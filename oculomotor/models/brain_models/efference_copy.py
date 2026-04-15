"""Efference copy — delays the combined motor command by tau_vis.

Architecture
────────────
Motor commands (u_burst + u_pursuit) drive eye movement AND contaminate the
retinal signals.  To cancel this contamination the combined motor command must
be added back to the delayed retinal signals — but AFTER the visual delay, not
before it.

Correct signal flow:
    u_motor = u_burst + u_pursuit                    (combined motor command)
    u_motor  → [EC delay, tau_vis] → u_motor_delayed (this module)

    e_slip_corrected = slip_delayed + scene · u_motor_delayed    (→ VS)
    e_combined       = vel_delayed  + scene · u_motor_delayed    (→ pursuit)

The EC delay uses the same gamma-distributed cascade (N_STAGES first-order LP
filters, tau_vis total delay) as the visual pathway — ensuring cancellation at
the same temporal phase.

VS correction (additive):
    slip_delayed ≈ −u_motor(t−τ)  →  slip + u_motor_ec ≈ 0  (VS silent) ✓

Pursuit Smith predictor (brain_model.py):
    e_combined ≈ v_target  (self-motion removed)
    u_pu_now   = x_pursuit + K_phasic · e_vel_pred      (current pursuit command)
    e_vel_pred = (e_combined − x_pursuit) / (1 + K_phasic)   (closed-form, no circularity)
    → integrator receives 1/(1+K_phasic) ≈ 45 % of the error at onset,
      eliminating delay-induced oscillation while preserving fast onset.

State: x_ec = (N_STAGES × 3,) = (120,) delay cascade (3-D).

Parameters:
    tau_vis — total delay (s).  Must match the visual delay parameter.
              Default: 0.08 s.
"""

from oculomotor.models.sensory_models.retina import delay_cascade_step, delay_cascade_read
from oculomotor.models.sensory_models.retina import _N_PER_SIG

N_STATES  = _N_PER_SIG   # 120  — one delay cascade for 3-D combined motor command
N_INPUTS  = 3             # u_motor = u_burst + u_pursuit
N_OUTPUTS = 3             # u_motor_delayed


def read_delayed(x_ec):
    """Read the current delayed motor output from the EC state.

    Pure state readout — no derivative computation.  Returns u_motor as it was
    tau_vis seconds ago.  Called at the top of the brain step (before dx_ec is
    computed) so the correction uses the same-time delayed signal.

    Args:
        x_ec: (120,)  efference copy cascade state

    Returns:
        u_motor_delayed: (3,)  combined motor command delayed by tau_vis
    """
    return delay_cascade_read(x_ec)


def step(x_ec, u_motor, brain_params):
    """Advance the efference copy delay cascade by one ODE step.

    Args:
        x_ec:         (120,)       EC cascade state
        u_motor:      (3,)         combined motor command = u_burst + u_pursuit (deg/s)
        brain_params: BrainParams  model parameters (reads tau_vis)

    Returns:
        dx_ec:           (120,)  dx_ec/dt
        u_motor_delayed: (3,)    motor command delayed by tau_vis (from current state)
    """
    # ── Dynamics ──────────────────────────────────────────────────────────────
    dx_ec            = delay_cascade_step(x_ec, u_motor, brain_params.tau_vis)
    u_motor_delayed  = delay_cascade_read(x_ec)   # pure state read — no lag vs dx_ec
    return dx_ec, u_motor_delayed
