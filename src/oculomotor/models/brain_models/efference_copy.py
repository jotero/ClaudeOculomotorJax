"""Efference copy — delays a 3-D motor command by tau_vis seconds.

Architecture
────────────
A motor command (u_burst or u_pursuit) drives eye movement AND contaminates
the retinal signals.  To cancel this contamination the command must be added
back to the delayed retinal signals — but AFTER the visual delay, not before.

This module is a reusable 120-state delay cascade.  brain_model instantiates
it TWICE — once for u_burst (saccade EC) and once for u_pursuit (pursuit EC):

    saccade_ec = ec.read_delayed(x_ec)        # delay(u_burst)
    pursuit_ec = ec.read_delayed(x_pu_ec)     # delay(u_pursuit)

    VS correction  (scene-gated):
        e_slip_corrected = slip_delayed + scene · (saccade_ec + pursuit_ec)
        slip + (saccade+pursuit)_ec ≈ 0  during any self-generated eye motion ✓

    Pursuit Smith predictor (target-gated):
        e_combined = vel_delayed + target · pursuit_ec   ≈ v_target
        e_vel_pred = (e_combined − x_pursuit) / (1 + K_phasic)
        → burst EC deliberately excluded: u_burst in e_combined would
          drive the pursuit integrator spuriously after each saccade.

The EC delay uses the same gamma-distributed cascade (N_STAGES first-order LP
filters, tau_vis total delay) as the visual pathway.

State: x_ec = (N_STAGES × 3,) = (120,) delay cascade for one 3-D signal.

Parameters:
    tau_vis — total delay (s).  Must match the visual delay parameter.
              Default: 0.08 s.
"""

from oculomotor.models.sensory_models.retina import delay_cascade_step, delay_cascade_read
from oculomotor.models.sensory_models.retina import _N_PER_SIG

N_STATES  = _N_PER_SIG   # 120  — one delay cascade for one 3-D signal
N_INPUTS  = 3             # u  (u_burst OR u_pursuit depending on which instance)
N_OUTPUTS = 3             # u_delayed


def read_delayed(x_ec):
    """Read the delayed output from an EC cascade state (pure state readout).

    Returns the signal as it was tau_vis seconds ago.  Called at the top of
    the brain step (before dx_ec is computed) so the correction uses the
    same-time delayed signal.

    Args:
        x_ec: (120,)  efference copy cascade state

    Returns:
        u_delayed: (3,)  signal delayed by tau_vis
    """
    return delay_cascade_read(x_ec)


def step(x_ec, u, brain_params):
    """Advance one EC delay cascade by one ODE step.

    Args:
        x_ec:         (120,)       EC cascade state
        u:            (3,)         signal to delay (u_burst OR u_pursuit)
        brain_params: BrainParams  model parameters (reads tau_vis)

    Returns:
        dx_ec:     (120,)  dx_ec/dt
        u_delayed: (3,)    signal delayed by tau_vis (from current state)
    """
    dx_ec     = delay_cascade_step(x_ec, u, brain_params.tau_vis)
    u_delayed = delay_cascade_read(x_ec)   # pure state read — no lag vs dx_ec
    return dx_ec, u_delayed
