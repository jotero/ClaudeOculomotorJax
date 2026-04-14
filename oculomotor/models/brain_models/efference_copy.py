"""Efference copy — delays u_burst by tau_vis to match the visual delay cascade.

Architecture
────────────
The saccade burst command (u_burst) drives eye movement AND contaminates the
retinal slip signal.  To cancel this contamination, u_burst must be subtracted
from the delayed retinal slip — but AFTER the visual delay, not before it.

Correct signal flow:
    raw_slip = w_scene − w_head − dx_plant          (no correction here)
    raw_slip  → [visual delay, tau_vis] → raw_slip_delayed
    u_burst   → [EC delay,    tau_vis] → u_burst_delayed     (this module)

    e_slip_corrected = raw_slip_delayed − u_burst_delayed     (in brain_model)
    e_slip_corrected → VS

The EC delay uses the same gamma-distributed cascade (N_STAGES first-order LP
filters, tau_vis total delay) as the visual pathway — ensuring that the
subtraction cancels at the same temporal phase.

State: x_ec = (N_STAGES × 3,) = (120,) delay cascade for u_burst (3-D).

Parameters:
    tau_vis — total delay (s).  Must match the visual delay parameter.
              Default: 0.08 s.  Shared via THETA_DEFAULT.
"""

from oculomotor.models.sensory_models.sensory_model import delay_cascade_step, delay_cascade_read
from oculomotor.models.sensory_models.sensory_model import _N_PER_SIG

N_STATES  = _N_PER_SIG   # 120  — one delay cascade for 3-D u_burst
N_INPUTS  = 3             # u_burst
N_OUTPUTS = 3             # u_burst_delayed


def read_delayed(x_ec):
    """Read the current delayed burst output from the EC state.

    Pure state readout — no derivative computation.  Returns u_burst as it was
    tau_vis seconds ago.  Called at the top of the brain step (before dx_ec is
    computed) so the subtraction uses the same-time delayed signal.

    Args:
        x_ec: (120,)  efference copy cascade state

    Returns:
        u_burst_delayed: (3,)  burst command delayed by tau_vis
    """
    return delay_cascade_read(x_ec)


def step(x_ec, u_burst, theta):
    """Advance the efference copy delay cascade by one ODE step.

    Args:
        x_ec:    (120,)  EC cascade state
        u_burst: (3,)    saccade burst velocity command (deg/s)
        theta:   dict    model parameters (reads 'tau_vis')

    Returns:
        dx_ec:           (120,)  dx_ec/dt
        u_burst_delayed: (3,)    burst command delayed by tau_vis (from current state)
    """
    # ── Dynamics ──────────────────────────────────────────────────────────────
    dx_ec           = delay_cascade_step(x_ec, u_burst, theta)
    u_burst_delayed = delay_cascade_read(x_ec)   # pure state read — no lag vs dx_ec
    return dx_ec, u_burst_delayed
