"""Brain model — velocity storage, neural integrator, saccade generator, efference copy.

Aggregates velocity_storage, neural_integrator, saccade_generator, and
efference_copy into a single SSM with one state vector and one step() function.

Signal flow:
    y_canals         (6,)   canal afferents         → VS
    raw_slip_delayed (3,)   delayed raw retinal slip → VS (after EC correction)
    e_cmd            (3,)   motor error command      → SG

Efference copy correction (inside step, before VS):
    u_burst_delayed  = ec.read_delayed(x_ec)          from current EC state
    e_slip_corrected = raw_slip_delayed − u_burst_delayed
    VS receives e_slip_corrected

Internal flow:
    VS  →  w_est  →  u_vel (with u_burst)  →  NI  →  u_p  →  (plant)
    SG  →  u_burst  (saccade velocity command → EC delay → next-step correction)
    EC  →  delays u_burst by tau_vis for next-step slip cancellation

State vector  x_brain = [x_vs (3) | x_ni (3) | x_sg (9) | x_ec (120)]  — N_STATES = 135

Index constants (relative to x_brain):
    _IDX_VS   — velocity storage states  (3,)
    _IDX_NI   — neural integrator states (3,)
    _IDX_SG   — saccade generator states (9,)
    _IDX_EC   — efference copy states    (120,)

Outputs of step():
    dx_brain  (135,)  state derivative
    u_p       (3,)    pulse-step motor command → plant
    u_burst   (3,)    saccade burst velocity command
"""

import jax.numpy as jnp

from oculomotor.models import velocity_storage as vs
from oculomotor.models import neural_integrator as ni
from oculomotor.models import saccade_generator as sg
from oculomotor.models import efference_copy as ec
from oculomotor.models.sensory_model import SensoryOutput  # noqa: F401 (re-exported)

# ── State layout ───────────────────────────────────────────────────────────────

N_STATES = vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES   # 3+3+9+120 = 135

# Index constants — relative to x_brain
_IDX_VS = slice(0,             vs.N_STATES)                                   # (3,)
_IDX_NI = slice(vs.N_STATES,   vs.N_STATES + ni.N_STATES)                    # (3,)
_IDX_SG = slice(vs.N_STATES + ni.N_STATES,
                vs.N_STATES + ni.N_STATES + sg.N_STATES)                      # (9,)
_IDX_EC = slice(vs.N_STATES + ni.N_STATES + sg.N_STATES,
                vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES)        # (120,)


# ── Step function ──────────────────────────────────────────────────────────────

def step(x_brain, sensory_out, e_cmd, scene_present, theta):
    """Single ODE step for the brain subsystem.

    Args:
        x_brain:      (135,)        brain state [x_vs | x_ni | x_sg | x_ec]
        sensory_out:  SensoryOutput bundled canal afferents + delayed visual signals
                        .canal        (6,)  canal afferent rates
                        .slip_delayed (3,)  delayed retinal slip (no EC correction yet)
                        .pos_visible  (3,)  delayed position error, gated by visual field
        e_cmd:        (3,)    motor error command for the saccade generator
                              (computed in simulator: orbital gate applied to pos_visible)
        scene_present: scalar  0=dark, 1=lit — gates EC slip correction
        theta:        Params   model parameters

    Returns:
        dx_brain:  (135,)  dx_brain/dt
        u_p:       (3,)    pulse-step motor command → plant
        u_burst:   (3,)    saccade burst velocity command
    """
    x_vs = x_brain[_IDX_VS]
    x_ni = x_brain[_IDX_NI]
    x_sg = x_brain[_IDX_SG]
    x_ec = x_brain[_IDX_EC]

    # ── Efference copy correction: add delayed burst to delayed slip ──────────
    # u_burst_delayed matches the phase of slip_delayed (both delayed by tau_vis).
    # Cancels burst-driven eye motion from retinal slip before VS.
    # Gated by scene_present: correction only applies when a visual scene is visible.
    u_burst_delayed  = ec.read_delayed(x_ec)
    e_slip_corrected = sensory_out.slip_delayed + scene_present * u_burst_delayed

    # ── Velocity storage: canal + corrected slip → velocity estimate ──────────
    dx_vs, w_est = vs.step(x_vs, jnp.concatenate([sensory_out.canal, e_slip_corrected]), theta)

    # ── Saccade generator ─────────────────────────────────────────────────────
    dx_sg, u_burst = sg.step(x_sg, e_cmd, theta)

    # ── Neural integrator: combined eye-velocity command ──────────────────────
    u_vel      = -w_est + u_burst
    dx_ni, u_p = ni.step(x_ni, u_vel, theta)

    # ── Efference copy: advance delay cascade with current burst ──────────────
    dx_ec, _ = ec.step(x_ec, u_burst, theta)

    # ── Pack state derivative ─────────────────────────────────────────────────
    dx_brain = jnp.concatenate([dx_vs, dx_ni, dx_sg, dx_ec])

    return dx_brain, u_p, u_burst
