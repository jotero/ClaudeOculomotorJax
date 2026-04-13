"""Brain model — velocity storage, neural integrator, saccade generator, efference copy.

Aggregates velocity_storage, neural_integrator, saccade_generator, and
efference_copy into a single SSM with one state vector and one step() function.

Signal flow:
    y_canals        (6,)  canal afferents        → VS
    e_slip_delayed  (3,)  delayed retinal slip   → VS (OKR)
    e_cmd           (3,)  motor error command    → SG  (computed by simulator via target_selector)

Internal flow:
    VS  →  w_est  →  u_vel (with u_burst)  →  NI  →  u_p  →  (plant)
    SG  →  u_burst  (saccade velocity command)
    EC  driven by u_burst  (efference copy for retinal slip cancellation)

State vector  x_brain = [x_vs (3) | x_ni (3) | x_sg (9) | x_ec (6)]  — N_STATES = 21

Index constants (relative to x_brain):
    _IDX_VS   — velocity storage states (3,)
    _IDX_NI   — neural integrator states (3,)
    _IDX_SG   — saccade generator states (9,)
    _IDX_EC   — efference copy states (6,)

Efference copy sub-layout (relative to x_brain[_IDX_EC]):
    _IDX_NI_PC  — NI copy states (3,)
    _IDX_PC     — plant copy states (3,)

Outputs of step():
    dx_brain  (21,)  state derivative
    u_p       (3,)   pulse-step motor command → plant
    u_burst   (3,)   saccade burst velocity command → retinal slip cancellation
"""

import jax.numpy as jnp

from oculomotor.models import velocity_storage as vs
from oculomotor.models import neural_integrator as ni
from oculomotor.models import saccade_generator as sg
from oculomotor.models import efference_copy as ec

# ── State layout ───────────────────────────────────────────────────────────────

N_STATES = vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES   # 3+3+9+6 = 21

# Index constants — relative to x_brain
_IDX_VS = slice(0,             vs.N_STATES)                                   # (3,)
_IDX_NI = slice(vs.N_STATES,   vs.N_STATES + ni.N_STATES)                    # (3,)
_IDX_SG = slice(vs.N_STATES + ni.N_STATES,
                vs.N_STATES + ni.N_STATES + sg.N_STATES)                      # (9,)
_IDX_EC = slice(vs.N_STATES + ni.N_STATES + sg.N_STATES,
                vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES)        # (6,)

# Efference copy sub-layout — relative to x_brain[_IDX_EC]
_IDX_NI_PC = slice(0, 3)   # x_ni_pc within ec block
_IDX_PC    = slice(3, 6)   # x_pc    within ec block


# ── Step function ──────────────────────────────────────────────────────────────

def step(x_brain, y_canals, e_slip_delayed, e_cmd, theta):
    """Single ODE step for the brain subsystem.

    Args:
        x_brain:        (21,)  brain state [x_vs | x_ni | x_sg | x_ec]
        y_canals:       (6,)   canal afferent firing rates
        e_slip_delayed: (3,)   delayed retinal slip   (for VS / OKR)
        e_cmd:          (3,)   motor error command for saccade generator
                               (computed upstream by target_selector.select())
        theta:          dict   model parameters

    Returns:
        dx_brain:  (21,)  dx_brain/dt
        u_p:       (3,)   pulse-step motor command → plant
        u_burst:   (3,)   saccade burst velocity command → efference slip cancellation
    """
    x_vs = x_brain[_IDX_VS]
    x_ni = x_brain[_IDX_NI]
    x_sg = x_brain[_IDX_SG]
    x_ec = x_brain[_IDX_EC]

    # ── Velocity storage: canal + visual → angular velocity estimate ──────────
    dx_vs, w_est = vs.step(x_vs, jnp.concatenate([y_canals, e_slip_delayed]), theta)

    # ── Saccade generator ─────────────────────────────────────────────────────
    dx_sg, u_burst = sg.step(x_sg, e_cmd, theta)

    # ── Neural integrator: combined eye-velocity command ──────────────────────
    u_vel      = -w_est + u_burst
    dx_ni, u_p = ni.step(x_ni, u_vel, theta)

    # ── Efference copy: mirrors NI + plant response to burst ──────────────────
    dx_ec, _ = ec.step(x_ec, u_burst, theta)

    # ── Pack state derivative ─────────────────────────────────────────────────
    dx_brain = jnp.concatenate([dx_vs, dx_ni, dx_sg, dx_ec])

    return dx_brain, u_p, u_burst
