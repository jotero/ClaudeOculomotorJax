"""Efference copy: NI + plant copy driven by saccade burst command.

State-space model (SSM) that mirrors the NI and plant response to u_burst alone,
so that the predicted burst-driven eye velocity cancels exactly in the retinal
slip signal:

    e_slip = scene_gain * (w_scene - w_head - w_eye + w_burst_pred)

State vector  x_ec = [x_ni_pc (3) | x_pc (3)]   (N_STATES = 6)

    x_ni_pc — efference-copy NI: integrates u_burst with tau_i leak.
    x_pc    — efference-copy plant: follows x_ni_pc with tau_p lag.

Equations:
    dx_ni_pc = -1/tau_i * x_ni_pc + u_burst
    dx_pc    =  1/tau_p * x_ni_pc - 1/tau_p * x_pc + u_burst

Output:
    w_burst_pred = (x_ni_pc - x_pc) / tau_p + u_burst  ≡  dx_pc

Matrices (see get_A, get_B, get_C, get_D):
    A (6×6): block diagonal [ -1/tau_i · I₃  |  0           ]
                             [  1/tau_p · I₃  | -1/tau_p · I₃]
    B (6×3): [I₃ ; I₃]
    C (3×6): [1/tau_p · I₃ | -1/tau_p · I₃]
    D (3×3): I₃   (u_burst feedthrough to w_burst_pred)
"""

import jax.numpy as jnp

N_STATES  = 6   # [x_ni_pc (3) | x_pc (3)]
N_INPUTS  = 3   # u_burst
N_OUTPUTS = 3   # w_burst_pred


def step(x_ec, u_burst, theta):
    """Advance efference copy one ODE step.

    Args:
        x_ec:    state vector (6,): [x_ni_pc (3) | x_pc (3)]
        u_burst: saccade burst command (3,)
        theta:   parameter dict with keys tau_i, tau_p

    Returns:
        dx_ec:        state derivative (6,)
        w_burst_pred: predicted burst-driven eye velocity (3,)
    """
    # ── System matrices ───────────────────────────────────────────────────────
    tau_i = theta['tau_i']
    tau_p = theta['tau_p']
    I3 = jnp.eye(3)
    Z3 = jnp.zeros((3, 3))
    A = jnp.concatenate([
        jnp.concatenate([(-1/tau_i) * I3,  Z3          ], axis=1),  # NI copy row
        jnp.concatenate([( 1/tau_p) * I3, (-1/tau_p)*I3], axis=1),  # plant copy row
    ], axis=0)                                           # (6, 6)
    B = jnp.concatenate([I3, I3], axis=0)               # (6, 3) — burst drives both
    C = jnp.concatenate([(1/tau_p)*I3, (-1/tau_p)*I3], axis=1)  # (3, 6) velocity readout
    # D = I (identity feedthrough — omitted)

    # ── Dynamics ──────────────────────────────────────────────────────────────
    dx_ec        = A @ x_ec + B @ u_burst
    w_burst_pred = C @ x_ec + u_burst
    return dx_ec, w_burst_pred
