"""Visual delay SSM — gamma-distributed delay (method of stages).

A cascade of N_STAGES first-order LP filters, each with time constant
τ/N_STAGES, approximates a pure delay of τ seconds.

    Single LP (N=1):  impulse response ~ Exp(τ),     std = τ
    Cascade (N=4):    impulse response ~ Gamma(4, τ), std = τ/2
    Cascade (N→∞):    impulse response → δ(t − τ)    (pure delay)

With N_STAGES = 4 and τ = 0.08 s:
    Mean delay   = 0.08 s
    Spread std   = 0.04 s       (vs 0.08 s for single LP)
    −3 dB BW    ≈ 22 Hz         (visual reflexes operate below ~2 Hz → effectively exact)
    Heun-stable  ✓ (τ_stage = 0.02 s, dt/τ_stage = 0.25)

Signal flow
───────────
    e_slip → x_vis[0] → x_vis[1] → x_vis[2] → x_vis[3]
             (3,)        (3,)        (3,)        (3,)

    e_delayed = C @ x_vis = x_vis[-3:]       (delayed retinal slip)

State (N_STAGES × 3 = 12): four 3-D delay stages [x0|x1|x2|x3]
Input (3):  e_slip   — instantaneous retinal slip (deg/s), from simulator
Output (3): e_delayed — delayed retinal slip (deg/s)

Parameters
──────────
    tau_vis  — total visual delay (s).  Default: 0.08 s.
"""

import jax.numpy as jnp

# ── Cascade parameters ─────────────────────────────────────────────────────────

N_STAGES = 4              # cascade depth (4 → std = τ/2, BW ≈ 22 Hz for τ=0.08s)
N_STATES = N_STAGES * 3  # 12 total states

# ── Structural matrices (computed once at import, scaled by theta at runtime) ──

# B structure: e_slip enters only the first stage
_B_STRUCT = jnp.zeros((N_STATES, 3)).at[:3, :].set(jnp.eye(3))

# C: select last stage output (delayed retinal slip)
C = jnp.zeros((3, N_STATES)).at[:, -3:].set(jnp.eye(3))


# ── ABCD matrices ──────────────────────────────────────────────────────────────

def get_A(theta):
    """(N_STATES × N_STATES) cascade state matrix.

    Block lower-bidiagonal:
        diagonal blocks    = −k · I₃     (decay)
        subdiagonal blocks = +k · I₃     (feed from previous stage)
    where k = N_STAGES / tau_vis.
    """
    tau = theta.get('tau_vis', 0.08)
    k   = N_STAGES / tau
    A   = -k * jnp.eye(N_STATES)
    for i in range(1, N_STAGES):
        A = A.at[i*3:(i+1)*3, (i-1)*3:i*3].set(k * jnp.eye(3))
    return A


def get_B(theta):
    """(N_STATES × 3) input matrix: e_slip drives first stage only."""
    tau = theta.get('tau_vis', 0.08)
    k   = N_STAGES / tau
    return k * _B_STRUCT


def step(x_vis, e_slip, theta):
    """Single ODE step: state derivative + delayed retinal slip output.

    Args:
        x_vis:  (N_STATES,)  delay cascade state
        e_slip: (3,)         instantaneous retinal slip (deg/s)
        theta:  dict         model parameters

    Returns:
        dx:        (N_STATES,)  dx_vis/dt
        e_delayed: (3,)         delayed retinal slip  C@x_vis
    """
    dx        = get_A(theta) @ x_vis + get_B(theta) @ e_slip
    e_delayed = C @ x_vis
    return dx, e_delayed
