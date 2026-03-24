"""OKR visual delay SSM — gamma-distributed delay for optokinetic reflex.

A cascade of N_STAGES first-order LP filters, each with time constant
τ/N_STAGES, approximates a pure delay of τ seconds (method of stages).

    Single LP (N=1):  impulse response ~ Exp(τ),     std = τ
    Cascade (N=4):    impulse response ~ Gamma(4, τ), std = τ/2
    Cascade (N→∞):    impulse response → δ(t − τ)    (pure delay)

With N_STAGES = 4 and τ = 0.08 s:
    Mean delay   = 0.08 s
    Spread std   = 0.04 s       (vs 0.08 s for single LP)
    −3 dB BW    ≈ 22 Hz         (OKR operates below ~2 Hz → effectively exact)
    Heun-stable  ✓ (τ_stage = 0.02 s, dt/τ_stage = 0.25)

Signal flow
───────────
    v_eye (3,)  →  x_vis[0] → x_vis[1] → x_vis[2] → x_vis[3]
                   (3,)        (3,)        (3,)        (3,) delayed eye vel

    e     = v_scene − x_vis[N−1]          retinal slip
    u_okr = g_okr · e                     OKR VS drive (deg/s)

State (N_STAGES × 3 = 12): four 3-D delay stages [x0|x1|x2|x3]
Input (3):  v_eye — plant derivative (eye angular velocity, deg/s)
Output (3): u_okr — OKR contribution to VS input (deg/s)

VS sign convention
──────────────────
    The OKR drive enters VS as −u_okr (rightward scene → rightward eye,
    equivalent to leftward head VOR response through existing pathway).
    The subtraction is done in ocular_motor_simulator, not here.

Parameters
──────────
    tau_okr_del  — total visual delay (s).   Default: 0.08 s.
    g_okr        — OKR gain (unitless).       Default: 0.7.
"""

import jax.numpy as jnp

# ── Cascade parameters ─────────────────────────────────────────────────────────

N_STAGES = 4              # cascade depth (4 → std = τ/2, BW ≈ 22 Hz for τ=0.08s)
N_STATES = N_STAGES * 3  # 12 total states

# ── Structural matrices (computed once at import, scaled by theta at runtime) ──

# B structure: v_eye enters only the first stage
_B_STRUCT = jnp.zeros((N_STATES, 3)).at[:3, :].set(jnp.eye(3))

# C: select last stage output (delayed eye velocity)
C = jnp.zeros((3, N_STATES)).at[:, -3:].set(jnp.eye(3))


# ── ABCD matrices ──────────────────────────────────────────────────────────────

def get_A(theta):
    """(N_STATES × N_STATES) cascade state matrix.

    Block lower-bidiagonal:
        diagonal blocks    = −k · I₃     (decay)
        subdiagonal blocks = +k · I₃     (feed from previous stage)
    where k = N_STAGES / tau_okr_del.

    Note: cascade_deriv() is more efficient for ODE integration.
    """
    tau = theta.get('tau_okr_del', 0.08)
    k   = N_STAGES / tau
    A   = -k * jnp.eye(N_STATES)
    for i in range(1, N_STAGES):
        A = A.at[i*3:(i+1)*3, (i-1)*3:i*3].set(k * jnp.eye(3))
    return A


def get_B(theta):
    """(N_STATES × 3) input matrix: v_eye drives first stage only."""
    tau = theta.get('tau_okr_del', 0.08)
    k   = N_STAGES / tau
    return k * _B_STRUCT


def get_D(theta):
    """(3 × 3) gain matrix: retinal slip → OKR VS drive."""
    return theta.get('g_okr', 0.0) * jnp.eye(3)


# ── Efficient cascade derivative ───────────────────────────────────────────────

def cascade_deriv(x_vis, v_eye, theta):
    """Compute dx_vis/dt for the N-stage gamma-delay cascade.

    Equivalent to get_A(theta) @ x_vis + get_B(theta) @ v_eye
    but O(N·3) instead of O((N·3)²) — avoids the full matrix multiply.

    Args:
        x_vis: (N_STATES,) current cascade state
        v_eye: (3,)        eye angular velocity = plant derivative (deg/s)
        theta: parameter dict

    Returns:
        dx_vis: (N_STATES,)
    """
    tau = theta.get('tau_okr_del', 0.08)
    k   = N_STAGES / tau
    x   = x_vis.reshape(N_STAGES, 3)                          # (N, 3)
    u   = jnp.concatenate([v_eye[None, :], x[:-1]], axis=0)  # (N, 3) stage inputs
    return (k * (u - x)).reshape(-1)                          # (N_STATES,)


# ── Output helpers ─────────────────────────────────────────────────────────────

def delayed_eye_vel(x_vis):
    """Delayed eye velocity: last cascade stage output, shape (3,)."""
    return x_vis[-3:]


def retinal_slip(x_vis, v_scene):
    """Retinal slip = v_scene − delayed_eye_vel, shape (3,)."""
    return v_scene - delayed_eye_vel(x_vis)


def okr_drive(x_vis, v_scene, theta):
    """OKR VS drive = g_okr · retinal_slip, shape (3,)."""
    return get_D(theta) @ retinal_slip(x_vis, v_scene)
