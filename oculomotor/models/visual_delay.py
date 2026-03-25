"""Visual delay SSM — gamma-distributed delay for two retinal signals.

A cascade of N_STAGES first-order LP filters, each with time constant
τ/N_STAGES, approximates a pure delay of τ seconds.

    Single LP (N=1):  impulse response ~ Exp(τ),     std = τ
    Cascade (N=4):    impulse response ~ Gamma(4, τ), std = τ/2
    Cascade (N→∞):    impulse response → δ(t − τ)    (pure delay)

With N_STAGES = 40 and τ = 0.08 s:
    Mean delay   = 0.08 s
    Spread std   = τ/√N ≈ 0.013 s   (vs 0.04 s for N=4 — much sharper step)
    −3 dB BW    ≈ 66 Hz
    Heun-stable  ✓ (τ_stage = 0.002 s, requires dt ≤ 0.004 s — use dt=0.001 s)

Two signals are delayed independently through the same cascade:
    Signal 0 — e_slip  (3,)  retinal velocity slip  → OKR drive
    Signal 1 — e_pos   (3,)  retinal position error → saccade motor error

Signal flow
───────────
    e_slip → [stage 0..39] → e_slip_delayed     (for OKR)
    e_pos  → [stage 0..39] → e_pos_delayed      (for saccades)

State layout (N_STAGES × 3 × N_SIG = 240):
    x_vis = [x_slip_0 (3) | … | x_slip_39 (3) |   (120 states)
             x_pos_0  (3) | … | x_pos_39  (3)]     (120 states)

Parameters
──────────
    tau_vis  — total visual delay (s).  Default: 0.08 s; ~80 ms latency to
               OKR onset (Cohen et al. 1977 J Neurophysiol; Miles, Kawano &
               Optican 1986 J Neurophysiol for smooth pursuit; Ilg 1997
               Curr Opin Neurobiol for OKR).
"""

import jax.numpy as jnp

# ── Cascade parameters ─────────────────────────────────────────────────────────

N_STAGES = 40             # cascade depth — sharp step (std ≈ τ/√N ≈ 13 ms)
N_SIG    = 2              # number of signals (slip + position error)
N_STATES = N_STAGES * 3 * N_SIG   # 240 total states

_N_PER_SIG = N_STAGES * 3   # 120 states per signal

# ── Structural matrices (computed once at import) ─────────────────────────────

def _make_cascade_A_struct():
    """Block bidiagonal A for one signal's N_STAGES cascade (12×12)."""
    A = -jnp.eye(_N_PER_SIG)   # diagonal = −1 (scaled by k at runtime)
    for i in range(1, N_STAGES):
        A = A.at[i*3:(i+1)*3, (i-1)*3:i*3].set(jnp.eye(3))
    return A

_A_STRUCT = _make_cascade_A_struct()   # (12, 12) — shared by both signals

# B structure: input enters only the first stage of each signal's cascade
_B_STRUCT_SIG = jnp.zeros((_N_PER_SIG, 3)).at[:3, :].set(jnp.eye(3))   # (12, 3)

# C: select last stage of each signal's cascade
C_slip = jnp.zeros((3, N_STATES)).at[:, _N_PER_SIG - 3 : _N_PER_SIG].set(jnp.eye(3))
C_pos  = jnp.zeros((3, N_STATES)).at[:, N_STATES - 3 :              ].set(jnp.eye(3))


# ── ABCD matrices ──────────────────────────────────────────────────────────────

def get_A(theta):
    """(N_STATES × N_STATES) block-diagonal cascade state matrix.

    Two identical cascades — one per signal — on the diagonal.
    """
    tau = theta.get('tau_vis', 0.08)
    k   = N_STAGES / tau
    A_blk = k * _A_STRUCT   # (12, 12) for one signal
    return jnp.block([[A_blk,           jnp.zeros((_N_PER_SIG, _N_PER_SIG))],
                      [jnp.zeros((_N_PER_SIG, _N_PER_SIG)), A_blk          ]])


def get_B(theta):
    """(N_STATES × 6) input matrix: each signal drives its own first stage."""
    tau   = theta.get('tau_vis', 0.08)
    k     = N_STAGES / tau
    B_blk = k * _B_STRUCT_SIG   # (12, 3) for one signal
    return jnp.block([[B_blk,                   jnp.zeros((_N_PER_SIG, 3))],
                      [jnp.zeros((_N_PER_SIG, 3)), B_blk                  ]])


def step(x_vis, e_slip, e_pos, theta):
    """Single ODE step: state derivative + both delayed outputs.

    Args:
        x_vis:  (N_STATES,)  delay cascade state (24,)
        e_slip: (3,)          instantaneous retinal slip (deg/s)
        e_pos:  (3,)          instantaneous retinal position error (deg)
        theta:  dict          model parameters

    Returns:
        dx:             (N_STATES,)  dx_vis/dt
        e_slip_delayed: (3,)         delayed retinal slip   (for OKR)
        e_pos_delayed:  (3,)         delayed position error (for saccades)
    """
    u            = jnp.concatenate([e_slip, e_pos])   # (6,)
    dx           = get_A(theta) @ x_vis + get_B(theta) @ u
    e_slip_delayed = C_slip @ x_vis
    e_pos_delayed  = C_pos  @ x_vis
    return dx, e_slip_delayed, e_pos_delayed
