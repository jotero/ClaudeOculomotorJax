"""Canal SSMs — array of bandpass semicircular canal filters.

Each canal is a second-order system: the classic torsion-pendulum model
(Steinhausen 1931) cascades a long-TC high-pass (cupula stiffness) with a
short-TC low-pass (endolymph inertia), giving a bandpass response.

Per-canal signal:
    h_i     = n_i[0] · head_vel              projected input (1-D stimulus)
    dx1_i   = (h_i − x1_i) / τ_c            slow LP state (cupula adaptation)
    y_hp_i  = h_i − x1_i                    HP intermediate (long time constant)
    dx2_i   = (y_hp_i − x2_i) / τ_s         fast LP state (endolymph inertia)
    y_i     = nl(x2_i, floor) · g_i         bandpass output + nonlinearity

Transfer function per canal (head_vel → y):
    H(s) = τ_c · s / [(1 + s·τ_c)(1 + s·τ_s)]   — bandpass
    Low-cut  at 1/(2π·τ_c) ≈ 0.032 Hz  (τ_c ≈ 5 s, cupula stiffness)
    High-cut at 1/(2π·τ_s) ≈ 32 Hz     (τ_s ≈ 0.005 s, endolymph inertia)

Canal nonlinearity nl(y, floor) — smooth, differentiable everywhere:
    y < −floor  →  −floor              inhibitory saturation (Ewald's 2nd law)
    |y| ≤ floor →  y                   linear regime
    y >  floor  →  2y − floor          excitatory gain doubles past threshold

For a push-pull pair the excitatory doubling exactly compensates the
inhibitory loss, keeping combined VOR gain = 1 for all amplitudes.
Use floor=1e6 to disable the nonlinearity (linear mode for fitting).

State layout (per canal: [x1, x2], ordered as x_c = [x1_all, x2_all]):
    x_c = [x1_c0, x1_c1, x2_c0, x2_c1]    shape (N_STATES,) = (4,)

Combined vestibular signal fed into VS:
    u_vs = PINV_SENS_1D @ y_canals    pseudo-inverse mixing → velocity estimate
"""

import jax.numpy as jnp
from jax.nn import softplus

# ── Canal geometry ────────────────────────────────────────────────────────────

ORIENTATIONS = jnp.array([
    [ 1.,  0.,  0.],   # canal 0 — right / ipsilateral  horizontal
    [-1.,  0.,  0.],   # canal 1 — left  / contralateral horizontal
])

N_CANALS           = ORIENTATIONS.shape[0]
N_STATES_PER_CANAL = 2              # x1 (adaptation LP) + x2 (inertia LP)
N_STATES           = N_CANALS * N_STATES_PER_CANAL   # 4 total
FLOOR              = 80.0           # deg/s — physiological resting discharge

# 1-D (horizontal) sensitivity and its pseudo-inverse.
# For [[+1], [−1]]:  PINV_SENS_1D = [[0.5, −0.5]]
_SENS_1D     = ORIENTATIONS[:, 0]                          # (N_CANALS,)
PINV_SENS_1D = jnp.linalg.pinv(_SENS_1D.reshape(-1, 1))   # (1, N_CANALS)

# Smoothing sharpness for the nonlinearity (s/deg).
# Transition width ≈ 1/_SOFTNESS ≈ 2 deg/s.  Increase for sharper kinks.
_SOFTNESS = 0.5


# ── SSM matrices ──────────────────────────────────────────────────────────────

def get_A(theta):
    """(N_STATES, N_STATES) state matrix — block structure.

    For state ordering [x1_c0, x1_c1, x2_c0, x2_c1]:

        d/dt [x1]   =  [−I/τc    0  ] [x1]   +  [S/τc] · h
             [x2]      [ I/τs  −I/τs] [x2]      [S/τs]

    where I = identity(N_CANALS), S = diag(_SENS_1D).
    """
    I  = jnp.eye(N_CANALS)
    Z  = jnp.zeros((N_CANALS, N_CANALS))
    ac = -1.0 / theta['tau_c']
    as_ = -1.0 / theta['tau_s']
    top = jnp.concatenate([ac * I,   Z       ], axis=1)
    bot = jnp.concatenate([as_ * I,  as_ * I ], axis=1)
    return jnp.concatenate([top, bot], axis=0)


def get_B(theta):
    """(N_STATES, 1) input matrix.

    B = [_SENS_1D/τc, _SENS_1D/τs]^T  — maps scalar head_vel to all states.
    """
    return jnp.concatenate([
        _SENS_1D / theta['tau_c'],
        _SENS_1D / theta['tau_s'],
    ]).reshape(-1, 1)


# ── Canal output (nonlinear) ──────────────────────────────────────────────────

def canal_outputs(x_c, floor, gains):
    """Compute bandpass afferent outputs for all canals.

    Args:
        x_c:   (N_STATES,)  full canal state [x1_c0, x1_c1, x2_c0, x2_c1]
        floor: scalar        inhibition depth / excitatory threshold (deg/s).
                             Use 1e6 for linear behaviour (no nonlinearity).
        gains: (N_CANALS,)  per-canal scale; 0.0 = complete loss.

    Returns:
        y: (N_CANALS,)  nonlinear bandpass afferent outputs
    """
    x2   = x_c[N_CANALS:]                              # (N_CANALS,) inertia states
    k    = _SOFTNESS
    y_nl = -floor + softplus(k * (x2 + floor)) / k + softplus(k * (x2 - floor)) / k
    return gains * y_nl
