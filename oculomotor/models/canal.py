"""Canal SSMs — array of bandpass semicircular canal filters.

Each canal is a second-order system (torsion-pendulum model, Steinhausen 1931):
a long-TC adaptation LP (cupula stiffness) cascades with a short-TC inertia LP
(endolymph), giving a bandpass response.

Per-canal signal (scalar — projection onto canal axis):
    h_i     = n_i · ω_head              projected 3-D head angular velocity
    dx1_i   = (h_i − x1_i) / τ_c       adaptation LP state (cupula)
    y_hp_i  = h_i − x1_i               HP intermediate (long time constant)
    dx2_i   = (y_hp_i - x2_i) / τ_s   inertia LP state
    y_i     = (nl(x2_i) + floor) · g_i   absolute firing rate; rest = floor

Transfer function per canal (ω_head → y):
    H(s) = τ_c·s / [(1+s·τ_c)(1+s·τ_s)]   — bandpass
    Low-cut  at 1/(2π·τ_c) ≈ 0.032 Hz   (τ_c ≈ 5 s)
    High-cut at 1/(2π·τ_s) ≈ 32  Hz    (τ_s ≈ 0.005 s)

Canal nonlinearity nl(y, floor) — smooth, differentiable everywhere:
    y < −floor  →  −floor              inhibitory saturation (Ewald's 2nd law)
    |y| ≤ floor →  y                   linear regime
    y >  floor  →  2y − floor          excitatory gain doubles past threshold

For a push-pull pair the excitatory doubling exactly compensates the
inhibitory loss, keeping combined VOR gain = 1 for all amplitudes.
Use floor=1e6 to disable the nonlinearity (linear mode for fitting).

Canal geometry (semi-anatomical, 45° vertical canals):
    Horizontal canals lie in the yaw plane.
    Vertical canals form two coplanar push-pull pairs (LARP and RALP), each
    tilted 45° between the pitch and roll axes — matching the standard
    anatomical description of anterior/posterior canal planes.

    Convention: x = rightward yaw, y = upward pitch, z = CW roll (right ear down)

    RALP pair (Right Anterior + Left Posterior):
        RAC excited by pitch-up + CW roll (right ear down) → [0, +s, +s]
        LPC opposite                                        → [0, −s, −s]
    LARP pair (Left Anterior + Right Posterior):
        LAC excited by pitch-up + CCW roll (left ear down) → [0, +s, −s]
        RPC opposite                                        → [0, −s, +s]

    Because ORIENTATIONS^T @ ORIENTATIONS = 2·I₃, the pseudo-inverse is
    exactly PINV_SENS = (1/2)·ORIENTATIONS^T — no numerical ambiguity.

State layout (per canal: [x1, x2], stacked as [x1_all | x2_all]):
    x_c = [x1_c0..x1_c5 | x2_c0..x2_c5]    shape (N_STATES,) = (12,)

Mixing to 3-D velocity estimate fed to VS:
    u_vs = PINV_SENS @ y_canals    (3,)  pseudo-inverse → angular velocity
"""

import jax.numpy as jnp
from jax.nn import softplus

# ── Canal geometry ─────────────────────────────────────────────────────────────

_S = 2 ** -0.5    # sin/cos 45° = 1/√2

ORIENTATIONS = jnp.array([
    [ 1.,   0.,   0.],   # canal 0 — RHC  right horizontal   (yaw+)
    [-1.,   0.,   0.],   # canal 1 — LHC  left  horizontal   (yaw−)
    [ 0.,  _S,   _S],   # canal 2 — RAC  right anterior     (RALP, pitch+ & CW roll+)
    [ 0., -_S,  -_S],   # canal 3 — LPC  left  posterior    (RALP, pitch− & CW roll−)
    [ 0.,  _S,  -_S],   # canal 4 — LAC  left  anterior     (LARP, pitch+ & CCW roll)
    [ 0., -_S,   _S],   # canal 5 — RPC  right posterior    (LARP, pitch− & CW roll)
])  # shape (N_CANALS, 3)

N_CANALS           = ORIENTATIONS.shape[0]          # 6
N_STATES_PER_CANAL = 2                              # x1 (adaptation LP) + x2 (inertia LP)
N_STATES           = N_CANALS * N_STATES_PER_CANAL  # 12
FLOOR              = 80.0   # deg/s — physiological resting discharge

# Pseudo-inverse: maps (N_CANALS,) canal outputs → (3,) angular velocity estimate.
# For orthogonal push-pull pairs PINV_SENS = (1/2) ORIENTATIONS^T, shape (3, 6).
PINV_SENS = jnp.linalg.pinv(ORIENTATIONS)   # (3, 6)

# Smoothing sharpness for the nonlinearity (s/deg).
# Transition width ≈ 1/_SOFTNESS ≈ 2 deg/s.
_SOFTNESS = 0.5


# ── SSM matrices ───────────────────────────────────────────────────────────────

def get_A(theta):
    """(N_STATES, N_STATES) state matrix — block structure.

    State ordering [x1_c0..x1_c5 | x2_c0..x2_c5]:

        d/dt [x1]   =  [−I/τc    0  ] [x1]   +  [ORIENTATIONS/τc] · ω
             [x2]      [ I/τs  −I/τs] [x2]      [ORIENTATIONS/τs]

    where I = identity(N_CANALS).
    """
    I   = jnp.eye(N_CANALS)
    Z   = jnp.zeros((N_CANALS, N_CANALS))
    ac  = -1.0 / theta['tau_c']
    as_ = -1.0 / theta['tau_s']
    top = jnp.concatenate([ac * I,   Z       ], axis=1)
    bot = jnp.concatenate([as_ * I,  as_ * I ], axis=1)
    return jnp.concatenate([top, bot], axis=0)   # (N_STATES, N_STATES)


def get_B(theta):
    """(N_STATES, 3) input matrix — maps 3-D head angular velocity to all states.

    B = [ORIENTATIONS/τc]   each row i: canal i's response per unit head vel
        [ORIENTATIONS/τs]
    """
    return jnp.concatenate([
        ORIENTATIONS / theta['tau_c'],
        ORIENTATIONS / theta['tau_s'],
    ], axis=0)   # (N_STATES, 3)


# ── Canal output (nonlinear) ───────────────────────────────────────────────────

def canal_nonlinearity(x_c, gains):
    """Compute bandpass afferent outputs for all canals.

    Args:
        x_c:   (N_STATES,)  canal state [x1_c0..x1_c5 | x2_c0..x2_c5]
        gains: (N_CANALS,)  per-canal scale; 0.0 = complete loss.

    Returns:
        y: (N_CANALS,)  absolute afferent firing rate (deg/s equivalent)
           At rest (x2=0): y = FLOOR (resting discharge).
           Full inhibition: y → 0.  Excitation: y > FLOOR.
    """
    x2   = x_c[N_CANALS:]                              # (N_CANALS,) inertia states
    k    = _SOFTNESS
    f    = FLOOR
    y_nl = -f + softplus(k * (x2 + f)) / k + softplus(k * (x2 - f)) / k
    return gains * (y_nl + f)   # absolute firing rate: rest = FLOOR, inhibitory → 0


def step(x_c, w_head, theta, gains):
    """Single ODE step: state derivative + afferent output.

    Args:
        x_c:    (N_STATES,)  canal state
        w_head: (3,)         head angular velocity (deg/s)
        theta:  dict         model parameters
        gains:  (N_CANALS,)  per-canal scale factors

    Returns:
        dx:       (N_STATES,)  dx_c/dt
        y_canals: (N_CANALS,)  afferent firing rates
    """
    dx = get_A(theta) @ x_c + get_B(theta) @ w_head
    y  = canal_nonlinearity(x_c, gains)
    return dx, y
