"""Canal array SSM — Steinhausen torsion-pendulum model.

Each canal is a second-order bandpass filter (Steinhausen 1931; Fernandez &
Goldberg 1971 J Neurophysiol):

    H(s) = τ_c·s / [(1+s·τ_c)(1+s·τ_s)]

Two cascaded first-order LPs implement the bandpass:
    x1 (adaptation / cupula):  dx1 = -(1/τ_c)·x1 + (1/τ_c)·ORIENTATIONS·w_head
    x2 (inertia / endolymph):  dx2 = -(1/τ_s)·(x1 + x2) + (1/τ_s)·ORIENTATIONS·w_head
                                    = (1/τ_s)·(w_head − x1) − (1/τ_s)·x2

Output (afferent rate) = canal_nonlinearity(x2):  soft push-pull rectification
around resting discharge FLOOR = 80 spk/s.

State layout (12,): [x1_c0..x1_c5 | x2_c0..x2_c5]
    x1 — adaptation LP (cupula), per canal
    x2 — inertia LP (endolymph), per canal — READ for afferent output

Canal geometry (semi-anatomical, 45° vertical canals):
    Convention: x = rightward yaw, y = upward pitch, z = CW roll
    _S = 1/√2 (sin/cos 45°)

    Left canals  (indices 0–2):
    canal 0 — LHC  left  horizontal  (yaw−)
    canal 1 — LAC  left  anterior    (LARP, pitch+ & CCW roll)
    canal 2 — LPC  left  posterior   (RALP, pitch− & CW roll−)

    Right canals (indices 3–5):
    canal 3 — RHC  right horizontal  (yaw+)
    canal 4 — RAC  right anterior    (RALP, pitch+ & CW roll+)
    canal 5 — RPC  right posterior   (LARP, pitch− & CW roll)

    Coplanar pairs: RALP = {RAC(4), LPC(2)};  LARP = {LAC(1), RPC(5)}

Because ORIENTATIONS^T @ ORIENTATIONS = 2·I₃, the pseudo-inverse is
exactly PINV_SENS = (1/2)·ORIENTATIONS^T.
"""

import jax.numpy as jnp
from jax.nn import softplus

# ── Canal geometry ─────────────────────────────────────────────────────────────

_S = 2 ** -0.5    # sin/cos 45° = 1/√2

ORIENTATIONS = jnp.array([
    [-1.,   0.,   0.],   # canal 0 — LHC  left  horizontal
    [ 0.,  _S,  -_S],   # canal 1 — LAC  left  anterior
    [ 0., -_S,  -_S],   # canal 2 — LPC  left  posterior
    [ 1.,   0.,   0.],   # canal 3 — RHC  right horizontal
    [ 0.,  _S,   _S],   # canal 4 — RAC  right anterior
    [ 0., -_S,   _S],   # canal 5 — RPC  right posterior
])  # (N_CANALS, 3)

N_CANALS  = ORIENTATIONS.shape[0]   # 6
N_STATES  = N_CANALS * 2            # 12  [x1 (6) | x2 (6)]


# ── State NamedTuple ──────────────────────────────────────────────────────────
from typing import NamedTuple


class State(NamedTuple):
    """Canal state — bandpass two-stage SSM (Steinhausen)."""
    x1: jnp.ndarray   # (N_CANALS,) first stage
    x2: jnp.ndarray   # (N_CANALS,) second stage (drives nonlinearity)


def rest_state():
    """Zero state — used for SimState initialisation."""
    return State(x1=jnp.zeros(N_CANALS), x2=jnp.zeros(N_CANALS))


def to_array(state):
    """canal.State → (12,) flat array — legacy adapter; SimState uses the NT directly."""
    return jnp.concatenate([state.x1, state.x2])

FLOOR     = 80.0   # deg/s — default resting discharge (Goldberg & Fernandez 1971); used as SensoryParams default
_SOFTNESS = 0.5    # nonlinearity sharpness (s/deg)

# Pseudo-inverse: maps (6,) afferents → (3,) angular velocity estimate
PINV_SENS = jnp.linalg.pinv(ORIENTATIONS)   # (3, 6)


# ── Nonlinearity ───────────────────────────────────────────────────────────────

def nonlinearity(x_c, gains, floor):
    """Soft push-pull rectification: maps inertia states → afferent firing rates.

    gains scales only the MODULATION above/below the resting discharge floor,
    not the resting discharge itself.  This models UVH as reduced sensitivity
    (fewer functioning hair cells → less head-velocity signal) while preserving
    the resting discharge of the surviving afferents.  At rest, all canals output
    floor (deg/s) regardless of gains, so there is no spurious DC signal from a
    unilateral gain reduction.  Complete deafferentation is modelled via b_vs → 0
    (VN tonic firing rate) rather than canal_gains.

    canal_gains = 1.0 → y = y_nl + floor   (healthy: resting + modulation)
    canal_gains = 0.1 → y = 0.1*y_nl + floor  (UVH: reduced sensitivity, normal resting)
    canal_gains = 0.0 → y = 0 + floor = floor  (paretic but resting preserved)

    Args:
        x_c:   (12,)        canal state [x1 (6) | x2 (6)]
        gains: (N_CANALS,)  per-canal scale; 0 = complete paresis
        floor: scalar       resting discharge (deg/s); inhibitory saturation point

    Returns:
        y: (N_CANALS,)  absolute afferent firing rate (deg/s equivalent)
    """
    x2   = x_c[N_CANALS:]                              # inertia states (6,)
    k    = _SOFTNESS
    f    = floor
    y_nl = -f + softplus(k * (x2 + f)) / k + softplus(k * (x2 - f)) / k
    return gains * y_nl + f   # resting discharge (f) is always present; gains scales modulation only


# ── SSM step ───────────────────────────────────────────────────────────────────

def step(state, w_head, sensory_params):
    """Single ODE step: canal state derivative + afferent output.

    Args:
        state:  canal.State  (x1, x2) two-stage bandpass states
        w_head: (3,)         head angular velocity (deg/s)
        theta:  Params       model parameters (reads phys.tau_c, phys.tau_s, phys.canal_gains)

    Returns:
        dstate:   canal.State   state derivative
        y_canals: (N_CANALS,)   afferent firing rates
    """
    tau_c = sensory_params.tau_c
    tau_s = sensory_params.tau_s

    # H(s) = tau_c*s / [(1+tau_c*s)(1+tau_s*s)]  — bandpass, zero at DC ✓
    # dx1 = -x1/tau_c + ORIENTATIONS·w/tau_c
    # dx2 = -(x1+x2)/tau_s + ORIENTATIONS·w/tau_s
    dx1 = (-state.x1 + ORIENTATIONS @ w_head) / tau_c
    dx2 = (-(state.x1 + state.x2) + ORIENTATIONS @ w_head) / tau_s

    # nonlinearity reads x2 internally; reuse via flat-vector convention for now
    y_canals = nonlinearity(jnp.concatenate([state.x1, state.x2]),
                             sensory_params.canal_gains, sensory_params.canal_floor)
    # Excitatory ceiling — sensor-side saturation
    y_canals = jnp.clip(y_canals, -sensory_params.canal_v_max, sensory_params.canal_v_max)
    return State(x1=dx1, x2=dx2), y_canals
