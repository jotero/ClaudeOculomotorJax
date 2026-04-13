"""Sensory model — canal array + visual delay cascade.

Aggregates the canal SSM and visual delay cascade into a single module.

Signal flow:
    w_head  → [Canal array]         → y_canals (6,)   afferent firing rates
    e_slip  → [Visual delay, sig 0] → e_slip_delayed   (for VS / OKR)
    e_pos   → [Visual delay, sig 1] → e_pos_delayed    (for saccade generator)

State vector  x_sensory = [x_c (12) | x_vis (240)]  — N_STATES = 252

Index constants (relative to x_sensory):
    _IDX_C   — canal states (12,)
    _IDX_VIS — visual delay cascade states (240,)

Inputs to step():
    w_head  (3,)  head angular velocity (deg/s)
    e_slip  (3,)  retinal velocity slip — instantaneous, efference-corrected
    e_pos   (3,)  retinal position error — instantaneous

Outputs of step():
    dx_sensory      (252,)  state derivative
    y_canals        (6,)    canal afferent firing rates
    e_slip_delayed  (3,)    delayed retinal slip   → VS / OKR
    e_pos_delayed   (3,)    delayed position error → saccade generator

Note: e_slip_delayed and e_pos_delayed are pure state readouts (C @ x_vis).
They are computed from the state at time t and reflect the signals as they
were tau_vis seconds ago — they do NOT depend on the new e_slip / e_pos
inputs passed in this step.

──────────────────────────────────────────────────────────────────────────────
Canal array (Steinhausen torsion-pendulum model)
──────────────────────────────────────────────────────────────────────────────

Each canal is a second-order bandpass filter (Steinhausen 1931; Fernandez &
Goldberg 1971 J Neurophysiol): adaptation LP (cupula) cascaded with inertia
LP (endolymph), giving the bandpass transfer function:

    H(s) = τ_c·s / [(1+s·τ_c)(1+s·τ_s)]

State layout (12,): [x1_c0..x1_c5 | x2_c0..x2_c5]
    x1 — adaptation LP state (cupula)
    x2 — inertia LP state

Canal geometry (semi-anatomical, 45° vertical canals):
    Convention: x = rightward yaw, y = upward pitch, z = CW roll
    _S = 1/√2 (sin/cos 45°)

    canal 0 — RHC  right horizontal  (yaw+)
    canal 1 — LHC  left  horizontal  (yaw−)
    canal 2 — RAC  right anterior    (RALP, pitch+ & CW roll+)
    canal 3 — LPC  left  posterior   (RALP, pitch− & CW roll−)
    canal 4 — LAC  left  anterior    (LARP, pitch+ & CCW roll)
    canal 5 — RPC  right posterior   (LARP, pitch− & CW roll)

Because ORIENTATIONS^T @ ORIENTATIONS = 2·I₃, the pseudo-inverse is
exactly PINV_SENS = (1/2)·ORIENTATIONS^T.

──────────────────────────────────────────────────────────────────────────────
Visual delay cascade (gamma-distributed delay)
──────────────────────────────────────────────────────────────────────────────

A cascade of N_STAGES first-order LP filters approximates a pure delay of
tau_vis seconds.  With N_STAGES=40 and tau_vis=0.08 s:
    Mean delay = 0.08 s,  std ≈ 0.013 s,  −3 dB BW ≈ 66 Hz

Two signals are delayed independently:
    Signal 0 — e_slip (3,)  retinal velocity slip   → OKR drive
    Signal 1 — e_pos  (3,)  retinal position error  → saccade motor error

State layout (240,): [x_slip_stage0..39 (120) | x_pos_stage0..39 (120)]

Module-level readout matrices (exported for demo scripts):
    C_slip  (3, 240)  selects last stage of slip cascade
    C_pos   (3, 240)  selects last stage of pos  cascade
"""

import jax.numpy as jnp
from jax.nn import softplus

# ── Canal geometry ─────────────────────────────────────────────────────────────

_S = 2 ** -0.5    # sin/cos 45° = 1/√2

ORIENTATIONS = jnp.array([
    [ 1.,   0.,  0.],   # canal 0 — RHC  right horizontal
    [-1.,   0.,  0.],   # canal 1 — LHC  left  horizontal
    [ 0.,  _S,  _S],   # canal 2 — RAC  right anterior
    [ 0., -_S, -_S],   # canal 3 — LPC  left  posterior
    [ 0.,  _S, -_S],   # canal 4 — LAC  left  anterior
    [ 0., -_S,  _S],   # canal 5 — RPC  right posterior
])  # (N_CANALS, 3)

N_CANALS           = ORIENTATIONS.shape[0]           # 6
_N_CANAL_STATES    = N_CANALS * 2                    # 12  [x1 (6) | x2 (6)]
FLOOR              = 80.0   # deg/s — resting discharge (Goldberg & Fernandez 1971)
_SOFTNESS          = 0.5    # nonlinearity sharpness (s/deg)

# Pseudo-inverse: maps (6,) canal outputs → (3,) angular velocity estimate
PINV_SENS = jnp.linalg.pinv(ORIENTATIONS)   # (3, 6)

# ── Visual delay cascade parameters ───────────────────────────────────────────

N_STAGES  = 40    # cascade depth
_N_SIG    = 2     # number of signals (slip + position)
_N_VIS_STATES = N_STAGES * 3 * _N_SIG   # 240
_N_PER_SIG    = N_STAGES * 3            # 120 states per signal

# ── State layout ───────────────────────────────────────────────────────────────

N_STATES = _N_CANAL_STATES + _N_VIS_STATES   # 12 + 240 = 252

# Index constants — relative to x_sensory
_IDX_C   = slice(0,               _N_CANAL_STATES)                     # (12,)
_IDX_VIS = slice(_N_CANAL_STATES, _N_CANAL_STATES + _N_VIS_STATES)     # (240,)

# ── Structural matrices (computed once at import) ─────────────────────────────

def _make_cascade_A_struct():
    """Block bidiagonal A for one signal's N_STAGES cascade."""
    A = -jnp.eye(_N_PER_SIG)
    for i in range(1, N_STAGES):
        A = A.at[i*3:(i+1)*3, (i-1)*3:i*3].set(jnp.eye(3))
    return A

_A_STRUCT     = _make_cascade_A_struct()                                   # (120, 120)
_B_STRUCT_SIG = jnp.zeros((_N_PER_SIG, 3)).at[:3, :].set(jnp.eye(3))     # (120, 3)

# Readout matrices — exported so demo scripts can apply them to x_vis directly
C_slip = jnp.zeros((3, _N_VIS_STATES)).at[:, _N_PER_SIG - 3 : _N_PER_SIG].set(jnp.eye(3))
C_pos  = jnp.zeros((3, _N_VIS_STATES)).at[:, _N_VIS_STATES - 3 :          ].set(jnp.eye(3))


# ── Shared delay cascade utilities (used by efference_copy and sensory_model) ─

def delay_cascade_step(x_cascade, signal, theta):
    """Advance one (N_STAGES × 3 = 120) delay cascade by one ODE step.

    Shared utility — called by sensory_model and efference_copy so both use
    an identical gamma-distributed delay with the same tau_vis parameter.

    Args:
        x_cascade: (120,)  cascade state
        signal:    (3,)    input signal to delay
        theta:     dict    model parameters (reads 'tau_vis')

    Returns:
        dx_cascade: (120,)  state derivative
    """
    k = N_STAGES / theta.get('tau_vis', 0.08)
    return k * _A_STRUCT @ x_cascade + k * _B_STRUCT_SIG @ signal


def delay_cascade_read(x_cascade):
    """Read the delayed output (last stage) of a cascade.

    Args:
        x_cascade: (120,)  cascade state

    Returns:
        delayed: (3,)  signal delayed by tau_vis seconds
    """
    return x_cascade[_N_PER_SIG - 3 : _N_PER_SIG]   # last 3 states


# ── Delayed signal readout ────────────────────────────────────────────────────

def read_delayed(x_sensory):
    """Read delayed retinal signals from the visual cascade state.

    Pure state readout — no derivative computation.  Returns signals as they
    were tau_vis seconds ago, independent of the current e_slip / e_pos inputs.

    Args:
        x_sensory: (252,)  sensory state [x_c (12) | x_vis (240)]

    Returns:
        e_slip_delayed: (3,)  delayed retinal slip   (for VS / OKR)
        e_pos_delayed:  (3,)  delayed position error (for saccade generator)
    """
    x_vis = x_sensory[_IDX_VIS]
    return C_slip @ x_vis, C_pos @ x_vis


# ── Canal afferent readout ────────────────────────────────────────────────────

def read_canals(x_sensory, theta):
    """Read canal afferent firing rates from the current sensory state.

    Pure state readout — no derivative computation.  y_canals is a nonlinear
    function of x_c (the inertia state) only; it does not depend on w_head.

    Args:
        x_sensory: (252,)  sensory state [x_c (12) | x_vis (240)]
        theta:     dict    model parameters (reads 'canal_gains')

    Returns:
        y_canals: (N_CANALS,)  canal afferent firing rates
    """
    x_c   = x_sensory[_IDX_C]
    gains = theta.get('canal_gains', jnp.ones(N_CANALS))
    return canal_nonlinearity(x_c, gains)


# ── Canal nonlinearity ─────────────────────────────────────────────────────────

def canal_nonlinearity(x_c, gains):
    """Compute bandpass afferent outputs for all canals.

    Args:
        x_c:   (12,)        canal state [x1_c0..x1_c5 | x2_c0..x2_c5]
        gains: (N_CANALS,)  per-canal scale; 0.0 = complete loss.

    Returns:
        y: (N_CANALS,)  absolute afferent firing rate (deg/s equivalent)
    """
    x2  = x_c[N_CANALS:]                              # (N_CANALS,) inertia states
    k   = _SOFTNESS
    f   = FLOOR
    y_nl = -f + softplus(k * (x2 + f)) / k + softplus(k * (x2 - f)) / k
    return gains * (y_nl + f)


# ── Step function ──────────────────────────────────────────────────────────────

def step(x_sensory, w_head, e_slip, e_pos, theta):
    """Single ODE step for the sensory subsystem.

    Args:
        x_sensory:  (252,)  sensory state [x_c (12) | x_vis (240)]
        w_head:     (3,)    head angular velocity (deg/s)
        e_slip:     (3,)    instantaneous retinal slip, efference-corrected (deg/s)
        e_pos:      (3,)    instantaneous retinal position error (deg)
        theta:      dict    model parameters

    Returns:
        dx_sensory:     (252,)  dx_sensory/dt
        y_canals:       (6,)    canal afferent firing rates
        e_slip_delayed: (3,)    delayed retinal slip   (for VS / OKR)
        e_pos_delayed:  (3,)    delayed position error (for saccade generator)
    """
    x_c   = x_sensory[_IDX_C]
    x_vis = x_sensory[_IDX_VIS]

    # ── Canal: system matrices ────────────────────────────────────────────────
    tau_c = theta['tau_c']
    tau_s = theta['tau_s']
    I_c = jnp.eye(N_CANALS)
    Z_c = jnp.zeros((N_CANALS, N_CANALS))
    A_c = jnp.concatenate([
        jnp.concatenate([-I_c/tau_c,   Z_c        ], axis=1),
        jnp.concatenate([ I_c/tau_s,  -I_c/tau_s  ], axis=1),
    ], axis=0)                                          # (12, 12)
    B_c = jnp.concatenate([ORIENTATIONS/tau_c,
                            ORIENTATIONS/tau_s], axis=0)  # (12, 3)

    # ── Canal: dynamics ───────────────────────────────────────────────────────
    gains    = theta.get('canal_gains', jnp.ones(N_CANALS))
    dx_c     = A_c @ x_c + B_c @ w_head
    y_canals = canal_nonlinearity(x_c, gains)

    # ── Visual delay: system matrices ─────────────────────────────────────────
    k     = N_STAGES / theta.get('tau_vis', 0.08)
    A_blk = k * _A_STRUCT                              # (120, 120)
    B_blk = k * _B_STRUCT_SIG                          # (120, 3)
    Z_sq  = jnp.zeros((_N_PER_SIG, _N_PER_SIG))
    Z_in  = jnp.zeros((_N_PER_SIG, 3))
    A_v = jnp.block([[A_blk, Z_sq], [Z_sq, A_blk]])   # (240, 240) block-diagonal
    B_v = jnp.block([[B_blk, Z_in], [Z_in, B_blk]])   # (240, 6)

    # ── Visual delay: dynamics ────────────────────────────────────────────────
    # C_slip, C_pos — module-level constants; select last stage of each cascade
    u_v            = jnp.concatenate([e_slip, e_pos])
    dx_vis         = A_v @ x_vis + B_v @ u_v
    e_slip_delayed = C_slip @ x_vis    # pure state readout — signals from tau_vis ago
    e_pos_delayed  = C_pos  @ x_vis

    # ── Pack state derivative ─────────────────────────────────────────────────
    dx_sensory = jnp.concatenate([dx_c, dx_vis])

    return dx_sensory, y_canals, e_slip_delayed, e_pos_delayed
