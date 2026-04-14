"""Retinal geometry + visual delay cascade.

Two responsibilities:

  1. Geometry — convert Cartesian target position to angular gaze error.
     The actual retinal error signals (e_pos, e_slip) are subtracted
     inline in the simulator so the signal-flow context is visible there.

  2. Visual delay cascade — gamma-distributed delay implemented as a
     cascade of N_STAGES first-order LP filters.  Approximates a pure
     delay of tau_vis seconds.  With N_STAGES=40 and tau_vis=0.08 s:
         Mean delay = 0.08 s,  std ≈ 0.013 s,  −3 dB BW ≈ 66 Hz

     Two signals are delayed independently (interleaved in state vector):
         Signal 0 — e_slip (3,)  retinal velocity slip   → OKR drive
         Signal 1 — e_pos  (3,)  retinal position error  → saccade motor error

     State layout (240,): [x_slip_stage0..39 (120) | x_pos_stage0..39 (120)]

     Module-level readout matrices (exported for demo scripts / efference copy):
         C_slip  (3, 240)  selects last stage of slip cascade
         C_pos   (3, 240)  selects last stage of pos  cascade
"""

import jax.numpy as jnp

# ── Cascade parameters ──────────────────────────────────────────────────────────

N_STAGES      = 40    # cascade depth
_N_SIG        = 2     # number of signals (slip + position)
N_STATES      = N_STAGES * 3 * _N_SIG   # 240  total visual delay states
_N_PER_SIG    = N_STAGES * 3            # 120  states per signal

# ── Structural matrices (computed once at import) ───────────────────────────────

def _make_cascade_A_struct():
    """Block bidiagonal A for one signal's N_STAGES cascade."""
    A = -jnp.eye(_N_PER_SIG)
    for i in range(1, N_STAGES):
        A = A.at[i*3:(i+1)*3, (i-1)*3:i*3].set(jnp.eye(3))
    return A

_A_STRUCT     = _make_cascade_A_struct()                                   # (120, 120)
_B_STRUCT_SIG = jnp.zeros((_N_PER_SIG, 3)).at[:3, :].set(jnp.eye(3))     # (120, 3)

# Readout matrices — exported so demo scripts / efference_copy can use them directly
C_slip = jnp.zeros((3, N_STATES)).at[:, _N_PER_SIG - 3 : _N_PER_SIG].set(jnp.eye(3))
C_pos  = jnp.zeros((3, N_STATES)).at[:, N_STATES - 3 :               ].set(jnp.eye(3))


# ── Geometry ────────────────────────────────────────────────────────────────────

def target_to_angle(p_target):
    """Convert Cartesian target position to angular gaze direction (deg).

    Args:
        p_target: (3,)  [x (rightward), y (upward), z (forward/depth)]

    Returns:
        (3,)  [yaw (rightward+), pitch (upward+), roll=0]  in degrees
    """
    x, y, z = p_target[0], p_target[1], p_target[2]
    yaw   = jnp.degrees(jnp.arctan2(x, z))
    pitch = jnp.degrees(jnp.arctan2(y, z))
    return jnp.array([yaw, pitch, 0.0])


# ── Shared cascade utilities (used by sensory_model and efference_copy) ─────────

def delay_cascade_step(x_cascade, signal, theta):
    """Advance one (N_STAGES × 3 = 120) delay cascade by one ODE step.

    Shared utility — called by sensory_model and efference_copy so both use
    an identical gamma-distributed delay with the same tau_vis parameter.

    Args:
        x_cascade: (120,)  cascade state
        signal:    (3,)    input signal to delay
        theta:     Params  model parameters (reads phys.tau_vis)

    Returns:
        dx_cascade: (120,)  state derivative
    """
    k = N_STAGES / theta.phys.tau_vis
    return k * _A_STRUCT @ x_cascade + k * _B_STRUCT_SIG @ signal


def delay_cascade_read(x_cascade):
    """Read the delayed output (last stage) of a cascade.

    Args:
        x_cascade: (120,)  cascade state

    Returns:
        delayed: (3,)  signal delayed by tau_vis seconds
    """
    return x_cascade[_N_PER_SIG - 3 : _N_PER_SIG]   # last 3 states


# ── Combined visual delay step ───────────────────────────────────────────────────

def step(x_vis, e_slip, e_pos, theta):
    """Single ODE step for the full visual delay cascade (both signals).

    Args:
        x_vis:  (240,)  visual cascade state [x_slip (120) | x_pos (120)]
        e_slip: (3,)    instantaneous retinal slip (deg/s)
        e_pos:  (3,)    instantaneous retinal position error (deg)
        theta:  Params  model parameters (reads phys.tau_vis)

    Returns:
        dx_vis:         (240,)  dx_vis/dt
        e_slip_delayed: (3,)    delayed retinal slip   (for VS / OKR)
        e_pos_delayed:  (3,)    delayed position error (for saccade generator)
    """
    k     = N_STAGES / theta.phys.tau_vis
    A_blk = k * _A_STRUCT                              # (120, 120)
    B_blk = k * _B_STRUCT_SIG                          # (120, 3)
    Z_sq  = jnp.zeros((_N_PER_SIG, _N_PER_SIG))
    Z_in  = jnp.zeros((_N_PER_SIG, 3))
    A_v   = jnp.block([[A_blk, Z_sq], [Z_sq, A_blk]]) # (240, 240) block-diagonal
    B_v   = jnp.block([[B_blk, Z_in], [Z_in, B_blk]]) # (240, 6)

    u_v            = jnp.concatenate([e_slip, e_pos])
    dx_vis         = A_v @ x_vis + B_v @ u_v
    e_slip_delayed = C_slip @ x_vis   # pure state readout — signals from tau_vis ago
    e_pos_delayed  = C_pos  @ x_vis
    return dx_vis, e_slip_delayed, e_pos_delayed
