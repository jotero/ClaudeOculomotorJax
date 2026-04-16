"""Retinal geometry + visual delay cascade.

Two responsibilities:

  1. Geometry — convert Cartesian target position to angular gaze error,
     compute retinal velocity of the tracked target (pursuit drive), and apply
     the visual-field gate (eccentricity limit of the retina).

  2. Visual delay cascade — gamma-distributed delay implemented as a
     cascade of N_STAGES first-order LP filters.  Approximates a pure
     delay of tau_vis seconds.  With N_STAGES=40 and tau_vis=0.08 s:
         Mean delay = 0.08 s,  std ≈ 0.013 s,  −3 dB BW ≈ 66 Hz

     Four signals are delayed independently:
         Signal 0 — e_slip    (3,)  retinal velocity slip        → OKR / VS drive
         Signal 1 — e_pos_vis (3,)  gate_vf * e_pos              → saccade error
         Signal 2 — e_vel     (3,)  target velocity on retina    → pursuit drive
         Signal 3 — gate_vf   (1,)  visual-field gate scalar     → SG target select

     Signal 1 and 3 are computed from instantaneous e_pos before entering the
     cascade so the gate is a geometric (retinal) property, not a brain decision.

     State layout (400,): [x_slip (120) | x_pos_vis (120) | x_vel (120) | x_gate (40)]

     Module-level readout matrices (exported for sensory_model / efference copy):
         C_slip  (3, 400)  selects last stage of slip cascade
         C_pos   (3, 400)  selects last stage of pos_vis cascade (gated pos)
         C_vel   (3, 400)  selects last stage of vel cascade
         C_gate  (1, 400)  selects last stage of gate cascade
"""

import jax
import jax.numpy as jnp

# ── Cascade parameters ──────────────────────────────────────────────────────────

N_STAGES      = 40    # cascade depth
_N_SIG        = 3     # number of 3-D signals (slip + pos_vis + vel)
_N_PER_SIG    = N_STAGES * 3            # 120  states per 3-D signal
_N_GATE       = N_STAGES               # 40   states for the 1-D gate channel
N_STATES      = _N_SIG * _N_PER_SIG + _N_GATE   # 360 + 40 = 400

# ── Structural matrices (computed once at import) ───────────────────────────────

def _make_cascade_A_struct():
    """Block bidiagonal A for one 3-D signal's N_STAGES cascade."""
    A = -jnp.eye(_N_PER_SIG)
    for i in range(1, N_STAGES):
        A = A.at[i*3:(i+1)*3, (i-1)*3:i*3].set(jnp.eye(3))
    return A

def _make_gate_A_struct():
    """Scalar bidiagonal A for the 1-D gate cascade (N_STAGES stages)."""
    A = -jnp.eye(_N_GATE)
    for i in range(1, _N_GATE):
        A = A.at[i, i-1].set(1.0)
    return A

_A_STRUCT     = _make_cascade_A_struct()                                    # (120, 120)
_B_STRUCT_SIG = jnp.zeros((_N_PER_SIG, 3)).at[:3, :].set(jnp.eye(3))      # (120, 3)
_A_GATE       = _make_gate_A_struct()                                       # (40, 40)
_B_GATE       = jnp.zeros(_N_GATE).at[0].set(1.0)                          # (40,)

# Readout matrices — exported so demo scripts / efference_copy can use them directly.
# Each selects the last stage of its respective cascade channel.
C_slip = jnp.zeros((3, N_STATES)).at[:, 1*_N_PER_SIG-3 : 1*_N_PER_SIG].set(jnp.eye(3))
C_pos  = jnp.zeros((3, N_STATES)).at[:, 2*_N_PER_SIG-3 : 2*_N_PER_SIG].set(jnp.eye(3))
C_vel  = jnp.zeros((3, N_STATES)).at[:, 3*_N_PER_SIG-3 : 3*_N_PER_SIG].set(jnp.eye(3))
C_gate = jnp.zeros((1, N_STATES)).at[0, N_STATES-1].set(1.0)               # (1, 400)


# ── Geometry ────────────────────────────────────────────────────────────────────

def retinal_signals(p_target, q_head, w_head, q_eye, w_eye, w_scene, v_target,
                    scene_present, vf_limit, k_vf):
    """Compute instantaneous retinal signals and visual-field gate.

    These are the raw sensory signals before neural processing or delay.
    They feed directly into the visual delay cascade.

    Visual-field gate (retinal property):
        gate_vf = 1 − σ(k_vf · (|e_pos| − vf_limit))
        ≈ 1 when target is within vf_limit deg of fovea (in visual field)
        ≈ 0 when target is beyond vf_limit deg (outside visual field)
    e_pos_vis = gate_vf · e_pos is the gated position error that enters the cascade.

    Args:
        p_target:      (3,)    Cartesian target position
        q_head:        (3,)    head angular position (deg)
        w_head:        (3,)    head angular velocity (deg/s)
        q_eye:         (3,)    eye angular position — plant state (deg)
        w_eye:         (3,)    eye angular velocity — plant derivative (deg/s)
        w_scene:       (3,)    scene angular velocity (deg/s)
        v_target:      (3,)    target angular velocity in world frame (deg/s)
        scene_present: scalar  0=dark, 1=lit — gates the slip signal
        vf_limit:      float   visual field half-width (deg); from SensoryParams
        k_vf:          float   sigmoid steepness (1/deg);     from SensoryParams

    Returns:
        e_pos:     (3,)    raw retinal position error (deg) = target − head − eye
        e_pos_vis: (3,)    visual-field-gated position error = gate_vf · e_pos
        raw_slip:  (3,)    retinal velocity slip (deg/s), zero in the dark
        e_vel:     (3,)    target velocity on retina (deg/s)
        gate_vf:   scalar  visual-field gate ∈ [0, 1]
    """
    e_pos    = target_to_angle(p_target) - q_head - q_eye
    raw_slip = scene_present * (w_scene - w_head - w_eye)
    e_vel    = v_target - w_head - w_eye

    e_mag     = jnp.linalg.norm(e_pos) + 1e-9
    gate_vf   = 1.0 - jax.nn.sigmoid(k_vf * (e_mag - vf_limit))
    e_pos_vis = gate_vf * e_pos

    return e_pos, e_pos_vis, raw_slip, e_vel, gate_vf


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

def delay_cascade_step(x_cascade, signal, tau_vis):
    """Advance one (N_STAGES × 3 = 120) 3-D delay cascade by one ODE step.

    Shared utility — called by sensory_model (with SensoryParams.tau_vis) and
    efference_copy (with BrainParams.tau_vis) so each uses its own copy.

    Args:
        x_cascade: (120,)  cascade state
        signal:    (3,)    input signal to delay
        tau_vis:   float   total cascade delay (s)

    Returns:
        dx_cascade: (120,)  state derivative
    """
    k = N_STAGES / tau_vis
    return k * _A_STRUCT @ x_cascade + k * _B_STRUCT_SIG @ signal


def delay_cascade_step_1d(x_gate, signal_1d, tau_vis):
    """Advance the 1-D gate cascade (N_STAGES = 40 scalar stages).

    Args:
        x_gate:    (40,)   gate cascade state
        signal_1d: scalar  instantaneous gate_vf value
        tau_vis:   float   total cascade delay (s) — same as the 3-D cascades

    Returns:
        dx_gate: (40,)  state derivative
    """
    k = N_STAGES / tau_vis
    return k * (_A_GATE @ x_gate + _B_GATE * signal_1d)


def delay_cascade_read(x_cascade):
    """Read the delayed output (last stage) of a 3-D cascade.

    Args:
        x_cascade: (120,)  cascade state

    Returns:
        delayed: (3,)  signal delayed by tau_vis seconds
    """
    return x_cascade[_N_PER_SIG - 3 : _N_PER_SIG]   # last 3 states


# ── Combined visual delay step ───────────────────────────────────────────────────

def step(x_vis, e_slip, e_pos_vis, e_vel, gate_vf, tau_vis):
    """Single ODE step for the full visual delay cascade (four signals).

    State layout: [x_slip (120) | x_pos_vis (120) | x_vel (120) | x_gate (40)]

    Signals delayed:
        e_slip    (3,) — retinal velocity slip          → OKR / VS
        e_pos_vis (3,) — gate_vf · e_pos (gated pos)   → SG error
        e_vel     (3,) — target velocity on retina      → pursuit
        gate_vf  scalar — visual-field gate             → SG target selection

    Args:
        x_vis:    (400,)  visual cascade state
        e_slip:   (3,)    instantaneous retinal slip (deg/s)
        e_pos_vis:(3,)    gate_vf · e_pos — gated position error (deg)
        e_vel:    (3,)    target velocity on retina (deg/s)
        gate_vf:  scalar  visual-field gate ∈ [0, 1]
        tau_vis:  float   total cascade delay (s)

    Returns:
        dx_vis:         (400,)  dx_vis/dt
        e_slip_delayed: (3,)    delayed retinal slip   (for VS / OKR)
        e_pos_delayed:  (3,)    delayed gated pos error (for saccade generator)
    """
    x_slip = x_vis[:_N_PER_SIG]
    x_pos  = x_vis[  _N_PER_SIG : 2*_N_PER_SIG]
    x_vel  = x_vis[2*_N_PER_SIG : 3*_N_PER_SIG]
    x_gate = x_vis[3*_N_PER_SIG:]                  # (40,)

    dx_slip = delay_cascade_step(x_slip, e_slip,    tau_vis)
    dx_pos  = delay_cascade_step(x_pos,  e_pos_vis, tau_vis)
    dx_vel  = delay_cascade_step(x_vel,  e_vel,     tau_vis)
    dx_gate = delay_cascade_step_1d(x_gate, gate_vf, tau_vis)

    dx_vis         = jnp.concatenate([dx_slip, dx_pos, dx_vel, dx_gate])
    e_slip_delayed = delay_cascade_read(x_slip)   # pure state readouts
    e_pos_delayed  = delay_cascade_read(x_pos)
    return dx_vis, e_slip_delayed, e_pos_delayed
