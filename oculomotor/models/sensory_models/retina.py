"""Retinal geometry + visual delay cascade.

Two responsibilities:

  1. Geometry — convert Cartesian target position to angular gaze error and
     compute retinal velocity of the tracked target (pursuit drive).

  2. Visual delay cascade — gamma-distributed delay implemented as a
     cascade of N_STAGES first-order LP filters.  Approximates a pure
     delay of tau_vis seconds.  With N_STAGES=40 and tau_vis=0.08 s:
         Mean delay = 0.08 s,  std ≈ 0.013 s,  −3 dB BW ≈ 66 Hz

     Three signals are delayed independently (interleaved in state vector):
         Signal 0 — e_slip (3,)  retinal velocity slip   → OKR / VS drive
         Signal 1 — e_pos  (3,)  retinal position error  → saccade motor error
         Signal 2 — e_vel  (3,)  target velocity on retina → pursuit drive

     State layout (360,): [x_slip_stage0..39 (120) | x_pos_stage0..39 (120)
                                                    | x_vel_stage0..39 (120)]

     Module-level readout matrices (exported for demo scripts / efference copy):
         C_slip  (3, 360)  selects last stage of slip cascade
         C_pos   (3, 360)  selects last stage of pos  cascade
         C_vel   (3, 360)  selects last stage of vel  cascade
"""

import jax.numpy as jnp

# ── Cascade parameters ──────────────────────────────────────────────────────────

N_STAGES      = 40    # cascade depth
_N_SIG        = 3     # number of signals (slip + position + target velocity)
N_STATES      = N_STAGES * 3 * _N_SIG   # 360  total visual delay states
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
# Each selects the last 3 states of its respective 120-state channel.
C_slip = jnp.zeros((3, N_STATES)).at[:, 1*_N_PER_SIG-3 : 1*_N_PER_SIG].set(jnp.eye(3))
C_pos  = jnp.zeros((3, N_STATES)).at[:, 2*_N_PER_SIG-3 : 2*_N_PER_SIG].set(jnp.eye(3))
C_vel  = jnp.zeros((3, N_STATES)).at[:, 3*_N_PER_SIG-3 : 3*_N_PER_SIG].set(jnp.eye(3))


# ── Geometry ────────────────────────────────────────────────────────────────────

def retinal_signals(p_target, q_head, w_head, q_eye, w_eye, w_scene, v_target, scene_present):
    """Compute instantaneous retinal position error, velocity slip, and target velocity.

    These are the raw sensory signals before any neural processing or delay.
    They feed directly into the visual delay cascade.

    Args:
        p_target:      (3,)    Cartesian target position
        q_head:        (3,)    head angular position (deg)
        w_head:        (3,)    head angular velocity (deg/s)
        q_eye:         (3,)    eye angular position — plant state (deg)
        w_eye:         (3,)    eye angular velocity — plant derivative (deg/s)
        w_scene:       (3,)    scene angular velocity (deg/s)
        v_target:      (3,)    target angular velocity in world frame (deg/s)
                               Analogous to w_scene but for the foveal target.
                               → pursuit drive via target velocity on retina
        scene_present: scalar  0=dark, 1=lit — gates the slip signal

    Returns:
        e_pos:    (3,)  retinal position error (deg)
                        = target_direction − head_position − eye_position
        raw_slip: (3,)  retinal velocity slip (deg/s), zero in the dark
                        = scene_present · (w_scene − w_head − w_eye)
        e_vel:    (3,)  target velocity on retina (deg/s)
                        = v_target − w_head − w_eye
                        → drives smooth pursuit; zero when eye perfectly tracks target
                        Gating (scene vs. target presence) is handled downstream:
                        brain_model gates e_combined by target_present before driving pursuit.
    """
    e_pos    = target_to_angle(p_target) - q_head - q_eye
    raw_slip = scene_present * (w_scene - w_head - w_eye)
    e_vel    = v_target - w_head - w_eye
    return e_pos, raw_slip, e_vel


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
    """Advance one (N_STAGES × 3 = 120) delay cascade by one ODE step.

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


def delay_cascade_read(x_cascade):
    """Read the delayed output (last stage) of a cascade.

    Args:
        x_cascade: (120,)  cascade state

    Returns:
        delayed: (3,)  signal delayed by tau_vis seconds
    """
    return x_cascade[_N_PER_SIG - 3 : _N_PER_SIG]   # last 3 states


# ── Combined visual delay step ───────────────────────────────────────────────────

def step(x_vis, e_slip, e_pos, e_vel, tau_vis):
    """Single ODE step for the full visual delay cascade (three signals).

    Each of the three signals is delayed independently through its own
    120-state cascade.  State layout: [x_slip (120) | x_pos (120) | x_vel (120)].

    Args:
        x_vis:   (360,)  visual cascade state
        e_slip:  (3,)    instantaneous retinal slip (deg/s)     → OKR / VS
        e_pos:   (3,)    instantaneous retinal position error (deg) → SG
        e_vel:   (3,)    target velocity on retina (deg/s)      → pursuit
        tau_vis: float   total cascade delay (s)

    Returns:
        dx_vis:         (360,)  dx_vis/dt
        e_slip_delayed: (3,)    delayed retinal slip   (for VS / OKR)
        e_pos_delayed:  (3,)    delayed position error (for saccade generator)
    """
    x_slip = x_vis[:_N_PER_SIG]
    x_pos  = x_vis[_N_PER_SIG : 2*_N_PER_SIG]
    x_vel  = x_vis[2*_N_PER_SIG:]

    dx_slip = delay_cascade_step(x_slip, e_slip, tau_vis)
    dx_pos  = delay_cascade_step(x_pos,  e_pos,  tau_vis)
    dx_vel  = delay_cascade_step(x_vel,  e_vel,  tau_vis)

    dx_vis         = jnp.concatenate([dx_slip, dx_pos, dx_vel])
    e_slip_delayed = delay_cascade_read(x_slip)   # pure state readout
    e_pos_delayed  = delay_cascade_read(x_pos)
    return dx_vis, e_slip_delayed, e_pos_delayed
