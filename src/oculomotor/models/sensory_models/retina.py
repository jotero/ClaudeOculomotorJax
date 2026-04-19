"""Retinal geometry + visual delay cascade.

Two responsibilities:

  1. Geometry — convert Cartesian target position to angular gaze error,
     compute retinal velocity of the tracked target (pursuit drive), and apply
     the visual-field gate (eccentricity limit of the retina).

  2. Visual delay cascade — gamma-distributed delay implemented as a
     cascade of N_STAGES first-order LP filters.  Approximates a pure
     delay of tau_vis seconds.  With N_STAGES=40 and tau_vis=0.08 s:
         Mean delay = 0.08 s,  std ≈ 0.013 s,  −3 dB BW ≈ 66 Hz

     Six signals are delayed independently:
         Signal 0 — scene_vel       (3,)  scene velocity on retina    → OKR / VS drive
         Signal 1 — target_pos      (3,)  target position on retina   → saccade error
         Signal 2 — target_vel      (3,)  target velocity on retina   → pursuit drive
         Signal 3 — target_in_vf         (1,)  geometric visual-field gate → SG target select
         Signal 4 — scene_visible   (1,)  delay(scene_present) — presence only, no geometry
         Signal 5 — target_visible  (1,)  delay(target_present × target_in_vf) — presence × geometry

     All signals enter the cascade raw (no pre-gating).  Signals 4 and 5 are
     the delayed "observed" flags; sensory_model.read_outputs() multiplies
     them by the raw delayed signals at readout so that zero in the cascade
     unambiguously means "the signal was zero", not "it was unobserved".

     State layout (480,):
         [x_scene_vel (120) | x_target_pos (120) | x_target_vel (120)
          | x_target_in_vf (40) | x_scene_visible (40) | x_target_visible (40)]

     Module-level readout matrices (exported for sensory_model / efference copy):
         C_slip           (3, 480)  last stage of scene_vel cascade
         C_pos            (3, 480)  last stage of target_pos cascade
         C_vel            (3, 480)  last stage of target_vel cascade
         C_target_in_vf           (1, 480)  last stage of target_in_vf cascade
         C_scene_visible  (1, 480)  last stage of scene_visible cascade
         C_target_visible (1, 480)  last stage of target_visible cascade
"""

import jax
import jax.numpy as jnp

from oculomotor.models.plant_models.readout import rotation_matrix

# ── Cascade parameters ──────────────────────────────────────────────────────────

N_STAGES      = 40    # cascade depth (stages per signal)
_N_SIG        = 3     # number of 3-D signals (scene_vel, target_pos, target_vel)
_N_PER_SIG    = N_STAGES * 3     # 120  states per 3-D signal
_N_SCALAR     = N_STAGES         # 40   states per scalar signal
N_STATES      = _N_SIG * _N_PER_SIG + 3 * _N_SCALAR  # 360 + 120 = 480

# ── State offsets ───────────────────────────────────────────────────────────────

_OFF_GATE         = _N_SIG * _N_PER_SIG          # 360
_OFF_SCENE_VIS    = _OFF_GATE + _N_SCALAR         # 400
_OFF_TARGET_VIS   = _OFF_SCENE_VIS + _N_SCALAR    # 440

# ── Readout matrices ────────────────────────────────────────────────────────────
# Exported so sensory_model / efference_copy can read cascade outputs directly.

C_slip           = jnp.zeros((3, N_STATES)).at[:, _N_PER_SIG-3       : _N_PER_SIG      ].set(jnp.eye(3))
C_pos            = jnp.zeros((3, N_STATES)).at[:, 2*_N_PER_SIG-3     : 2*_N_PER_SIG    ].set(jnp.eye(3))
C_vel            = jnp.zeros((3, N_STATES)).at[:, 3*_N_PER_SIG-3     : 3*_N_PER_SIG    ].set(jnp.eye(3))
C_target_in_vf   = jnp.zeros((1, N_STATES)).at[0, _OFF_GATE + _N_SCALAR - 1             ].set(1.0)
C_scene_visible  = jnp.zeros((1, N_STATES)).at[0, _OFF_SCENE_VIS + _N_SCALAR - 1        ].set(1.0)
C_target_visible = jnp.zeros((1, N_STATES)).at[0, _OFF_TARGET_VIS + _N_SCALAR - 1       ].set(1.0)


# ── Geometry ────────────────────────────────────────────────────────────────────

def retinal_signals(p_target, eye_offset_head, q_head, w_head, q_eye, w_eye,
                    w_scene, v_target, vf_limit, k_vf):
    """Compute instantaneous retinal signals and visual-field gate.

    All outputs are in EYE coordinates [yaw, pitch, roll] (deg or deg/s).
    All head and eye geometry is handled here — sensory_model only passes
    anatomical offsets (eye_offset_head) and does not touch rotation matrices.

    Inputs
    ------
    p_target:        world frame, HCR origin (stereographic, x=right y=up z=fwd)
    eye_offset_head: this eye's fixed position in head frame (m)
                     left=[-ipd/2,0,0]  right=[+ipd/2,0,0]
    q_head:          head rotation vector (deg, world frame)
    w_head:          head angular velocity (deg/s, world frame)
    q_eye:           eye rotation vector relative to head (deg, head frame)
    w_eye:           eye angular velocity relative to head (deg/s, head frame)
    w_scene:         scene angular velocity (deg/s, world frame)
    v_target:        target angular velocity (deg/s, world frame)
    vf_limit:        visual field half-width (deg)
    k_vf:            visual field gate sigmoid steepness (1/deg)

    Geometry
    --------
    R_head : world ← head  (from q_head rotation vector)
    R_eye  : head  ← eye   (from q_eye  rotation vector)
    R_gaze = R_head @ R_eye : world ← eye

    Target position in eye frame (exact, no small-angle approximation):
        eye_world  = R_head @ eye_offset_head     eye position in world frame
        p_from_eye = p_target − eye_world         target direction from this eye
        p_eye      = R_gaze.T @ p_hat             target direction in eye frame
        target_pos = [arctan2(x,z), arctan2(y,√(x²+z²)), 0]  (deg, eye frame)

    Retinal velocities in eye frame (exact):
        w_eye_world = w_head + R_head @ w_eye     total eye velocity, world frame
        scene_vel   = R_gaze.T @ (w_scene  − w_eye_world)   eye-frame, deg/s
        target_vel  = R_gaze.T @ (v_target − w_eye_world)   eye-frame, deg/s

    Returns  (all in eye coordinates)
    -----------------------------------
        target_pos: (3,)   target direction [yaw, pitch, 0] (deg)
        scene_vel:  (3,)   full-field scene velocity on retina (deg/s)
        target_vel: (3,)   target velocity on retina (deg/s)
        target_in_vf:    scalar geometric visual-field gate ∈ [0, 1]
    """
    # ── Rotation matrices ─────────────────────────────────────────────────────
    R_head   = rotation_matrix(q_head)      # world ← head
    R_eye    = rotation_matrix(q_eye)       # head  ← eye
    R_gaze_T = R_eye.T @ R_head.T          # world → eye frame

    # ── Target position in eye frame ──────────────────────────────────────────
    eye_world  = R_head @ eye_offset_head                    # eye position, world frame
    p_from_eye = p_target - eye_world                        # target from this eye, world frame
    p_hat      = p_from_eye / (jnp.linalg.norm(p_from_eye) + 1e-9)
    p_eye      = R_gaze_T @ p_hat                            # target direction, eye frame

    yaw_e   = jnp.degrees(jnp.arctan2(p_eye[0], p_eye[2]))
    pitch_e = jnp.degrees(jnp.arctan2(p_eye[1], jnp.sqrt(p_eye[0]**2 + p_eye[2]**2)))
    target_pos = jnp.array([yaw_e, pitch_e, 0.0])           # roll=0: target direction has only 2 DOF

    # ── Retinal velocities in eye frame ───────────────────────────────────────
    w_eye_world = w_head + R_head @ w_eye                    # total eye velocity, world frame
    scene_vel   = R_gaze_T @ (w_scene  - w_eye_world)       # eye frame, deg/s
    target_vel  = R_gaze_T @ (v_target - w_eye_world)       # eye frame, deg/s

    # ── Visual-field gate ─────────────────────────────────────────────────────
    e_mag  = jnp.linalg.norm(target_pos) + 1e-9
    target_in_vf = 1.0 - jax.nn.sigmoid(k_vf * (e_mag - vf_limit))

    return target_pos, scene_vel, target_vel, target_in_vf


# ── Delay cascade ───────────────────────────────────────────────────────────────

def delay_cascade_step(x, u, tau_vis, N=N_STAGES):
    """Advance a delay cascade of N stages for any signal shape.

    Works for both 3-D signals (120 states each) and scalar signals (40 states)
    with the same code.  A and B are built from N and the signal width at JAX
    trace time, so there is no runtime overhead vs pre-computed matrices.

    Args:
        x:       (N*n,)          cascade state  (n = signal width)
        u:       (n,) or scalar  input signal
        tau_vis: float           total cascade delay (s)
        N:       int             number of stages (default N_STAGES=40)

    Returns:
        dx: (N*n,)  state derivative
    """
    u1d = jnp.atleast_1d(u)
    n   = u1d.shape[0]        # signal width — static at JAX trace time
    ns  = N * n               # total states
    A = -jnp.eye(ns)
    for i in range(1, N):
        A = A.at[i*n:(i+1)*n, (i-1)*n:i*n].set(jnp.eye(n))
    B = jnp.zeros((ns, n)).at[:n].set(jnp.eye(n))
    k = N / tau_vis
    return k * (A @ x + B @ u1d)


def delay_cascade_read(x_cascade):
    """Read the delayed output (last stage) of a 3-D cascade.

    Args:
        x_cascade: (120,)  cascade state

    Returns:
        delayed: (3,)  signal delayed by tau_vis seconds
    """
    return x_cascade[_N_PER_SIG - 3 : _N_PER_SIG]


# ── Combined visual delay step ───────────────────────────────────────────────────

def step(x_vis, scene_vel, target_pos, target_vel, target_in_vf,
         scene_present, target_visible, tau_vis):
    """Single ODE step for the full visual delay cascade (six signals).

    All signals enter the cascade raw — no pre-gating.  Gating is applied at
    readout (sensory_model.read_outputs) using the delayed presence signals
    scene_visible and target_visible.

    State layout: [x_scene_vel(120) | x_target_pos(120) | x_target_vel(120)
                   | x_target_in_vf(40) | x_scene_visible(40) | x_target_visible(40)]

    Args:
        x_vis:          (480,)  visual cascade state
        scene_vel:      (3,)    scene velocity on retina, eye frame (deg/s)
        target_pos:     (3,)    target position on retina, eye frame (deg)
        target_vel:     (3,)    target velocity on retina, eye frame (deg/s)
        target_in_vf:        scalar  geometric visual-field gate ∈ [0, 1]
        scene_present:  scalar  is scene physically present? ∈ {0, 1}
        target_visible: scalar  target_present × target_in_vf ∈ [0, 1]
        tau_vis:        float   total cascade delay (s)

    Returns:
        dx_vis: (480,)  dx_vis/dt
    """
    x_scene_vel   = x_vis[                 :   _N_PER_SIG]
    x_target_pos  = x_vis[    _N_PER_SIG   : 2*_N_PER_SIG]
    x_target_vel  = x_vis[  2*_N_PER_SIG   : 3*_N_PER_SIG]
    x_target_in_vf     = x_vis[ _OFF_GATE        : _OFF_SCENE_VIS]
    x_scene_vis   = x_vis[ _OFF_SCENE_VIS   : _OFF_TARGET_VIS]
    x_target_vis  = x_vis[ _OFF_TARGET_VIS  :]

    dx_scene_vel  = delay_cascade_step(x_scene_vel,  scene_vel,       tau_vis)
    dx_target_pos = delay_cascade_step(x_target_pos, target_pos,      tau_vis)
    dx_target_vel = delay_cascade_step(x_target_vel, target_vel,      tau_vis)
    dx_target_in_vf    = delay_cascade_step(x_target_in_vf,    target_in_vf,         tau_vis)
    dx_scene_vis  = delay_cascade_step(x_scene_vis,  scene_present,   tau_vis)
    dx_target_vis = delay_cascade_step(x_target_vis, target_visible,  tau_vis)

    return jnp.concatenate([dx_scene_vel, dx_target_pos, dx_target_vel,
                             dx_target_in_vf, dx_scene_vis, dx_target_vis])
