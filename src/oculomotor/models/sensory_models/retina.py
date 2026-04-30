"""Retinal geometry + visual delay cascade.

Two responsibilities:

  1. Geometry — convert Cartesian target position to angular gaze error,
     compute retinal velocity of the tracked target (pursuit drive), and apply
     the visual-field gate (eccentricity limit of the retina).

  2. Visual delay cascade — gamma-distributed delay implemented as a
     cascade of N_STAGES first-order LP filters.  Approximates a pure
     delay of tau_vis seconds.  With N_STAGES=40 and tau_vis=0.08 s:
         Mean delay = 0.08 s,  std ≈ 0.013 s,  −3 dB BW ≈ 66 Hz

     Eight signals are delayed in the SINGLE CYCLOPEAN cascade (binocular
     fusion happens BEFORE the delay in cyclopean_vision.pre_delay_fusion):
         Signal 0 — scene_angular_vel  (3,)  cyclopean rotational optic flow  → OKR / VS
         Signal 1 — scene_linear_vel   (3,)  cyclopean translational optic flow → looming
         Signal 2 — target_pos         (3,)  cyclopean target position          → saccade error
         Signal 3 — target_vel         (3,)  cyclopean target velocity (strobe-gated) → pursuit
         Signal 4 — target_disparity   (3,)  vergence disparity (diplopia-gated) → vergence
         Signal 5 — scene_visible      (1,)  delay(scene_present) — cyclopean
         Signal 6 — target_visible     (1,)  delay(target_present × target_in_vf) — cyclopean
         Signal 7 — target_motion_vis  (1,)  delay(target_visible × (1−strobe)) — pursuit gate

     State layout (720,):
         [scene_angular_vel(120) | scene_linear_vel(120) | target_pos(120)
          | target_vel(120) | target_disparity(120)
          | scene_visible(40) | target_visible(40) | target_motion_visible(40)]

     Module-level readout matrices (exported for sensory_model / efference copy):
         C_slip             (3, 720)  last stage of scene_angular_vel cascade
         C_scene_linear_vel (3, 720)  last stage of scene_linear_vel cascade
         C_pos              (3, 720)  last stage of target_pos cascade
         C_vel              (3, 720)  last stage of target_vel cascade
         C_target_disp      (3, 720)  last stage of target_disparity cascade
         C_scene_visible    (1, 720)  last stage of scene_visible cascade
         C_target_visible   (1, 720)  last stage of target_visible cascade
         C_target_motion_visible (1, 720)  last stage of target_motion_visible cascade
"""

import jax
import jax.numpy as jnp

from oculomotor.models.plant_models.readout import rotation_matrix


# ── Cascade parameters ──────────────────────────────────────────────────────────

N_STAGES      = 40    # cascade depth (stages per signal)
_N_SIG        = 5     # number of 3-D signals: scene_angular_vel, scene_linear_vel, target_pos, target_vel, target_disparity
_N_PER_SIG    = N_STAGES * 3     # 120  states per 3-D signal
_N_SCALAR     = N_STAGES         # 40   states per scalar signal
N_STATES      = _N_SIG * _N_PER_SIG + 3 * _N_SCALAR  # 5*120 + 3*40 = 720

# ── State offsets (3-D signals) ─────────────────────────────────────────────────
# State layout: [scene_angular_vel(120) | scene_linear_vel(120) | target_pos(120)
#                | target_vel(120) | target_disparity(120)
#                | scene_visible(40) | target_visible(40) | target_motion_visible(40)]

_OFF_SCENE_LINEAR = _N_PER_SIG                        # 120
_OFF_TARGET_POS   = 2 * _N_PER_SIG                    # 240
_OFF_TARGET_VEL   = 3 * _N_PER_SIG                    # 360
_OFF_TARGET_DISP  = 4 * _N_PER_SIG                    # 480  NEW: target disparity cascade
_OFF_SCENE_VIS    = 5 * _N_PER_SIG                    # 600
_OFF_TARGET_VIS   = _OFF_SCENE_VIS  + _N_SCALAR       # 640
_OFF_STROBED      = _OFF_TARGET_VIS + _N_SCALAR       # 680

# ── Readout matrices ────────────────────────────────────────────────────────────
# Exported so sensory_model / efference_copy can read cascade outputs directly.

C_slip             = jnp.zeros((3, N_STATES)).at[:, _N_PER_SIG-3         : _N_PER_SIG        ].set(jnp.eye(3))
C_scene_linear_vel = jnp.zeros((3, N_STATES)).at[:, 2*_N_PER_SIG-3      : 2*_N_PER_SIG      ].set(jnp.eye(3))
C_pos              = jnp.zeros((3, N_STATES)).at[:, 3*_N_PER_SIG-3      : 3*_N_PER_SIG      ].set(jnp.eye(3))
C_vel              = jnp.zeros((3, N_STATES)).at[:, 4*_N_PER_SIG-3      : 4*_N_PER_SIG      ].set(jnp.eye(3))
C_target_disp      = jnp.zeros((3, N_STATES)).at[:, 5*_N_PER_SIG-3      : 5*_N_PER_SIG      ].set(jnp.eye(3))
C_scene_visible    = jnp.zeros((1, N_STATES)).at[0, _OFF_SCENE_VIS  + _N_SCALAR - 1          ].set(1.0)
C_target_visible   = jnp.zeros((1, N_STATES)).at[0, _OFF_TARGET_VIS + _N_SCALAR - 1          ].set(1.0)
C_target_motion_visible = jnp.zeros((1, N_STATES)).at[0, _OFF_STROBED + _N_SCALAR - 1].set(1.0)


# ── Coordinate helpers ──────────────────────────────────────────────────────────
#
# World frame is LEFT-HANDED: x=right, y=up, z=forward  (x × y = −z).
#
# Angular vectors are stored as [yaw, pitch, roll] — NOT the same index order
# as xyz.  The mapping to xyz rotation axes used by rotation_matrix / cross():
#
#   ypr_to_xyz([yaw, pitch, roll]) = [−pitch,  yaw,  roll]
#   xyz_to_ypr([x,   y,     z  ]) = [y,       −x,   z   ]
#
#   yaw   (idx 0): rotation about +y  (left-hand: forward → right = rightward turn)
#   pitch (idx 1): rotation about −x  (left-hand: forward → up   = look up)
#   roll  (idx 2): rotation about +z  (left-hand: right → up)
#
# Always call ypr_to_xyz() before matrix ops; xyz_to_ypr() after.

def ypr_to_xyz(q):
    """[yaw, pitch, roll] (deg or deg/s) → xyz rotation-axis vector (same units)."""
    return jnp.array([-q[1], q[0], q[2]])


def xyz_to_ypr(v):
    """xyz rotation-axis vector → [yaw, pitch, roll] (same units). Inverse of ypr_to_xyz."""
    return jnp.array([v[1], -v[0], v[2]])


# ── Geometry ────────────────────────────────────────────────────────────────────

def world_to_retina(p_target, eye_offset_head, q_head, w_head, x_head,
                    q_eye, w_eye, w_scene, v_scene, dp_dt, vf_limit, k_vf):
    """Compute instantaneous retinal signals and visual-field gate.

    Angular outputs (scene_angular_vel, target_vel) are in EYE coordinates
    [yaw, pitch, roll] (deg/s).  scene_linear_vel is in eye-frame xyz [right,
    up, forward] (m/s) — no ypr reordering since it is a translational, not
    rotational, quantity.  target_pos is [yaw, pitch, 0] (deg).

    All head and eye geometry is handled here — sensory_model only passes
    anatomical offsets (eye_offset_head) and does not touch rotation matrices.

    World frame is LEFT-HANDED: x=right, y=up, z=forward  (x × y = −z).
    See module-level ypr_to_xyz / xyz_to_ypr for the [yaw,pitch,roll] ↔ xyz mapping.

    Inputs
    ------
    p_target:        target 3-D position in world frame (m)  [x=right, y=up, z=fwd]
    eye_offset_head: this eye's fixed position in head frame (m)
                     left=[-ipd/2,0,0]  right=[+ipd/2,0,0]
    q_head:          head rotation vector [yaw,pitch,roll]  (deg, world frame)
    w_head:          head angular velocity [yaw,pitch,roll]  (deg/s, world frame)
    x_head:          head linear position  [x,y,z]           (m, world frame)
    q_eye:           eye rotation vector relative to head (deg, head frame)
    w_eye:           eye angular velocity relative to head (deg/s, head frame)
    w_scene:         scene angular velocity [yaw,pitch,roll] (deg/s, world frame)
    v_scene:         scene linear velocity  [x,y,z]          (m/s,   world frame)
    dp_dt:           target Cartesian velocity [x,y,z] (m/s, world frame)
    vf_limit:        visual field half-width (deg)
    k_vf:            visual field gate sigmoid steepness (1/deg)

    Geometry
    --------
    R_head : world ← head  (from q_head rotation vector)
    R_eye  : head  ← eye   (from q_eye  rotation vector)
    R_gaze = R_head @ R_eye : world ← eye

    Target position in eye frame (exact, no small-angle approximation):
        eye_world  = x_head + R_head @ eye_offset_head   eye position in world frame
        p_from_eye = p_target − eye_world                target direction from this eye
        p_eye      = R_gaze.T @ p_hat                   target direction in eye frame
        target_pos = [arctan2(x,z), arctan2(y,√(x²+z²)), 0]  (deg, eye frame)

    Target angular velocity (computed from Cartesian position + velocity):
        v_target = xyz_to_ypr( cross(p_target, dp_dt) / |p_target|² )  [deg/s, world frame]

    Retinal velocities in eye frame:
        w_eye_world     = w_head + R_head @ w_eye             total eye angular velocity, world frame
        scene_angular_vel = R_gaze.T @ (w_scene − w_eye_world)  rotational optic flow, [yaw,pitch,roll] deg/s
        scene_linear_vel  = R_gaze.T @ v_scene                  translational optic flow, [x,y,z] m/s
        target_vel        = R_gaze.T @ (v_target − w_eye_world) target velocity on retina, [yaw,pitch,roll] deg/s

    Returns
    -------
        target_pos:       (3,)   target direction [yaw, pitch, 0] (deg)
        scene_angular_vel:(3,)   rotational optic flow [yaw,pitch,roll] (deg/s)
        scene_linear_vel: (3,)   translational optic flow [x,y,z] (m/s, eye frame)
        target_vel:       (3,)   target velocity on retina [yaw,pitch,roll] (deg/s)
        target_in_vf:     scalar geometric visual-field gate ∈ [0, 1]
    """
    # ── Rotation matrices ─────────────────────────────────────────────────────
    R_head   = rotation_matrix(ypr_to_xyz(q_head))   # world ← head
    R_eye    = rotation_matrix(ypr_to_xyz(q_eye))    # head  ← eye
    R_gaze_T = R_eye.T @ R_head.T                    # world → eye frame

    # ── Target position in eye frame ──────────────────────────────────────────
    eye_world  = x_head + R_head @ eye_offset_head           # eye position, world frame
    p_from_eye = p_target - eye_world                        # target from this eye, world frame
    p_hat      = p_from_eye / (jnp.linalg.norm(p_from_eye) + 1e-9)
    p_eye      = R_gaze_T @ p_hat                            # target direction, eye frame

    yaw_e   = jnp.degrees(jnp.arctan2(p_eye[0], p_eye[2]))
    pitch_e = jnp.degrees(jnp.arctan2(p_eye[1], jnp.sqrt(p_eye[0]**2 + p_eye[2]**2)))
    target_pos = jnp.array([yaw_e, pitch_e, 0.0])           # roll=0: target direction has only 2 DOF

    # ── Retinal velocities in eye frame ───────────────────────────────────────
    # Angular velocities: convert ypr→xyz before rotation matrix ops, xyz→ypr after.
    # Without this, sustained rotation rotates the yaw axis into pitch/roll, causing
    # OKR to fight VOR.
    w_head_xyz  = ypr_to_xyz(w_head)
    w_eye_xyz   = ypr_to_xyz(w_eye)
    w_scene_xyz = ypr_to_xyz(w_scene)
    target_dist = jnp.sqrt(jnp.dot(p_target, p_target)) + 1e-9
    v_target    = jnp.degrees(xyz_to_ypr(jnp.cross(p_target, dp_dt)) / target_dist ** 2)
    vt_xyz      = ypr_to_xyz(v_target)
    w_eye_world = w_head_xyz + R_head @ w_eye_xyz

    scene_angular_vel = xyz_to_ypr(R_gaze_T @ (w_scene_xyz - w_eye_world))  # [yaw,pitch,roll] deg/s
    scene_linear_vel  = R_gaze_T @ v_scene                                   # [x,y,z] m/s, eye frame
    target_vel        = xyz_to_ypr(R_gaze_T @ (vt_xyz - w_eye_world))        # [yaw,pitch,roll] deg/s
    target_vel        = target_vel.at[2].set(0.0)   # retina is 2D: target translates H/V only

    # ── Visual-field gate ─────────────────────────────────────────────────────
    e_mag  = jnp.linalg.norm(target_pos) + 1e-9
    target_in_vf = 1.0 - jax.nn.sigmoid(k_vf * (e_mag - vf_limit))

    return target_pos, scene_angular_vel, scene_linear_vel, target_vel, target_in_vf


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


