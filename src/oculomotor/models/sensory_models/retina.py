"""Retinal geometry + visual delay cascade.

Two responsibilities:

  1. Geometry — convert Cartesian target position to angular gaze error,
     compute retinal velocity of the tracked target (pursuit drive), and apply
     the visual-field gate (eccentricity limit of the retina).

  2. Visual delay cascade — gamma-distributed delay implemented as a
     cascade of N_STAGES first-order LP filters.  Approximates a pure
     delay of tau_vis seconds.  With N_STAGES=40 and tau_vis=0.08 s:
         Mean delay = 0.08 s,  std ≈ 0.013 s,  −3 dB BW ≈ 66 Hz

     Ten signals are delayed in the SINGLE CYCLOPEAN cascade (binocular
     fusion happens BEFORE the delay in cyclopean_vision.pre_delay_fusion):
         Signal 0 — scene_angular_vel  (3,)  cyclopean rotational optic flow  → OKR / VS
         Signal 1 — scene_linear_vel   (3,)  cyclopean translational optic flow → looming
         Signal 2 — target_pos         (3,)  cyclopean target position          → saccade error
         Signal 3 — target_vel         (3,)  cyclopean target velocity (strobe-gated) → pursuit
         Signal 4 — target_disparity   (3,)  vergence disparity (diplopia-gated) → vergence
         Signal 5 — scene_disp_rate    (3,)  per-eye scene-flow differential  → vergence + HE
         Signal 6 — scene_visible      (1,)  delay(scene_present) — cyclopean
         Signal 7 — target_visible     (1,)  delay(target_present × target_in_vf) — cyclopean
         Signal 8 — target_motion_vis  (1,)  delay(target_visible × (1−strobe)) — pursuit gate
         Signal 9 — target_fusable     (1,)  delay(NPC fusion gate) — binocular fusability

Implicit depth-map assumption
-----------------------------
The very fact that this retina extracts both *rotational* (scene_angular_vel) and
*translational* (scene_linear_vel) optic flow as separable quantities is itself a
strong assumption: it requires the visual system to have already solved the depth
problem.  In a depthless world you can only infer rigid-body angular flow; you
cannot decompose head-translation parallax from rotation without depth structure
(the heading direction needs the focus-of-expansion of the depth-aware flow field).

We side-step this by assuming the brain has a depth map (e.g., from binocular
disparity, motion parallax, accommodation cues) and can therefore expose:
    - clean angular flow (scene_angular_vel) as if from rigid rotation
    - clean translational flow (scene_linear_vel) as the head-translation parallax
    - per-eye flow differential (scene_disp_rate) as a depth-rate cue

In practice for a depthless or uniform scene, scene_disp_rate degenerates to 0,
which the brain correctly interprets as "no depth-rate change" → constrains
heading-z and vergence-rate estimates.  This is the right zero-point behavior;
in a depth-structured scene the same signal would carry rich heading information.
"""

import jax
import jax.numpy as jnp

from oculomotor.models.plant_models.readout import rotation_matrix


# ── Cascade parameters ──────────────────────────────────────────────────────────

N_STAGES      = 40    # cascade depth (stages per signal)
_N_SIG        = 6     # 3-D signals: scene_angular_vel, scene_linear_vel, target_pos,
                      #              target_vel, target_disparity, scene_disp_rate
_N_PER_SIG    = N_STAGES * 3     # 120  states per 3-D signal
_N_SCALAR     = N_STAGES         # 40   states per scalar signal
N_STATES      = _N_SIG * _N_PER_SIG + 5 * _N_SCALAR  # 6*120 + 5*40 = 920

# ── State offsets ───────────────────────────────────────────────────────────────
# State layout: [scene_angular_vel(120) | scene_linear_vel(120) | target_pos(120)
#                | target_vel(120) | target_disparity(120) | scene_disp_rate(120)
#                | scene_visible(40) | target_visible(40) | target_motion_visible(40)
#                | target_fusable(40) | defocus(40)]

_OFF_SCENE_LINEAR    = _N_PER_SIG                            # 120
_OFF_TARGET_POS      = 2 * _N_PER_SIG                        # 240
_OFF_TARGET_VEL      = 3 * _N_PER_SIG                        # 360
_OFF_TARGET_DISP     = 4 * _N_PER_SIG                        # 480
_OFF_SCENE_DISP_RATE = 5 * _N_PER_SIG                        # 600
_OFF_SCENE_VIS       = 6 * _N_PER_SIG                        # 720
_OFF_TARGET_VIS      = _OFF_SCENE_VIS    + _N_SCALAR          # 760
_OFF_STROBED         = _OFF_TARGET_VIS   + _N_SCALAR          # 800
_OFF_TARGET_FUSABLE  = _OFF_STROBED      + _N_SCALAR          # 840
_OFF_DEFOCUS         = _OFF_TARGET_FUSABLE + _N_SCALAR        # 880

# ── Readout matrices ────────────────────────────────────────────────────────────
# Exported so sensory_model / efference_copy can read cascade outputs directly.

C_slip             = jnp.zeros((3, N_STATES)).at[:, _N_PER_SIG-3         : _N_PER_SIG        ].set(jnp.eye(3))
C_scene_linear_vel = jnp.zeros((3, N_STATES)).at[:, 2*_N_PER_SIG-3      : 2*_N_PER_SIG      ].set(jnp.eye(3))
C_pos              = jnp.zeros((3, N_STATES)).at[:, 3*_N_PER_SIG-3      : 3*_N_PER_SIG      ].set(jnp.eye(3))
C_vel              = jnp.zeros((3, N_STATES)).at[:, 4*_N_PER_SIG-3      : 4*_N_PER_SIG      ].set(jnp.eye(3))
C_target_disp      = jnp.zeros((3, N_STATES)).at[:, 5*_N_PER_SIG-3      : 5*_N_PER_SIG      ].set(jnp.eye(3))
C_scene_disp_rate  = jnp.zeros((3, N_STATES)).at[:, 6*_N_PER_SIG-3      : 6*_N_PER_SIG      ].set(jnp.eye(3))
C_scene_visible    = jnp.zeros((1, N_STATES)).at[0, _OFF_SCENE_VIS     + _N_SCALAR - 1].set(1.0)
C_target_visible   = jnp.zeros((1, N_STATES)).at[0, _OFF_TARGET_VIS    + _N_SCALAR - 1].set(1.0)
C_target_motion_visible = jnp.zeros((1, N_STATES)).at[0, _OFF_STROBED   + _N_SCALAR - 1].set(1.0)
C_target_fusable   = jnp.zeros((1, N_STATES)).at[0, _OFF_TARGET_FUSABLE + _N_SCALAR - 1].set(1.0)
C_defocus          = jnp.zeros((1, N_STATES)).at[0, _OFF_DEFOCUS        + _N_SCALAR - 1].set(1.0)


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

def world_to_retina(p_target, eye_offset_head, q_head, w_head, x_head, v_head,
                    q_eye, w_eye, w_scene, v_scene, dp_dt,
                    scene_present, target_present, vf_limit, k_vf):
    """Compute instantaneous retinal signals and per-eye visibility gates.

    Angular outputs (scene_angular_vel, target_vel) are in EYE coordinates
    [yaw, pitch, roll] (deg/s).  scene_linear_vel is in HEAD-frame xyz [right,
    up, forward] (m/s), with per-eye parallax (each eye is offset from the head
    centre, so head rotation moves the two eyes at different velocities — they
    therefore see different translational optic flow even when v_head is shared).
    Computing head-frame at retina (using actual q_eye implicitly via R_eye in
    R_gaze) is mathematically equivalent to delaying an eye-frame signal and
    derotating later by a delay-matched ec_pos, but is simpler to implement.
    target_pos is [yaw, pitch, 0] (deg).

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
    v_head:          head linear velocity  [x,y,z]           (m/s, world frame)
    q_eye:           eye rotation vector relative to head (deg, head frame)
    w_eye:           eye angular velocity relative to head (deg/s, head frame)
    w_scene:         scene angular velocity [yaw,pitch,roll] (deg/s, world frame)
    v_scene:         scene linear velocity  [x,y,z]          (m/s,   world frame)
    dp_dt:           target Cartesian velocity [x,y,z] (m/s, world frame)
    scene_present:   scalar ∈ [0,1] — is the scene lit for this eye?
    target_present:  scalar ∈ [0,1] — is the target visible (not occluded) for this eye?
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
        v_eye_world     = v_head + ω_head × (R_head @ eye_offset_head)
                          per-eye linear velocity in world frame; second term is the
                          parallax velocity from head rotation moving an eccentric eye.
        scene_angular_vel = R_gaze.T @ (w_scene − w_eye_world)  rotational optic flow, [yaw,pitch,roll] deg/s
        scene_linear_vel  = R_head.T @ (v_scene − v_eye_world)  translational optic flow, [x,y,z] m/s, HEAD frame, per-eye
        target_vel        = R_gaze.T @ (v_target − w_eye_world) target velocity on retina, [yaw,pitch,roll] deg/s

    Returns
    -------
        target_pos:       (3,)   target direction [yaw, pitch, 0] (deg)
        scene_angular_vel:(3,)   rotational optic flow [yaw,pitch,roll] (deg/s)
        scene_linear_vel: (3,)   translational optic flow [x,y,z] (m/s, head frame, per-eye)
        target_vel:       (3,)   target velocity on retina [yaw,pitch,roll] (deg/s)
        scene_vis:        scalar scene presence gate = scene_present ∈ [0,1]
        target_vis:       scalar combined target gate = target_present × target_in_vf ∈ [0,1]
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

    # Rotation-vector extraction (axis-angle), CONSISTENT with how q_eye and other
    # angular positions are treated throughout the code.  Fick-style arctan2 would
    # give numerically different yaw/pitch at non-primary positions (~1° at 30° gaze),
    # and rotation_matrix(ypr_to_xyz(target_pos)) applied to (0,0,1) wouldn't recover
    # the target direction.  With rotation-vector extraction this is exact.
    #
    # IMPLICIT LISTING'S LAW: this extraction inherently produces a Listing-compliant
    # target position.  The rotation that takes the primary direction (0,0,1) to the
    # target direction has its axis in the (x,y) plane — i.e., perpendicular to the
    # primary direction — which is exactly Listing's plane.  So target_pos[2] (torsion)
    # is 0 not because "target has only 2 DOF" but because Listing's law says the
    # eye rotation from primary stays in Listing's plane.  A non-Listing eye position
    # would require a non-(x,y) rotation axis, which this representation can't express.
    #
    # TODO: factor into a helper function `direction_to_rotvec_ypr(p_eye, primary)`
    # that takes a unit gaze direction and returns the YPR-style rotation vector.
    # Same logic should be reusable for any "look-at" geometry.
    #
    # Algorithm: rotation that takes (0,0,1) to p_eye is
    #   axis  = (0,0,1) × p_eye  = (−p_eye[1], p_eye[0], 0)
    #   angle = arctan2(|axis|, p_eye[2])     (robust)
    #   q_xyz = axis · (angle / |axis|)        (rotation vector, rad)
    # Then xyz_to_ypr → [yaw, pitch, roll=0] in degrees.
    axis_unscaled = jnp.array([-p_eye[1], p_eye[0], 0.0])
    r             = jnp.sqrt(p_eye[0]**2 + p_eye[1]**2)
    angle         = jnp.arctan2(r, p_eye[2])
    scale         = jnp.where(r > 1e-9, angle / (r + 1e-9), 1.0)
    q_xyz         = axis_unscaled * scale
    q_ypr         = jnp.degrees(xyz_to_ypr(q_xyz))    # [yaw, pitch, roll=0]
    target_pos    = jnp.array([q_ypr[0], q_ypr[1], 0.0])  # roll=0: implicit Listing's compliance

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
    # Per-eye linear velocity in world frame: v_head + ω_head × eye_offset_world.
    # The cross-product term is the parallax velocity — for a head rotating about its
    # own centre, an eccentric eye traces a small arc, and that motion contributes to
    # the optic flow at that eye even when the head is not translating.
    omega_head_rad = jnp.radians(w_head_xyz)
    v_eye_world    = v_head + jnp.cross(omega_head_rad, R_head @ eye_offset_head)
    scene_linear_vel  = R_head.T @ (v_scene - v_eye_world)                   # [x,y,z] m/s, HEAD frame, per-eye
    target_vel        = xyz_to_ypr(R_gaze_T @ (vt_xyz - w_eye_world))        # [yaw,pitch,roll] deg/s
    target_vel        = target_vel.at[2].set(0.0)   # retina is 2D: target translates H/V only

    # ── Visibility gates ──────────────────────────────────────────────────────
    e_mag      = jnp.linalg.norm(target_pos) + 1e-9
    target_in_vf = 1.0 - jax.nn.sigmoid(k_vf * (e_mag - vf_limit))
    scene_vis  = jnp.asarray(scene_present, dtype=jnp.float32)
    target_vis = jnp.asarray(target_present, dtype=jnp.float32) * target_in_vf

    return target_pos, scene_angular_vel, scene_linear_vel, target_vel, scene_vis, target_vis


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


