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
#
# Two-tier transmission model per signal:
#   - SHARP cascade: high-N (or moderate-N) gamma cascade modelling photo-
#     transduction + axonal/synaptic transport delay (Pugh & Lamb 1993, Dunn &
#     Rieke 2006). Produces a near-pure transport delay of mean = tau_sharp.
#   - SMOOTH LP: optional 1-pole leaky integrator after the cascade modelling
#     channel-specific neural integration (MT/MST motion window, V1 stereo
#     correspondence, accommodation circuit). Adds smoothness without changing
#     the dead-zone character.
#
# target_pos KEEPS the legacy 40-stage cascade (no LP) — saccade targeting needs
# a sharp transport delay with no extra smoothing.

# Legacy stage count for target_pos
N_STAGES      = 40

# Sharp-cascade stage count for all OTHER signals (Pugh-Lamb photo-transduction:
# 4-6 stage biochemical cascade gives the right rise shape).
_N_STAGES_OTHER = 6

# ── Per-eye retina state layout ────────────────────────────────────────────────
# Each eye has its own sharp gamma cascade per signal (N = _N_STAGES_OTHER stages
# × τ_retina). target_disparity is NOT here — it's a binocular construction
# computed in perception_cyclopean from delayed per-eye target_pos.
_RETINA_PER_EYE_LAYOUT = [
    ('scene_angular_vel', _N_STAGES_OTHER, 3),   # 18
    ('scene_linear_vel',  _N_STAGES_OTHER, 3),   # 18
    ('target_pos',        _N_STAGES_OTHER, 3),   # 18
    ('target_vel',        _N_STAGES_OTHER, 3),   # 18
    ('scene_visible',     _N_STAGES_OTHER, 1),   #  6
    ('target_visible',    _N_STAGES_OTHER, 1),   #  6
    ('defocus',           _N_STAGES_OTHER, 1),   #  6
]

_RETINA_PER_EYE_SIZES = {name: N * n for name, N, n in _RETINA_PER_EYE_LAYOUT}
_RETINA_PER_EYE_OFFSETS = {}
_offset = 0
for name, N, n in _RETINA_PER_EYE_LAYOUT:
    _RETINA_PER_EYE_OFFSETS[name] = _offset
    _offset += _RETINA_PER_EYE_SIZES[name]
N_STATES_PER_EYE = _offset   # 90 — sharp cascade states per eye

_RET_OFF_SCENE_ANGULAR_VEL = _RETINA_PER_EYE_OFFSETS['scene_angular_vel']
_RET_OFF_SCENE_LINEAR      = _RETINA_PER_EYE_OFFSETS['scene_linear_vel']
_RET_OFF_TARGET_POS        = _RETINA_PER_EYE_OFFSETS['target_pos']
_RET_OFF_TARGET_VEL        = _RETINA_PER_EYE_OFFSETS['target_vel']
_RET_OFF_SCENE_VIS         = _RETINA_PER_EYE_OFFSETS['scene_visible']
_RET_OFF_TARGET_VIS        = _RETINA_PER_EYE_OFFSETS['target_visible']
_RET_OFF_DEFOCUS           = _RETINA_PER_EYE_OFFSETS['defocus']

_RET_END_SCENE_ANGULAR_VEL = _RET_OFF_SCENE_ANGULAR_VEL + _RETINA_PER_EYE_SIZES['scene_angular_vel']
_RET_END_SCENE_LINEAR      = _RET_OFF_SCENE_LINEAR      + _RETINA_PER_EYE_SIZES['scene_linear_vel']
_RET_END_TARGET_POS        = _RET_OFF_TARGET_POS        + _RETINA_PER_EYE_SIZES['target_pos']
_RET_END_TARGET_VEL        = _RET_OFF_TARGET_VEL        + _RETINA_PER_EYE_SIZES['target_vel']
_RET_END_SCENE_VIS         = _RET_OFF_SCENE_VIS         + _RETINA_PER_EYE_SIZES['scene_visible']
_RET_END_TARGET_VIS        = _RET_OFF_TARGET_VIS        + _RETINA_PER_EYE_SIZES['target_visible']
_RET_END_DEFOCUS           = _RET_OFF_DEFOCUS           + _RETINA_PER_EYE_SIZES['defocus']

_N_SCALAR = _N_STAGES_OTHER     # used in retina geometry below for scaling — keeps
                                # the visual-field gate sigmoid the same as before

# Backward-compat alias for legacy code (efference_copy.py) that hardcodes the
# 40-stage 3-axis cascade size.
_N_PER_SIG = N_STAGES * 3       # 120


# ── Sensor saturation ───────────────────────────────────────────────────────────

def velocity_saturation(v, v_sat, v_zero=None, v_offset=None):
    """Smooth velocity saturation: passes at low speed, gain ramps to zero at high speed.

    Models NOT/AOS / MT-MST firing-rate ceiling — neurons are band-pass tuned
    for speed and stop firing for implausibly fast retinal motion.

        |v| ≤ v_sat          → output = v           (gain = 1)
        v_sat < |v| < v_zero → output = v · gain    (cosine rolloff, 1 → 0)
        |v| ≥ v_zero         → output = 0           (gain = 0)

    Args:
        v:        (N,) velocity vector (deg/s); norm computed over the full vector
        v_sat:    saturation onset (deg/s) — gain is exactly 1 below this
        v_zero:   speed where gain reaches 0 (deg/s); default = 2 × v_sat
        v_offset: (N,) background velocity to shift the clip window (deg/s)

    Returns:
        Same shape as v, scaled by smooth gain ∈ [0, 1], plus v_offset if given.
    """
    if v_zero is None:
        v_zero = 2.0 * v_sat
    if v_offset is not None:
        v_rel = v - v_offset
    else:
        v_rel = v
    speed = jnp.linalg.norm(v_rel)
    t     = jnp.clip((speed - v_sat) / (v_zero - v_sat), 0.0, 1.0)
    gain  = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    result = v_rel * gain
    if v_offset is not None:
        result = result + v_offset
    return result


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


def cascade_lp_step(x_block, u, tau_sharp, tau_smooth, N, n_axes, N_lp):
    """Sharp gamma cascade + optional multi-stage smoothing.

    Block layout (concatenated cascade then LP):
        [sharp cascade (N · n_axes) | smoothing cascade (N_lp · n_axes)]

    The smoothing stage is itself a gamma cascade of N_lp poles with total mean
    delay tau_smooth (so per-stage TC = tau_smooth/N_lp). N_lp = 1 reduces to a
    single 1-pole LP (exponential rise/fall, long tail). N_lp ≥ 2 gives a
    sharper rolloff and a more concentrated impulse response — the right
    choice when residual signal after the input goes to zero must drain quickly
    (e.g. binocular disparity after one eye closes).

    Args:
        x_block:    block state ((N + N_lp)·n_axes,)
        u:          (n_axes,) or scalar  current input
        tau_sharp:  total sharp-cascade mean delay (s)
        tau_smooth: total smoothing-cascade mean delay (s); ignored if N_lp=0
        N:          sharp-cascade stage count
        n_axes:     1 (scalar) or 3 (3-vector)
        N_lp:       smoothing-cascade stage count; 0 = no smoothing,
                    1 = single 1-pole LP, ≥2 = multi-pole gamma smoothing

    Returns:
        dx_block:   state derivative, same shape as x_block
    """
    n_cascade = N * n_axes
    x_cascade = x_block[:n_cascade]
    dx_cascade = delay_cascade_step(x_cascade, u, tau_sharp, N=N)

    if N_lp == 0:
        return dx_cascade

    # Smoothing stage: feed the sharp-cascade output into another gamma cascade.
    cascade_out = x_cascade[n_cascade - n_axes : n_cascade]
    x_lp = x_block[n_cascade : n_cascade + N_lp * n_axes]
    dx_lp = delay_cascade_step(x_lp, cascade_out, tau_smooth, N=N_lp)
    return jnp.concatenate([dx_cascade, dx_lp])


# ── Per-eye retina step ─────────────────────────────────────────────────────────

from typing import NamedTuple


class RetinaOut(NamedTuple):
    """Per-eye delayed retinal signals after the sharp gamma cascade.

    All signals are in EYE FRAME at the delayed time (≈ τ_retina ago). Field
    order matches the cascade state-block layout for consistency. Read into
    perception_cyclopean for binocular fusion + brain LP smoothing.
    """
    scene_angular_vel: jnp.ndarray  # (3,) [yaw, pitch, roll] (deg/s) — gated by scene_visible + saturated
    scene_linear_vel:  jnp.ndarray  # (3,) [x, y, z] (m/s, head frame, per-eye) — gated by scene_visible
    target_pos:        jnp.ndarray  # (3,) [yaw, pitch, 0] (deg) — gated by target_visible
    target_vel:        jnp.ndarray  # (3,) [yaw, pitch, 0] (deg/s) — gated by target_motion_vis + saturated
    scene_visible:     jnp.ndarray  # scalar — delayed scene_present
    target_visible:    jnp.ndarray  # scalar — delayed target_present × target_in_vf (NOT strobe-gated)
    defocus:           jnp.ndarray  # scalar — delayed defocus (D)


def step(x_retina_eye,
         eye_offset_head, q_head, w_head, x_head, v_head,
         q_eye, w_eye,
         w_scene, v_scene, p_target, dp_dt,
         defocus_eye,
         scene_present, target_present, target_strobed,
         sensory_params):
    """Per-eye retina step: world_to_retina + sensor saturation + sharp cascade.

    Each eye runs an independent sharp gamma cascade (N = _N_STAGES_OTHER stages
    × τ_retina). Binocular fusion happens downstream in perception_cyclopean,
    so this function knows nothing about the other eye.

    Args:
        x_retina_eye:    (N_STATES_PER_EYE,) per-eye retina cascade state
        eye_offset_head: (3,) this eye's anatomical offset in head frame (m)
        q_head, w_head, x_head, v_head: head pose / velocity (world frame)
        q_eye, w_eye:    eye pose / velocity (head frame)
        w_scene, v_scene, p_target, dp_dt: world-frame scene / target stimulus
        defocus_eye:     scalar — instantaneous per-eye defocus (D)
        scene_present, target_present: scalar visibility flags (this eye)
        target_strobed:  scalar global strobe gate; (1−strobed) gates target_vel only
        sensory_params:  SensoryParams — reads tau_vis_sharp, v_max_scene_vel,
                         v_max_target_vel, visual_field_limit, k_visual_field

    Returns:
        dx_retina_eye: (N_STATES_PER_EYE,) state derivative
        retina_out:    RetinaOut bundle of delayed per-eye signals
    """
    # ── 1. Geometry — world_to_retina projection ─────────────────────────────
    target_pos, scene_angular_vel, scene_linear_vel, target_vel, scene_vis, target_vis = \
        world_to_retina(
            p_target, eye_offset_head, q_head, w_head, x_head, v_head,
            q_eye, w_eye, w_scene, v_scene, dp_dt,
            scene_present, target_present,
            sensory_params.visual_field_limit, sensory_params.k_visual_field,
        )

    # ── 2. Per-eye gating + sensor saturation ────────────────────────────────
    # Strobe gate: only target_vel sees it (so SG still sees target_visible during
    # strobed pursuit and can re-target accurately).
    target_motion_vis  = target_vis * (1.0 - target_strobed)
    scene_angular_in   = velocity_saturation(scene_angular_vel * scene_vis, sensory_params.v_max_scene_vel)
    target_vel_in      = velocity_saturation(target_vel * target_motion_vis, sensory_params.v_max_target_vel)
    scene_linear_in    = scene_linear_vel * scene_vis
    target_pos_in      = target_pos * target_vis
    defocus_in         = defocus_eye

    # ── 3. Slice per-eye cascade state ───────────────────────────────────────
    x_scene_ang  = x_retina_eye[_RET_OFF_SCENE_ANGULAR_VEL : _RET_END_SCENE_ANGULAR_VEL]
    x_scene_lin  = x_retina_eye[_RET_OFF_SCENE_LINEAR      : _RET_END_SCENE_LINEAR]
    x_target_pos = x_retina_eye[_RET_OFF_TARGET_POS        : _RET_END_TARGET_POS]
    x_target_vel = x_retina_eye[_RET_OFF_TARGET_VEL        : _RET_END_TARGET_VEL]
    x_scene_vis  = x_retina_eye[_RET_OFF_SCENE_VIS         : _RET_END_SCENE_VIS]
    x_target_vis = x_retina_eye[_RET_OFF_TARGET_VIS        : _RET_END_TARGET_VIS]
    x_defocus    = x_retina_eye[_RET_OFF_DEFOCUS           : _RET_END_DEFOCUS]

    # ── 4. Advance sharp cascades (N stages × τ_retina, per signal) ──────────
    tau_retina = sensory_params.tau_vis_sharp
    N          = _N_STAGES_OTHER
    dx_scene_ang  = delay_cascade_step(x_scene_ang,  scene_angular_in, tau_retina, N=N)
    dx_scene_lin  = delay_cascade_step(x_scene_lin,  scene_linear_in,  tau_retina, N=N)
    dx_target_pos = delay_cascade_step(x_target_pos, target_pos_in,    tau_retina, N=N)
    dx_target_vel = delay_cascade_step(x_target_vel, target_vel_in,    tau_retina, N=N)
    dx_scene_vis  = delay_cascade_step(x_scene_vis,  scene_vis,        tau_retina, N=N)
    dx_target_vis = delay_cascade_step(x_target_vis, target_vis,       tau_retina, N=N)
    dx_defocus    = delay_cascade_step(x_defocus,    defocus_in,       tau_retina, N=N)

    dx_retina_eye = jnp.concatenate([
        dx_scene_ang, dx_scene_lin, dx_target_pos, dx_target_vel,
        dx_scene_vis, dx_target_vis, dx_defocus,
    ])

    # ── 5. Read delayed signals (last n_axes of each cascade) ────────────────
    out = RetinaOut(
        scene_angular_vel = x_scene_ang[-3:],
        scene_linear_vel  = x_scene_lin[-3:],
        target_pos        = x_target_pos[-3:],
        target_vel        = x_target_vel[-3:],
        scene_visible     = x_scene_vis[-1],
        target_visible    = x_target_vis[-1],
        defocus           = x_defocus[-1],
    )
    return dx_retina_eye, out


def read_outputs(x_retina_eye):
    """Pure state readout — returns RetinaOut from a per-eye cascade state.

    Last n_axes of each block = sharp-cascade output (delayed per-eye signal).
    """
    return RetinaOut(
        scene_angular_vel = x_retina_eye[_RET_END_SCENE_ANGULAR_VEL - 3 : _RET_END_SCENE_ANGULAR_VEL],
        scene_linear_vel  = x_retina_eye[_RET_END_SCENE_LINEAR      - 3 : _RET_END_SCENE_LINEAR],
        target_pos        = x_retina_eye[_RET_END_TARGET_POS        - 3 : _RET_END_TARGET_POS],
        target_vel        = x_retina_eye[_RET_END_TARGET_VEL        - 3 : _RET_END_TARGET_VEL],
        scene_visible     = x_retina_eye[_RET_END_SCENE_VIS         - 1],
        target_visible    = x_retina_eye[_RET_END_TARGET_VIS        - 1],
        defocus           = x_retina_eye[_RET_END_DEFOCUS           - 1],
    )


