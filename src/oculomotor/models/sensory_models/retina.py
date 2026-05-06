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

# Per-signal layout: (N_sharp, N_lp, n_axes)  in declaration order.
# N_lp = 0 → no smoothing stage; N_lp = 1 → single 1-pole LP; N_lp ≥ 2 → gamma
# cascade (N_lp poles, total mean delay = tau_smooth) — sharper rolloff than a
# single LP, less exponential tail.  n_axes=3 → 3-D signal; n_axes=1 → scalar.
_SIG_LAYOUT = [
    ('scene_angular_vel',    _N_STAGES_OTHER, 1, 3),
    ('scene_linear_vel',     _N_STAGES_OTHER, 1, 3),
    ('target_pos',           N_STAGES,        0, 3),   # legacy 40-stage, no LP — sharp targeting
    ('target_vel',           _N_STAGES_OTHER, 1, 3),
    ('target_disparity',     _N_STAGES_OTHER, 4, 3),   # 4-pole smoothing — sharper than 1-pole;
                                                       # less residual disparity after eye closure
    # Visibility flags consumed by brain (HE, SG): full 40-stage cascade (no LP)
    # so brief target pulses propagate without amplitude loss. target_motion_visible
    # and target_fusable are NOT cascaded — they're used inline inside cyclopean_vision
    # to gate target_slip and target_disparity before those go through the cascade.
    ('scene_visible',        N_STAGES,        0, 1),
    ('target_visible',       N_STAGES,        0, 1),
    ('defocus',              _N_STAGES_OTHER, 4, 1),   # 4-pole smoothing — sharper than 1-pole
]

def _sig_size(N, N_lp, n_axes):
    """States = (cascade + LP) × n_axes."""
    return (N + N_lp) * n_axes

# Compute per-signal sizes and offsets at module load time
_SIG_SIZES   = {name: _sig_size(N, lp, n) for name, N, lp, n in _SIG_LAYOUT}
_SIG_OFFSETS = {}
_offset = 0
for name, N, lp, n in _SIG_LAYOUT:
    _SIG_OFFSETS[name] = _offset
    _offset += _SIG_SIZES[name]
N_STATES = _offset    # total cascade-substate count

# Convenience module-level offset constants used by cyclopean_vision.step().
# Each `_OFF_<SIGNAL>` is the start index of the signal's block in x_vis;
# the *block* contains [cascade (N×n) | LP (n if has_lp)].
_OFF_SCENE_ANGULAR_VEL    = _SIG_OFFSETS['scene_angular_vel']
_OFF_SCENE_LINEAR         = _SIG_OFFSETS['scene_linear_vel']
_OFF_TARGET_POS           = _SIG_OFFSETS['target_pos']
_OFF_TARGET_VEL           = _SIG_OFFSETS['target_vel']
_OFF_TARGET_DISP          = _SIG_OFFSETS['target_disparity']
_OFF_SCENE_VIS            = _SIG_OFFSETS['scene_visible']
_OFF_TARGET_VIS           = _SIG_OFFSETS['target_visible']
_OFF_DEFOCUS              = _SIG_OFFSETS['defocus']

# Block-end indices (= offset + size) for slicing each signal's block.
_END_SCENE_ANGULAR_VEL    = _OFF_SCENE_ANGULAR_VEL    + _SIG_SIZES['scene_angular_vel']
_END_SCENE_LINEAR         = _OFF_SCENE_LINEAR         + _SIG_SIZES['scene_linear_vel']
_END_TARGET_POS           = _OFF_TARGET_POS           + _SIG_SIZES['target_pos']
_END_TARGET_VEL           = _OFF_TARGET_VEL           + _SIG_SIZES['target_vel']
_END_TARGET_DISP          = _OFF_TARGET_DISP          + _SIG_SIZES['target_disparity']
_END_SCENE_VIS            = _OFF_SCENE_VIS            + _SIG_SIZES['scene_visible']
_END_TARGET_VIS           = _OFF_TARGET_VIS           + _SIG_SIZES['target_visible']
_END_DEFOCUS              = _OFF_DEFOCUS              + _SIG_SIZES['defocus']

# Legacy alias for code that still references the old "120 states per 3-D signal"
# constant — only target_pos still has this size.
_N_PER_SIG    = N_STAGES * 3        # 120 (target_pos block size)
_N_SCALAR     = _N_STAGES_OTHER     # used in retina geometry below for scaling — keeps
                                    # the visual-field gate sigmoid the same as before


def _make_C_last_n(end_idx, n_axes):
    """Read-off matrix that selects the LAST n_axes elements of a signal block.

    For signals with LP: last n_axes states = LP states (the smoothed output).
    For signals without LP (target_pos): last n_axes states = last cascade stage.
    Either way, this is the channel's "current observed value".
    """
    C = jnp.zeros((n_axes, N_STATES))
    for i in range(n_axes):
        C = C.at[i, end_idx - n_axes + i].set(1.0)
    return C


# ── Readout matrices ────────────────────────────────────────────────────────────
# Exported so sensory_model / efference_copy can read cascade outputs directly.

C_slip             = _make_C_last_n(_END_SCENE_ANGULAR_VEL, 3)
C_scene_linear_vel = _make_C_last_n(_END_SCENE_LINEAR,      3)
C_pos              = _make_C_last_n(_END_TARGET_POS,        3)
C_vel              = _make_C_last_n(_END_TARGET_VEL,        3)
C_target_disp      = _make_C_last_n(_END_TARGET_DISP,       3)
C_scene_visible    = _make_C_last_n(_END_SCENE_VIS,         1)
C_target_visible   = _make_C_last_n(_END_TARGET_VIS,        1)
C_defocus          = _make_C_last_n(_END_DEFOCUS,           1)


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


