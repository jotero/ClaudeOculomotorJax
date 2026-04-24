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
         Signal 3 — scene_visible   (1,)  delay(scene_present) — presence only, no geometry
         Signal 4 — target_visible  (1,)  delay(target_present × target_in_vf) — presence × geometry
         Signal 5 — target_strobed  (1,)  delay(target_strobed) — stroboscopic flag,
                                          used by brain to gate EC alongside delayed velocity

     All signals enter the cascade raw.  Signal 5 is delayed so the brain can gate
     the efference copy at exactly the time the (already-zeroed) velocity arrives.

     State layout (480,):
         [x_scene_vel (120) | x_target_pos (120) | x_target_vel (120)
          | x_scene_visible (40) | x_target_visible (40) | x_strobed (40)]

     Module-level readout matrices (exported for sensory_model / efference copy):
         C_slip           (3, 480)  last stage of scene_vel cascade
         C_pos            (3, 480)  last stage of target_pos cascade
         C_vel            (3, 480)  last stage of target_vel cascade
         C_scene_visible  (1, 480)  last stage of scene_visible cascade
         C_target_visible (1, 480)  last stage of target_visible cascade
         C_strobed        (1, 480)  last stage of strobed cascade
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

_OFF_SCENE_VIS    = _N_SIG * _N_PER_SIG          # 360
_OFF_TARGET_VIS   = _OFF_SCENE_VIS + _N_SCALAR    # 400
_OFF_STROBED      = _OFF_TARGET_VIS + _N_SCALAR   # 440

# ── Readout matrices ────────────────────────────────────────────────────────────
# Exported so sensory_model / efference_copy can read cascade outputs directly.

C_slip           = jnp.zeros((3, N_STATES)).at[:, _N_PER_SIG-3       : _N_PER_SIG      ].set(jnp.eye(3))
C_pos            = jnp.zeros((3, N_STATES)).at[:, 2*_N_PER_SIG-3     : 2*_N_PER_SIG    ].set(jnp.eye(3))
C_vel            = jnp.zeros((3, N_STATES)).at[:, 3*_N_PER_SIG-3     : 3*_N_PER_SIG    ].set(jnp.eye(3))
C_scene_visible  = jnp.zeros((1, N_STATES)).at[0, _OFF_SCENE_VIS + _N_SCALAR - 1        ].set(1.0)
C_target_visible = jnp.zeros((1, N_STATES)).at[0, _OFF_TARGET_VIS + _N_SCALAR - 1       ].set(1.0)
C_strobed        = jnp.zeros((1, N_STATES)).at[0, _OFF_STROBED    + _N_SCALAR - 1       ].set(1.0)


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
    # q = [yaw, pitch, roll] but Rodrigues uses [x, y, z] axis components.
    # In the world frame (x=right, y=up, z=fwd): yaw rotates around y-axis,
    # pitch around x-axis.  Rodrigues R_x(+θ) maps forward [0,0,1] →
    # [0,-sinθ,cosθ] (looks DOWN), so we negate pitch to get R_x(−pitch)
    # which correctly maps [0,0,1] → [0,sinθ,cosθ] (looks UP for positive pitch).
    def _q2rv(q): return jnp.array([-q[1], q[0], q[2]])      # [yaw,pitch,roll] → xyz
    def _rv2q(v): return jnp.array([v[1], -v[0], v[2]])     # xyz → [yaw,pitch,roll]
    R_head   = rotation_matrix(_q2rv(q_head))   # world ← head
    R_eye    = rotation_matrix(_q2rv(q_eye))    # head  ← eye
    R_gaze_T = R_eye.T @ R_head.T              # world → eye frame

    # ── Target position in eye frame ──────────────────────────────────────────
    eye_world  = R_head @ eye_offset_head                    # eye position, world frame
    p_from_eye = p_target - eye_world                        # target from this eye, world frame
    p_hat      = p_from_eye / (jnp.linalg.norm(p_from_eye) + 1e-9)
    p_eye      = R_gaze_T @ p_hat                            # target direction, eye frame

    yaw_e   = jnp.degrees(jnp.arctan2(p_eye[0], p_eye[2]))
    pitch_e = jnp.degrees(jnp.arctan2(p_eye[1], jnp.sqrt(p_eye[0]**2 + p_eye[2]**2)))
    target_pos = jnp.array([yaw_e, pitch_e, 0.0])           # roll=0: target direction has only 2 DOF

    # ── Retinal velocities in eye frame ───────────────────────────────────────
    # Velocity vectors are in [yaw,pitch,roll] notation but R_head / R_gaze_T
    # are built in xyz (via _q2rv). Convert velocities to xyz for the matrix
    # operations so the transformation is correct at large head angles.
    # Without this, sustained rotation (e.g. 90° cumulative yaw) rotates the
    # yaw velocity into the pitch/roll directions, causing OKR to fight VOR.
    w_head_xyz  = _q2rv(w_head)
    w_eye_xyz   = _q2rv(w_eye)
    w_scene_xyz = _q2rv(w_scene)
    vt_xyz      = _q2rv(v_target)
    w_eye_world = w_head_xyz + R_head @ w_eye_xyz            # total eye velocity, xyz world frame
    scene_vel   = _rv2q(R_gaze_T @ (w_scene_xyz - w_eye_world))   # eye frame, [yaw,pitch,roll]
    target_vel  = _rv2q(R_gaze_T @ (vt_xyz      - w_eye_world))   # eye frame, [yaw,pitch,roll]

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

def step(x_vis, scene_vel, target_pos, target_vel,
         scene_present, target_visible, target_strobed, tau_vis):
    """Single ODE step for the full visual delay cascade (six signals).

    Signals are gated at the cascade input so delayed outputs are already clean:
        scene_vel   × scene_present                          → OKR / VS
        target_pos  × target_visible                         → saccade error
        target_vel  × target_visible × (1 − target_strobed) → pursuit drive
    Presence flags and strobe are delayed raw so the brain has timing information.

    State layout: [x_scene_vel(120) | x_target_pos(120) | x_target_vel(120)
                   | x_scene_visible(40) | x_target_visible(40) | x_strobed(40)]

    Args:
        x_vis:           (480,)  visual cascade state
        scene_vel:       (3,)    scene velocity on retina, eye frame (deg/s)
        target_pos:      (3,)    target position on retina, eye frame (deg)
        target_vel:      (3,)    target velocity on retina, eye frame (deg/s)
        scene_present:   scalar  is scene physically present? ∈ {0, 1}
        target_visible:  scalar  target_present × target_in_vf ∈ [0, 1]
        target_strobed:  scalar  1 = stroboscopic illumination ∈ {0, 1}
        tau_vis:         float   total cascade delay (s)

    Returns:
        dx_vis: (480,)  dx_vis/dt
    """
    x_scene_vel   = x_vis[              :   _N_PER_SIG]
    x_target_pos  = x_vis[ _N_PER_SIG   : 2*_N_PER_SIG]
    x_target_vel  = x_vis[2*_N_PER_SIG  : 3*_N_PER_SIG]
    x_scene_vis   = x_vis[_OFF_SCENE_VIS : _OFF_TARGET_VIS]
    x_target_vis  = x_vis[_OFF_TARGET_VIS: _OFF_STROBED]
    x_strobed     = x_vis[_OFF_STROBED   :]

    scene_vel_in  = scene_vel  * scene_present
    target_pos_in = target_pos * target_visible
    target_vel_in = target_vel * target_visible * (1.0 - target_strobed)

    dx_scene_vel  = delay_cascade_step(x_scene_vel,  scene_vel_in,   tau_vis)
    dx_target_pos = delay_cascade_step(x_target_pos, target_pos_in,  tau_vis)
    dx_target_vel = delay_cascade_step(x_target_vel, target_vel_in,  tau_vis)
    dx_scene_vis  = delay_cascade_step(x_scene_vis,  scene_present,  tau_vis)
    dx_target_vis = delay_cascade_step(x_target_vis, target_visible, tau_vis)
    dx_strobed    = delay_cascade_step(x_strobed,    target_strobed, tau_vis)

    return jnp.concatenate([dx_scene_vel, dx_target_pos, dx_target_vel,
                             dx_scene_vis, dx_target_vis, dx_strobed])
