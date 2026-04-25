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


# ── Velocity saturation ─────────────────────────────────────────────────────────

def velocity_saturation(v, v_sat, v_zero=None):
    """Smooth velocity saturation: passes at low speed, gain ramps to zero at high speed.

    For a velocity vector v:
        |v| ≤ v_sat          → output = v           (gain = 1)
        v_sat < |v| < v_zero → output = v * gain    (cosine rolloff, 1 → 0)
        |v| ≥ v_zero         → output = 0            (gain = 0)

    The cosine rolloff keeps gain and its derivative continuous at both endpoints.

    Contrast with jnp.clip: clipping keeps output at ±v_sat for large inputs,
    which lets a step-function target spike (e.g. 9000 deg/s central-difference
    artifact) drive the pursuit integrator at full saturation velocity.  This
    function returns zero instead, faithfully modelling MT/MST insensitivity to
    implausibly fast retinal motion.

    Speed tuning background
    -----------------------
    MT/MST neurons are band-pass tuned for speed, not low-pass.  The population
    response peaks around 10–40 deg/s and falls off sharply above ~80–100 deg/s.
    Pursuit consequently saturates: human gain ≈ 1 below ~30 deg/s, dropping to
    ~0.5 at 60 deg/s and near zero above ~100 deg/s.

    NOT/AOS neurons driving OKR have broader tuning, peaking ~40–80 deg/s and
    falling off above ~160 deg/s.

    References
    ----------
    Maunsell JHR & Van Essen DC (1983) J Neurophysiol 49:1127-1147
        — MT speed tuning in macaque; peak ~30 deg/s, ~50 % at 8 and 100 deg/s.
    Lisberger SG et al. (1987) Annu Rev Neurosci 10:97-129
        — Review of pursuit velocity range and MT contribution.
    Priebe NJ & Lisberger SG (2004) J Neurosci 24:4907-4926
        — Population speed coding in MT; pursuit gain × speed relationship.
    Buettner U et al. (1976) Brain Res 108:359-377
        — OKN gain as a function of stimulus velocity; falls above 80 deg/s.

    Args:
        v:      (N,) velocity vector (deg/s); norm computed over the full vector
        v_sat:  saturation onset (deg/s) — gain is exactly 1 below this
        v_zero: speed where gain reaches 0 (deg/s); default = 2 × v_sat

    Returns:
        Same shape as v, scaled by smooth gain ∈ [0, 1].

    Example (pursuit):
        velocity_saturation(target_slip, v_sat=40.0)   # v_zero defaults to 80.0
    Example (OKR/NOT):
        velocity_saturation(scene_slip, v_sat=80.0, v_zero=160.0)
    """
    if v_zero is None:
        v_zero = 2.0 * v_sat
    speed = jnp.linalg.norm(v)
    t     = jnp.clip((speed - v_sat) / (v_zero - v_sat), 0.0, 1.0)
    gain  = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    return v * gain


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

def retinal_signals(p_target, eye_offset_head, q_head, w_head, x_head, v_head,
                    q_eye, w_eye, w_scene, v_target, vf_limit, k_vf):
    """Compute instantaneous retinal signals and visual-field gate.

    All outputs are in EYE coordinates [yaw, pitch, roll] (deg or deg/s).
    All head and eye geometry is handled here — sensory_model only passes
    anatomical offsets (eye_offset_head) and does not touch rotation matrices.

    World-frame convention: x=right, y=up, z=forward.

    Inputs
    ------
    p_target:        target 3-D position in world frame (m)  [x=right, y=up, z=fwd]
    eye_offset_head: this eye's fixed position in head frame (m)
                     left=[-ipd/2,0,0]  right=[+ipd/2,0,0]
    q_head:          head rotation vector [yaw,pitch,roll]  (deg, world frame)
    w_head:          head angular velocity [yaw,pitch,roll]  (deg/s, world frame)
    x_head:          head linear position  [x,y,z]           (m, world frame)
    v_head:          head linear velocity  [x,y,z]           (m/s, world frame)
                     stored for future looming estimation; not used in current geometry
    q_eye:           eye rotation vector relative to head (deg, head frame)
    w_eye:           eye angular velocity relative to head (deg/s, head frame)
    w_scene:         scene angular velocity [yaw,pitch,roll] (deg/s, world frame)
    v_target:        target angular velocity [yaw,pitch,roll] (deg/s, world frame)
                     = _rv2q( cross(p_target, dp_target/dt) / |p_target|² )
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

    Retinal velocities in eye frame (exact):
        w_eye_world = w_head + R_head @ w_eye     total eye velocity, world frame
        scene_vel   = R_gaze.T @ (w_scene  − w_eye_world)   eye-frame, deg/s
        target_vel  = R_gaze.T @ (v_target − w_eye_world)   eye-frame, deg/s

    Returns  (all in eye coordinates)
    -----------------------------------
        target_pos:   (3,)   target direction [yaw, pitch, 0] (deg)
        scene_vel:    (3,)   full-field scene velocity on retina (deg/s)
        target_vel:   (3,)   target velocity on retina (deg/s)
        target_in_vf: scalar geometric visual-field gate ∈ [0, 1]
    """
    # ── Rotation matrices ─────────────────────────────────────────────────────
    # World frame: x=right, y=up, z=fwd.
    # [yaw,pitch,roll] → xyz rotation vector: yaw→y, pitch→-x, roll→z.
    # Rodrigues R_x(+θ) maps [0,0,1] → [0,-sinθ,cosθ] (looks DOWN), so negate
    # pitch to get R_x(−pitch) which maps [0,0,1]→[0,sinθ,cosθ] (looks UP).
    def _q2rv(q): return jnp.array([-q[1], q[0], q[2]])   # [yaw,pitch,roll] → xyz
    def _rv2q(v): return jnp.array([v[1], -v[0], v[2]])   # xyz → [yaw,pitch,roll]
    R_head   = rotation_matrix(_q2rv(q_head))   # world ← head
    R_eye    = rotation_matrix(_q2rv(q_eye))    # head  ← eye
    R_gaze_T = R_eye.T @ R_head.T              # world → eye frame

    # ── Target position in eye frame ──────────────────────────────────────────
    eye_world  = x_head + R_head @ eye_offset_head           # eye position, world frame
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

def step(x_vis,
         p_target, eye_offset,
         q_head, w_head, x_head, v_head,
         q_eye, w_eye,
         w_scene, v_target,
         scene_present, target_present, target_strobed,
         tau_vis, visual_field_limit, k_visual_field,
         v_max_scene_vel=80.0, v_max_target_vel=40.0):
    """Single ODE step for the full visual delay cascade (six signals).

    Computes retinal geometry (via retinal_signals) then gates, speed-saturates,
    and advances six delay cascades:
        scene_vel   × scene_present → velocity_saturation(·, v_max_scene_vel)   → OKR / VS
        target_pos  × target_visible                                              → saccade error
        target_vel  × target_visible × (1 − target_strobed)
                    → velocity_saturation(·, v_max_target_vel)                  → pursuit drive

    where target_visible = target_present × target_in_vf (retinal field-of-view gate).

    Pre-delay saturation ensures that step-target velocity spikes (e.g. 9000 deg/s
    from central-difference numerics) never enter the cascade.  The cascade output
    is then always ≤ v_max, matching the physiological speed tuning of MT/MST
    (target) and NOT/AOS (scene) neurons.

    State layout: [x_scene_vel(120) | x_target_pos(120) | x_target_vel(120)
                   | x_scene_visible(40) | x_target_visible(40) | x_strobed(40)]

    Args:
        x_vis:               (480,)  visual cascade state
        p_target:            (3,)    target 3-D position (m, world frame)
        eye_offset:          (3,)    eye centre in head frame (m); e.g. [±ipd/2, 0, 0]
        q_head:              (3,)    head rotation vector (deg)
        w_head:              (3,)    head angular velocity (deg/s)
        x_head:              (3,)    head linear position (m, world frame)
        v_head:              (3,)    head linear velocity (m/s, world frame)
        q_eye:               (3,)    eye rotation vector (deg, head frame)
        w_eye:               (3,)    eye angular velocity (deg/s, head frame)
        w_scene:             (3,)    scene angular velocity (deg/s, world frame)
        v_target:            (3,)    target angular velocity (deg/s, world frame)
        scene_present:       scalar  is scene physically present? ∈ {0, 1}
        target_present:      scalar  is target present (not occluded)? ∈ {0, 1}
        target_strobed:      scalar  1 = stroboscopic illumination ∈ {0, 1}
        tau_vis:             float   total cascade delay (s)
        visual_field_limit:  float   retinal eccentricity limit (deg)
        k_visual_field:      float   sigmoid steepness for visual field gate (1/deg)
        v_max_scene_vel:     float   NOT/AOS speed ceiling (deg/s); default 80.0
        v_max_target_vel:    float   MT/MST speed ceiling (deg/s);  default 40.0

    Returns:
        dx_vis: (480,)  dx_vis/dt
    """
    target_pos, scene_vel, target_vel, target_in_vf = retinal_signals(
        p_target, eye_offset, q_head, w_head, x_head, v_head,
        q_eye, w_eye, w_scene, v_target,
        visual_field_limit, k_visual_field)

    x_scene_vel   = x_vis[              :   _N_PER_SIG]
    x_target_pos  = x_vis[ _N_PER_SIG   : 2*_N_PER_SIG]
    x_target_vel  = x_vis[2*_N_PER_SIG  : 3*_N_PER_SIG]
    x_scene_vis   = x_vis[_OFF_SCENE_VIS : _OFF_TARGET_VIS]
    x_target_vis  = x_vis[_OFF_TARGET_VIS: _OFF_STROBED]
    x_strobed     = x_vis[_OFF_STROBED   :]

    target_visible = target_present * target_in_vf

    scene_vel_in  = velocity_saturation(scene_vel  * scene_present,                           v_max_scene_vel)
    target_pos_in = target_pos * target_visible
    target_vel_in = velocity_saturation(target_vel * target_visible * (1.0 - target_strobed), v_max_target_vel)

    dx_scene_vel  = delay_cascade_step(x_scene_vel,  scene_vel_in,   tau_vis)
    dx_target_pos = delay_cascade_step(x_target_pos, target_pos_in,  tau_vis)
    dx_target_vel = delay_cascade_step(x_target_vel, target_vel_in,  tau_vis)
    dx_scene_vis  = delay_cascade_step(x_scene_vis,  scene_present,  tau_vis)
    dx_target_vis = delay_cascade_step(x_target_vis, target_visible, tau_vis)
    dx_strobed    = delay_cascade_step(x_strobed,    target_strobed, tau_vis)

    return jnp.concatenate([dx_scene_vel, dx_target_pos, dx_target_vel,
                             dx_scene_vis, dx_target_vis, dx_strobed])
