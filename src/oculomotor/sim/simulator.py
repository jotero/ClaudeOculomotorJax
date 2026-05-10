"""Full oculomotor simulator — wires sensory_model, brain_model, and plant_model.

World frame is LEFT-HANDED: x=right, y=up, z=forward  (x × y = −z).

Coordinate conventions
----------------------
Positional / linear vectors  (p_target, x_head, …):   [x, y, z] = [right, up, fwd]

Angular vectors  (w_head, w_eye, motor_cmd, …):        [yaw, pitch, roll]
    ≠ xyz order — element 0 (yaw) is rotation about +y, not +x.

    ypr_to_xyz([yaw, pitch, roll]) = [−pitch,  yaw,  roll]   # → xyz rotation axis
    xyz_to_ypr([x,   y,     z  ]) = [y,       −x,   z   ]   # → [yaw,pitch,roll]

    yaw   (idx 0): rotation about +y  (left-hand: forward → right = rightward turn)
    pitch (idx 1): rotation about −x  (left-hand: forward → up   = look up)
    roll  (idx 2): rotation about +z  (left-hand: right → up)

All rotation matrices (Rodrigues) and cross products operate in xyz.
Call ypr_to_xyz() before matrix ops; xyz_to_ypr() after.

Signal flow (3-D, binocular):

    VOR pathway:
        w_head (3,) → [Sensory: Canal Array] → y_canals (6,)

    Visual delay (inside sensory_model, per eye):
        e_slip (3,)  = scene_present · (w_scene − w_eye_world)
        e_slip → [Sensory: Visual delay L/R] → e_slip_delayed   (VS / OKR)
        e_pos  → [Sensory: Visual delay L/R] → e_pos_delayed    (SG)

    Brain model (VS + NI + SG + EC) — driven by L+R averages:
        [y_canals | e_slip_delayed] → VS  → w_est
        e_pos_delayed → target selector → SG → u_burst
        u_vel = −w_est + u_burst → NI → motor_cmd

    Plant model (binocular — same motor_cmd drives both eyes):
        motor_cmd → Plant_L → q_eye_L
        motor_cmd → Plant_R → q_eye_R

    v_target (target angular velocity for pursuit) is computed in the ODE
    from the Cartesian target position and velocity:
        v_target = xyz_to_ypr( cross(p_target, dp_target/dt) / |p_target|² )  [deg/s]

State structure — SimState NamedTuple:

    sensory  (978):  [x_c (12) | x_oto (6) | x_vis_L (480) | x_vis_R (480)]
    brain    (156):  [x_vs (9) | x_ni (9) | x_sg (9) | x_ec (120) | x_grav (3) | x_pursuit (3) | x_verg (3)]
    plant      (6):  [x_p_L (3) | x_p_R (3)]

Stimulus inputs to simulate():
    head:   KinematicTrajectory  — 6-DOF head pose + derivatives
    scene:  KinematicTrajectory  — 6-DOF scene pose + derivatives
    target: TargetTrajectory     — 3-DOF target position + velocity
    scene_present_*/target_present_* arrays — per-eye visibility flags
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax

from oculomotor.sim.kinematics import KinematicTrajectory, TargetTrajectory, build_kinematics, build_target
from oculomotor.models.sensory_models.retina import ypr_to_xyz, xyz_to_ypr
from oculomotor.models.plant_models.readout import rotation_matrix as _rotation_matrix

from oculomotor.models.sensory_models.sensory_model import SensoryParams
from oculomotor.models.sensory_models               import canal   as _canal
from oculomotor.models.sensory_models               import otolith as _otolith
from oculomotor.models.sensory_models               import retina  as _retina
from oculomotor.models.brain_models.brain_model    import BrainParams
from oculomotor.models.plant_models.plant_model_first_order import PlantParams
from oculomotor.models.plant_models.muscle_geometry import M_PLANT_EYE_L, M_PLANT_EYE_R
from oculomotor.models.sensory_models import sensory_model
from oculomotor.models.brain_models   import brain_model
from oculomotor.models.plant_models   import plant_model_first_order as plant_model
from oculomotor.models.plant_models   import accommodation_plant     as acc_plant_mod


# ── Swappable brain step ────────────────────────────────────────────────────
# Default: brain_model.step. Set via set_brain_step() to test alternatives
# (e.g. unified_brain.step). Must have the same I/O signature.
_BRAIN_STEP = brain_model.step


def set_brain_step(fn):
    """Swap the brain step function used by simulate().

    Pass any callable with the same signature as brain_model.step:
        fn(x_brain, sensory_out, brain_params, noise_acc) ->
            (dx_brain, nerves, ec_vel, ec_pos, ec_verg, u_acc)

    Call set_brain_step(brain_model.step) to restore the default.
    """
    global _BRAIN_STEP
    _BRAIN_STEP = fn


# ── Prism helper ───────────────────────────────────────────────────────────────

def _apply_prism(q_eye_ypr, prism_ypr):
    """Return effective eye rotation after optical prism insertion (prism in head frame).

    The prism is mounted on glasses — fixed relative to the head, not the eye.
    It rotates the apparent direction of all visual inputs in HEAD frame before
    they reach eye optics:

        R_gaze_T_eff = R_eye.T @ R_prism @ R_head.T

    Equivalent to rotating p_from_eye by R_prism in head frame ("virtual target"):
        p_hat_virtual_head = R_prism @ R_head.T @ p_from_eye / |...|
        p_eye = R_eye.T @ p_hat_virtual_head

    Implemented by setting R_eye_eff = R_prism.T @ R_eye so that
    R_gaze_T_eff = R_eye_eff.T @ R_head.T automatically propagates through
    ALL of world_to_retina: target_pos, scene_angular_vel, scene_linear_vel,
    and target_vel are all rotated by R_prism.

    The eye's physical position (IPD offset, actual w_eye) is unaffected.

    Args:
        q_eye_ypr:  (3,) actual eye rotation [yaw, pitch, roll] deg
        prism_ypr:  (3,) prism deviation     [yaw, pitch, roll] deg (head frame);
                    positive yaw  = apparent field shifted rightward,
                    positive pitch = apparent field shifted upward.

    Returns:
        q_eff_ypr: (3,) effective eye rotation to pass to world_to_retina
    """
    # rotation_matrix expects DEGREES (it converts to rad internally) — no deg2rad here
    R_eye   = _rotation_matrix(ypr_to_xyz(q_eye_ypr))
    R_prism = _rotation_matrix(ypr_to_xyz(prism_ypr))
    R_eff   = R_prism.T @ R_eye          # R_prism^{-1} @ R_eye  (prism in head frame)

    # Rotation vector from matrix — stable via angle-axis formula.
    # skew = [R32-R23, R13-R31, R21-R12] = 2 sin(θ) * axis
    trace    = R_eff[0,0] + R_eff[1,1] + R_eff[2,2]
    cos_th   = jnp.clip(0.5 * (trace - 1.0), -1.0, 1.0)
    theta    = jnp.arccos(cos_th)
    skew     = jnp.array([R_eff[2,1] - R_eff[1,2],
                           R_eff[0,2] - R_eff[2,0],
                           R_eff[1,0] - R_eff[0,1]])
    # axis * theta = skew * (theta / (2 sin θ)); limit → 0.5 as θ → 0
    half_sinc_inv = jnp.where(theta > 1e-7, theta / (2.0 * jnp.sin(theta)), 0.5)
    q_xyz_deg = skew * half_sinc_inv * (180.0 / jnp.pi)
    return xyz_to_ypr(q_xyz_deg)


# ── Simulation config ───────────────────────────────────────────────────────────

class SimConfig(NamedTuple):
    """Solver / run settings — not model parameters, not learnable.

    dt_solve: Heun fixed step (s).  Must satisfy dt < 2 * tau_stage_vis.
              With N_STAGES=40 and tau_vis=0.08 s → tau_stage = 0.002 s
              → dt_max = 0.004 s.  Default 0.001 s gives 4× safety margin.

    warmup_s: Settling period (s) prepended before t=0.  The stimulus is
              held at its t=0 value for this duration so that fast states
              (visual delay cascades, canal, NI) reach steady state before
              the plotted output begins.  The warmup window is stripped from
              all returned arrays — callers always see t starting at t_array[0].
              Vergence is initialised analytically to tonic_verg (its TC ~6 s is
              too slow to settle in a short warmup).
              Set to 0.0 to disable.  Default: 3.0 s.
    """
    dt_solve: float = 0.001
    warmup_s: float = 3.0


# ── Top-level parameter container ──────────────────────────────────────────────

class Params(NamedTuple):
    sensory: SensoryParams = SensoryParams()
    plant:   PlantParams   = PlantParams()
    brain:   BrainParams   = BrainParams()


def _tonic_verg_from_ipd(ipd: float, dist_m: float = 1.0) -> float:
    """Tonic vergence angle for a given IPD and dark-vergence resting distance."""
    import math
    return 2.0 * math.degrees(math.atan(ipd / 2.0 / dist_m))


def default_params() -> Params:
    """Healthy primate default parameters.

    tonic_verg is computed from SensoryParams.ipd so it stays consistent
    when IPD is changed via with_sensory(..., ipd=X) + default_params().
    """
    sp = SensoryParams()
    return Params(
        sensory=sp,
        plant=PlantParams(),
        brain=BrainParams(tonic_verg=_tonic_verg_from_ipd(sp.ipd)),
    )


def with_brain(params: Params, **kwargs) -> Params:
    return params._replace(brain=params.brain._replace(**kwargs))


def with_sensory(params: Params, **kwargs) -> Params:
    new = params._replace(sensory=params.sensory._replace(**kwargs))
    if 'ipd' in kwargs:
        # Keep tonic_verg consistent: rescale assuming same 1m dark-vergence distance.
        new = new._replace(brain=new.brain._replace(
            tonic_verg=_tonic_verg_from_ipd(kwargs['ipd'])))
    return new


def with_plant(params: Params, **kwargs) -> Params:
    return params._replace(plant=params.plant._replace(**kwargs))


def with_cerebellum(params: Params, **kwargs) -> Params:
    """Alias of `with_brain` for cerebellum fields (currently K_cereb_pu).

    Re-routes through `with_brain` for naming clarity at call sites.
    """
    return with_brain(params, **kwargs)


SIM_CONFIG_DEFAULT = SimConfig()
PARAMS_DEFAULT     = default_params()


# ── Vestibular lesion helpers ───────────────────────────────────────────────────

def with_uvh(params: Params, side: str = 'left',
             canal_gain_frac: float = 0.1,
             b_lesion: float = 70.0) -> Params:
    """Unilateral vestibular hypofunction."""
    import numpy as np
    b_healthy = float(np.mean(np.broadcast_to(
        np.asarray(params.brain.b_vs, dtype=float), (6,))))
    cg = np.array(params.sensory.canal_gains, dtype=float)

    if side == 'left':
        cg[:3] *= canal_gain_frac
        b_vs = jnp.array([b_healthy, b_healthy, b_healthy, b_lesion, b_lesion, b_lesion], dtype=jnp.float32)
    elif side == 'right':
        cg[3:] *= canal_gain_frac
        b_vs = jnp.array([b_lesion, b_lesion, b_lesion, b_healthy, b_healthy, b_healthy], dtype=jnp.float32)
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    return params._replace(
        sensory = params.sensory._replace(canal_gains=jnp.array(cg, dtype=jnp.float32)),
        brain   = params.brain._replace(b_vs=b_vs),
    )


def with_vn_lesion(params: Params, side: str = 'left') -> Params:
    """Unilateral VN infarct — silences the affected population entirely."""
    import numpy as np
    b_healthy = float(np.mean(np.broadcast_to(
        np.asarray(params.brain.b_vs, dtype=float), (6,))))

    if side == 'left':
        b_vs = jnp.array([b_healthy, b_healthy, b_healthy, 0.0, 0.0, 0.0], dtype=jnp.float32)
    elif side == 'right':
        b_vs = jnp.array([0.0, 0.0, 0.0, b_healthy, b_healthy, b_healthy], dtype=jnp.float32)
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    return params._replace(brain=params.brain._replace(b_vs=b_vs))


__all__ = [
    'SimState', 'ODE_ocular_motor', 'simulate',
    'Params', 'SimConfig', 'SensoryParams', 'PlantParams', 'BrainParams',
    'default_params', 'with_brain', 'with_sensory', 'with_plant', 'with_cerebellum',
    'with_uvh', 'with_vn_lesion',
    'PARAMS_DEFAULT', 'SIM_CONFIG_DEFAULT',
]


# ── SimState ───────────────────────────────────────────────────────────────────

class SimState(NamedTuple):
    """Structured ODE state — JAX-compatible pytree (NamedTuple).

    All four fields are nested NamedTuples (or scalar arrays).  No flat-array
    layouts — read fields directly via `state.brain.<sub>.<field>`,
    `state.sensory.canal.x1`, etc.

    Groups:
        sensory   sensory_model.State NT — canal, otolith, retina_L, retina_R
        brain     brain_model.BrainState NT — pc, sm, pt, sg, pu, va, ni, fcp, ec_scene, ec_target
        plant     plant_model.State NT — left, right eye rotation vectors (deg)
        acc_plant (1,)  Lens accommodation plant state (D)
    """
    sensory:   'sensory_model.State'    # nested per-sensor
    brain:     'brain_model.BrainState' # nested per-subsystem
    plant:     'plant_model.State'      # binocular eye rotation vectors
    acc_plant: jnp.ndarray              # (1,) lens accommodation (D)


# ── ODE vector field ───────────────────────────────────────────────────────────

def ODE_ocular_motor(t, state, args):
    """ODE right-hand side: chains sensory, brain, and plant models.

    Compatible with diffrax: signature f(t, state, args).

    World frame is LEFT-HANDED: x=right, y=up, z=forward  (x × y = −z).

    Evaluation order:
        1. read_outputs  — slice delayed signals from sensory state for brain
        2. Compute v_target from cross(p_target, dp_target/dt) / |p_target|²
        3. brain_model   — VS + NI + SG + EC → motor_cmd
        4. plant_model   — motor_cmd → dx_plant_L/R; w_eye = dx_plant (deg/s)
        5. sensory_model — canal + visual delay cascades driven by w_eye

    Args:
        t:     scalar time (s)
        state: SimState pytree
        args:  31-element tuple — see simulate() for layout

    Returns:
        SimState of derivatives (dsensory, dbrain, dplant)
    """
    (theta,
     head_q_interp, head_w_interp, head_x_interp, head_v_interp, head_a_interp,
     scene_q_L_interp, scene_w_L_interp, scene_x_L_interp, scene_v_L_interp,
     scene_q_R_interp, scene_w_R_interp, scene_x_R_interp, scene_v_R_interp,
     target_p_L_interp, target_dv_L_interp,
     target_p_R_interp, target_dv_R_interp,
     prism_L_interp, prism_R_interp,
     lens_L_interp, lens_R_interp,
     scene_present_L_interp, scene_present_R_interp,
     target_present_L_interp, target_present_R_interp,
     target_strobed_interp,
     noise_canal_interp, noise_slip_interp, noise_pos_interp,
     noise_vel_interp,
     noise_acc_interp) = args

    # ── External inputs at time t ────────────────────────────────────────────
    q_head  = head_q_interp.evaluate(t)       # (3,) [yaw,pitch,roll] deg
    w_head  = head_w_interp.evaluate(t)       # (3,) angular velocity deg/s
    x_head  = head_x_interp.evaluate(t)       # (3,) linear position m
    v_head  = head_v_interp.evaluate(t)       # (3,) linear velocity m/s
    a_head  = head_a_interp.evaluate(t)       # (3,) linear acceleration m/s²

    # Per-eye scene (identical in monocular mode; diverge in stereo or OKN-monocular)
    q_scene_L = scene_q_L_interp.evaluate(t); w_scene_L = scene_w_L_interp.evaluate(t)
    x_scene_L = scene_x_L_interp.evaluate(t); v_scene_L = scene_v_L_interp.evaluate(t)
    q_scene_R = scene_q_R_interp.evaluate(t); w_scene_R = scene_w_R_interp.evaluate(t)
    x_scene_R = scene_x_R_interp.evaluate(t); v_scene_R = scene_v_R_interp.evaluate(t)

    # Per-eye target (identical in monocular mode; diverge in dichoptic / stereo)
    p_target_L = target_p_L_interp.evaluate(t); dp_dt_L = target_dv_L_interp.evaluate(t)
    p_target_R = target_p_R_interp.evaluate(t); dp_dt_R = target_dv_R_interp.evaluate(t)

    scene_present_L  = scene_present_L_interp.evaluate(t)
    scene_present_R  = scene_present_R_interp.evaluate(t)
    target_present_L = target_present_L_interp.evaluate(t)
    target_present_R = target_present_R_interp.evaluate(t)
    target_strobed   = target_strobed_interp.evaluate(t)
    lens_L           = lens_L_interp.evaluate(t)
    lens_R           = lens_R_interp.evaluate(t)

    # ── Sensory: read delayed cascade outputs ────────────────────────────────
    sensory_out = sensory_model.read_outputs(state.sensory, theta.sensory, q_head, a_head)

    # ── Sensory noise ─────────────────────────────────────────────────────────
    # Canal noise → afferent rates (cyclopean already at this stage).
    # Visual noise is applied to BOTH eyes' retina outputs identically — this is
    # a stand-in for cyclopean noise that occurs post-fusion in the brain. With
    # equal noise on L and R, the binocular fusion policy averages it through
    # cleanly to the cyclopean output.
    nslip = noise_slip_interp.evaluate(t)
    nvel  = noise_vel_interp.evaluate(t)
    npos  = noise_pos_interp.evaluate(t)
    sensory_out = sensory_out._replace(
        canal    = sensory_out.canal + noise_canal_interp.evaluate(t),
        retina_L = sensory_out.retina_L._replace(
            scene_angular_vel = sensory_out.retina_L.scene_angular_vel + nslip,
            target_vel        = sensory_out.retina_L.target_vel        + nvel,
            target_pos        = sensory_out.retina_L.target_pos        + npos,
        ),
        retina_R = sensory_out.retina_R._replace(
            scene_angular_vel = sensory_out.retina_R.scene_angular_vel + nslip,
            target_vel        = sensory_out.retina_R.target_vel        + nvel,
            target_pos        = sensory_out.retina_R.target_pos        + npos,
        ),
    )

    # ── Brain: VS + NI + SG + pursuit + vergence + accommodation ─────────────
    dbrain, nerves, ec_vel, ec_pos, ec_verg, u_acc = _BRAIN_STEP(
        state.brain, sensory_out, theta.brain, noise_acc_interp.evaluate(t))

    # ── Plant ─────────────────────────────────────────────────────────────────
    dx_p_L, q_eye_L, w_eye_L = plant_model.step(state.plant.left,  nerves[:6], theta.plant, M_PLANT_EYE_L)
    dx_p_R, q_eye_R, w_eye_R = plant_model.step(state.plant.right, nerves[6:], theta.plant, M_PLANT_EYE_R)

    # ── Accommodation plant ────────────────────────────────────────────────────
    # u_acc = brain neural command + CA/C feedforward (combined inside va.step).
    dx_acc_plant, _ = acc_plant_mod.step(
        state.acc_plant, u_acc, theta.brain.tau_acc_plant)

    # ── Optical interventions — applied after plant, before sensory step ─────
    # Prisms are head-frame mounted (glasses); they rotate the apparent gaze direction
    # without changing the physical eye velocity. All of world_to_retina propagates
    # through the effective eye orientation automatically.
    q_eye_L_eff = _apply_prism(q_eye_L, prism_L_interp.evaluate(t))
    q_eye_R_eff = _apply_prism(q_eye_R, prism_R_interp.evaluate(t))

    # ── Per-eye defocus: acc_demand + refractive_error − x_plant ─────────────
    # Defocus is the blur signal at the retina. Computed here (using current
    # x_plant from state) and passed to sensory_model.step() which gates it by
    # defocus_visible and delays it through the cyclopean cascade.
    # refractive_error (D): >0 hyperopia (needs more acc), <0 myopia (needs less).
    x_plant_now = state.acc_plant[0]
    re = theta.brain.refractive_error
    defocus_L = 1.0 / (jnp.linalg.norm(p_target_L) + 1e-9) + lens_L + re - x_plant_now
    defocus_R = 1.0 / (jnp.linalg.norm(p_target_R) + 1e-9) + lens_R + re - x_plant_now

    # ── Sensory: ODE step — must follow plant ────────────────────────────────
    dx_sensory = sensory_model.step(
        state.sensory,
        q_head, w_head, x_head, v_head, a_head,
        q_eye_L_eff, w_eye_L, q_eye_R_eff, w_eye_R,
        q_scene_L, w_scene_L, x_scene_L, v_scene_L,
        q_scene_R, w_scene_R, x_scene_R, v_scene_R,
        p_target_L, dp_dt_L,
        p_target_R, dp_dt_R,
        defocus_L, defocus_R,
        scene_present_L, scene_present_R,
        target_present_L, target_present_R, target_strobed,
        ec_vel, ec_pos, ec_verg,
        theta.sensory)

    return SimState(
        sensory   = dx_sensory,
        brain     = dbrain,
        plant     = plant_model.State(left=dx_p_L, right=dx_p_R),
        acc_plant = dx_acc_plant,
    )


# ── Simulation entry point ─────────────────────────────────────────────────────

def simulate(
    params,
    t_array,
    head:   KinematicTrajectory = None,
    scene:  KinematicTrajectory = None,
    target: TargetTrajectory    = None,
    scene_present_array=None,
    scene_present_L_array=None,
    scene_present_R_array=None,
    target_present_array=None,
    target_present_L_array=None,
    target_present_R_array=None,
    target_strobed_array=None,
    # ── Optical interventions (time-varying, per-eye stimulus arrays) ─────────
    prism_L_array=None,     # (T, 3) prism deviation [yaw, pitch, roll] deg, L eye. None → no prism.
    prism_R_array=None,     # (T, 3) prism deviation [yaw, pitch, roll] deg, R eye. None → no prism.
    lens_L_array=None,      # (T,)   accommodation demand offset (diopters), L eye. None → no lens.
    lens_R_array=None,      # (T,)   accommodation demand offset (diopters), R eye. None → no lens.
    # ── Stereo per-eye scene / target overrides ───────────────────────────────
    scene_L: KinematicTrajectory = None,   # L-eye scene. None → both eyes use shared `scene`.
    scene_R: KinematicTrajectory = None,   # R-eye scene. None → both eyes use shared `scene`.
    target_L: TargetTrajectory   = None,   # L-eye target. None → both eyes use shared `target`.
    target_R: TargetTrajectory   = None,   # R-eye target. None → both eyes use shared `target`.
    max_steps=10000,
    sim_config=None,
    return_states=False,
    key=None,
):
    """Integrate the oculomotor ODE and return eye rotation vectors.

    World frame is LEFT-HANDED: x=right, y=up, z=forward  (x × y = −z).

    Args:
        params:     Params — model parameters (see default_params()).
        t_array:    (T,) time array (s).
        head:       KinematicTrajectory or None (stationary).
                    Carries 6-DOF head pose + all derivatives.
                    Build with head_rotation_step(), head_impulse(), etc.
                    from oculomotor.sim.kinematics.
        scene:      KinematicTrajectory or None (dark, all zeros).
                    Build with scene_rotation_step(), scene_stationary(), etc.
        target:     TargetTrajectory or None (straight-ahead [0,0,1] m).
                    Build with target_stationary(), target_steps(),
                    target_ramp(), or build_target().
        scene_present_array:    (T,) in [0,1] — both-eye visibility.
                    None → 1.0 if scene is provided and has non-zero rot_vel,
                           0.0 otherwise (dark).
        scene_present_L/R_array: per-eye override. None → scene_present_array.
        target_present_array:   (T,) in [0,1]. None → 1.0 (target always visible).
        target_present_L/R_array: per-eye override.
        target_strobed_array:   (T,) ∈ {0,1}. None → 0.
        max_steps:  ODE solver step budget.
        sim_config: SimConfig — solver settings. Default: SIM_CONFIG_DEFAULT.
        return_states: if True, return full SimState trajectory instead of
                       just eye rotation (T, 6).
        key:        jax.random.PRNGKey for sensory noise. Default PRNGKey(0).

    Returns:
        If return_states=False (default):
            eye_rot: (T, 6) [left(3) | right(3)] eye rotation vectors (deg)
        If return_states=True:
            SimState pytree, each field shape (T, N).
    """
    import numpy as np

    cfg = sim_config if sim_config is not None else SIM_CONFIG_DEFAULT
    dt  = cfg.dt_solve
    T   = len(t_array)

    # ── Normalise scalar parameters to their required array shapes ────────────
    # Done once here so ODE step functions receive ready-to-use arrays.
    b_vs_6 = jnp.broadcast_to(jnp.asarray(params.brain.b_vs, dtype=jnp.float32), (6,))
    params = params._replace(brain=params.brain._replace(b_vs=b_vs_6))

    # ── Default trajectories ──────────────────────────────────────────────────
    if head is None:
        head = build_kinematics(t_array)
    if scene is None:
        scene = build_kinematics(t_array)
    if target is None:
        target = build_target(t_array, lin_pos=jnp.tile(jnp.array([0.0, 0.0, 1.0]), (T, 1)))

    # ── Scene presence ────────────────────────────────────────────────────────
    if scene_present_array is not None:
        sg_both = jnp.asarray(scene_present_array, dtype=jnp.float32)
    else:
        has_motion = jnp.any(jnp.asarray(scene.rot_vel) != 0.0).item()
        sg_both = jnp.ones(T, dtype=jnp.float32) if has_motion else jnp.zeros(T, dtype=jnp.float32)
    sg_L = jnp.asarray(scene_present_L_array, dtype=jnp.float32) if scene_present_L_array is not None else sg_both
    sg_R = jnp.asarray(scene_present_R_array, dtype=jnp.float32) if scene_present_R_array is not None else sg_both

    # ── Target presence ───────────────────────────────────────────────────────
    tg_both = jnp.asarray(target_present_array, dtype=jnp.float32) if target_present_array is not None else jnp.ones(T, dtype=jnp.float32)
    tg_L = jnp.asarray(target_present_L_array, dtype=jnp.float32) if target_present_L_array is not None else tg_both
    tg_R = jnp.asarray(target_present_R_array, dtype=jnp.float32) if target_present_R_array is not None else tg_both

    # ── Strobe flag ───────────────────────────────────────────────────────────
    ts = jnp.asarray(target_strobed_array, dtype=jnp.float32) if target_strobed_array is not None else jnp.zeros(T, dtype=jnp.float32)

    # ── Extract trajectory arrays ─────────────────────────────────────────────
    head_q = jnp.asarray(head.rot_pos, dtype=jnp.float32)   # (T,3) deg
    head_w = jnp.asarray(head.rot_vel, dtype=jnp.float32)   # (T,3) deg/s
    head_x = jnp.asarray(head.lin_pos, dtype=jnp.float32)   # (T,3) m
    head_v = jnp.asarray(head.lin_vel, dtype=jnp.float32)   # (T,3) m/s
    head_a = jnp.asarray(head.lin_acc, dtype=jnp.float32)   # (T,3) m/s²

    scene_q = jnp.asarray(scene.rot_pos, dtype=jnp.float32)
    scene_w = jnp.asarray(scene.rot_vel, dtype=jnp.float32)
    scene_x = jnp.asarray(scene.lin_pos, dtype=jnp.float32)
    scene_v = jnp.asarray(scene.lin_vel, dtype=jnp.float32)

    tgt_p  = jnp.asarray(target.lin_pos, dtype=jnp.float32)   # (T,3) m
    tgt_dv = jnp.asarray(target.lin_vel, dtype=jnp.float32)   # (T,3) m/s

    # ── Per-eye scene (default: both eyes use shared scene) ───────────────────
    scene_q_L = scene_q if scene_L is None else jnp.asarray(scene_L.rot_pos, dtype=jnp.float32)
    scene_w_L = scene_w if scene_L is None else jnp.asarray(scene_L.rot_vel, dtype=jnp.float32)
    scene_x_L = scene_x if scene_L is None else jnp.asarray(scene_L.lin_pos, dtype=jnp.float32)
    scene_v_L = scene_v if scene_L is None else jnp.asarray(scene_L.lin_vel, dtype=jnp.float32)
    scene_q_R = scene_q if scene_R is None else jnp.asarray(scene_R.rot_pos, dtype=jnp.float32)
    scene_w_R = scene_w if scene_R is None else jnp.asarray(scene_R.rot_vel, dtype=jnp.float32)
    scene_x_R = scene_x if scene_R is None else jnp.asarray(scene_R.lin_pos, dtype=jnp.float32)
    scene_v_R = scene_v if scene_R is None else jnp.asarray(scene_R.lin_vel, dtype=jnp.float32)

    # ── Per-eye target (default: both eyes use shared target) ─────────────────
    tgt_p_L  = tgt_p  if target_L is None else jnp.asarray(target_L.lin_pos, dtype=jnp.float32)
    tgt_dv_L = tgt_dv if target_L is None else jnp.asarray(target_L.lin_vel, dtype=jnp.float32)
    tgt_p_R  = tgt_p  if target_R is None else jnp.asarray(target_R.lin_pos, dtype=jnp.float32)
    tgt_dv_R = tgt_dv if target_R is None else jnp.asarray(target_R.lin_vel, dtype=jnp.float32)

    # ── Prism: deviation in [yaw, pitch, roll] deg per eye ───────────────────
    # Positive yaw = apparent field shifted rightward; positive pitch = upward.
    # Use prism_from_pd() in stimuli.py to convert from clinical PD + base angle.
    prism_L = jnp.asarray(prism_L_array, dtype=jnp.float32) if prism_L_array is not None \
              else jnp.zeros((T, 3), dtype=jnp.float32)
    prism_R = jnp.asarray(prism_R_array, dtype=jnp.float32) if prism_R_array is not None \
              else jnp.zeros((T, 3), dtype=jnp.float32)

    # ── Lens: accommodation demand offset (diopters) per eye ─────────────────
    lens_L_arr = jnp.asarray(lens_L_array, dtype=jnp.float32) if lens_L_array is not None \
                 else jnp.zeros(T, dtype=jnp.float32)
    lens_R_arr = jnp.asarray(lens_R_array, dtype=jnp.float32) if lens_R_array is not None \
                 else jnp.zeros(T, dtype=jnp.float32)

    # ── Sensory noise ─────────────────────────────────────────────────────────
    if key is None:
        key = jax.random.PRNGKey(0)
    k_canal, k_slip, k_pos, k_vel, k_acc_n = jax.random.split(key, 5)

    # All four sensory noise sources are Ornstein-Uhlenbeck processes:
    #   x_{n+1} = α·x_n + √(1−α²)·σ·w_n,   α = exp(−dt/τ)
    # Stationary stddev = σ. Short τ → ~white noise (band-limited at 1/(2π·τ)).
    def _ou_noise(rng_key, shape, sigma, tau):
        alpha = jnp.exp(-dt / tau)
        drive = jnp.sqrt(1.0 - alpha ** 2) * sigma
        white = jax.random.normal(rng_key, shape)
        x0    = jnp.zeros(shape[1:])
        def step(carry, w):
            x = alpha * carry + drive * w
            return x, x
        _, ou = jax.lax.scan(step, x0, white)
        return ou

    sp = params.sensory
    noise_canal = _ou_noise(k_canal, (T, 6), sp.sigma_canal, sp.tau_canal_drift)
    noise_slip  = _ou_noise(k_slip,  (T, 3), sp.sigma_slip,  sp.tau_slip_drift)
    noise_pos   = _ou_noise(k_pos,   (T, 3), sp.sigma_pos,   sp.tau_pos_drift)
    noise_vel   = _ou_noise(k_vel,   (T, 3), sp.sigma_vel,   sp.tau_vel_drift)
    # Accumulator diffusion noise: pre-scaled so that after ODE multiply-by-dt gives
    # N(0, sigma_acc·√dt) per step — standard Euler-Maruyama / Langevin scaling.
    noise_acc   = jax.random.normal(k_acc_n, (T,))   * (params.brain.sigma_acc / jnp.sqrt(dt))

    # ── Warmup prepend ────────────────────────────────────────────────────────
    warmup_s = cfg.warmup_s
    warmup_T = int(round(warmup_s / dt))

    if warmup_T > 0:
        t_warmup = t_array[0] + dt * (jnp.arange(warmup_T) - warmup_T)
        t_full   = jnp.concatenate([t_warmup, t_array])

        def _prepend(arr):
            reps = (warmup_T,) + (1,) * (arr.ndim - 1)
            return jnp.concatenate([jnp.tile(arr[0:1], reps), arr], axis=0)

        head_q = _prepend(head_q); head_w = _prepend(head_w)
        head_x = _prepend(head_x); head_v = _prepend(head_v); head_a = _prepend(head_a)
        scene_q = _prepend(scene_q); scene_w = _prepend(scene_w)
        scene_x = _prepend(scene_x); scene_v = _prepend(scene_v)
        scene_q_L = _prepend(scene_q_L); scene_w_L = _prepend(scene_w_L)
        scene_x_L = _prepend(scene_x_L); scene_v_L = _prepend(scene_v_L)
        scene_q_R = _prepend(scene_q_R); scene_w_R = _prepend(scene_w_R)
        scene_x_R = _prepend(scene_x_R); scene_v_R = _prepend(scene_v_R)
        tgt_p  = _prepend(tgt_p);  tgt_dv  = _prepend(tgt_dv)
        tgt_p_L  = _prepend(tgt_p_L);  tgt_dv_L = _prepend(tgt_dv_L)
        tgt_p_R  = _prepend(tgt_p_R);  tgt_dv_R = _prepend(tgt_dv_R)
        prism_L = _prepend(prism_L)
        prism_R = _prepend(prism_R)
        lens_L_arr = _prepend(lens_L_arr[:, None])[:, 0]
        lens_R_arr = _prepend(lens_R_arr[:, None])[:, 0]
        sg_L = _prepend(sg_L[:, None])[:, 0]; sg_R = _prepend(sg_R[:, None])[:, 0]
        tg_L = _prepend(tg_L[:, None])[:, 0]; tg_R = _prepend(tg_R[:, None])[:, 0]
        ts   = _prepend(ts[:, None])[:, 0]

        _z6 = jnp.zeros((warmup_T, 6))
        _z3 = jnp.zeros((warmup_T, 3))
        noise_canal = jnp.concatenate([_z6, noise_canal], axis=0)
        noise_slip  = jnp.concatenate([_z3, noise_slip],  axis=0)
        noise_pos   = jnp.concatenate([_z3, noise_pos],   axis=0)
        noise_vel   = jnp.concatenate([_z3, noise_vel],   axis=0)
        noise_acc   = jnp.concatenate([jnp.zeros(warmup_T), noise_acc])
    else:
        t_full   = t_array
        warmup_T = 0

    # ── Build interpolants ────────────────────────────────────────────────────
    def _interp(ys):
        return diffrax.LinearInterpolation(ts=t_full, ys=ys)

    head_q_interp   = _interp(head_q);  head_w_interp = _interp(head_w)
    head_x_interp   = _interp(head_x);  head_v_interp = _interp(head_v)
    head_a_interp   = _interp(head_a)
    scene_q_L_interp  = _interp(scene_q_L); scene_w_L_interp = _interp(scene_w_L)
    scene_x_L_interp  = _interp(scene_x_L); scene_v_L_interp = _interp(scene_v_L)
    scene_q_R_interp  = _interp(scene_q_R); scene_w_R_interp = _interp(scene_w_R)
    scene_x_R_interp  = _interp(scene_x_R); scene_v_R_interp = _interp(scene_v_R)
    target_p_L_interp  = _interp(tgt_p_L);  target_dv_L_interp = _interp(tgt_dv_L)
    target_p_R_interp  = _interp(tgt_p_R);  target_dv_R_interp = _interp(tgt_dv_R)
    prism_L_interp = _interp(prism_L)
    prism_R_interp = _interp(prism_R)
    lens_L_interp      = _interp(lens_L_arr)
    lens_R_interp      = _interp(lens_R_arr)
    sp_L_interp     = _interp(sg_L);    sp_R_interp  = _interp(sg_R)
    tp_L_interp     = _interp(tg_L);    tp_R_interp  = _interp(tg_R)
    ts_interp       = _interp(ts)
    noise_canal_interp = _interp(noise_canal)
    noise_slip_interp  = _interp(noise_slip)
    noise_pos_interp   = _interp(noise_pos)
    noise_vel_interp   = _interp(noise_vel)
    noise_acc_interp   = _interp(noise_acc)

    # ── Initial state ─────────────────────────────────────────────────────────
    sensory_x0 = sensory_model.State(
        canal    = _canal.rest_state(),
        otolith  = _otolith.rest_state(),   # both sides settled to gravity
        retina_L = _retina.rest_state(),
        retina_R = _retina.rest_state(),
    )
    brain_x0 = brain_model.make_x0(params.brain)
    plant_x0 = plant_model.rest_state()

    x0 = SimState(
        sensory   = sensory_x0,
        brain     = brain_x0,
        plant     = plant_x0,
        acc_plant = jnp.array([params.brain.tonic_acc]),  # start at dark focus (D)
    )

    # ── Solve ─────────────────────────────────────────────────────────────────
    total_steps = int(jnp.ceil((t_full[-1] - t_full[0]) / dt).item()) + 100
    max_steps   = max(max_steps, total_steps)

    ode_args = (
        params,
        head_q_interp, head_w_interp, head_x_interp, head_v_interp, head_a_interp,
        scene_q_L_interp, scene_w_L_interp, scene_x_L_interp, scene_v_L_interp,
        scene_q_R_interp, scene_w_R_interp, scene_x_R_interp, scene_v_R_interp,
        target_p_L_interp, target_dv_L_interp,
        target_p_R_interp, target_dv_R_interp,
        prism_L_interp, prism_R_interp,
        lens_L_interp, lens_R_interp,
        sp_L_interp, sp_R_interp, tp_L_interp, tp_R_interp, ts_interp,
        noise_canal_interp, noise_slip_interp, noise_pos_interp,
        noise_vel_interp,
        noise_acc_interp,
    )

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(ODE_ocular_motor),
        diffrax.Heun(),
        t0=t_full[0],
        t1=t_full[-1],
        dt0=dt,
        y0=x0,
        args=ode_args,
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=t_full),
        max_steps=max_steps,
    )

    # `solution.ys` is a SimState pytree with a leading time axis on every leaf.
    # All four fields are nested NamedTuples (or plain ndarray for acc_plant) —
    # tree_map slices each leaf along the time axis.
    ys = jax.tree_util.tree_map(lambda x: x[warmup_T:], solution.ys)

    if return_states:
        return ys
    # Default return: flat (T, 6) plant trajectory [L (3) | R (3)] — convenience
    # for legacy callers that just want eye position over time.
    return jnp.concatenate([ys.plant.left, ys.plant.right], axis=1)   # (T, 6)
