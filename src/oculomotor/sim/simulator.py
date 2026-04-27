"""Full oculomotor simulator — wires sensory_model, brain_model, and plant_model.

World-frame convention: x=right, y=up, z=forward.

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
        v_target = _rv2q( cross(p_target, dp_target/dt) / |p_target|² )  [deg/s]

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

from oculomotor.models.sensory_models.sensory_model import (
    _IDX_C, _IDX_OTO, _IDX_VIS, _IDX_VIS_L, _IDX_VIS_R, SensoryParams,
)
from oculomotor.models.sensory_models               import otolith as _otolith
from oculomotor.models.brain_models.brain_model    import (
    _IDX_VS, _IDX_VS_L, _IDX_VS_R, _IDX_VS_NULL,
    _IDX_NI, _IDX_NI_L, _IDX_NI_R, _IDX_NI_NULL,
    _IDX_SG, _IDX_EC, _IDX_EC_OKR, _IDX_GRAV, _IDX_PURSUIT, _IDX_VERG, _IDX_ACC,
    BrainParams,
)
from oculomotor.models.plant_models.plant_model_first_order import PlantParams, _IDX_P_L, _IDX_P_R
from oculomotor.models.plant_models.muscle_geometry import M_PLANT_EYE_L, M_PLANT_EYE_R
from oculomotor.models.sensory_models import sensory_model
from oculomotor.models.brain_models   import brain_model
from oculomotor.models.plant_models   import plant_model_first_order as plant_model


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
              Vergence is initialised analytically to phoria (its TC ~25 s is
              far too slow to settle in a short warmup).
              Set to 0.0 to disable.  Default: 3.0 s.
    """
    dt_solve: float = 0.001
    warmup_s: float = 3.0


# ── Top-level parameter container ──────────────────────────────────────────────

class Params(NamedTuple):
    sensory: SensoryParams = SensoryParams()
    plant:   PlantParams   = PlantParams()
    brain:   BrainParams   = BrainParams()


def default_params() -> Params:
    """Healthy primate default parameters."""
    return Params()


def with_brain(params: Params, **kwargs) -> Params:
    return params._replace(brain=params.brain._replace(**kwargs))


def with_sensory(params: Params, **kwargs) -> Params:
    return params._replace(sensory=params.sensory._replace(**kwargs))


def with_plant(params: Params, **kwargs) -> Params:
    return params._replace(plant=params.plant._replace(**kwargs))


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
    '_IDX_C', '_IDX_OTO', '_IDX_VIS', '_IDX_VIS_L', '_IDX_VIS_R',
    '_IDX_VS', '_IDX_VS_L', '_IDX_VS_R', '_IDX_VS_NULL',
    '_IDX_NI', '_IDX_NI_L', '_IDX_NI_R', '_IDX_NI_NULL',
    '_IDX_SG', '_IDX_EC', '_IDX_EC_OKR', '_IDX_GRAV', '_IDX_PURSUIT', '_IDX_VERG', '_IDX_ACC',
    '_IDX_P_L', '_IDX_P_R',
    'Params', 'SimConfig', 'SensoryParams', 'PlantParams', 'BrainParams',
    'default_params', 'with_brain', 'with_sensory', 'with_plant',
    'with_uvh', 'with_vn_lesion',
    'PARAMS_DEFAULT', 'SIM_CONFIG_DEFAULT',
]


# ── SimState ───────────────────────────────────────────────────────────────────

class SimState(NamedTuple):
    """Structured ODE state — JAX-compatible pytree (NamedTuple).

    Groups:
        sensory  (978)  Canal + otolith + two retinal delay cascades (L and R).
        brain    (156)  Central computation: VS, NI, SG, EC, gravity, pursuit, vergence.
        plant      (6)  Two extraocular plants — [left (3) | right (3)] eye rotation (deg).
    """
    sensory: jnp.ndarray   # (978,)
    brain:   jnp.ndarray   # (156,)
    plant:   jnp.ndarray   #   (6,)


# ── ODE vector field ───────────────────────────────────────────────────────────

def ODE_ocular_motor(t, state, args):
    """ODE right-hand side: chains sensory, brain, and plant models.

    Compatible with diffrax: signature f(t, state, args).

    World-frame convention: x=right, y=up, z=forward.

    Evaluation order:
        1. read_outputs  — slice delayed signals from sensory state for brain
        2. Compute v_target from cross(p_target, dp_target/dt) / |p_target|²
        3. brain_model   — VS + NI + SG + EC → motor_cmd
        4. plant_model   — motor_cmd → dx_plant_L/R; w_eye = dx_plant (deg/s)
        5. sensory_model — canal + visual delay cascades driven by w_eye

    Args:
        t:     scalar time (s)
        state: SimState pytree
        args:  25-element tuple — see simulate() for layout

    Returns:
        SimState of derivatives (dsensory, dbrain, dplant)
    """
    (theta,
     head_q_interp, head_w_interp, head_x_interp, head_v_interp, head_a_interp,
     scene_q_interp, scene_w_interp, scene_x_interp, scene_v_interp,
     target_p_interp, target_dv_interp,
     scene_present_L_interp, scene_present_R_interp,
     target_present_L_interp, target_present_R_interp,
     target_strobed_interp,
     noise_canal_interp, noise_slip_L_interp, noise_slip_R_interp,
     noise_pos_L_interp, noise_pos_R_interp,
     noise_vel_L_interp, noise_vel_R_interp,
     noise_acc_interp) = args

    # ── External inputs at time t ────────────────────────────────────────────
    q_head  = head_q_interp.evaluate(t)       # (3,) [yaw,pitch,roll] deg
    w_head  = head_w_interp.evaluate(t)       # (3,) angular velocity deg/s
    x_head  = head_x_interp.evaluate(t)       # (3,) linear position m
    v_head  = head_v_interp.evaluate(t)       # (3,) linear velocity m/s
    a_head  = head_a_interp.evaluate(t)       # (3,) linear acceleration m/s²

    q_scene = scene_q_interp.evaluate(t)      # (3,) scene rotation vector deg
    w_scene = scene_w_interp.evaluate(t)      # (3,) scene angular velocity deg/s
    x_scene = scene_x_interp.evaluate(t)      # (3,) scene position m
    v_scene = scene_v_interp.evaluate(t)      # (3,) scene velocity m/s

    p_target = target_p_interp.evaluate(t)    # (3,) target position m (world frame)
    dp_dt    = target_dv_interp.evaluate(t)   # (3,) target Cartesian velocity m/s

    scene_present_L  = scene_present_L_interp.evaluate(t)
    scene_present_R  = scene_present_R_interp.evaluate(t)
    target_present_L = target_present_L_interp.evaluate(t)
    target_present_R = target_present_R_interp.evaluate(t)
    target_strobed   = target_strobed_interp.evaluate(t)

    # ── v_target: angular velocity of target direction (deg/s, world frame) ──
    # v_target_ypr = _rv2q( cross(p, dp/dt) / |p|² )
    # where _rv2q([x,y,z]) = [y, -x, z]  (xyz → [yaw,pitch,roll])
    p_norm_sq  = jnp.dot(p_target, p_target) + 1e-9
    cross_xyz  = jnp.cross(p_target, dp_dt)                    # (3,) in xyz frame
    v_target   = jnp.degrees(
        jnp.array([cross_xyz[1], -cross_xyz[0], cross_xyz[2]]) / p_norm_sq
    )

    # ── Accommodation demand: 1/z_depth (diopters) ───────────────────────────
    acc_demand = 1.0 / jnp.maximum(p_target[2], 0.05)

    # ── Sensory: read delayed cascade outputs ────────────────────────────────
    sensory_out = sensory_model.read_outputs(state.sensory, theta.sensory, q_head, a_head)
    sensory_out = sensory_out._replace(acc_demand=acc_demand)

    # ── Sensory noise ─────────────────────────────────────────────────────────
    sensory_out = sensory_out._replace(
        canal   = sensory_out.canal  + noise_canal_interp.evaluate(t),
        slip_L  = sensory_out.slip_L + noise_slip_L_interp.evaluate(t),
        slip_R  = sensory_out.slip_R + noise_slip_R_interp.evaluate(t),
        vel_L   = sensory_out.vel_L  + noise_vel_L_interp.evaluate(t),
        vel_R   = sensory_out.vel_R  + noise_vel_R_interp.evaluate(t),
        pos_L   = sensory_out.pos_L  + noise_pos_L_interp.evaluate(t),
        pos_R   = sensory_out.pos_R  + noise_pos_R_interp.evaluate(t),
    )

    # ── Brain: VS + NI + SG + EC + vergence ──────────────────────────────────
    dx_brain, motor_cmd_L, motor_cmd_R = brain_model.step(
        state.brain, sensory_out, theta.brain, noise_acc_interp.evaluate(t))

    # ── Plant ─────────────────────────────────────────────────────────────────
    dx_p_L, q_eye_L, w_eye_L = plant_model.step(state.plant[_IDX_P_L], motor_cmd_L, theta.plant, M_PLANT_EYE_L)
    dx_p_R, q_eye_R, w_eye_R = plant_model.step(state.plant[_IDX_P_R], motor_cmd_R, theta.plant, M_PLANT_EYE_R)
    dx_plant = jnp.concatenate([dx_p_L, dx_p_R])

    # ── Sensory: ODE step — must follow plant ────────────────────────────────
    dx_sensory = sensory_model.step(
        state.sensory,
        q_head, w_head, x_head, v_head, a_head,
        q_eye_L, w_eye_L, q_eye_R, w_eye_R,
        q_scene, w_scene, x_scene, v_scene,
        v_target, p_target,
        scene_present_L, scene_present_R,
        target_present_L, target_present_R, target_strobed,
        theta.sensory)

    return SimState(
        sensory = dx_sensory,
        brain   = dx_brain,
        plant   = dx_plant,
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
    max_steps=10000,
    sim_config=None,
    return_states=False,
    key=None,
):
    """Integrate the oculomotor ODE and return eye rotation vectors.

    World-frame convention: x=right, y=up, z=forward.

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

    # ── Sensory noise ─────────────────────────────────────────────────────────
    if key is None:
        key = jax.random.PRNGKey(0)
    k_canal, k_slip, k_pos, k_vel, k_acc_n = jax.random.split(key, 5)
    k_slip_L, k_slip_R = jax.random.split(k_slip, 2)
    k_vel_L,  k_vel_R  = jax.random.split(k_vel,  2)
    k_pos_L,  k_pos_R  = jax.random.split(k_pos,  2)

    noise_canal  = jax.random.normal(k_canal,  (T, 6)) * params.sensory.sigma_canal
    noise_slip_L = jax.random.normal(k_slip_L, (T, 3)) * params.sensory.sigma_slip
    noise_slip_R = jax.random.normal(k_slip_R, (T, 3)) * params.sensory.sigma_slip
    noise_vel_L  = jax.random.normal(k_vel_L,  (T, 3)) * params.sensory.sigma_vel
    noise_vel_R  = jax.random.normal(k_vel_R,  (T, 3)) * params.sensory.sigma_vel
    # Accumulator diffusion noise: pre-scaled so that after ODE multiply-by-dt gives
    # N(0, sigma_acc·√dt) per step — standard Euler-Maruyama / Langevin scaling.
    noise_acc    = jax.random.normal(k_acc_n,  (T,))   * (params.brain.sigma_acc / jnp.sqrt(dt))

    alpha_ou = jnp.exp(-dt / params.sensory.tau_pos_drift)
    ou_drive = jnp.sqrt(1.0 - alpha_ou ** 2) * params.sensory.sigma_pos

    def _ou_step(carry, w):
        x = alpha_ou * carry + ou_drive * w
        return x, x

    _, noise_pos_L = jax.lax.scan(_ou_step, jnp.zeros(3), jax.random.normal(k_pos_L, (T, 3)))
    _, noise_pos_R = jax.lax.scan(_ou_step, jnp.zeros(3), jax.random.normal(k_pos_R, (T, 3)))

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
        tgt_p  = _prepend(tgt_p);  tgt_dv  = _prepend(tgt_dv)
        sg_L = _prepend(sg_L[:, None])[:, 0]; sg_R = _prepend(sg_R[:, None])[:, 0]
        tg_L = _prepend(tg_L[:, None])[:, 0]; tg_R = _prepend(tg_R[:, None])[:, 0]
        ts   = _prepend(ts[:, None])[:, 0]

        _z6 = jnp.zeros((warmup_T, 6))
        _z3 = jnp.zeros((warmup_T, 3))
        noise_canal  = jnp.concatenate([_z6, noise_canal],  axis=0)
        noise_slip_L = jnp.concatenate([_z3, noise_slip_L], axis=0)
        noise_slip_R = jnp.concatenate([_z3, noise_slip_R], axis=0)
        noise_pos_L  = jnp.concatenate([_z3, noise_pos_L],  axis=0)
        noise_pos_R  = jnp.concatenate([_z3, noise_pos_R],  axis=0)
        noise_vel_L  = jnp.concatenate([_z3, noise_vel_L],  axis=0)
        noise_vel_R  = jnp.concatenate([_z3, noise_vel_R],  axis=0)
        noise_acc    = jnp.concatenate([jnp.zeros(warmup_T), noise_acc])
    else:
        t_full   = t_array
        warmup_T = 0

    # ── Build interpolants ────────────────────────────────────────────────────
    def _interp(ys):
        return diffrax.LinearInterpolation(ts=t_full, ys=ys)

    head_q_interp   = _interp(head_q);  head_w_interp = _interp(head_w)
    head_x_interp   = _interp(head_x);  head_v_interp = _interp(head_v)
    head_a_interp   = _interp(head_a)
    scene_q_interp  = _interp(scene_q); scene_w_interp = _interp(scene_w)
    scene_x_interp  = _interp(scene_x); scene_v_interp = _interp(scene_v)
    target_p_interp = _interp(tgt_p);   target_dv_interp = _interp(tgt_dv)
    sp_L_interp     = _interp(sg_L);    sp_R_interp  = _interp(sg_R)
    tp_L_interp     = _interp(tg_L);    tp_R_interp  = _interp(tg_R)
    ts_interp       = _interp(ts)
    noise_canal_interp  = _interp(noise_canal)
    noise_slip_L_interp = _interp(noise_slip_L);  noise_slip_R_interp = _interp(noise_slip_R)
    noise_pos_L_interp  = _interp(noise_pos_L);   noise_pos_R_interp  = _interp(noise_pos_R)
    noise_vel_L_interp  = _interp(noise_vel_L);   noise_vel_R_interp  = _interp(noise_vel_R)
    noise_acc_interp    = _interp(noise_acc)

    # ── Initial state ─────────────────────────────────────────────────────────
    sensory_x0 = jnp.zeros(sensory_model.N_STATES)
    sensory_x0 = sensory_x0.at[_IDX_OTO].set(_otolith.X0)

    brain_x0 = brain_model.make_x0(params.brain)

    x0 = SimState(
        sensory = sensory_x0,
        brain   = brain_x0,
        plant   = jnp.zeros(plant_model.N_STATES),
    )

    # ── Solve ─────────────────────────────────────────────────────────────────
    total_steps = int(jnp.ceil((t_full[-1] - t_full[0]) / dt).item()) + 100
    max_steps   = max(max_steps, total_steps)

    ode_args = (
        params,
        head_q_interp, head_w_interp, head_x_interp, head_v_interp, head_a_interp,
        scene_q_interp, scene_w_interp, scene_x_interp, scene_v_interp,
        target_p_interp, target_dv_interp,
        sp_L_interp, sp_R_interp, tp_L_interp, tp_R_interp, ts_interp,
        noise_canal_interp, noise_slip_L_interp, noise_slip_R_interp,
        noise_pos_L_interp, noise_pos_R_interp,
        noise_vel_L_interp, noise_vel_R_interp,
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

    ys = SimState(
        sensory = solution.ys.sensory[warmup_T:],
        brain   = solution.ys.brain[warmup_T:],
        plant   = solution.ys.plant[warmup_T:],
    )

    if return_states:
        return ys
    return ys.plant   # (T, 6)
