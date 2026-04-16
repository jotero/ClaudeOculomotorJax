"""Full oculomotor simulator — wires sensory_model, brain_model, and plant_model.

Signal flow (3-D):

    VOR pathway:
        w_head (3,) → [Sensory: Canal Array] → y_canals (6,)

    Visual delay (inside sensory_model):
        e_slip (3,)  = scene_present · (w_scene − w_head − dx_p + u_burst)
        e_slip → [Sensory: Visual delay, sig 0] → e_slip_delayed  (for VS / OKR)
        e_pos  → [Sensory: Visual delay, sig 1] → e_pos_delayed   (for SG)

    Brain model (VS + NI + SG + EC):
        [y_canals | e_slip_delayed] → VS  → w_est
        e_pos_delayed → target selector → SG → u_burst
        u_vel = −w_est + u_burst → NI → motor_cmd
        u_burst → efference copy (slip cancellation)

    Plant model:
        motor_cmd → Plant → q_eye (eye rotation vector)

State structure — SimState NamedTuple with three groups:

    sensory  (252):  [x_c (12) | x_vis (240)]
                      canal       retinal-delay cascade
                      _IDX_C      _IDX_VIS          (indices into sensory)

    brain    (135):  [x_vs (3) | x_ni (3) | x_sg (9) | x_ec (120)]
                      vel-store   NI          sacc-gen   EC delay cascade
                      _IDX_VS     _IDX_NI     _IDX_SG    _IDX_EC          (indices into brain)

    plant      (3):  x_p — eye rotation vector (deg); directly observable

    x_sg sub-layout: [x_copy(3) | z_ref(1) | e_held(3) | z_sac(1) | z_acc(1)]

Head velocity input:
    Accepts 1-D (T,) horizontal-only array — padded to (T, 3) internally.
    Accepts 3-D (T, 3) array directly for full 3-D stimulation.
    Accepts a Stimulus object (oculomotor.sim.stimuli) for full 6-DOF + visual input.

Output:
    simulate() returns eye rotation vector, shape (T, 3), or a SimState trajectory
    when return_states=True.  Access subsystem states via:
        states.plant                → (T, 3)  eye rotation
        states.brain[:, _IDX_VS]   → (T, 3)  velocity storage
        states.sensory[:, _IDX_C]  → (T, 12) canal states
        etc.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax

from oculomotor.models.sensory_models.sensory_model import _IDX_C, _IDX_OTO, _IDX_VIS, SensoryParams
from oculomotor.models.sensory_models               import otolith as _otolith
from oculomotor.models.brain_models.brain_model    import _IDX_VS, _IDX_NI, _IDX_SG, _IDX_EC, _IDX_GRAV, _IDX_PURSUIT, BrainParams
from oculomotor.models.plant_models.plant_model_first_order import PlantParams
from oculomotor.models.sensory_models import sensory_model
from oculomotor.models.brain_models   import brain_model
from oculomotor.models.plant_models   import plant_model_first_order as plant_model


# ── Simulation config ───────────────────────────────────────────────────────────

class SimConfig(NamedTuple):
    """Solver / run settings — not model parameters, not learnable.

    dt_solve: Heun fixed step (s).  Must satisfy dt < 2 * tau_stage_vis.
              With N_STAGES=40 and tau_vis=0.08 s → tau_stage = 0.002 s
              → dt_max = 0.004 s.  Default 0.001 s gives 4× safety margin.
    """
    dt_solve: float = 0.001


# ── Top-level parameter container ──────────────────────────────────────────────

class Params(NamedTuple):
    """Top-level parameter container — a JAX pytree.

    Each field is a NamedTuple defined in the corresponding model module.
    This separation allows swapping one component (e.g. plant) without
    touching the others, and keeps parameter documentation co-located
    with the code that uses it.

    jax.tree_util.tree_leaves(params) returns all float/array leaves from
    all three sub-containers.  For partial-parameter optimisation, pass
    only params.brain to optax.
    """
    sensory: SensoryParams = SensoryParams()
    plant:   PlantParams   = PlantParams()
    brain:   BrainParams   = BrainParams()


def default_params() -> Params:
    """Healthy primate default parameters."""
    return Params()


def with_brain(params: Params, **kwargs) -> Params:
    """Return a new Params with brain fields updated.

    Example:
        p = with_brain(default_params(), g_burst=0.0)  # disable saccades
    """
    return params._replace(brain=params.brain._replace(**kwargs))


def with_sensory(params: Params, **kwargs) -> Params:
    """Return a new Params with sensory fields updated.

    Example:
        p = with_sensory(default_params(), canal_gains=jnp.array([0,0,1,1,1,1.]))
    """
    return params._replace(sensory=params.sensory._replace(**kwargs))


def with_plant(params: Params, **kwargs) -> Params:
    """Return a new Params with plant fields updated.

    Example:
        p = with_plant(default_params(), tau_p=0.20)
    """
    return params._replace(plant=params.plant._replace(**kwargs))


SIM_CONFIG_DEFAULT = SimConfig()
PARAMS_DEFAULT     = default_params()

# Re-export params API so callers can import everything from simulator
__all__ = [
    'SimState', 'ODE_ocular_motor', 'simulate',
    '_IDX_C', '_IDX_OTO', '_IDX_VIS', '_IDX_VS', '_IDX_NI', '_IDX_SG', '_IDX_EC', '_IDX_GRAV', '_IDX_PURSUIT',
    # params
    'Params', 'SimConfig', 'SensoryParams', 'PlantParams', 'BrainParams',
    'default_params', 'with_brain', 'with_sensory', 'with_plant',
    'PARAMS_DEFAULT', 'SIM_CONFIG_DEFAULT',
]

# ── SimState: structured state split by functional group ──────────────────────

class SimState(NamedTuple):
    """Structured ODE state — a JAX-compatible pytree (NamedTuple).

    Groups:
        sensory  (378)  Canal + otolith + retinal delay cascade (slip, pos, vel).
        brain    (141)  Central computation: VS, NI, SG, EC, gravity, pursuit.
        plant      (3)  Extraocular plant — eye rotation vector (deg)
    """
    sensory: jnp.ndarray   # (418,)  [x_c (12) | x_oto (6) | x_vis (400)]
    brain:   jnp.ndarray   # (141,)  [x_vs (3) | x_ni (3) | x_sg (9) | x_ec (120) | x_grav (3) | x_pursuit (3)]
    plant:   jnp.ndarray   #   (3,)  eye rotation vector (deg)




# ── ODE vector field ───────────────────────────────────────────────────────────

def ODE_ocular_motor(t, state, args):
    """ODE right-hand side: chains sensory, brain, and plant models.

    Compatible with diffrax: signature f(t, state, args).

    Evaluation order:
        1. read_outputs  — slice delayed signals from sensory state for brain
        2. brain_model   — VS + NI + SG + EC → motor_cmd
        3. plant_model   — motor_cmd → dx_plant;  dx_plant[_IDX_Q] = w_true (deg/s)
        4. sensory_model — canal + visual delay cascade driven by w_true

    sensory_model.step must follow plant_model.step because it requires
    w_eye = dx_plant[_IDX_Q] (instantaneous eye velocity, no lag).  Using the
    tracked velocity in state.plant[_IDX_W] instead would introduce a ~5 ms
    lag that breaks efference-copy cancellation and causes a post-saccade blip.

    Args:
        t:     scalar time (s)
        state: SimState pytree with fields (sensory, brain, plant)
        args:  (theta, hv_interp, hp_interp, ha_interp, vs_interp, target_interp,
                vt_interp, scene_present_interp, target_present_interp)

    Returns:
        SimState of derivatives (dsensory, dbrain, dplant)
    """
    (theta, hv_interp, hp_interp, ha_interp, vs_interp, target_interp,
     vt_interp, scene_present_interp, target_present_interp,
     noise_canal_interp, noise_slip_interp, noise_pos_interp,
     noise_vel_interp) = args

    # ── External inputs at time t ────────────────────────────────────────────
    w_head        = hv_interp.evaluate(t)              # (3,) head angular velocity (deg/s)
    q_head        = hp_interp.evaluate(t)              # (3,) head angular position (deg)
    a_head        = ha_interp.evaluate(t)              # (3,) head linear acceleration (m/s²)
    w_scene       = vs_interp.evaluate(t)              # (3,) scene angular velocity (deg/s)
    p_target      = target_interp.evaluate(t)          # (3,) Cartesian target position
    v_target      = vt_interp.evaluate(t)              # (3,) target angular velocity (deg/s)
    scene_present  = scene_present_interp.evaluate(t)         # scalar: 0=dark, 1=lit
    target_present = target_present_interp.evaluate(t)        # scalar: 0=no target, 1=present

    # ── Unpack plant state ────────────────────────────────────────────────────
    q_eye = state.plant   # (3,) eye rotation vector (deg)

    # ── Sensory: read bundled outputs for brain ───────────────────────────────
    # f_gia from otolith LP state; gate_vf from |pos_delayed|; target selection in SG
    sensory_out = sensory_model.read_outputs(
        state.sensory, scene_present, target_present, theta.sensory)

    # ── Sensory noise ─────────────────────────────────────────────────────────
    sensory_out = sensory_out._replace(
        canal        = sensory_out.canal        + noise_canal_interp.evaluate(t),
        slip_delayed = sensory_out.slip_delayed + noise_slip_interp.evaluate(t),
        pos_delayed  = sensory_out.pos_delayed  + noise_pos_interp.evaluate(t),
        vel_delayed  = sensory_out.vel_delayed  + noise_vel_interp.evaluate(t),
    )

    # ── Brain: VS + NI + SG + EC ──────────────────────────────────────────────
    dx_brain, motor_cmd = brain_model.step(state.brain, sensory_out, theta.brain)

    # ── Plant ─────────────────────────────────────────────────────────────────
    dx_plant, _, w_eye = plant_model.step(state.plant, motor_cmd, theta.plant)

    # ── Sensory: retinal signals + canal + otolith + visual delay ────────────
    # w_eye = dx_plant = w_true (algebraic, no lag) — must follow plant step.
    dx_sensory, _, _, _ = sensory_model.step(
        state.sensory, q_head, w_head, a_head, q_eye, w_eye,
        w_scene, v_target, p_target, scene_present, theta.sensory)

    return SimState(
        sensory = dx_sensory,
        brain   = dx_brain,
        plant   = dx_plant,
    )


# ── Simulation entry point ─────────────────────────────────────────────────────

def simulate(params, t_array_or_stimulus, head_vel_array=None,
             head_accel_array=None,
             v_scene_array=None,
             p_target_array=None,
             v_target_array=None,
             scene_present_array=None,
             target_present_array=None,
             max_steps=10000, sim_config=None,
             return_states=False,
             key=None):
    """Integrate the oculomotor ODE and return eye rotation vector.

    Args:
        params:               Params — model parameters (see default_params()).
        t_array_or_stimulus:  1-D time array (s), shape (T,)  — OR —
                              a Stimulus object (oculomotor.sim.stimuli).
        head_vel_array:       head angular velocity, shape (T,) or (T, 3).
                              If 1-D, treated as horizontal (yaw); pitch/roll = 0.
        head_accel_array:     head linear acceleration, shape (T, 3) or None (m/s²).
                              None (default) = zeros (no translational motion).
        v_scene_array:        visual scene angular velocity, shape (T, 3) or None.
                              None (default) = dark — also sets scene_present=0
                              unless scene_present_array is given explicitly.
        p_target_array:       Cartesian target position, shape (T, 3) or (3,) or None.
                              None = straight ahead [0, 0, 1].
                              Shape (3,) = constant target (replicated over time).
        v_target_array:       target angular velocity in world frame, shape (T, 3) or (T,) or None.
                              None (default) = stationary target (no pursuit drive).
                              Shape (T,) = horizontal-only velocity (yaw); pitch/roll = 0.
        scene_present_array:  scene visibility gain, shape (T,), values in [0, 1].
                              None (default) = inferred: 1.0 if v_scene_array was
                              provided, 0.0 (dark) if v_scene_array is None.
        target_present_array: target visibility gain, shape (T,), values in [0, 1].
                              None (default) = 1.0 (target always present).
        max_steps:            ODE solver step budget (≥ duration / dt_solve).
        sim_config:           SimConfig — solver settings. Default: SIM_CONFIG_DEFAULT.
        return_states:        if True, return full state trajectory as a SimState
                              instead of just eye rotation (T, 3).

    Returns:
        If return_states=False (default):
            eye_rot: eye rotation vector (deg), shape (T, 3)
        If return_states=True:
            states: SimState pytree, each field has shape (T, N):
                      states.plant                      → (T, 3)   eye rotation vector
                      states.brain[:, _IDX_VS]         → (T, 3)   velocity storage
                      states.brain[:, _IDX_NI]         → (T, 3)   neural integrator
                      states.brain[:, _IDX_SG]         → (T, 9)   saccade generator
                      states.brain[:, _IDX_EC]         → (T, 120) efference copy cascade
                      states.brain[:, _IDX_GRAV]       → (T, 3)   gravity estimate (m/s²)
                      states.brain[:, _IDX_PURSUIT]    → (T, 3)   pursuit velocity memory
                      states.sensory[:, _IDX_C]        → (T, 12)  canal states
                      states.sensory[:, _IDX_OTO]      → (T, 6)   otolith LP states
                      states.sensory[:, _IDX_VIS]      → (T, 400) visual delay cascade
    """
    cfg = sim_config if sim_config is not None else SIM_CONFIG_DEFAULT
    dt  = cfg.dt_solve

    # ── Accept Stimulus object ────────────────────────────────────────────────
    if hasattr(t_array_or_stimulus, 'omega') and hasattr(t_array_or_stimulus, 't'):
        stim           = t_array_or_stimulus
        t_array        = stim.t
        head_vel_array = stim.omega
        v_scene_array  = stim.v_scene
    else:
        t_array = t_array_or_stimulus

    T = len(t_array)

    if head_vel_array is None:
        head_vel_array = jnp.zeros(T)

    # ── Head linear acceleration (zeros if not provided) ─────────────────────
    if head_accel_array is None:
        ha3 = jnp.zeros((T, 3))
    else:
        ha3 = jnp.asarray(head_accel_array)

    # ── Pad 1-D head velocity to 3-D ─────────────────────────────────────────
    if jnp.ndim(head_vel_array) == 1:
        hv3 = jnp.stack([head_vel_array,
                          jnp.zeros_like(head_vel_array),
                          jnp.zeros_like(head_vel_array)], axis=1)
    else:
        hv3 = head_vel_array

    # ── Visual scene velocity (zeros if dark) ─────────────────────────────────
    if v_scene_array is None:
        vs3 = jnp.zeros((T, 3))
    elif jnp.ndim(v_scene_array) == 1:
        vs3 = jnp.stack([v_scene_array,
                          jnp.zeros_like(v_scene_array),
                          jnp.zeros_like(v_scene_array)], axis=1)
    else:
        vs3 = jnp.asarray(v_scene_array)

    # ── Target position (straight ahead = [0, 0, 1] if not specified) ─────────
    if p_target_array is None:
        pt3 = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (T, 1))
    elif jnp.ndim(jnp.asarray(p_target_array)) == 1:
        pt3 = jnp.tile(jnp.asarray(p_target_array), (T, 1))
    else:
        pt3 = jnp.asarray(p_target_array)

    # ── Target angular velocity (zeros if stationary) ────────────────────────
    if v_target_array is None:
        vt3 = jnp.zeros((T, 3))
    elif jnp.ndim(jnp.asarray(v_target_array)) == 1:
        vt3 = jnp.stack([jnp.asarray(v_target_array),
                          jnp.zeros_like(jnp.asarray(v_target_array)),
                          jnp.zeros_like(jnp.asarray(v_target_array))], axis=1)
    else:
        vt3 = jnp.asarray(v_target_array)

    # ── Head position: trapezoidal integral of head velocity ──────────────────
    dt_arr = jnp.diff(t_array)                                     # (T-1,)
    hp3    = jnp.concatenate([
        jnp.zeros((1, 3)),
        jnp.cumsum(0.5 * (hv3[:-1] + hv3[1:]) * dt_arr[:, None], axis=0),
    ])                                                              # (T, 3)

    # ── Scene presence gain ───────────────────────────────────────────────────
    if scene_present_array is not None:
        sg1 = jnp.asarray(scene_present_array, dtype=jnp.float32)
    elif v_scene_array is not None:
        sg1 = jnp.ones(T, dtype=jnp.float32)
    else:
        sg1 = jnp.zeros(T, dtype=jnp.float32)   # dark

    # ── Target presence gain ──────────────────────────────────────────────────
    if target_present_array is not None:
        tg1 = jnp.asarray(target_present_array, dtype=jnp.float32)
    else:
        tg1 = jnp.ones(T, dtype=jnp.float32)

    # ── Sensory noise arrays (pre-generated; zero when sigma=0) ──────────────
    if key is None:
        key = jax.random.PRNGKey(0)
    k_canal, k_slip, k_pos, k_vel = jax.random.split(key, 4)

    noise_canal = jax.random.normal(k_canal, (T, 6)) * params.sensory.sigma_canal  # (T, 6) deg/s
    noise_slip  = jax.random.normal(k_slip,  (T, 3)) * params.sensory.sigma_slip   # (T, 3) deg/s
    noise_pos   = jax.random.normal(k_pos,   (T, 3)) * params.sensory.sigma_pos    # (T, 3) deg
    noise_vel   = jax.random.normal(k_vel,   (T, 3)) * params.sensory.sigma_vel    # (T, 3) deg/s

    noise_canal_interp = diffrax.LinearInterpolation(ts=t_array, ys=noise_canal)
    noise_slip_interp  = diffrax.LinearInterpolation(ts=t_array, ys=noise_slip)
    noise_pos_interp   = diffrax.LinearInterpolation(ts=t_array, ys=noise_pos)
    noise_vel_interp   = diffrax.LinearInterpolation(ts=t_array, ys=noise_vel)

    hv_interp             = diffrax.LinearInterpolation(ts=t_array, ys=hv3)
    hp_interp             = diffrax.LinearInterpolation(ts=t_array, ys=hp3)
    ha_interp             = diffrax.LinearInterpolation(ts=t_array, ys=ha3)
    vs_interp             = diffrax.LinearInterpolation(ts=t_array, ys=vs3)
    target_interp         = diffrax.LinearInterpolation(ts=t_array, ys=pt3)
    vt_interp             = diffrax.LinearInterpolation(ts=t_array, ys=vt3)
    scene_present_interp  = diffrax.LinearInterpolation(ts=t_array, ys=sg1)
    target_present_interp = diffrax.LinearInterpolation(ts=t_array, ys=tg1)

    sensory_x0 = jnp.zeros(sensory_model.N_STATES)
    sensory_x0 = sensory_x0.at[_IDX_OTO].set(_otolith.X0)   # otolith settled at gravity

    x0 = SimState(
        sensory = sensory_x0,                          # (418,) otolith init at [9.81,0,0, ...]
        brain   = brain_model.make_x0(),               # (141,) gravity init at [9.81,0,0]
        plant   = jnp.zeros(plant_model.N_STATES),     #   (3,)  eye rotation vector
    )

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(ODE_ocular_motor),
        diffrax.Heun(),
        t0=t_array[0],
        t1=t_array[-1],
        dt0=dt,
        y0=x0,
        args=(params, hv_interp, hp_interp, ha_interp, vs_interp, target_interp,
              vt_interp, scene_present_interp, target_present_interp,
              noise_canal_interp, noise_slip_interp, noise_pos_interp,
              noise_vel_interp),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=t_array),
        max_steps=max_steps,
    )

    if return_states:
        return solution.ys                          # SimState, each field (T, N)
    return solution.ys.plant                        # (T, 3) eye rotation vector
