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
        u_vel = −w_est + u_burst → NI → u_p
        u_burst → efference copy (slip cancellation)

    Plant model:
        u_p → Plant → q_eye (eye rotation vector)

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
    Accepts a Stimulus object (oculomotor.sim.stimulus) for full 6-DOF + visual input.

Output:
    simulate() returns eye rotation vector, shape (T, 3), or a SimState trajectory
    when return_states=True.  Access subsystem states via:
        states.plant                → (T, 3)  eye rotation
        states.brain[:, _IDX_VS]   → (T, 3)  velocity storage
        states.sensory[:, _IDX_C]  → (T, 12) canal states
        etc.
"""

from typing import NamedTuple

import jax.numpy as jnp
import diffrax

from oculomotor.models import retina
from oculomotor.models import target_selector as ts
from oculomotor.models.sensory_model import _IDX_C, _IDX_VIS
from oculomotor.models.brain_model   import _IDX_VS, _IDX_NI, _IDX_SG, _IDX_EC
import oculomotor.models.sensory_model as sensory_model
import oculomotor.models.brain_model   as brain_model
import oculomotor.models.plant_model_first_order as plant_model
from oculomotor.params import (
    Params, SimConfig, PhysParams, BrainParams,
    default_params, with_brain, with_phys,
    PARAMS_DEFAULT, SIM_CONFIG_DEFAULT,
)

# Re-export params API so callers can import everything from simulator
__all__ = [
    'SimState', 'ODE_ocular_motor', 'simulate',
    '_IDX_C', '_IDX_VIS', '_IDX_VS', '_IDX_NI', '_IDX_SG', '_IDX_EC',
    # params
    'Params', 'SimConfig', 'PhysParams', 'BrainParams',
    'default_params', 'with_brain', 'with_phys',
    'PARAMS_DEFAULT', 'SIM_CONFIG_DEFAULT',
]

# ── SimState: structured state split by functional group ──────────────────────

class SimState(NamedTuple):
    """Structured ODE state — a JAX-compatible pytree (NamedTuple).

    Groups:
        sensory  (252)  Canal transducers + retinal delay cascade.
        brain     (21)  Central computation: VS, NI, SG, efference copy.
        plant      (3)  Extraocular plant — directly observable as eye position.
    """
    sensory: jnp.ndarray   # (252,)  [x_c (12) | x_vis (240)]
    brain:   jnp.ndarray   #  (21,)  [x_vs (3) | x_ni (3) | x_sg (9) | x_ec (6)]
    plant:   jnp.ndarray   #   (3,)  x_p — eye rotation vector (deg)




# ── ODE vector field ───────────────────────────────────────────────────────────

def ODE_ocular_motor(t, state, args):
    """ODE right-hand side: chains sensory, brain, and plant models.

    Compatible with diffrax: signature f(t, state, args).

    Args:
        t:     scalar time (s)
        state: SimState pytree with fields (sensory, brain, plant)
        args:  (theta, hv_interp, hp_interp, vs_interp, target_interp,
                scene_present_interp, target_present_interp)

    Returns:
        SimState of derivatives (dsensory, dbrain, dplant)
    """
    theta, hv_interp, hp_interp, vs_interp, target_interp, scene_present_interp, target_present_interp = args

    # ── External inputs at time t ────────────────────────────────────────────
    w_head          = hv_interp.evaluate(t)              # (3,) head angular velocity (deg/s)
    q_head          = hp_interp.evaluate(t)              # (3,) head angular position (deg)
    w_scene         = vs_interp.evaluate(t)              # (3,) scene angular velocity (deg/s)
    p_target        = target_interp.evaluate(t)          # (3,) Cartesian target position
    scene_present   = scene_present_interp.evaluate(t)   # scalar: 0=dark, 1=scene present
    target_present  = target_present_interp.evaluate(t)  # scalar: 0=no target, 1=target present

    # ── Sensory: read bundled outputs from current state ─────────────────────
    sensory_out = sensory_model.read_outputs(state.sensory, theta)

    # ── Target selector: visually-gated position error + orbital state → motor cmd
    e_cmd = ts.select(sensory_out.pos_visible, state.plant, theta)

    # ── Brain: VS + NI + SG + EC ──────────────────────────────────────────────
    dx_brain, motor_commands, u_burst = brain_model.step(state.brain, sensory_out, e_cmd, scene_present, theta)

    # ── Plant ─────────────────────────────────────────────────────────────────
    dx_plant, _ = plant_model.step(state.plant, motor_commands, theta)

    # ── Retinal signals → visual delay cascade ────────────────────────────────
    e_pos     = retina.target_to_angle(p_target) - q_head - state.plant
    raw_slip  = scene_present * (w_scene - w_head - dx_plant)

    # ── Sensory: full step (canal derivative + visual delay derivative) ────────
    dx_sensory, _, _, _ = sensory_model.step(state.sensory, w_head, raw_slip, e_pos, theta)

    return SimState(
        sensory = dx_sensory,
        brain   = dx_brain,
        plant   = dx_plant,
    )


# ── Simulation entry point ─────────────────────────────────────────────────────

def simulate(params, t_array_or_stimulus, head_vel_array=None,
             v_scene_array=None,
             p_target_array=None,
             scene_present_array=None,
             target_present_array=None,
             max_steps=10000, sim_config=None,
             return_states=False):
    """Integrate the oculomotor ODE and return eye rotation vector.

    Args:
        params:               Params — model parameters (see default_params()).
        t_array_or_stimulus:  1-D time array (s), shape (T,)  — OR —
                              a Stimulus object (oculomotor.sim.stimulus).
        head_vel_array:       head angular velocity, shape (T,) or (T, 3).
                              If 1-D, treated as horizontal (yaw); pitch/roll = 0.
        v_scene_array:        visual scene angular velocity, shape (T, 3) or None.
                              None (default) = dark — also sets scene_present=0
                              unless scene_present_array is given explicitly.
        p_target_array:       Cartesian target position, shape (T, 3) or (3,) or None.
                              None = straight ahead [0, 0, 1].
                              Shape (3,) = constant target (replicated over time).
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
                      states.plant                → (T, 3)  eye rotation vector
                      states.brain[:, _IDX_VS]   → (T, 3)  velocity storage
                      states.brain[:, _IDX_NI]   → (T, 3)  neural integrator
                      states.brain[:, _IDX_SG]   → (T, 9)  saccade generator
                      states.brain[:, _IDX_EC]   → (T, 120) efference copy cascade
                      states.sensory[:, _IDX_C]  → (T, 12) canal states
                      states.sensory[:, _IDX_VIS]→ (T,240) visual delay cascade
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

    hv_interp          = diffrax.LinearInterpolation(ts=t_array, ys=hv3)
    hp_interp          = diffrax.LinearInterpolation(ts=t_array, ys=hp3)
    vs_interp          = diffrax.LinearInterpolation(ts=t_array, ys=vs3)
    target_interp      = diffrax.LinearInterpolation(ts=t_array, ys=pt3)
    scene_present_interp  = diffrax.LinearInterpolation(ts=t_array, ys=sg1)
    target_present_interp = diffrax.LinearInterpolation(ts=t_array, ys=tg1)

    x0 = SimState(
        sensory = jnp.zeros(sensory_model.N_STATES),   # (252,)
        brain   = jnp.zeros(brain_model.N_STATES),     # (21,)
        plant   = jnp.zeros(plant_model.N_STATES),     # (3,)
    )

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(ODE_ocular_motor),
        diffrax.Heun(),
        t0=t_array[0],
        t1=t_array[-1],
        dt0=dt,
        y0=x0,
        args=(params, hv_interp, hp_interp, vs_interp, target_interp,
              scene_present_interp, target_present_interp),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=t_array),
        max_steps=max_steps,
    )

    if return_states:
        return solution.ys                  # SimState, each field (T, N)
    return solution.ys.plant               # (T, 3) eye rotation vector
