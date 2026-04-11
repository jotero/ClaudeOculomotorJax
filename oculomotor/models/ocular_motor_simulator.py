"""Full oculomotor simulator: Canal Array → VS → NI → Plant, with OKR and Saccades.

Signal flow (3-D):

    VOR pathway:
        w_head (3,) → [Canal Array] → y_canals (6,)

    Visual delay:
        e_slip (3,)  = scene_gain · (w_scene − w_head − w_eye + w_burst_pred)
        x_vis  (240) cascade state; e_slip_delayed = C_slip @ x_vis

    Velocity storage (unified canal + visual):
        VS input: [y_canals (6,) | e_slip_delayed (3,)]
        VS owns PINV_SENS mixing (canal) and visual gains K_vis / g_vis.
        w_est (3,) = VS output: velocity estimate → NI with sign flip −g_vor.
        OKAN driven by x_vs decaying with τ_eff = 1/(1/τ_vs + K_vs).

    Saccade pathway  [see models/saccade_generator.py]:
        p_target  (3,)   Cartesian target position → theta_target (deg) via atan2
        e_pos_delayed (3,) delayed retinal position error → saccade motor error
        u_burst   (3,)   saccade velocity command from local-feedback burst model

    Downstream (VOR + OKR + Saccade combined):
        u_vel = −g_vor · w_est + u_burst        combined eye-velocity command
        u_vel → [NI] → x_ni (3,)               position command
        u_p  = C·x_ni + τ_p·u_vel              pulse-step to plant
        u_p  → [Plant] → q (3,)                eye rotation vector (deg)

Global state vector (276 states):
    x = [x_c (12) | x_vs (3) | x_ni (3) | x_p (3) | x_vis (240) | x_sg (9) | x_ec (6)]
         ─ canal ─   ─ VS ─   ─  NI  ─   plant      vis delay      sacc gen   efference copy
    x_sg = [x_copy(3) | z_ref(1) | e_held(3) | z_sac(1) | z_acc(1)]
    x_ec = [x_ni_pc(3) | x_pc(3)]
                                           (120 slip + 120 pos error, 40 stages × 3 axes × 2 sig)

    x_pc    — efference-copy plant position (burst-driven): follows x_ni_pc with tau_p lag.
    x_ni_pc — efference-copy NI (burst-driven): integrates u_burst with tau_i leak.

    Together they mirror the full NI+plant response to u_burst alone, so that
        w_burst_pred = (x_ni_pc − x_pc)/tau_p + u_burst
    exactly equals dx_p_burst, giving perfect burst cancellation in e_slip.

Head velocity input:
    Accepts 1-D (T,) horizontal-only array — padded to (T, 3) internally.
    Accepts 3-D (T, 3) array directly for full 3-D stimulation.
    Accepts a Stimulus object (oculomotor.sim.stimulus) for full 6-DOF + visual input.

Scene angular velocity input:
    Accepts (T, 3) array or None (dark = zeros).
    Or pass a Stimulus object that carries w_scene.

Target position input:
    Accepts (T, 3) Cartesian array or None (saccades disabled, g_burst must be 0).
    Pass a fixed (3,) array for a constant target (replicated across time).

Output:
    simulate() returns eye rotation vector, shape (T, 3), or the full state
    trajectory, shape (T, 270), when return_states=True.
    Use readout module to convert to Fick angles, Listing deviation, etc.
"""

import jax.numpy as jnp
import diffrax

from oculomotor.models import canal
from oculomotor.models import velocity_storage as vs
from oculomotor.models import neural_integrator as ni
from oculomotor.models import plant
from oculomotor.models import retina
from oculomotor.models import visual_delay
from oculomotor.models import saccade_generator as sg
from oculomotor.models import efference_copy as ec
from oculomotor.models import target_selector as ts

# ── Canonical model parameters ─────────────────────────────────────────────────

THETA_DEFAULT = {
    # ── Semicircular canals ───────────────────────────────────────────────────
    # Torsion-pendulum model: Steinhausen (1931); Fernandez & Goldberg (1971)
    'tau_c':          5.0,    # cupula adaptation TC (s); HP corner ≈ 0.03 Hz
    'tau_s':          0.005,  # endolymph inertia TC (s); LP corner ≈ 32 Hz
    # Per-canal gains (6,): scale afferent output; 1 = intact, 0 = lesioned.
    # Sets effective VOR gain. Healthy primate ≈ 1.0 (Collewijn et al. 1983).
    'canal_gains':    jnp.ones(canal.N_CANALS),

    # ── Velocity storage (VS) — Raphan, Matsuo & Cohen (1979) architecture ──
    # Leaky integrator: A = −1/τ_vs  (τ_vs IS the storage TC directly).
    # K_vs controls canal coupling; independent of τ_vs (no compound formula).
    # Architecture: Raphan, Matsuo & Cohen (1979 Exp Brain Res)
    'tau_vs':        20.0,    # storage / OKAN TC (s); ~20 s in monkey
                              # Cohen, Matsuo & Raphan (1977 J Neurophysiol);
                              # Raphan, Matsuo & Cohen (1979 Exp Brain Res)
    'K_vs':           0.1,    # canal coupling gain (1/s); x_vs_ss = K_vs·τ_vs·u_canal
                              # K_vs=0.1 → VS charges to ~19 deg/s in 15-s rotation

    # ── OKR — visual pathway into VS ─────────────────────────────────────────
    # Retinal slip drives VS with two components:
    #   K_vis: state gain — charges x_vs, sustains OKAN after scene off
    #   g_vis: direct feedthrough — fast OKR onset, contributes to SS gain
    # SS OKR gain ≈ (τ_vs·K_vis + g_vis) / (1 + τ_vs·K_vis + g_vis) ≈ 86%
    # OKAN TC = τ_vs = 20 s (independent of K_vis and K_vs)
    # Data: Raphan et al. (1979); Cohen et al. (1977); Waespe & Henn (1977
    # Exp Brain Res)
    'K_vis':          0.3,    # visual state gain (1/s)
    'g_vis':          0.3,    # visual direct feedthrough (unitless)

    # ── Neural integrator (NI) ────────────────────────────────────────────────
    # Robinson (1975); leak fitted in monkey by Cannon & Robinson (1985
    # Biol Cybern)
    'tau_i':         25.0,    # leak TC (s); ~25 s in healthy brainstem

    # ── Ocular plant ──────────────────────────────────────────────────────────
    # First-order plant: Robinson (1964 IEEE Trans Biomed Eng; 1981 Ann Rev
    # Neurosci); τ_p ≈ 150 ms (Goldstein 1983 Biol Cybern)
    'tau_p':          0.15,   # plant TC (s)

    # ── Visual pathway ────────────────────────────────────────────────────────
    # OKR/SP onset latency ~80 ms (Cohen et al. 1977 J Neurophysiol;
    # Miles, Kawano & Optican 1986 J Neurophysiol)
    'tau_vis':        0.08,   # visual delay TC (s)

    # ── Saccade generator — enabled by default ───────────────────────────────
    # Ballistic local-feedback burst model: Robinson (1975); burst neurons:
    # Fuchs, Scudder & Kaneko (1988 J Neurophysiol); main sequence: Bahill,
    # Clark & Stark (1975 Math Biosci).
    # Saccades are ballistic: error is latched at onset (sample-and-hold),
    # burst runs against the held target, not live visual feedback.
    'g_burst':        700.0,  # burst ceiling (deg/s); set to 0 to disable saccades
    'threshold_sac':    0.5,  # trigger threshold (deg); dead-zone ~0.5°
                              # (Steinman et al. 1967 Science)
    'threshold_stop':   0.1,  # stopping threshold (deg); endpoint accuracy ~0.1–0.2°
                              # (Becker 1989). Against HELD residual → always clean stop.
    'k_sac':          200.0,  # sigmoid steepness (1/deg)
    'e_sat_sac':        7.0,  # main-sequence saturation (deg); Bahill et al. (1975)
    'tau_reset_fast':   0.05, # x_copy reset TC between bursts (s)
    'tau_ref':          0.15, # refractory decay TC (s); ISI ~150–200 ms
                              # (Fischer & Ramsperger 1984 Exp Brain Res)
    'tau_ref_charge':   0.001,# OPN charge TC (s); z_ref rises in ~1 ms at burst end
    'k_ref':           50.0,  # bistable OPN gate steepness (1/z_ref)
    'threshold_ref':    0.1,  # OPN threshold: z_ref < 0.1 → burst allowed, > 0.1 → blocked
                              # z_ref decays from ~0.7 to 0.1 in ~240 ms = refractory period
    'tau_hold':         0.005,# sample-and-hold tracking TC (s) between saccades (5 ms)
    'tau_sac':              0.001,# saccade latch TC (s): z_sac rise/fall time (1 ms)
    'threshold_sac_release': 0.4, # z_ref level that releases z_sac (must be >> threshold_ref)
                                  # In the continuous-burst fixed point, z_ref only reaches
                                  # ~0.08–0.10, so z_sac stays locked (e_held frozen) until
                                  # the burst truly stops and z_ref charges past 0.4.
    # Rise-to-bound accumulator (z_acc) — delays z_sac to let visual cascade settle
    'tau_acc':       0.080,  # accumulator rise TC (s): ~80 ms to fill (cascade ~99% settled)
    'tau_drain':     0.120,  # accumulator drain TC (s) when gate off or burst active
    'threshold_acc': 0.5,    # accumulator threshold to fire z_sac
    'k_acc':         50.0,   # accumulator sigmoid steepness

    # ── Orbital limits + target selector ─────────────────────────────────────
    # target_selector.select() blends visual error with orbital reset signal and
    # clips to prevent saccade commands that exceed the orbital range.
    # plant.soft_limit() saturates eye position to ±orbital_limit via tanh.
    # Ref: Goldberg & Fernandez (1971); Robinson (1975) motor range.
    'orbital_limit':  50.0,  # half-range of orbital limit (deg); typical ±50° in monkey
    'k_orbital':       0.1,  # sigmoid steepness for reset gate (1/deg)
    'alpha_reset':     1.0,  # orbital reset gain; e_reset = -alpha_reset * x_p
}

# ── State vector layout ────────────────────────────────────────────────────────

_NC      = canal.N_STATES           # 12  (x1+x2 per canal, 6 canals)
_NVS     = vs.N_STATES              #  3
_NNI     = ni.N_STATES              #  3
_NP      = plant.N_STATES           #  3
_NVis    = visual_delay.N_STATES    # 240 visual delay (120 slip + 120 pos error)
_NSG     = sg.N_STATES              #   9 saccade generator [x_copy(3)|z_ref|e_held(3)|z_sac|z_acc]
_NEC     = ec.N_STATES              #   6 efference copy [x_ni_pc(3) | x_pc(3)]
_N_TOTAL = _NC + _NVS + _NNI + _NP + _NVis + _NSG + _NEC              # 276

_IDX_C     = slice(0,                                          _NC)
_IDX_VS    = slice(_NC,                                        _NC + _NVS)
_IDX_NI    = slice(_NC + _NVS,                                 _NC + _NVS + _NNI)
_IDX_P     = slice(_NC + _NVS + _NNI,                          _NC + _NVS + _NNI + _NP)
_IDX_VIS   = slice(_NC + _NVS + _NNI + _NP,                    _NC + _NVS + _NNI + _NP + _NVis)
_IDX_SG    = slice(_NC + _NVS + _NNI + _NP + _NVis,            _NC + _NVS + _NNI + _NP + _NVis + _NSG)
_IDX_EC    = slice(_NC + _NVS + _NNI + _NP + _NVis + _NSG,     _N_TOTAL)
# Sub-slices within the efference copy block: ec module stores [x_ni_pc(3) | x_pc(3)]
_IDX_NI_PC = slice(_NC + _NVS + _NNI + _NP + _NVis + _NSG,     _NC + _NVS + _NNI + _NP + _NVis + _NSG + 3)
_IDX_PC    = slice(_NC + _NVS + _NNI + _NP + _NVis + _NSG + 3, _N_TOTAL)

_DT_SOLVE = 0.001  # Heun fixed step (s); must satisfy dt < 2*tau_stage_vis = 0.004 s


# ── ODE vector field ───────────────────────────────────────────────────────────

def ODE_ocular_motor(t, x, args):
    """ODE right-hand side: chains all SSMs.

    Compatible with diffrax: signature f(t, x, args).

    Args:
        t:    scalar time (s)
        x:    global state vector, shape (_N_TOTAL,)
        args: (theta, hv_interp, hp_interp, vs_interp, target_interp,
               scene_gain_interp, target_gain_interp)

              scene_gain_interp  — LinearInterpolation of a (T,) scalar array,
              values in [0, 1].  0 = dark (no retinal image), 1 = full scene.
              Gates the retinal slip entering the visual delay cascade.

              target_gain_interp — LinearInterpolation of a (T,) scalar array,
              values in [0, 1].  0 = no visible target, 1 = target present.
              Gates e_pos_delayed inside target_selector before driving saccades.

    Returns:
        dx_dt: shape (_N_TOTAL,)
    """
    theta, hv_interp, hp_interp, vs_interp, target_interp, scene_gain_interp, target_gain_interp = args

    x_c     = x[_IDX_C]
    x_vs    = x[_IDX_VS]
    x_ni    = x[_IDX_NI]
    x_p     = x[_IDX_P]
    x_vis   = x[_IDX_VIS]
    x_sg    = x[_IDX_SG]
    x_ec    = x[_IDX_EC]

    w_head      = hv_interp.evaluate(t)           # (3,) head angular velocity (deg/s)
    q_head      = hp_interp.evaluate(t)           # (3,) head angular position (deg)
    w_scene     = vs_interp.evaluate(t)           # (3,) scene angular velocity (deg/s)
    p_target    = target_interp.evaluate(t)       # (3,) Cartesian target position
    scene_gain  = scene_gain_interp.evaluate(t)   # scalar: 0=dark, 1=full scene
    target_gain = target_gain_interp.evaluate(t)  # scalar: 0=no target, 1=target present

    # ── Read delayed signals from visual delay state ─────────────────────────
    e_slip_delayed = visual_delay.C_slip @ x_vis   # (3,) delayed retinal slip
    e_pos_delayed  = visual_delay.C_pos  @ x_vis   # (3,) delayed position error

    # ── Vestibular sensing ────────────────────────────────────────────────────
    dx_c, y_canals = canal.step(x_c, w_head, theta)

    # ── Velocity storage: unified canal + visual input ────────────────────────
    # VS owns PINV_SENS mixing and visual gains (K_vis, g_vis).
    # w_est is the velocity estimate: positive = head/scene moving rightward.
    dx_vs, w_est = vs.step(x_vs, jnp.concatenate([y_canals, e_slip_delayed]), theta)

    # ── Target selector: blend visual error + orbital reset, anti-windup clip ──
    # Produces e_cmd for the SG.  Within ±orbital_limit → pure visual mode.
    # At/past limit → smooth transition to orbital reset (drives eye back to center).
    e_cmd = ts.select(e_pos_delayed, x_p, theta, target_gain)

    # ── Saccade generator (motor error command → Robinson local feedback) ─────
    dx_sg, u_burst = sg.step(x_sg, e_cmd, theta)

    # ── Neural integrator + plant ─────────────────────────────────────────────
    # VOR: sign flip (eyes oppose head). Gain lives in canal_gains (theta).
    # Burst already in eye coordinates — unit gain, no sign flip needed.
    dx_ni, u_p = ni.step(x_ni, -w_est + u_burst, theta)
    dx_p,  _   = plant.step(x_p, u_p, theta)

    # ── Efference copy: NI+plant copy driven by burst only ───────────────────
    dx_ec, w_burst_pred = ec.step(x_ec, u_burst, theta)

    # ── Retinal signals (position error + velocity slip) → visual delay ──────
    e_pos  = retina.target_to_angle(p_target) - q_head - x_p          # position error (deg)
    e_slip = scene_gain * (w_scene - w_head - dx_p + w_burst_pred)    # gated by scene: 0 in dark
    dx_vis, _, _ = visual_delay.step(x_vis, e_slip, e_pos, theta)

    return jnp.concatenate([dx_c, dx_vs, dx_ni, dx_p, dx_vis, dx_sg, dx_ec])


# ── Simulation entry point ─────────────────────────────────────────────────────

def simulate(theta, t_array_or_stimulus, head_vel_array=None,
             v_scene_array=None,
             p_target_array=None,
             scene_present_array=None,
             target_present_array=None,
             max_steps=10000, dt_solve=None,
             return_states=False):
    """Integrate the oculomotor ODE and return eye rotation vector.

    Args:
        theta:                dict with keys tau_c, tau_s, canal_gains, tau_i, tau_p,
                              tau_vs, K_vs, K_vis, g_vis, tau_vis, and optionally
                              g_burst, threshold_sac, k_sac, e_sat_sac, tau_reset_*.
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
                              Pass explicitly to simulate a stationary lit scene
                              (v_scene=zeros) or a scene that turns on/off mid-trial.
        target_present_array: target visibility gain, shape (T,), values in [0, 1].
                              None (default) = 1.0 (target always present).
                              Pass zeros for dark conditions where no target is visible
                              — gates e_pos_delayed inside target_selector so no
                              visually-driven saccades fire in the dark.
        max_steps:            ODE solver step budget (≥ duration / dt_solve).
        dt_solve:             Heun fixed step size (s). Default: _DT_SOLVE (0.001).
        return_states:        if True, return full state trajectory (T, 270) instead
                              of just eye rotation (T, 3).  Use _IDX_* slices to
                              extract individual subsystem states.

    Returns:
        If return_states=False (default):
            eye_rot: eye rotation vector (deg), shape (T, 3)
                     columns: [yaw, pitch, roll]
                     Use oculomotor.models.readout for Fick/Helmholtz/etc.
        If return_states=True:
            states: full global state trajectory, shape (T, 270)
                    Use _IDX_C, _IDX_VS, _IDX_NI, _IDX_P, _IDX_VIS,
                    _IDX_SG, _IDX_EC to slice subsystem states.
    """
    dt = _DT_SOLVE if dt_solve is None else dt_solve

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

    # ── Scene presence gain (time-varying, [0,1]) ─────────────────────────────
    # None → infer: 1 if v_scene_array was provided, 0 (dark) if None.
    # Pass explicitly for stationary-world or mid-trial light-on/off scenarios.
    if scene_present_array is not None:
        sg1 = jnp.asarray(scene_present_array, dtype=jnp.float32)
    elif v_scene_array is not None:
        sg1 = jnp.ones(T, dtype=jnp.float32)
    else:
        sg1 = jnp.zeros(T, dtype=jnp.float32)   # dark

    # ── Target presence gain (time-varying, [0,1]) ────────────────────────────
    # None → infer: 1 if p_target_array was provided explicitly, else 1 by default.
    # Pass target_present_array=zeros to suppress saccades in dark (no visible target).
    if target_present_array is not None:
        tg1 = jnp.asarray(target_present_array, dtype=jnp.float32)
    else:
        tg1 = jnp.ones(T, dtype=jnp.float32)    # target present by default

    hv_interp          = diffrax.LinearInterpolation(ts=t_array, ys=hv3)
    hp_interp          = diffrax.LinearInterpolation(ts=t_array, ys=hp3)
    vs_interp          = diffrax.LinearInterpolation(ts=t_array, ys=vs3)
    target_interp      = diffrax.LinearInterpolation(ts=t_array, ys=pt3)
    scene_gain_interp  = diffrax.LinearInterpolation(ts=t_array, ys=sg1)
    target_gain_interp = diffrax.LinearInterpolation(ts=t_array, ys=tg1)
    x0                 = jnp.zeros(_N_TOTAL)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(ODE_ocular_motor),
        diffrax.Heun(),
        t0=t_array[0],
        t1=t_array[-1],
        dt0=dt,
        y0=x0,
        args=(theta, hv_interp, hp_interp, vs_interp, target_interp,
              scene_gain_interp, target_gain_interp),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=t_array),
        max_steps=max_steps,
    )

    if return_states:
        return solution.ys           # (T, 270) full state trajectory
    return solution.ys[:, _IDX_P]   # (T, 3) eye rotation vector
