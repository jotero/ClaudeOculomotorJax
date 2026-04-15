"""Brain model — velocity storage, neural integrator, saccade generator, efference copy,
gravity estimator, and smooth pursuit.

Aggregates all brain subsystems into a single SSM with one state vector and one
step() function.

Signal flow:
    y_canals         (6,)   canal afferents                   → VS
    raw_slip_delayed (3,)   delayed raw retinal slip           → VS (after EC)
    vel_delayed      (3,)   delayed target velocity on retina  → pursuit (Smith predictor)
    e_cmd            (3,)   motor error command                → SG

One efference copy cascade (120 states), two uses with different gates:
    motor_ec = ec.read_delayed(x_ec)          # delay(u_burst + u_pursuit)

    OKR / VS correction  — scene-gated (full scene slip):
        e_slip_corrected = slip_delayed + scene_present · motor_ec
        slip_delayed ≈ −(u_burst+u_pursuit)(t−τ)  →  corrected ≈ 0  ✓

    Pursuit Smith predictor — target-gated (foveal target slip only):
        e_combined = target_present · (vel_delayed + motor_ec)   ≈ v_target when target on
        Full signal gated by target_present → zero drive during VOR / OKN / fixation
        e_vel_pred = (e_combined − x_pursuit) / (1 + K_phasic)
        → at onset:        ~45 % of v_target drives integrator  (less oscillation)
        → at steady state: e_vel_pred → 0  (integrator at rest, u_pursuit ≈ v_target)
        u_pursuit = x_pursuit + K_phasic · e_vel_pred

EC advance (end of step):
    dx_ec = ec.step(x_ec, u_burst + u_pursuit)   # combined motor command

Internal flow:
    VS  →  w_est  →  −w_est + u_burst + u_pursuit  →  NI  →  motor_cmd
    SG  →  u_burst    (saccade burst → EC cascade)
    Pursuit → u_pursuit  (→ NI + EC cascade)
    EC  →  delays (u_burst + u_pursuit) by tau_vis
           read used for: VS (scene-gated) and pursuit (target-gated)
    GE  →  g_hat (gravity estimate, cross-product dynamics)

State vector  x_brain = [x_vs (3) | x_ni (3) | x_sg (9) | x_ec (120) | x_grav (3) | x_pursuit (3)]
N_STATES = 141

Index constants (relative to x_brain):
    _IDX_VS      — velocity storage states   (3,)
    _IDX_NI      — neural integrator states  (3,)
    _IDX_SG      — saccade generator states  (9,)
    _IDX_EC      — efference copy states     (120,)
    _IDX_GRAV    — gravity estimator states  (3,)
    _IDX_PURSUIT — pursuit velocity memory   (3,)

Outputs of step():
    dx_brain   (141,)  state derivative
    motor_cmd  (3,)    pulse-step motor command → plant
"""

from typing import NamedTuple

import jax.numpy as jnp

from oculomotor.models.brain_models import velocity_storage    as vs
from oculomotor.models.brain_models import neural_integrator   as ni
from oculomotor.models.brain_models import saccade_generator   as sg
from oculomotor.models.brain_models import efference_copy      as ec
from oculomotor.models.brain_models import gravity_estimator   as ge
from oculomotor.models.brain_models import pursuit             as pu
from oculomotor.models.sensory_models.sensory_model import SensoryOutput  # noqa: F401 (re-exported)


# ── Brain parameters ────────────────────────────────────────────────────────────

class BrainParams(NamedTuple):
    """Learnable central parameters — fit to patient eye-movement data."""

    # Velocity storage — Raphan, Matsuo & Cohen (1979)
    tau_vs:                float = 20.0   # storage / OKAN TC (s); ~20 s monkey (Cohen 1977)
    K_vs:                  float = 0.1    # canal-to-VS gain (1/s); controls charging speed
    K_vis:                 float = 1.0    # visual-to-VS gain (1/s); OKR / OKAN charging
                                          # OKN SS gain ≈ (K_vis·τ_vs + g_vis)/(1 + K_vis·τ_vs + g_vis)
                                          # K_vis=1.0 → L=20.3 → gain ≈ 0.95  (was 0.3 → 0.86)
    g_vis:                 float = 0.3    # visual feedthrough (unitless); fast OKR onset

    # Neural integrator — Robinson (1975)
    tau_i:                 float = 25.0   # leak TC (s); healthy >20 s (Cannon & Robinson 1985)
    tau_p:                 float = 0.15   # plant TC copy — NI feedthrough for lag cancellation
    tau_vis:               float = 0.08   # visual delay copy — EC delay must match retinal delay
                                          # should match PlantParams.tau_p in healthy subjects;
                                          # may differ in pathology (imperfect internal model)

    # Saccade generator — Robinson (1975) local-feedback burst model
    g_burst:               float = 700.0  # burst ceiling (deg/s); 0 disables saccades
    e_sat_sac:             float = 7.0    # main-sequence saturation (deg)
    k_sac:                 float = 200.0  # trigger sigmoid steepness (1/deg)
    threshold_sac:         float = 0.5    # retinal error trigger threshold (deg)
    threshold_stop:        float = 0.1    # burst-stop threshold (deg)
    threshold_sac_release: float = 0.4    # OPN latch release threshold
    tau_reset_fast:        float = 0.05   # inter-saccade x_copy reset TC (s)
    tau_ref:               float = 0.15   # refractory (OPN) decay TC (s); ~150 ms ISI
    tau_ref_charge:        float = 0.001  # OPN charge TC (s)
    k_ref:                 float = 50.0   # bistable OPN gate steepness (1/z_ref)
    threshold_ref:         float = 0.1    # OPN threshold
    tau_hold:              float = 0.005  # sample-and-hold tracking TC (s)
    tau_sac:               float = 0.001  # saccade latch TC (s)
    tau_acc:               float = 0.080  # accumulator rise TC (s)
    tau_drain:             float = 0.120  # accumulator drain TC (s)
    threshold_acc:         float = 0.5    # accumulator trigger threshold
    k_acc:                 float = 50.0   # accumulator sigmoid steepness

    # Orbital reset — centering saccade policy at orbital limit
    alpha_reset:           float = 1.0    # centering-saccade gain; 0=suppress, 1=active centering

    # Otolith / gravity estimation — Laurens & Angelaki (2011, 2017)
    K_grav:                float = 0.5    # otolith correction gain (1/s); TC = 1/K_grav ≈ 2 s
    K_gd:                  float = 0.0    # gravity dumping gain (1/s); 0 = disabled
    g_ocr:                 float = 0.0    # OCR gain (dimensionless); 0 = disabled until verified

    # Smooth pursuit — leaky integrator + direct feedthrough (Lisberger 1988)
    K_pursuit:             float = 4.0    # pursuit integration gain (1/s); rise TC ≈ 1/K_pursuit
    K_phasic_pursuit:      float = 5.0    # pursuit direct feedthrough (dim'less); fast onset
    tau_pursuit:           float = 40.0   # pursuit leak TC (s); ~40 s → ~97.5% gain at 1 Hz


# ── State layout ───────────────────────────────────────────────────────────────

N_STATES = vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES + ge.N_STATES + pu.N_STATES
#        = 3 + 3 + 9 + 120 + 3 + 3 = 141

# Index constants — relative to x_brain
_IDX_VS      = slice(0,             vs.N_STATES)                                        # (3,)
_IDX_NI      = slice(vs.N_STATES,   vs.N_STATES + ni.N_STATES)                         # (3,)
_IDX_SG      = slice(vs.N_STATES + ni.N_STATES,
                     vs.N_STATES + ni.N_STATES + sg.N_STATES)                           # (9,)
_IDX_EC      = slice(vs.N_STATES + ni.N_STATES + sg.N_STATES,
                     vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES)             # (120,)
_IDX_GRAV    = slice(vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES,
                     vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES + ge.N_STATES)  # (3,)
_IDX_PURSUIT = slice(vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES + ge.N_STATES,
                     N_STATES)                                                           # (3,)


def make_x0():
    """Default initial brain state — gravity estimator pointing down (upright head)."""
    x0 = jnp.zeros(N_STATES)
    return x0.at[_IDX_GRAV].set(ge.X0)


# ── Step function ──────────────────────────────────────────────────────────────

def step(x_brain, sensory_out, brain_params):
    """Single ODE step for the brain subsystem.

    Args:
        x_brain:     (141,)        brain state [x_vs | x_ni | x_sg | x_ec | x_grav | x_pursuit]
        sensory_out: SensoryOutput bundled canal afferents + delayed visual signals
                       .canal         (6,)    canal afferent rates
                       .slip_delayed  (3,)    delayed retinal slip (no EC correction yet)
                       .pos_visible   (3,)    delayed position error, gated by visual field
                       .e_cmd         (3,)    motor error command for the saccade generator
                       .vel_delayed   (3,)    delayed target velocity on retina → pursuit
                       .f_otolith     (3,)    specific force in head frame (m/s²)
                       .scene_present scalar  0=dark, 1=lit — gates EC slip correction
        brain_params: BrainParams   model parameters

    Returns:
        dx_brain:   (141,)  dx_brain/dt
        motor_cmd:  (3,)    pulse-step motor command → plant
    """
    x_vs      = x_brain[_IDX_VS]
    x_ni      = x_brain[_IDX_NI]
    x_sg      = x_brain[_IDX_SG]
    x_ec      = x_brain[_IDX_EC]
    x_grav    = x_brain[_IDX_GRAV]
    x_pursuit = x_brain[_IDX_PURSUIT]

    # ── One EC, two corrections with separate gates ───────────────────────────
    # motor_ec = delay(u_burst + u_pursuit) — one cascade, read once, used twice.
    motor_ec = ec.read_delayed(x_ec)

    # OKR / VS: scene-gated — cancels all self-generated motion in the visual scene
    #   slip_delayed ≈ −(u_burst+u_pursuit)(t−τ)  →  corrected ≈ 0 during any motion ✓
    e_slip_corrected = sensory_out.slip_delayed + sensory_out.scene_present * motor_ec

    # Pursuit: target-gated — foveal target slip only (excludes VOR, OKN, fixation)
    #   Gate the *entire* signal by target_present so that when there is no moving
    #   foveal target (saccade to stationary point, VOR, OKN) the pursuit integrator
    #   receives zero input.  EC cancellation of saccadic eye motion still works when
    #   target_present=1: vel_delayed ≈ v_target − w_eye(t−τ), motor_ec ≈ +w_eye(t−τ)
    #   → e_combined ≈ v_target ✓
    #   Smith predictor lives inside pu.step(): e_pred = (e_combined − x_p)/(1+K_ph)
    e_combined = sensory_out.target_present * (sensory_out.vel_delayed + motor_ec)
    dx_pursuit, u_pursuit = pu.step(x_pursuit, e_combined, brain_params)

    # ── Velocity storage: canal + EC-corrected scene slip + g_hat → ω̂ ─────────
    dx_vs, w_est = vs.step(
        x_vs,
        jnp.concatenate([sensory_out.canal, e_slip_corrected, x_grav]),
        brain_params)

    # ── Gravity estimator: cross-product transport + otolith correction ────────
    dx_grav, g_hat = ge.step(
        x_grav,
        jnp.concatenate([w_est, sensory_out.f_otolith]),
        brain_params)

    # ── Saccade generator ─────────────────────────────────────────────────────
    dx_sg, u_burst = sg.step(x_sg, sensory_out.e_cmd, brain_params)

    # ── OCR / somatogravic: gravity-driven eye position command ───────────────
    # g_hat = specific force (+x upright).  Tilt signals are normalised components:
    #   tilt_roll  = g_hat[1]/|g_hat| < 0 for CW head roll → eye rolls CCW (−z) ✓
    #   tilt_pitch = g_hat[2]/|g_hat| > 0 for nose-up tilt → eye pitches down (−y) ✓
    # (g_ocr = 0 by default; set to ~0.3 to enable)
    g_norm  = jnp.linalg.norm(g_hat) + 1e-9
    ocr_pos = brain_params.g_ocr * (180.0 / jnp.pi) * jnp.array([
        0.0,
        -g_hat[2] / g_norm,   # pitch: nose-up tilt → eye pitches down (−y)
         g_hat[1] / g_norm,   # roll:  CW tilt (g_hat[1]<0) → eye CCW (−z)
    ])

    # ── Neural integrator: VOR + saccades + pursuit → motor command ───────────
    dx_ni, motor_cmd_ni = ni.step(x_ni, -w_est + u_burst + u_pursuit, brain_params)

    # Add OCR position offset directly to motor command (bypasses NI leak)
    motor_cmd = motor_cmd_ni + ocr_pos

    # ── Efference copy: advance delay cascade with combined motor command ────────
    dx_ec, _ = ec.step(x_ec, u_burst + u_pursuit, brain_params)

    # ── Pack state derivative ─────────────────────────────────────────────────
    dx_brain = jnp.concatenate([dx_vs, dx_ni, dx_sg, dx_ec, dx_grav, dx_pursuit])

    return dx_brain, motor_cmd
