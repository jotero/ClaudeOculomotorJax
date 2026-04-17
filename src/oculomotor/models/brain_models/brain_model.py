"""Brain model — velocity storage, neural integrator, saccade generator, efference copy,
gravity estimator, smooth pursuit, and vergence.

Aggregates all brain subsystems into a single SSM with one state vector and one
step() function.

Signal flow:
    y_canals         (6,)   canal afferents                   → VS
    raw_slip_delayed (3,)   delayed raw retinal slip           → VS (after EC)
    vel_delayed      (3,)   delayed target velocity on retina  → pursuit (Smith predictor)
    e_cmd            (3,)   motor error command                → SG
    pos_delayed_L/R  (3,)   per-eye delayed position error     → vergence

One efference copy cascade (120 states), two uses with different gates:
    motor_ec = ec.read_delayed(x_ec)          # delay(u_burst + u_pursuit)

    OKR / VS correction  — scene-gated (full scene slip):
        e_slip_corrected = scene_visible · (slip_delayed + motor_ec)
        slip_delayed ≈ −(u_burst+u_pursuit)(t−τ)  →  corrected ≈ 0  ✓

    Pursuit Smith predictor — target-gated (foveal target slip only):
        e_combined = target_visible · (vel_delayed + motor_ec)   ≈ v_target when target on
        Full signal gated by target_visible (= gate_vf) → zero drive when no target in field
        e_vel_pred = (e_combined − x_pursuit) / (1 + K_phasic)
        → at onset:        ~45 % of v_target drives integrator  (less oscillation)
        → at steady state: e_vel_pred → 0  (integrator at rest, u_pursuit ≈ v_target)
        u_pursuit = x_pursuit + K_phasic · e_vel_pred

Vergence:
    e_disp = pos_delayed_L − pos_delayed_R   (binocular disparity, deg)
    Smith predictor identical to pursuit but position-driven:
        e_pred = (e_disp − x_verg) / (1 + K_phasic_verg)
    dx_verg = −x_verg/τ_verg + K_verg · e_pred
    u_verg  = x_verg + K_phasic_verg · e_pred
    motor_cmd_L = motor_cmd_version + ½ · u_verg   (L eye converges rightward)
    motor_cmd_R = motor_cmd_version − ½ · u_verg   (R eye converges leftward)

EC advance (end of step):
    dx_ec = ec.step(x_ec, u_burst + u_pursuit)   # version motor command only

Internal flow:
    VS  →  w_est  →  −w_est + u_burst + u_pursuit  →  NI  →  motor_cmd_version
    SG  →  u_burst    (saccade burst → EC cascade)
    Pursuit → u_pursuit  (→ NI + EC cascade)
    EC  →  delays (u_burst + u_pursuit) by tau_vis
           read used for: VS (scene-gated) and pursuit (target-gated)
    GE  →  g_hat (gravity estimate, cross-product dynamics)
    Vergence → u_verg → split ±½ to L/R motor commands

State vector  x_brain = [x_vs (3) | x_ni (3) | x_sg (9) | x_ec (120) | x_grav (3) | x_pursuit (3) | x_verg (3)]
N_STATES = 144

Index constants (relative to x_brain):
    _IDX_VS      — velocity storage states   (3,)
    _IDX_NI      — neural integrator states  (3,)
    _IDX_SG      — saccade generator states  (9,)
    _IDX_EC      — efference copy states     (120,)
    _IDX_GRAV    — gravity estimator states  (3,)
    _IDX_PURSUIT — pursuit velocity memory   (3,)
    _IDX_VERG    — vergence position memory  (3,)

Outputs of step():
    dx_brain     (144,)  state derivative
    motor_cmd_L  (3,)    pulse-step motor command → left  plant
    motor_cmd_R  (3,)    pulse-step motor command → right plant
"""

from typing import NamedTuple

import jax.numpy as jnp

from oculomotor.models.brain_models import velocity_storage    as vs
from oculomotor.models.brain_models import neural_integrator   as ni
from oculomotor.models.brain_models import saccade_generator   as sg
from oculomotor.models.brain_models import efference_copy      as ec
from oculomotor.models.brain_models import gravity_estimator   as ge
from oculomotor.models.brain_models import pursuit             as pu
from oculomotor.models.brain_models import vergence            as vg
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

    # Saccade target selection — handled inside the saccade generator
    orbital_limit:         float = 50.0   # oculomotor range half-width (deg); clip e_cmd to ±limit
    alpha_reset:           float = 1.0    # centering gain (0–1); e_center = −α·x_ni when out-of-field

    # Otolith / gravity estimation — Laurens & Angelaki (2011, 2017)
    K_grav:                float = 0.5    # otolith correction gain (1/s); TC = 1/K_grav ≈ 2 s
    K_gd:                  float = 0.0    # gravity dumping gain (1/s); 0 = disabled
    g_ocr:                 float = 0.0    # OCR gain (dimensionless); 0 = disabled until verified

    # Smooth pursuit — leaky integrator + direct feedthrough (Lisberger 1988)
    K_pursuit:             float = 4.0    # pursuit integration gain (1/s); rise TC ≈ 1/K_pursuit
    K_phasic_pursuit:      float = 5.0    # pursuit direct feedthrough (dim'less); fast onset
    tau_pursuit:           float = 40.0   # pursuit leak TC (s); ~40 s → ~97.5% gain at 1 Hz

    # Vergence — disparity-driven position integrator + Smith predictor (Patel et al. 1997)
    K_verg:                float = 4.0    # integration gain (1/s)
    K_phasic_verg:         float = 1.0    # direct feedthrough (dim'less); Smith predictor
    tau_verg:              float = 25.0   # vergence position leak TC (s); stable hold


# ── State layout ───────────────────────────────────────────────────────────────

N_STATES = vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES + ge.N_STATES + pu.N_STATES + vg.N_STATES
#        = 3 + 3 + 9 + 120 + 3 + 3 + 3 = 144

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
                     vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES + ge.N_STATES + pu.N_STATES)  # (3,)
_IDX_VERG    = slice(vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES + ge.N_STATES + pu.N_STATES,
                     N_STATES)                                                           # (3,)


def make_x0():
    """Default initial brain state — gravity estimator pointing down (upright head)."""
    x0 = jnp.zeros(N_STATES)
    return x0.at[_IDX_GRAV].set(ge.X0)


# ── Step function ──────────────────────────────────────────────────────────────

def step(x_brain, sensory_out, brain_params):
    """Single ODE step for the brain subsystem.

    Args:
        x_brain:     (144,)        brain state [x_vs | x_ni | x_sg | x_ec | x_grav | x_pursuit | x_verg]
        sensory_out: SensoryOutput bundled canal afferents + delayed visual signals
                       .canal          (6,)    canal afferent rates
                       .slip_delayed   (3,)    delayed retinal slip (no EC correction yet)
                       .pos_delayed    (3,)    L+R averaged delayed position error → SG
                       .vel_delayed    (3,)    delayed target velocity on retina → pursuit
                       .f_otolith      (3,)    specific force in head frame (m/s²)
                       .scene_visible  scalar  0=dark, 1=lit — gates EC slip correction
                       .target_visible scalar  0=no target in field, 1=target visible (= gate_vf)
                       .pos_delayed_L  (3,)    left  eye delayed position error → vergence
                       .pos_delayed_R  (3,)    right eye delayed position error → vergence
        brain_params: BrainParams   model parameters

    Returns:
        dx_brain:    (144,)  dx_brain/dt
        motor_cmd_L: (3,)    pulse-step motor command → left  plant
        motor_cmd_R: (3,)    pulse-step motor command → right plant
    """
    x_vs      = x_brain[_IDX_VS]
    x_ni      = x_brain[_IDX_NI]
    x_sg      = x_brain[_IDX_SG]
    x_ec      = x_brain[_IDX_EC]
    x_grav    = x_brain[_IDX_GRAV]
    x_pursuit = x_brain[_IDX_PURSUIT]
    x_verg    = x_brain[_IDX_VERG]

    # ── One EC, two corrections with separate gates ───────────────────────────
    # motor_ec = delay(u_burst + u_pursuit) — one cascade, read once, used twice.
    motor_ec = ec.read_delayed(x_ec)

    # OKR / VS: scene-gated — slip and EC correction both gated by scene_visible.
    #   Parallel to the pursuit path (target_visible gates vel_delayed + motor_ec).
    #   When dark: zero visual input to VS; x_vs decays freely with τ_vs → clean OKAN.
    #   When lit:  slip_delayed ≈ −(u_burst+u_pursuit)(t−τ)  →  corrected ≈ 0 ✓
    e_slip_corrected = sensory_out.scene_visible * (sensory_out.slip_delayed + motor_ec)

    # Pursuit: target-gated — foveal target slip only (excludes VOR, OKN, fixation)
    #   Gate the *entire* signal by target_visible (= gate_vf, delayed in-field gate).
    #   When target_present=0 in retina step, gate_vf → 0 → target_visible → 0.
    #   EC cancellation still works: vel_delayed ≈ v_target − w_eye(t−τ),
    #   motor_ec ≈ +w_eye(t−τ) → e_combined ≈ v_target ✓
    #   Smith predictor lives inside pu.step(): e_pred = (e_combined − x_p)/(1+K_ph)
    e_combined = sensory_out.target_visible * (sensory_out.vel_delayed + motor_ec)
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

    # ── Saccade generator (target selection handled internally) ───────────────
    # x_ni is the brain's proxy for current eye position (avoids plant state dependency)
    dx_sg, u_burst = sg.step(x_sg, sensory_out.pos_delayed, sensory_out.gate_vf, x_ni, brain_params)

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

    # ── Neural integrator: VOR + saccades + pursuit → version motor command ───
    dx_ni, motor_cmd_ni = ni.step(x_ni, -w_est + u_burst + u_pursuit, brain_params)

    # Add OCR position offset directly to motor command (bypasses NI leak)
    motor_cmd_version = motor_cmd_ni + ocr_pos

    # ── Vergence: binocular disparity → disconjugate eye commands ─────────────
    # e_disp = pos_delayed_L − pos_delayed_R; gated by target_present upstream.
    # EC correction: add x_verg to cancel the eye's own convergence contribution,
    # analogous to motor_ec correcting vel_delayed for pursuit.
    # Without: e_disp = ψ_d − ψ → closed-loop gain ≈ 0.5
    # With:    e_disp + x_verg ≈ ψ_d → gain ≈ 0.99
    e_disp = sensory_out.pos_delayed_L - sensory_out.pos_delayed_R
    dx_verg, u_verg = vg.step(x_verg, e_disp + x_verg, brain_params)

    # Split vergence ±½ around the version command
    motor_cmd_L = motor_cmd_version + 0.5 * u_verg
    motor_cmd_R = motor_cmd_version - 0.5 * u_verg

    # ── Efference copy: advance delay cascade with version motor command ──────
    # EC tracks the conjugate (version) command: u_burst + u_pursuit.
    # Vergence is disconjugate and does not contaminate VS or pursuit EC.
    dx_ec, _ = ec.step(x_ec, u_burst + u_pursuit, brain_params)

    # ── Pack state derivative ─────────────────────────────────────────────────
    dx_brain = jnp.concatenate([dx_vs, dx_ni, dx_sg, dx_ec, dx_grav, dx_pursuit, dx_verg])

    return dx_brain, motor_cmd_L, motor_cmd_R
