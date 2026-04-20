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
        Full signal gated by target_visible (= target_in_vf) → zero drive when no target in field
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

State vector  x_brain = [x_vs (9) | x_ni (9) | x_sg (9) | x_ec (120) | x_grav (3) | x_pursuit (3) | x_verg (3)]
N_STATES = 156

Index constants (relative to x_brain):
    _IDX_VS       — velocity storage states   (9,)  = left(3) + right(3) + null(3)
    _IDX_VS_L     — left  VN population       (3,)
    _IDX_VS_R     — right VN population       (3,)
    _IDX_VS_NULL  — VS null adaptation state  (3,)
    _IDX_NI       — neural integrator states  (9,)  = left(3) + right(3) + null(3)
    _IDX_NI_L     — left  NPH population      (3,)
    _IDX_NI_R     — right NPH population      (3,)
    _IDX_NI_NULL  — NI null adaptation state  (3,)
    _IDX_SG       — saccade generator states  (9,)
    _IDX_EC       — efference copy states     (120,)
    _IDX_GRAV     — gravity estimator states  (3,)
    _IDX_PURSUIT  — pursuit velocity memory   (3,)
    _IDX_VERG     — vergence position memory  (3,)

Outputs of step():
    dx_brain     (156,)  state derivative
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
    K_vis:                 float = 0.1    # visual-to-VS gain (1/s); OKR / OKAN charging
                                          # OKN SS gain ≈ (2·K_vis·τ_vs + g_vis)/(1 + 2·K_vis·τ_vs + g_vis)
                                          # K_vis=0.1, g_vis=0.6 → SS gain ≈ 0.82  (Raphan 1979)
    g_vis:                 float = 0.6    # direct visual pathway gain (Raphan 1979, Fig. 8: gl = 0.6)
                                          # OKR inner loop: L(jω) ≈ g_vis·exp(−jω·τ_vis) → stable iff g_vis < 1
                                          # g_vis=1.0 → zero gain margin → sustained ~6 Hz onset ringing
                                          # g_vis=0.6 → τ_decay = τ_vis/ln(1/0.6) ≈ 157 ms (~1 cycle, acceptable)
                                          # SS OKN: gain ≈ 0.82, INT/SPV ≈ 0.87
    b_vs:                  float = 100.0  # VN resting bias AND population gain (deg/s).
                                          # Scalar broadcasts to all 6 states; pass a (6,) array for asymmetry.
                                          # velocity_storage scales canal drive by b_vs / B_NOMINAL, so b_vs
                                          # simultaneously sets the equilibrium firing rate and the input
                                          # responsiveness of each population — one parameter controls both.
    tau_vs_adapt:          float = 600.0  # VS null adaptation TC (s); >> tau_vs → negligible in normal demos
                                          # reduce to ~30–60 s to engage PAN-like slow oscillation

    # Neural integrator — bilateral push-pull + null adaptation (Robinson 1975; rebound: Zee et al. 1980)
    tau_i:                 float = 25.0   # leak TC (s); healthy >20 s (Cannon & Robinson 1985)
    tau_p:                 float = 0.15   # plant TC copy — NI feedthrough for lag cancellation
    tau_vis:               float = 0.08   # visual delay copy — EC delay must match retinal delay
                                          # should match PlantParams.tau_p in healthy subjects;
                                          # may differ in pathology (imperfect internal model)
    b_ni:                  float = 0.0    # NPH intrinsic resting bias (deg); 0 = no net bias at centre gaze
                                          # future: set >0 for unilateral NPH lesion modelling
    tau_ni_adapt:          float = 20.0   # NI null adaptation TC (s); controls rebound nystagmus amplitude
                                          # τ_ni_adapt → ∞: no rebound; τ_ni_adapt ~10–30 s: visible rebound

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
    K_verg:                float        = 4.0             # integration gain (1/s)
    K_phasic_verg:         float        = 1.0             # direct feedthrough (dim'less)
    tau_verg:              float        = 25.0            # vergence position leak TC (s)
    phoria:                jnp.ndarray  = jnp.zeros(3)    # resting vergence (deg); 0=orthophoria
                                                           # phoria[0]>0 esophoria, <0 exophoria


# ── State layout ───────────────────────────────────────────────────────────────

N_STATES = vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES + ge.N_STATES + pu.N_STATES + vg.N_STATES
#        = 9 + 9 + 9 + 120 + 3 + 3 + 3 = 156

# ── Index constants — relative to x_brain ─────────────────────────────────────
# Computed from module N_STATES to stay in sync automatically.

_o_vs  = 0
_o_ni  = _o_vs + vs.N_STATES   # 9
_o_sg  = _o_ni + ni.N_STATES   # 18
_o_ec  = _o_sg + sg.N_STATES   # 27
_o_gv  = _o_ec + ec.N_STATES   # 147
_o_pu  = _o_gv + ge.N_STATES   # 150
_o_vg  = _o_pu + pu.N_STATES   # 153

# Velocity storage (9 states: L pop + R pop + null)
_IDX_VS      = slice(_o_vs,     _o_vs + 9)   # (9,)
_IDX_VS_L    = slice(_o_vs,     _o_vs + 3)   # (3,) left  VN pop
_IDX_VS_R    = slice(_o_vs + 3, _o_vs + 6)   # (3,) right VN pop
_IDX_VS_NULL = slice(_o_vs + 6, _o_vs + 9)   # (3,) null adaptation

# Neural integrator (9 states: L pop + R pop + null)
_IDX_NI      = slice(_o_ni,     _o_ni + 9)   # (9,)
_IDX_NI_L    = slice(_o_ni,     _o_ni + 3)   # (3,) left  NPH pop
_IDX_NI_R    = slice(_o_ni + 3, _o_ni + 6)   # (3,) right NPH pop
_IDX_NI_NULL = slice(_o_ni + 6, _o_ni + 9)   # (3,) null adaptation

# Remaining subsystems
_IDX_SG      = slice(_o_sg, _o_sg + sg.N_STATES)   # (9,)
_IDX_EC      = slice(_o_ec, _o_ec + ec.N_STATES)   # (120,)
_IDX_GRAV    = slice(_o_gv, _o_gv + ge.N_STATES)   # (3,)
_IDX_PURSUIT = slice(_o_pu, _o_pu + pu.N_STATES)   # (3,)
_IDX_VERG    = slice(_o_vg, N_STATES)               # (3,)


def make_x0(brain_params=None):
    """Default initial brain state.

    VS populations initialised to b_vs (bilateral equilibrium — both pops at resting bias).
    VS/NI null adaptation states initialised to 0 (no initial adaptation).
    NI populations initialised to 0 (b_ni=0 → net=0 at centre gaze).
    Gravity estimator initialised pointing down (upright head).

    Args:
        brain_params: BrainParams NamedTuple.  If None, uses b_vs=0 (zero bias; old behaviour).
    """
    x0 = jnp.zeros(N_STATES)
    x0 = x0.at[_IDX_GRAV].set(ge.X0)
    if brain_params is not None:
        # VS: both populations start at resting bias; null starts at 0.
        b6 = jnp.broadcast_to(jnp.asarray(brain_params.b_vs, dtype=jnp.float32), (6,))
        x0 = x0.at[_IDX_VS_L].set(b6[:3])
        x0 = x0.at[_IDX_VS_R].set(b6[3:])
        # _IDX_VS_NULL stays at 0 (no initial adaptation)
        # NI: b_ni populations (b_ni=0 default → stays zero)
        # _IDX_NI_L/R/NULL all stay at 0
    return x0


# ── Step function ──────────────────────────────────────────────────────────────

def step(x_brain, sensory_out, brain_params):
    """Single ODE step for the brain subsystem.

    Args:
        x_brain:     (156,)        brain state [x_vs (9) | x_ni (9) | x_sg | x_ec | x_grav | x_pursuit | x_verg]
        sensory_out: SensoryOutput bundled canal afferents + per-eye raw signals
                       .canal          (6,)    canal afferent rates
                       .f_otolith      (3,)    specific force in head frame (m/s²)
                       .slip_L/R       (3,)    per-eye raw delayed scene velocity
                       .pos_L/R        (3,)    per-eye raw delayed target position
                       .vel_L/R        (3,)    per-eye raw delayed target velocity
                       .scene_vis_L/R  scalar  delay(scene_present)
                       .target_vis_L/R scalar  delay(target_present × target_in_vf)
        brain_params: BrainParams   model parameters

    Returns:
        dx_brain:    (144,)  dx_brain/dt
        motor_cmd_L: (3,)    pulse-step motor command → left  plant
        motor_cmd_R: (3,)    pulse-step motor command → right plant
    """
    x_vs      = x_brain[_IDX_VS]      # (9,): bilateral VS + null
    x_ni      = x_brain[_IDX_NI]      # (9,): bilateral NI + null
    x_ni_net  = x_ni[:3] - x_ni[3:6]  # (3,): net eye position (L pop − R pop)
    x_sg      = x_brain[_IDX_SG]
    x_ec      = x_brain[_IDX_EC]
    x_grav    = x_brain[_IDX_GRAV]
    x_pursuit = x_brain[_IDX_PURSUIT]
    x_verg    = x_brain[_IDX_VERG]

    # ── Binocular combining — gate × signal, then visibility-weighted average ──
    sv_L, sv_R = sensory_out.scene_vis_L, sensory_out.scene_vis_R
    tv_L, tv_R = sensory_out.target_vis_L, sensory_out.target_vis_R

    sv_sum  = sv_L + sv_R
    tv_sum  = tv_L + tv_R
    sv_norm = jnp.maximum(sv_sum, 1e-6)
    tv_norm = jnp.maximum(tv_sum, 1e-6)

    slip_delayed   = (sv_L * sensory_out.slip_L + sv_R * sensory_out.slip_R) / sv_norm
    scene_visible  = 0.5 * sv_sum
    pos_delayed    = (tv_L * sensory_out.pos_L + tv_R * sensory_out.pos_R) / tv_norm
    vel_delayed    = (tv_L * sensory_out.vel_L + tv_R * sensory_out.vel_R) / tv_norm
    target_visible = 0.5 * tv_sum
    target_in_vf   = jnp.clip(tv_sum, 0.0, 1.0)
    pos_delayed_L  = tv_L * sensory_out.pos_L   # per-eye gated positions for vergence
    pos_delayed_R  = tv_R * sensory_out.pos_R

    # ── One EC, two corrections with separate gates ───────────────────────────
    # motor_ec = delay(u_burst + u_pursuit) — one cascade, read once, used twice.
    motor_ec = ec.read_delayed(x_ec)

    # OKR / VS: scene-gated — slip and EC correction both gated by scene_visible.
    #   When dark: zero visual input to VS; x_vs decays freely with τ_vs → clean OKAN.
    #   When lit:  slip_delayed ≈ −(u_burst+u_pursuit)(t−τ)  →  corrected ≈ 0 ✓
    e_slip_corrected = scene_visible * (slip_delayed + motor_ec)

    # Pursuit: target-gated — foveal target slip only (excludes VOR, OKN, fixation)
    #   Gate the *entire* signal by target_visible.
    #   EC cancellation: vel_delayed ≈ v_target − w_eye(t−τ), motor_ec ≈ +w_eye(t−τ) ✓
    #   Smith predictor lives inside pu.step(): e_pred = (e_combined − x_p)/(1+K_ph)
    e_combined = target_visible * (vel_delayed + motor_ec)
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
    # x_ni_net is the brain's proxy for current eye position (avoids plant state dependency)
    dx_sg, u_burst = sg.step(x_sg, pos_delayed, target_in_vf, x_ni_net, brain_params)

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
    # ni.step takes (9,) x_ni [L|R|null] and returns (9,) dx + (3,) motor_cmd on net
    dx_ni, motor_cmd_ni = ni.step(x_ni, -w_est + u_burst + u_pursuit, brain_params)

    # Add OCR position offset directly to motor command (bypasses NI leak)
    motor_cmd_version = motor_cmd_ni + ocr_pos

    # ── Vergence: binocular disparity → disconjugate eye commands ─────────────
    # Binocularity gate: vergence drive requires both eyes to see the target.
    #   bino = tv_L * tv_R ≈ 1 when both eyes fuse, 0 when either covered.
    # When bino = 0: e_disp = 0 → dx_verg = −(x_verg − phoria)/τ_verg ✓
    # EC correction: add x_verg so that, when bino>0, e_pred = e_disp/(1+K_ph).
    bino   = tv_L * tv_R
    e_disp = bino * (pos_delayed_L - pos_delayed_R)
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
