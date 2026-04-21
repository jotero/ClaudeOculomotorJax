"""Brain model — velocity storage, neural integrator, saccade generator, efference copy,
gravity estimator, smooth pursuit, and vergence.

Aggregates all brain subsystems into a single SSM with one state vector and one
step() function.

Signal flow:
    y_canals         (6,)   canal afferents                   → VS
    slip             (3,)   delayed raw retinal slip           → VS (after EC)
    target_slip      (3,)   delayed target velocity on retina  → pursuit (Smith predictor)
    e_cmd            (3,)   motor error command                → SG
    pos_L/R          (3,)   per-eye delayed position error     → vergence

One efference copy cascade (120 states), two uses with different gates:
    motor_ec = ec.read_delayed(x_ec)          # delay(u_burst + u_pursuit)

    OKR / VS correction  — scene-gated (full scene slip):
        okr = scene_visible · (slip + motor_ec)
        slip ≈ −(u_burst+u_pursuit)(t−τ)  →  corrected ≈ 0  ✓

    Pursuit Smith predictor — target-gated (foveal target slip only):
        e_combined = target_visible · (target_slip + motor_ec)   ≈ v_target when target on
        Full signal gated by target_visible → zero drive when no target in field
        e_vel_pred = (e_combined − x_pursuit) / (1 + K_phasic)
        → at onset:        ~45 % of v_target drives integrator  (less oscillation)
        → at steady state: e_vel_pred → 0  (integrator at rest, u_pursuit ≈ v_target)
        u_pursuit = x_pursuit + K_phasic · e_vel_pred

Vergence:
    e_disp = pos_L − pos_R   (binocular disparity, deg)
    PI controller (no efference copy / no Smith predictor — may be added later):
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
from oculomotor.models.brain_models import ocr                 as ocr_mod
from oculomotor.models.brain_models import optokinetic         as okr_mod
from oculomotor.models.sensory_models.sensory_model import SensoryOutput  # noqa: F401 (re-exported)
from oculomotor.models.plant_models.readout import rotation_matrix
from oculomotor.models.plant_models.muscle_geometry import (
    M_NUCLEUS, M_NERVE_PROJ, G_NUCLEUS_DEFAULT, G_NERVE_DEFAULT,
    CN3_MR_L, CN3_MR_R,
)


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
    g_ocr:                 float = 0.0    # OCR amplitude (deg); healthy ~10°; 0 = disabled until verified

    # Smooth pursuit — leaky integrator + direct feedthrough (Lisberger 1988)
    K_pursuit:             float = 4.0    # pursuit integration gain (1/s); rise TC ≈ 1/K_pursuit
    K_phasic_pursuit:      float = 5.0    # pursuit direct feedthrough (dim'less); fast onset
    tau_pursuit:           float = 40.0   # pursuit leak TC (s); ~40 s → ~97.5% gain at 1 Hz
    v_max_pursuit:         float = 40.0   # MT/MST velocity saturation (deg/s); clip on e_combined
                                           # Pursuit gain ≈1 up to ~30–40 deg/s, then falls (Fuchs 1967 J Physiol;
                                           # Lisberger & Westbrook 1985 J Neurosci). MT tuned 10–64 deg/s
                                           # (Newsome et al. 1988 J Neurosci); 40 deg/s is conservative.
    v_max_okr:             float = 80.0   # NOT/AOS velocity saturation (deg/s); clip on visual slip to VS
                                           # NOT neurons saturate ~80 deg/s (Hoffmann 1979 Exp Brain Res).
                                           # OKR gain ≈1 below 30 deg/s, half-max ~60 deg/s, near-zero ~100 deg/s
                                           # (Cohen, Matsuo & Raphan 1977 J Neurophysiol; Demer & Zee 1984 J Neurophysiol).

    # Vergence — single leaky integrator with dual-range nonlinear drive (Schor 1979)
    # Rashbass & Westheimer 1961; Jones 1980; Hung & Semmlow 1980; Judge & Miles 1985
    K_verg:                float        = 5.0             # fusional integration gain (1/s); high gain for fine disparity
    K_verg_prox:           float        = 1.0             # proximal integration gain (1/s); lower gain for full range
    K_phasic_verg:         float        = 1.0             # phasic feedthrough (dim'less); applied to fusional clip only
    tau_verg:              float        = 25.0            # vergence leak TC (s); leaks to phoria
    disp_max_verg_fus:        float        = 1.0             # fusional disparity saturation (deg); Panum's ~±1 deg (Jones 1980)
    disp_max_verg_prox:       float        = 20.0            # proximal disparity saturation (deg); full vergence range (Hung & Semmlow 1980)
    phoria:                jnp.ndarray  = jnp.zeros(3)    # resting vergence (deg); tonic setpoint; 0=orthophoria
                                                           # phoria[0]>0 esophoria, <0 exophoria

    # Motor nucleus and nerve gains — two-stage encode (see muscle_geometry.py)
    # Stage 1 — g_nucleus (12,): per-nucleus gain [0,1]. Zero = nucleus lesion.
    #   Nucleus order: ABN_L(0), ABN_R(1), CN4_L(2), CN4_R(3),
    #                  CN3_MR_L(4), CN3_MR_R(5), CN3_SR_L(6), CN3_SR_R(7),
    #                  CN3_IR_L(8), CN3_IR_R(9), CN3_IO_L(10), CN3_IO_R(11)
    #   ABN nucleus lesion isolates ipsilateral LR only (MLF not modelled separately).
    # Stage 2 — g_nerve (12,): per-nerve gain [0,1]. Zero = nerve/fascicular lesion.
    #   Nerve order: [LR_L,MR_L,SR_L,IR_L,SO_L,IO_L, LR_R,MR_R,SR_R,IR_R,SO_R,IO_R]
    #   CN nerve lesion isolates individual muscles without affecting other nuclei.
    # Healthy default: all ones → transparent round-trip through plant.
    g_nucleus:             jnp.ndarray  = G_NUCLEUS_DEFAULT  # (12,) motor nucleus gains
    g_nerve:               jnp.ndarray  = G_NERVE_DEFAULT    # (12,) per-nerve gains

    # INO (internuclear ophthalmoplegia) — version_yaw gain for each MR subnucleus.
    # The MLF (ABN → contralateral MR) is modelled as the version component of CN3_MR.
    # Zeroing g_mlf_ver_L cuts version drive to left MR → left eye can't adduct on
    # rightward gaze; vergence (vrg_yaw = +½) is preserved.
    #   g_mlf_ver_L = 0 → left  INO  (right MLF pathway cut: ABN_R → MR_L)
    #   g_mlf_ver_R = 0 → right INO  (left  MLF pathway cut: ABN_L → MR_R)
    g_mlf_ver_L:           float        = 1.0   # version_yaw gain for CN3_MR_L
    g_mlf_ver_R:           float        = 1.0   # version_yaw gain for CN3_MR_R


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
        # Vergence: initialise to phoria — tau_verg=25s is too slow to settle from zero.
        x0 = x0.at[_IDX_VERG].set(jnp.asarray(brain_params.phoria, dtype=jnp.float32))
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
        dx_brain:    (156,)  dx_brain/dt
        motor_cmd_L: (6,)    per-muscle activation vector → left  plant
        motor_cmd_R: (6,)    per-muscle activation vector → right plant
    """
    x_vs      = x_brain[_IDX_VS]      # (9,): bilateral VS + null
    x_ni      = x_brain[_IDX_NI]      # (9,): bilateral NI + null
    x_ni_net  = x_ni[:3] - x_ni[3:6]  # (3,): net eye position (L pop − R pop)
    x_sg      = x_brain[_IDX_SG]
    x_ec      = x_brain[_IDX_EC]
    x_grav    = x_brain[_IDX_GRAV]
    x_pursuit = x_brain[_IDX_PURSUIT]
    x_verg    = x_brain[_IDX_VERG]

    # ── Binocular combining — version (average) and vergence (difference) ────────
    # Each signal is gated by the per-eye delayed visibility (tv_L/R or sv_L/R),
    # then summed and divided by the number of eyes that can see (1 or 2).
    # scene/target_visible clip to [0,1] so they act as binary presence gates.
    # Vergence (e_disp) uses the difference of the per-eye gated positions,
    # further gated by bino = tv_L * tv_R so it is only active when both eyes
    # see the target (binocular fusion required for disparity).
    # TODO (diplopia): when disparity is too large the two retinal images are not
    # fused and the version error from each eye may diverge. In that regime the
    # weighted average is ambiguous — one option is to fall back to the dominant
    # eye's signal (e.g. whichever has higher tv) and suppress the other, similar
    # to what happens perceptually during suppression of the non-dominant eye.
    sv_L, sv_R = sensory_out.scene_vis_L, sensory_out.scene_vis_R
    tv_L, tv_R = sensory_out.target_vis_L, sensory_out.target_vis_R

    sv_sum  = sv_L + sv_R
    tv_sum  = tv_L + tv_R
    sv_norm = jnp.maximum(sv_sum, 1e-6)
    tv_norm = jnp.maximum(tv_sum, 1e-6)

    pos_L = tv_L * sensory_out.pos_L
    pos_R = tv_R * sensory_out.pos_R

    scene_slip     = (sv_L * sensory_out.slip_L + sv_R * sensory_out.slip_R) / sv_norm
    scene_visible  = jnp.clip(sv_sum, 0.0, 1.0)
    pos            = (pos_L + pos_R) / tv_norm
    target_slip    = (tv_L * sensory_out.vel_L + tv_R * sensory_out.vel_R) / tv_norm
    target_visible = jnp.clip(tv_sum, 0.0, 1.0)

    bino   = tv_L * tv_R
    e_disp = bino * (pos_L - pos_R)

    # ── One EC, two corrections with separate gates ───────────────────────────
    # motor_ec: efference copy of predicted eye velocity from saccades (u_burst) and smooth
    #   pursuit (u_pursuit), delayed to match visual processing lag.  Used to cancel
    #   self-generated retinal slip in both OKR (scene gate) and pursuit (target gate).
    motor_ec = ec.read_delayed(x_ec)

    # ── Optokinetic: scene-gated EC-corrected slip → visual drive for VS ─────────
    okr = okr_mod.compute(scene_slip, motor_ec, scene_visible, brain_params)

    # ── Velocity storage: combines VOR (canal) + OKR (visual) reflexes → ω̂ head ──
    # Canal provides vestibular drive; OKR provides visual drive; together they give
    # VS an estimate of head angular velocity that is more accurate than either alone.
    dx_vs, w_est = vs.step(
        x_vs,
        jnp.concatenate([sensory_out.canal, okr, x_grav]),
        brain_params)

    # ── Gravity estimator: cross-product transport + otolith correction ────────
    dx_grav, g_hat = ge.step(
        x_grav,
        jnp.concatenate([w_est, sensory_out.f_otolith]),
        brain_params)

    # ── OCR: torsional counter-roll driven by head tilt ──────────────────────
    ocr = ocr_mod.compute(g_hat, brain_params)

    # ── Pursuit: target-gated EC-corrected velocity → pursuit integrator ─────────
    dx_pursuit, u_pursuit = pu.step(x_pursuit, target_slip, motor_ec, target_visible, brain_params)

    # ── Saccade generator (target selection handled internally) ───────────────
    # x_ni_net is the brain's proxy for current eye position (avoids plant state dependency)
    dx_sg, u_burst = sg.step(x_sg, pos, target_visible, x_ni_net, brain_params)

    # ── Neural integrator: VOR + saccades + pursuit → version motor command ───
    # ocr / tau_i is a tonic drive that settles the NI at ocr in steady
    # state, acting as a torsional setpoint without bypassing the integrator leak.
    dx_ni, motor_cmd_ni = ni.step(x_ni, -w_est + u_burst + u_pursuit + ocr / brain_params.tau_i, brain_params)

    # ── Vergence: binocular disparity → disconjugate eye commands ─────────────
    # bino = tv_L * tv_R ≈ 1 when both eyes fuse, 0 when either covered.
    # When bino = 0: e_disp = 0 → x_verg drifts to phoria with TC = tau_verg.
    # High K·τ ≈ 150 → CL gain ~99% without state correction.
    dx_verg, u_verg = vg.step(x_verg, e_disp, brain_params)

    # ── Two-stage motor nucleus encode → per-muscle nerve activations ────────
    # Stage 1: [version, vergence] (6,) → nuclei (12,) via M_NUCLEUS
    # Stage 2: nuclei → nerves (12,) via M_NERVE_PROJ (CN4 → contralateral SO;
    #          all other projections ipsilateral)
    # Non-negative encoding: relu + factor-2 is exact for all antipodal pairs
    # (M_NUCLEUS rows are antipodal by construction — see muscle_geometry.py).
    version_vergence = jnp.concatenate([motor_cmd_ni, u_verg])               # (6,)
    # Apply INO gains: scale version_yaw (col 0) of CN3_MR rows before projecting.
    m_nuc = M_NUCLEUS \
        .at[CN3_MR_L, 0].mul(brain_params.g_mlf_ver_L) \
        .at[CN3_MR_R, 0].mul(brain_params.g_mlf_ver_R)
    nuclei_raw = m_nuc @ version_vergence                                         # (12,) signed
    nuclei   = jnp.diag(brain_params.g_nucleus) @ (2 * jnp.maximum(nuclei_raw, 0.0))  # (12,) non-neg
    nerves   = jnp.diag(brain_params.g_nerve) @ (M_NERVE_PROJ @ nuclei)          # (12,)
    motor_cmd_L = nerves[:6]   # (6,) left  eye nerve activations
    motor_cmd_R = nerves[6:]   # (6,) right eye nerve activations

    # ── Efference copy: advance delay cascade with version motor command ──────
    # Rotate (u_burst + u_pursuit) from head frame into eye frame before delaying.
    # slip is in eye frame (retinal_signals applies R_gaze_T); the EC must
    # delay the SAME frame.  Rotating at cascade INPUT ensures motor_ec at readout
    # carries R_eye_T(t−τ) @ u(t−τ), which matches slip(t) exactly —
    # both use the gaze angle from the same past time t−τ.
    # Approximation: R_head ≈ I (head stationary during saccades) → R_gaze_T ≈ R_eye_T.
    # x_ni_net proxies current gaze; [yaw,pitch,roll] → permute for rotation_matrix.
    _q2rv_ec = lambda q: jnp.array([-q[1], q[0], q[2]])
    R_eye_T  = rotation_matrix(_q2rv_ec(x_ni_net)).T   # head → eye frame
    dx_ec, _ = ec.step(x_ec, R_eye_T @ (u_burst + u_pursuit), brain_params)

    # ── Pack state derivative ─────────────────────────────────────────────────
    dx_brain = jnp.concatenate([dx_vs, dx_ni, dx_sg, dx_ec, dx_grav, dx_pursuit, dx_verg])

    return dx_brain, motor_cmd_L, motor_cmd_R
