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
    GE  →  g_est (gravity estimate, cross-product dynamics)
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

import jax
import jax.numpy as jnp

from oculomotor.models.brain_models import velocity_storage    as vs
from oculomotor.models.brain_models import neural_integrator   as ni
from oculomotor.models.brain_models import saccade_generator   as sg
from oculomotor.models.brain_models import gravity_estimator   as ge
from oculomotor.models.brain_models import heading_estimator   as he
from oculomotor.models.brain_models import pursuit             as pu
from oculomotor.models.brain_models import vergence            as vg
from oculomotor.models.brain_models import accommodation           as acc_mod
from oculomotor.models.brain_models import final_common_pathway    as fcp

from oculomotor.models.sensory_models.sensory_model import SensoryOutput
from oculomotor.models.brain_models.final_common_pathway import G_NUCLEUS_DEFAULT, G_NERVE_DEFAULT


# ── State layout ───────────────────────────────────────────────────────────────

N_STATES = vs.N_STATES + ni.N_STATES + sg.N_STATES + ge.N_STATES + he.N_STATES + pu.N_STATES + vg.N_STATES + acc_mod.N_STATES
#        = 9 + 9 + 20 + 9 + 3 + 3 + 9 + 2 = 64
# EC delay cascades removed — EC subtraction happens pre-delay in cyclopean_vision.step().

# ── Index constants — relative to x_brain ─────────────────────────────────────
# Computed from module N_STATES to stay in sync automatically.

_o_vs = 0
_o_ni = _o_vs + vs.N_STATES    #  9
_o_sg = _o_ni + ni.N_STATES    # 18
_o_gv = _o_sg + sg.N_STATES    # 38
_o_hd = _o_gv + ge.N_STATES    # 47  (ge.N_STATES=9: g_est+a_lin+rf)
_o_pu = _o_hd + he.N_STATES    # 50
_o_vg = _o_pu + pu.N_STATES    # 53

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
_IDX_SG      = slice(_o_sg, _o_sg + sg.N_STATES)        # (20,) [18:38]
_IDX_GRAV    = slice(_o_gv, _o_gv + ge.N_STATES)        # (9,)  [38:47]
_IDX_HEAD    = slice(_o_hd, _o_hd + he.N_STATES)        # (3,)  [47:50]
_IDX_PURSUIT = slice(_o_pu, _o_pu + pu.N_STATES)        # (3,)  [50:53]
_IDX_VERG    = slice(_o_vg, _o_vg + vg.N_STATES)        # (3,)  [53:56]
_o_acc       = _o_vg + vg.N_STATES                      # 45
_IDX_ACC     = slice(_o_acc, _o_acc + acc_mod.N_STATES)  # (2,)  [45:47]


# ── Brain parameters ────────────────────────────────────────────────────────────

class BrainParams(NamedTuple):
    """Learnable central parameters — fit to patient eye-movement data."""

    # Velocity storage — Raphan, Matsuo & Cohen (1979)
    # Bilateral push-pull (L/R VN populations); net = x_L − x_R; OKAN decays with tau_vs.
    #
    # VS time constants — per-axis via fractions of the main (yaw) TC.
    #   tau_vs             : yaw   ~15–20 s  (Raphan et al. 1979; Cohen et al. 1981)
    #   tau_vs * f_pitch   : pitch ~5–10 s   (Hess & Dieringer 1991; Angelaki & Henn 2000)
    #   tau_vs * f_roll    : roll  ~2–5 s    (Dai et al. 1991; Angelaki et al. 1995)
    # Disorders that shorten tau_vs (nodulus lesion, UVH) only need to set tau_vs —
    # the ratios stay fixed, so all axes scale together and no existing code breaks.
    tau_vs:                float = 20.0   # yaw (main) VS TC (s); ~20 s monkey (Cohen 1977)
    tau_vs_pitch_frac:     float = 1.0    # pitch TC = tau_vs × this  → 20 s
    tau_vs_roll_frac:      float = 1.0    # roll  TC = tau_vs × this  → 20 s
    b_vs:                  float = 100.0  # VN resting bias AND population gain (deg/s).
                                          # Scalar broadcasts to all 6 states; pass a (6,) array for asymmetry.
                                          # velocity_storage scales canal drive by b_vs / B_NOMINAL, so b_vs
                                          # simultaneously sets the equilibrium firing rate and the input
                                          # responsiveness of each population — one parameter controls both.
    tau_vs_adapt:          float = 600.0  # VS null adaptation TC (s); >> tau_vs → negligible in normal demos
                                          # reduce to ~30–60 s to engage PAN-like slow oscillation

    # Vestibular reflex (VOR) — semicircular canal drive to velocity storage
    g_vor:                 float = 1.0    # VOR central gain (dim'less); scales canal bypass to VS output.
                                          # 1.0 = healthy; <1 = hypofunction / adaptation down;
                                          # >1 = adaptation up. Distinct from peripheral canal_gains
                                          # (SensoryParams), which reflect transduction sensitivity.
    v_max_vor:             float = 400.0  # excitatory canal afferent saturation (deg/s).
                                          # Inhibitory saturation (~80 deg/s, Ewald's 2nd law) is already
                                          # implemented as FLOOR in canal.nonlinearity() — not modelled here.
                                          # Excitatory ceiling ~300–600 deg/s (Goldberg & Fernández 1971
                                          # J Neurophysiol 34:635); 400 is conservative.
                                          # At typical stimulus velocities (<200 deg/s) this clip is inert.
    K_vs:                  float = 0.1    # canal-to-VS integration gain (1/s); controls charging speed
                                          # Bilateral push-pull: effective net gain = 2·K_vs = 0.2.

    # Optokinetic reflex (OKR/OKAN) — NOT/AOS visual drive to velocity storage
    # Pathways: direct (g_vis, fast) + integrating (K_vis → VS, slow OKAN buildup)
    K_vis:                 float = 0.1    # visual-to-VS gain (1/s); OKR / OKAN charging
                                          # Bilateral push-pull: effective net gain = 2·K_vis = 0.2
                                          # OKN SS gain ≈ (2·K_vis·τ_vs + g_vis)/(1 + 2·K_vis·τ_vs + g_vis)
                                          # K_vis=0.1, g_vis=0.6 → SS gain ≈ 0.82  (Raphan 1979)
    g_vis:                 float = 0.6    # direct visual pathway gain (Raphan 1979, Fig. 8: gl = 0.6)
                                          # OKR inner loop: L(jω) ≈ g_vis·exp(−jω·τ_vis) → stable iff g_vis < 1
                                          # g_vis=1.0 → zero gain margin → sustained ~6 Hz onset ringing
                                          # g_vis=0.6 → τ_decay = τ_vis/ln(1/0.6) ≈ 157 ms (~1 cycle, acceptable)
                                          # SS OKN: gain ≈ 0.82, INT/SPV ≈ 0.87
    v_max_okr:             float = 80.0   # NOT/AOS velocity saturation (deg/s); clip on visual slip to VS
                                          # NOT neurons saturate ~80 deg/s (Hoffmann 1979 Exp Brain Res).
                                          # OKR gain ≈1 below 30 deg/s, half-max ~60 deg/s, near-zero ~100 deg/s
                                          # (Cohen, Matsuo & Raphan 1977 J Neurophysiol; Demer & Zee 1984 J Neurophysiol).

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
    tau_latch:             float = 0.003  # unused; kept for backward compat
    tau_hold:              float = 0.020  # e_held tracking TC (s) between saccades.
                                           # 180ms accumulation / 20ms TC = 9 TCs → 99.99% capture.
                                           # Burst (normalized_opn=0) freezes tracking during saccade.
    tau_sac:               float = 0.001  # saccade latch TC (s)
    k_tonic_opn:           float = 0.5    # OPN tonic recovery gain; recovery TC = tau_sac/k_tonic = 2 ms
                                           # always active: keeps z_opn at 100 unless IBN overcomes it.
                                           # Suppression condition: g_ibn_opn > k_tonic*100 (200>50).
    tau_bn:                float = 0.003  # EBN/IBN state TC (s); BN states track error drive with lag.
                                           # Heun stability limit: (1+g_opn_bn·100)·dt/tau_bn < 2
                                           #   → tau_bn > (1+4)*0.001/2 = 0.0025 s. Current 3 ms is safely above.
    g_opn_bn:              float = 0.04   # OPN→BN multiplicative suppression (per unit, act_opn∈[0,100]).
                                           # Heun stability: (1+g_opn_bn·100)·dt/tau_bn < 2 → g_opn_bn < 0.09.
    g_opn_bn_hold:         float = 0.4    # OPN→BN additive offset (per unit, act_opn∈[0,100]).
                                           # At tonic (act_opn=100): opn_inh=40°. Tonic BN_eq = (e_held−40)/5 ≈ −8°.
                                           # Keeps BNs negative for fixation errors up to ~40° → IBN=0 between saccades.
    g_ibn_bn:              float = 0.143  # IBN→BN contralateral inhibition gain (= g_ci/g_burst = 100/700).
                                           # Element-wise: L yaw IBN → R yaw BN, etc. Absorbs g_burst normalisation.
    g_opn_pause:           float = 500.0  # OPN inhibitory overshoot (fr units); IBN inhibition drives OPN
                                           # membrane potential to −g_opn_pause (below spike threshold).
                                           # Firing rate clips at 0; large value → OPN pauses in ~2 ms.
    tau_acc:               float = 0.180  # accumulator rise TC (s); ~120 ms to threshold → cascade fully settled before e_held freezes
    tau_burst_drain:       float = 0.002  # accumulator burst drain TC (s); drives z_acc → acc_burst_floor while OPN paused
    acc_burst_floor:       float = -0.5   # accumulator target level during burst; negative = must re-climb after saccade
                                           # ISI ≈ (threshold_acc − acc_burst_floor) * tau_acc = 1.5 * 0.18 = 270 ms
                                           # No idle drain needed — every saccade resets the accumulator
    threshold_acc:         float = 1.0    # accumulator trigger threshold (natural unit: triggers at 1)
    threshold_sac_qp:      float = 2.0    # error threshold to START accumulating for quick phases (target_visible≈0)
                                           # Higher → requires larger drift error before accumulation begins → fewer spurious resets
    k_acc:                 float = 500.0  # accumulator→OPN sigmoid steepness; high value = near-hard threshold
                                           # OPN only starts dropping in the last ~0.2 ms before z_acc crosses
                                           # threshold_acc, so x_copy barely grows during the OPN transition
    sigma_acc:             float = 0.2    # accumulator diffusion noise (1/√s); adds RT variability
                                           # ~0.15–0.25 gives ±15–25 ms SD; 0 = deterministic
    g_ibn_opn:             float = 200.0  # IBN→OPN inhibition gain (OPN units/s / tau_sac).
                                           # IBN total = |act_ibn_R| + |act_ibn_L| (scalar, deg/s units).
                                           # Schmitt trigger: burst→IBN keeps OPN suppressed; burst ends→IBN→0→OPN recovers.
    tau_trig:              float = 0.002  # z_trig rise TC (s); smooth onset for OPN suppression.
                                           # Heun stability: (1+g_ibn_trig)·dt/tau_trig < 2 → tau_trig > 1.5ms. Current 2ms ✓.
    g_ibn_trig:            float = 2.0    # IBN→z_trig drain gain (dimensionless, ibn_norm∈[0,1]).
                                           # z_trig TC during burst: tau_trig/(1+g_ibn_trig) ≈ 0.7ms → fast drain by IBN.
    g_acc_drain:           float = 3.0    # IBN drain multiplier for z_acc (dimensionless).
                                           # Boosts drain for small saccades (weak IBN) without breaking triggering.
                                           # Heun bound: g_acc_drain · dt / tau_burst_drain < 2 → max ≈ 4 at current params.
    tau_acc_leak:          float = 5.0    # accumulator passive leak TC (s); pulls z_acc → 0 between saccades.
                                           # ~5% effect on 270 ms refractory. Keep ≥ 5 s: shorter values starve
                                           # small saccades (0.5°, gate_err=0.5) during OPN suppression phase.

    # Saccade target selection — handled inside the saccade generator
    orbital_limit:         float = 50.0   # oculomotor range half-width (deg); clip e_cmd to ±limit
    alpha_reset:           float = 1.25   # centering gain; e_center = −α·x_ni when out-of-field
    k_center_vel:          float = 0.75   # quick-phase prediction gain (0=no look-ahead, 1=full τ_ref look-ahead)
                                          # look-ahead = k·τ_ref; aims past centre to compensate for slow-phase
                                          # drift during refractory.  Only applied in out-of-field (quick-phase) path.

    # Otolith / gravity estimation — Laurens & Angelaki (2011, 2017)
    tau_grav:              float = 5.0    # gravity estimate TC (s); somatogravic bandwidth = 1/(2π·tau_grav) ≈ 0.032 Hz
                                          # sets how fast g_est tracks GIA changes (OCR rise time, OVAR following)
    K_lin:                 float = 0.1    # linear acceleration adaptation gain (1/s); 0 disables â state
                                          # smaller → more somatogravic effect; larger → faster a_lin adaptation
    K_gd:                  float = 0.0    # gravity dumping gain (1/s); 0 = disabled
    g_ocr:                 float = 0.0    # OCR amplitude (deg); healthy ~10°; 0 = disabled until verified

    # Heading estimator — linear velocity in head-fixed frame
    tau_head:              float = 2.0    # linear velocity integration TC (s); corner freq ≈ 0.08 Hz
                                          # leaky integral of a_est = GIA − g_est; prevents drift

    # Listing's law — torsional constraint (Listing 1854; Tweed et al. 1998)
    listing_primary:       jnp.ndarray = jnp.zeros(2)  # primary position [yaw₀, pitch₀] (deg)
                                                         # shifts the centre of Listing's plane
                                                         # 0 = straight ahead (healthy default)
    listing_l2_frac:       float = 0.0    # L2 cyclovergence fraction (0=off, 0.5=physiological)
                                          # Listing's plane tilts ±l2_frac·φ/2 per eye with vergence
                                          # Disabled until validated with binocular torsion data

    # Smooth pursuit — leaky integrator + Smith predictor (Lisberger 1988)
    K_pursuit:             float = 4.0    # pursuit integration gain (1/s); rise TC ≈ 1/K_pursuit
    K_phasic_pursuit:      float = 5.0    # pursuit direct feedthrough (dim'less); fast onset
    tau_pursuit:           float = 40.0   # pursuit leak TC (s); ~40 s → ~97.5% gain at 1 Hz
    v_max_pursuit:         float = 40.0   # MT/MST velocity saturation (deg/s); clip on target slip + EC
                                           # Pursuit gain ≈1 up to ~30–40 deg/s, then falls (Fuchs 1967 J Physiol;
                                           # Lisberger & Westbrook 1985 J Neurosci). MT tuned 10–64 deg/s
                                           # (Newsome et al. 1988 J Neurosci); 40 deg/s is conservative.

    # Vergence — single leaky integrator with dual-range nonlinear drive (Schor 1979)
    # Rashbass & Westheimer 1961; Jones 1980; Hung & Semmlow 1980; Judge & Miles 1985
    K_verg:                float        = 4.0             # fusional integration gain (1/s); high gain for fine disparity
    K_verg_prox:           float        = 3.0             # proximal integration gain (1/s); lower gain for full range
    K_phasic_verg:         float        = 1.0             # phasic feedthrough (dim'less); applied to fusional clip only
    tau_verg:              float        = 6.0             # vergence leak TC (s); leaks to tonic_verg [Semmlow 1986: ~5–7 s]
    # ── Sensory limits (relative retinal disparity — what the visual system measures) ──────
    #    Fusion limits (Panum's area): within these, the fine slow-vergence servo is active.
    #    Disparity limit (prox_sat): the coarse drive saturates here; beyond it the rate
    #    is capped but the system keeps driving toward fusion.
    #    All in disparity space, not eye-position space.
    panum_h:               float        = 2.0             # horizontal fusion limit (deg); Panum's area; fused/fusable boundary
                                                          # Jones (1980): ~±1° (total 2°)
    panum_v:               float        = 3.0             # vertical fusion limit ±(deg); ~2–3° clinical norm
                                                          # (Saladin 1988 Optom Vis Sci)
    panum_t:               float        = 5.0             # torsional fusion limit ±(deg); ~5–8° max
                                                          # (van Rijn & van den Berg 1993 Vision Res; Kertesz 1983)
    prox_sat:              float        = 20.0            # disparity limit: coarse (proximal) drive saturation (deg)
                                                          # Hung & Semmlow 1980; max disparity input to coarse drive; rate-caps beyond
    tonic_verg:            float        = 3.67            # tonic (brainstem) vergence baseline (deg); resting dark vergence
                                                          # = 2·arctan(IPD/2 / 1 m); recomputed from IPD in default_params()
                                                          # Riggs & Niehl 1960, Morgan 1944: dark vergence ≈ 1 m
                                                          # NOTE: phoria is a *measurement* (cover test outcome), not a model param;
                                                          # it emerges from the balance of tonic, AC/A, and fusional drives
    g_burst_verg:          float        = 1.6             # vergence saccade pulse gain (deg/s per deg residual)
                                                          # Zee (1992): vergence 2–3× faster during saccades (peak ~12 deg/s for 5° demand)
                                                          # τ_burst = 1/g = 625 ms → covers ~11% of demand during 70 ms saccade
                                                          # 0 = Zee mechanism disabled (isolated fusional vergence only)
    D_verg:                float        = 1.0             # velocity damping coefficient (dim'less); divides dx_verg by (1+D)
                                                          # D=0 → underdamped; D=1 → ξ≈0.9 (near-critical); D=2 → overdamped
                                                          # Schor (1979): vergence has significant viscous damping to prevent overshoot


    # Accommodation — Schor dual-interaction model (Schor 1979; Schor & Ciuffreda 1983)
    # Cross-coupled to vergence via AC/A and CA/C ratios.
    tau_acc_fast:          float = 0.3    # fast (phasic) lens TC (s) [Hung & Semmlow 1980]
    tau_acc_slow:          float = 30.0   # slow tonic adaptation TC (s) [Schor 1979]
    K_acc_fast:            float = 1.5    # fast blur-to-accommodation gain (1/s)
    K_acc_slow:            float = 0.3    # slow integration gain (1/s)
    AC_A:                  float = 5.0    # AC/A ratio (prism diopters / diopter); typical 4–6
                                          # At 40 cm (2.5 D): AC/A drive ≈ 5×0.573×2.5 ≈ 7.2°
                                          # At   6 m (0.17 D): drive ≈ 0.5° — explains why
                                          # IXT decompensates preferentially at distance.
    CA_C:                  float = 0.4    # CA/C ratio (diopters / prism diopter); typical 0.3–0.5
                                          # Drives accommodation when vergence is disparity-driven;
                                          # reduces defocus during vergence eye movements.

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
    g_nerve:               jnp.ndarray  = G_NERVE_DEFAULT    # (12,) per-nerve ceiling fraction: clips nerve at g_nerve×_NERVE_MAX
                                                              # 1.0 = transparent (ceiling >> normal burst); <1 = conduction block
                                                              # Nerve order: [LR_L,MR_L,SR_L,IR_L,SO_L,IO_L, LR_R,MR_R,SR_R,IR_R,SO_R,IO_R]
                                                              # INO: g_nerve[MR_L or MR_R] ↓  →  adducting saccades slow, fixation preserved
                                                              # CN6: g_nerve[LR_L or LR_R] ↓  →  abduction limited

    # INO (internuclear ophthalmoplegia) — version_yaw gain for each MR subnucleus.
    # The MLF (ABN → contralateral MR) is modelled as the version component of CN3_MR.
    # Zeroing g_mlf_ver_L cuts version drive to left MR → left eye can't adduct on
    # rightward gaze; vergence (vrg_yaw = +½) is preserved.
    #   g_mlf_ver_L = 0 → left  INO  (right MLF pathway cut: ABN_R → MR_L)
    #   g_mlf_ver_R = 0 → right INO  (left  MLF pathway cut: ABN_L → MR_R)
    g_mlf_ver_L:           float        = 1.0   # version_yaw gain for CN3_MR_L
    g_mlf_ver_R:           float        = 1.0   # version_yaw gain for CN3_MR_R


# ── Initialization ─────────────────────────────────────────────────────────────

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
    x0 = x0.at[_IDX_GRAV].set(ge.X0)   # [G0,0,0, 0,0,0] — upright gravity, zero linear accel
    if brain_params is not None:
        # VS: both populations start at resting bias; null starts at 0.
        # b_vs is (6,) float32 — normalised by simulate() before this is called.
        x0 = x0.at[_IDX_VS_L].set(brain_params.b_vs[:3])
        x0 = x0.at[_IDX_VS_R].set(brain_params.b_vs[3:])
        # _IDX_VS_NULL stays at 0 (no initial adaptation)
        # NI: b_ni populations (b_ni=0 default → stays zero)
        # _IDX_NI_L/R/NULL all stay at 0
        # OPN: initialise to tonic firing rate (100); stays there between saccades.
        x0 = x0.at[_IDX_SG.start + 3].set(100.0)
        # Vergence: initialise x_verg to tonic_verg (H only); e_held/x_copy start at zero.
        x0 = x0.at[_o_vg].set(jnp.float32(brain_params.tonic_verg))
        # Accommodation: initialise fast component at 1 D (1 m target), slow at 0.
        # Warmup will settle both to the correct steady state for the actual target depth.
        x0 = x0.at[_IDX_ACC].set(jnp.array([1.0, 0.0]))
    return x0


# ── Step function ──────────────────────────────────────────────────────────────

def step(x_brain, sensory_out, brain_params, noise_acc=0.0):
    """Single ODE step for the brain subsystem.

    Args:
        x_brain:     (287,)        brain state [x_vs (9) | x_ni (9) | x_sg | x_ec | x_grav | x_pursuit | x_verg]
        sensory_out: SensoryOutput bundled canal afferents + cyclopean delayed signals
                       .canal            (6,)    canal afferent rates
                       .otolith          (3,)    specific force in head frame (m/s²)
                       .scene_slip       (3,)    delayed cyclopean scene angular velocity
                       .scene_linear_vel (3,)    delayed cyclopean scene linear velocity
                       .target_pos       (3,)    delayed cyclopean target position
                       .target_slip      (3,)    delayed cyclopean target velocity
                       .target_disparity (3,)    delayed cyclopean vergence disparity
                       .scene_visible    scalar  delay(scene_present)
                       .target_visible   scalar  delay(target_present × target_in_vf)
                       .target_motion_visible  scalar  delay(target_visible × (1−strobe))
        brain_params: BrainParams   model parameters

    Returns:
        dx_brain: (287,)  dx_brain/dt
        nerves:   (12,)   per-muscle nerve activations [L6 | R6] → plant (split in simulator)
    """
    x_vs      = x_brain[_IDX_VS]      # (9,): bilateral VS + null
    x_ni      = x_brain[_IDX_NI]      # (9,): bilateral NI + null
    x_ni_net  = x_ni[:3] - x_ni[3:6]  # (3,): net eye position (L pop − R pop)
    x_sg      = x_brain[_IDX_SG]
    x_grav    = x_brain[_IDX_GRAV]
    x_head    = x_brain[_IDX_HEAD]
    x_pursuit = x_brain[_IDX_PURSUIT]
    rf_state  = x_grav[ge._IDX_RF]   # (3,) rotational feedback: 1-step delayed, owned by GE
    x_verg    = x_brain[_IDX_VERG]
    x_acc     = x_brain[_IDX_ACC]     # (2,): [x_fast, x_slow] diopters

    # ── Velocity storage + gravity estimator + OCR ───────────────────────────
    # sensory_out.scene_slip is already EC-corrected (pre-delay EC in cyclopean_vision).
    # Ordering: VS runs first (using rf_state from ODE state, 1 ODE step delayed),
    # then GE runs with the accurate w_est from VS this step.
    # rf_state (read from x_grav above) is 1 ODE step delayed — negligible vs gravity dynamics.
    # This breaks the algebraic loop: VS needs rf (from GE), GE needs w_est (from VS).
    dx_vs, w_est   = vs.step(x_vs, jnp.concatenate([sensory_out.canal, sensory_out.scene_slip, rf_state]), brain_params)
    dx_grav, g_est = ge.step(x_grav, jnp.concatenate([w_est, sensory_out.otolith]), brain_params)
    dx_head, v_lin = he.step(x_head, jnp.concatenate([g_est, sensory_out.otolith]), brain_params)
    # OCR: world frame [x=right, y=up, z=fwd]. Right-ear-down → g_est[0] < 0 → -g_est[0] > 0.
    # Positive motor roll = left-ear-down (left-hand rule); negative = right-ear-down (same as head tilt).
    # Partial compensatory: eyes roll right-ear-down when head is right-ear-down (−g_est[0] < 0 → roll < 0).
    ocr            = jnp.array([0.0, 0.0, -brain_params.g_ocr * g_est[0]])

    # ── Pursuit: sensory_out.target_slip already EC-corrected (pre-delay) ────
    dx_pursuit, u_pursuit = pu.step(x_pursuit, sensory_out.target_slip, x_ni_net, brain_params)

    # ── Saccade generator (target selection + Listing's corrections handled internally) ──
    dx_sg, u_burst = sg.step(x_sg, sensory_out.target_pos, sensory_out.target_visible, x_ni_net, ocr[2], w_est, brain_params, noise_acc)

    # ── Neural integrator: VOR + saccades + pursuit → version motor command ───
    # OCR is a tonic position offset (gravity-driven); passed as u_tonic so it is added
    # to the output but not integrated into the NI state (avoids 25 s TC attenuation).
    dx_ni, motor_cmd_ni = ni.step(x_ni, -w_est + u_burst + u_pursuit, brain_params, u_tonic=ocr)

    # ── Accommodation: blur-driven lens adjustment (AC/A and CA/C cross-links disabled) ──
    # TODO: re-enable cross-links once accommodation–vergence interaction is validated.
    dx_acc, _ = acc_mod.step(x_acc, sensory_out.acc_demand, 0.0, brain_params)   # 0.0: CA/C off

    # ── Vergence: disparity + Zee saccade pulse + L2 cyclovergence ────────────
    # bino = tv_L * tv_R ≈ 1 when both eyes fuse, 0 when either covered.
    # z_act: 0=idle (OPN tonic), 1=saccade (OPN paused); normalised from z_opn ∈ [0,100].
    z_act_verg = 1.0 - jnp.clip(x_sg[3], 0.0, 100.0) / 100.0
    dx_verg, u_verg = vg.step(x_verg, sensory_out.target_disparity, 0.0, z_act_verg, x_ni_net[:2], brain_params)

    # ── Final common pathway: nucleus encode → nerve activations ─────────────
    nerves = fcp.step(jnp.concatenate([motor_cmd_ni, u_verg]), brain_params)   # (12,) [L6|R6]

    # ── Efference copy signals for pre-delay EC in cyclopean_vision ──────────
    # Rotation to eye frame is done inside cyclopean_vision.step().
    ec_vel  = u_burst + u_pursuit   # version velocity efference (head frame, [yaw,pitch,roll] deg/s)
    ec_pos  = x_ni_net + ocr        # eye position efference     (head frame, [yaw,pitch,roll] deg)
    ec_verg = x_verg                # vergence efference         ([H,V,T] deg)

    # ── Pack state derivative ─────────────────────────────────────────────────
    dx_brain = jnp.concatenate([dx_vs, dx_ni, dx_sg, dx_grav, dx_head, dx_pursuit, dx_verg, dx_acc])

    return dx_brain, nerves, ec_vel, ec_pos, ec_verg
