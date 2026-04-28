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

import jax.numpy as jnp

from oculomotor.models.sensory_models.retina import velocity_saturation
from oculomotor.models.brain_models import velocity_storage    as vs
from oculomotor.models.brain_models import neural_integrator   as ni
from oculomotor.models.brain_models import saccade_generator   as sg
from oculomotor.models.brain_models import efference_copy      as ec
from oculomotor.models.brain_models import gravity_estimator   as ge
from oculomotor.models.brain_models import pursuit             as pu
from oculomotor.models.brain_models import vergence            as vg
from oculomotor.models.brain_models import accommodation       as acc_mod
from oculomotor.models.brain_models import listing

from oculomotor.models.sensory_models.sensory_model import SensoryOutput, PINV_SENS  # noqa: F401 (re-exported)
from oculomotor.models.plant_models.readout import rotation_matrix
from oculomotor.models.plant_models.muscle_geometry import (
    M_NUCLEUS, M_NERVE_PROJ, G_NUCLEUS_DEFAULT, G_NERVE_DEFAULT,
    CN3_MR_L, CN3_MR_R,
)


# ── State layout ───────────────────────────────────────────────────────────────

N_STATES = vs.N_STATES + ni.N_STATES + sg.N_STATES + 2*ec.N_STATES + ge.N_STATES + pu.N_STATES + vg.N_STATES + acc_mod.N_STATES
#        = 9 + 9 + 9 + 240 + 6 + 3 + 3 + 2 = 281

# ── Index constants — relative to x_brain ─────────────────────────────────────
# Computed from module N_STATES to stay in sync automatically.

_o_vs    = 0
_o_ni    = _o_vs  + vs.N_STATES    # 9
_o_sg    = _o_ni  + ni.N_STATES    # 18
_o_ec    = _o_sg  + sg.N_STATES    # 27   pursuit EC
_o_ec_ok = _o_ec  + ec.N_STATES    # 147  OKR EC
_o_gv    = _o_ec_ok + ec.N_STATES  # 267
_o_pu    = _o_gv  + ge.N_STATES    # 270
_o_vg    = _o_pu  + pu.N_STATES    # 273

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
_IDX_SG      = slice(_o_sg,   _o_sg   + sg.N_STATES)   # (9,)
_IDX_EC      = slice(_o_ec,   _o_ec   + ec.N_STATES)   # (120,) pursuit EC  [27:147]
_IDX_EC_OKR  = slice(_o_ec_ok, _o_ec_ok + ec.N_STATES) # (120,) OKR EC      [147:267]
_IDX_GRAV    = slice(_o_gv,   _o_gv   + ge.N_STATES)   # (3,)               [267:270]
_IDX_PURSUIT = slice(_o_pu,   _o_pu   + pu.N_STATES)   # (3,)               [270:273]
_IDX_VERG    = slice(_o_vg,   _o_vg   + vg.N_STATES)   # (3,)               [273:276]
_o_acc       = _o_vg + vg.N_STATES                     # 276
_IDX_ACC     = slice(_o_acc,  _o_acc  + acc_mod.N_STATES)  # (2,)            [276:278]


# ── Brain parameters ────────────────────────────────────────────────────────────

class BrainParams(NamedTuple):
    """Learnable central parameters — fit to patient eye-movement data."""

    # Velocity storage — Raphan, Matsuo & Cohen (1979)
    # Bilateral push-pull (L/R VN populations); net = x_L − x_R; OKAN decays with tau_vs.
    tau_vs:                float = 20.0   # storage / OKAN TC (s); ~20 s monkey (Cohen 1977)
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
    threshold_sac_release: float = 0.4    # OPN latch release threshold
    tau_reset_fast:        float = 0.05   # inter-saccade x_copy reset TC (s)
    tau_ref:               float = 0.15   # refractory (OPN) decay TC (s); ~150 ms ISI
    tau_ref_charge:        float = 0.001  # OPN charge TC (s)
    k_ref:                 float = 50.0   # bistable OPN gate steepness (1/z_ref)
    threshold_ref:         float = 0.1    # OPN threshold
    tau_hold:              float = 0.005  # sample-and-hold tracking TC (s)
    tau_sac:               float = 0.001  # saccade latch TC (s)
    g_opn_pause:           float = 500.0  # OPN inhibitory overshoot (fr units); IBN inhibition drives OPN
                                           # membrane potential to −g_opn_pause (below spike threshold).
                                           # Firing rate clips at 0; large value → OPN pauses in ~2 ms.
    tau_acc:               float = 0.130  # accumulator rise TC (s); ~80 ms to threshold → ~200 ms RT with visual delay
    tau_drain:             float = 0.200  # accumulator drain TC (s)
    threshold_acc:         float = 0.5    # accumulator trigger threshold
    threshold_sac_qp:      float = 2.0    # error threshold to START accumulating for quick phases (target_visible≈0)
                                           # Higher → requires larger drift error before accumulation begins → fewer spurious resets
    k_acc:                 float = 500.0  # accumulator→OPN sigmoid steepness; high value = near-hard threshold
                                           # OPN only starts dropping in the last ~0.2 ms before z_acc crosses
                                           # threshold_acc, so x_copy barely grows during the OPN transition
    sigma_acc:             float = 0.0    # accumulator diffusion noise (1/√s); adds RT variability
                                           # ~0.15–0.25 gives ±15–25 ms SD; 0 = deterministic

    # Saccade target selection — handled inside the saccade generator
    orbital_limit:         float = 50.0   # oculomotor range half-width (deg); clip e_cmd to ±limit
    alpha_reset:           float = 1.25   # centering gain; e_center = −α·x_ni when out-of-field
    k_center_vel:          float = 0.75   # quick-phase prediction gain (0=no look-ahead, 1=full τ_ref look-ahead)
                                          # look-ahead = k·τ_ref; aims past centre to compensate for slow-phase
                                          # drift during refractory.  Only applied in out-of-field (quick-phase) path.

    # Otolith / gravity estimation — Laurens & Angelaki (2011, 2017)
    K_grav:                float = 0.6    # otolith correction gain for gravity (1/s); go=0.6 Laurens & Angelaki 2011
    K_lin:                 float = 0.1    # linear acceleration adaptation gain (1/s); 0 disables â state
                                          # smaller → more somatogravic effect; larger → faster a_lin adaptation
    K_gd:                  float = 0.0    # gravity dumping gain (1/s); 0 = disabled
    g_ocr:                 float = 0.0    # OCR amplitude (deg); healthy ~10°; 0 = disabled until verified

    # Listing's law — torsional constraint (Listing 1854; Tweed et al. 1998)
    listing_primary:       jnp.ndarray = jnp.zeros(2)  # primary position [yaw₀, pitch₀] (deg)
                                                         # shifts the centre of Listing's plane
                                                         # 0 = straight ahead (healthy default)

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
    tau_verg:              float        = 6.0             # vergence leak TC (s); leaks to phoria [Semmlow 1986: ~5–7 s]
    disp_max_verg_fus:     float        = 2.0             # fusional disparity saturation (deg); Panum's ~±1 deg (Jones 1980)
    disp_max_verg_prox:    float        = 20.0            # proximal disparity saturation (deg); full vergence range (Hung & Semmlow 1980)
    eye_dominant:          float        = 1.0             # 1.0 = right dominant, 0.0 = left dominant
    phoria:                jnp.ndarray  = jnp.zeros(3)    # resting vergence (deg); tonic setpoint; 0=orthophoria
                                                          # phoria[0]>0 esophoria, <0 exophoria

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
    g_nerve:               jnp.ndarray  = G_NERVE_DEFAULT    # (12,) per-nerve gains

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
        b6 = jnp.broadcast_to(jnp.asarray(brain_params.b_vs, dtype=jnp.float32), (6,))
        x0 = x0.at[_IDX_VS_L].set(b6[:3])
        x0 = x0.at[_IDX_VS_R].set(b6[3:])
        # _IDX_VS_NULL stays at 0 (no initial adaptation)
        # NI: b_ni populations (b_ni=0 default → stays zero)
        # _IDX_NI_L/R/NULL all stay at 0
        # OPN: initialise to tonic firing rate (100); stays there between saccades.
        x0 = x0.at[_IDX_SG.start + 7].set(100.0)
        # Vergence: initialise to phoria — tau_verg too slow to settle from zero.
        x0 = x0.at[_IDX_VERG].set(jnp.asarray(brain_params.phoria, dtype=jnp.float32))
        # Accommodation: initialise fast component at 1 D (1 m target), slow at 0.
        # Warmup will settle both to the correct steady state for the actual target depth.
        x0 = x0.at[_IDX_ACC].set(jnp.array([1.0, 0.0]))
    return x0


# ── Binocular perception ───────────────────────────────────────────────────────

class CyclopeanPercept(NamedTuple):
    scene_slip:            jnp.ndarray   # (3,)  OKR / VS drive
    scene_visible:         jnp.ndarray   # ()    scene presence gate
    target_pos:            jnp.ndarray   # (3,)  SG position error (cyclopean)
    target_slip:           jnp.ndarray   # (3,)  pursuit velocity drive
    target_visible:        jnp.ndarray   # ()    target presence gate
    target_motion_visible: jnp.ndarray   # ()    pursuit gate (visible + continuous)
    target_disparity:      jnp.ndarray   # (3,)  vergence disparity (bino-gated)


def _cyclopean_percept(sensory_out, brain_params) -> CyclopeanPercept:
    """Combine per-eye retinal signals into cyclopean percepts.

    Each signal is weighted by per-eye visibility and divided by the number of
    visible eyes so monocular and binocular viewing produce the same amplitude.
    When target disparity exceeds disp_max_verg_fus, the non-dominant eye is
    suppressed and version signals come from the dominant eye only.
    """
    sv_L, sv_R = sensory_out.scene_vis_L,  sensory_out.scene_vis_R
    tv_L, tv_R = sensory_out.target_vis_L, sensory_out.target_vis_R

    # ── Binocular gate — both eyes must see the target ────────────────────────
    bino_raw = tv_L * tv_R   # 1 only when both eyes see target

    # ── Target disparity — vergence drive (uses raw binocular gate, unaffected by suppression) ─
    # Position disparity only. A velocity disparity term (vel_L − vel_R, gated by bino_raw)
    # could be added as a phasic drive (cf. Schor dual-mode model; Hung & Semmlow 1980;
    # Regan & Beverley looming channel). Evidence is weaker than for pursuit: vergence
    # responds faster to ramp than step stimuli, and MT/MST carry disparity-velocity
    # signals (Bradley et al. 1995), but the motor contribution is hard to isolate and
    # position error alone fits most vergence dynamics. Add if ramp responses are too sluggish.
    target_disparity = bino_raw * (sensory_out.pos_L - sensory_out.pos_R)

    # ── Scene — average both eyes by scene visibility (independent of target fusion) ─
    sv_norm       = jnp.maximum(sv_L + sv_R, 1e-6)
    scene_slip    = (sv_L * sensory_out.slip_L + sv_R * sensory_out.slip_R) / sv_norm
    scene_visible = jnp.clip(sv_L + sv_R, 0.0, 1.0)

    # ── Version position and velocity — binocular average (unsuppressed) ────────
    # Direction of version saccades/pursuit uses the cyclopean average regardless of
    # diplopia: in a symmetric vergence step pos_L and pos_R point opposite directions
    # and cancel to zero — suppressing one eye would create a spurious version error.
    tv_norm     = jnp.maximum(tv_L + tv_R, 1e-6)
    target_pos  = (tv_L * sensory_out.pos_L + tv_R * sensory_out.pos_R) / tv_norm
    target_slip = (tv_L * sensory_out.vel_L  + tv_R * sensory_out.vel_R)  / tv_norm

    # ── Diplopia suppression — gates only (not direction) ─────────────────────
    # Smooth gate: fuse=1 when disparity < disp_max_verg_fus (Panum's area), fuse→0 when diplopic.
    # Monocular viewing (bino_raw=0) → disp_mag=0 → fuse=1 → no suppression.
    # Suppression controls WHETHER to trigger (visibility), not WHERE to look (direction).
    #
    # TODO: temporal suppression buildup — replace instantaneous fuse with a low-pass state x_supp:
    #   dx_supp = ((1 - fuse) - x_supp) / tau_supp     tau_supp ~ 0.5 s
    #   suppression = x_supp   (instead of 1 - fuse directly)
    # Rationale: if vergence corrects the disparity within ~200–400 ms (Rashbass & Westheimer
    # 1961 J Physiol), x_supp never builds and both eyes remain visible. If vergence fails
    # (large strabismus, muscle palsy), suppression accumulates over tau_supp and the dominant
    # eye takes over version — consistent with clinical strabismus (von Noorden & Campos 2002).
    # The instantaneous model is adequate for short simulations; tau_supp matters mainly for
    # strabismus modelling where sustained diplopia drives the fixation shift.
    # Hysteresis in fusion (Fender & Julesz 1967 J Opt Soc Am) also implies a temporal process:
    # re-fusion requires smaller disparity than fusion break, suggesting suppression is not
    # reset instantly once it builds.
    fuse   = 1.0 / (1.0 + jnp.exp(10.0 * (jnp.linalg.norm(target_disparity) - brain_params.disp_max_verg_fus)))
    dom_L  = 1.0 - brain_params.eye_dominant
    dom_R  = brain_params.eye_dominant
    tv_L_s = fuse * tv_L + (1.0 - fuse) * tv_L * dom_L   # suppressed visibility
    tv_R_s = fuse * tv_R + (1.0 - fuse) * tv_R * dom_R

    tv_norm_s      = jnp.maximum(tv_L_s + tv_R_s, 1e-6)
    target_visible = jnp.clip(tv_L_s + tv_R_s, 0.0, 1.0)
    strobe         = jnp.clip(
        (tv_L_s * sensory_out.strobe_delayed_L + tv_R_s * sensory_out.strobe_delayed_R) / tv_norm_s,
        0.0, 1.0)
    target_motion_visible = target_visible * (1.0 - strobe)

    return CyclopeanPercept(
        scene_slip=scene_slip, scene_visible=scene_visible,
        target_pos=target_pos, target_slip=target_slip,
        target_visible=target_visible, target_motion_visible=target_motion_visible,
        target_disparity=target_disparity,
    )


# ── Step function ──────────────────────────────────────────────────────────────

def step(x_brain, sensory_out, brain_params, noise_acc=0.0):
    """Single ODE step for the brain subsystem.

    Args:
        x_brain:     (156,)        brain state [x_vs (9) | x_ni (9) | x_sg | x_ec | x_grav | x_pursuit | x_verg]
        sensory_out: SensoryOutput bundled canal afferents + per-eye raw signals
                       .canal          (6,)    canal afferent rates
                       .otolith      (3,)    specific force in head frame (m/s²)
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
    x_ec      = x_brain[_IDX_EC]      # (120,): pursuit EC cascade
    x_ec_okr  = x_brain[_IDX_EC_OKR]  # (120,): OKR EC cascade
    x_grav    = x_brain[_IDX_GRAV]
    x_pursuit = x_brain[_IDX_PURSUIT]
    x_verg    = x_brain[_IDX_VERG]
    x_acc     = x_brain[_IDX_ACC]     # (2,): [x_fast, x_slow] diopters

    # ── Binocular combining — version (average) and vergence (difference) ────────
    percept = _cyclopean_percept(sensory_out, brain_params)

    # ── Two ECs, two corrections with separate clips and gates ───────────────
    # EC cascade inputs are background-shifted: vel_sat(u_motor - bg, v_max) + bg.
    # This centres the clip window on the background retinal slip rather than zero,
    # so the saccade component (u_motor - bg) is clipped at the same effective level
    # as the scene/target velocity cascade, ensuring cancellation holds when the eye
    # is already moving (OKN fast phases, saccades during pursuit).
    #   OKR background  = percept.scene_slip  (delayed scene velocity on retina)
    #   Pursuit bg      = percept.vel_delayed (delayed target velocity on retina)
    motor_ec_pursuit = ec.read_delayed(x_ec)
    motor_ec_okr     = ec.read_delayed(x_ec_okr)

    # ── Velocity storage + gravity estimator + OCR ───────────────────────────
    # Canal velocity for gravity transport (decays with tau_c~5s).
    canal_vel = PINV_SENS @ sensory_out.canal
    # GE convention [x=up, y=interaural(leftward), z=fwd]: w_x=canal[0] (yaw), w_y=-canal[1] (pitch sign flip).
    w_for_ge  = jnp.array([canal_vel[0], -canal_vel[1], canal_vel[2]])
    # sensory_out.otolith = f_gia in world convention x=right, y=up, z=fwd.
    # Convert to GE/VS convention [x=up, y=interaural(leftward), z=fwd]: ge_x=f[1], ge_y=−f[0], ge_z=f[2].
    # At upright: f_gia=[0,G0,0] → f_oto_ge=[G0,0,0]=X0 ✓; left-ear-down: f_oto_ge[1]<0 → OCR<0 ✓.
    f_oto_ge  = jnp.array([sensory_out.otolith[1], -sensory_out.otolith[0], sensory_out.otolith[2]])

    okr      = percept.scene_slip + motor_ec_okr * percept.scene_visible
    g_est_now = x_grav[ge._IDX_G]   # (3,) gravity estimate from GE state (first 3 of 6)
    dx_vs,   w_est = vs.step(x_vs,   jnp.concatenate([sensory_out.canal, okr, g_est_now, f_oto_ge]), brain_params)
    dx_grav, g_est = ge.step(x_grav, jnp.concatenate([w_for_ge, f_oto_ge]), brain_params)
    # OCR: g_est[1] < 0 when left-ear-down (positive roll) → torsion negative = eye rolls left-ear-down.
    ocr            = jnp.array([0.0, 0.0, brain_params.g_ocr * g_est[1]])

    # ── Pursuit: target-gated EC-corrected velocity → pursuit integrator ─────────
    # Strobe gate: when target is strobed, EC is also zeroed — stroboscopic illumination
    # provides no continuous motion signal, so eye-movement EC would create a spurious drive.
    # strobe_delayed matches the timing of the already-zeroed target_slip (same delay cascade).
    # pursuit EC cascade already carries no torsion (zeroed at input, matching retina's 2D surface)
    dx_pursuit, u_pursuit = pu.step(x_pursuit, percept.target_slip, motor_ec_pursuit*percept.target_motion_visible, brain_params)

    # ── Listing's law — inject torsional target into the saccade generator ───
    # The retina cannot sense torsion, so torsional errors from Listing's law
    # (T_required = OCR − H·V·π/360) must be injected explicitly as the [2]
    # component of the SG error signal.
    # x_ni_for_sg has its torsion expressed as the full listing_error relative to
    # T_required (which includes OCR), so the SG aims at zero Listing's error.
    ocr_val     = ocr[2]
    primary_pos = brain_params.listing_primary
    # OCR is added to motor_cmd AFTER the NI step, so x_ni_net[2] does not include
    # the OCR offset. The Listing's calculation needs the full torsion estimate
    # (NI + OCR) so listing_error is zero when the eye is correctly at OCR torsion.
    # Without this, listing_error = -ocr_val at primary position → SG fires spurious
    # torsional saccades that fight the OCR motor command.
    x_ni_net_with_ocr = x_ni_net.at[2].add(ocr_val)
    pos_for_sg, x_ni_for_sg, vel_torsion = listing.corrections(
        x_ni_net_with_ocr,
        (-w_est + u_pursuit)[:2],   # smooth H/V velocity (no saccade burst)
        percept.target_pos,
        ocr_val,
        primary_pos)
    u_pursuit_listing = u_pursuit.at[2].add(vel_torsion)

    # ── Saccade generator (target selection handled internally) ───────────────
    # x_ni_for_sg is the brain's proxy for current eye position (avoids plant state dependency)
    # Use VS STATE (tonic slow phase) for prediction, NOT full w_est which includes the
    # fast canal D-feedthrough (~head velocity during VOR). The D-feedthrough is phasic —
    # a correct compensatory response — not a drift to correct for.
    w_vs_tonic = x_vs[:3] - x_vs[3:6]   # VS state net (tonic imbalance source only)
    dx_sg, u_burst = sg.step(x_sg, pos_for_sg, percept.target_visible, x_ni_for_sg, w_vs_tonic, brain_params, noise_acc)

    # ── Neural integrator: VOR + saccades + pursuit → version motor command ───
    dx_ni, motor_cmd_ni = ni.step(x_ni, -w_est + u_burst + u_pursuit_listing, brain_params)
    # OCR is a tonic position offset (gravity-driven, ~100–500 ms latency via plant TC).
    # Added directly to motor command rather than through the NI (25 s TC) so that
    # dynamic OCR (somatogravic, OVAR) is not attenuated above the NI passband (~0.006 Hz).
    motor_cmd_ni = motor_cmd_ni + ocr

    # ── Accommodation: blur-driven lens adjustment (AC/A and CA/C cross-links disabled) ──
    # TODO: re-enable cross-links once accommodation–vergence interaction is validated.
    dx_acc, _ = acc_mod.step(
        x_acc, sensory_out.acc_demand, 0.0, brain_params)   # 0.0: CA/C off

    # ── Vergence: binocular disparity only (AC/A drive disconnected) ─────────
    # bino = tv_L * tv_R ≈ 1 when both eyes fuse, 0 when either covered.
    # TODO: add saccade vergence interactions from zee's paper: transient vergence 
    # response to saccades, even in darkness (Zee et al. 1987 J Neurophysiol).
    # TODO: add L2: extended listings law, how vergence affects listings plane tilt and therefore saccade torsion. Extended horopter paper
    dx_verg, u_verg = vg.step(x_verg, percept.target_disparity, 0.0, brain_params)   # 0.0: AC/A off

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
    # Rotate (u_burst + u_pursuit) into eye frame before delaying, to match the
    # frame of scene_vel / target_vel computed by retinal_signals (which applies
    # R_gaze_T = R_eye.T @ R_head.T).
    #
    # Correct formula: motor_ec = R_gaze_T @ u_burst  →  scene_slip + motor_ec ≈ 0 ✓
    #     R_gaze_T ≈ R_eye.T   (head stationary during saccades: R_head ≈ I)
    #     R_eye = rotation_matrix(x_ni_net)    (x_ni_net proxies eye rotation vector)
    #
    # Note: do NOT permute x_ni_net before passing to rotation_matrix.  The
    # previous code used _q2rv_ec([yaw, pitch, roll]) = [-pitch, yaw, roll],
    # which created a pitch-axis rotation for a yaw gaze angle — wrong axis —
    # introducing ~sin(gaze)·g_burst of spurious roll into the EC cascade.
    R_gaze_T    = rotation_matrix(x_ni_net).T   # R_eye.T  (head → eye frame)
    u_motor     = R_gaze_T @ (u_burst + u_pursuit)
    u_motor_2d  = u_motor.at[2].set(0.0)   # pursuit EC carries no torsion (retina is 2D)
    dx_ec     = ec.step(x_ec,     velocity_saturation(u_motor_2d, brain_params.v_max_pursuit, v_offset=percept.target_slip), brain_params)
    dx_ec_okr = ec.step(x_ec_okr, velocity_saturation(u_motor,    brain_params.v_max_okr,     v_offset=percept.scene_slip),  brain_params)

    # ── Pack state derivative ─────────────────────────────────────────────────
    dx_brain = jnp.concatenate([dx_vs, dx_ni, dx_sg, dx_ec, dx_ec_okr, dx_grav, dx_pursuit, dx_verg, dx_acc])

    return dx_brain, motor_cmd_L, motor_cmd_R
