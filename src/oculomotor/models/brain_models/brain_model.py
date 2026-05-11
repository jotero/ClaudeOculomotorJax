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

EC delay cascades live inside the cerebellum module (cb), since they ARE
the cerebellum's forward-model output (a delayed prediction of the retinal
self-motion contribution).  Two paths run in parallel:
    acts.cb.ec_scene  = delayed EC matched to scene_angular_vel cascade
    acts.cb.ec_target = delayed EC matched to target_vel cascade

    OKR / VS correction  — scene-gated (full scene slip):
        okr = scene_visible · (slip + acts.cb.ec_scene)
        slip ≈ −(u_burst+u_pursuit)(t−τ)  →  corrected ≈ 0  ✓

    Pursuit cerebellar drive — target-gated trust signal applied internally:
        cerebellum exposes acts.cb.drive (gated forward-model prediction);
        pursuit input = K_pursuit_direct·target_slip + acts.cb.drive
        Implicit cancellation of self-motion via target_slip ≈ −ec_target
        (delayed by the same cascade).

Vergence (Schor 1986 dual integrator + Robinson direct phasic path):
    e_disp = pos_L − pos_R   (binocular disparity, deg)
    u_phasic = K_phasic_verg · e_disp                     (deg/s, no state)
    dx_v       = −x_v / τ_verg + K_verg · e_disp                       (vergence integrator)
    dx_v_tonic = −x_v_tonic / τ_verg_tonic + K_verg_tonic · e_disp     (tonic vergence integrator)
    u_verg   = tonic_verg + x_fast + x_slow + τ_vp · u_phasic
    motor_cmd_L = motor_cmd_version + ½ · u_verg   (L eye converges rightward)
    motor_cmd_R = motor_cmd_version − ½ · u_verg   (R eye converges leftward)

Cerebellum cascade advance (end of step):
    dcb = cb.step(brain_state.cb, ec_vel, ec_pos, brain_params)
    where ec_vel = u_burst + u_pursuit + omega_tvor (version velocity command).

Internal flow:
    VS  →  w_est  →  −w_est + u_burst + u_pursuit  →  NI  →  motor_cmd_version
    SG  →  u_burst    (saccade burst → cerebellum EC cascade)
    Pursuit → u_pursuit  (→ NI + cerebellum EC cascade)
    cerebellum → delays (u_burst + u_pursuit + omega_tvor) by tau_vis;
                  ec_scene used by VS (scene path);
                  ec_target used by pursuit forward model (drive output)
    GE  →  g_est (gravity estimate, cross-product dynamics)
    Vergence → u_verg → split ±½ to L/R motor commands

Brain state — `BrainState` NamedTuple, perception → motor order:
    pc   — perception_cyclopean (binocular fusion + brain LP)
    sm   — self-motion observer (VS bilateral pops + GE + HE)
    pt   — target perception working memory
    sg   — saccade generator (e_held + OPN + accumulators + EBN/IBN)
    pu   — bilateral pursuit pops
    va   — vergence + accommodation
    ni   — bilateral NI pops + null adaptation
    fcp  — motor neurons (final common pathway)
    cb   — cerebellum (EC cascades + pursuit forward model)

Each subsystem owns its own `State` NamedTuple; `BrainState` aggregates them
under named fields.  No flat-array layout — read fields directly via
`brain_state.<sub>.<field>`.  `N_STATES = 163` is kept for legacy info only.

Outputs of step():
    dbrain      BrainState  state derivative
    nerves      (12,)       per-muscle nerve activations [L6 | R6] → plant
    ec_vel      (3,)        version velocity efference (head frame, deg/s)
    ec_pos      (3,)        eye position efference (= NI net)
    ec_verg     (3,)        vergence efference (full vergence command)
    u_acc       scalar      total lens-plant input (D)
"""

from typing import NamedTuple

import jax.numpy as jnp

from oculomotor.models.brain_models  import perception_self_motion as sm
from oculomotor.models.brain_models  import perception_target      as pt
from oculomotor.models.brain_models  import perception_cyclopean   as pc
from oculomotor.models.brain_models  import vergence_accommodation as va
from oculomotor.models.brain_models  import neural_integrator      as ni
from oculomotor.models.brain_models  import saccade_generator      as sg
from oculomotor.models.brain_models  import pursuit                as pu
from oculomotor.models.brain_models  import tvor                   as tv
from oculomotor.models.brain_models  import final_common_pathway   as fcp
from oculomotor.models.brain_models  import cerebellum               as cb
from oculomotor.models.brain_models  import listing

from oculomotor.models.sensory_models.sensory_model import SensoryOutput
from oculomotor.models.brain_models.final_common_pathway import G_NUCLEUS_DEFAULT, G_NERVE_DEFAULT
from oculomotor.models.plant_models.muscle_geometry           import R_BASELINE_DEFAULT
# EC cascade math (rotation, saturation, cascade_lp_step) and the cascade
# state both live in `cerebellum.py` now — same module that owns the pursuit
# forward model (which uses the cascade tails as its prediction).


# ── State layout ───────────────────────────────────────────────────────────────
# Brain state is a nested NamedTuple (BrainState below) — no flat array, no
# `_IDX_*` slice constants.  Each subsystem owns its `State` NT; this module
# aggregates them under named fields.

N_STATES = (sm.N_STATES + ni.N_STATES + sg.N_STATES + pu.N_STATES + va.N_STATES
            + pt.N_STATES + pc.N_STATES + cb.N_STATES + fcp.N_STATES)
#        = 21 + 9 + 18 + 3 + 11 + 4 + 43 + 21 + 21 + 12 = 163  (kept for legacy info)


# ── Brain state (nested NamedTuple) ───────────────────────────────────────────
# Each subsystem owns its `State` NamedTuple.  `BrainState` aggregates them
# under per-subsystem fields, eliminating the flat-array layout and the
# `_IDX_*` slice constants.
#
# Diffrax handles arbitrary PyTrees natively — `BrainState` is just a deeper
# tree.  EC cascades (ec_scene, ec_target) stay as flat arrays since they're
# pure delay buffers with no internal sub-structure to surface.

class BrainState(NamedTuple):
    """Top-level brain state — one field per subsystem.

    Ordered perception → cognition → motor → cerebellum:
        pc   perception_cyclopean (binocular fusion + brain LP)
        sm   self-motion observer (VS + GE + HE)
        pt   target working memory (FEF/dlPFC)
        sg   saccade generator
        pu   smooth pursuit
        va   vergence + accommodation
        ni   neural integrator (final premotor stage)
        fcp  motor neurons (final common pathway)
        cb   cerebellum (EC delay cascades + pursuit forward model)
    """
    # Perception
    pc:   pc.State              # perception_cyclopean brain LP
    sm:   sm.State              # self-motion (VS + GE + HE)
    pt:   pt.State              # target perception working memory
    # Motor planning + execution
    sg:   sg.State              # saccade generator
    pu:   pu.State              # smooth pursuit
    va:   va.State              # vergence + accommodation
    ni:   ni.State              # neural integrator
    fcp:  fcp.State             # motor neurons
    # Cerebellum (subsumes the prior efference_copy module: EC cascades +
    # pursuit forward-model prediction-error correction)
    cb:   cb.State              # scene + target EC cascades


def rest_brain_state():
    """Initial BrainState — all zeros for cascade buffers and pops."""
    return BrainState(
        pc   = pc.rest_state(),
        sm   = sm.rest_state(),
        pt   = pt.rest_state(),
        sg   = sg.rest_state(),
        pu   = pu.rest_state(),
        va   = va.rest_state(),
        ni   = ni.rest_state(),
        fcp  = fcp.zero_state(),
        cb   = cb.rest_state(),
    )


# ── Phase-2 registries (Activations / Decoded / Weights) ─────────────────────
# Aggregator over per-module registries.  Each subsystem owns its local
# Activations / Decoded / Weights NamedTuples and reader functions; this
# module just collects them under named subsystem fields.

class Activations(NamedTuple):
    """Brain-wide firing rates — perception → motor order."""
    # Perception
    pc:  pc.Activations    # cyclopean cascade-tail delayed signals
    sm:  sm.Activations    # VS + GE + HE
    pt:  pt.Activations    # target memory
    # Motor
    sg:  sg.Activations    # saccade generator
    pu:  pu.Activations    # bilateral pursuit
    va:  va.Activations    # vergence + accommodation
    ni:  ni.Activations    # bilateral NI
    fcp: fcp.Activations   # motor neurons
    # Cerebellum (stateless prediction-error correction; activations are
    # input-driven, computed from BrainState pc + ec at the top of step)
    cb:  cb.Activations    # pursuit pred_err / drive / gate


class Decoded(NamedTuple):
    """Brain-wide push-pull decoded nets — perception → motor order."""
    sm: sm.Decoded   # vs_net  (perception)
    pu: pu.Decoded   # pu_net  (motor)
    ni: ni.Decoded   # ni_net  (motor)


class Weights(NamedTuple):
    """Brain-wide tonic/null/setpoint registers (long-term: learned weights)."""
    sm: sm.Weights   # vs_null
    ni: ni.Weights   # ni_null
    sg: sg.Weights   # e_held


def read_activations(brain_state, brain_params):
    """Aggregate per-module activations into the brain-wide registry.

    Single canonical state→firing-rate projection; called once per ODE step.

    `brain_params` is forwarded only to subsystems whose activations require
    parameter values (currently just `cb`, which is stateless and computes
    its activations from BrainState pc + ec via state-derived inputs).  All
    other subsystems are pure state→firing-rate projections.
    """
    return Activations(
        pc  = pc.read_activations(brain_state.pc),
        sm  = sm.read_activations(brain_state.sm),
        pt  = pt.read_activations(brain_state.pt),
        sg  = sg.read_activations(brain_state.sg),
        pu  = pu.read_activations(brain_state.pu),
        va  = va.read_activations(brain_state.va),
        ni  = ni.read_activations(brain_state.ni),
        fcp = fcp.read_activations(brain_state.fcp),
        cb  = cb.read_activations(brain_state, brain_params),
    )


def decode_activations(acts):
    """Aggregate per-module decoded readouts (push-pull L−R / R−L nets).

    Pure function of `acts` — no raw state involvement.
    """
    return Decoded(
        sm = sm.decode_states(acts.sm),
        pu = pu.decode_states(acts.pu),
        ni = ni.decode_states(acts.ni),
    )


def read_weights(brain_state):
    """Aggregate per-module tonic/null/setpoint registers."""
    return Weights(
        sm = sm.read_weights(brain_state.sm),
        ni = ni.read_weights(brain_state.ni),
        sg = sg.read_weights(brain_state.sg),
    )


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
    tau_vs:                float = 20.0   # yaw (main) VS TC (s); Cohen 1977 monkey ~20 s
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
    # Note: canal afferent saturation (was v_max_vor here) lives in
    # SensoryParams.canal_v_max — sensor-side ceiling, applied in canal.step.
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

    # EC contribution to the post-delay slip — pure prediction error.
    #
    #     PE_scene  = scene_slip  + scene_visible  · ec_d_scene   → VS / OKR
    #     PE_target = target_slip + target_visible · ec_d_target  → pursuit
    #
    # No multiplicative gain, no saccadic-suppression Hill — just the
    # classical Robinson/Smith formulation.  The visibility scalar keeps the
    # EC contribution zero when the scene/target was invisible.

    # Neural integrator — bilateral push-pull + null adaptation (Robinson 1975; rebound: Zee et al. 1980)
    # Per-axis TCs via fractions of tau_i (matches the VS pattern). Yaw uses tau_i directly;
    # torsional NI in the INC is reported as substantially leakier (Crawford & Vilis 1991:
    # ~5–10 s vs 20–25 s for horizontal NPH; Suzuki et al. 1995; Anastasopoulos & Mergner 1982).
    tau_i:                 float = 25.0   # yaw leak TC (s); healthy >20 s (Cannon & Robinson 1985)
    tau_i_pitch_frac:      float = 1.0    # pitch TC = tau_i × this  → 25 s (vertical NI similar to horizontal)
    tau_i_roll_frac:       float = 0.3    # roll  TC = tau_i × this  → 7.5 s (Crawford & Vilis 1991)
    tau_p:                 float = 0.15   # plant TC copy — NI feedthrough for lag cancellation
    tau_vis:               float = 0.08   # visual delay copy — EC delay must match retinal delay
                                          # should match PlantParams.tau_p in healthy subjects;
                                          # may differ in pathology (imperfect internal model)
    # Internal model copies of the sensory cascade shape used by the post-delay
    # EC subtraction in brain_model.step. cascade_lp_step on ec_vel_eye must match
    # the matched perception_cyclopean LP cascade for clean cancellation.
    tau_vis_sharp:              float = 0.05   # sharp cascade mean delay (s); 6 stages × this/6
                                                # Shared by both scene and target EC cascades.
    tau_vis_smooth_motion:      float = 0.02   # smoothing LP TC (s) for SCENE EC cascade.
                                                # Must match SensoryParams.tau_vis_smooth_motion so
                                                # the EC and slip cascades have the same shape →
                                                # clean post-delay subtraction in brain_model.step.
    tau_vis_smooth_target_vel:  float = 0.075  # smoothing LP TC (s) for TARGET EC cascade.
                                                # Must match SensoryParams.tau_vis_smooth_target_vel.
                                                # Heavier smoothing than scene because the visual
                                                # target-velocity pathway is intrinsically slower
                                                # (Krauzlis & Lisberger 1994).
    tau_vis_smooth_disparity:   float = 0.15   # smoothing LP TC (s) for target_disparity cyclopean LP.
                                                # V1 stereo correspondence is genuinely slow (~150 ms;
                                                # Cumming & DeAngelis 2001).
    tau_vis_smooth_defocus:     float = 0.20   # smoothing LP TC (s) for defocus cyclopean LP.
                                                # Sloppy accommodation channel; combined with the lens
                                                # plant TC (Schor & Bharadwaj 2006) produces realistic
                                                # sluggish open-loop accommodation responses.
    tau_brain_pos:              float = 0.015  # N-stage gamma TC (s) for target_pos / visibility brain
                                                # phase (post-fusion). Total visual mean delay for these
                                                # signals ≈ tau_vis_sharp + tau_brain_pos = 0.08 s.

    # Binocular fusion policy (perception_cyclopean.binocular_fusion_policy)
    # Cortical decisions about fusion limits and eye dominance — moved here
    # because they're brain-side, not peripheral sensor properties.
    npc:                        float = 50.0   # near point of convergence (deg); convergence motor limit
                                                # 50° ≈ NPC 7 cm (IPD=64 mm); physiological 40–55° young adults
    div_max:                    float = 6.0    # maximum divergence (deg); ~6° young adults
    vert_max:                   float = 5.0    # maximum vertical vergence ±(deg); ~3–5° clinical range
    tors_max:                   float = 8.0    # maximum cyclovergence ±(deg); ~5–8° max
    eye_dominant:               float = 1.0    # 1.0 = right dominant, 0.0 = left dominant
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

    # Burst-neuron dynamic gain (short-term facilitation + depression).
    # g_dyn = 1 + alpha_fac·z_fac − alpha_dep·z_dep multiplies the BN drive
    # (relu(±e_held)). z_fac tracks (1 − act_opn/100) with τ_fac (rises during
    # OPN pause, decays after); z_dep follows z_fac with τ_dep (slower) so a
    # post-saccadic dip in g_dyn persists into the inter-saccadic interval.
    # Defaults alpha_fac=alpha_dep=0 → g_dyn≡1, feature OFF (existing benches
    # unaffected). Heun stability: dt/τ_fac = 0.05, dt/τ_dep = 0.01.
    tau_fac:               float = 0.020  # BN facilitation rise/decay TC (s)
    tau_dep:               float = 0.100  # BN depression follower TC (s) — sets post-saccadic recovery duration
    alpha_fac:             float = 1.0    # facilitation gain: g_dyn = 1 + alpha_fac·z_fac during burst —
                                            # post-inhibitory-rebound boost of the burst neurons (z_fac rises
                                            # toward 1 while OPN is paused).  ~1.0 steepens the main sequence
                                            # toward 700(1−e^{−A/7}) without de-stabilising the burst loop.
    alpha_dep:             float = 0.0    # depression gain: g_dyn dips by alpha_dep × z_dep after burst

    # Saccade target selection — handled inside the saccade generator
    orbital_limit:         float = 50.0   # oculomotor range half-width (deg); clip e_cmd to ±limit
    alpha_reset:           float = 1.25   # centering gain; e_center = −α·x_ni when out-of-field
    k_center_vel:          float = 0.75   # quick-phase prediction gain (0=no look-ahead, 1=full τ_ref look-ahead)
                                          # look-ahead = k·τ_ref; aims past centre to compensate for slow-phase
                                          # drift during refractory.  Only applied in out-of-field (quick-phase) path.

    # Otolith / gravity estimation — Laurens & Angelaki (2011, 2013, 2017)
    # Values held to Laurens-published numbers, no hand-tuning compromises.
    K_grav:                float = 0.6    # somatogravic gain "go" — Laurens & Angelaki (2011) Table 1.
                                          # Sets corner frequency f_c = K_grav/(2π) ≈ 0.095 Hz for
                                          # tilt-percept commitment.
    K_lin:                 float = 0.1    # linear-acc adaptation gain — Laurens & Angelaki (2011) "ka".
                                          # Static value (canal-gating below modulates it state-dependently).
    w_canal_gate:          float = 2.0    # canal-gating threshold (deg/s) — Laurens 2017 Bayesian
                                          # disambiguation extension. Used by the dynamic gate
                                          # ρ = |ω × ĝ| / w_canal_gate; K_grav_eff = K_grav · √(1+ρ),
                                          # K_lin_eff = K_lin / √(1+ρ). Lower threshold → gating
                                          # engages at slower rotations.
                                          # Set to 1e6 to disable for pure Laurens-2011 testing.
    tau_a_lin:             float = 1.5    # translation-duration prior τ_a — Laurens, Meng & Angelaki (2013).
    K_gd:                  float = 2.86   # gravity dumping gain — Laurens & Angelaki (2011) 0.05 rad/s
                                          # converted to deg/s units (0.05 × 180/π ≈ 2.86). Drives VS dumping
                                          # ⊥ gravity (tilt suppression) and OVAR sustained nystagmus.
                                          # 0 = disabled / gravity-blind VS.
    g_ocr:                 float = 1.019  # OCR gain (deg/(m/s²)): 10°/9.81 → ~10° torsion at 90° lateral tilt
                                          # (Howard & Templeton 1966; Diamond, Markham et al. 1979).
                                          # 0 = disabled; useful for benches that want eye torsion off.

    # Heading estimator — linear velocity in head-fixed frame
    tau_head:              float = 2.0    # linear velocity integration TC (s).  Shorter τ trades off
                                          # sustained-motion DARK T-VOR gain (~0.5 of LIT for a 1.6 s
                                          # pulse) against post-pulse drift containment.  Matches
                                          # Paige & Tomko (1991) DARK T-VOR gain of ~0.5–0.7 and
                                          # bounds the closed-loop drift from VS-OKR contamination
                                          # of the gravity transport input.
    K_he_vis:              float = 5.0    # visual-velocity pull gain into v_lin (1/s); fuses scene
                                          # translational flow with vestibular integration.  In dark
                                          # (scene_lin_vel=0) the pull is toward zero, suppressing
                                          # drift during pure rotation (OVAR, tilt suppression).

    # Listing's law — torsional constraint (Listing 1854; Tweed et al. 1998)
    listing_primary:       jnp.ndarray = jnp.zeros(2)  # primary position [yaw₀, pitch₀] (deg)
                                                         # shifts the centre of Listing's plane
                                                         # 0 = straight ahead (healthy default)
    listing_l2_frac:       float = 0.0    # L2 cyclovergence fraction (0=off, 0.5=physiological)
                                          # Listing's plane tilts ±l2_frac·φ/2 per eye with vergence
                                          # Disabled until validated with binocular torsion data
    listing_gain:          float = 1.0    # master gain on the Listing's corrections (cyc_torsion_vel
                                          # and cyclo_verg_rate). Set to 0 to disable both for debugging.

    # Smooth pursuit — leaky integrator + Smith predictor (Lisberger 1988)
    K_pursuit:             float = 4.0    # pursuit integration gain (1/s); rise TC ≈ 1/K_pursuit
    K_phasic_pursuit:      float = 5.0    # pursuit direct feedthrough (dim'less); fast onset
    tau_pursuit:           float = 40.0   # pursuit-pop open-loop leak TC (s).  The
                                            # behaviorally observed pursuit memory TC is the
                                            # CLOSED-loop tau_eff = tau·(1+K_ph) / [(1+K_ph)+K_pursuit·tau],
                                            # which at defaults (K_pursuit=4, K_ph=5) ≈ 1.45 s
                                            # — matches Bennett & Barnes (2003) extra-retinal
                                            # velocity-memory τ ≈ 1–3 s.  Long `tau_pursuit` here
                                            # keeps steady-state pursuit gain near unity (~99%)
                                            # without affecting the closed-loop decay shape.
    v_max_pursuit:         float = 40.0   # MT/MST velocity saturation (deg/s); clip on target slip + EC
                                           # Pursuit gain ≈1 up to ~30–40 deg/s, then falls (Fuchs 1967 J Physiol;
                                           # Lisberger & Westbrook 1985 J Neurosci). MT tuned 10–64 deg/s
                                           # (Newsome et al. 1988 J Neurosci); 40 deg/s is conservative.


    # Vergence — Schor (1986) dual integrator + Robinson (1975) direct phasic path
    # Structure: disparity → fast int + slow int + direct path (τ_vp·K_phasic) → plant
    # Step 1 of rebuild — Zee saccade burst (Step 2) parameters added later.
    # References: Schor & Kotulak 1986; Read & Schor 2022; Rashbass & Westheimer 1961
    # Direct phasic path provides the ~150–200 ms fast onset (Rashbass & Westheimer 1961
    # J Physiol 159:339; achieved via plant-cancellation, not via integrator dynamics).
    # Schor 1999 Table 1 (vergence column) — TCs and dimensionless K gains.
    #   Kb (proportional/direct) = 1 (NI-matched, NOT Schor's 1.5 — we use NI's
    #     pure pass-through form: direct contribution = τ_p · rate_drive).
    #   Kf (phasic) = 2.5, Ks (tonic) = 1.5  (Schor 1999 Table 1)
    #   Tf (phasic) = 5 s, Ts (tonic) = 20 s
    K_phasic_verg:         float        = 1.0             # Kb (direct path); set to NI unit gain
    K_verg:                float        = 1.25            # vergence integrator gain (was 2.5; halved to compensate
                                                            # for the doubled disparity cascade delay so loop-gain·delay
                                                            # product stays in a stable closed-loop range)
    K_verg_tonic:          float        = 1.5             # tonic vergence integrator gain (was K_verg_slow / Ks in Schor 1999)
    tau_verg:              float        = 5.0             # vergence integrator TC (s) (was tau_verg_fast / Tf in Schor 1999)
    tau_verg_tonic:        float        = 20.0            # tonic vergence integrator TC (s) (was tau_verg_slow / Ts in Schor 1999)
    tau_vp:                float        = 0.15            # legacy alias of tau_p; va.step uses brain_params.tau_p directly
    tonic_verg:            float        = 3.67            # tonic (brainstem) vergence baseline (deg); resting dark vergence
                                                          # = 2·arctan(IPD/2 / 1 m); recomputed from IPD in default_params()
                                                          # Riggs & Niehl 1960, Morgan 1944: dark vergence ≈ 1 m
                                                          # NOTE: phoria is a *measurement* (cover test outcome), not a model param;
                                                          # it emerges from the balance of tonic, AC/A, and fusional drives
    proximal_d:            float        = 0.0             # perceived-distance prior (D = 1/m). Drives BOTH
                                                          # vergence (residual = proximal·IPD·180/π − u_verg[H])
                                                          # and accommodation (residual = proximal − u_acc) toward
                                                          # a perceived-near target. The fast integrators absorb
                                                          # the residual; vision wins when present (closed-loop).
                                                          # Distinct from motor priors tonic_verg / tonic_acc:
                                                          # proximal_d is a perceptual-cue parameter (HMD, instrument
                                                          # myopia, awareness-of-near). 0 = no proximal effect.
    # Zee (1992) SVBN saccadic vergence burst — asymmetric saturating gain
    # y = sign(disp) · g · (1 − exp(−|disp|/X))     (deg/s); fires while OPN pauses (z_act≈1)
    # Convergence much stronger than divergence (Zee Table 1: ~50°/s vs ~12°/s peaks).
    # Set g_svbn_conv = g_svbn_div = 0 to disable the burst (slow vergence only).
    # Zee (1992) Table 1: pure conv peak velocity 41–58 °/s for 10° amplitude;
    # pure div 9.5–13 °/s for 2.5°. Saturating exponential y = g·(1 − exp(−|disp|/X))
    # tuned so y(10°, conv) ≈ 60·(1−e^{-10/3}) ≈ 58 °/s and y(2.5°, div) ≈ 14·(1−e^{-2.5/1}) ≈ 13 °/s.
    g_svbn_conv:           float        = 30.0            # convergence asymptotic burst velocity (deg/s)
    X_svbn_conv:           float        = 3.0             # convergence saturating scale (deg)
    g_svbn_div:            float        = 14.0            # divergence  asymptotic burst velocity (deg/s)
    X_svbn_div:            float        = 1.0             # divergence  saturating scale (deg)
    g_burst_verg:          float        = 0.0             # legacy: kept for bench compatibility; set 0 (SVBN replaces)

    # Translational VOR — Paige & Tomko (1991), Angelaki et al., Laurens & Angelaki (2017)
    # Reuses heading_estimator's v_lin (gravity-corrected linear velocity, τ_head ≈ 2 s)
    # as the vestibular estimate; fuses with visual translational scene flow via constant
    # Kalman-like weights; divides by vergence-derived 1/distance for compensatory eye
    # velocity. Output added to NI input alongside −w_est (canal-VOR). A soft tanh
    # saturation caps each output axis at ω_max_tvor — ample headroom for any real
    # T-VOR response, prevents runaway from gravity-mismatch artifacts.
    # Vest+visual fusion of head linear velocity is now done in heading_estimator
    # (K_he_vis); T-VOR consumes the combined v_lin directly.
    K_phasic_tvor:         float        = 1.0             # T-VOR direct (phasic) pathway gain (s) — multiplies a_lin in the
                                                          # cross-product formula to give a fast onset velocity that bypasses
                                                          # the heading-estimator (v_lin) integrator. Implicit units: s, since
                                                          # it converts a_lin (m/s²) into a v-equivalent (m/s) for the cross
                                                          # product. ~0.1–0.3 matches the ~100 ms short-latency T-VOR onset
                                                          # reported by Paige & Tomko (1991), Angelaki et al. (1999).
                                                          # Set 0 to disable direct pathway (integrator-only T-VOR).
    g_tvor:                float        = 1.0             # T-VOR version output gain (cross product); ~unity per Paige & Tomko 1991.
                                                          # Was lowered to 0.5 while debugging tilt-suppression with K_gd off and
                                                          # the old position-integrator TVOR architecture; restored now that
                                                          # K_gd default is non-zero and the NPC gate + velocity-output TVOR
                                                          # prevent the runaway that originally motivated halving.
    g_tvor_verg:           float        = 0.4             # T-VOR vergence-rate gain (dot product · IPD/D²); kept below unity
                                                          # because vergence drift from sustained-tilt v_lin leak is harder to
                                                          # contain than the version output. NPC gate handles overshoot but
                                                          # the steady-state drive can still pull vergence away from tonic.
    g_tvor_l2_cyclo:       float        = 0.0             # L2 cyclo-vergence cross-coupling gain in T-VOR.  Set to 0 because the
                                                          # exact geometry creates a positive feedback loop with FCP's linear
                                                          # Hering's during head tilts (OCR cascade roll cyclo-vergence drift).
                                                          # Re-enable when FCP uses rotation-matrix Hering's.
    tau_tvor_pos:          float        = 2.0             # T-VOR low-pass TC for v_lin (s); short enough to track translation transients,
                                                          # long enough to smooth out canal noise / quick-phase chatter
    ipd_brain:             float        = 0.064           # interpupillary distance (m) used for vergence→distance calculation;
                                                          # mirror of SensoryParams.ipd — kept here so brain_model.step can convert
                                                          # vergence angle to distance for T-VOR without threading sensory_params through
    distance_npc:          float        = 0.08            # near point of convergence (m); anatomical floor on T-VOR distance.
                                                          # Below this depth no real target can be fused, so T-VOR distance is
                                                          # clipped here to prevent 1/D blowup as vergence wanders past NPC.
                                                          # Clinical Sheard / Gulick range 6–10 cm; default 8 cm.


    # Accommodation — Schor dual-interaction model + plant dynamics
    # Read, Kaspiris-Rousellis et al. (2022) J Vision 22(9):4; Schor (1979)
    # Neural controller: dual integrators (fast + slow) → plant (lens/ciliary muscle)
    tonic_acc:             float = 1.0    # dark-focus resting level (D); ~1 D ≈ 1 m for young adults
                                          # x_fast and x_slow represent DEVIATIONS from this baseline.
                                          # Calibrate to match: tonic_acc ≈ IPD / (2·tan(tonic_verg·π/360))
    # Schor 1999 Table 1 (accommodation column) — Kf, Ks, TCs;
    # plant kept at 0.156 s (Schor & Bharadwaj 2006).
    #   Kb (direct/proportional) = 1 (implicit, defocus added directly to output)
    #   Kf (phasic) = 2.5, Ks (tonic) = 1.5
    #   Tf (phasic) = 5 s, Ts (tonic) = 20 s
    tau_acc_plant:         float = 0.156  # lens / ciliary muscle TC (s) [Schor & Bharadwaj 2006]
    tau_acc_fast:          float = 5.0    # phasic Tf (s) [Schor 1999 Table 1]
    tau_acc_slow:          float = 20.0   # tonic  Ts (s) [Schor 1999 Table 1]
    K_acc_fast:            float = 2.5    # Kf (phasic gain) [Schor 1999 Table 1]
    K_acc_slow:            float = 1.5    # Ks (tonic gain)  [Schor 1999 Table 1]
    AC_A:                  float = 5.0    # AC/A ratio (prism diopters / diopter); typical 4–6
                                          # At 40 cm (2.5 D): AC/A drive ≈ 5×0.573×2.5 ≈ 7.2°
                                          # At   6 m (0.17 D): drive ≈ 0.5° — explains why
                                          # IXT decompensates preferentially at distance.
    CA_C:                  float = 0.08   # CA/C ratio (diopters / prism diopter). Literature value
                                          # (Schor & Kotulak 1986: 0.5 D/MA × 1 MA / 6.4 pd at IPD 64 mm;
                                          # consistent with Schor 1992, Daum 1983). Behavioral effect is
                                          # small in our model because the closed defocus loop clamps
                                          # accommodation near optical demand — without an open-loop
                                          # mechanism (pinhole/DoG), measured CA/C is suppressed ~10×
                                          # CA/C only contributes a small, hard-to-measure perturbation.
                                          # When enabled (≈0.08): drives accommodation when vergence is
                                          # disparity-driven; AC_A·CA_C ≈ 0.4 D/D < 1 keeps the cross-loop stable.
    refractive_error:      float = 0.0   # patient refractive error (diopters); >0 hyperopia, <0 myopia
                                          # Added to 1/z before defocus = 1/z + RE − x_plant.
                                          # Hyperope needs more accommodation at every distance;
                                          # myope needs less (natural far point = 1/|RE| m for myopia).

    # Motor nucleus and nerve gains — two-stage encode (see muscle_geometry.py)
    # Stage 1 — g_nucleus (12,): per-nucleus gain [0,1]. Zero = nucleus lesion.
    #   Nucleus order: ABN_L(0), ABN_R(1), CN4_L(2), CN4_R(3),
    #                  CN3_MR_L(4), CN3_MR_R(5), CN3_SR_L(6), CN3_SR_R(7),
    #                  CN3_IR_L(8), CN3_IR_R(9), CN3_IO_L(10), CN3_IO_R(11)
    #   ABN gain is shared with the co-located AIN (abducens internuclear neurons)
    #   on the same side: any real abducens-nucleus lesion silences BOTH
    #   ipsilateral LR motoneurons AND the MLF outflow to contralateral MR
    #   (textbook horizontal gaze palsy — neither eye saccades to that side).
    # Stage 2 — g_nerve (12,): per-nerve gain [0,1]. Zero = nerve/fascicular lesion.
    #   Nerve order: [LR_L,MR_L,SR_L,IR_L,SO_L,IO_L, LR_R,MR_R,SR_R,IR_R,SO_R,IO_R]
    #   CN nerve lesion isolates individual muscles without affecting other nuclei.
    # Healthy default: all ones → transparent round-trip through plant.
    g_nucleus:             jnp.ndarray  = G_NUCLEUS_DEFAULT  # (12,) motor nucleus gains (one per side)
    g_nerve:               jnp.ndarray  = G_NERVE_DEFAULT    # (12,) per-nerve ceiling fraction: clips nerve at g_nerve×_NERVE_MAX
    # Per-nucleus tonic baseline firing rate (deg/s equiv).  Symmetric default
    # (50 across all 12) gives zero plant effect (uniform → zero-sum decode).
    # Asymmetric values produce tonic strabismus without lesioning gains:
    #   r_baseline[ABN_R] > r_baseline[ABN_L]  →  R eye drifts abducted (exo)
    #   r_baseline[CN3_MR_R] > r_baseline[CN3_MR_L]  →  R eye drifts adducted
    # Indices match g_nucleus: [ABN_L, ABN_R, CN4_L, CN4_R, CN3_MR_L, CN3_MR_R,
    # CN3_SR_L, CN3_SR_R, CN3_IR_L, CN3_IR_R, CN3_IO_L, CN3_IO_R]
    r_baseline:            jnp.ndarray  = R_BASELINE_DEFAULT  # (12,) tonic firing rate at primary position
    tau_mn:                float        = 0.005              # MN membrane TC (s); per-nerve low-pass on the
                                                              # smooth-clipped brainstem drive. ~5 ms matches
                                                              # oculomotor MN membrane TCs (Robinson 1981; Sylvestre
                                                              # & Cullen 1999). Behaviorally invisible at healthy
                                                              # plant TCs (~150–200 ms), but a clean state-level
                                                              # hook for MN-pathology modelling (e.g. fatigue).
    # Cerebellum MN forward-model: per-axis MLF participation in the conjugate
    # motor pathway.  Yaw uses 0.5·(mn_lp1+mn_lp2) — half the agonist force
    # (LR_R for right eye) is single-MN filtered, the other half (MR_L for
    # left eye, via MLF) is double-MN filtered.  Pitch/roll use mn_lp1 alone
    # (no MLF stage modelled on the vertical/torsional axes).
    mlf_axis_weight:       jnp.ndarray  = jnp.array([0.5, 0.0, 0.0])

    # Saccadic-suppression gate strength (contrast amplification on the raw
    # `1 − cascaded_sat_flag` gate).  Applied as:
    #     gate = clip((raw_gate − threshold) / (1 − threshold), 0, 1) ** steepness
    # threshold = 0 + steepness = 1 reproduces the raw linear gate.  Larger
    # threshold or steepness pushes intermediate gate values toward 0, so
    # only near-fully-open raw gates (cascaded_sat ≈ 0) pass through.
    saccadic_suppression_threshold:    float = 0.85  # raw_gate ≤ this → fully suppressed
    saccadic_suppression_steepness:    float = 6.0   # power applied to thresholded gate
                                                              # 1.0 = transparent (ceiling >> normal burst); <1 = conduction block
                                                              # Nerve order: [LR_L,MR_L,SR_L,IR_L,SO_L,IO_L, LR_R,MR_R,SR_R,IR_R,SO_R,IO_R]
                                                              # INO: g_nerve[MR_L or MR_R] ↓  →  adducting saccades slow, fixation preserved
                                                              # CN6: g_nerve[LR_L or LR_R] ↓  →  abduction limited

    # INO (internuclear ophthalmoplegia) — synaptic gain on the MLF input to
    # each MR motoneuron pool.  AIN_L → MR_R (right MLF) and AIN_R → MR_L
    # (left MLF) are explicit edges in M_NERVE_PROJ; g_mlf_L/R scale them.
    #   g_mlf_L = 0 → left  MLF cut → left  MR fails to adduct on rightward gaze
    #   g_mlf_R = 0 → right MLF cut → right MR fails to adduct on leftward  gaze
    # Vergence is preserved in either case (CN3_MR drives MR directly).
    g_mlf_L:               float        = 1.0   # left  MLF synaptic gain (AIN_R → MR_L)
    g_mlf_R:               float        = 1.0   # right MLF synaptic gain (AIN_L → MR_R)

    # ── Cerebellum — prediction-error correction (cerebellum.md) ──────────────
    # Currently only the pursuit region is wired (paraflocculus_ventral /
    # vermis VI–VII).  Other regions (FL / VPF / V/CFN / NU) will be re-added
    # when there's a brainstem subsystem ready to consume their drive.
    # K_cereb_pu = 0 simulates a cerebellar pursuit-region lesion: the
    # cerebellar drive vanishes and pursuit falls back to the brainstem
    # direct path alone (raw self-motion-contaminated slip → reduced gain).
    K_cereb_pu:            float        = 1.0   # cerebellar pursuit prediction gain.
                                                  # Cerebellum's drive is the
                                                  # FORWARD-MODEL PREDICTION itself
                                                  # (negated so it adds to brainstem
                                                  # slip, gated by trust):
                                                  #   u_cb = gate · K_cereb_pu · (−s_pred)
                                                  #        = gate · K_cereb_pu
                                                  #          · visible · ec_d_target_no_torsion
                                                  # Combined with the brainstem direct
                                                  # path:
                                                  #   u_total = K_pursuit_direct·target_slip
                                                  #             + gate · K_cereb_pu · ec_d_target
                                                  # No explicit cancellation term — the
                                                  # brainstem's raw target_slip and the
                                                  # cerebellum's EC prediction naturally
                                                  # cancel during saccades because
                                                  # target_slip ≈ −eye_velocity and
                                                  # ec_d_target ≈ +eye_velocity (both
                                                  # delayed by the same cascade).
                                                  # Lesion (K_cereb_pu = 0) zeros this →
                                                  # pursuit falls back to raw target_slip
                                                  # → reduced gain (flocculus phenotype).
    K_pursuit_direct:      float        = 1.0   # brainstem reactive gain on
                                                  # gated raw target_slip (= sat ·
                                                  # cyc.target_vel) — the direct path.
                                                  # Saccadic suppression still acts on
                                                  # this term via `sat`.
    K_vor_direct:          float        = 1.0   # brainstem reactive gain on the gated
                                                  # raw scene slip (= sat · cyc.scene_angular_vel)
                                                  # feeding VS / OKR — analog of
                                                  # K_pursuit_direct for the scene path.
    K_cereb_okr:           float        = 1.0   # cerebellar OKR EC-correction gain.
                                                  # Multiplies acts.cb.fl_okr_drive
                                                  # (= sat · scene_visible · ec_scene)
                                                  # in the scene PE assembly.  Lesion
                                                  # (K=0): no scene EC correction, VS
                                                  # driven by gated raw slip only.
    K_cereb_fl:            float        = 1.0   # floccular NI leak-cancellation gain
                                                  # (Cannon & Robinson 1985).  Adds
                                                  # positive feedback K · (x_net − x_null)
                                                  # / tau_i_per_axis to NI input.
                                                  # K = 1 → NI ≈ perfect integrator;
                                                  # K = 0 → floccular lesion, gaze-evoked
                                                  # nystagmus.
    K_cereb_fl_vs:         float        = 0.0   # floccular VS leak-cancellation gain.
                                                  # Same Cannon-Robinson architecture
                                                  # applied to velocity storage. Default
                                                  # 0 because the current tau_vs = 20 s
                                                  # already represents the FL-extended
                                                  # effective TC; enabling this on top
                                                  # would push tau_vs → ∞ and break OKAN.
                                                  # To opt in: shorten tau_vs to its
                                                  # intrinsic brainstem value (~5 s)
                                                  # and set K_cereb_fl_vs = 0.75 to
                                                  # restore effective tau_vs ≈ 20 s with
                                                  # an explicit FL contribution that
                                                  # can be lesioned (K = 0 → tau_vs ≈
                                                  # intrinsic 5 s, shortened OKAN).
    K_cereb_nu:            float        = 1.0   # nodulus + uvula axis-dumping gain
                                                  # (Cohen, Raphan, Wearne).  Multiplies
                                                  # K_gd · rf (the VS gravity-axis
                                                  # damping signal) before it enters VS.
                                                  # K_cereb_nu = 1 → full normal dumping
                                                  # (tilt suppression, OVAR works);
                                                  # K_cereb_nu = 0 → nodular lesion →
                                                  # PAN (periodic alternating nystagmus)
                                                  # + prolonged tau_vs + loss of tilt
                                                  # suppression of post-rotatory nystagmus.


# ── Initialization ─────────────────────────────────────────────────────────────

def make_x0(brain_params=None):
    """Default initial BrainState.

    VS populations initialised to b_vs (bilateral equilibrium — both pops at resting bias).
    VS/NI null adaptation states initialised to 0 (no initial adaptation).
    NI populations initialised to 0 (b_ni=0 → net=0 at centre gaze).
    Gravity estimator initialised pointing down (upright head).

    Args:
        brain_params: BrainParams NamedTuple.  If None, uses b_vs=0 (zero bias).

    Returns:
        BrainState — nested NamedTuple with each subsystem's initial state.
    """
    # Self-motion: VS at b_vs equilibrium, GE pointing down (upright gravity).
    if brain_params is not None:
        vs_L = brain_params.b_vs[:3]
        vs_R = brain_params.b_vs[3:]
    else:
        vs_L = jnp.zeros(3)
        vs_R = jnp.zeros(3)
    sm_state = sm.State(
        vs_L    = vs_L,
        vs_R    = vs_R,
        vs_null = jnp.zeros(3),
        g_est   = sm.GRAV_X0[0:3],
        a_lin   = sm.GRAV_X0[3:6],
        rf      = sm.GRAV_X0[6:9],
        v_lin   = jnp.zeros(3),
    )

    # SG: OPN tonic = 100 (blocks burst between saccades).
    sg_state = sg.State(
        e_held = jnp.zeros(3),
        z_opn  = jnp.float32(100.0),
        z_acc  = jnp.float32(0.0),
        z_trig = jnp.float32(0.0),
        z_fac  = jnp.float32(0.0),
        z_dep  = jnp.float32(0.0),
        ebn_R  = jnp.zeros(3),
        ebn_L  = jnp.zeros(3),
        ibn_R  = jnp.zeros(3),
        ibn_L  = jnp.zeros(3),
    )

    # Vergence: x_slow[H] holds tonic_verg so u_verg = tonic_verg at rest.
    # Accommodation: x_slow holds tonic_acc baseline.
    if brain_params is not None:
        verg_tonic_init = jnp.zeros(3).at[0].set(brain_params.tonic_verg)
        acc_slow_init   = jnp.float32(brain_params.tonic_acc)
    else:
        verg_tonic_init = jnp.zeros(3)
        acc_slow_init   = jnp.float32(0.0)
    va_state = va.State(
        verg_fast  = jnp.zeros(3),
        verg_tonic = verg_tonic_init,
        verg_copy  = jnp.zeros(3),
        acc_fast   = jnp.float32(0.0),
        acc_slow   = acc_slow_init,
    )

    # MN: start on the slow manifold for centre-gaze + tonic vergence so the
    # nerves have their resting bias from t=0 (skips a ~5·tau_mn start-up
    # transient at the very beginning of every simulation).
    if brain_params is not None:
        vv_rest = jnp.array([0.0, 0.0, 0.0, brain_params.tonic_verg, 0.0, 0.0])
        fcp_state = fcp.State(mn=fcp.rest_state(vv_rest, brain_params))
    else:
        fcp_state = fcp.zero_state()

    return BrainState(
        pc   = pc.rest_state(),
        sm   = sm_state,
        pt   = pt.rest_state(),
        sg   = sg_state,
        pu   = pu.rest_state(),
        va   = va_state,
        ni   = ni.rest_state(),
        fcp  = fcp_state,
        cb   = cb.rest_state(),
    )


# ── Step function ──────────────────────────────────────────────────────────────

def step(brain_state, sensory_out, brain_params, noise_acc=0.0):
    """Single ODE step for the brain subsystem.

    Args:
        brain_state:  BrainState NamedTuple — one field per subsystem
        sensory_out:  SensoryOutput bundled canal afferents + cyclopean delayed signals
                        .canal            (6,)    canal afferent rates
                        .otolith          (3,)    specific force in head frame (m/s²)
                        .retina_L         RetinaOut for left eye
                        .retina_R         RetinaOut for right eye
        brain_params: BrainParams   model parameters
        noise_acc:    scalar  accumulator diffusion noise sample (pre-scaled)

    Returns:
        dbrain_state: BrainState  state derivative
        nerves:       (12,)  per-muscle nerve activations [L6 | R6] → plant
        ec_vel:       (3,)   version velocity efference (head frame, deg/s)
        ec_pos:       (3,)   eye position efference
        ec_verg:      (3,)   vergence efference
        u_acc:        scalar total lens-plant input (D) — neural + CA/C, drives acc_plant
    """
    # ── Activation / Decoded / Weights registries ────────────────────────────
    # Built once per step.  Subsystems read these instead of raw state.
    # `weights` holds setpoint-like registers (vs_null, ni_null, e_held);
    # long-term these become learnable parameters of the network.
    acts     = read_activations(brain_state, brain_params)
    decoded  = decode_activations(acts)
    weights  = read_weights(brain_state)

    # ── Efference-copy reads (all four assembled in one place) ──────────────
    # ec_pos / ec_verg are state-only readouts of the current (or 1-step-lagged)
    # version / vergence eye position — used by perception_cyclopean for the
    # post-delay EC subtraction.  ec_d_scene / ec_d_target are the delayed EC
    # cascade outputs (matched-impulse-response copies of the version-velocity
    # command), used to cancel self-generated motion in the slip cascades.
    x_ni_net    = decoded.ni.net
    ec_pos      = decoded.ni.net
    ec_verg     = (acts.va.verg_fast + acts.va.verg_tonic
                   + jnp.array([brain_params.tonic_verg, 0.0, 0.0]))
    # EC cascade tails come from cerebellum activations (acts.cb), already
    # built by read_activations(brain_state, brain_params) at the top of step.
    # ec_scene / ec_target are still consumed by pt.step (target perception
    # memory) and by the cerebellum's own internal computations.
    ec_d_target = acts.cb.ec_target

    # ── Cyclopean perception: binocular fusion + brain LP smoothing ──────────
    # ec_pos / ec_verg are 1-step-delayed via the ODE state read order — the
    # exact same-step ec_verg from this step's va.step isn't available yet, so
    # we use the state-based reconstruction above.  Negligible lag (one dt).
    dpc, cyc = pc.step(
        state        = brain_state.pc,
        retina_L     = sensory_out.retina_L,
        retina_R     = sensory_out.retina_R,
        ec_pos       = ec_pos,
        ec_verg      = ec_verg,
        brain_params = brain_params,
    )

    # ── Target path: working memory only (FEF/dlPFC) → SG ────────────────────
    # The pursuit-side EC subtraction + Hill/directional gates moved into the
    # cerebellum (pursuit region — see cb.step below).  pt.step now produces
    # only the SG-relevant signals.
    dpt, tgt_pos_eff, tgt_vis_eff = pt.step(
        activations    = acts.pt,
        target_visible = cyc.target_visible,
        target_pos     = cyc.target_pos,
        ec_d_target    = ec_d_target,
    )

    # ── Cerebellum: pursuit-region activations are read at the top of step()
    # via `acts.cb` (computed by cerebellum.read_activations from BrainState
    # pc + ec).  No state to update; pursuit input is assembled below.

    # ── Scene-path PE assembly (saccadic suppression on RAW slip + cerebellar
    #    gated EC correction).  Mirrors the pursuit-side pattern below:
    #        slip_pe_for_vs = K_vor_direct · sat · cyc.scene_angular_vel
    #                       + K_cereb_okr  · acts.cb.fl_okr_drive
    #    where fl_okr_drive is the cerebellum's pre-gated scene EC correction
    #    (= sat · scene_visible · ec_scene).  At default gains the sum equals
    #    sat · (cyc.scene_angular_vel + scene_visible · ec_scene).  Flocculus-
    #    OKR lesion (K_cereb_okr = 0): VS driven by raw gated slip only.
    sat_vs        = acts.cb.saccadic_suppression_scene
    slip_pe_for_vs = (brain_params.K_vor_direct * sat_vs * cyc.scene_angular_vel
                     + brain_params.K_cereb_okr * acts.cb.fl_okr_drive)
    scene_lin_pe  = sat_vs * cyc.scene_linear_vel

    # ── Self-motion observer (VS + GE + HE) — single unified step ────────────
    # Activation-driven: VS pops + GE/HE observer states come from `acts.sm`;
    # the vs_null setpoint comes from `weights.sm`.
    dsm, w_est, g_est, v_lin, a_lin_est = sm.step(
        acts.sm, weights.sm,
        canal           = sensory_out.canal,
        gia             = sensory_out.otolith,
        slip_pe_for_vs  = slip_pe_for_vs,
        scene_lin_pe    = scene_lin_pe,
        scene_visible   = cyc.scene_visible,
        fl_vs_drive     = acts.cb.fl_vs_drive,
        nu_drive        = acts.cb.nu_drive,
        brain_params    = brain_params,
    )

    # OCR: world frame [x=right, y=up, z=fwd]. Right-ear-down → g_est[0] < 0 → -g_est[0] > 0.
    # Positive motor roll = left-ear-down (left-hand rule); negative = right-ear-down.
    ocr = jnp.array([0.0, 0.0, -brain_params.g_ocr * g_est[0]])

    # ── Pursuit input: saccadically-gated brainstem direct + cerebellar EC
    #   pursuit_in = K_pursuit_direct · sat · cyc.target_vel + K_cereb_pu · vpf_drive
    # where vpf_drive = sat · target_visible · ec_no_torsion, so both terms
    # scale with the saccadic-suppression gate `sat` (≈ slip + vis·ec at K=1,
    # ×sat).  IMPORTANT: the gate strengthen MUST be mild
    # (`saccadic_suppression_threshold ≈ 0`, `steepness ≈ 1`) — an aggressive
    # gate throttles pursuit between catch-up saccades, where the gate cascade
    # has not fully reopened.  Floccular lesion (K_cereb_pu = 0): gated raw
    # slip only — no EC correction, reduced gain during head motion.
    sat_pu                  = acts.cb.saccadic_suppression_target
    target_slip_for_pursuit = (brain_params.K_pursuit_direct * sat_pu * cyc.target_vel
                               + brain_params.K_cereb_pu     * acts.cb.vpf_drive)
    dpu, u_pursuit = pu.step(
        activations    = acts.pu,
        target_slip_ec = target_slip_for_pursuit,
        brain_params   = brain_params,
    )

    # ── Saccade generator (target selection + Listing's corrections internal) ──
    dsg, u_burst = sg.step(acts.sg, weights.sg, tgt_pos_eff, tgt_vis_eff,
                            x_ni_net, ocr, w_est, brain_params, noise_acc)

    # ── Translational VOR (T-VOR): vestibular + visual fusion, distance-scaled ───
    # Uses heading_estimator's v_lin (already gravity-corrected, τ_head ≈ 2 s) as the
    # vestibular linear-velocity estimate.
    # Distance proxy: actual current vergence ≈ u_verg[0] computed from state:
    #   x_v[H] (fast integrator) + x_tonic[H] (tonic adapter) + AC/A·acc_fast (cross-link).
    # Using only x_tonic[H] missed the fast-integrator and ACA contributions, so during
    # a transient the inferred distance was wrong (e.g. eyes fused on a 10 m target but
    # x_tonic still near +15° tonic → T-VOR thought distance was 24 cm).
    # Direct phasic path (τ_p·disp) is dropped — small and not state-computable.
    _DEG_PER_PD = 0.5729
    aca_term = brain_params.AC_A * _DEG_PER_PD * acts.va.acc_fast
    current_vergence_yaw = acts.va.verg_fast[0] + acts.va.verg_tonic[0] + aca_term
    omega_tvor, verg_rate_tvor = tv.step(
        v_lin    = v_lin,
        a_lin    = a_lin_est,
        verg_yaw = current_vergence_yaw,
        eye_pos  = x_ni_net,
        brain_params = brain_params,
    )

    # ── Neural integrator: VOR + saccades + pursuit + T-VOR → version motor command ───
    # OCR is a tonic position-offset set-point (gravity-driven); passed as u_tonic so it
    # shifts the NI leak target via x_null_eff (saccade landing on OCR is then stable).
    # T-VOR contributes a VELOCITY (omega_tvor, deg/s) that NI integrates alongside
    # the other velocity drives — no longer a position bypass.
    # Torsional VOR gain is ~half horizontal (Crawford 1991, Misslisch 1994); apply that
    # attenuation only at the VS→NI connection so w_est elsewhere keeps full magnitude.
    vor_torsion_gain = jnp.array([1.0, 1.0, 0.5])
    # Floccular leak-cancellation feedback (Cannon & Robinson 1985):
    # acts.cb.fl_drive = K_cereb_fl · (ni_net − ni_null) / tau_i_per_axis
    # Added to NI input so the leak term cancels out and the NI behaves as
    # a near-perfect integrator (effective TC → ∞ at K_cereb_fl = 1).
    # K_cereb_fl = 0 → floccular lesion → intrinsic NI leak → gaze-evoked
    # nystagmus (eye drifts back to centre with TC ~ tau_i).
    u_ni_in = (-w_est * vor_torsion_gain + u_burst + u_pursuit + omega_tvor
               + acts.cb.fl_drive)

    # ── Listing's law — saccades aim at LL via SG's e_target (handled inside
    # sg.step). The velocity-level correction below is for the SMOOTH pathways
    # (pursuit + T-VOR) that need to STAY on Listing's plane during continuous
    # tracking — fundamentally different from saccades, which need to MOVE TO
    # the plane. Burst is excluded from vel_hv so the saccade's H/V command
    # doesn't trigger a redundant velocity-level correction (the burst already
    # ends on the plane via the LL-aware e_target).
    smooth_vel_hv = (u_pursuit + omega_tvor)[:2]
    ll_u_ni, ll_u_tonic, ll_verg_rate = listing.listing_corrections(
        x_ni_net, smooth_vel_hv, current_vergence_yaw,
        brain_params.listing_primary, brain_params.listing_l2_frac,
        brain_params.listing_gain,
    )
    u_ni_in        = u_ni_in        + ll_u_ni
    verg_rate_tvor = verg_rate_tvor + ll_verg_rate

    # NI tonic = OCR (gravity) + Listing's prescribed torsion (gaze-dependent).
    # Both shift the NI's leak target so x_net leaks toward the correct torsion
    # at SS without relying purely on velocity-level corrections to maintain it.
    u_tonic = ocr + ll_u_tonic
    dni, motor_cmd_ni = ni.step(acts.ni, weights.ni, u_ni_in, brain_params, u_tonic=u_tonic)

    # ── Vergence + Accommodation — single unified step ────────────────────────
    # Internal sequencing (state-based AC/A & CA/C → vergence → accommodation)
    # and CA/C application to the lens plant are all owned by va.step.
    # Cross-couplings happen one-step-delayed via integrator state, matching
    # synaptic latency and avoiding intra-step iteration.
    # gate_opn is the unclipped OPN-membrane proxy (z_opn/100); vergence wants
    # it as a firing rate ∈ [0, 1] to compute the saccadic-vergence-burst gate.
    z_act_verg = 1.0 - jnp.clip(acts.sg.gate_opn, 0.0, 1.0)
    dva, u_verg, u_acc = va.step(
        acts.va,
        defocus          = cyc.defocus,
        target_disparity = cyc.target_disparity,
        verg_rate_tvor   = verg_rate_tvor,
        z_act            = z_act_verg,
        brain_params     = brain_params,
    )

    # ── Final common pathway: nucleus encode → MN low-pass → nerve activations ─
    # MN firing rates come from the brain-wide activations registry (`acts.fcp`),
    # which applies the cell-body f-I curve clip via fcp.read_activations.
    # FCP step is fully activation-driven — it does NOT need raw brain_state.fcp.
    dfcp, nerves = fcp.step(acts.fcp,
                             jnp.concatenate([motor_cmd_ni, u_verg]),
                             brain_params)

    # ── Cerebellum: cascade advance + return signals ─────────────────────────
    # ec_vel = version-velocity efference; drives both EC cascades inside cb.step
    # (which applies frame rotation, retinal saturation, and matched LP cascades).
    # ec_pos was assembled at the top from decoded.ni.net (same as x_ni_net).
    # ec_verg returned to caller is the FULL vergence command (u_verg from
    # va.step) — distinct from the lagged ec_verg used by perception_cyclopean
    # at the top of step (which was state-based to break the va↔pc loop).
    ec_vel       = u_burst + u_pursuit + omega_tvor
    ec_verg_cmd  = u_verg
    ni_net_full  = brain_state.ni.L - brain_state.ni.R
    ni_null_full = brain_state.ni.null
    vs_net_full  = brain_state.sm.vs_L - brain_state.sm.vs_R
    vs_null_full = brain_state.sm.vs_null
    rf_full      = brain_state.sm.rf
    dcb, _ = cb.step(
        state          = brain_state.cb,
        ec_vel         = ec_vel,
        ec_pos         = ec_pos,
        ni_net         = ni_net_full,
        ni_null        = ni_null_full,
        vs_net         = vs_net_full,
        vs_null        = vs_null_full,
        rf             = rf_full,
        w_est          = w_est,
        target_slip    = cyc.target_vel,
        target_visible = cyc.target_visible,
        scene_visible  = cyc.scene_visible,
        brain_params   = brain_params,
    )

    # ── Pack state derivative ─────────────────────────────────────────────────
    dbrain = BrainState(
        pc   = dpc,
        sm   = dsm,
        pt   = dpt,
        sg   = dsg,
        pu   = dpu,
        va   = dva,
        ni   = dni,
        fcp  = dfcp,
        cb   = dcb,
    )

    return dbrain, nerves, ec_vel, ec_pos, ec_verg_cmd, u_acc
