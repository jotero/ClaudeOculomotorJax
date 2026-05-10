"""Saccade Generator SSM — Robinson (1975) local-feedback burst model.

Robinson DА (1975) "Oculomotor control signals" in "Basic Mechanisms of Ocular
Motility and Their Clinical Implications", Pergamon, pp. 337–374.
Burst neuron recordings: Fuchs, Scudder & Kaneko (1988 J Neurophysiol).
Main sequence (velocity–amplitude): Bahill, Clark & Stark (1975 Math Biosci).

Architecture
────────────
Saccades are BALLISTIC: once triggered, they run to completion based on the
error at onset, not ongoing visual feedback.  A sample-and-hold state (e_held)
latches the retinal error at trigger onset and is frozen during the burst.
The Robinson residual (e_held − x_copy) drives the burst and decreases
monotonically to zero at burst end, regardless of target velocity.

State:  x_sg = [e_held(3) | z_opn(1) | z_acc(1) | z_trig(1) | z_fac(1) | z_dep(1) |
                 x_ebn_R(3) | x_ebn_L(3) | x_ibn_R(3) | x_ibn_L(3)]  (20 states)

    e_held  — Robinson's resettable integrator: latches e_cur at trigger onset; integrates −u_burst during burst.
               Sample-and-hold: between saccades tracks e_cur via τ_hold; frozen when OPN paused.
               No overshoot: e_held monotonically decrements to 0, burst ends naturally.
    z_opn   — OPN membrane potential: 100=tonic (burst blocked), near-0 or negative=paused (burst active)
    z_acc   — rise-to-bound accumulator: integrates gate_err; drains when OPN is suppressed.
               Natural refractory: (threshold_acc − acc_burst_floor) · τ_acc ≈ 270 ms.
    z_trig  — intermediate trigger state: rises from charge_sac with TC τ_trig; drained by IBN.
               Provides smooth onset for OPN suppression, decoupled from z_acc drain.
               Sequence: z_acc > threshold → charge_sac → z_trig rises → OPN drops → IBN fires
               → IBN latches OPN + drains z_trig; OPN suppression drains z_acc.
    z_fac   — BN facilitation: tracks (1 − act_opn/100) with TC τ_fac. Rises during OPN pause
               (saccade), decays back when OPN returns. Modulates BN drive via g_dyn.
               Not a firing rate — short-term presynaptic facilitation / use-dependent gain.
    z_dep   — BN depression: follows z_fac with slower TC τ_dep. Persists after burst end,
               leaving a transient post-saccadic dip in g_dyn (residual depression) that
               recovers slowly. Models post-saccadic refractory weakness.
    x_ebn_R — right EBN membrane potentials (3,): driven by relu(+e_held) when act_opn<1; held negative by OPN
    x_ebn_L — left  EBN membrane potentials (3,): driven by relu(−e_held) when act_opn<1; held negative by OPN
    x_ibn_R — right IBN membrane potentials (3,): same drive as x_ebn_R; held negative by OPN
    x_ibn_L — left  IBN membrane potentials (3,): same drive as x_ebn_L; held negative by OPN

Input:  pos_delayed (3,)   delayed retinal position error (deg)
Output: u_burst (3,)       saccade velocity command (deg/s)

──────────────────────────────────────────────────────────────────────────────
Burst neuron populations
──────────────────────────────────────────────────────────────────────────────
States are membrane potentials; activations are the burst nonlinearity applied
to those states (burst_velocity).

    src_R = relu(+e_res)   — drives right BNs during rightward e_res
    src_L = relu(−e_res)   — drives left  BNs during leftward  e_res

BNs are always driven by e_held directly; OPN suppression acts on x_bn via opn_gain
(multiplicative TC shortening) and opn_inh (additive offset) — not by gating src.

Schmitt-trigger sequence:
    Between saccades: act_opn=1 → opn_gain large → x_bn clamped to −0.4 → ibn_total=0.
    At trigger: charge_sac→1 → z_opn drops → act_opn falls → opn_gain drops → x_bn rises.
    IBN latches OPN paused; burst fires. Saccade ends: e_res→0 → src→0 → x_bn decays → burst ends.

BN dynamics (directional selectivity from relu; off-side src is 0 so off-side BN decays):
    dx_ebn_R/dt = (src_R − inh_from_L − opn_inh − x_ebn_R · opn_gain) / τ_bn

Burst output (no explicit gating — natural termination via BN state decay):
    u_burst = act_ebn_R − act_ebn_L
    act_ebn = burst_velocity(x_ebn, p)

OPN inhibition (z_trig + IBN Schmitt trigger):
    ibn_total = Σ(act_ibn_R + act_ibn_L)
    dz_opn = (k_tonic·(100−z_opn) − (z_opn+g_opn_pause)·z_trig − g_ibn_opn·ibn_total) / τ_sac
    z_trig provides smooth delayed onset (τ_trig); IBN latches OPN paused via g_ibn_opn.
    Accumulator drain (dz_acc) uses ibn_norm — zero between saccades, active only when burst runs.
    This prevents the sub-threshold equilibrium that sigmoid-tailed drain formulas create.

──────────────────────────────────────────────────────────────────────────────
OPN latch (z_opn) — the key to the ballistic design
──────────────────────────────────────────────────────────────────────────────
    Sequence:
        1. Target appears → gate_err→1 → z_acc accumulates (τ_acc)
        2. z_acc > threshold_acc → charge_sac→1 → z_opn drops (OPN pauses)
        3. act_opn falls → src_R = relu(e_res) → x_ebn/x_ibn rise → burst fires
        4. IBN keeps z_opn suppressed; act_opn≈0 sustains burst
        5. x_copy integrates toward e_held → e_res→0 → src→0 → x_bn decays → burst ends
        6. x_ibn→0 → ibn_total→0 → k_tonic restores z_opn→100 in ~2 ms
        7. z_acc re-climbs from floor to threshold → ~270 ms refractory

──────────────────────────────────────────────────────────────────────────────
Burst nonlinearity  (main sequence)
──────────────────────────────────────────────────────────────────────────────
    Element-wise (per-axis independent): g_burst · (1 − exp(−relu(e) / e_sat_sac))
    Each axis computes its own burst — no cross-axis coupling via 3D norm.
    Applied to EBN/IBN states.  At burst onset x_ebn ≈ e_held; tapers to 0
    as e_held→0 → bell-shaped velocity profile, clean natural stop.

──────────────────────────────────────────────────────────────────────────────
Output insertion (in simulator)
──────────────────────────────────────────────────────────────────────────────
    dx_ni  += u_burst            NI integrates burst → holds post-saccade position
    u_p    += tau_p · u_burst    velocity pulse → plant (cancels LP lag)

Parameters
──────────
    tau_fac        BN facilitation TC (s)             default 0.020
    tau_dep        BN depression TC (s)               default 0.100
    alpha_fac      facilitation gain (dimensionless)  default 0.0   (off; >0 boosts burst)
    alpha_dep      depression gain (dimensionless)    default 0.0   (off; >0 dips post-burst)
    g_burst        burst ceiling (deg/s)              default 700.0
    threshold_sac  retinal-error trigger (deg)        default 0.5
    k_sac          sigmoid steepness (1/deg)          default 200.0
    e_sat_sac      main-sequence saturation (deg)     default 7.0
    tau_i          NI leak TC — shared with NI (s)    default 25.0
    tau_latch      sample-and-hold TC (s)               default 0.003
    tau_hold       sample-and-hold tracking TC (s)    default 0.005
    tau_sac        saccade latch TC (s)               default 0.001
    tau_acc        accumulator rise TC (s)            default 0.180
    tau_burst_drain accumulator burst drain TC (s)    default 0.005
    acc_burst_floor accumulator target level          default -0.5
    threshold_acc  accumulator trigger threshold      default 1.0
    k_acc          accumulator sigmoid steepness      default 500.0
    k_tonic_opn    OPN tonic recovery gain            default 0.5
    tau_bn         burst-neuron state TC (s)          default 0.005
    g_opn_bn       OPN→BN multiplicative suppression  default 0.04 (act_opn∈[0,100])
    g_opn_bn_hold  OPN→BN additive offset             default 0.4  (act_opn∈[0,100])
    g_ibn_bn       contralateral IBN→BN gain          default 0.143
    g_ibn_opn      IBN→OPN inhibition gain            default 200.0
    g_opn_pause    OPN inhibitory overshoot           default 500.0
"""

from typing import NamedTuple

import jax.numpy as jnp
import jax

from oculomotor.models.brain_models import listing

# OPN tonic firing ≈ 100 sp/s; paused ≈ 0.  Gate output = z_opn / 100, clipped.
_OPN_TONIC = 100.0


# ── State + registries ────────────────────────────────────────────────────────

class State(NamedTuple):
    """SG state — Robinson held error + OPN/accumulator scalars + EBN/IBN pops."""
    e_held: jnp.ndarray   # (3,)    signed  Robinson local-feedback held error / setpoint
    z_opn:  jnp.ndarray   # scalar          OPN membrane (≈100 tonic, ≈0 paused)
    z_acc:  jnp.ndarray   # scalar          rise-to-bound accumulator
    z_trig: jnp.ndarray   # scalar          trigger membrane (charges from acc)
    z_fac:  jnp.ndarray   # scalar          BN facilitation (rises when OPN paused)
    z_dep:  jnp.ndarray   # scalar          BN depression (slow follower of z_fac)
    ebn_R:  jnp.ndarray   # (3,)            right EBN membrane potentials
    ebn_L:  jnp.ndarray   # (3,)            left  EBN membrane potentials
    ibn_R:  jnp.ndarray   # (3,)            right IBN membrane potentials
    ibn_L:  jnp.ndarray   # (3,)            left  IBN membrane potentials


class Activations(NamedTuple):
    """SG firing rates."""
    gate_opn: jnp.ndarray   # scalar  OPN gate output (1=tonic, 0=paused)  [nucleus raphe interpositus]
    z_acc:    jnp.ndarray   # scalar  rise-to-bound accumulator            [SC build-up cells, plausible]
    z_trig:   jnp.ndarray   # scalar  trigger membrane                     [SC burst cells, plausible]
    ebn_R:    jnp.ndarray   # (3,)    right EBN burst                      [PPRF | riMLF]
    ebn_L:    jnp.ndarray   # (3,)    left  EBN burst                      [PPRF | riMLF]
    ibn_R:    jnp.ndarray   # (3,)    right IBN burst                      [PPRF | riMLF (inhib)]
    ibn_L:    jnp.ndarray   # (3,)    left  IBN burst                      [PPRF | riMLF (inhib)]


class Weights(NamedTuple):
    """SG tonic / null / setpoint registers (long-term: learned weights).

    z_fac and z_dep are short-term modulatory states (not firing rates) that
    set the dynamic BN gain g_dyn. They live here rather than in Activations
    because no downstream subsystem reads them as a population rate — they
    only modulate BN drive inside this module.
    """
    e_held: jnp.ndarray   # (3,) signed   Robinson local-feedback held error / setpoint
    z_fac:  jnp.ndarray   # scalar        BN facilitation register
    z_dep:  jnp.ndarray   # scalar        BN depression register


def rest_state():
    """Zero state (z_opn = 100 = OPN tonic, blocking burst)."""
    return State(
        e_held = jnp.zeros(3),
        z_opn  = jnp.float32(_OPN_TONIC),
        z_acc  = jnp.float32(0.0),
        z_trig = jnp.float32(0.0),
        z_fac  = jnp.float32(0.0),
        z_dep  = jnp.float32(0.0),
        ebn_R  = jnp.zeros(3),
        ebn_L  = jnp.zeros(3),
        ibn_R  = jnp.zeros(3),
        ibn_L  = jnp.zeros(3),
    )


def read_activations(state):
    """Project SG state → SG Activations.

    NOTE: gate_opn is NOT clipped to [0, 1] here because the SG step()
    reconstructs the OPN membrane potential as gate_opn · _OPN_TONIC for its
    own dynamics, and the dynamics need the RAW z_opn (which dips deep below
    zero during saccades — equilibrium of the IBN-driven inhibition is at
    z_opn = −g_opn_pause = −500).  Clipping here was a bug: it left the
    recovery rate computed at the clipped value (k_tonic · (100 − 0)) while
    the true state was at large negative values, so OPN took 100s of ms to
    return to tonic and e_held never restarted tracking → no catch-up
    saccades during pursuit.

    External consumers that interpret gate_opn AS A FIRING RATE must clip
    themselves: `gate_opn_fr = jnp.clip(acts.sg.gate_opn, 0.0, 1.0)`.
    Currently only vergence_accommodation (z_act_verg = 1 − gate_opn) reads
    this — and that's robust since gate_opn ≥ 1 doesn't occur in normal
    operation, only the negative excursion is the issue.
    """
    return Activations(
        gate_opn = state.z_opn / _OPN_TONIC,    # unclipped — dynamics need raw z_opn
        z_acc    = state.z_acc,
        z_trig   = state.z_trig,
        ebn_R    = state.ebn_R,
        ebn_L    = state.ebn_L,
        ibn_R    = state.ibn_R,
        ibn_L    = state.ibn_L,
    )


def read_weights(state):
    """SG Robinson held-error setpoint + BN facilitation/depression registers."""
    return Weights(
        e_held = state.e_held,
        z_fac  = state.z_fac,
        z_dep  = state.z_dep,
    )


# ── Burst velocity (magnitude + direction) ────────────────────────────────────

def burst_velocity(e, p):
    """Main-sequence nonlinearity: element-wise per-axis burst velocity.

    Rectifies first (negative membrane potential = OPN-inhibited = silent),
    then applies the exponential main-sequence saturation independently on each axis.
    No cross-axis coupling: horizontal and vertical components compute their own burst.

    Args:
        e: (3,)  input vector (EBN/IBN membrane potential)
        p: BrainParams

    Returns:
        (3,)  burst velocity (deg/s), zero for negative states
    """
    e = jax.nn.relu(e)
    return p.g_burst * (1.0 - jnp.exp(-e / p.e_sat_sac))


# ── SSM step ──────────────────────────────────────────────────────────────────

def step(activations, weights, pos_delayed, target_visible, x_ni, ocr, w_est,
         p, noise_acc=0.0):
    """Single ODE step: target selection + burst dynamics + burst output.

    Activation-driven: burst neuron firing rates and trigger/accumulator
    activations come from `activations` (acts.sg).  The Robinson held-error
    setpoint comes from `weights` (weights.sg).

    Note: gate_opn in `activations` is the clipped firing-rate proxy; the
    underlying OPN membrane state (z_opn) is recovered from gate_opn via the
    inverse projection (gate × _OPN_TONIC) — this is the firing rate's
    interpretation as state for the LP dynamics, identical for the typical
    operating range.

    Target selection (uses brain-internal x_ni, not plant state):
        target_visible ≈ 1  (in visual field):
            e_cmd = clip(pos_delayed + ocr, −orbital_limit − x_ni, +orbital_limit − x_ni)
        target_visible ≈ 0  (quick-phase generator — target outside ~90° visual field):
            e_cmd = −alpha_reset · (x_ni − k_center_vel·τ_ref · w_vs)
            Predictive centripetal quick phase.

    Args:
        activations:   sg.Activations  gate_opn, z_acc, z_trig, ebn_R/L, ibn_R/L
        weights:       sg.Weights      e_held (Robinson held-error setpoint)
        pos_delayed:   (3,)         delayed retinal position error (deg)
        target_visible: scalar       visual-field gate (≈1 in-field, ≈0 out-of-field)
        x_ni:          (3,)         NI net state — brain's eye-position estimate (deg)
        ocr:           (3,)         OCR vector (rotation-vec) — added to target so the
                                    saccade aims at (H, V, T_with_OCR). Only torsion is
                                    non-zero in practice.
        w_est:         (3,)         velocity storage head-velocity estimate (deg/s)
        p:             BrainParams
        noise_acc:     scalar        pre-generated accumulator diffusion term

    Returns:
        dstate:  sg.State    state derivative
        u_burst: (3,)        saccade velocity command (deg/s)
    """
    # Listing's law is applied centrally in brain_model.step (added to the
    # summed velocity command before NI integration), so SG aims at H/V only
    # and torsion drops out of the velocity-level rule here.

    # ── Activation reads ──────────────────────────────────────────────────────
    e_held  = weights.e_held
    z_fac   = weights.z_fac
    z_dep   = weights.z_dep
    z_opn   = activations.gate_opn * _OPN_TONIC   # gate_opn = clip(z_opn/100); invert
    z_acc   = activations.z_acc
    z_trig  = activations.z_trig
    x_ebn_R = activations.ebn_R
    x_ebn_L = activations.ebn_L
    x_ibn_R = activations.ibn_R
    x_ibn_L = activations.ibn_L

    # ── Target selection ──────────────────────────────────────────────────────
    # In-field: saccade aims at the landed gaze (current x_ni + retinal error
    # pos_delayed for H/V) with torsion set to OCR + Listing's-prescribed
    # torsion at the landed (H, V). This way each saccade ENDS on Listing's
    # plane, not just on OCR.
    #
    # OCR (gravity-driven) and T_LL(H_landed, V_landed) are both ABSOLUTE
    # torsion targets in head frame. Convert to a delta from current x_ni[2]:
    #   delta_T = OCR + T_LL(landed gaze) − x_ni[2]
    H_landed = x_ni[0] + pos_delayed[0] - p.listing_primary[0]
    V_landed = x_ni[1] + pos_delayed[1] - p.listing_primary[1]
    T_LL_landed = -(jnp.pi / 360.0) * H_landed * V_landed
    listing_target_delta = jnp.array([0.0, 0.0,
                                       p.listing_gain * T_LL_landed - x_ni[2]])
    ocr_delta = ocr + listing_target_delta
    e_target = jnp.clip(pos_delayed + ocr_delta, -p.orbital_limit - x_ni, p.orbital_limit - x_ni)

    tau_refractory = (p.threshold_acc - p.acc_burst_floor) * p.tau_acc
    x_ni_pred = x_ni - p.k_center_vel * tau_refractory * w_est
    # Quick-phase reset target: centripetal toward 0 in H/V, but toward OCR setpoint
    # in torsion (head-tilt-driven). Without OCR here, dark quick phases would drag
    # the eye to 0 torsion every reset and OCR would never reach the eye.
    e_center  = -p.alpha_reset * (x_ni_pred - ocr)
    
    doing_saccade = target_visible;
    doing_quick_phase = 1.0 - target_visible;
    e_cur     =  doing_saccade * e_target + doing_quick_phase * e_center
    e_cur_mag = jnp.linalg.norm(e_cur)

    # ── Trigger gate (accumulator source) ────────────────────────────────────
    threshold_sac_eff = doing_saccade * p.threshold_sac + doing_quick_phase * p.threshold_sac_qp
    gate_err = jax.nn.sigmoid(p.k_sac * (e_cur_mag - threshold_sac_eff))

    # ── Activations ───────────────────────────────────────────────────────────────
    act_opn   = jnp.clip(z_opn, 0.0, 100.0)              # 0=paused (burst active), 100=tonic (burst blocked)
    act_ebn_R = burst_velocity(x_ebn_R, p)
    act_ebn_L = burst_velocity(x_ebn_L, p)
    act_ibn_R = burst_velocity(x_ibn_R, p)
    act_ibn_L = burst_velocity(x_ibn_L, p)

    # ── Burst output ──────────────────────────────────────────────────────────
    u_burst = act_ebn_R - act_ebn_L

    # ── Burst neuron dynamics ─────────────────────────────────────────────────
    # Three-gain inhibitory network (all within tau_bn):
    #   g_opn_bn:      OPN→BN multiplicative — clamping TC=tau_bn/(1+g_opn_bn·act_opn). Must satisfy Heun stability:
    #                  (1+g_opn_bn·100)·dt/tau_bn < 2 → g_opn_bn < 0.09 with dt=1ms, tau_bn=3ms.
    #   g_opn_bn_hold: OPN→BN additive offset — keeps BN at (e_held−g_opn_bn_hold·100)/(1+g_opn_bn·100) < 0 between saccades.
    #   Combined equilibrium: x_eq = (e_held − 40)/(1+4) = −8 at fixation → IBN firmly off.
    #   g_ibn_bn:      IBN→BN contralateral, element-wise (axis-matched); g_ci/g_burst absorbed.
    #   g_ibn_opn:     IBN→OPN Schmitt-trigger latch.
    inh_from_L = p.g_ibn_bn * act_ibn_L   # (3,) L IBN → R BNs, one-to-one per axis
    inh_from_R = p.g_ibn_bn * act_ibn_R   # (3,) R IBN → L BNs, one-to-one per axis
    opn_gain   = 1.0 + p.g_opn_bn * act_opn        # multiplicative: TC = tau_bn/(1+g_opn_bn) when tonic
    opn_inh    = p.g_opn_bn_hold * act_opn          # additive offset: 0 during saccade

    # Dynamic BN drive gain — short-term presynaptic facilitation/depression.
    # g_dyn rises during OPN pause (z_fac builds), then dips below 1 after burst
    # (z_dep persists as z_fac decays). alpha_fac=alpha_dep=0 → g_dyn≡1 (off).
    g_dyn = 1.0 + p.alpha_fac * z_fac - p.alpha_dep * z_dep

    dx_ebn_R = (g_dyn * jax.nn.relu( e_held) - inh_from_L - opn_inh - x_ebn_R * opn_gain) / p.tau_bn
    dx_ebn_L = (g_dyn * jax.nn.relu(-e_held) - inh_from_R - opn_inh - x_ebn_L * opn_gain) / p.tau_bn
    dx_ibn_R = (g_dyn * jax.nn.relu( e_held) - inh_from_L - opn_inh - x_ibn_R * opn_gain) / p.tau_bn
    dx_ibn_L = (g_dyn * jax.nn.relu(-e_held) - inh_from_R - opn_inh - x_ibn_L * opn_gain) / p.tau_bn

    # ── Resettable integrator (Robinson 1975) ────────────────────────────────
    # Continuous pre-loading: e_held tracks pos_delayed (=e_cur) between saccades.
    # By the time z_acc reaches threshold (~270 ms accumulation), e_held ≈ e_cur.
    # OPN pause (normalized_opn→0) freezes tracking during burst; u_burst decrements e_held.
    normalized_opn = act_opn / 100.0   # 0 = paused, 1 = tonic
    de_held = -u_burst + normalized_opn**2 * (e_cur - e_held) / p.tau_hold

    # ── Trigger state + accumulator ───────────────────────────────────────────
    # charge_sac: relu (not sigmoid) → exactly 0 below threshold, no pre-threshold tail.
    #   This is essential: any pre-threshold z_trig suppresses OPN → opn_paused > 0 → drain
    #   with tau_burst_drain=2ms creates 500x amplified drain that always overwhelms charging.
    # z_trig: smooth delayed OPN suppression signal (trigger IBN population).
    #   Charges from charge_sac (only when z_acc > threshold_acc, no sigmoid bleed).
    #   Drained by IBN: burst suppresses trigger → OPN recovers when burst ends.
    #   Heun stability: (1+g_ibn_trig)·dt/tau_trig < 2 → tau_trig > 1.5ms. Current 2ms ✓.
    charge_sac   = jnp.clip(p.k_acc * (z_acc - p.threshold_acc), 0.0, 1.0)
    ibn_total    = jnp.sum(act_ibn_R) + jnp.sum(act_ibn_L)
    ibn_norm     = jnp.clip(ibn_total / (2.0 * p.g_burst), 0.0, 1.0)
    dz_trig = (charge_sac - z_trig * (1.0 + p.g_ibn_trig * ibn_norm)) / p.tau_trig

    # z_acc: charging GATED by normalized_opn (1=tonic, 0=paused during burst).
    #   Drain by g_acc_drain * ibn_norm only: exactly 0 between saccades (IBN silent), only
    #   active during burst. g_acc_drain boosts drain for small saccades (weak IBN); within
    #   Heun bound: g_acc_drain · dt / tau_burst_drain < 2 → g_acc_drain < 4 at current params.
    dz_acc = (gate_err * normalized_opn / p.tau_acc
              - p.g_acc_drain * ibn_norm * (z_acc - p.acc_burst_floor) / p.tau_burst_drain
              - z_acc / p.tau_acc_leak
              + noise_acc)

    # ── BN dynamic-gain dynamics (facilitation + depression) ─────────────────
    # z_fac tracks (1 − act_opn/100) with τ_fac (fast LP):
    #   OPN paused (saccade) → drive≈1 → z_fac rises toward 1.
    #   OPN tonic (between saccades) → drive≈0 → z_fac decays toward 0.
    # z_dep follows z_fac with τ_dep (slow LP) — persists after z_fac falls,
    # producing transient post-saccadic depression in g_dyn.
    pause_drive = 1.0 - normalized_opn
    dz_fac = (pause_drive - z_fac) / p.tau_fac
    dz_dep = (z_fac - z_dep) / p.tau_dep

    # ── OPN latch dynamics ────────────────────────────────────────────────────
    # z_trig suppresses OPN initially (smooth onset via tau_trig, replaces raw charge_sac).
    # IBN latches OPN suppressed throughout burst (Schmitt trigger).
    # At burst end: e_held→0 → IBN→0 → z_trig drained by IBN → both release OPN.
    dz_opn = (p.k_tonic_opn * (100.0 - z_opn)
              - (z_opn + p.g_opn_pause) * z_trig
              - p.g_ibn_opn * ibn_total) / p.tau_sac

    dstate = State(
        e_held = de_held,
        z_opn  = dz_opn,
        z_acc  = dz_acc,
        z_trig = dz_trig,
        z_fac  = dz_fac,
        z_dep  = dz_dep,
        ebn_R  = dx_ebn_R,
        ebn_L  = dx_ebn_L,
        ibn_R  = dx_ibn_R,
        ibn_L  = dx_ibn_L,
    )
    return dstate, u_burst


# ── Legacy flat-array adapters (deleted once brain_model migrates to BrainState) ─

N_STATES  = 20
N_INPUTS  = 9
N_OUTPUTS = 3


def from_array(x_sg):
    """(20,) flat array → sg.State."""
    return State(
        e_held = x_sg[0:3],
        z_opn  = x_sg[3],
        z_acc  = x_sg[4],
        z_trig = x_sg[5],
        z_fac  = x_sg[6],
        z_dep  = x_sg[7],
        ebn_R  = x_sg[8:11],
        ebn_L  = x_sg[11:14],
        ibn_R  = x_sg[14:17],
        ibn_L  = x_sg[17:20],
    )


def to_array(state):
    """sg.State → (20,) flat array."""
    return jnp.concatenate([
        state.e_held,
        jnp.array([state.z_opn, state.z_acc, state.z_trig, state.z_fac, state.z_dep]),
        state.ebn_R, state.ebn_L, state.ibn_R, state.ibn_L,
    ])
