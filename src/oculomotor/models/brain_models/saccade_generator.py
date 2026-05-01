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

State:  x_sg = [e_held(3) | z_opn(1) | z_acc(1) |
                 x_ebn_R(3) | x_ebn_L(3) | x_ibn_R(3) | x_ibn_L(3)]  (17 states)

    e_held  — error estimator: tracks e_cur between saccades; integrates −u_burst during burst.
               Replaces the old x_copy+e_held pair: e_held IS the residual error (e_res = e_held).
               Between saccades: de_held/dt = act_opn²·(e_cur−e_held)/τ_reset_fast (fast tracking).
               During burst:     de_held/dt = −u_burst   (decrements as burst fires → e_held→0).
               No overshoot: e_held monotonically approaches 0, so no corrective opposite burst.
    z_opn   — OPN membrane potential: 100=tonic (burst blocked), near-0 or negative=paused (burst active)
    z_acc   — rise-to-bound accumulator: integrates gate_err; drains to floor during burst.
               Natural refractory: (threshold_acc − acc_burst_floor) · τ_acc ≈ 270 ms.
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

    src_R = (1−act_opn) · relu(+e_res)   — drives right BNs during rightward e_res
    src_L = (1−act_opn) · relu(−e_res)   — drives left  BNs during leftward  e_res

act_opn gating (the Schmitt-trigger mechanism):
    Between saccades: act_opn=1 → src=0 → x_bn→0 → ibn_total=0 → OPN stable at 100.
    At trigger: charge_sac→1 → z_opn drops → act_opn falls → x_bn rises → burst fires.
    Saccade ends: e_res→0 → src→0 → x_bn decays → IBN→0 → OPN recovers.

BN dynamics (no explicit contralateral inhibition — directional selectivity comes
from relu; off-side src is 0, so off-side BN decays naturally):
    dx_ebn_R/dt = (src_R − x_ebn_R) / τ_bn

Burst output (no explicit gating — natural termination via BN state decay):
    u_burst = act_ebn_R − act_ebn_L
    act_ebn = burst_velocity(x_ebn, p)

OPN inhibition (Schmitt trigger via IBN):
    ibn_total = |act_ibn_R| + |act_ibn_L|
    dz_opn = (k_tonic·(100−z_opn) − (z_opn+g_opn_pause)·charge_sac − g_ibn_opn·ibn_total) / τ_sac

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
    Magnitude: g_burst · (1 − exp(−|e| / e_sat_sac))
    Direction: e / |e|   (unit vector of input)
    Applied to EBN/IBN states.  At burst onset |x_ebn| ≈ |e_held|; tapers to 0
    as x_copy→e_held → bell-shaped velocity profile, clean natural stop.

──────────────────────────────────────────────────────────────────────────────
Output insertion (in simulator)
──────────────────────────────────────────────────────────────────────────────
    dx_ni  += u_burst            NI integrates burst → holds post-saccade position
    u_p    += tau_p · u_burst    velocity pulse → plant (cancels LP lag)

Parameters
──────────
    g_burst        burst ceiling (deg/s)              default 700.0
    threshold_sac  retinal-error trigger (deg)        default 0.5
    k_sac          sigmoid steepness (1/deg)          default 200.0
    e_sat_sac      main-sequence saturation (deg)     default 7.0
    tau_i          NI leak TC — shared with NI (s)    default 25.0
    tau_reset_fast copy integrator reset TC (s)       default 0.05
    tau_hold       sample-and-hold tracking TC (s)    default 0.005
    tau_sac        saccade latch TC (s)               default 0.001
    tau_acc        accumulator rise TC (s)            default 0.180
    tau_burst_drain accumulator burst drain TC (s)    default 0.005
    acc_burst_floor accumulator target level          default -0.5
    threshold_acc  accumulator trigger threshold      default 1.0
    k_acc          accumulator sigmoid steepness      default 500.0
    k_tonic_opn    OPN tonic recovery gain            default 0.5
    tau_bn         burst-neuron state TC (s)          default 0.005
    g_opn_bn       OPN→BN multiplicative suppression  default 100.0
    g_opn_bn_hold  OPN→BN additive offset             default 2.0
    g_ci           contralateral IBN→BN gain          default 100.0
    g_ibn_opn      IBN→OPN inhibition gain            default 200.0
    g_opn_pause    OPN inhibitory overshoot           default 500.0
"""

import jax.numpy as jnp
import jax

from oculomotor.models.brain_models import listing

N_STATES  = 17  # e_held(3) + z_opn(1) + z_acc(1) + x_ebn_R(3) + x_ebn_L(3) + x_ibn_R(3) + x_ibn_L(3)
N_INPUTS  = 9   # pos_delayed(3) + target_visible(1) + x_ni(3)
N_OUTPUTS = 3   # u_burst (3,)


# ── Burst velocity (magnitude + direction) ────────────────────────────────────

def burst_velocity(e, p):
    """Main-sequence nonlinearity: maps input vector to burst velocity.

    Rectifies first (negative membrane potential = OPN-inhibited = silent),
    then applies the exponential main-sequence magnitude with the original direction.

    Args:
        e: (3,)  input vector (EBN/IBN membrane potential)
        p: BrainParams

    Returns:
        (3,)  burst velocity (deg/s), zero for negative states
    """
    e = jax.nn.relu(e)
    mag = jnp.linalg.norm(e)
    return p.g_burst * (1.0 - jnp.exp(-mag / p.e_sat_sac)) * e / (mag + 1e-6)


# ── SSM step ──────────────────────────────────────────────────────────────────

def step(x_sg, pos_delayed, target_visible, x_ni, ocr, w_est, p, noise_acc=0.0):
    """Single ODE step: target selection + burst dynamics + burst output.

    Target selection (inside step — uses brain-internal x_ni, not plant state):
        target_visible ≈ 1  (in visual field):
            e_cmd = clip(pos_delayed, −orbital_limit − x_ni,  +orbital_limit − x_ni)
        target_visible ≈ 0  (quick-phase generator — target outside ~90° visual field):
            e_cmd = −alpha_reset · (x_ni − k_center_vel·τ_ref · w_vs)
            Predictive centripetal quick phase.

    Args:
        x_sg:          (N_STATES,)  SG state vector (see module docstring)
        pos_delayed:   (3,)         delayed retinal position error (deg)
        target_visible: scalar       visual-field gate (≈1 in-field, ≈0 out-of-field)
        x_ni:          (3,)         NI net state — brain's eye-position estimate (deg)
        ocr:           scalar        torsional OCR offset (deg)
        w_est:         (3,)         velocity storage head-velocity estimate (deg/s)
        p:             BrainParams
        noise_acc:     scalar        pre-generated accumulator diffusion term

    Returns:
        dx_sg:   (N_STATES,)  state derivative
        u_burst: (3,)         saccade velocity command (deg/s)
    """
    # ── Listing's law corrections ─────────────────────────────────────────────
    eye_pos = x_ni.at[2].add(ocr)
    pos_delayed, x_ni = listing.saccade_corrections(
        eye_pos, pos_delayed, ocr, p.listing_primary)

    # ── State extraction ──────────────────────────────────────────────────────
    e_held  = x_sg[0:3]    # (3,) error estimator = residual error (e_res = e_held)
    z_opn   = x_sg[3]      # scalar OPN: 100=tonic (blocked), ≈0 or negative=paused (active)
    z_acc   = x_sg[4]      # scalar rise-to-bound accumulator
    x_ebn_R = x_sg[5:8]    # (3,) right EBN membrane potentials
    x_ebn_L = x_sg[8:11]   # (3,) left  EBN membrane potentials
    x_ibn_R = x_sg[11:14]  # (3,) right IBN membrane potentials
    x_ibn_L = x_sg[14:17]  # (3,) left  IBN membrane potentials

    # OPN firing rate (clipped — membrane can go negative but firing rate cannot)
    z_opn_fr = jnp.clip(z_opn, 0.0, 100.0)

    # ── Target selection ──────────────────────────────────────────────────────
    # In-field: clip to oculomotor range; Quick-phase: predictive centripetal reset.
    e_target = jnp.clip(pos_delayed, -p.orbital_limit - x_ni, p.orbital_limit - x_ni)

    tau_refractory = (p.threshold_acc - p.acc_burst_floor) * p.tau_acc
    x_ni_pred = x_ni - p.k_center_vel * tau_refractory * w_est
    e_center  = -p.alpha_reset * x_ni_pred
    
    refixation = target_visible;
    quick_phase = 1.0 - refixation;
    e_cur     =  refixation * e_target + quick_phase * e_center
    e_cur_mag = jnp.linalg.norm(e_cur)
    # e_held IS the residual error (e_res = e_held); no separate x_copy state.

    # ── Trigger gate (accumulator source) ────────────────────────────────────
    threshold_sac_eff = (target_visible * p.threshold_sac
                         + (1.0 - target_visible) * p.threshold_sac_qp)
    gate_err = jax.nn.sigmoid(p.k_sac * (e_cur_mag - threshold_sac_eff))

    # ── Activations ───────────────────────────────────────────────────────────────
    act_opn   = z_opn_fr / 100.0    # 0=paused (burst active), 1=tonic (burst blocked)
    act_ebn_R = burst_velocity(x_ebn_R, p)
    act_ebn_L = burst_velocity(x_ebn_L, p)
    act_ibn_R = burst_velocity(x_ibn_R, p)
    act_ibn_L = burst_velocity(x_ibn_L, p)

    # ── Burst output ──────────────────────────────────────────────────────────
    u_burst = act_ebn_R - act_ebn_L

    # ── Burst neuron dynamics ─────────────────────────────────────────────────

    # Three-gain inhibitory network (all within tau_bn):
    #   g_opn_bn:      OPN→BN multiplicative — clamping TC=tau_bn/(1+g_opn_bn). Must satisfy Heun stability:
    #                  (1+g_opn_bn)*dt/tau_bn < 2 → g_opn_bn < 9 with dt=1ms, tau_bn=5ms.
    #   g_opn_bn_hold: OPN→BN additive offset — keeps BN at −g_opn_bn_hold/(1+g_opn_bn) ≈ −0.4 between saccades.
    #   Combined equilibrium: x_eq = −g_opn_bn_hold/(1+g_opn_bn) < 0 (slightly negative, fires in ~0.4ms at trigger).
    #   g_ci:          IBN→BN contralateral, element-wise (axis-matched, normalised by g_burst).
    #   g_ibn_opn:     IBN→OPN Schmitt-trigger latch.
    inh_from_L = p.g_ci * act_ibn_L / p.g_burst   # (3,) L IBN → R BNs, one-to-one per axis
    inh_from_R = p.g_ci * act_ibn_R / p.g_burst   # (3,) R IBN → L BNs, one-to-one per axis
    opn_gain   = 1.0 + p.g_opn_bn * act_opn        # multiplicative: TC = tau_bn/(1+g_opn_bn) when tonic
    opn_inh    = p.g_opn_bn_hold * act_opn          # additive offset: 0 during saccade

    dx_ebn_R = (jax.nn.relu( e_held) - inh_from_L - opn_inh - x_ebn_R * opn_gain) / p.tau_bn
    dx_ebn_L = (jax.nn.relu(-e_held) - inh_from_R - opn_inh - x_ebn_L * opn_gain) / p.tau_bn
    dx_ibn_R = (jax.nn.relu( e_held) - inh_from_L - opn_inh - x_ibn_R * opn_gain) / p.tau_bn
    dx_ibn_L = (jax.nn.relu(-e_held) - inh_from_R - opn_inh - x_ibn_L * opn_gain) / p.tau_bn

    # ── Error estimator ───────────────────────────────────────────────────────
    # During burst:    u_burst > 0  → de_held = −u_burst (e_held decrements to 0)
    # Between saccades: BNs held negative by OPN → u_burst = 0 → de_held tracks e_cur
    # No separate x_copy → no overshoot: e_held can only decrement while BNs are positive.
    de_held = -u_burst + act_opn**2 * (e_cur - e_held) / p.tau_reset_fast

    # ── Rise-to-bound accumulator (z_acc) ────────────────────────────────────
    # OPN pause (1−act_opn) provides reliable drain for all amplitudes: even a 1° saccade
    # gets full drain when OPN commits. IBN adds amplitude-proportional contribution.
    # No equilibrium: OPN drops essentially in one step (tau_sac=1ms), IBN latches OPN
    # before z_acc can recharge → z_acc drains to floor before oscillation can develop.
    # ISI ≈ (threshold − floor)·tau_acc once floor is reached.
    ibn_drain = jnp.sum(act_ibn_R + act_ibn_L) / p.g_burst
    dz_acc = (gate_err / p.tau_acc
              - ((1.0 - act_opn) + ibn_drain) * (z_acc - p.acc_burst_floor) / p.tau_burst_drain
              + noise_acc)

    # ── OPN latch dynamics ────────────────────────────────────────────────────
    # ibn_total from rectified acts: negative (OPN-inhibited) BN states contribute zero.
    ibn_total = jnp.sum(act_ibn_R) + jnp.sum(act_ibn_L)
    charge_sac = jax.nn.sigmoid(p.k_acc * (z_acc - p.threshold_acc))
    dz_opn = (p.k_tonic_opn * (100.0 - z_opn)
              - (z_opn + p.g_opn_pause) * charge_sac
              - p.g_ibn_opn * ibn_total) / p.tau_sac

    dx_sg = jnp.concatenate([de_held,
                              jnp.array([dz_opn, dz_acc]),
                              dx_ebn_R, dx_ebn_L, dx_ibn_R, dx_ibn_L])
    return dx_sg, u_burst
