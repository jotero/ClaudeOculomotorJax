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

    e_held  — Robinson's resettable integrator: latches e_cur at trigger onset; integrates −u_burst during burst.
               Sample-and-hold: de_held = charge_sac·(e_cur−e_held)/τ_latch − u_burst.
               At trigger (charge_sac→1, τ_latch=3ms): e_held snaps to e_cur in ~3ms.
               During burst (z_acc drains → charge_sac→0): tracking off → e_held decrements by −u_burst only.
               Between saccades (charge_sac=0): frozen — stale visual delay cannot corrupt e_held.
               No overshoot: e_held monotonically decrements to 0, burst ends naturally.
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

OPN inhibition (Schmitt trigger via IBN):
    ibn_total = Σ(act_ibn_R + act_ibn_L)
    dz_opn = (k_tonic·(100−z_opn) − (z_opn+g_opn_pause)·charge_sac − g_ibn_opn·ibn_total) / τ_sac
    Accumulator is the sole initial trigger (charge_sac→1); IBN then latches OPN paused via g_ibn_opn.

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

import jax.numpy as jnp
import jax

from oculomotor.models.brain_models import listing

N_STATES  = 17  # e_held(3) + z_opn(1) + z_acc(1) + x_ebn_R(3) + x_ebn_L(3) + x_ibn_R(3) + x_ibn_L(3)
N_INPUTS  = 9   # pos_delayed(3) + target_visible(1) + x_ni(3)
N_OUTPUTS = 3   # u_burst (3,)


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

    # ── Target selection ──────────────────────────────────────────────────────
    # In-field: clip to oculomotor range; Quick-phase: predictive centripetal reset.
    e_target = jnp.clip(pos_delayed, -p.orbital_limit - x_ni, p.orbital_limit - x_ni)

    tau_refractory = (p.threshold_acc - p.acc_burst_floor) * p.tau_acc
    x_ni_pred = x_ni - p.k_center_vel * tau_refractory * w_est
    e_center  = -p.alpha_reset * x_ni_pred
    
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
    #                  (1+g_opn_bn·100)·dt/tau_bn < 2 → g_opn_bn < 0.09 with dt=1ms, tau_bn=5ms.
    #   g_opn_bn_hold: OPN→BN additive offset — keeps BN at (e_held−g_opn_bn_hold·100)/(1+g_opn_bn·100) < 0 between saccades.
    #   Combined equilibrium: x_eq = (e_held − 40)/(1+4) = −8 at fixation → IBN firmly off.
    #   g_ibn_bn:      IBN→BN contralateral, element-wise (axis-matched); g_ci/g_burst absorbed.
    #   g_ibn_opn:     IBN→OPN Schmitt-trigger latch.
    inh_from_L = p.g_ibn_bn * act_ibn_L   # (3,) L IBN → R BNs, one-to-one per axis
    inh_from_R = p.g_ibn_bn * act_ibn_R   # (3,) R IBN → L BNs, one-to-one per axis
    opn_gain   = 1.0 + p.g_opn_bn * act_opn        # multiplicative: TC = tau_bn/(1+g_opn_bn) when tonic
    opn_inh    = p.g_opn_bn_hold * act_opn          # additive offset: 0 during saccade

    dx_ebn_R = (jax.nn.relu( e_held) - inh_from_L - opn_inh - x_ebn_R * opn_gain) / p.tau_bn
    dx_ebn_L = (jax.nn.relu(-e_held) - inh_from_R - opn_inh - x_ebn_L * opn_gain) / p.tau_bn
    dx_ibn_R = (jax.nn.relu( e_held) - inh_from_L - opn_inh - x_ibn_R * opn_gain) / p.tau_bn
    dx_ibn_L = (jax.nn.relu(-e_held) - inh_from_R - opn_inh - x_ibn_L * opn_gain) / p.tau_bn

    # ── Resettable integrator (Robinson 1975) ────────────────────────────────
    # Sample-and-hold: charge_sac gates the latch; frozen between saccades.
    # At trigger (charge_sac→1, tau_latch=3ms): e_held snaps to e_cur.
    # During burst (z_acc drains → charge_sac→0): tracking off; u_burst decrements e_held.
    # Between saccades: charge_sac=0 → frozen; stale pos_delayed cannot corrupt e_held.
    charge_sac = jax.nn.sigmoid(p.k_acc * (z_acc - p.threshold_acc))
    de_held = -u_burst + charge_sac * (e_cur - e_held) / p.tau_latch

    # ── Rise-to-bound accumulator (z_acc) ────────────────────────────────────
    # IBN provides amplitude-proportional drain during burst. With g_opn_bn_hold≥orbital_limit,
    # IBN is zero between saccades (x_bn<0 → relu→0) so accumulator rises freely.
    # During burst (OPN paused): BNs drive up → IBN fires → drain >> gate_err → floor reached.
    # Drain ratio: ibn_drain ≥ 0.13 for 1° saccade → drain 39/s >> gate_err 5.6/s ✓
    # ISI ≈ (threshold − floor)·tau_acc once floor is reached.
    ibn_drain = jnp.sum(act_ibn_R + act_ibn_L) / p.g_burst
    dz_acc = (gate_err / p.tau_acc
              - ibn_drain * (z_acc - p.acc_burst_floor) / p.tau_burst_drain
              + noise_acc)

    # ── OPN latch dynamics ────────────────────────────────────────────────────
    # IBN→OPN naturally zero between saccades: with g_opn_bn_hold·100 > ~orbital_limit, BN equilibrium
    # is negative (x_bn<0) → relu→0 → ibn_total=0 → no direct OPN suppression.
    # During burst: act_opn drops → opn_inh drops → x_bn rises → IBN fires → latches OPN.
    # At burst end: e_held→0 → src→0 → x_bn driven negative by recovering OPN → IBN decays.
    # charge_sac already computed above (shared with de_held latch).
    ibn_total = jnp.sum(act_ibn_R) + jnp.sum(act_ibn_L)
    dz_opn = (p.k_tonic_opn * (100.0 - z_opn)
              - (z_opn + p.g_opn_pause) * charge_sac
              - p.g_ibn_opn * ibn_total) / p.tau_sac

    dx_sg = jnp.concatenate([de_held,
                              jnp.array([dz_opn, dz_acc]),
                              dx_ebn_R, dx_ebn_L, dx_ibn_R, dx_ibn_L])
    return dx_sg, u_burst
