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
The Robinson residual (e_held − x_copy) drives the burst and always decreases
monotonically to zero at burst end, regardless of target velocity.

State:  x_sg = [x_copy (3) | z_ref (1) | e_held (3) | z_opn (1) | z_acc (1)]   (9 states)

    x_copy  — internal copy integrator; mirrors NI during burst, resets between
    z_ref   — OPN refractory state: 0=ready, rises to ~0.7 after each saccade,
               decays with τ_ref (~150 ms); bistable OPN gate blocks burst while
               z_ref > threshold_ref
    e_held  — sample-and-hold: tracks e_pos_delayed between saccades (τ_hold=5ms),
               frozen during burst (z_opn=0)
    z_opn   — OPN state: 100=tonic firing (burst blocked), 0=paused (burst active)
               drops to 0 (1ms) when accumulator crosses threshold (z_acc > threshold_acc)
               recovers to 100 when z_ref crosses threshold_sac_release
    z_acc   — rise-to-bound accumulator: integrates gate_err × gate_opn with τ_acc.
               Drains when gate_err=0 or z_opn=0 (burst active).  Noise-robust: brief
               spikes (< τ_acc) cannot cross threshold_acc.  Also delays z_opn drop by
               τ_acc after gate_err first fires, giving the visual cascade time to settle
               before e_held freezes.

Input:  u_sg  = e_pos_delayed (3,)   delayed retinal position error (deg)
Output: u_burst (3,)                  saccade velocity command (deg/s)

──────────────────────────────────────────────────────────────────────────────
Gates
──────────────────────────────────────────────────────────────────────────────
    gate_err  = σ(k · (|e_pos_delayed| − threshold_sac))
        Current retinal error trigger.  Checks live error for new targets.

    gate_opn  = σ(−k_ref · (z_ref − threshold_ref))
        Bistable OPN gate (hard switch at z_ref = threshold_ref):
            z_ref < threshold_ref → gate_opn ≈ 1  (burst allowed)
            z_ref > threshold_ref → gate_opn ≈ 0  (refractory)

    gate_res  = σ(k · (|e_held − x_copy| − threshold_stop))
        Stopping gate on held residual.  Because e_held is frozen during
        the burst, this residual decreases monotonically 0 at burst end.

    gate_dir  = relu(ê_held · ê_res)
        Suppresses burst if copy overshoots held target.

    gate_active = z_sac · gate_res · gate_dir
        z_sac (not gate_opn) sustains the burst — breaks the circularity
        between gate_opn and e_held freeze.

──────────────────────────────────────────────────────────────────────────────
OPN latch (z_opn) — the key to the ballistic design
──────────────────────────────────────────────────────────────────────────────
    z_opn = 100 at rest (tonic OPN firing); z_opn = 0 during saccade (OPN paused)

    dz_opn/dt = (100−z_opn) · release_sac / τ_sac
              −       z_opn  · charge_sac  / τ_sac

    charge_sac  = σ(k_acc · (z_acc − threshold_acc)):  accumulator crossed threshold
    release_sac = σ(k_ref · (z_ref − threshold_sac_release)): refractory reached peak

    Sequence:
        1. Target appears → gate_err→1, gate_opn≈1 → z_acc accumulates
        2. z_acc > threshold_acc → charge_sac→1 → z_opn→0 in ~1ms (OPN pauses)
        3. e_held freezes (z_opn/100 ≈ 0 → tracking term off)
        4. Burst runs: gate_active = (1−z_opn/100) · gate_res · gate_dir ≈ 1
        5. x_copy integrates toward e_held → e_res decreases → gate_res→0
        6. z_ref charges to ~0.7 in ~1ms
        7. z_ref > threshold_sac_release → release_sac→1 → z_opn→100 (OPN resumes)
        8. e_held unfreezes → tracks epd → prepares next saccade target
        9. z_ref decays below threshold_ref in ~175ms → gate_opn→1 → z_acc can fire again

    Initialization: z_opn=100 (set in make_x0).  Between saccades z_opn stays at 100
    because both charge_sac and release_sac are 0 → dz_opn=0.

──────────────────────────────────────────────────────────────────────────────
Refractory (OPN) dynamics
──────────────────────────────────────────────────────────────────────────────
    dz_ref/dt = (1−z_ref) · charge / τ_charge  −  z_ref / τ_ref
    charge = z_sac · (1 − gate_res)

    At burst end: gate_res→0, gate_opn≈1 → charge≈1 → z_ref charges to ~0.7 in 1ms.
    During refrac: gate_opn→0 → charge=0 → z_ref decays freely with τ_ref=150ms.
    z_ref decays from 0.7 to threshold_ref=0.1 in ~0.15·ln(7)=0.29s → ~180ms ISI.

──────────────────────────────────────────────────────────────────────────────
Burst nonlinearity  (main sequence)
──────────────────────────────────────────────────────────────────────────────
    Magnitude: g_burst · (1 − exp(−|e_res| / e_sat_sac))
        At onset x_copy=0, |e_res|=|e_held| → main sequence.  Tapers to 0 as
        x_copy→e_held → bell-shaped velocity profile, clean stop.

    Direction: e_res / |e_res|   (held residual direction)

──────────────────────────────────────────────────────────────────────────────
Output insertion (in simulator)
──────────────────────────────────────────────────────────────────────────────
    dx_ni  += u_burst            NI integrates burst → holds post-saccade position
    u_p    += tau_p · u_burst    velocity pulse → plant (cancels LP lag)

Parameters
──────────
    g_burst        burst ceiling (deg/s)              default 700.0
    threshold_sac  retinal-error trigger (deg)        default 0.5
    threshold_stop residual stopping threshold (deg)  default 0.1
    k_sac          sigmoid steepness (1/deg)          default 200.0
    e_sat_sac      main-sequence saturation (deg)     default 7.0
    tau_i          NI leak TC — shared with NI (s)    default 25.0
    tau_reset_fast copy integrator reset TC (s)       default 0.05
    tau_ref        refractory decay TC (s)            default 0.15
    tau_ref_charge OPN charge TC (s)                  default 0.001
    k_ref          bistable OPN steepness             default 50.0
    threshold_ref  OPN refractory threshold           default 0.1
    tau_hold       sample-and-hold tracking TC (s)    default 0.005
    tau_sac        saccade latch TC (s)               default 0.001
    tau_acc        accumulator rise TC (s)             default 0.060
    tau_drain      accumulator drain TC (s)            default 0.080
    threshold_acc  accumulator trigger threshold       default 0.5
    k_acc          accumulator sigmoid steepness       default 50.0
"""

import jax.numpy as jnp
import jax

N_STATES  = 9   # x_copy(3) + z_ref(1) + e_held(3) + z_sac(1) + z_acc(1)
N_INPUTS  = 9   # pos_delayed(3) + target_visible(1) + x_ni(3) → e_cmd(3) computed internally
N_OUTPUTS = 3   # u_burst (3,)


# ── Burst velocity (magnitude + direction) ────────────────────────────────────

def burst_velocity(e_residual, brain_params):
    """Burst velocity from held residual.  Main-sequence nonlinearity, no gating.

    Args:
        e_residual: (3,)  e_held − x_copy
        theta:      dict  model parameters

    Returns:
        u_burst_raw: (3,)  burst velocity vector (deg/s), gates NOT applied
    """
    g_burst   = brain_params.g_burst
    e_sat_sac = brain_params.e_sat_sac
    e_res_mag = jnp.linalg.norm(e_residual)
    burst_mag = g_burst * (1.0 - jnp.exp(-e_res_mag / e_sat_sac))
    e_res_dir = e_residual / (e_res_mag + 1e-6)
    return burst_mag * e_res_dir


# ── SSM step ──────────────────────────────────────────────────────────────────

def step(x_sg, pos_delayed, target_visible, x_ni, w_est, brain_params, noise_acc=0.0):
    """Single ODE step: target selection + burst dynamics + burst output.

    Target selection (inside step — uses brain-internal x_ni, not plant state):
        target_visible ≈ 1  (in visual field):
            e_cmd = clip(pos_delayed, −orbital_limit − x_ni,  +orbital_limit − x_ni)
            Parks the eye at the oculomotor boundary when target is visible but
            beyond mechanical reach; otherwise a normal saccade error.
        target_visible ≈ 0  (quick-phase generator — target outside ~90° visual field):
            e_cmd = −alpha_reset · (x_ni − k_center_vel·τ_ref · w_vs)
            Predictive centripetal quick phase: aims past centre to compensate for
            slow-phase drift during the refractory period.  Prediction only applies
            here (not in the in-field path) so voluntary saccades are unaffected.

    Args:
        x_sg:         (N_STATES,)  [x_copy(3) | z_ref(1) | e_held(3) | z_sac(1) | z_acc(1)]
        pos_delayed:  (3,)         delayed retinal position error (deg, raw)
        target_visible: scalar       visual-field gate (≈1 in-field, ≈0 out-of-field)
        x_ni:         (3,)         neural integrator state — brain's eye-position estimate (deg)
        w_est:        (3,)         velocity storage output — head velocity estimate (deg/s)
        brain_params: BrainParams

    Returns:
        dx_sg:   (N_STATES,)  state derivative
        u_burst: (3,)         saccade velocity command (deg/s)
    """
    x_copy = x_sg[:3]    # (3,) internal copy integrator
    z_ref  = x_sg[3]     # scalar refractory state
    e_held = x_sg[4:7]   # (3,) held target error (frozen during burst)
    z_opn  = x_sg[7]     # scalar OPN state: 100=tonic (burst blocked), 0=paused (burst active)
    z_acc  = x_sg[8]     # scalar rise-to-bound accumulator

    # Firing-rate clip: IBN inhibition drives z_opn membrane potential below 0,
    # but OPN firing rate cannot go negative — clip before computing gates.
    z_opn_fr = jnp.clip(z_opn, 0.0, 100.0)
    z_act = (100.0 - z_opn_fr) / 100.0   # 0=idle, 1=saccade active
    z_idl = z_opn_fr / 100.0             # 0=during saccade, 1=idle

    # ── Target selection ──────────────────────────────────────────────────────
    # Use x_ni (NI state) as brain's proxy for current eye position.

    orbital_limit = brain_params.orbital_limit
    alpha_reset   = brain_params.alpha_reset

    # In-field saccade: aim for raw retinal error; clip to orbital range at current position.
    # No velocity prediction here — voluntary saccades go where the retina says.
    e_target = jnp.clip(pos_delayed, -orbital_limit - x_ni, orbital_limit - x_ni)

    # ── Quick-phase generator (out-of-field / no target) ──────────────────────
    # Predictive centripetal reset: aim past centre to compensate for slow-phase
    # drift that accumulates during the refractory period (tau_ref).
    #   x_ni_dot ≈ −w_vs  (VOR or spontaneous slow phase driven by VS)
    #   x_ni_pred = x_ni − look_ahead · w_vs   (where NI will be when eye lands)
    #   e_center  = −alpha_reset · x_ni_pred    (aim at predicted post-drift position)
    # look_ahead = k_center_vel · tau_ref:  0 = no prediction, 1 = full τ_ref look-ahead.
    look_ahead = brain_params.k_center_vel * brain_params.tau_ref
    x_ni_pred  = x_ni - look_ahead * w_est
    e_center   = -alpha_reset * x_ni_pred

    # Blend by gate: target_visible=1 → track target; target_visible=0 → center
    e_cur     = target_visible * e_target + (1.0 - target_visible) * e_center
    e_cur_mag = jnp.linalg.norm(e_cur)

    e_res     = e_held - x_copy          # ballistic residual (against HELD target)
    e_res_mag = jnp.linalg.norm(e_res)
    e_res_dir = e_res / (e_res_mag + 1e-6)

    # ── Gates ─────────────────────────────────────────────────────────────────

    k_sac          = brain_params.k_sac
    threshold_sac  = brain_params.threshold_sac
    threshold_stop = brain_params.threshold_stop
    k_ref          = brain_params.k_ref
    threshold_ref  = brain_params.threshold_ref

    # Trigger: is current retinal error large enough to warrant a saccade?
    # Quick phases (target_visible≈0) use a higher threshold — require larger drift error
    # before starting to accumulate, reducing spurious centripetal resets during slow phases.
    threshold_sac_eff = (target_visible * threshold_sac
                         + (1.0 - target_visible) * brain_params.threshold_sac_qp)
    gate_err = jax.nn.sigmoid(k_sac * (e_cur_mag - threshold_sac_eff))

    # OPN bistable gate: hard switch at z_ref = threshold_ref.
    gate_opn = jax.nn.sigmoid(-k_ref * (z_ref - threshold_ref))

    # Stopping gate: has the INTERNAL COPY reached the HELD TARGET?
    gate_res = jax.nn.sigmoid(k_sac * (e_res_mag - threshold_stop))

    # Direction gate: prevent backward burst if copy overshoots held target.
    e_held_dir = e_held / (jnp.linalg.norm(e_held) + 1e-6)
    gate_dir   = jax.nn.relu(jnp.dot(e_held_dir, e_res_dir))

    # Within-saccade burst gate: stopping (gate_res) and direction (gate_dir).
    # z_sac is NOT in gate_active_burst — it controls the copy dynamics separately.
    gate_active_burst = gate_res * gate_dir

    # ── Burst output ──────────────────────────────────────────────────────────

    u_burst_raw = burst_velocity(e_res, brain_params)
    u_burst     = z_act * gate_active_burst * u_burst_raw

    # ── Copy integrator ───────────────────────────────────────────────────────
    # Four regimes via z_sac and gate_active_burst:
    #
    #   z_sac=1, gate_active_burst≈1  (normal burst, x_copy < e_held):
    #       NI dynamics driven by burst → x_copy integrates toward e_held
    #
    #   z_sac=1, gate_active_burst≈0  (saccade ended OR overshoot x_copy > e_held):
    #       Spring toward e_held with τ_fast → x_copy settles at e_held, not 0.
    #       This prevents the old bug where (1-gate_active)*reset sent x_copy→0
    #       during overshoot, creating a rapid oscillation.
    #
    #   z_sac=0 (idle / refractory):
    #       Fast reset to 0 with τ_fast → clears copy for next saccade.
    #
    # Note: (1 − z_sac)² not (1 − z_sac) for the idle reset — same reason as de_held.
    # z_sac settles at ~0.955 (not 1.0) due to ODE step size, so (1−z_sac)=0.045 lets
    # the idle-reset term bleed at 4.5%.  For large x_copy the reset force 0.045·x/τ_fast
    # exceeds the burst force, preventing convergence to e_held:
    #     equilibrium e_res = 0.9·x_copy / (95.5·tau_fast) = 0.009·x_copy
    # For a 30° saccade (x_copy≈26°): e_res* = 0.24° > threshold_stop = 0.1°  → burst stuck.
    # Squaring drives bleedthrough to (0.045)² = 0.002 → e_res* = 0.011°  ✓

    tau_fast = brain_params.tau_reset_fast
    A_ni     = (-1.0 / brain_params.tau_i) * jnp.eye(3)
    # B = I (identity — omitted)
    dx_copy  = (z_act * (gate_active_burst * (A_ni @ x_copy + u_burst_raw)
                         + (1.0 - gate_active_burst) * (-(x_copy - e_held) / tau_fast))
               + z_idl**2 * (-x_copy / tau_fast))

    # ── Sample-and-hold ───────────────────────────────────────────────────────
    # z_sac=1 (burst active):   de_held ≈ 0  → e_held frozen at onset value
    # z_sac=0 (idle/refractory): e_held tracks e_cur with τ_hold → samples next target
    #
    # (1 - z_sac)² instead of (1 - z_sac): z_sac never numerically reaches 1.0
    # (settles at ~0.955 due to ODE step size), so (1-z_sac)=0.045 allows 9/s
    # leakthrough with tau_hold=0.005 — enough to chase a 60 deg/s target.
    # Squaring drives leakthrough to (0.045)²/0.005 = 0.4/s: effectively frozen.
    # Between saccades z_sac≈0 so (1-0)²=1: tracking rate unchanged.

    tau_hold = brain_params.tau_hold
    de_held  = z_idl**2 * (e_cur - e_held) / tau_hold

    # ── Refractory (OPN) dynamics ─────────────────────────────────────────────
    # charge = z_act · (1 − gate_active_burst)
    #
    #   Key insight: charge fires whenever the BURST IS INACTIVE, for any reason:
    #     • Normal completion:   gate_res → 0       (x_copy reached e_held)
    #     • Overshoot stop:      gate_dir → 0       (x_copy passed e_held)
    #   Using gate_active_burst = gate_res × gate_dir instead of gate_res alone
    #   prevents the OPN from getting stuck when overshoot stops the burst before
    #   gate_res can reach 0 — the old (1−gate_res) condition left z_ref uncharged
    #   in the overshoot case, so z_opn never returned to 100.
    #
    #   During burst:          gate_active_burst ≈ 1  →  charge ≈ 0   (protected)
    #   At landing (any stop): gate_active_burst → 0, z_act = 1  →  charge = 1
    #                          z_ref charges to ~1 in τ_ref_charge ≈ 1ms
    #   After z_ref → 1: release_sac → 1 → z_opn → 100 (OPN resumes)
    #   During refrac:  z_act = 0  →  charge = 0, z_ref decays with τ_ref
    #   At rest:        z_act = 0  →  charge = 0

    tau_ref_charge = brain_params.tau_ref_charge
    tau_ref        = brain_params.tau_ref
    charge         = z_act * (1.0 - gate_active_burst)
    dz_ref         = (1.0 - z_ref) * charge / tau_ref_charge  -  z_ref / tau_ref

    # ── Rise-to-bound accumulator (z_acc) ────────────────────────────────────
    # Integrates gate_err × gate_opn; drains when gate is off or burst active.
    # z_sac fires only when z_acc crosses threshold_acc — requires sustained
    # sub-threshold error for ~τ_acc before a saccade is committed.
    #
    # Benefits:
    #   1. Noise robustness: brief spikes (< τ_acc) cannot trigger a saccade.
    #   2. Cascade settling: during the τ_acc accumulation window z_sac=0 so
    #      e_held keeps tracking the still-rising visual cascade.  When z_sac
    #      finally fires, e_held holds a more accurate estimate of target angle.

    tau_acc       = brain_params.tau_acc
    tau_drain     = brain_params.tau_drain
    threshold_acc = brain_params.threshold_acc
    k_acc         = brain_params.k_acc

    # Accumulate while gate is on AND OPN is tonic (not paused); drain otherwise.
    # noise_acc is a pre-generated diffusion term for RT variability (units: 1/s, scaled by sqrt(dt)).
    gate_drive = gate_err * gate_opn * z_idl
    dz_acc     = gate_drive / tau_acc  -  z_acc / tau_drain  +  noise_acc

    # ── Saccade latch dynamics ────────────────────────────────────────────────
    # z_sac fires (fast, 1ms) when accumulator crosses threshold.
    # z_opn recovers to 100 when z_ref crosses threshold_sac_release ≫ threshold_ref.

    tau_sac               = brain_params.tau_sac
    threshold_sac_release = brain_params.threshold_sac_release
    charge_sac  = jax.nn.sigmoid(k_acc * (z_acc - threshold_acc))           # fires when accumulated
    release_sac = jax.nn.sigmoid(k_ref * (z_ref - threshold_sac_release))  # fires when refractory
    # IBN inhibitory overshoot: drive z_opn toward −g_opn_pause when charge fires.
    # State can go negative (hyperpolarised); firing rate is clipped at 0 above (z_opn_fr).
    g_opn_pause = brain_params.g_opn_pause
    dz_opn      = ((100.0 - z_opn) * release_sac  -  (z_opn + g_opn_pause) * charge_sac) / tau_sac

    dx_sg = jnp.concatenate([dx_copy, jnp.array([dz_ref]), de_held,
                              jnp.array([dz_opn]), jnp.array([dz_acc])])
    return dx_sg, u_burst
