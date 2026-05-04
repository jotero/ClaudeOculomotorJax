"""Vergence SSM — Schor (1986) dual-integrator + Robinson direct phasic + Zee (1992) SVBN burst.

Steps 1 + 2 of the vergence rebuild:
  Step 1 — slow disparity vergence (dual integrator + direct path).
  Step 2 — saccadic vergence burst (Zee 1992 SVBN), gated by OPN release.

Architecture
────────────
Schor & Kotulak (1986) dual-controller model: a fast phasic integrator (sub-second
to ~2 s) summed with a slow tonic integrator (tens of seconds) that adapts the
resting bias.  Both leaky and driven by binocular disparity.  A direct phasic
feedthrough cancels the plant lag (Robinson 1975 NI pulse-step):

    u_phasic = K_phasic_verg · e_disp                    (deg/s, no state)
    dx_fast  = -x_fast/τ_fast + K_fast · e_disp + u_svbn (fast integrator + SVBN boost)
    dx_slow  = -x_slow/τ_slow + K_slow · e_disp          (slow integrator)
    u_verg   = tonic_vec + x_fast + x_slow + τ_vp · u_phasic

Zee SVBN burst (added to fast-integrator drive during OPN release):

    Asymmetric saturating gain (Zee 1992 Table 1: peak conv ≈ 50°/s for 10°,
    peak div ≈ 12°/s for 2.5°):
        u_svbn = z_act · sign(disp) · g · (1 − exp(−|disp|/X))     (deg/s)
        g, X chosen separately for convergence (disp > 0) and divergence.

    Fires whenever OPN pauses (z_act ≈ 1) — same gate that releases the
    saccadic burst neurons.  Drive is by *current* disparity (not a held value),
    so it self-terminates as eyes converge during the saccade.  By boosting
    the fast integrator (rather than adding a position pulse), the burst's
    contribution persists naturally after OPN re-engages.

State layout (9 slots in brain.x_verg, kept at 9 so brain_model is unchanged):
    [0:3]   x_fast    fast integrator [H, V, T] (deg, deviation from tonic)
    [3:6]   x_slow    slow integrator [H, V, T] (deg, deviation from tonic)
    [6:9]   x_copy    integrated SVBN burst for observability (no role in u_verg)

Tonic baseline:
    tonic_verg (deg) is the dark-vergence resting level (~3.67° for 1 m, IPD 64 mm).
    Enters the OUTPUT only — the integrators are deviations from tonic, both zero
    at rest.  At rest (e_disp = 0) the system holds at tonic_verg.

Step 1 closure analysis:
    SS (e_disp = e_ss, all derivatives zero):
        x_fast = K_fast · τ_fast · e_ss = G_fast · e_ss
        x_slow = K_slow · τ_slow · e_ss = G_slow · e_ss
        u_verg = tonic + (G_fast + G_slow + τ_vp · K_phasic) · e_ss
    For 9° geometric demand and tonic = 3.67° → vergence offset 5.33°
        e_ss = 5.33 / G_total
    With G_fast=20, G_slow=30, τ_vp·K_phasic ≈ 0.15: G_total ≈ 50, e_ss ≈ 0.11° (~6 arcmin) ✓

References
──────────
Schor CM, Kotulak JC (1986) Vision Res 26:927–942 — dual-controller framework
Read JCA, Kaspiris-Rousellis C et al. (2022) J Vision 22(9):4 — modern dual loop
Robinson DA (1975) Oculomotor control signals — NI pulse-step / direct path
Rashbass C, Westheimer G (1961) J Physiol 159:339–360 — vergence dynamics
Zee DS et al. (1992) J Neurophysiol 68:1624–1641 — saccade burst (Step 2)
"""

import jax.numpy as jnp


N_STATES  = 9   # [x_fast(3) | x_slow(3) | x_copy(3)]
N_INPUTS  = 13  # see _IDX_INPUT_* below
N_OUTPUTS = 3

# Sub-state slices within x_verg
_IDX_FAST = slice(0, 3)
_IDX_SLOW = slice(3, 6)
_IDX_COPY = slice(6, 9)

# Bundled-input layout — match the SSM convention: step(x, u, theta).
_IDX_INPUT_E_DISP            = slice(0, 3)    # delayed binocular disparity (deg)
_IDX_INPUT_ACA_DRIVE         = 3              # AC/A drive (deg); scalar
_IDX_INPUT_VERG_RATE_TVOR    = slice(4, 7)    # T-VOR vergence rate (deg/s, [H,V,T])
_IDX_INPUT_Z_ACT             = 7              # OPN gate (0=idle, 1=saccade); scalar
_IDX_INPUT_EYE_HV            = slice(8, 10)   # gaze [H,V] (deg); reserved for L2
_IDX_INPUT_SCENE_DISP_RATE   = slice(10, 13)  # per-eye scene-flow diff (m/s); pass zeros if N/A

# x_copy decays slowly between saccades so accumulated burst contribution doesn't drift forever
_TAU_COPY_RESET = 30.0   # s


def step(x_verg, u, brain_params):
    """Vergence step: dual integrator + direct phasic + tonic baseline.

    Args:
        x_verg:       (9,)  [x_fast(3) | x_slow(3) | x_copy(3)]
        u:            (13,) bundled input vector:
                            [_IDX_INPUT_E_DISP]          = binocular disparity (3,) (deg) = pos_L − pos_R
                            [_IDX_INPUT_ACA_DRIVE]       = AC/A drive (deg); positive = converging
                            [_IDX_INPUT_VERG_RATE_TVOR]  = T-VOR vergence rate (3,) (deg/s, [H,V,T])
                            [_IDX_INPUT_Z_ACT]           = OPN gate (0=idle, 1=saccade); SVBN trigger
                            [_IDX_INPUT_EYE_HV]          = gaze [H,V] (deg); reserved for L2
                            [_IDX_INPUT_SCENE_DISP_RATE] = per-eye scene-flow diff (3,) (m/s);
                                                           pass zeros if no depth/parallax evidence
        brain_params: BrainParams (reads tonic_verg, tau_vp, tau_verg_fast, tau_verg_slow,
                                          K_phasic_verg, K_verg_fast, K_verg_slow,
                                          g_svbn_conv, X_svbn_conv, g_svbn_div, X_svbn_div,
                                          K_visual_verg, ipd_brain)

    Returns:
        dx_verg: (9,)  state derivative
        u_verg:  (3,)  vergence position command (deg) → split ±½ in final_common_pathway
    """
    e_disp          = u[_IDX_INPUT_E_DISP]
    ac_a_drive      = u[_IDX_INPUT_ACA_DRIVE]
    verg_rate_tvor  = u[_IDX_INPUT_VERG_RATE_TVOR]
    z_act           = u[_IDX_INPUT_Z_ACT]
    _eye_hv         = u[_IDX_INPUT_EYE_HV]   # reserved for future L2 cyclovergence
    scene_disp_rate = u[_IDX_INPUT_SCENE_DISP_RATE]

    x_fast = x_verg[_IDX_FAST]
    x_slow = x_verg[_IDX_SLOW]
    x_copy = x_verg[_IDX_COPY]

    # Disparity drives the dual integrator + direct phasic path (closed loop with retina).
    # AC/A drive (from accommodation cross-link) does NOT enter here — it bypasses the
    # high-gain integrator stack and is added to u_verg directly below.  Reason: in
    # open-loop conditions (cover test, dark) AC/A would otherwise be amplified by
    # G_total ≈ 15 and run vergence to the orbital walls.  Direct addition gives
    # unity gain for AC/A (clinically: 1.5 D over dark focus → ~7° convergence per Hofstetter).
    e_total = e_disp

    # Phasic direct path — provides plant-canceling pulse (no state)
    u_phasic = brain_params.K_phasic_verg * e_total                    # deg/s

    # ── Zee (1992) SVBN burst — saccade-gated, asymmetric saturating gain ──
    # Conv much stronger than div (Zee Table 1: peak ~50°/s for 10° conv,
    # ~12°/s for 2.5° div). Fires only when OPN pauses (z_act ≈ 1).
    disp_h    = e_total[0]
    is_conv   = (disp_h > 0).astype(jnp.float32)
    g_eff     = is_conv * brain_params.g_svbn_conv + (1.0 - is_conv) * brain_params.g_svbn_div
    X_eff     = is_conv * brain_params.X_svbn_conv + (1.0 - is_conv) * brain_params.X_svbn_div
    u_svbn_h  = z_act * jnp.sign(disp_h) * g_eff * (1.0 - jnp.exp(-jnp.abs(disp_h) / X_eff))
    u_svbn    = jnp.array([u_svbn_h, 0.0, 0.0])                        # horizontal-only

    # Fast integrator — sub-second tracking; SVBN boosts during saccade for persistence
    dx_fast = -x_fast / brain_params.tau_verg_fast + brain_params.K_verg_fast * e_total + u_svbn

    # Slow integrator — tonic adapter; T-VOR 3D vergence rate adds open-loop integration
    # so eyes converge as head approaches near target (H component) and pick up L2 cross-
    # coupling at off-primary gaze (V, T components — usually small).
    # Visual evidence: scene_disp_rate is per-eye scene-flow difference.  In a uniform
    # depthless scene it's 0, providing a damping signal that constrains T-VOR-driven
    # vergence drives when the visual world doesn't agree.  Convert from m/s differential
    # to deg/s vergence rate via 1/IPD (small-angle): per-eye flow difference of v m/s
    # implies vergence rate of v/IPD rad/s if interpreted as approaching a target.
    # Project per-eye flow differential onto the horizontal vergence axis.
    # For pure translation in a uniform scene scene_disp_rate is 0 → damps
    # spurious T-VOR drift. Caller passes zeros if scene/depth signal absent.
    K_visual_verg     = brain_params.K_visual_verg
    visual_evidence_H = scene_disp_rate[0] / brain_params.ipd_brain * jnp.degrees(1.0)  # deg/s
    verg_rate_tvor    = verg_rate_tvor.at[0].add(K_visual_verg * (visual_evidence_H - verg_rate_tvor[0]))
    dx_slow = (-x_slow / brain_params.tau_verg_slow
               + brain_params.K_verg_slow * e_total
               + verg_rate_tvor)

    # x_copy: integrated SVBN burst for observability; slow decay between saccades
    dx_copy = u_svbn - x_copy / _TAU_COPY_RESET

    # Output: tonic baseline + integrators + direct path (phasic + SVBN burst).
    # Robinson NI structure: SVBN appears in BOTH the integrator (for lasting position
    # memory after the saccade) and the direct path (for fast within-saccade onset, scaled
    # by τ_vp like the phasic drive).  During saccade both contribute → fast burst.
    # After saccade z_act → 0 so SVBN drops out of direct path; x_fast keeps the integral.
    tonic_vec    = jnp.array([brain_params.tonic_verg, 0.0, 0.0])
    aca_vec      = jnp.array([ac_a_drive, 0.0, 0.0])                 # AC/A bypass (direct, not integrated)
    u_verg_drive = u_phasic + u_svbn                                 # total velocity drive
    u_verg       = tonic_vec + aca_vec + x_fast + x_slow + brain_params.tau_vp * u_verg_drive

    dx_verg = jnp.concatenate([dx_fast, dx_slow, dx_copy])
    return dx_verg, u_verg
