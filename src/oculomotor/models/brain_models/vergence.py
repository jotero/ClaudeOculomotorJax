"""Vergence SSM — single leaky integrator with dual-range nonlinear drive.

Drives disconjugate eye movements to align both foveas on a binocular target.

Architecture:
    One integrator state x_verg holds the tonic vergence position memory.
    The input is a piecewise-linear (nonlinear) combination of two clipped
    versions of e_disp, implementing dual-range gain:

            e_fus  = clip(e_disp, ±disp_max_verg_fus)    ← fusional range  (~±1 deg)
        e_prox = clip(e_disp, ±disp_max_verg_prox)  ← full range      (~±20 deg)
        e_pred = K_verg · e_fus + K_verg_prox · e_prox   ← deg/s drive

    Gain profile (deg/s per deg disparity):
        |e_disp| < 1 deg:    K_verg + K_verg_prox  [high — fusional + proximal active]
        1 < |e_disp| < 20:   K_verg_prox           [lower — proximal only, fusional saturated]
        |e_disp| > 20:       0                      [fully saturated]

Dynamics:
    dx_verg  = −(x_verg − phoria) / τ  +  e_pred
    u_verg   = x_verg + K_phasic · e_fus          ← phasic feedthrough from fusional range only

Phasic term (K_phasic · e_fus):
    Provides fast vergence onset for small disparities (within fusional range).
    Saturates at ±K_phasic · e_sat_fus for large disparities — no phasic overshoot
    on large depth steps where the integrator drives most of the response.

Steady-state gain (approximate, within fusional range):
    CL gain ≈ G / (1 + G)   where G ≈ (K_verg + K_verg_prox) · τ + K_phasic
    With K_verg=5, K_verg_prox=1, τ=25, K_phasic=1:  G ≈ 151 → gain > 99 %

State:
    x_verg = [H_verg, V_verg, torsional_verg]  (3,)  deg
    Positive H = converged (eyes rotated inward for near target).

Parameters:
    K_verg         — fusional integration gain (1/s).  Default 5.
                     Combined gain in fusional range = K_verg + K_verg_prox ≈ 6 /s →
                     convergence TC ≈ 160 ms. [Rashbass & Westheimer 1961]
    K_verg_prox    — proximal/full-range integration gain (1/s).  Default 1.
                     Active alone outside fusional range: TC ≈ 1 s.
                     [Judge & Miles 1985: slow vergence ~500 ms–1 s]
    K_phasic_verg  — phasic feedthrough (dim'less); default 1.
                     Applied to fusional clip only → fast onset for small steps.
    tau_verg       — leak TC (s); default 25 s.  Stable vergence hold; drifts to phoria
                     slowly when fusion is lost. [Schor 1979]
    disp_max_verg_fus  — fusional disparity saturation (deg); default 1 deg.
                         Panum's fusional area ~±0.5–1 deg horizontally. [Jones 1980]
    disp_max_verg_prox — proximal disparity saturation (deg); default 20 deg.
                         Full physiological vergence range ~0–25 deg (≈15 deg at 40 cm,
                         IPD 6.3 cm). [Hung & Semmlow 1980]
    phoria         — (3,) resting vergence (deg); tonic setpoint in absence of fusion.
                     Orthophoria = [0,0,0]; esophoria > 0; exophoria < 0.

References:
    Schor CM (1979) Vision Res 19:1359–1367          — dual-range vergence model
    Rashbass C, Westheimer G (1961) J Physiol 159:361–364  — vergence TC ~160 ms
    Jones R (1980) Am J Optom Physiol Opt 57:636–645  — fusional range ~±1 deg
    Hung GK, Semmlow JL (1980) IEEE Trans Biomed Eng 27:722–728  — vergence dynamics
    Judge SJ, Miles FA (1985) Exp Brain Res 60:184–203  — proximal/tonic TC ~500 ms
"""

import jax.numpy as jnp

N_STATES  = 3   # x_verg: [H_verg, V_verg, torsional_verg]  (deg)
N_INPUTS  = 3   # e_disp = pos_delayed_L − pos_delayed_R (deg)
N_OUTPUTS = 3   # u_verg: vergence position command (deg)


def step(x_verg, e_disp, brain_params):
    """Single-integrator vergence controller with dual-range nonlinear drive.

    Args:
        x_verg:       (3,)   vergence position memory (deg); positive = converged
        e_disp:       (3,)   binocular disparity = pos_delayed_L − pos_delayed_R (deg)
                             Gated by target_present via the retinal cascade.
        brain_params: BrainParams  (reads K_verg, K_verg_prox, K_phasic_verg, tau_verg,
                                         disp_max_verg_fus, disp_max_verg_prox, phoria)

    Returns:
        dx_verg: (3,)  dx_verg/dt  (deg/s)
        u_verg:  (3,)  vergence position command (deg) → split ±½ in brain_model
    """
    # ── Dual-range nonlinear gain ───────────────────────────────────────────────
    # e_disp > 0: need to converge more (target closer than current fixation).
    # e_disp < 0: need to diverge (target farther).
    # x_verg > 0: currently commanding convergence (motor output to plant).
    #
    # e_fus:  disparity clipped to fusional range (±disp_max_verg_fus, ~±1 deg).
    #         High-gain path: drives fast convergence for fine disparity.
    # e_prox: disparity clipped to full vergence range (±disp_max_verg_prox, ~±20 deg).
    #         Low-gain path: drives slow convergence for large depth steps.
    #
    # Gain profile (effective deg/s per deg of disparity):
    #   |e_disp| < 1 deg:   K_verg + K_verg_prox ≈ 6/s  → TC ~160 ms  [Rashbass & Westheimer 1961]
    #   1 < |e_disp| < 20:  K_verg_prox          ≈ 1/s  → TC ~1 s     [Judge & Miles 1985]
    # CL gain ≈ K_total·τ / (K_total·τ + 1 + K_phasic) ≈ 150/152 ≈ 99 % without state correction.
    e_fus  = jnp.clip(e_disp, -brain_params.disp_max_verg_fus,  brain_params.disp_max_verg_fus)
    e_prox = jnp.clip(e_disp, -brain_params.disp_max_verg_prox, brain_params.disp_max_verg_prox)

    e_pred = brain_params.K_verg * e_fus + brain_params.K_verg_prox * e_prox

    # Leak toward phoria (resting vergence when fusion is absent) with TC = tau_verg.
    dx_verg = -(1.0 / brain_params.tau_verg) * (x_verg - brain_params.phoria) + e_pred

    # Phasic feedthrough from fusional range only: fast onset for small disparity steps;
    # saturates at ±K_phasic_verg × disp_max_verg_fus for large depth changes.
    u_verg  = x_verg + brain_params.K_phasic_verg * e_fus

    return dx_verg, u_verg
