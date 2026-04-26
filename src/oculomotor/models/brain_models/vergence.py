"""Vergence SSM — single leaky integrator with dual-range nonlinear drive.

Drives disconjugate eye movements to align both foveas on a binocular target.

Architecture:
    One integrator state x_verg holds the tonic vergence position memory.
    The input is a windowed (zero-outside-range) combination of two gated
    versions of e_disp, implementing dual-range gain:

            e_fus  = disp_sat(e_disp, v_sat=disp_max_verg_fus)   ← cosine rolloff 1→2 deg
            e_prox = disp_sat(e_disp, v_sat=disp_max_verg_prox) ← cosine rolloff 20→40 deg
            e_pred = K_verg · e_fus + K_verg_prox · e_prox     ← deg/s drive

    Gain profile (deg/s per deg disparity):
        |e_disp| < 1 deg:    K_verg + K_verg_prox  [high — fusional + proximal active]
        1–2 deg rolloff:     K_verg decays smoothly (cosine), K_verg_prox still full
        2 < |e_disp| < 20:   K_verg_prox           [lower — fusional gone, proximal full]
        20–40 deg rolloff:   K_verg_prox decays smoothly to zero
        |e_disp| > 40:       0                      [outside vergence range — no drive]

Dynamics:
    dx_verg  = −(x_verg − phoria) / τ  +  e_pred
    u_verg   = x_verg + K_phasic · e_fus          ← phasic feedthrough from fusional range only

Phasic term (K_phasic · e_fus):
    Provides fast vergence onset for small disparities (within fusional range).
    Saturates at ±K_phasic · e_sat_fus for large disparities — no phasic overshoot
    on large depth steps where the integrator drives most of the response.

Steady-state gain (approximate, within fusional range):
    CL gain ≈ G / (1 + G)   where G ≈ (K_verg + K_verg_prox) · τ + K_phasic
    With K_verg=4, K_verg_prox=3, τ=6, K_phasic=1:  G ≈ 43 → gain ~98 %

State:
    x_verg = [H_verg, V_verg, torsional_verg]  (3,)  deg
    Positive H = converged (eyes rotated inward for near target).

Parameters:
    K_verg         — fusional integration gain (1/s).  Default 4.
                     Combined gain in fusional range = K_verg + K_verg_prox = 7 /s →
                     convergence TC ≈ 143 ms. [Rashbass & Westheimer 1961]
    K_verg_prox    — proximal/full-range integration gain (1/s).  Default 3.
                     Active alone outside fusional range: TC ≈ 330 ms;
                     initial velocity ≈ 30°/s for 10° step. [Collewijn et al. 1988]
    K_phasic_verg  — phasic feedthrough (dim'less); default 1.
                     Applied to fusional clip only → fast onset for small steps.
    tau_verg       — leak TC (s); default 6 s.  Tonic vergence hold; drifts to phoria
                     when fusion is lost. [Semmlow et al. 1986: ~5–7 s]
                     (Schor 1979 reports ~25 s for slow fusional *adaptation*, not
                     the integrator TC itself.)
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
    Semmlow JL et al. (1986) Invest Ophthalmol Vis Sci 27:558–564  — tonic vergence TC ~5–7 s
"""

import jax.numpy as jnp

from oculomotor.models.sensory_models.retina import velocity_saturation as _disp_sat

N_STATES  = 3   # x_verg: [H_verg, V_verg, torsional_verg]  (deg)
N_INPUTS  = 3   # e_disp = pos_delayed_L − pos_delayed_R (deg)
N_OUTPUTS = 3   # u_verg: vergence position command (deg)


def step(x_verg, e_disp, ac_a_drive, brain_params):
    """Single-integrator vergence controller with dual-range nonlinear drive.

    Args:
        x_verg:       (3,)   vergence position memory (deg); positive = converged
        e_disp:       (3,)   binocular disparity = pos_delayed_L − pos_delayed_R (deg)
                             Gated by bino = tv_L × tv_R; zero when one eye is suppressed.
        ac_a_drive:   scalar  accommodative convergence drive (deg) from accommodation.py.
                             Active even when e_disp=0 (suppressed / monocular) — this
                             is the primary re-fusion mechanism in intermittent exotropia.
        brain_params: BrainParams  (reads K_verg, K_verg_prox, K_phasic_verg, tau_verg,
                                         disp_max_verg_fus, disp_max_verg_prox, phoria)

    Returns:
        dx_verg: (3,)  dx_verg/dt  (deg/s)
        u_verg:  (3,)  vergence position command (deg) → split ±½ in brain_model
    """
    # ── AC/A + disparity: total vergence error ────────────────────────────────
    # ac_a_drive acts on the horizontal axis only; broadcast to (3,) for uniform clipping.
    # The dual-range nonlinearity then clips the combined signal appropriately.
    e_total = e_disp + jnp.array([ac_a_drive, 0.0, 0.0])

    # ── Dual-range cosine-rolloff nonlinearity ──────────────────────────────────
    # Same shape as velocity_saturation in retina.py: gain=1 below v_sat, cosine
    # rolloff to 0 at v_zero=2×v_sat, zero beyond.  This avoids the saturation
    # plateau that clip() creates (which drives vergence at constant max force for
    # arbitrarily large disparities).
    #
    # Gain profile (based on |e_total|):
    #   |e_total| < 1 deg:       K_verg + K_verg_prox ≈ 6/s  → TC ~160 ms [Rashbass 1961]
    #   1 < |e_total| < 2 deg:   K_verg × rolloff + K_verg_prox  (fusional rolling off)
    #   2 < |e_total| < 20 deg:  K_verg_prox          ≈ 1/s  → TC ~1 s    [Judge 1985]
    #   20 < |e_total| < 40 deg: K_verg_prox × rolloff (proximal rolling off)
    #   |e_total| > 40 deg:      0                            → no drive
    e_fus  = _disp_sat(e_total, v_sat=brain_params.disp_max_verg_fus)
    e_prox = _disp_sat(e_total, v_sat=brain_params.disp_max_verg_prox)

    e_pred = brain_params.K_verg * e_fus + brain_params.K_verg_prox * e_prox

    # Leak toward phoria (resting vergence when fusion is absent) with TC = tau_verg.
    dx_verg = -(1.0 / brain_params.tau_verg) * (x_verg - brain_params.phoria) + e_pred

    # Phasic feedthrough from fusional range only: fast onset for small disparity steps;
    # saturates at ±K_phasic_verg × disp_max_verg_fus for large depth changes.
    u_verg  = x_verg + brain_params.K_phasic_verg * e_fus

    return dx_verg, u_verg
