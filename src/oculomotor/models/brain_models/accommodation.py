"""Accommodation SSM — Schor dual-interaction model (Schor 1979; Schor & Ciuffreda 1983).

Blur-driven lens adjustment with fast phasic + slow tonic components,
cross-coupled to vergence via two ratios:

    AC/A  (accommodative convergence / accommodation):
        When the eye accommodates to bring a near target into focus, vergence
        is reflexively driven inward.  This cross-link provides convergence even
        when binocular disparity is absent (e.g. one eye suppressed in IXT),
        making it the primary recovery mechanism from the deviated state.

        ac_a_drive (deg) = AC_A (pd/D) × 0.573 (deg/pd) × x_acc (D)

    CA/C  (convergence accommodation / accommodation):
        When vergence is driven by disparity, the lens is co-driven to
        maintain focus at the new depth.  Reduces blur during vergence eye
        movements; stabilises the coupled loop.

        A_cac (D) = CA_C (D/pd) × x_verg_yaw (deg) / 0.573 (deg/pd)

Dual-component structure (Schor 1979):
    Fast phasic component  (τ_fast ~0.3 s):  responds quickly to blur; fatigues.
    Slow tonic  component  (τ_slow ~30 s):   integrates residual blur; builds up
                                              slowly; models vergence / accommodation
                                              adaptation (Maddox tonic vergence analogue).

Dynamics:
    e_blur  = acc_demand − (x_fast + x_slow) − CA_C · (x_verg_yaw / 0.573)
    dx_fast = −x_fast / τ_fast  +  K_fast · e_blur
    dx_slow = −x_slow / τ_slow  +  K_slow · e_blur
    x_acc   = x_fast + x_slow                        (total accommodation, D)

    ac_a_drive (deg) = AC_A · 0.573 · x_acc          (fed into vergence.step)

State:
    x_acc = [x_fast (D), x_slow (D)]   (2 scalar states)

Input:
    acc_demand  (D)  = 1 / z_depth (m)  — target-driven blur demand
    x_verg_yaw  (deg)                   — current vergence yaw state (CA/C feedback)

Output:
    x_acc_total (D)  — total accommodation → AC/A drive for vergence

Unit note:
    1 prism diopter (pd) = arctan(0.01) rad ≈ 0.573 deg.
    AC/A ratio is conventionally in pd/D; multiply by 0.573 to get deg/D.
    CA/C ratio is in D/pd; divide vergence by 0.573 to convert deg → pd.

Parameters (BrainParams fields):
    tau_acc_fast  (s)     phasic lens TC;           default 0.3 s
    tau_acc_slow  (s)     tonic adaptation TC;       default 30 s
    K_acc_fast    (1/s)   fast blur gain
    K_acc_slow    (1/s)   slow integration gain
    AC_A          (pd/D)  AC/A ratio;                default 5.0
    CA_C          (D/pd)  CA/C ratio;                default 0.4

References:
    Schor CM (1979) Vision Res 19:1359–1367            — dual slow/fast model
    Schor CM, Ciuffreda KJ (1983) Vergence Eye Movements — textbook
    Hung GK, Semmlow JL (1980) IEEE Trans Biomed Eng 27:722–728
    Semmlow JL, Hung GK, Ciuffreda KJ (1986) Invest Ophthalmol Vis Sci 27:558
"""

import jax.numpy as jnp

N_STATES  = 2   # [x_fast (D), x_slow (D)]
N_INPUTS  = 2   # acc_demand (D), x_verg_yaw (deg)
N_OUTPUTS = 1   # x_acc_total (D)

# 1 prism diopter = arctan(0.01) rad ≈ 0.5729 deg
_DEG_PER_PD = 0.5729


def step(x_acc, acc_demand, x_verg_yaw, brain_params):
    """Single ODE step for dual-component accommodation.

    Args:
        x_acc:        (2,)   [x_fast, x_slow] in diopters
        acc_demand:   scalar  1 / z_depth (D); target-driven blur demand
        x_verg_yaw:   scalar  vergence yaw state (deg) → CA/C feedback
        brain_params: BrainParams  (reads tau_acc_fast, tau_acc_slow,
                                          K_acc_fast, K_acc_slow, CA_C)

    Returns:
        dx_acc:      (2,)   state derivative (D/s)
        x_acc_total: scalar  total accommodation (D) — used for AC/A drive
    """
    x_fast, x_slow = x_acc[0], x_acc[1]
    x_total = x_fast + x_slow

    # CA/C: vergence (deg → pd) drives accommodation co-operatively
    A_cac  = brain_params.CA_C * (x_verg_yaw / _DEG_PER_PD)   # D

    e_blur = acc_demand - x_total - A_cac

    dx_fast = -x_fast / brain_params.tau_acc_fast + brain_params.K_acc_fast * e_blur
    dx_slow = -x_slow / brain_params.tau_acc_slow + brain_params.K_acc_slow * e_blur

    return jnp.array([dx_fast, dx_slow]), x_total


def ac_a_drive(x_acc_total, brain_params):
    """Convert total accommodation (D) to vergence drive (deg) via AC/A.

    Args:
        x_acc_total:  scalar  total accommodation (D)
        brain_params: BrainParams  (reads AC_A)

    Returns:
        drive: scalar  vergence drive (deg); positive = converging
    """
    return brain_params.AC_A * _DEG_PER_PD * x_acc_total
