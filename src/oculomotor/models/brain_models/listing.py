"""Listing's law — torsional constraint for eye positions.

In rotation-vector coordinates, Listing's law states that all achievable eye
positions lie on a sphere such that the rotation axis from the PRIMARY POSITION
to any eye position lies in Listing's plane (a frontal plane).

Primary position (H₀, V₀): the gaze direction that defines the "zero" of
Listing's plane.  Normally straight ahead (0°, 0°); can be shifted via
``BrainParams.listing_primary``.

Half-angle rule (in rotation-vector space):

    T_required = OCR  −  (H − H₀) · (V − V₀) · π/360

where all quantities are in degrees and π/360 ≈ 0.00873 converts the deg²
product back to degrees.  Accuracy: < 5 % error for |H−H₀|, |V−V₀| < 20°.

Single entry point used by brain_model:

    cyc_torsion_vel, cyclo_verg_rate = listing_corrections(
        eye_pos, vel_hv, vergence_angle, primary_pos, l2_frac)

No state — purely algebraic.  All functions are JAX-compatible.

References:
    Listing J B (1854) Beitrag zur physiologischen Optik, Göttingen.
    Tweed, Haslwanter & Fetter (1998) Invest Ophthalmol Vis Sci 39:1500.
    van Rijn & van den Berg (1993) Exp Brain Res 95:154.
"""

import jax.numpy as jnp

# Half-angle conversion factor  (π/360 ≈ 0.00873  deg² → deg)
HALF_ANGLE = jnp.pi / 360.0


def listing_corrections(eye_pos, vel_hv, vergence_angle, primary_pos, l2_frac):
    """Combined Listing's-law corrections — three outputs used by brain_model.

      cyc_torsion_vel    → add to the SUMMED velocity command's torsion axis
                           (u_total[2]) BEFORE NI integration. Ensures the
                           cyclopean eye position satisfies the half-angle rule
                           DURING the gaze shift.

      cyc_torsion_target → add to NI's u_tonic torsion axis. Acts as a Listing-
                           consistent set-point so the NI's leak naturally pulls
                           torsion toward Listing's plane between/after movements,
                           rather than relying on the velocity correction alone
                           to fight the leak.

      cyclo_verg_rate    → add to the vergence cyclo-vergence axis input
                           (verg_rate_tvor[2] in the vergence step). Drives the
                           per-eye torsion difference (T_L − T_R) that arises
                           when each eye's Listing plane tilts ±verg/2 around
                           head-vertical and the eye moves vertically.

    Math:
      Cyclopean half-angle (set-point and rate):
        T_LL(H, V)   =  −(H − H₀)·(V − V₀)·π/360                (set-point)
        Ṫ_LL         =  −π/360 · [(H − H₀)·V̇ + (V − V₀)·Ḣ]     (rate)

      L2 (each eye's Listing plane tilts ±verg/2 about head-vertical):
        Cyclo-vergence rate (T_L − T_R) = −V̇ · sin(verg) ≈ −V̇ · verg [rad]
        Convert verg from deg → rad:  rate = −l2_frac · V̇ · radians(verg)

    Args:
        eye_pos        : (3,) cyclopean eye position [H, V, T] (deg)
        vel_hv         : (2,) summed H/V velocity command [Ḣ, V̇] (deg/s)
        vergence_angle : scalar absolute vergence (deg)
        primary_pos    : (2,) [H₀, V₀] primary position (deg)
        l2_frac        : scalar [0..1] L2 fraction (0 = disabled, 0.5 = physiological)

    Returns:
        cyc_torsion_vel    : float  cyclopean torsional velocity correction (deg/s)
        cyc_torsion_target : float  Listing-prescribed torsion set-point (deg)
        cyclo_verg_rate    : float  cyclo-vergence rate from L2 (deg/s)
    """
    dH = eye_pos[0] - primary_pos[0]
    dV = eye_pos[1] - primary_pos[1]
    H_dot, V_dot = vel_hv[0], vel_hv[1]

    cyc_torsion_vel    = -HALF_ANGLE * (dH * V_dot + dV * H_dot)
    cyc_torsion_target = -HALF_ANGLE * dH * dV
    cyclo_verg_rate    = -l2_frac * V_dot * jnp.radians(vergence_angle)
    return cyc_torsion_vel, cyc_torsion_target, cyclo_verg_rate
