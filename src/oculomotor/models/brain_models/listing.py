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


def listing_corrections(eye_pos, vel_hv, vergence_angle, primary_pos, l2_frac, gain):
    """Combined Listing's-law corrections — three 3-D vectors ready to add.

      u_ni_correction → add directly to u_ni_in.  Cyclopean half-angle
                        velocity correction on the torsion axis (axis 2).
                        Ensures eye position satisfies the half-angle rule
                        DURING the gaze shift.

      u_tonic_listing → add directly to u_tonic.  Listing-prescribed torsion
                        set-point that the NI's leak pulls toward, so torsion
                        relaxes onto Listing's plane between/after movements.

      verg_rate_listing → add directly to verg_rate_tvor.  Cyclo-vergence
                        rate from the L2 component (each eye's Listing plane
                        tilted ±verg/2 about head-vertical → per-eye torsion
                        difference T_L − T_R when the eye moves vertically).

    All three returns are pre-multiplied by `gain` and embedded into 3-D
    vectors with the non-torsion axes zero, so the caller just does
    `u_ni_in += listing.listing_corrections(...)[0]` without any indexing.

    Math:
      T_LL(H, V)  =  −(H − H₀)·(V − V₀)·π/360                 (set-point)
      Ṫ_LL        =  −π/360 · [(H − H₀)·V̇ + (V − V₀)·Ḣ]     (rate)
      L2 rate     =  −l2_frac · V̇ · radians(verg)              (cyclo-vergence)

    Args:
        eye_pos        : (3,) cyclopean eye position [H, V, T] (deg)
        vel_hv         : (2,) summed H/V velocity command [Ḣ, V̇] (deg/s)
        vergence_angle : scalar absolute vergence (deg)
        primary_pos    : (2,) [H₀, V₀] primary position (deg)
        l2_frac        : scalar [0..1] L2 fraction (0 = disabled, 0.5 = physiological)
        gain           : scalar  multiplier applied to all three outputs
                                 (`brain_params.listing_gain`)

    Returns:
        u_ni_correction   : (3,) [0, 0, gain · cyc_torsion_vel]      → u_ni_in
        u_tonic_listing   : (3,) [0, 0, gain · cyc_torsion_target]   → u_tonic
        verg_rate_listing : (3,) [0, 0, gain · cyclo_verg_rate]      → verg_rate_tvor
    """
    dH = eye_pos[0] - primary_pos[0]
    dV = eye_pos[1] - primary_pos[1]
    H_dot, V_dot = vel_hv[0], vel_hv[1]

    cyc_torsion_vel    = -HALF_ANGLE * (dH * V_dot + dV * H_dot)
    cyc_torsion_target = -HALF_ANGLE * dH * dV
    cyclo_verg_rate    = -l2_frac * V_dot * jnp.radians(vergence_angle)

    z = jnp.float32(0.0)
    g = gain
    return (
        jnp.array([z, z, g * cyc_torsion_vel]),     # → u_ni_in
        jnp.array([z, z, g * cyc_torsion_target]),  # → u_tonic
        jnp.array([z, z, g * cyclo_verg_rate]),     # → verg_rate_tvor
    )
