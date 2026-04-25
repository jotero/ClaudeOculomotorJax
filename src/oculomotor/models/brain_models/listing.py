"""Listing's law — torsional constraint for eye positions.

In rotation-vector coordinates, Listing's law states that all achievable eye
positions lie on a sphere such that the rotation axis from the PRIMARY POSITION
to any eye position lies in Listing's plane (a frontal plane).

Primary position (H₀, V₀): the gaze direction that defines the "zero" of
Listing's plane.  Normally straight ahead (0°, 0°); can be shifted in
strabismus, after adaptation, or via listing_primary in BrainParams.

Half-angle rule: in rotation-vector space, the torsional component required by
Listing's law at gaze (H, V) with head-tilt OCR is:

    T_required = OCR  −  (H − H₀) · (V − V₀) · π/360

where all quantities are in degrees and π/360 ≈ 0.00873 converts the deg²
product back to degrees.  Accuracy: < 5 % error for |H−H₀|, |V−V₀| < 20°.

Single entry point for brain_model:

    pos_for_sg, x_ni_for_sg, vel_torsion = listing.corrections(...)

No state — purely algebraic.  All functions are JAX-compatible.

References:
    Listing J B (1854) Beitrag zur physiologischen Optik, Göttingen.
    Tweed, Haslwanter & Fetter (1998) Invest Ophthalmol Vis Sci 39:1500.
    van Rijn & van den Berg (1993) Exp Brain Res 95:154.
"""

import jax.numpy as jnp

N_STATES  = 0
N_INPUTS  = 8   # [eye_pos(3), eye_vel_hv(2), target_pos(3)] + scalars (conceptual)
N_OUTPUTS = 3   # pos_for_sg(3), x_ni_for_sg(3), vel_torsion(1) — returned as tuple

# Half-angle conversion factor  (π/360 ≈ 0.00873  deg² → deg)
HALF_ANGLE = jnp.pi / 360.0


def required_torsion(eye_h, eye_v, ocr, primary_pos):
    """Torsion required by Listing's law + OCR at gaze (eye_h, eye_v).

    T_required = OCR  −  (H − H₀) · (V − V₀) · π/360

    Args:
        eye_h       : horizontal gaze (deg, rotation-vector component)
        eye_v       : vertical gaze (deg, rotation-vector component)
        ocr         : OCR torsional shift from head tilt (deg; +CW from front)
        primary_pos : (2,) [yaw₀, pitch₀] primary position (deg)

    Returns:
        T_required (deg)
    """
    dH = eye_h - primary_pos[0]
    dV = eye_v - primary_pos[1]
    return ocr - dH * dV * HALF_ANGLE


def listing_error(eye_pos, ocr, primary_pos):
    """Torsional deviation from Listing's law at the current eye position.

    Args:
        eye_pos     : (3,) [yaw, pitch, roll] deg — current eye position
        ocr         : scalar OCR (deg)
        primary_pos : (2,) [yaw₀, pitch₀] primary position (deg)

    Returns:
        error (deg): positive = excess CCW torsion (needs CW correction)
    """
    T_req = required_torsion(eye_pos[0], eye_pos[1], ocr, primary_pos)
    return eye_pos[2] - T_req


def corrections(eye_pos, eye_vel_hv, target_pos, ocr, primary_pos):
    """All Listing's law corrections — single call for brain_model.

    Computes three outputs in one pass:

    1. pos_for_sg  — target_pos with torsional landing correction in [2].
       The torsion to add so the saccade lands on Listing's plane at (H+ΔH, V+ΔV).

    2. x_ni_for_sg — eye_pos with listing_error in [2].
       Shifts the SG's eye-position reference so the out-of-field centering
       saccade targets OCR rather than zero.

    3. vel_torsion — torsional velocity (deg/s) to maintain Listing's plane
       during smooth pursuit.  Equals dT_required/dt for the current eye
       velocity:

           vel_torsion = −π/360 · [ (H−H₀)·V̇  +  (V−V₀)·Ḣ ]

       Pass this as an additive torsional drive to the NI input.

    Args:
        eye_pos     : (3,) current eye position [H, V, T] deg (rotation-vector)
        eye_vel_hv  : (2,) smooth H/V eye velocity [Ḣ, V̇] deg/s
                      (−w_est + u_pursuit, excluding saccade burst)
        target_pos  : (3,) SG position error [e_H, e_V, ~0] deg
        ocr         : scalar OCR from gravity estimator (deg)
        primary_pos : (2,) [H₀, V₀] primary position (deg)

    Returns:
        pos_for_sg   : (3,)  corrected SG position error
        x_ni_for_sg  : (3,)  corrected eye-position proxy for SG
        vel_torsion  : float  torsional velocity demand (deg/s)
    """
    # Listing error at current position (for SG centering reference)
    err_now = listing_error(eye_pos, ocr, primary_pos)

    # Torsional target at saccade landing position
    T_req_land = required_torsion(
        eye_pos[0] + target_pos[0],
        eye_pos[1] + target_pos[1],
        ocr, primary_pos)

    # Torsional velocity demand to keep T on Listing's plane during smooth pursuit
    # dT_required/dt = −HALF_ANGLE · [(H−H₀)·V̇  +  (V−V₀)·Ḣ]
    dH = eye_pos[0] - primary_pos[0]
    dV = eye_pos[1] - primary_pos[1]
    vel_torsion = -HALF_ANGLE * (dH * eye_vel_hv[1] + dV * eye_vel_hv[0])

    return (
        target_pos.at[2].set(T_req_land - eye_pos[2]),   # pos_for_sg
        eye_pos.at[2].set(err_now),                       # x_ni_for_sg
        vel_torsion,                                      # pursuit torsional demand
    )
