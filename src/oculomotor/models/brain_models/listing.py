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


def saccade_corrections(eye_pos, target_pos, ocr, primary_pos):
    """Listing's law corrections for saccade targeting.

    Args:
        eye_pos     : (3,) current eye position [H, V, T] deg
        target_pos  : (3,) SG position error [e_H, e_V, ~0] deg
        ocr         : scalar OCR from gravity estimator (deg)
        primary_pos : (2,) [H₀, V₀] primary position (deg)

    Returns:
        pos_for_sg   : (3,)  target_pos with torsional landing correction in [2]
        x_ni_for_sg  : (3,)  eye_pos with listing_error in [2] (SG centering reference)
    """
    err_now    = listing_error(eye_pos, ocr, primary_pos)
    T_req_land = required_torsion(
        eye_pos[0] + target_pos[0],
        eye_pos[1] + target_pos[1],
        ocr, primary_pos)
    return (
        target_pos.at[2].set(T_req_land - eye_pos[2]),
        eye_pos.at[2].set(err_now),
    )


def pursuit_torsion(eye_pos, eye_vel_hv, primary_pos):
    """Torsional velocity (deg/s) to maintain Listing's plane during smooth pursuit.

    vel_torsion = −π/360 · [ (H−H₀)·V̇  +  (V−V₀)·Ḣ ]

    Args:
        eye_pos     : (3,) current eye position [H, V, T] deg
        eye_vel_hv  : (2,) smooth pursuit H/V velocity [Ḣ, V̇] deg/s
                      (pursuit only — VOR follows its own 3D canal structure)
        primary_pos : (2,) [H₀, V₀] primary position (deg)

    Returns:
        vel_torsion : float  torsional velocity demand (deg/s)
    """
    dH = eye_pos[0] - primary_pos[0]
    dV = eye_pos[1] - primary_pos[1]
    return -HALF_ANGLE * (dH * eye_vel_hv[1] + dV * eye_vel_hv[0])
