"""Translational VOR (T-VOR) — gaze- and distance-dependent compensation for head translation.

Geometry:
    ω_eye      = − ĝ × v_head / D       (head-frame angular velocity, rad/s)
    verg_rate  = IPD · (ĝ · v) / D²     (vergence change rate, rad/s)

Architecture (DIRECT pathway — bypasses NI):
    The instantaneous eye-velocity command from the cross product is **integrated
    in T-VOR's own state** (x_tvor_pos, deg) and the resulting position offset is
    added DIRECTLY to the motor command, alongside NI's output.  Bypassing NI
    avoids the slow leak (τ_i = 25 s) and the disruption from saccade-burst inputs
    that would otherwise eclipse the smooth T-VOR slow phase.

    State:    x_tvor_pos (3,)  — accumulated T-VOR eye position offset (deg)
    Dynamics: dx_tvor_pos/dt = ω_tvor − x_tvor_pos / τ_tvor_pos
                               (slow leak; default τ ≈ 30 s lets the offset reset
                                between movements without bleeding off during the motion)
    Output:   pos_tvor   (3,)  — added directly to motor_cmd in brain_model
              ω_tvor     (3,)  — also returned as a velocity hint (currently unused
                                  downstream so the slow-phase signal flows through
                                  the bypass path only)

References
    Paige & Tomko (1991) J Neurophysiol 65:1170 — vergence-modulated T-VOR gain
    Angelaki et al. (1999, 2000) — otolith afferent dynamics and T-VOR
    Crane & Demer (1997, 1998) Vision Res — T-VOR distance dependence

Note on EC and parallax (future improvement):
    ω_tvor is added to ec_vel in brain_model.py so OKR/pursuit don't fight T-VOR's
    angular eye motion.  This is a workaround for the simplified scene model in
    retina.py: scene_angular_vel from head translation is currently 0 (no parallax),
    so T-VOR's eye motion creates angular slip out of nothing in the model.

    A more physiological fix is to make scene_angular_vel reflect head-translation-
    induced motion parallax — e.g. add a term −R_gaze.T @ (v_head × g_hat) / D_scene
    using a reference scene depth, so head translation naturally creates angular
    flow on the retina.  Then T-VOR's eye motion would naturally cancel the
    parallax flow at the target's depth, just as RVOR cancels canal-driven slip,
    without needing T-VOR EC.  Until that's done, the EC fix keeps the version
    loops (OKR + pursuit) from fighting T-VOR.
"""

import jax.numpy as jnp

from oculomotor.models.plant_models.readout import gaze_unit_vector
from oculomotor.models.sensory_models.retina import xyz_to_ypr
from oculomotor.models.brain_models import listing


_HALF_ANGLE = jnp.pi / 360.0   # deg² → deg conversion for half-angle rule


N_STATES  = 3   # x_tvor_pos (3,) — accumulated eye-position offset from T-VOR (deg)
N_INPUTS  = 6
N_OUTPUTS = 3


def step(x_tvor, v_lin, x_verg_yaw, eye_pos, ipd, brain_params):
    """T-VOR step: integrate ω_tvor into a position offset, return that + vergence rate.

    Args:
        x_tvor:        (3,)   accumulated T-VOR eye-position offset (deg)
        v_lin:         (3,)   head linear velocity (m/s, head frame); from HE
        x_verg_yaw:    scalar absolute vergence (deg) → distance D
        eye_pos:       (3,)   gaze direction [yaw, pitch, roll] (deg, head frame)
        ipd:           scalar interpupillary distance (m)
        brain_params:  reads g_tvor, g_tvor_verg, tau_tvor_pos

    Returns:
        dx_tvor:        (3,)  state derivative (deg/s)
        pos_tvor:       (3,)  position offset (deg) — added DIRECTLY to motor_cmd
        verg_rate_tvor: scalar vergence rate (deg/s) → vergence slow integrator
    """
    # ── Vergence-derived distance and gaze unit vector ──────────────────────
    distance = ipd / (2.0 * jnp.tan(jnp.radians(x_verg_yaw) * 0.5))
    g_hat    = gaze_unit_vector(eye_pos)

    # ── Cross product → instantaneous version eye velocity (rad/s, xyz) ─────
    omega_xyz   = -jnp.cross(g_hat, v_lin) / distance
    DEG_PER_RAD = 180.0 / jnp.pi
    omega_tvor  = brain_params.g_tvor * DEG_PER_RAD * xyz_to_ypr(omega_xyz)

    # ── Direct-pathway integrator: x_tvor_pos accumulates ω_tvor with slow leak.
    #    The slow leak (τ_tvor_pos, default ≈ 30 s) lets the offset bleed off
    #    between bouts of head motion — but is long enough that it doesn't
    #    decay during the motion itself.
    dx_tvor_pos = omega_tvor - x_tvor / brain_params.tau_tvor_pos
    pos_tvor    = x_tvor

    # ── Dot product → horizontal vergence rate (rad/s) → deg/s ──────────────
    # Surge along the gaze direction changes target distance → H-vergence change.
    # V- and T-vergence rates are 0 from translation in our scene model (no
    # asymmetric parallax / depth structure); kept as a 3-vector for future L2
    # extensions and for downstream consumers that want a 3D vergence drive.
    dot_gv          = jnp.dot(g_hat, v_lin)
    verg_rate_rad_H = ipd * dot_gv / (distance * distance)
    verg_rate_tvor  = brain_params.g_tvor_verg * DEG_PER_RAD * jnp.array([
        verg_rate_rad_H,   # horizontal vergence rate (deg/s)
        0.0,               # vertical   vergence rate (deg/s) — reserved
        0.0,               # cyclo-     vergence rate (deg/s) — reserved
    ])

    # ── Listing's law correction (extended L2 for vergence) ─────────────────
    # 1) pos_tvor torsion: enforce the half-angle rule on the FULL cyclopean
    #    eye position (eye_pos + pos_tvor), not just on pos_tvor's own H,V.
    #    Listing's law is a constraint on the actual eye orientation, not an
    #    additive correction — so SET (don't ADD) the T-VOR torsion to make
    #    the total satisfy the half-angle rule:
    #       T_required_total = -(H_total − H₀)·(V_total − V₀)·π/360
    #    pos_tvor provides the *difference* from the existing eye_pos torsion.
    #    L2 (per-eye plane tilts by ±verg/2) collapses to standard Listing's
    #    at the cyclopean level because the ±verg/2 tilts cancel in (L+R)/2.
    primary   = brain_params.listing_primary    # (2,) [H₀, V₀]
    H_total   = (eye_pos[0] + pos_tvor[0]) - primary[0]
    V_total   = (eye_pos[1] + pos_tvor[1]) - primary[1]
    T_required_total = -H_total * V_total * _HALF_ANGLE
    pos_tvor  = pos_tvor.at[2].set(T_required_total - eye_pos[2])

    # 2) verg_rate L2 cross-coupling — vertical motion at vergence drives cyclo-verg.
    #    Each eye's Listing's plane is tilted around the head's VERTICAL axis by
    #    ±verg/2.  The "horizontal rotation axis" for that eye (used for vertical
    #    motion) is therefore tilted forward/back by verg/2:
    #       L_horiz_axis ≈ (cos(verg/2), 0, −sin(verg/2))
    #       R_horiz_axis ≈ (cos(verg/2), 0, +sin(verg/2))
    #    A vertical rotation ΔV on this tilted axis has a torsional component:
    #       L torsion rate: −ΔV · sin(verg/2) ≈ −ΔV · verg/2
    #       R torsion rate: +ΔV · sin(verg/2) ≈ +ΔV · verg/2
    #    Cyclo-vergence rate (T_L − T_R) = −ΔV · verg  (in radians).
    #    ΔV here is the cyclopean V-rate from T-VOR: omega_tvor[1] (deg/s).
    #    Gated by g_tvor_l2_cyclo (default 0): the geometry is correct but enabling
    #    creates a positive feedback loop with FCP's linear Hering's during head tilts
    #    (cyclo-vergence drift in OCR cascade roll column).  Re-enable once FCP uses
    #    rotation-matrix Hering's instead of linear motor split.
    verg_rad        = jnp.radians(x_verg_yaw)   # absolute vergence in radians
    Tverg_from_L2   = brain_params.g_tvor_l2_cyclo * (-omega_tvor[1] * verg_rad)
    verg_rate_tvor  = verg_rate_tvor.at[2].set(Tverg_from_L2)

    return dx_tvor_pos, pos_tvor, verg_rate_tvor
