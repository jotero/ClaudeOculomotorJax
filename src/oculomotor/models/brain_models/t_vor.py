"""Translational VOR (T-VOR) — gaze- and distance-dependent compensation for head translation.

Geometry:
    ω_eye      = − ĝ × v_head / D       (head-frame angular velocity, rad/s)
    verg_rate  = IPD · (ĝ · v) / D²     (vergence change rate, rad/s)

Architecture (VELOCITY pathway):
    No position integrator inside T-VOR.  The state is a low-pass filter on the
    head linear-velocity estimate (m/s) — it tracks v_lin and decays to 0 in the
    absence of head translation.  The output is an angular EYE VELOCITY command
    (deg/s) computed algebraically from the cross product, and is integrated
    downstream by NI (added to NI's velocity input alongside −w_est + u_burst).

    Vergence-derived distance D enters only on the OUTPUT; it does not appear
    in the integrator dynamics, so eye convergence cannot shrink D and amplify
    the integrated command.

    State:    x_tvor_v (3,)    — low-pass filter on v_lin (m/s, head frame)
    Dynamics: dx_tvor_v/dt = (v_lin − x_tvor_v) / τ_tvor_pos
                               (lowpass; default τ ≈ 30 s — leaks toward 0 in
                                absence of head translation)
    Output:   omega_tvor (3,)  = (−ĝ × x_tvor_v) / D · g_tvor [deg/s, ypr]
                                fed into NI as a velocity input (NI integrates it).
              verg_rate_tvor   — H/V/cyclo-vergence rate from translation

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

import jax
import jax.numpy as jnp

from oculomotor.models.plant_models.readout import gaze_unit_vector
from oculomotor.models.sensory_models.retina import xyz_to_ypr
from oculomotor.models.brain_models import listing


_HALF_ANGLE = jnp.pi / 360.0   # deg² → deg conversion for half-angle rule


N_STATES  = 3   # x_tvor_v (3,) — low-pass-filtered head linear velocity estimate (m/s)
N_INPUTS  = 6
N_OUTPUTS = 3


def step(x_tvor, v_lin, x_verg_yaw, eye_pos, ipd, brain_params):
    """T-VOR step: low-pass v_lin and convert to an angular eye VELOCITY command.

    No position integrator inside T-VOR. The state is a low-pass filter on v_lin
    (the head-frame linear velocity estimate). The output omega_tvor is an angular
    eye-velocity command (deg/s) that NI integrates downstream.

    Args:
        x_tvor:        (3,)   low-pass-filtered head linear velocity (m/s, head frame)
        v_lin:         (3,)   head linear velocity (m/s, head frame); from HE
        x_verg_yaw:    scalar absolute vergence (deg) → distance D
        eye_pos:       (3,)   gaze direction [yaw, pitch, roll] (deg, head frame)
        ipd:           scalar interpupillary distance (m)
        brain_params:  reads g_tvor, g_tvor_verg, tau_tvor_pos

    Returns:
        dx_tvor:        (3,)  state derivative (m/s²) — lowpass dynamics
        omega_tvor:     (3,)  angular eye-velocity command (deg/s) → NI velocity input
        verg_rate_tvor: (3,)  H/V/cyclo-vergence rates (deg/s) → vergence slow integrator
    """
    # ── Vergence-derived distance and effective 1/distance ──────────────────
    # The math everywhere below needs 1/D, never D itself, so build an inverse-
    # distance gain directly and gate it at both ends:
    #   d → 0  (vergence past NPC, or divergence giving negative D): gain → 0
    #   d → ∞ (no convergence): 1/d → 0 naturally
    # Aggressive variant: gate center sits at 1.5·NPC and is steep, so the
    # gain has already dropped sharply by the time vergence reaches NPC and
    # the peak of (gate/d) sits well above NPC (≈ 2·NPC) instead of right at it.
    distance_raw = ipd / (2.0 * jnp.tan(jnp.radians(x_verg_yaw) * 0.5))
    # Guard against zero / negative distance (small or divergent vergence).
    d_safe       = jnp.maximum(distance_raw, 1e-6)
    npc          = brain_params.distance_npc
    npc_gate     = jax.nn.sigmoid((d_safe - 1.5 * npc) / (0.1 * npc))
    inv_distance = npc_gate / d_safe
    g_hat        = gaze_unit_vector(eye_pos)
    DEG_PER_RAD  = 180.0 / jnp.pi

    # ── No second integrator on v_lin: pass it straight through to the cross
    #    product. The heading_estimator already lowpass-integrates a_lin → v_lin,
    #    so a second lowpass here would just slow the TVOR response without
    #    adding filtering value. State slot kept for layout compat (inert).
    dx_tvor = jnp.zeros_like(x_tvor)

    # ── Algebraic output: convert linear velocity → angular eye velocity ─────
    #    omega_xyz_rad = (−ĝ × v_lin) · (1/D)     (instantaneous parallax velocity)
    omega_xyz_rad = -jnp.cross(g_hat, v_lin) * inv_distance
    omega_tvor    = brain_params.g_tvor * DEG_PER_RAD * xyz_to_ypr(omega_xyz_rad)

    # ── Dot product → horizontal vergence rate (rad/s) → deg/s ──────────────
    # Surge along the gaze direction changes target distance → H-vergence change.
    # V- and T-vergence rates are 0 from translation in our scene model (no
    # asymmetric parallax / depth structure); kept as a 3-vector for future L2
    # extensions and for downstream consumers that want a 3D vergence drive.
    dot_gv          = jnp.dot(g_hat, v_lin)
    verg_rate_rad_H = ipd * dot_gv * inv_distance * inv_distance
    verg_rate_tvor  = brain_params.g_tvor_verg * DEG_PER_RAD * jnp.array([
        verg_rate_rad_H,   # horizontal vergence rate (deg/s)
        0.0,               # vertical   vergence rate (deg/s) — reserved
        0.0,               # cyclo-     vergence rate (deg/s) — reserved
    ])

    # ── Listing's law correction removed ──────────────────────────────────────
    # The previous architecture computed an angular position offset (pos_tvor)
    # and applied the half-angle rule to it. With the new velocity-output
    # architecture there is no position to correct here — Listing's must be
    # enforced wherever the integrated eye position is read (e.g. on x_ni).

    # ── verg_rate L2 cross-coupling — vertical motion at vergence drives cyclo-verg.
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

    return dx_tvor, omega_tvor, verg_rate_tvor
