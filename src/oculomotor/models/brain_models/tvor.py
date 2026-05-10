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

# NPC gate (smooth shut-off of T-VOR scaling near vergence-derived NPC):
#   gate(d) = sigmoid((d − _NPC_GATE_CENTER_FRAC·NPC) / (_NPC_GATE_SHARPNESS_FRAC·NPC))
# Center sits ABOVE NPC so the gain has already dropped sharply by the time
# vergence reaches NPC and the peak of (gate/d) lands well above NPC.
_NPC_GATE_CENTER_FRAC    = 1.5    # gate half-point at 1.5·NPC
_NPC_GATE_SHARPNESS_FRAC = 0.1    # sigmoid width = 0.1·NPC (sharp transition)
_DISTANCE_EPSILON        = 1e-6   # m — guard against zero/negative distance


N_STATES  = 0   # T-VOR is stateless (was 3 — see history; the lowpass on v_lin was inert)
N_OUTPUTS = 3


def step(v_lin, a_lin, verg_yaw, eye_pos, brain_params):
    """T-VOR step: convert head linear velocity / acceleration to angular eye VELOCITY.

    Stateless. Two parallel pathways feeding the same cross-product formula:

      INTEGRATING path (slow but sustained):
        v_lin already low-pass integrated by heading_estimator (TC = τ_head).
        omega_int = g_tvor · (−ĝ × v_lin) / D

      DIRECT path (fast onset, decays with τ_a_lin):
        a_lin is the gravity-estimator's linear-acc estimate (m/s²).
        omega_dir = K_phasic_tvor · (−ĝ × a_lin) / D
        K_phasic_tvor implicitly carries seconds (a_lin → v-equivalent), so
        the result has the same deg/s units as omega_int.

      omega_tvor = omega_int + omega_dir

    The direct path matches the short-latency component of T-VOR (~30–100 ms)
    that the multi-stage cascade alone can't deliver. Set K_phasic_tvor = 0
    to disable.

    Args:
        v_lin:        (3,)   head linear velocity (m/s, head frame) — from
                              brain-wide acts.sm.v_lin
        a_lin:        (3,)   head linear acceleration (m/s², head frame) —
                              from acts.sm.a_lin
        verg_yaw:     scalar absolute vergence (deg) → distance D
        eye_pos:      (3,)   gaze direction [yaw, pitch, roll] (deg) —
                              decoded.ni.net (NI net eye position)
        brain_params: BrainParams (reads g_tvor, K_phasic_tvor, g_tvor_verg,
                                   g_tvor_l2_cyclo, distance_npc, ipd_brain)

    Returns:
        omega_tvor:     (3,)  angular eye-velocity command (deg/s) → NI velocity input
        verg_rate_tvor: (3,)  H/V/cyclo-vergence rates (deg/s) → vergence slow integrator
    """
    x_verg_yaw = verg_yaw
    ipd        = brain_params.ipd_brain
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
    d_safe       = jnp.maximum(distance_raw, _DISTANCE_EPSILON)
    npc          = brain_params.distance_npc
    npc_gate     = jax.nn.sigmoid(
        (d_safe - _NPC_GATE_CENTER_FRAC * npc) / (_NPC_GATE_SHARPNESS_FRAC * npc)
    )
    inv_distance = npc_gate / d_safe
    g_hat        = gaze_unit_vector(eye_pos)
    DEG_PER_RAD  = 180.0 / jnp.pi

    # ── Integrating path: convert linear velocity → angular eye velocity ────
    #    omega_int_xyz_rad = (−ĝ × v_lin) · (1/D)    (sustained parallax velocity)
    omega_int_xyz_rad = -jnp.cross(g_hat, v_lin) * inv_distance
    omega_int         = brain_params.g_tvor * DEG_PER_RAD * xyz_to_ypr(omega_int_xyz_rad)

    # ── Direct (phasic) path: convert linear acceleration → angular eye velocity ─
    #    Bypasses the heading-estimator integrator. K_phasic_tvor implicitly
    #    carries seconds (a_lin → v-equivalent), so omega_dir is in deg/s.
    omega_dir_xyz_rad = -jnp.cross(g_hat, a_lin) * inv_distance
    omega_dir         = brain_params.K_phasic_tvor * DEG_PER_RAD * xyz_to_ypr(omega_dir_xyz_rad)

    omega_tvor = omega_int + omega_dir

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

    # ── Listing's law (cyclopean + L2) is now applied centrally in brain_model
    # via listing.listing_corrections, NOT here. T-VOR returns the raw
    # version omega and translation-derived vergence rate; brain_model adds
    # the cyclopean torsion velocity to NI's input and the L2 cyclo-vergence
    # rate to verg_rate_tvor[2] before vergence integrates them.
    return omega_tvor, verg_rate_tvor
