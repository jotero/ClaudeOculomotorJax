"""Target selector — upstream feedthrough for the saccade generator.

No internal states.  Pure function, JAX/jit/grad compatible.
Does NOT follow the SSM step() convention — it is a feedthrough, not an SSM.

Coordinate systems
──────────────────
Two distinct coordinate frames matter here:

    Retinal / gaze-centered  (e_pos_delayed):
        Angle of the target from the fovea.  This is what the retina reports.
        Independent of where the eye is pointing in the head.
        Large |e_pos_delayed| means the target is far from where the eye
        is currently looking — not necessarily that the eye is eccentric.

    Head-centered  (x_p):
        Where the eye points relative to the head (plant state, deg).
        The orbital mechanical range lives here: the eye cannot rotate
        beyond ±orbital_limit regardless of what the retina reports.

Three cases handled
───────────────────
1. Target visible, within oculomotor range  (most common):
        pos_visible ≠ 0, x_p + pos_visible within ±orbital_limit.
        Anti-windup clip is inactive.  Normal saccade to target.

2. Target visible, beyond oculomotor range:
        pos_visible ≠ 0, x_p + pos_visible would exceed ±orbital_limit.
        Anti-windup clip activates → saccade parks the eye at the orbital
        boundary.  No centering saccade.  Eye stays at limit until the
        target re-enters the reachable range.

3. Target outside visual field entirely  (|e_pos_delayed| > vf_limit):
        gate_vf ≈ 0  →  pos_visible ≈ 0.
        Centering saccade fires: e_reset = −alpha_reset · x_p.
        Pulls the eye back toward primary position.

Implicit straight-ahead target
───────────────────────────────
When no explicit fixation target is provided, the simulator defaults to
p_target = [0, 0, 1] (straight ahead).  In that case
    e_pos_delayed ≈ −x_p  (head stationary)
so pos_visible ≈ −x_p (always within the visual field for normal eccentricities).
This drives the eye back toward center WITHOUT the e_reset centering path.
That is the mechanism behind VOR/OKN fast phases and orbital-reset saccades —
no special logic needed, and e_reset is NOT triggered.

Signal flow
───────────
    e_pos_delayed  (retinal — from visual delay cascade)
         │
         ▼
    [visual field gate]  → pos_visible ≈ 0 when |e_pos_delayed| > vf_limit
         │                 gate_vf ≈ 0 when out-of-field
         ▼
    [centering gate]     → e_cmd = pos_visible − alpha_reset·x_p·(1 − gate_vf)
         │                 Centering fires only when target leaves visual field.
         ▼
    [anti-windup clip]   → hard clip: x_p + e_cmd stays within ±orbital_limit
         │
         ▼
    e_cmd → saccade generator

Parameters
──────────
    orbital_limit  (deg)  mechanical half-range of orbital limit  default 50.0
                          Used only for the anti-windup clip.
    alpha_reset    (-)    centering gain; e_reset = −alpha_reset · x_p
                          Applies only when target is outside the visual field.
                          0 = suppress centering; 1 = full centripetal saccade.  default 1.0
    visual_field_limit and k_visual_field are in SensoryParams; gate_vf is
    computed in sensory_model.read_outputs() and passed here.
"""

import jax.numpy as jnp


def select(pos_visible, x_p, gate_vf, plant_params, brain_params):
    """Compute motor error command for the saccade generator.

    Args:
        pos_visible  : (3,)   visually-gated position error (deg, gaze-centered).
                              ≈ pos_delayed when target is in-field; → 0 when out-of-field.
        x_p          : (3,)   plant state (deg, head-centered).
        gate_vf      : scalar visual-field gate value in [0, 1].
                              ≈ 1 when target is within visual field;
                              ≈ 0 when target is outside visual field.
                              Computed in sensory_model.read_outputs().
        plant_params : PlantParams   orbital mechanics (orbital_limit)
        brain_params : BrainParams   reset policy (alpha_reset)

    Returns:
        e_cmd : (3,)  motor error command (deg), clipped so x_p + e_cmd ∈ ±orbital_limit.
    """
    orbital_limit = plant_params.orbital_limit
    alpha_reset   = brain_params.alpha_reset

    # ── Centering (out-of-field only) ────────────────────────────────────────
    # e_reset fires only when the target has left the visual field (gate_vf → 0).
    # When the target is in the visual field (gate_vf ≈ 1), centering is fully
    # suppressed and the eye parks at the orbital limit via the clip below.
    # When the default straight-ahead target is in use (VOR/OKN fast phases),
    # gate_vf ≈ 1 and pos_visible ≈ −x_p already provides centripetal drive —
    # no e_reset needed.
    e_reset = -alpha_reset * x_p * (1.0 - gate_vf)   # (3,)

    e_cmd = pos_visible + e_reset

    # ── Anti-windup clip (head-centered) ─────────────────────────────────────
    # Hard guarantee: landing position x_p + e_cmd stays within ±orbital_limit.
    # This is the sole mechanism that parks the eye at the orbital boundary
    # when the target is visible but beyond the motor range (case 2 above).
    e_cmd_clipped = jnp.clip(e_cmd, -orbital_limit - x_p, orbital_limit - x_p)

    return e_cmd_clipped
