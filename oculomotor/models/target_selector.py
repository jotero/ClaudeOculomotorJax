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

Two distinct limits
───────────────────
1. Visual field gate  (retinal — gaze-centered):
        |e_pos_delayed| > visual_field_limit  (~90 deg)
        The target is outside the visual field entirely — behind the head,
        or so far eccentric that no retinal signal exists.  Saccade should
        be suppressed completely.
        Implemented as a smooth gate on |e_pos_delayed|, per axis.

2. Orbital motor limit  (head-centered — x_p):
        |x_p| > orbital_limit  (~50 deg)
        The eye is at its mechanical range.  The motor command must be
        suppressed or redirected regardless of what the retina sees.
        The anti-windup clip below guarantees the landing position x_p + e_cmd
        never exceeds ±orbital_limit.

The range in between
─────────────────────
Target IS visible (|e_pos_delayed| < visual_field_limit) but landing position
x_p + e_pos_delayed would exceed the orbital limit.  Correct behavior: execute
the best saccade possible within range (command is clipped to the limit), then
re-evaluate on the next cycle.  No special logic needed — the clip handles it.

Implicit straight-ahead target
───────────────────────────────
When no explicit fixation target is provided, the simulator defaults to
p_target = [0, 0, 1] (straight ahead).  In that case
    e_pos_delayed ≈ −x_p  (head stationary)
so the saccade command drives the eye back toward center.
This gives the reflexive centration saccades (fast phases) needed for VOR
nystagmus, OKN fast phases, and orbital-reset saccades — WITHOUT requiring an
explicit fixation target.  "target_present" no longer gates the saccade path.

Signal flow
───────────
    e_pos_delayed  (retinal — from visual delay cascade)
         │
         ▼
    [visual field gate]  → suppress if |e_pos_delayed| > visual_field_limit
         │
         ▼
    [orbital gate blend] → blend toward e_reset as |x_p| → orbital_limit
         │
         ▼
    [anti-windup clip]   → hard clip: x_p + e_cmd stays within ±orbital_limit
         │
         ▼
    e_cmd → saccade generator

Parameters
──────────
    orbital_limit       (deg)   mechanical half-range of orbital limit  default 50.0
                                Used for the orbital gate AND the anti-windup clip.
    k_orbital           (1/deg) sigmoid steepness for orbital gate       default 1.0
                                At k=1, gate goes 5%→95% over ±3° around limit.
    alpha_reset         (-)     orbital reset gain; e_reset = -alpha_reset*x_p
                                alpha_reset=0 → pure suppression at the limit.
                                alpha_reset=1 → active centering saccade.    default 1.0
    visual_field_limit  (deg)   retinal eccentricity beyond which the target   default 90.0
                                is considered outside the visual field and the
                                saccade command is suppressed.  Applied to the
                                norm of e_pos_delayed (isotropic, not per-axis).

──────────────────────────────────────────────────────────────────────────────
DEFERRED — Option 2: efference copy anti-windup
──────────────────────────────────────────────────────────────────────────────
If x_p saturates past orbital_limit despite the clip here (e.g. NI drift or
overshoot), the efference copy plant state x_pc will also exceed the limit.
Since the real plant reports soft_limit(x_p) but the efference copy tracks the
unbounded x_pc, the slip cancellation identity w_burst_pred ≡ dx_pc breaks.
Fix: apply plant.soft_limit() to x_pc inside efference_copy.step().
Implement only if e_slip artifacts appear during orbital saturation.
──────────────────────────────────────────────────────────────────────────────
"""

import jax
import jax.numpy as jnp


def select(e_pos_delayed, x_p, theta):
    """Compute motor error command for the saccade generator.

    The implicit target is always "straight ahead" (the default p_target when
    none is specified).  e_pos_delayed already encodes the retinal error from
    whatever target is active, so no separate target_gain gate is needed here.

    Args:
        e_pos_delayed : (3,)   delayed retinal position error (deg)
                               Gaze-centered: angle of target from fovea.
                               With default target [0,0,1]: ≈ −x_p − q_head.
        x_p           : (3,)   plant state (deg)
                               Head-centered: where the eye points in the head.
        theta         : Params  model parameters

    Returns:
        e_cmd : (3,)  motor error command (deg), clipped to ±orbital_limit.
    """
    orbital_limit = theta.brain.orbital_limit
    k             = theta.brain.k_orbital
    alpha_reset   = theta.brain.alpha_reset
    vf_limit      = theta.brain.visual_field_limit

    # ── 1. Visual field gate (retinal / gaze-centered) ────────────────────────
    # Suppress if target is outside the visual field.
    # Uses norm of e_pos_delayed (isotropic visual field).
    e_mag = jnp.linalg.norm(e_pos_delayed) + 1e-9
    gate_vf  = 1.0 - jax.nn.sigmoid(k * (e_mag - vf_limit))   # scalar; ≈1 in-field, ≈0 out

    # Visual error: gated by visual field only.
    # No target_gain gate here — the implicit target (straight-ahead by default)
    # always provides a centration drive for reflexive fast phases.
    e_visual = gate_vf * e_pos_delayed            # (3,)

    # ── 2. Orbital gate (head-centered) ──────────────────────────────────────
    # Blends e_visual toward e_reset as the eye approaches the orbital limit.
    # Per-axis: each axis saturates independently.
    gate_orbital = jax.nn.sigmoid(k * (jnp.abs(x_p) - orbital_limit))   # (3,)
    e_reset      = -alpha_reset * x_p                                     # (3,)

    e_cmd = (1.0 - gate_orbital) * e_visual + gate_orbital * e_reset

    # ── 3. Anti-windup clip (head-centered) ───────────────────────────────────
    # Hard guarantee: landing position x_p + e_cmd stays within ±orbital_limit.
    # Handles the "visible but unreachable" case: clips to best saccade in range.
    e_cmd_clipped = jnp.clip(e_cmd, -orbital_limit - x_p, orbital_limit - x_p)

    return e_cmd_clipped
