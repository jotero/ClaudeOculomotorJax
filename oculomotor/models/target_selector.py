"""Target selector — upstream feedthrough for the saccade generator.

No internal states.  Pure function, JAX/jit/grad compatible.
Does NOT follow the SSM step() convention — it is a feedthrough, not an SSM.

Sits between the visual delay cascade output and the saccade generator input.
Combines two motor error signals into a single e_cmd that the SG treats as a
normal retinal position error:

    Visual mode    (|x_p| << orbital_limit):
        e_cmd ≈ e_pos_delayed
        The eye is within range; drive foveation saccades to the visual target.

    Orbital limit mode  (|x_p| ≥ orbital_limit):
        e_cmd ≈ alpha_reset * (−x_p)
        The eye is at the orbital limit; with alpha_reset ≈ 0 this suppresses
        the command; increase alpha_reset to add an active centering component.

Mode selection is smooth (per axis):
    gate_reset = sigmoid(k_orbital * (|x_p| − orbital_limit))
        ≈ 0  when eye is well within range  → full visual mode
        ≈ 1  when eye is at or past limit   → command blended toward e_reset

    e_cmd = (1 − gate_reset) * e_visual + gate_reset * e_reset
    e_reset = −alpha_reset * x_p

Anti-windup clip (per axis):
    e_cmd_clipped = clip(e_cmd, −limit − x_p, +limit − x_p)
    Ensures the commanded landing position (x_p + e_cmd) stays within ±limit.
    Prevents the SG from generating a burst that the plant cannot execute.
    jnp.clip gradient = 0 at the boundary — acceptable for fitting.

──────────────────────────────────────────────────────────────────────────────
Option 2 (DEFERRED): efference copy anti-windup
──────────────────────────────────────────────────────────────────────────────
If x_p saturates past orbital_limit despite the clip here (e.g. NI drift or
overshoot), the efference copy plant state x_pc will also exceed the limit.
Since the real plant reports soft_limit(x_p) but the efference copy tracks the
unbounded x_pc, the slip cancellation identity w_burst_pred ≡ dx_pc breaks.
Fix: apply plant.soft_limit() to x_pc inside efference_copy.step().
Implement only if e_slip artifacts appear during orbital saturation.
──────────────────────────────────────────────────────────────────────────────

Parameters
──────────
    orbital_limit  (deg)    half-range of orbital limit             default 50.0
    k_orbital      (1/deg)  sigmoid steepness for suppression gate  default 1.0
    alpha_reset    (-)      reset gain; e_reset = -alpha_reset*x_p  default 0.1
                            Set close to 0 to suppress; increase for active centering.

Runtime inputs
──────────────
    target_gain    scalar   0 = dark (no visible target), 1 = target present.
                            Passed from simulate() target_present_array.
                            Gates e_visual; does not affect the orbital reset term.
"""

import jax
import jax.numpy as jnp


def select(e_pos_delayed, x_p, theta, target_gain=1.0):
    """Compute motor error command for the saccade generator.

    Args:
        e_pos_delayed : (3,)   delayed retinal position error (deg)
        x_p           : (3,)   raw (unsaturated) plant state (deg)
        theta         : dict
        target_gain   : scalar in [0, 1].  0 = no visible target (dark);
                        1 = target present.  Gates e_visual before blending.
                        Analogous to scene_gain for the slip pathway.

    Returns:
        e_cmd : (3,)  motor error command (deg), clipped to orbital range
    """
    limit       = theta.get('orbital_limit', 50.0)
    k           = theta.get('k_orbital',      10.0)
    alpha_reset = theta.get('alpha_reset',    1.0)

    # Per-axis gate: smoothly transitions from visual to orbital reset mode.
    gate_reset = jax.nn.sigmoid(k * (jnp.abs(x_p) - limit))   # (3,)

    e_visual = target_gain * e_pos_delayed           # zero when no target visible
    e_reset  = -alpha_reset * x_p

    e_cmd = (1.0 - gate_reset) * e_visual + gate_reset * e_reset

    # Per-axis anti-windup clip: landing position stays within ±limit
    e_cmd_clipped = jnp.clip(e_cmd, -limit - x_p, limit - x_p)

    return e_cmd_clipped
