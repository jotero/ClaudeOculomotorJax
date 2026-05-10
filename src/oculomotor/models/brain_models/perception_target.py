"""Target perception — FEF/dlPFC working memory feeding the saccade generator.

Pursuit drive (post-delay EC subtraction + magnitude/directional gates) used
to live here but moved into `cerebellum.py` (pursuit region — paraflocculus
ventral / vermis VI–VII).  This module now owns only the working-memory layer
that lets brief target flashes drive a saccade after the flash ends, and that
decays slowly when the target is gone.

Component (working memory):
  4-state cognitive layer (3-D last-seen position + trust scalar).  The SG
  uses these to fire a saccade toward the remembered location after the
  flash ends.  Memory drains proportional to |ec_d_target|, so any eye
  movement (saccade, fast pursuit, head-impulse fast-phase) consumes the
  memory and prevents re-triggering on the residual.

State layout (N_STATES = 4):
    x_target_mem = [x_mem_pos (3) | trust (1)]

Outputs of step():
    dstate         pt.State derivative
    tgt_pos_eff   (3,)   blended raw + memory target position → SG
    tgt_vis_eff   scalar blended visibility                   → SG
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ── Working memory time constants — local module hacks, not in BrainParams ──
# These shape FEF/dlPFC-style gaze evidence accumulation so brief flashes can
# accumulate enough trust to fire a saccade, then the saccade burst consumes
# the memory so it doesn't re-trigger.
_TAU_TARGET_MEM_UPDATE       = 0.05    # memory lock TC when target is visible (s)
_TAU_TARGET_MEM_TRUST_RISE   = 0.10    # trust rise TC when target is seen (s)
_TAU_TARGET_MEM_TRUST_DECAY  = 5.0     # trust decay TC when target is unseen (s)
_TARGET_MEM_TRUST_THRESHOLD  = 0.2     # trust level above which SG commits to saccade mode

# ── State + registries ────────────────────────────────────────────────────────

class State(NamedTuple):
    """Target perception state — working-memory population + confidence."""
    mem_pos:   jnp.ndarray   # (3,)   target working memory pop  [dlPFC / FEF]
    mem_trust: jnp.ndarray   # scalar memory confidence ∈ [0,1]  [dlPFC]


# State == Activations: working-memory pop is itself a firing-rate signal.
Activations = State


def rest_state():
    """Zero state — used for SimState initialisation."""
    return State(mem_pos=jnp.zeros(3), mem_trust=jnp.float32(0.0))


def read_activations(state):
    """Working-memory IS the activation — identity projection."""
    return state


def step(activations,
         target_visible, target_pos,
         ec_d_target):
    """Single ODE step for target-side perception.

    Activation-driven: working-memory pop firing rates come from `activations`
    (acts.pt).  State == Activations for pt today (identity projection).

    Args:
        activations:     pt.Activations  mem_pos (3,) | mem_trust (scalar)
        target_visible:  scalar  delayed cyclopean target visibility gate ∈ [0,1]
        target_pos:      (3,)    delayed cyclopean retinal target position (eye frame, deg)
        ec_d_target:     (3,)    delayed EC (cascade-matched to target_vel; eye frame, deg/s)

    Returns:
        dstate:        pt.State  derivative
        tgt_pos_eff:   (3,)    effective target position → SG
        tgt_vis_eff:   scalar  effective target visibility → SG
    """
    x_mem = activations.mem_pos
    trust = activations.mem_trust

    # Memory drain proportional to delayed-EC magnitude. Whenever the eye is
    # moving (saccade burst, fast pursuit overshoot, head-impulse fast-phase),
    # |ec_d_target| is large in deg/s → drain rate grows in 1/s, draining the
    # remembered position fast even for small saccades whose cascade-delayed
    # EC peaks at only tens of deg/s. Between saccades the LP cascade has
    # decayed and inv_consume → 0, so the memory holds.
    # Trust decays naturally with τ_trust_decay when no flashes arrive —
    # that's the "give up and look home" timescale, not driven by the EC.
    inv_consume = jnp.linalg.norm(ec_d_target)
    dx_mem = target_visible * (target_pos - x_mem) / _TAU_TARGET_MEM_UPDATE \
             - inv_consume * x_mem

    inv_rise   = 1.0 / _TAU_TARGET_MEM_TRUST_RISE
    inv_decay  = 1.0 / _TAU_TARGET_MEM_TRUST_DECAY
    gain_trust = target_visible * inv_rise + (1.0 - target_visible) * inv_decay
    dx_trust   = gain_trust * (target_visible - trust)
    dstate     = State(mem_pos=dx_mem, mem_trust=dx_trust)

    # Memory commits to "saccade mode" once trust crosses a small threshold,
    # binarising via a steep sigmoid so brief flashes accumulate evidence and
    # fire one saccade.
    mem_active  = jax.nn.sigmoid(50.0 * (trust - _TARGET_MEM_TRUST_THRESHOLD))
    tgt_pos_eff = target_visible * target_pos + (1.0 - target_visible) * mem_active * x_mem
    tgt_vis_eff = jnp.maximum(target_visible, mem_active)

    return dstate, tgt_pos_eff, tgt_vis_eff


# ── Legacy flat-array adapters (deleted once brain_model migrates to BrainState) ─

N_STATES = 4   # 3-D last-seen position + 1 trust scalar


def from_array(x_target_mem):
    """(4,) flat array → pt.State."""
    return State(mem_pos=x_target_mem[0:3], mem_trust=x_target_mem[3])


def to_array(state):
    """pt.State → (4,) flat array."""
    return jnp.concatenate([state.mem_pos, jnp.array([state.mem_trust])])
