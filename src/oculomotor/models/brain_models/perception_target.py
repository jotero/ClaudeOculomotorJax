"""Target perception — target-side post-delay processing.

Encapsulates the target pathway from sensory_out delayed signals to:
  - Pursuit drive (EC-corrected + magnitude-gated + directional-gated target slip)
  - Saccade generator inputs (effective target position + visibility, with
    short-term working memory across brief target occlusions)

Components (per the bigger picture):
  1. EC subtraction         target_slip + visible · ec_d_target  (post-delay)
  2. Hill magnitude gate    K_mag(|ec_d_target|) — closes during fast self-motion
  3. Directional gate       sign-sensitive sigmoid on slip · ec_d_target alignment
                            — suppresses harder when slip and ec_d are opposite
                            (self-motion case), passes when aligned or slip ≈ 0
                            (so steady pursuit is unaffected)
  4. Target working memory  4-state cognitive layer (3-D last-seen position +
                            trust scalar) — lets brief target flashes drive a
                            saccade after the flash ends, and decays slowly when
                            target is gone

State layout (N_STATES = 4):
    x_target_mem = [x_mem_pos (3) | trust (1)]

Outputs of step():
    dx_target_mem            (4,)   state derivative
    target_slip_for_pursuit  (3,)   gated EC-corrected slip → pursuit integrator
    tgt_pos_eff              (3,)   blended raw + memory target position → SG
    tgt_vis_eff              scalar  blended visibility → SG
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


def step(state,
         target_slip, target_visible, target_pos,
         ec_d_target,
         brain_params):
    """Single ODE step for target-side perception.

    Args:
        state:           pt.State   working memory (mem_pos(3) | mem_trust scalar)
        target_slip:     (3,)    raw delayed retinal target velocity (eye frame, deg/s)
        target_visible:  scalar  delayed cyclopean target visibility gate ∈ [0,1]
        target_pos:      (3,)    delayed cyclopean retinal target position (eye frame, deg)
        ec_d_target:     (3,)    delayed EC (cascade-matched to target_vel; eye frame, deg/s)
        brain_params:    BrainParams — reads v_crit_ec_gate, n_ec_gate,
                                       alpha_ec_dir, bias_ec_dir

    Returns:
        dstate:                  pt.State  derivative
        target_slip_for_pursuit: (3,)  EC-corrected + gated → pursuit
        tgt_pos_eff:             (3,)  effective target position → SG
        tgt_vis_eff:             scalar effective target visibility → SG
    """
    # ── 1. Post-delay EC subtraction ──────────────────────────────────────────
    # target_slip has no torsion (retina is 2D); zero the torsion of the EC sub.
    # Visibility-gating ensures EC contribution = 0 when target was invisible
    # at the delayed time (raw target_slip is also zero in that case).
    target_slip_corr = target_slip + target_visible * ec_d_target.at[2].set(0.0)

    # ── 2. Hill magnitude gate ───────────────────────────────────────────────
    K_mag = 1.0 / (1.0 + (jnp.linalg.norm(ec_d_target) / brain_params.v_crit_ec_gate)
                   ** brain_params.n_ec_gate)

    # ── 3. Directional gate (sign-sensitive scalar) ──────────────────────────
    # Gate signal = signed projection of the *raw* delayed target slip onto
    # ec_d_target's direction. Slip and ec_d opposite (self-motion case) →
    # close gate. Slip aligned with ec_d (real motion in same direction as eye,
    # e.g., steady pursuit where raw target_slip ≈ 0) → keep gate open.
    ec_norm  = jnp.linalg.norm(ec_d_target) + 1e-9
    ec_hat   = ec_d_target / ec_norm
    slip_dot = jnp.dot(target_slip, ec_hat)
    K_dir    = jax.nn.sigmoid((slip_dot + brain_params.bias_ec_dir) * brain_params.alpha_ec_dir)

    target_slip_for_pursuit = K_mag * K_dir * target_slip_corr

    # ── 4. Target working memory ─────────────────────────────────────────────
    # Memory is the LAST-SEEN cyclopean retinal target error (deg). The SG
    # uses this to fire a saccade toward the remembered location after the
    # flash ends. The next flash overwrites the memory based on the new
    # retinal error. No efference copy: if a saccade lands accurately, the
    # next flash will reset the memory to ~0; otherwise the residual error
    # drives a corrective saccade.
    x_mem = state.mem_pos
    trust = state.mem_trust

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

    return dstate, target_slip_for_pursuit, tgt_pos_eff, tgt_vis_eff


# ── Legacy flat-array adapters (deleted once brain_model migrates to BrainState) ─

N_STATES = 4   # 3-D last-seen position + 1 trust scalar


def from_array(x_target_mem):
    """(4,) flat array → pt.State."""
    return State(mem_pos=x_target_mem[0:3], mem_trust=x_target_mem[3])


def to_array(state):
    """pt.State → (4,) flat array."""
    return jnp.concatenate([state.mem_pos, jnp.array([state.mem_trust])])
