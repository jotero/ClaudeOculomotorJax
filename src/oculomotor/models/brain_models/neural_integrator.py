"""Neural Integrator SSM — bilateral push-pull with null-point adaptation.

Mirrors the velocity_storage bilateral architecture.  Two populations (Left,
Right) model the bilateral nucleus prepositus hypoglossi (NPH) / interstitial
nucleus of Cajal (INC) organisation.

State:  x_ni = [x_L (3,) | x_R (3,) | x_null (3,)]         (9,)
Input:  u_vel                                                 (3,)  velocity command
Output: u_p — pulse-step motor command to plant               (3,)

ABCD system (bilateral core):
────────────────────────────────────────────────────────────────────────
    dx_L /dt = −(1/τ_i)·(x_L − b_ni − x_null/2) + u_vel/2
    dx_R /dt = −(1/τ_i)·(x_R − b_ni + x_null/2) − u_vel/2

    Net position:  x_net = x_L − x_R   (identical to old scalar x_ni)
    d(x_net)/dt  = −(1/τ_i)·(x_net − x_null) + u_vel      ← leaks toward null, not 0

    u_p  =  x_net  +  τ_p · u_vel     (pulse-step: lag-cancelled motor command)

Null-point adaptation:
────────────────────────────────────────────────────────────────────────
    dx_null/dt = (x_net − x_null) / τ_ni_adapt

    The null slowly tracks the current net position.  During sustained eccentric
    gaze (x_net = E), x_null → E.  On return to centre (x_net = 0), the NI leaks
    toward x_null = E → eye drifts back (slow phase eccentric) → fast phase toward
    centre → rebound nystagmus. ✓

    τ_ni_adapt default 20 s: ~half-period rebound after sustained right gaze.
    τ_ni_adapt → ∞ (very large):  null frozen at 0 → reverts to old NI behaviour.

Bilateral conventions (mirror velocity_storage):
    Model LEFT pop  (x_L, 0:3) codes RIGHTWARD gaze = anatomical RIGHT NPH.
    Model RIGHT pop (x_R, 3:6) codes LEFTWARD  gaze = anatomical LEFT  NPH.
    Net x_L − x_R > 0  →  rightward eye position command.

    b_ni (default 0):  NPH intrinsic resting bias.  At net level (C @ x = 0) the
    resting discharge cancels, so b_ni=0 is physiologically appropriate unless
    modelling unilateral NPH lesions (future work).

Anti-windup:
    Applied to the net derivative d(x_net)/dt before distributing back to
    individual populations.  Prevents integrator wind-up beyond ±orbital_limit.

Parameters:
    τ_i         (s)   — leak TC.         Default 25 s (healthy).
    τ_p         (s)   — plant TC copy.   Default 0.15 s.
    b_ni        (deg) — NPH resting bias. Default 0.
    τ_ni_adapt  (s)   — null adaptation TC.  Default 20 s.
    orbital_limit (deg) — oculomotor range half-width.  Default 50 deg.
"""

from typing import NamedTuple

import jax.numpy as jnp

# ── State + registries ────────────────────────────────────────────────────────

class State(NamedTuple):
    """NI state — bilateral push-pull pops + null adaptation register."""
    L:    jnp.ndarray   # (3,) left  NPH/INC pop  (rectified ≥ 0 by construction)
    R:    jnp.ndarray   # (3,) right NPH/INC pop  (rectified ≥ 0 by construction)
    null: jnp.ndarray   # (3,) signed adaptation register (drifts toward x_net)


class Activations(NamedTuple):
    """NI firing rates — bilateral pops only (null is a setpoint, in Weights)."""
    L: jnp.ndarray   # (3,) left  NPH/INC pop
    R: jnp.ndarray   # (3,) right NPH/INC pop


class Decoded(NamedTuple):
    """NI decoded readout — net eye position consumed by FCP."""
    net: jnp.ndarray   # (3,) signed = L − R   eye position estimate (deg)


class Weights(NamedTuple):
    """NI tonic / null / setpoint registers (long-term: learned weights)."""
    null: jnp.ndarray   # (3,) signed   slow null adaptation register


def rest_state():
    """Zero state — used for SimState initialisation."""
    return State(L=jnp.zeros(3), R=jnp.zeros(3), null=jnp.zeros(3))


def read_activations(state):
    """NI bilateral pops are firing rates by construction — direct projection."""
    return Activations(L=state.L, R=state.R)


def decode_states(acts):
    """NI net eye position from bilateral pops."""
    return Decoded(net=acts.L - acts.R)


def read_weights(state):
    """NI null adaptation register."""
    return Weights(null=state.null)


def step(state, u_vel, brain_params, u_tonic=0.0):
    """Single ODE step: bilateral NI dynamics + null adaptation + motor command.

    Args:
        state:        ni.State   (L, R, null) — each (3,)
        u_vel:        (3,)  combined eye-velocity command (deg/s) — sign-flipped upstream
        brain_params: BrainParams
        u_tonic:      (3,)  tonic position-offset set-point (e.g. OCR).
                            Acts as a shift on x_null for the population leak target,
                            so x_net leaks toward (x_null + u_tonic). A saccade landing
                            at the OCR position is therefore stable (no drift back to 0).
                            Not added to u_p directly — it flows through the integrator,
                            so x_ni already reflects the offset and ec_pos stays
                            consistent with the actual eye position.

    Returns:
        dstate: ni.State   state derivative
        u_p:    (3,)       pulse-step motor command to plant
    """
    x_L    = state.L
    x_R    = state.R
    x_null = state.null

    b_ni   = jnp.asarray(brain_params.b_ni,  dtype=jnp.float32)
    L      = brain_params.orbital_limit
    # Per-axis NI leak TC: yaw uses tau_i directly; pitch/roll scale by their fractions.
    # Torsional integrator is leakier (~7.5 s vs 25 s) per Crawford & Vilis 1991.
    tau_i  = brain_params.tau_i * jnp.array([1.0,
                                              brain_params.tau_i_pitch_frac,
                                              brain_params.tau_i_roll_frac])

    # u_tonic shifts the effective null/leak target without altering the stored
    # x_null state. Without quick-phase resets, x_net only reaches a fraction
    # τ_ni_adapt / (τ_i + τ_ni_adapt) ≈ 0.44 of u_tonic at SS — saccades and
    # quick phases drive the rest of the way (visible in the OCR cascade bench).
    x_null_eff = x_null + u_tonic

    # ── Population equilibria: leak toward b_ni ± half-(shifted)-null ────────
    # b_eff_L = b_ni + x_null_eff/2   (left  pop target rises with rightward null)
    # b_eff_R = b_ni - x_null_eff/2   (right pop target falls with rightward null)
    dx_L_raw = -(1.0 / tau_i) * (x_L - b_ni - x_null_eff / 2.0) + u_vel / 2.0
    dx_R_raw = -(1.0 / tau_i) * (x_R - b_ni + x_null_eff / 2.0) - u_vel / 2.0

    # ── Anti-windup on net ────────────────────────────────────────────────────
    x_net   = x_L - x_R                      # current net position
    dx_net  = dx_L_raw - dx_R_raw            # net derivative before clipping
    dx_sum  = dx_L_raw + dx_R_raw            # common-mode: unaffected by windup

    dx_net  = jnp.where(x_net >= L,  jnp.minimum(dx_net, 0.0), dx_net)
    dx_net  = jnp.where(x_net <= -L, jnp.maximum(dx_net, 0.0), dx_net)

    # Reconstruct individual derivatives from clipped net + unchanged sum
    dx_L = (dx_net + dx_sum) / 2.0
    dx_R = (dx_sum - dx_net) / 2.0

    # ── Null adaptation: null tracks (x_net − x_null_eff) ────────────────────
    # With sustained u_tonic and no input the system has a 1-D family of
    # equilibria along x_net = x_null + u_tonic. Starting from (0,0) it settles
    # at  x_net = u_tonic·τ_ni_adapt/(τ_i+τ_ni_adapt)  and
    #     x_null = -u_tonic·τ_i/(τ_i+τ_ni_adapt)  on TC τ_eff = τ_i·τ_ni_adapt/(τ_i+τ_ni_adapt).
    # So the null partially adapts to OCR — when OCR is later removed, x_null
    # stays negative briefly and drives a small post-OCR rebound, which is at
    # least directionally consistent with reported post-tilt-removal drift.
    dx_null = (x_net - x_null_eff) / brain_params.tau_ni_adapt

    # ── Pulse-step motor command: lag cancellation feedthrough ────────────────
    u_p = x_net + brain_params.tau_p * u_vel

    return State(L=dx_L, R=dx_R, null=dx_null), u_p


# ── Legacy flat-array adapters (deleted once brain_model migrates to BrainState) ─

N_STATES  = 9   # x_L(3) + x_R(3) + x_null(3)
N_INPUTS  = 3
N_OUTPUTS = 3


def from_array(x_ni):
    """(9,) flat array → ni.State."""
    return State(L=x_ni[0:3], R=x_ni[3:6], null=x_ni[6:9])


def to_array(state):
    """ni.State → (9,) flat array."""
    return jnp.concatenate([state.L, state.R, state.null])
