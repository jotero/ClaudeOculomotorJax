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

import jax.numpy as jnp

N_STATES  = 9   # x_L(3) + x_R(3) + x_null(3)
N_INPUTS  = 3
N_OUTPUTS = 3

# Sub-index slices (relative to x_ni)
_IDX_L    = slice(0, 3)
_IDX_R    = slice(3, 6)
_IDX_NULL = slice(6, 9)


def step(x_ni, u_vel, brain_params):
    """Single ODE step: bilateral NI dynamics + null adaptation + motor command.

    Args:
        x_ni:         (9,)  NI state [x_L (3,) | x_R (3,) | x_null (3,)]
        u_vel:        (3,)  combined eye-velocity command (deg/s) — sign-flipped upstream
        brain_params: BrainParams

    Returns:
        dx:  (9,)  dx_ni/dt
        u_p: (3,)  pulse-step motor command to plant
    """
    x_L    = x_ni[_IDX_L]    # (3,) left  pop
    x_R    = x_ni[_IDX_R]    # (3,) right pop
    x_null = x_ni[_IDX_NULL] # (3,) adapted null

    b_ni   = jnp.asarray(brain_params.b_ni,  dtype=jnp.float32)
    L      = brain_params.orbital_limit
    tau_i  = brain_params.tau_i

    # ── Population equilibria: leak toward b_ni ± half-null ──────────────────
    # b_eff_L = b_ni + x_null/2   (left  pop target rises with rightward null)
    # b_eff_R = b_ni - x_null/2   (right pop target falls with rightward null)
    dx_L_raw = -(1.0 / tau_i) * (x_L - b_ni - x_null / 2.0) + u_vel / 2.0
    dx_R_raw = -(1.0 / tau_i) * (x_R - b_ni + x_null / 2.0) - u_vel / 2.0

    # ── Anti-windup on net ────────────────────────────────────────────────────
    x_net   = x_L - x_R                      # current net position
    dx_net  = dx_L_raw - dx_R_raw            # net derivative before clipping
    dx_sum  = dx_L_raw + dx_R_raw            # common-mode: unaffected by windup

    dx_net  = jnp.where(x_net >= L,  jnp.minimum(dx_net, 0.0), dx_net)
    dx_net  = jnp.where(x_net <= -L, jnp.maximum(dx_net, 0.0), dx_net)

    # Reconstruct individual derivatives from clipped net + unchanged sum
    dx_L = (dx_net + dx_sum) / 2.0
    dx_R = (dx_sum - dx_net) / 2.0

    # ── Null adaptation: null slowly tracks net position ──────────────────────
    dx_null = (x_net - x_null) / brain_params.tau_ni_adapt

    # ── Pulse-step motor command: lag cancellation feedthrough ────────────────
    u_p = x_net + brain_params.tau_p * u_vel

    return jnp.concatenate([dx_L, dx_R, dx_null]), u_p
