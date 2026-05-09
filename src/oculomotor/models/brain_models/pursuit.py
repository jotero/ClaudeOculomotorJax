"""Smooth pursuit SSM — bilateral leaky integrator + Smith predictor.

Drives smooth eye movements to track a moving target.  Receives the EC-corrected
target slip (target_slip_ec ≈ v_target) and outputs a pursuit velocity command
that feeds into the neural integrator (NI).

Bilateral (push-pull) architecture
----------------------------------
Real MT / MST has direction-selective populations: rightward-preferring cells
fire for rightward target motion, leftward-preferring cells for leftward.
We split the pursuit integrator into two non-negative populations:

    x_R  (3,)  — rightward / upward / extorting pursuit memory
    x_L  (3,)  — leftward  / downward / intorting pursuit memory

Each population leaks with τ_pursuit and integrates the rectified part of
the Smith-predictor error:

    e_pred  =  (target_slip_ec − x_net) / (1 + K_phasic)
    dx_R/dt =  −x_R / τ  +  K_pursuit · max(0,  e_pred)
    dx_L/dt =  −x_L / τ  +  K_pursuit · max(0, −e_pred)

The NET pursuit memory drives the eye:

    x_net      = x_R − x_L
    u_pursuit  = x_net + K_phasic · e_pred

Identity check.  Because  max(0, e) − max(0, −e) = e  for any signed e,
the net dynamics dx_net/dt = −x_net/τ + K_pursuit · e_pred are EQUAL to the
single-integrator case.  Bilateralisation preserves the closed-form Smith
predictor and steady-state pursuit gain — it just exposes the per-side
populations as separate states.

Why bilateralise — clinical motivation
--------------------------------------
- Unilateral cerebellar / MT lesion → asymmetric pursuit gain (e.g. reduced
  rightward gain only).  In the single-integrator model you can only scale
  K_pursuit globally — no way to model directional asymmetry.
- With per-side populations, downstream lesion gains can target one side
  (g_pursuit_R, g_pursuit_L — future per-side gains, not implemented yet).

Smith predictor (closed-form, no circularity)
----------------------------------------------
    u_pu_now      = x_net + K_phasic · e_pred             (current motor output)
    e_pred        = target_slip_ec − u_pu_now             (Smith: error = ref − output)
    →  e_pred = (target_slip_ec − x_net) / (1 + K_phasic) (solved explicitly)

EC correction in VS
-------------------
brain_model feeds u_burst + u_pursuit into the EC cascade.
VS receives slip_delayed + scene · motor_ec_okr → cancels self-generated OKR.
Pursuit receives target_slip + motor_ec_pursuit → cancels self-generated pursuit.

Parameters (in BrainParams)
---------------------------
    K_pursuit         integration gain (1/s).  TC ≈ 1/K_pursuit (open-loop).
    K_phasic_pursuit  direct feedthrough (dim'less); sets Smith attenuation.
    tau_pursuit       leak TC (s).  Steady-state pursuit gain:
                          gain ≈ K_p·τ·(1+K_ph) / [(1+K_ph)² + K_p·τ]
                      With K_pursuit=4, K_phasic=5, tau_pursuit=40 s → ~98.8 %
"""

from typing import NamedTuple

import jax.numpy as jnp


# ── State layout ───────────────────────────────────────────────────────────────
# x_pursuit shape (6,) = [x_R (3) | x_L (3)]
N_STATES  = 6   # bilateral push-pull: 3 axes × 2 pops
N_INPUTS  = 6   # target_slip_ec(3) + eye_pos(3) — see _IDX_INPUT_* below
N_OUTPUTS = 3   # u_pursuit: net velocity command → NI

_IDX_PURSUIT_R = slice(0, 3)   # rightward / upward / extorting pop
_IDX_PURSUIT_L = slice(3, 6)   # leftward  / downward / intorting pop


# ── Local registries (Phase-2 contract) ──────────────────────────────────────

class Activations(NamedTuple):
    """Pursuit firing rates — bilateral push-pull pops."""
    R: jnp.ndarray   # (3,) right MT/MST pop  (rightward / upward / extorting)
    L: jnp.ndarray   # (3,) left  MT/MST pop  (leftward  / downward / intorting)


class Decoded(NamedTuple):
    """Pursuit decoded readout — net velocity command consumed by NI."""
    net: jnp.ndarray   # (3,) signed = R − L   pursuit velocity (deg/s)


def read_activations(x_pursuit):
    """Project pursuit raw state → Activations.

    Pops are non-negative by integration construction (rectified drives), so
    state == activation here.  No `max(0, ·)` needed.
    """
    return Activations(R=x_pursuit[_IDX_PURSUIT_R], L=x_pursuit[_IDX_PURSUIT_L])


def decode_states(acts):
    """Pursuit net velocity from bilateral pops."""
    return Decoded(net=acts.R - acts.L)

# Bundled-input layout — match the SSM convention: step(x, u, theta).
_IDX_INPUT_TARGET_SLIP_EC = slice(0, 3)   # EC-corrected target slip (deg/s)
_IDX_INPUT_EYE_POS        = slice(3, 6)   # current eye position [H, V, T] deg (NI net)


def step(x_pursuit, u, brain_params):
    """Single ODE step: bilateral pursuit integrator with Smith predictor.

    Args:
        x_pursuit:    (6,)  pursuit state [x_R (3) | x_L (3)] — non-negative pops
        u:            (6,)  bundled input vector:
                            [_IDX_INPUT_TARGET_SLIP_EC] = EC-corrected target slip (3,)
                                                          (target_slip + motor_ec · target_motion_visible)
                            [_IDX_INPUT_EYE_POS]        = current eye position (3,) [H, V, T] (deg)
        brain_params: BrainParams  (reads K_pursuit, K_phasic_pursuit, tau_pursuit)

    Returns:
        dx_pursuit: (6,)  state derivative [dx_R (3) | dx_L (3)]
        u_pursuit:  (3,)  net pursuit velocity command (deg/s) → NI
    """
    target_slip_ec = u[_IDX_INPUT_TARGET_SLIP_EC]
    eye_pos        = u[_IDX_INPUT_EYE_POS]

    x_R = x_pursuit[_IDX_PURSUIT_R]
    x_L = x_pursuit[_IDX_PURSUIT_L]
    x_net = x_R - x_L                                          # signed pursuit memory

    K_ph = brain_params.K_phasic_pursuit
    # Smith predictor on the NET memory (preserves single-integrator dynamics).
    e_pred = (target_slip_ec - x_net) / (1.0 + K_ph)

    # Rectified push-pull drive — each population integrates only its preferred direction.
    drive_R = jnp.maximum( e_pred, 0.0)
    drive_L = jnp.maximum(-e_pred, 0.0)

    leak = -1.0 / brain_params.tau_pursuit
    dx_R = leak * x_R + brain_params.K_pursuit * drive_R
    dx_L = leak * x_L + brain_params.K_pursuit * drive_L
    dx_pursuit = jnp.concatenate([dx_R, dx_L])

    # Net activation sent downstream (NI / EC).  Identical to old single-state output.
    u_pursuit = x_net + K_ph * e_pred

    # eye_pos kept in the interface for symmetry / future use (Listing's torsion
    # is now applied centrally in brain_model.step on the SUMMED velocity command).
    _ = eye_pos
    return dx_pursuit, u_pursuit


