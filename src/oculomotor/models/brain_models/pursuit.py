"""Smooth pursuit SSM — leaky integrator + direct feedthrough with Smith predictor.

Drives smooth eye movements to track a moving target.  Receives the EC-corrected
target slip (target_slip_ec ≈ v_target) and outputs a pursuit velocity command
that feeds into the neural integrator (NI).

Architecture:
  - Smith predictor (internal): removes the "already commanded" portion from the
    error, eliminating delay-induced oscillation without extra states
  - Integrator state x_p stores the pursuit velocity "memory"
  - Direct feedthrough K_phasic provides fast onset
  - Leak (1/tau_pursuit) prevents unbounded accumulation

Smith predictor (closed-form, no circularity):
    u_pu_now      = x_p + K_phasic · e_pred              (current motor output)
    e_pred        = target_slip_ec − u_pu_now             (Smith: error = reference − output)
    →  e_pred = (target_slip_ec − x_p) / (1 + K_phasic)  (solved explicitly)

Dynamics:
    dx_p/dt  = −x_p / τ_pursuit  +  K_pursuit · e_pred
    u_pursuit = x_p + K_phasic · e_pred

where:
    x_p              (3,)  pursuit velocity memory (deg/s)
    target_slip_ec   (3,)  EC-corrected target velocity error, assembled in brain_model.py:
                               target_slip + motor_ec_pursuit * target_motion_visible
                           target_slip pre-clipped at v_max_target_vel before visual cascade
                           motor_ec pre-clipped at v_max_pursuit before pursuit EC cascade
                           target_motion_visible gates out strobed/invisible targets
    u_pursuit        (3,)  pursuit velocity command → NI

Predictor effect:
    At onset:        e_pred = v_target / (1 + K_phasic)  (~45 % of raw error)
                     → integrator receives a gentler kick  → much less oscillation
    At steady state: e_pred → 0  (integrator at rest, u_pursuit ≈ v_target)

EC correction in VS:
    brain_model feeds u_burst + u_pursuit into the EC cascade.
    VS receives slip_delayed + scene · motor_ec_okr → cancels self-generated OKR.
    Pursuit receives target_slip + motor_ec_pursuit → cancels self-generated pursuit.

Parameters:
    K_pursuit        integration gain (1/s).  TC ≈ 1/K_pursuit (open-loop).
    K_phasic_pursuit direct feedthrough (dim'less); sets Smith attenuation.
    tau_pursuit      leak TC (s).  Steady-state pursuit gain:
                         gain ≈ K_p·τ·(1+K_ph) / [(1+K_ph)² + K_p·τ]
                     With K_pursuit=2, K_phasic=0.8, tau_pursuit=40 s → ~98.8 %
"""

import jax.numpy as jnp

from oculomotor.models.brain_models import listing

N_STATES  = 3   # x_p: pursuit velocity memory (deg/s, one per axis)
N_INPUTS  = 6   # target_slip_ec(3) + eye_pos(3) — see _IDX_INPUT_* below
N_OUTPUTS = 3   # u_pursuit: velocity command → NI and VS

# Bundled-input layout — match the SSM convention: step(x, u, theta).
_IDX_INPUT_TARGET_SLIP_EC = slice(0, 3)   # EC-corrected target slip (deg/s)
_IDX_INPUT_EYE_POS        = slice(3, 6)   # current eye position [H, V, T] deg (NI net)


def step(x_pursuit, u, brain_params):
    """Single ODE step: Smith predictor → pursuit derivative + output command.

    Args:
        x_pursuit:    (3,)  pursuit memory state (deg/s)
        u:            (6,)  bundled input vector:
                            [_IDX_INPUT_TARGET_SLIP_EC] = EC-corrected target slip (3,)
                                                          (target_slip + motor_ec · target_motion_visible)
                            [_IDX_INPUT_EYE_POS]        = current eye position (3,) [H, V, T] (deg)
        brain_params: BrainParams  (reads K_pursuit, K_phasic_pursuit, tau_pursuit,
                                          listing_primary)

    Returns:
        dx_pursuit: (3,)  dx_p/dt  (deg/s²)
        u_pursuit:  (3,)  pursuit velocity command (deg/s), torsional component includes
                          Listing's half-angle correction
    """
    target_slip_ec = u[_IDX_INPUT_TARGET_SLIP_EC]
    eye_pos        = u[_IDX_INPUT_EYE_POS]

    K_ph = brain_params.K_phasic_pursuit
    # Smith predictor: e_pred = (target_slip_ec − x_pursuit) / (1 + K_phasic)
    e_pred = (target_slip_ec - x_pursuit) / (1.0 + K_ph)

    A  = -(1.0 / brain_params.tau_pursuit) * jnp.eye(3)
    dx_pursuit = A @ x_pursuit + brain_params.K_pursuit * e_pred
    u_pursuit  = x_pursuit + K_ph * e_pred

    vel_torsion = listing.pursuit_torsion(eye_pos, u_pursuit, brain_params.listing_primary)
    u_pursuit   = u_pursuit.at[2].add(vel_torsion)
    return dx_pursuit, u_pursuit
