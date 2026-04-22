"""Smooth pursuit SSM — leaky integrator + direct feedthrough with Smith predictor.

Drives smooth eye movements to track a moving target.  Receives the EC-corrected
target velocity error (e_combined ≈ v_target) and outputs a pursuit velocity
command that feeds into the neural integrator (NI).

Architecture:
  - Smith predictor (internal): removes the "already commanded" portion from the
    error, eliminating delay-induced oscillation without extra states
  - Integrator state x_p stores the pursuit velocity "memory"
  - Direct feedthrough K_phasic provides fast onset
  - Leak (1/tau_pursuit) prevents unbounded accumulation

Smith predictor (closed-form, no circularity):
    u_pu_now  = x_p + K_phasic · e_pred         (current motor output)
    e_pred    = e_combined − u_pu_now            (Smith: error = reference − output)
    →  e_pred = (e_combined − x_p) / (1 + K_phasic)   (solved explicitly)

Dynamics:
    dx_p/dt  = −x_p / τ_pursuit  +  K_pursuit · e_pred
    u_pursuit = x_p + K_phasic · e_pred

where:
    x_p       (3,)   pursuit velocity memory (deg/s)
    e_combined (3,)  EC-corrected target velocity error, assembled in brain_model.py:
                         target_slip already gated (target_visible × (1−strobed) at retina)
                         motor_ec pre-gated by target_motion_visible before call
                     Ensures OKN (no foveal target) does not drive the pursuit integrator.
    u_pursuit (3,)   pursuit velocity command → NI

Predictor effect:
    At onset:        e_pred = v_target / (1 + K_phasic)  (~45 % of raw error)
                     → integrator receives a gentler kick  → much less oscillation
    At steady state: e_pred → 0  (integrator at rest, u_pursuit ≈ v_target)

EC correction in VS:
    brain_model feeds u_burst + u_pursuit into the EC cascade.
    VS receives slip_delayed + scene · motor_ec → cancels self-generated OKR.

Parameters:
    K_pursuit        integration gain (1/s).  TC ≈ 1/K_pursuit (open-loop).
    K_phasic_pursuit direct feedthrough (dim'less); sets Smith attenuation.
    tau_pursuit      leak TC (s).  Steady-state pursuit gain:
                         gain ≈ K_p·τ·(1+K_ph) / [(1+K_ph)² + K_p·τ]
                     With K_pursuit=2, K_phasic=0.8, tau_pursuit=40 s → ~98.8 %
"""

import jax.numpy as jnp

N_STATES  = 3   # x_p: pursuit velocity memory (deg/s, one per axis)
N_INPUTS  = 3   # e_vel_delayed after saccade EC subtraction
N_OUTPUTS = 3   # u_pursuit: velocity command → NI and VS


def step(x_pursuit, target_slip, motor_ec, brain_params):
    """Single ODE step: gating + Smith predictor → pursuit derivative + output command.

    Args:
        x_pursuit:    (3,)   pursuit memory state (deg/s)
        target_slip:  (3,)   delayed target velocity on retina (deg/s); already gated by
                             target_visible × (1 − strobed) at retinal cascade input
        motor_ec:     (3,)   efference copy pre-gated by target_motion_visible at call site
        brain_params: BrainParams  (reads K_pursuit, K_phasic_pursuit, tau_pursuit, v_max_pursuit)

    Returns:
        dx_pursuit: (3,)  dx_p/dt  (deg/s²)
        u_pursuit:  (3,)  pursuit velocity command (deg/s)
    """
    K_ph = brain_params.K_phasic_pursuit
    # MT/MST velocity saturation: clip slip and EC independently before combining.
    e_combined = (jnp.clip(target_slip, -brain_params.v_max_pursuit, brain_params.v_max_pursuit)
        + jnp.clip(motor_ec,  -brain_params.v_max_pursuit, brain_params.v_max_pursuit)
    )
    # Smith predictor: e_pred = (e_combined − x_pursuit) / (1 + K_phasic)
    e_pred = (e_combined - x_pursuit) / (1.0 + K_ph)

    A  = -(1.0 / brain_params.tau_pursuit) * jnp.eye(3)
    dx_pursuit = A @ x_pursuit + brain_params.K_pursuit * e_pred
    u_pursuit  = x_pursuit + K_ph * e_pred
    return dx_pursuit, u_pursuit
