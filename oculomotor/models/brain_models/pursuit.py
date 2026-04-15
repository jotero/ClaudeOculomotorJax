"""Smooth pursuit SSM — leaky integrator + direct feedthrough (Smith predictor).

Drives smooth eye movements to track a moving target.  Receives a prediction
error (computed in brain_model.py) and outputs a pursuit velocity command that
feeds into the neural integrator (NI).

Architecture:
  - Integrator state x_p stores the pursuit velocity "memory"
  - Direct feedthrough K_phasic provides fast onset without waiting for
    the integrator to charge
  - Leak (1/tau_pursuit) prevents unbounded accumulation; set large (≥30 s)
    for robust velocity maintenance

Dynamics:
    dx_p/dt = −x_p / τ_pursuit  +  K_pursuit · e_pred
    u_pursuit = x_p  +  K_phasic · e_pred

where:
    x_p    (3,)   pursuit velocity memory (deg/s)
    e_pred (3,)   Smith predictor error from brain_model.py (deg/s):
                      e_pred = (e_combined − x_p) / (1 + K_phasic)
                      e_combined = vel_delayed + scene · motor_ec  ≈ v_target
                  This removes the "already commanded" portion from the error,
                  eliminating delay-induced oscillation.
    u_pursuit (3,) pursuit velocity command → NI

Signal flow within brain_model (see brain_model.py for details):
    1. motor_ec = delay(u_burst + u_pursuit)  (combined EC read)
    2. e_combined = vel_delayed + scene · motor_ec  (≈ v_target)
    3. e_pred = (e_combined − x_pursuit) / (1 + K_phasic)   (Smith predictor)
    4. this step(x_pursuit, e_pred) → u_pursuit
    5. u_pursuit added to NI drive: −w_est + u_burst + u_pursuit
    6. u_burst + u_pursuit advanced into EC cascade

Pursuit EC to VS (handled by combined EC cascade, not x_pursuit directly):
    motor_ec ≈ delay(u_pursuit) during pursuit → cancels slip_delayed in VS.
    VS sees: slip_delayed + motor_ec ≈ −u_pursuit + u_pursuit = 0  ✓

Parameters:
    K_pursuit        integration gain (1/s).
    K_phasic_pursuit direct feedthrough gain (dim'less); also sets the
                     Smith predictor attenuation: onset drive = v_target/(1+K_ph)
    tau_pursuit      leak TC (s).  Steady-state pursuit gain:
                         gain ≈ K_p · τ / [(1+K_ph) + K_p·τ/(1+K_ph)]
                     With K_pursuit=2, K_phasic=0.8, tau_pursuit=40 s → ~98.8 %
"""

import jax.numpy as jnp

N_STATES  = 3   # x_p: pursuit velocity memory (deg/s, one per axis)
N_INPUTS  = 3   # e_vel_delayed after saccade EC subtraction
N_OUTPUTS = 3   # u_pursuit: velocity command → NI and VS


def step(x_pursuit, e_vel, brain_params):
    """Single ODE step: pursuit velocity derivative + output command.

    Args:
        x_pursuit:    (3,)   pursuit memory state (deg/s)
        e_vel:        (3,)   Smith predictor error: (e_combined − x_pursuit)/(1+K_ph)
                             Computed in brain_model.py before calling this function.
        brain_params: BrainParams  (reads K_pursuit, K_phasic_pursuit, tau_pursuit)

    Returns:
        dx_pursuit: (3,)  dx_p/dt  (deg/s²)
        u_pursuit:  (3,)  pursuit velocity command (deg/s)
    """
    A  = -(1.0 / brain_params.tau_pursuit) * jnp.eye(3)
    dx_pursuit = A @ x_pursuit + brain_params.K_pursuit * e_vel
    u_pursuit  = x_pursuit + brain_params.K_phasic_pursuit * e_vel
    return dx_pursuit, u_pursuit
