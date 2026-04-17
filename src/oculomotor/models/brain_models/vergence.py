"""Vergence SSM — disparity-driven position integrator + Smith predictor.

Drives disconjugate eye movements (convergence/divergence) to align both
foveas on a binocular target.  Structurally identical to smooth pursuit
(pursuit.py) but integrates *position* disparity rather than *velocity* slip.

Signal flow:
    binocular disparity  e_disp = pos_delayed_L − pos_delayed_R  (deg)
        ↓ Smith predictor (closed-form, no extra states)
    vergence integrator  x_verg  (3,)  vergence position memory (deg)
        ↓ output
    u_verg  (3,)  vergence position command  → split ± ½ in brain_model:
        motor_cmd_L = motor_cmd_version + ½ · u_verg   (L eye converges rightward)
        motor_cmd_R = motor_cmd_version − ½ · u_verg   (R eye converges leftward)

Smith predictor (same derivation as pursuit.py):
    u_verg_now  = x_verg + K_phasic_verg · e_pred
    e_pred      = e_disp − u_verg_now
    →  e_pred   = (e_disp − x_verg) / (1 + K_phasic_verg)

Dynamics:
    dx_verg/dt  = −x_verg / τ_verg  +  K_verg · e_pred
    u_verg      = x_verg + K_phasic_verg · e_pred

Steady-state vergence gain (fast-plant approximation):
    u_verg_ss / ψ_demand ≈ K_verg · τ_verg / (K_verg · τ_verg + 1 + K_phasic_verg)
    With K_verg=4, K_phasic=1, τ_verg=25:  gain ≈ 0.99

Closed-loop convergence TC:
    TC ≈ (1 + K_phasic) / (K_verg · (2 + K_phasic))
    With K_verg=4, K_phasic=1:  TC ≈ 2 / 12 ≈ 170 ms

EC correction (applied in brain_model.py before calling step):
    Without correction: e_disp = ψ_d − ψ (residual) → closed-loop gain ≈ 0.5
    With correction:    e_disp + x_verg ≈ ψ_d      → gain ≈ 0.99
    Analogous to motor_ec correcting vel_delayed for pursuit.
    x_verg is an acceptable approximation of the "vergence EC" because the plant
    time constant (0.15 s) is much shorter than the vergence settling time.

Gating:
    pos_delayed_L and pos_delayed_R are both gated by target_present upstream
    (sensory_model.step applies target_present before the cascade).
    When target_present = 0: e_disp = 0 → integrator holds last value.

State:
    x_verg = [H_verg, V_verg, torsional_verg]  (3,)  deg
    Positive H = converged (eyes rotated inward for near target).
    V and torsional driven by the same disparity mechanism; torsional ≈ 0
    in all standard paradigms.

Diopters (for T-VOR, derived externally):
    d_diopters ≈ radians(u_verg[0]) / ipd_m    (horizontal component)

Parameters:
  K_verg         — integration gain (1/s).
  K_phasic_verg  — direct feedthrough (dim'less); sets Smith attenuation.
  tau_verg       — leak TC (s); healthy long (>20 s) → stable vergence hold.
"""

import jax.numpy as jnp

N_STATES  = 3   # [H_verg, V_verg, torsional_verg]  vergence position (deg)
N_INPUTS  = 3   # e_disp = pos_delayed_L − pos_delayed_R (deg)
N_OUTPUTS = 3   # u_verg: vergence position command (deg)


def step(x_verg, e_disp, brain_params):
    """Single ODE step: Smith predictor → vergence derivative + position command.

    Args:
        x_verg:       (3,)   vergence position memory (deg); positive = converged
        e_disp:       (3,)   binocular disparity = pos_delayed_L − pos_delayed_R (deg)
                             Gated by target_present via the retinal cascade.
        brain_params: BrainParams  (reads K_verg, K_phasic_verg, tau_verg)

    Returns:
        dx_verg: (3,)  dx_verg/dt  (deg/s)
        u_verg:  (3,)  vergence position command (deg) → split ±½ in brain_model
    """
    K_ph = brain_params.K_phasic_verg
    # Smith predictor: e_pred = (e_disp − x_verg) / (1 + K_phasic)
    # Derivation: u_now = x + K_ph·e_pred;  e_pred = e_disp − u_now
    #             → e_pred·(1+K_ph) = e_disp − x  → solved below
    e_pred = (e_disp - x_verg) / (1.0 + K_ph)

    A       = -(1.0 / brain_params.tau_verg) * jnp.eye(3)
    dx_verg = A @ x_verg + brain_params.K_verg * e_pred
    u_verg  = x_verg + K_ph * e_pred
    return dx_verg, u_verg
