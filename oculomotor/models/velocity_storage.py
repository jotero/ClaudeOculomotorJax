"""Velocity Storage SSM — brainstem internal model estimator (3-D).

Implements the Laurens & Angelaki (2011) internal model / Kalman-filter
framework. The VS state ω̂ is the brain's 3-D estimate of true head angular
velocity, updated by canal afferents and OKR drive.

    dx_vs/dt = A_vs(θ) @ x_vs + B_vs(θ) @ u
      y_vs   = C_vs @ x_vs  +  D_vs @ u       (D feedthrough)

States:  x_vs = [ω̂_x, ω̂_y, ω̂_z]             (3,)   angular velocity estimate
Input:   u    = [y_canals (6,) | u_okr (3,)]  (9,)   stacked afferents + OKR
Output:  y_vs = ω̂  →  feeds Neural SSM

──────────────────────────────────────────────────────────────────────────
Derivation (Kalman predictor-corrector in continuous time):
    dω̂/dt = F · ω̂  +  K_vs · (u_canal − H · ω̂)  −  K_vs · u_okr
           = (F − K_vs · H) · ω̂  +  K_vs · u_canal  −  K_vs · u_okr

where u_canal = PINV_SENS @ y_canals  (3,)  is internal — VS owns the mixing.

In 3-D (independent axes, diagonal matrices):
    F   = −(1/τ_vs) · I₃
    H   = I₃
    ⟹  A_vs = −(1/τ_vs + K_vs) · I₃             (3×3)
        B_vs = K_vs · [PINV_SENS | −I₃]           (3×9)
        D_vs =        [PINV_SENS | −I₃]           (3×9)

Effective VS time constant (per axis):
    τ_eff = 1 / (1/τ_vs + K_vs)

Typical healthy: τ_vs = 50 s, K_vs = 0.03 /s  →  τ_eff ≈ 20 s

──────────────────────────────────────────────────────────────────────────
Canal feedthrough (D includes PINV_SENS):
  • y_vs = x_vs + D @ u — fast canal signal passes to NI even when x_vs≈0
    (Robinson 1977 / Raphan-Cohen 1979 architecture).
  • At HIT frequencies x_vs ≈ 0 so canal drives NI directly.
  • At low frequencies x_vs dominates (velocity storage extension).

3-D extension notes:
    F    → off-diagonal gravity coupling terms (axis-dependent VS)
    K_vs → 3×3 gain matrix
    These off-diagonal terms are zero for Level 1b; add later.

Parameters:
  τ_vs  — prior time constant (s).   Typical: 50 s.
  K_vs  — Kalman gain (1/s).         Typical: 0.03 /s.
"""

import jax.numpy as jnp
from oculomotor.models.canal import PINV_SENS, N_CANALS

N_STATES  = 3
N_INPUTS  = N_CANALS + 3    # 6 canal afferents + 3 OKR drive axes
N_OUTPUTS = 3

# D structure: [PINV_SENS (3×6) | −I₃ (3×3)] — maps stacked input to vel estimate
# Constant (geometry only, no theta dependence).
_D_STRUCT = jnp.concatenate([PINV_SENS, -jnp.eye(3)], axis=1)   # (3, 9)

C = jnp.eye(3)     # (3, 3) state output


def get_A(theta):
    """(3, 3) state matrix — prior decay + Kalman correction, diagonal."""
    a = -(1.0 / theta['tau_vs'] + theta['K_vs'])
    return a * jnp.eye(3)


def get_B(theta):
    """(3, 9) input matrix — K_vs · [PINV_SENS | −I₃].

    Maps stacked input u = [y_canals | u_okr] to state derivative.
    """
    return theta['K_vs'] * _D_STRUCT


def get_D():
    """(3, 9) direct feedthrough — [PINV_SENS | −I₃].

    Maps stacked input u = [y_canals | u_okr] to NI velocity command.
    """
    return _D_STRUCT


def step(x_vs, u, theta):
    """Single ODE step: state derivative + velocity command output.

    Args:
        x_vs:  (3,)  VS state (stored angular velocity estimate)
        u:     (9,)  stacked input [y_canals (6,) | u_okr (3,)]
        theta: dict  model parameters

    Returns:
        dx:   (3,)  dx_vs/dt
        u_ni: (3,)  velocity command to neural integrator
    """
    dx   = get_A(theta) @ x_vs + get_B(theta) @ u
    u_ni = C @ x_vs + _D_STRUCT @ u
    return dx, u_ni
