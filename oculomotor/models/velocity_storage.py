"""Velocity Storage SSM — brainstem internal model estimator (3-D).

Implements the Laurens & Angelaki (2011) internal model / Kalman-filter
framework. The VS state ω̂ is the brain's 3-D estimate of true head angular
velocity, updated by a canal prediction error.

    dx_vs/dt = A_vs(θ) @ x_vs + B_vs(θ) @ u_vs
      y_vs   = C_vs @ x_vs + D_vs @ u_vs    (D = I: direct canal feedthrough)

States:  x_vs = [ω̂_x, ω̂_y, ω̂_z]   (3,)   estimated head angular velocity, deg/s
Input:   u_vs = [canal]              (3,)   canal velocity estimate from Canal SSM
Output:  y_vs = ω̂ → feeds Neural SSM

──────────────────────────────────────────────────────────────────────────
Derivation (Kalman predictor-corrector in continuous time):
    dω̂/dt = F · ω̂  +  K_vs · (canal − H · ω̂)
           = (F − K_vs · H) · ω̂  +  K_vs · canal

In 3-D (independent axes, diagonal matrices):
    F   = −(1/τ_vs) · I₃       prior dynamics
    H   = I₃                   canal ≈ measures ω̂ directly
    ⟹  A_vs = −(1/τ_vs + K_vs) · I₃
        B_vs = K_vs · I₃

Effective VS time constant (per axis):
    τ_eff = 1 / (1/τ_vs + K_vs)

Typical healthy: τ_vs = 50 s, K_vs = 0.03 /s  →  τ_eff ≈ 20 s

──────────────────────────────────────────────────────────────────────────
Canal feedthrough (D = I):
  • y_vs = x_vs + u_vs — fast canal signal passes to NI even when x_vs≈0
    (Robinson 1977 / Raphan-Cohen 1979 architecture).
  • At HIT frequencies x_vs ≈ 0 so canal drives NI directly.
  • At low frequencies x_vs dominates (velocity storage extension).

3-D extension notes:
    F    → diagonal (axis-specific prior time constants) + off-diagonal gravity
            coupling terms (key for axis-dependent VS: horizontal > vertical)
    K_vs → 3×3 gain matrix
    H    → 3×3 canal geometry matrix (canal planes → head-frame axes)
    These off-diagonal terms are set to zero for Level 1b; add later.

Parameters:
  τ_vs  — prior time constant (s).   Typical: 50 s.
  K_vs  — Kalman gain (1/s).         Typical: 0.03 /s.
"""

import jax.numpy as jnp

N_STATES  = 3
N_INPUTS  = 3
N_OUTPUTS = 3


def get_A(theta):
    """(3, 3) state matrix — prior decay + Kalman correction, diagonal."""
    a = -(1.0 / theta['tau_vs'] + theta['K_vs'])
    return a * jnp.eye(3)


def get_B(theta):
    """(3, 3) input matrix — Kalman gain on canal prediction error."""
    return theta['K_vs'] * jnp.eye(3)


C = jnp.eye(3)   # (3, 3) state output
D = jnp.eye(3)   # (3, 3) direct canal feedthrough (D = I)
