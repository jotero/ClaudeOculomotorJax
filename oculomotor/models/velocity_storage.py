"""Velocity Storage SSM — brainstem internal model estimator.

Implements the Laurens & Angelaki (2011) internal model / Kalman-filter
framework. The VS state ω̂ is the brain's estimate of true head angular
velocity, updated by a canal prediction error.

    dx_vs/dt = A_vs(θ) @ x_vs + B_vs(θ) @ u_vs
      y_vs   = C_vs @ x_vs + D_vs @ u_vs            (D = 1: direct canal feedthrough)

States:  x_vs = [ω̂]       (1,)   estimated head angular velocity, deg/s
Input:   u_vs = [canal]    (1,)   canal afferent from Canal SSM
Output:  y_vs = ω̂ → feeds Neural SSM

──────────────────────────────────────────────────────────────────────────
Derivation (Kalman predictor-corrector in continuous time):
    dω̂/dt = F · ω̂  +  K_vs · (canal − H · ω̂)
           = (F − K_vs · H) · ω̂  +  K_vs · canal

In 1-D:
    F   = −1/τ_vs          prior dynamics  (head velocity decays slowly)
    H   = 1                canal ≈ measures ω̂ at high frequencies
    ⟹  A_vs = −(1/τ_vs + K_vs)
        B_vs = K_vs

Effective VS time constant:
    τ_eff = 1 / (1/τ_vs + K_vs)

Typical healthy: τ_vs = 50 s, K_vs = 0.03 /s  →  τ_eff ≈ 20 s
  (extends the canal time constant τ_c ≈ 5 s to observed τ_VOR ≈ 20 s)

──────────────────────────────────────────────────────────────────────────
Canal feedthrough (D = 1):
  • y_vs = x_vs + u_vs — the estimate passed to NI includes both the
    integrated VS state (slow velocity storage) AND the raw canal signal
    (fast direct path).  This is essential for HIT responses:
    at HIT frequencies (~10 Hz) x_vs ≈ 0 so the canal signal passes
    through directly; at low frequencies x_vs dominates (velocity storage).
  • This matches the Robinson (1977) / Raphan-Cohen (1979) architecture
    where NI receives canal + VS storage signal simultaneously.
  • Lower K_vs → slower VS state update → longer apparent VOR time constant.
  • The Kalman-gain structure is what makes this architecture directly
    extensible to 3-D (Laurens 2011):

    3-D extension:
      F    → diagonal (axis-specific prior time constants)
               + off-diagonal gravity coupling terms  ← key for
               axis-dependent VS (horizontal > vertical/torsional)
      K_vs → 3×3 gain matrix
      H    → 3×3 canal geometry matrix (canal planes → head-frame axes)

Parameters:
  τ_vs  — prior time constant (s).  Typical: 50 s.  Large = weak prior.
  K_vs  — Kalman gain (1/s).        Typical: 0.03 /s.  Small = slow update.
"""

import jax.numpy as jnp

N_STATES = 1
N_INPUTS = 1
N_OUTPUTS = 1


def get_A(theta):
    """(1, 1) state matrix — prior decay + Kalman correction."""
    return jnp.array([[-(1.0 / theta['tau_vs'] + theta['K_vs'])]])


def get_B(theta):
    """(1, 1) input matrix — Kalman gain on canal prediction error."""
    return jnp.array([[theta['K_vs']]])


C = jnp.array([[1.0]])   # (1, 1) state output
D = jnp.array([[1.0]])   # (1, 1) direct canal feedthrough (D = 1)
