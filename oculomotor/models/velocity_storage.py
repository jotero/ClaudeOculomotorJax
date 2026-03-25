"""Velocity Storage SSM — brainstem internal model estimator (3-D).

Implements the Laurens & Angelaki (2011 Exp Brain Res) internal model /
Kalman-filter framework, on top of the Robinson (1977) / Raphan, Matsuo &
Cohen (1979 Exp Brain Res) velocity-storage architecture.
The VS state ω̂ is the brain's 3-D estimate of true head angular velocity,
updated by canal afferents and delayed retinal slip.

    dx_vs/dt = A_vs(θ) @ x_vs + B_vs(θ) @ u
      y_vs   = C_vs @ x_vs  +  D_vs(θ) @ u       (D feedthrough)

States:  x_vs = [ω̂_x, ω̂_y, ω̂_z]                    (3,)   angular velocity estimate
Input:   u    = [y_canals (6,) | e_slip_delayed (3,)]  (9,)   stacked afferents + visual
Output:  y_vs = ω̂  →  feeds Neural SSM

──────────────────────────────────────────────────────────────────────────
Derivation (Kalman predictor-corrector in continuous time):

    Canal pathway (fast head-motion sensing):
        dω̂/dt += K_vs · (u_canal − ω̂)   [Kalman correction on canal estimate]

    Visual pathway (OKR drive — retinal slip charges VS negatively):
        dω̂/dt -= K_vis · e_slip_delayed  [direct drive from delayed retinal slip]

    Combined:
        dω̂/dt = F·ω̂ + K_vs·(u_canal − ω̂) − K_vis·e_slip_delayed
               = (F − K_vs)·ω̂ + K_vs·u_canal − K_vis·e_slip_delayed

where u_canal = PINV_SENS @ y_canals  (3,)  is internal — VS owns the mixing.

In 3-D (independent axes, diagonal matrices):
    F   = −(1/τ_vs) · I₃
    ⟹  A_vs = −(1/τ_vs + K_vs) · I₃               (3×3)
        B_vs = [K_vs·PINV_SENS | −K_vis·I₃]        (3×9)    θ-dependent
        D_vs = [PINV_SENS      | −g_vis·I₃]         (3×9)    θ-dependent

Effective VS time constant (canal pathway):
    τ_eff = 1 / (1/τ_vs + K_vs)

Typical healthy: τ_vs = 50 s, K_vs = 0.03 /s  →  τ_eff ≈ 20 s

──────────────────────────────────────────────────────────────────────────
Visual pathway (OKR / OKAN):

  Sign convention: e_slip_delayed > 0 when scene moves faster than eye.
  This charges x_vs NEGATIVELY (−K_vis term in B), so
  w_est = x_vs − g_vis·e_slip_delayed becomes more negative, and
  u_vel = −g_vor · w_est increases in the scene direction → eye follows. ✓

  OKR steady-state gain (τ_eff = 20 s, g_vor = 1):
      w_eye/w_scene ≈ (20·K_vis + g_vis) / (1 + 20·K_vis + g_vis)

  OKAN time constant = τ_eff = 1/(1/τ_vs + K_vs), independent of K_vis.
  When scene turns off, x_vs (negative) decays with τ_eff → OKAN. ✓

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
  τ_vs  — prior time constant (s).    Typical: 50 s  (Laurens & Angelaki 2011).
  K_vs  — canal Kalman gain (1/s).    Typical: 0.03 /s; yields τ_eff ≈ 20 s
           matching the measured VS time constant (Raphan et al. 1979;
           Cohen, Matsuo & Raphan 1977 J Neurophysiol).
  K_vis — visual state gain (1/s).    Default: 0.3 /s; charges VS from slip.
           Does NOT affect OKAN time constant (only τ_vs and K_vs do).
  g_vis — visual direct feedthrough.  Default: 0.3; provides fast OKR onset.
           Combined with K_vis sets steady-state OKR gain ≈ 86% at defaults.
"""

import jax.numpy as jnp
from oculomotor.models.canal import PINV_SENS, N_CANALS

N_STATES  = 3
N_INPUTS  = N_CANALS + 3    # 6 canal afferents + 3 delayed retinal slip axes
N_OUTPUTS = 3

C = jnp.eye(3)     # (3, 3) state output


def get_A(theta):
    """(3, 3) state matrix — prior decay + canal Kalman correction, diagonal."""
    a = -(1.0 / theta['tau_vs'] + theta['K_vs'])
    return a * jnp.eye(3)


def get_B(theta):
    """(3, 9) input matrix — [K_vs·PINV_SENS | −K_vis·I₃].

    Canal block drives VS state with Kalman gain K_vs.
    Visual block charges VS state negatively with gain K_vis.
    Maps stacked input u = [y_canals | e_slip_delayed] to state derivative.
    """
    k_vis = theta.get('K_vis', 0.3)
    return jnp.concatenate([theta['K_vs'] * PINV_SENS,
                             -k_vis * jnp.eye(3)], axis=1)


def get_D(theta):
    """(3, 9) direct feedthrough — [PINV_SENS | −g_vis·I₃].

    Canal block: passes head vel estimate to NI instantly (fast VOR).
    Visual block: instantaneous OKR contribution from delayed slip.
    """
    g_vis = theta.get('g_vis', 0.3)
    return jnp.concatenate([PINV_SENS, -g_vis * jnp.eye(3)], axis=1)


def step(x_vs, u, theta):
    """Single ODE step: state derivative + velocity command output.

    Args:
        x_vs:  (3,)  VS state (stored angular velocity estimate)
        u:     (9,)  stacked input [y_canals (6,) | e_slip_delayed (3,)]
        theta: dict  model parameters

    Returns:
        dx:    (3,)  dx_vs/dt
        w_est: (3,)  angular velocity estimate (deg/s)
    """
    dx    = get_A(theta) @ x_vs + get_B(theta) @ u
    w_est = C @ x_vs + get_D(theta) @ u
    return dx, w_est
