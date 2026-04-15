"""Velocity Storage SSM — Raphan/Cohen (1979) leaky-integrator architecture.

Implements the Raphan, Matsuo & Cohen (1979 Exp Brain Res) velocity-storage
model: a leaky integrator charged by semicircular canal afferents (and
retinal slip), with storage TC = τ_vs directly settable to match post-
rotatory nystagmus / OKAN data (~20 s in monkey: Cohen et al. 1977).

    dx_vs/dt = A_vs @ x_vs + B_vs(θ) @ u
      y_vs   = C_vs @ x_vs  +  D_vs(θ) @ u       (D feedthrough)

States:  x_vs = [ω̂_x, ω̂_y, ω̂_z]                    (3,)   angular velocity estimate
Input:   u    = [y_canals (6,) | e_slip_delayed (3,)]  (9,)   stacked afferents + visual
Output:  y_vs = ω̂  →  feeds Neural SSM

──────────────────────────────────────────────────────────────────────────
Architecture (Raphan et al. 1979):

    VS is a leaky integrator with storage TC τ_vs, driven by canal afferents
    (K_vs gain) and retinal slip (K_vis gain).  The two parameters are
    INDEPENDENT: τ_vs sets the storage/OKAN decay; K_vs sets how aggressively
    canal signals charge the store.

    Canal pathway:
        dω̂/dt += K_vs · u_canal       [canal charges VS; K_vs = coupling gain]

    Visual pathway (OKR / OKAN):
        dω̂/dt -= K_vis · e_slip_delayed

    Leak:
        dω̂/dt -= ω̂ / τ_vs            [τ_vs = storage/OKAN time constant directly]

    Combined:
        dω̂/dt = −(1/τ_vs)·ω̂ + K_vs·u_canal − K_vis·e_slip_delayed

where u_canal = PINV_SENS @ y_canals  (3,)  is internal — VS owns the mixing.

In 3-D (independent axes, diagonal matrices):
    A_vs = −(1/τ_vs) · I₃                     (3×3)  θ-dependent via τ_vs only
    B_vs = [K_vs·PINV_SENS | −K_vis·I₃]       (3×9)  θ-dependent
    D_vs = [PINV_SENS      | −g_vis·I₃]        (3×9)  θ-dependent

Storage/OKAN time constant = τ_vs (directly; no compound formula).
Typical: τ_vs = 20 s  (Cohen, Matsuo & Raphan 1977 J Neurophysiol;
                        Raphan, Matsuo & Cohen 1979 Exp Brain Res).

──────────────────────────────────────────────────────────────────────────
Why K_vs and τ_vs are independent:

  In the Laurens & Angelaki (2011) Kalman formulation, A = −(1/τ_vs + K_vs),
  so K_vs appears in BOTH the decay and the drive, creating a constraint:
      τ_eff = 1/(1/τ_vs + K_vs)
  To get τ_eff = 20 s one needs tiny K_vs ≈ 0.03 /s, which charges VS too
  slowly — the canal adapts (τ_c = 5 s) before VS can accumulate significant
  velocity.

  The Raphan model avoids this: A = −1/τ_vs, so τ_vs IS the storage TC
  directly, and K_vs is free to be large (e.g. 0.1 /s) for rapid charging.
  With K_vs = 0.1 and τ_vs = 20 s, VS charges to ~19 deg/s during a 15-s
  constant-velocity rotation vs ~6 deg/s with the Kalman formulation.

──────────────────────────────────────────────────────────────────────────
Visual pathway (OKR / OKAN):

  Sign convention: e_slip_delayed > 0 when scene moves faster than eye.
  This charges x_vs NEGATIVELY (−K_vis term in B), so
  w_est = x_vs − g_vis·e_slip_delayed becomes more negative, and
  u_vel = −g_vor · w_est increases in the scene direction → eye follows. ✓

  OKR steady-state gain (τ_vs = 20 s, g_vor = 1):
      w_eye/w_scene ≈ (τ_vs·K_vis + g_vis) / (1 + τ_vs·K_vis + g_vis)

  OKAN time constant = τ_vs, independent of K_vis.
  When scene turns off, x_vs (negative) decays with τ_vs → OKAN. ✓

──────────────────────────────────────────────────────────────────────────
Canal feedthrough (D includes PINV_SENS):
  • y_vs = x_vs + D @ u — fast canal signal passes to NI even when x_vs≈0
    (Robinson 1977 / Raphan-Cohen 1979 architecture).
  • At HIT frequencies x_vs ≈ 0 so canal drives NI directly.
  • At low frequencies x_vs dominates (velocity storage extension).

3-D extension notes:
    A_vs → off-diagonal gravity coupling terms (axis-dependent VS)
    K_vs → 3×3 gain matrix
    These off-diagonal terms are zero for Level 1b; add later.

Parameters:
  τ_vs  — storage time constant (s).  Typical: 20 s  (Raphan et al. 1979;
           Cohen et al. 1977).  This IS the OKAN TC — set directly from data.
  K_vs  — canal coupling gain (1/s).  Typical: 0.1 /s.  Controls how fast
           canal afferents charge the VS store.  Independent of τ_vs.
           Larger K_vs → VS charges faster but SS level = K_vs·τ_vs·u_canal
           (e.g. K_vs=0.05 → x_vs_ss = u_canal at constant canal drive).
  K_vis — visual state gain (1/s).    Default: 0.3 /s; charges VS from slip.
           Does NOT affect OKAN time constant (only τ_vs controls OKAN TC).
  g_vis — visual direct feedthrough.  Default: 0.3; provides fast OKR onset.
           Combined with K_vis sets steady-state OKR gain ≈ 86% at defaults.
"""

import jax.numpy as jnp
from oculomotor.models.sensory_models.sensory_model import PINV_SENS, N_CANALS

N_STATES  = 3
N_INPUTS  = N_CANALS + 3 + 3   # 6 canal afferents + 3 slip + 3 g_hat
N_OUTPUTS = 3

def step(x_vs, u, brain_params):
    """Single ODE step: state derivative + velocity command output.

    Args:
        x_vs:  (3,)    VS state (stored angular velocity estimate)
        u:     (12,)   stacked input [y_canals (6,) | e_slip_delayed (3,) | g_hat (3,)]
        brain_params: BrainParams  model parameters

    Returns:
        dx:    (3,)  dx_vs/dt
        w_est: (3,)  angular velocity estimate (deg/s)
    """
    canal_in   = u[:N_CANALS]       # (6,)  canal afferent rates
    slip_in    = u[N_CANALS:N_CANALS+3]  # (3,)  corrected retinal slip
    g_hat      = u[N_CANALS+3:]     # (3,)  gravity estimate (m/s²)

    u_cs = jnp.concatenate([canal_in, slip_in])   # (9,) — original linear inputs

    # ── Linear system matrices ─────────────────────────────────────────────────
    A = -(1.0 / brain_params.tau_vs) * jnp.eye(3)
    B = jnp.concatenate([brain_params.K_vs  * PINV_SENS, -brain_params.K_vis * jnp.eye(3)], axis=1)
    D = jnp.concatenate([PINV_SENS,                     -brain_params.g_vis  * jnp.eye(3)], axis=1)
    # C = I (identity — omitted)

    # ── Gravity dumping: preferential decay of components ⊥ to gravity ────────
    # cross(ĝ, cross(ĝ, x_vs)) / |ĝ|² = x_vs_parallel − x_vs = −x_vs_perp
    # Adds −K_gd · x_vs_perp to the dynamics → horizontal storage decays faster.
    g_norm_sq  = jnp.dot(g_hat, g_hat) + 1e-9
    gd_term    = brain_params.K_gd * jnp.cross(g_hat, jnp.cross(g_hat, x_vs)) / g_norm_sq

    # ── Dynamics ──────────────────────────────────────────────────────────────
    dx    = A @ x_vs + B @ u_cs + gd_term
    w_est = x_vs     + D @ u_cs
    return dx, w_est
