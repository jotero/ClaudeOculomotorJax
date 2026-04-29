"""Gravity estimator SSM — canal-otolith interaction via cross-product dynamics.

Tracks the gravity vector and linear acceleration in the head frame.

State:  x_grav = [ĝ (3,) | â (3,)]                                           (6,)
    ĝ  gravity estimate (m/s²)       — points UP (specific force convention)
    â  linear acceleration estimate (m/s²) — residual after subtracting ĝ from GIA

Dynamics (Laurens & Angelaki 2011, 2017):

    dĝ/dt = −(π/180) · (ω̂ × ĝ)  +  K_grav · (f − â − ĝ)
    dâ/dt =                          K_lin  · (f − â − ĝ)

where:
    ĝ       (3,)   gravity estimate in head frame (m/s²)
    â       (3,)   linear acceleration estimate in head frame (m/s²)
    ω̂       (3,)   angular velocity estimate (deg/s, converted to rad/s)
    f       (3,)   specific force (GIA) from otolith (m/s²)
    K_grav        otolith correction gain for gravity (1/s)
    K_lin         linear acceleration adaptation gain (1/s); typically K_lin < K_grav

The residual (f − â − ĝ) is the "unexplained" GIA — what's left after subtracting
both the gravity and linear acceleration estimates.  K_grav drives ĝ toward this
residual (gravity tracks the DC component); K_lin drives â toward the same residual
(linear acceleration tracks the slowly-varying component).

At rest (no acceleration): â → 0, ĝ → f, correct gravity estimate.
Sustained lateral acceleration: â adapts slowly, partially absorbing the acceleration,
so ĝ doesn't fully tilt toward the somatogravic "false vertical" — the degree of
somatogravic effect depends on K_lin / K_grav ratio.
High-frequency linear acceleration: â cannot follow → ĝ stays anchored by canal
transport → correctly interpreted as translation (no perceived tilt).

ĝ represents the SPECIFIC FORCE (what an accelerometer measures): it points UP
(opposing gravity) when the head is upright.

Axis convention — world frame (LEFT-HANDED: x=right, y=up, z=forward):
    All vectors in [x, y, z] = [right, up, fwd] order.
    Angular velocity input in [yaw, pitch, roll] order — converted via ypr_to_xyz()
    before cross-product transport (yaw→+y, pitch→−x, roll→+z).
    At rest upright: ĝ = [0, +9.81, 0] m/s²  (specific force = +y = up).

Default initial condition: ĝ₀ = [0, +9.81, 0] m/s²; â₀ = [0, 0, 0] m/s².

Derived quantities:
    tilt_roll  = -g_est[0] / |g_est|   (+ when right ear down; drives OCR via brain_params.g_ocr)
    tilt_pitch =  g_est[2] / |g_est|   (+ when nose up; future: Listing's plane tilt)
    a_trans    = f - g_est - a_lin      (LVOR drive; zero at rest)

Parameters (in BrainParams):
    K_grav (1/s)  gravity correction gain. Typical: 0.6 (TC ~1.7 s).
    K_lin  (1/s)  linear accel adaptation gain. Typical: 0.1-0.2; 0 = full somatogravic effect.
"""

import jax.numpy as jnp

from oculomotor.models.sensory_models.retina import ypr_to_xyz

# ── Standard gravity ────────────────────────────────────────────────────────────

G0 = 9.81   # standard gravity (m/s²)

# Default initial state: ĝ = upright specific force; â = 0 (no linear acceleration)
X0 = jnp.array([0.0, G0, 0.0, 0.0, 0.0, 0.0])

# ── State layout ────────────────────────────────────────────────────────────────

N_STATES  = 6   # [g_est (3,) | a_lin (3,)]
N_INPUTS  = 6   # [f_otolith (3,) | w_est (3,)]
N_OUTPUTS = 6   # [g_est (3,) | a_lin (3,)]

# Slice indices within x_grav
_IDX_G = slice(0, 3)
_IDX_A = slice(3, 6)


def step(x_grav, u, brain_params):
    """Single ODE step: gravity + linear acceleration derivative, gravity output.

    Args:
        x_grav:      (6,)  [g_est (3,) | a_lin (3,)]
                           g_est: gravity estimate, world frame [x=right, y=up, z=fwd] (m/s²)
                                  upright rest: [0, +9.81, 0]
                           a_lin: linear acceleration estimate, same world frame (m/s²)
                                  zero at rest
        u:           (6,)  [f_otolith (3, m/s²) | w_est (3, deg/s)]
                           f_otolith: specific force (GIA) from otolith, world frame (m/s²)
                                      [x=right, y=up, z=fwd]; upright rest: [0, +9.81, 0]
                           w_est:     angular velocity estimate from VS, world [yaw, pitch, roll] (deg/s)
        brain_params: BrainParams  (reads K_grav, K_lin)

    Returns:
        dx_grav: (6,)  d[ĝ; â]/dt  — world frame
        g_est:   (3,)  current gravity estimate ĝ (= x_grav[:3], passed through)
        a_lin:   (3,)  current linear acceleration estimate (= x_grav[3:], passed through)
    """
    g_est     = x_grav[_IDX_G]  # gravity estimate,              world frame (m/s²)
    a_lin     = x_grav[_IDX_A]  # linear acceleration estimate,  world frame (m/s²)
    f_otolith = u[:3]            # otolith GIA,      world frame [x=right, y=up, z=fwd] (m/s²)
    w_est     = u[3:]            # angular velocity from VS, world [yaw, pitch, roll] (deg/s)

    # Unexplained residual: GIA minus both estimates
    residual = f_otolith - a_lin - g_est

    # Transport: rotate gravity estimate with VS angular velocity estimate
    # ypr_to_xyz converts [yaw,pitch,roll] → xyz rotation-axis vector for cross product
    w_rad_xyz = jnp.radians(ypr_to_xyz(w_est))
    transport = -jnp.cross(w_rad_xyz, g_est)

    # Gravity correction: pull toward residual (DC component of GIA)
    dg = transport + brain_params.K_grav * residual

    # Linear acceleration adaptation: absorb slowly-varying non-gravity GIA
    da = brain_params.K_lin * residual

    return jnp.concatenate([dg, da]), g_est, a_lin
