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
(opposing gravity) when the head is upright.  At rest upright: ĝ = [G0, 0, 0].

Axis convention (matching canal.py):
    x = yaw / vertical axis      — specific force is +x when head is upright
    y = pitch / interaural axis
    z = roll / naso-occipital axis

Default initial condition: ĝ₀ = [+9.81, 0, 0] m/s²; â₀ = [0, 0, 0] m/s².

Derived quantities:
    tilt_roll  = ĝ[1] / |ĝ|       (+ when right ear down) → eye CCW (−z)
    tilt_pitch = ĝ[2] / |ĝ|       (+ when nose up)        → eye pitch down (−y)
    a_trans    = f − ĝ − â        → true residual linear accel (LVOR drive)

Parameters:
    K_grav — gravity correction gain (1/s).  Typical: 0.6 1/s (TC ≈ 1.7 s).
    K_lin  — linear acceleration adaptation gain (1/s).  Typical: 0.1–0.2 1/s.
             Set to 0 to disable linear acceleration estimation (somatogravic effect
             fully present, as in the original single-state formulation).
"""

import jax.numpy as jnp

# ── Standard gravity ────────────────────────────────────────────────────────────

G0 = 9.81   # standard gravity (m/s²)

# Default initial state: ĝ = upright specific force; â = 0 (no linear acceleration)
X0 = jnp.array([G0, 0.0, 0.0, 0.0, 0.0, 0.0])

# ── State layout ────────────────────────────────────────────────────────────────

N_STATES  = 6   # [g_est (3,) | a_lin (3,)]
N_INPUTS  = 6   # [w_est (3,) | f_otolith (3,)]
N_OUTPUTS = 3   # g_est  (first 3 states)

# Slice indices within x_grav
_IDX_G = slice(0, 3)
_IDX_A = slice(3, 6)


def step(x_grav, u, brain_params):
    """Single ODE step: gravity + linear acceleration derivative, gravity output.

    Args:
        x_grav:      (6,)  [g_est (3,) | a_lin (3,)]
                           g_est: gravity estimate, head frame [x=up, y=left, z=fwd] (m/s²)
                                  upright rest: [+9.81, 0, 0]
                           a_lin: linear acceleration estimate, same head frame (m/s²)
                                  zero at rest
        u:           (6,)  [w_est (3, deg/s) | f_otolith (3, m/s²)]
                           w_est:     angular velocity from canal (deg/s), head frame
                                      [x=rotation-around-up, y=rotation-around-interaural,
                                       z=rotation-around-naso-occipital]
                           f_otolith: specific force (GIA) from otolith, head frame (m/s²)
                                      [x=up/yaw, y=left/interaural, z=fwd/naso-occipital]
                                      supplied as [f_gia[1], −f_gia[0], f_gia[2]] to convert
                                      from world frame (x=right, y=up) → head frame (x=up, y=left)
        brain_params: BrainParams  (reads K_grav, K_lin)

    Returns:
        dx_grav: (6,)  d[ĝ; â]/dt  — same head frame
        g_est:   (3,)  current gravity estimate ĝ (= x_grav[:3], passed through)
    """
    g_est     = x_grav[_IDX_G]  # gravity estimate,              head frame (m/s²)
    a_lin     = x_grav[_IDX_A]  # linear acceleration estimate,  head frame (m/s²)
    w_est     = u[:3]            # angular velocity, head frame [x=up, y=left, z=fwd] (deg/s)
    f_otolith = u[3:]            # otolith GIA,      head frame [x=up, y=left, z=fwd] (m/s²)

    # Unexplained residual: GIA minus both estimates
    residual = f_otolith - a_lin - g_est

    # Transport: rotate gravity estimate with head motion (canal-driven)
    w_rad     = w_est * (jnp.pi / 180.0)
    transport = -jnp.cross(w_rad, g_est)

    # Gravity correction: pull toward residual (DC component of GIA)
    dg = transport + brain_params.K_grav * residual

    # Linear acceleration adaptation: absorb slowly-varying non-gravity GIA
    da = brain_params.K_lin * residual

    return jnp.concatenate([dg, da]), g_est
