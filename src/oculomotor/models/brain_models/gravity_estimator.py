"""Gravity estimator SSM — canal-otolith interaction via cross-product dynamics.

Tracks the gravity vector and linear acceleration in the head frame.

State:  x_grav = [ĝ (3,) | â (3,)]                                           (6,)
    ĝ  gravity estimate (m/s²)       — points UP (specific force convention)
    â  linear acceleration estimate (m/s²) — residual after subtracting ĝ from GIA

Dynamics:

    dĝ/dt = −(π/180) · (ω̂ × ĝ)  +  (1/τ_grav) · (f − ĝ)
    dâ/dt =                           K_lin      · (f − ĝ)

where:
    ĝ        (3,)   gravity estimate in head frame (m/s²)
    â        (3,)   linear acceleration estimate in head frame (m/s²)
    ω̂        (3,)   angular velocity estimate (deg/s, converted to rad/s)
    f        (3,)   specific force (GIA) from otolith (m/s²)
    τ_grav         gravity estimate time constant (s); somatogravic bandwidth = 1/(2π·τ_grav)
    K_lin          linear acceleration tracking gain (1/s)

ĝ tracks the full GIA with TC = τ_grav (DC gain = 1.0 → full somatogravic effect).
â accumulates (f − ĝ) and is used to compute a_trans = f − ĝ − â for LVOR drive.

At rest (no acceleration): â → 0, ĝ → f, correct gravity estimate.
Sustained lateral acceleration: ĝ fully tilts toward GIA (somatogravic "false vertical").
High-frequency linear acceleration: ĝ stays anchored by canal transport → correctly
interpreted as translation (no perceived tilt).

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
    a_trans    = gia - g_est - a_lin    (LVOR drive; zero at rest)

Parameters (in BrainParams):
    tau_grav (s)  gravity estimate TC. Default 5 s (somatogravic bandwidth ~0.032 Hz).
    K_lin  (1/s)  linear accel tracking gain. Typical: 0.1.
"""

import jax.numpy as jnp

from oculomotor.models.sensory_models.retina import ypr_to_xyz

# ── Standard gravity ────────────────────────────────────────────────────────────

G0 = 9.81   # standard gravity (m/s²)

# Default initial state: ĝ = upright specific force; â = 0 (no linear acceleration)
X0 = jnp.array([0.0, G0, 0.0, 0.0, 0.0, 0.0])

# ── State layout ────────────────────────────────────────────────────────────────

N_STATES  = 6   # [g_est (3,) | a_lin (3,)]
N_INPUTS  = 6   # [gia (3,) | w_est (3,)]
N_OUTPUTS = 6   # [g_est (3,) | a_est (3,)]

# Slice indices within x_grav
_IDX_G = slice(0, 3)
_IDX_A = slice(3, 6)   # a_lin state — updated but not read in step(); reserved for LVOR


def step(x_grav, u, brain_params):
    """Single ODE step: gravity + linear acceleration derivative, gravity output.

    Args:
        x_grav:      (6,)  [g_est (3,) | a_lin (3,)]
                           g_est: gravity estimate, world frame [x=right, y=up, z=fwd] (m/s²)
                                  upright rest: [0, +9.81, 0]
                           a_lin: linear acceleration state, same world frame (m/s²)
                                  zero at rest; reserved for LVOR
        u:           (6,)  [gia (3, m/s²) | w_est (3, deg/s)]
                           gia:   gravitoinertial acceleration from otolith, world frame (m/s²)
                                  [x=right, y=up, z=fwd]; upright rest: [0, +9.81, 0]
                           w_est: angular velocity estimate from VS, world [yaw, pitch, roll] (deg/s)
        brain_params: BrainParams  (reads tau_grav, K_lin)

    Returns:
        dx_grav: (6,)  d[ĝ; â]/dt  — world frame
        g_est:   (3,)  current gravity estimate ĝ (= x_grav[:3], passed through)
        a_est:   (3,)  estimated linear acceleration = gia − g_est (instantaneous)
    """
    g_est = x_grav[_IDX_G]  # gravity estimate,  world frame (m/s²)
    gia   = u[:3]            # otolith GIA,       world frame [x=right, y=up, z=fwd] (m/s²)
    w_est = u[3:]            # angular velocity from VS, world [yaw, pitch, roll] (deg/s)

    # Estimated linear acceleration: GIA minus gravity estimate.
    # g_est tracks full GIA with TC = tau_grav (DC gain = 1, full somatogravic effect).
    # a_lin accumulates a_est for LVOR: a_trans = gia − g_est − a_lin (zero at rest).
    a_est = gia - g_est

    # Transport: rotate gravity estimate with VS angular velocity estimate
    # ypr_to_xyz converts [yaw,pitch,roll] → xyz rotation-axis vector for cross product
    w_rad_xyz = jnp.radians(ypr_to_xyz(w_est))
    transport = -jnp.cross(w_rad_xyz, g_est)

    # Gravity correction: pull toward GIA with TC = tau_grav
    dg = transport + (1.0 / brain_params.tau_grav) * a_est

    # Linear acceleration: tracks a_est for LVOR / a_trans
    da = brain_params.K_lin * a_est

    return jnp.concatenate([dg, da]), g_est, a_est
