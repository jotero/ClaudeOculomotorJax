"""Gravity estimator SSM — canal-otolith interaction via cross-product dynamics.

Tracks the gravity vector in the head frame by rotating it with the angular
velocity estimate and correcting toward the otolith measurement.

Dynamics (Laurens & Angelaki 2011, 2017):

    dĝ/dt = −(π/180) · (ω̂ × ĝ)  +  K_grav · (f − ĝ)

where:
    ĝ       (3,)   gravity estimate in head frame (m/s²)
    ω̂       (3,)   angular velocity estimate from VS (deg/s, converted to rad/s)
    f       (3,)   specific force from otolith (m/s²)
    K_grav        otolith correction gain (1/s)

The cross-product term −(ω̂ × ĝ) is the transport equation: it rotates ĝ
consistent with head motion (canal-driven).  The K_grav term corrects ĝ toward
the raw otolith measurement f.  This implements tilt-translation disambiguation:
at low frequencies K_grav dominates (otolith = gravity + tilt); at high
frequencies the cross-product dominates (canal-driven rotation).

ĝ represents the SPECIFIC FORCE (what an accelerometer measures): it points UP
(opposing gravity) when the head is upright.  This matches the otolith afferent
convention: the otolith reads +G0 in the upward direction at rest.

Axis convention (matching canal.py):
    x = yaw / vertical axis      — specific force is +x when head is upright
    y = pitch / interaural axis
    z = roll / naso-occipital axis

Default initial condition: ĝ₀ = [+9.81, 0, 0] m/s² (upright head, +x = up).

Derived quantities:
    tilt_roll  = ĝ[1] / |ĝ|       (+ when right ear down) → eye CCW (−z)
    tilt_pitch = ĝ[2] / |ĝ|       (+ when nose up)        → eye pitch down (−y)
    a_trans    = f − ĝ             → residual linear accel (somatogravic / LVOR)

Parameters:
    K_grav — otolith correction gain (1/s).  Controls weighting of otolith vs
             canal.  Typical: 0.5 1/s (TC ≈ 2 s for gravity convergence).
"""

import jax.numpy as jnp

# ── Standard gravity ────────────────────────────────────────────────────────────

G0 = 9.81   # standard gravity (m/s²)

# Default initial specific force — upright head, specific force ∥ +x (opposing gravity)
X0 = jnp.array([G0, 0.0, 0.0])

# ── State layout ────────────────────────────────────────────────────────────────

N_STATES  = 3   # [g_hat_x, g_hat_y, g_hat_z]
N_INPUTS  = 6   # [w_est (3,) | f_otolith (3,)]
N_OUTPUTS = 3   # g_hat


def step(x_grav, u, brain_params):
    """Single ODE step: gravity estimate derivative + current estimate output.

    Args:
        x_grav:      (3,)  gravity estimate in head frame (m/s²)
        u:           (6,)  [w_est (3, deg/s) | f_otolith (3, m/s²)]
        brain_params: BrainParams  (reads K_grav)

    Returns:
        dx_grav: (3,)  dĝ/dt  (m/s³)
        g_hat:   (3,)  current gravity estimate ĝ (= x_grav, passed through)
    """
    w_est    = u[:3]     # angular velocity estimate (deg/s)
    f_otolith = u[3:]    # otolith specific force (m/s²)
    g_hat    = x_grav    # gravity estimate (m/s²)

    # Convert angular velocity to rad/s for cross product (ĝ is in m/s²)
    w_rad = w_est * (jnp.pi / 180.0)

    # Transport: rotate gravity estimate with head motion (canal-driven)
    transport = -jnp.cross(w_rad, g_hat)

    # Correction: pull toward otolith measurement
    correction = brain_params.K_grav * (f_otolith - g_hat)

    dx_grav = transport + correction
    return dx_grav, g_hat
