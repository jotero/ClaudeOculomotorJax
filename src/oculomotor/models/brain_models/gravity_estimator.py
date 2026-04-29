"""Gravity estimator SSM — canal-otolith interaction via cross-product dynamics.

Tracks the gravity vector, linear acceleration, and rotational feedback in the head frame.

State:  x_grav = [ĝ (3,) | â (3,) | rf (3,)]                                 (9,)
    ĝ   gravity estimate (m/s²)       — points UP (specific force convention)
    â   linear acceleration estimate (m/s²) — residual after subtracting ĝ from GIA
    rf  rotational feedback (deg/s) — Laurens & Angelaki (2011); read by VS as input

Dynamics:

    dĝ/dt  = −(π/180) · (ω̂ × ĝ)  +  (1/τ_grav) · (f − ĝ)
    dâ/dt  =                           K_lin      · (f − ĝ)
    drf/dt =  (rf_new − rf) / τ_rf_state

where:
    ĝ        (3,)   gravity estimate in head frame (m/s²)
    â        (3,)   linear acceleration estimate in head frame (m/s²)
    rf       (3,)   rotational feedback state (deg/s) — fast-lag of rf_new for VS
    ω̂        (3,)   VS net angular velocity (deg/s, converted to rad/s)
    f        (3,)   specific force (GIA) from otolith (m/s²)
    τ_grav         gravity estimate time constant (s); somatogravic bandwidth = 1/(2π·τ_grav)
    K_lin          linear acceleration tracking gain (1/s)
    τ_rf_state     rf state tracking TC = 5 ms (breaks VS/GE algebraic loop with negligible lag)

rf state design:
    VS needs rf from GE; GE needs w_est from VS (for gravity transport).
    Storing rf in the GE state breaks this loop: VS reads rf from the ODE state
    (previous timestep, 1 ODE step delayed), then GE uses the current w_est from VS.
    τ_rf_state = 5 ms → lag is negligible vs gravity dynamics (τ_grav ~ seconds).

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

Default initial condition: ĝ₀ = [0, +9.81, 0] m/s²; â₀ = [0, 0, 0] m/s²; rf₀ = [0, 0, 0].

Derived quantities:
    tilt_roll  = -g_est[0] / |g_est|   (+ when right ear down; drives OCR via brain_params.g_ocr)
    tilt_pitch =  g_est[2] / |g_est|   (+ when nose up; future: Listing's plane tilt)
    a_trans    = gia - g_est - a_lin    (LVOR drive; zero at rest)

Parameters (in BrainParams):
    tau_grav (s)  gravity estimate TC. Default 5 s (somatogravic BW ~0.032 Hz).
    K_lin  (1/s)  linear accel tracking gain. Typical: 0.1.
    K_gd          rotational feedback gain for VS (dim'less). Default 0 (disabled).
"""

import jax.numpy as jnp

from oculomotor.models.sensory_models.retina import ypr_to_xyz, xyz_to_ypr

# ── Standard gravity ────────────────────────────────────────────────────────────

G0 = 9.81   # standard gravity (m/s²)

# Default initial state: ĝ = upright specific force; â = 0; rf = 0
X0 = jnp.array([0.0, G0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0])

# ── State layout ────────────────────────────────────────────────────────────────

N_STATES  = 9   # [g_est (3,) | a_lin (3,) | rf (3,)]
N_INPUTS  = 6   # [w_est (3,) | gia (3,)]
N_OUTPUTS = 3   # [g_est (3,)]

# Slice indices within x_grav
_IDX_G  = slice(0, 3)   # g_est
_IDX_A  = slice(3, 6)   # a_lin state — updated but not read in step(); reserved for LVOR
_IDX_RF = slice(6, 9)   # rotational feedback state — read by brain_model for VS input

# rf state tracking TC — breaks VS/GE algebraic loop with negligible lag vs gravity dynamics
_TAU_RF_STATE = 0.005   # 5 ms


def step(x_grav, u, brain_params):
    """Single ODE step: gravity + linear acceleration + rf derivative, gravity output.

    Args:
        x_grav:      (9,)  [g_est (3,) | a_lin (3,) | rf (3,)]
                           g_est: gravity estimate, world frame [x=right, y=up, z=fwd] (m/s²)
                                  upright rest: [0, +9.81, 0]
                           a_lin: linear acceleration state, same world frame (m/s²)
                                  zero at rest; reserved for LVOR
                           rf:    rotational feedback state [yaw, pitch, roll] (deg/s)
                                  read by brain_model for VS input (VS/GE loop state)
        u:           (6,)  [w_est (3, deg/s) | gia (3, m/s²)]
                           w_est: VS net angular velocity, [yaw, pitch, roll] (deg/s)
                                  used for gravity transport (VN → uvula/nodulus)
                           gia:   gravitoinertial acceleration from otolith, world frame (m/s²)
                                  [x=right, y=up, z=fwd]; upright rest: [0, +9.81, 0]
        brain_params: BrainParams  (reads tau_grav, K_lin, K_gd)

    Returns:
        dx_grav: (9,)  d[ĝ; â; rf]/dt  — world frame
        g_est:   (3,)  current gravity estimate ĝ (= x_grav[:3], passed through)
    """
    g_est    = x_grav[_IDX_G]    # gravity estimate,  world frame (m/s²)
    rf_state = x_grav[_IDX_RF]   # current rf state — VS reads this from ODE state
    w_est    = u[:3]              # VS net angular velocity, [yaw, pitch, roll] (deg/s)
    gia      = u[3:]              # otolith GIA, world frame [x=right, y=up, z=fwd] (m/s²)

    # Estimated linear acceleration: GIA minus gravity estimate.
    # g_est tracks full GIA with TC = tau_grav (DC gain = 1, full somatogravic effect).
    # a_lin accumulates a_est for LVOR: a_trans = gia − g_est − a_lin (zero at rest).
    a_est = gia - g_est

    # Transport: rotate gravity estimate with VS angular velocity (VN → uvula/nodulus pathway)
    # ypr_to_xyz converts [yaw,pitch,roll] → xyz rotation-axis vector for cross product
    w_rad_xyz = jnp.radians(ypr_to_xyz(w_est))
    transport = -jnp.cross(w_rad_xyz, g_est)

    # Gravity correction: pull toward GIA with TC = tau_grav
    dg = transport + (1.0 / brain_params.tau_grav) * a_est

    # Linear acceleration: tracks a_est for LVOR / a_trans
    da = brain_params.K_lin * a_est

    # Rotational feedback for VS — Laurens & Angelaki (2011): GIA × G_down / G0²
    # G_down = −g_est (g_est is specific force UP; G_down points DOWN).
    # Zero at steady state (GIA ≈ −G_down); active when gravity estimate lags GIA.
    rf_new = xyz_to_ypr(jnp.cross(gia, -g_est)) / (G0 ** 2)

    # rf state: fast first-order tracking — breaks the VS/GE algebraic loop.
    # brain_model reads x_grav[_IDX_RF] at the top of step() and passes it to VS.
    drf = (rf_new - rf_state) / _TAU_RF_STATE

    return jnp.concatenate([dg, da, drf]), g_est
