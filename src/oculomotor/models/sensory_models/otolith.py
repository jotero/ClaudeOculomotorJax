"""Otolith SSM — bilateral gravitoinertial acceleration sensors.

Models the utricle and saccule as a single lumped 3-D GIA sensor per side
(left / right), following the Laurens & Angelaki (2011, 2017) framework.

──────────────────────────────────────────────────────────────────────────
Anatomy summary
    Utricle  — horizontal macula, primarily sensitive to x/y GIA
    Saccule  — vertical macula, primarily sensitive to z GIA
    Each side lumped into one 3-D sensor (full-axis sensitivity).

    Unlike semicircular canals (clean push-pull pairs), left and right
    otoliths measure the SAME GIA with the SAME sign — they are parallel,
    not opposing.  Their bilateral output supports unilateral-loss detection
    and averaging reduces noise, but there is no differencing operator.

    PINV mixes: f̂ = (y_L + y_R) / 2

──────────────────────────────────────────────────────────────────────────
Physical signal
    GIA in head frame:  f(t) = g_head(t) + a_head(t)
        g_head  — gravity resolved into head frame = R(q_head)ᵀ · g_world
        a_head  — head linear acceleration (m/s²)

    Axis convention (world frame: x=right, y=up, z=forward):
        specific force is +y when head is upright (y=up)
        g_world = [0, +9.81, 0] m/s²

    At rest, upright: f = g_world = [0, +9.81, 0] m/s²

──────────────────────────────────────────────────────────────────────────
SSM interface (follows canal.py convention)
    State layout (6,): [x_L (3) | x_R (3)]  — LP adaptation states only.
    Head orientation q_head is passed as input (already integrated externally).

    step(x_oto, u, sensory_params) → (dx_oto, f_gia)
        u = [a_head (3) | q_head (3)]  — 6-D input
        f_gia  (3,)  — LP-filtered GIA estimate passed to gravity_estimator

──────────────────────────────────────────────────────────────────────────
Dynamics  (first-order adaptation, Fernandez & Goldberg 1976)
    The otolithic membrane settles slowly to a sustained GIA:
        dx_oto/dt = (S · f − x_oto) / τ_oto
        y_oto     = x_oto                    (LP-filtered GIA)

    τ_oto large (10–100 s) → y_oto ≈ f at all frequencies above ~0.01 Hz
    τ_oto small             → high-pass (adapts away sustained tilt)

    For the Laurens tilt-translation model, large τ_oto is preferred so
    that the DC gravity component is preserved.
"""

import jax.numpy as jnp

from oculomotor.models.plant_models.readout import rotation_matrix

# ── Sensor geometry ────────────────────────────────────────────────────────────

G0        = 9.81   # standard gravity (m/s²)
G_WORLD   = jnp.array([0., G0, 0.])   # specific force at rest, world frame (y=up)

# Sensitivity matrices (per side): full 3-D, identity (all axes equally sensitive)
SENS_LEFT  = jnp.eye(3)   # (3, 3)
SENS_RIGHT = jnp.eye(3)   # (3, 3)

# Mixing: GIA estimate = average of left and right LP states
# y shape (6,): [x_L (0:3) | x_R (3:6)]
PINV_SENS = 0.5 * jnp.array([
    [1., 0., 0., 1., 0., 0.],
    [0., 1., 0., 0., 1., 0.],
    [0., 0., 1., 0., 0., 1.],
])   # (3, 6): averages left and right per axis

# ── SSM constants ──────────────────────────────────────────────────────────────

N_STATES  = 6    # [x_L (3) | x_R (3)] — LP adaptation states; q_head is an input
N_INPUTS  = 6    # [a_head (3) | q_head (3)]
N_OUTPUTS = 3    # f_gia (3,) — GIA estimate

_IDX_L = slice(0, 3)   # left  otolith adaptation state (m/s²)
_IDX_R = slice(3, 6)   # right otolith adaptation state (m/s²)

# Default initial state: both sides settled to gravity at rest, upright head
X0 = jnp.concatenate([SENS_LEFT @ G_WORLD, SENS_RIGHT @ G_WORLD])   # [9.81,0,0, 9.81,0,0]


# ── SSM step ───────────────────────────────────────────────────────────────────

def step(x_oto, u, sensory_params):
    """Single ODE step: otolith LP derivative + GIA estimate output.

    Args:
        x_oto:          (6,)  otolith state [x_L (3) | x_R (3)]  (m/s²)
        u:              (6,)  [a_head (3) | q_head (3)]
                              a_head — head linear acceleration (m/s²)
                              q_head — head orientation rotation vector (deg)
        sensory_params: SensoryParams  (reads tau_oto)

    Returns:
        dx_oto: (6,)  dx_oto/dt  (m/s³)
        f_gia:  (3,)  LP-filtered GIA estimate → gravity_estimator (m/s²)
    """
    a_head = u[:3]   # (3,) head linear acceleration (m/s²)
    q_head = u[3:]   # (3,) head orientation rotation vector (deg)

    x_L = x_oto[_IDX_L]   # (3,) left  adaptation state
    x_R = x_oto[_IDX_R]   # (3,) right adaptation state

    # Rotate q_head [yaw,pitch,roll] → xyz rotation vector for rotation_matrix.
    # Convention: yaw→y, pitch→-x, roll→z  (matches retina.py _q2rv).
    q_xyz = jnp.array([-q_head[1], q_head[0], q_head[2]])
    R      = rotation_matrix(q_xyz)    # (3,3) world←head rotation
    g_head = R.T @ G_WORLD             # (3,) specific force in head frame

    # Rotate world-frame linear acceleration to head frame
    a_head_frame = R.T @ a_head

    # Gravitoinertial acceleration (GIA) = gravity component + linear acceleration
    f = g_head + a_head_frame

    # LP adaptation dynamics: low-pass filter the raw GIA
    tau  = sensory_params.tau_oto
    dx_L = (SENS_LEFT  @ f - x_L) / tau
    dx_R = (SENS_RIGHT @ f - x_R) / tau

    dx_oto = jnp.concatenate([dx_L, dx_R])
    # Return the INSTANTANEOUS GIA (= f), not the LP-adapted state.
    # The gravity estimator needs the raw otolith reading so the correction
    # term K_grav·(f−ĝ) works correctly during sustained tilt — if we pass the
    # LP state (tau_oto=100 s), the correction pulls ĝ back toward upright
    # once the canal decays, causing visible drift.
    # x_oto remains as an adaptation state for future somatogravic illusion
    # modelling (comparing adapted vs raw response).
    f_gia  = f   # instantaneous GIA in head frame (m/s²)

    return dx_oto, f_gia
