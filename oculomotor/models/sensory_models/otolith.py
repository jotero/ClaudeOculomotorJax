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
    GIA in head frame:  f(t) = a_head(t) + g_head(t)
        a_head  — linear acceleration of head (m/s²)
        g_head  — gravity vector expressed in the instantaneous head frame
                  = R(q_head)ᵀ · g_world
                  rotates as the head tilts / rotates

    At rest, upright: f = g_world = [0, 0, −9.81] m/s²

──────────────────────────────────────────────────────────────────────────
Dynamics  (first-order adaptation, Fernandez & Goldberg 1976)
    The otolithic membrane settles slowly to a sustained GIA:
        dx_oto/dt = (S · f − x_oto) / τ_oto
        y_oto     = x_oto                    (LP-filtered GIA)

    τ_oto large (10–100 s) → y_oto ≈ f at all frequencies above ~0.01 Hz
    τ_oto small             → high-pass (adapts away sustained tilt)

    For the Laurens tilt-translation model, large τ_oto is preferred so
    that the DC gravity component is preserved.

──────────────────────────────────────────────────────────────────────────
Head orientation integration
    The orientation of the head relative to the world is needed to resolve
    gravity into the head frame.  A rotation vector q_head (deg) is
    integrated from the angular velocity input:

        dq_head/dt ≈ ω_head          (small-angle; valid for |q| ≲ 30°)

    Initial orientation set by initial_q_head parameter (default: upright).

──────────────────────────────────────────────────────────────────────────
State layout  (9 states)
    x_oto = [q_head (3) | x_L (3) | x_R (3)]
              head orient   left LP   right LP

Input  (6-DOF)
    omega_array : (T, 3) or (T,)   head angular velocity (deg/s)
    a_array     : (T, 3)            head linear acceleration (m/s²)
                  default zeros (pure rotation / tilt)

Parameters
    tau_oto   — adaptation time constant (s). Default: 100 s (near-static).
    g_world   — gravity vector in world frame (m/s²). Default: [0, 0, −9.81].
    initial_q_head — initial head orientation rotation vector (deg).
                     Default: [0, 0, 0] (upright).
"""

import jax
import jax.numpy as jnp
import diffrax

from oculomotor.models.plant_models.readout import rotation_matrix

# ── Sensor geometry ────────────────────────────────────────────────────────────

N_SIDES   = 2   # left, right
N_AXES    = 3   # x (lateral), y (fore-aft), z (vertical) per side
N_SENSORS = N_SIDES * N_AXES   # 6

G_MAGNITUDE = 9.81   # m/s²

# Sensitivity matrices (per side): full 3-D, no preferred axis
SENS_LEFT  = jnp.eye(3)   # (3, 3)
SENS_RIGHT = jnp.eye(3)   # (3, 3)

# Mixing: GIA estimate = average of left and right outputs
# y shape (6,): [y_L (0:3) | y_R (3:6)]
PINV_SENS = 0.5 * jnp.array([
    [1., 0., 0., 1., 0., 0.],
    [0., 1., 0., 0., 1., 0.],
    [0., 0., 1., 0., 0., 1.],
])   # (3, 6): averages left and right per axis

# ── State indexing ─────────────────────────────────────────────────────────────

N_STATES = 9
_IDX_Q = slice(0, 3)   # head orientation (rotation vector, deg)
_IDX_L = slice(3, 6)   # left  otolith adaptation state (m/s²)
_IDX_R = slice(6, 9)   # right otolith adaptation state (m/s²)


# ── Initial state ──────────────────────────────────────────────────────────────

def get_initial_state(g_world, initial_q_head=None):
    """Compute the settled initial state.

    Otolith LP states initialised to gravity in the initial head frame
    (i.e. the system starts fully adapted — no transient at t = 0).

    Args:
        g_world:        (3,) gravity in world frame (m/s²)
        initial_q_head: (3,) initial head orientation rotation vector (deg).
                        Default: [0, 0, 0] (upright, g_head = g_world).

    Returns:
        x0: (9,) initial state
    """
    if initial_q_head is None:
        initial_q_head = jnp.zeros(3)
    R      = rotation_matrix(initial_q_head)        # head-to-world rotation
    g_head = R.T @ g_world                          # gravity in initial head frame
    return jnp.concatenate([initial_q_head,
                             SENS_LEFT  @ g_head,   # left  settled to gravity
                             SENS_RIGHT @ g_head])  # right settled to gravity


# ── ODE vector field ───────────────────────────────────────────────────────────

def otolith_vector_field(t, x_oto, args):
    """ODE right-hand side for otolith dynamics.

    Args:
        t:     scalar time (s)
        x_oto: (N_STATES,) = [q_head | x_L | x_R]
        args:  (motion_interp, theta_oto, g_world)
               motion_interp evaluates (6,) = [omega (3) | a_lin (3)] at time t

    Returns:
        dx_oto: (N_STATES,)
    """
    motion_interp, theta_oto, g_world = args

    q_head = x_oto[_IDX_Q]   # (3,) head orientation (rotation vector, deg)
    x_L    = x_oto[_IDX_L]   # (3,) left  adaptation state
    x_R    = x_oto[_IDX_R]   # (3,) right adaptation state

    motion = motion_interp.evaluate(t)   # (6,)
    omega  = motion[:3]                  # (3,) head angular velocity (deg/s)
    a_lin  = motion[3:]                  # (3,) head linear acceleration (m/s²)

    # ── Head orientation ─────────────────────────────────────────────────────
    # Small-angle integration: dq/dt ≈ ω  (valid for |q| ≲ 30°).
    dq_head = omega

    # ── Gravity in current head frame ─────────────────────────────────────────
    R      = rotation_matrix(q_head)    # (3, 3) head-to-world rotation
    g_head = R.T @ g_world              # gravity resolved into head frame

    # ── Gravitoinertial acceleration (GIA) in head frame ──────────────────────
    f = a_lin + g_head                  # GIA = linear accel + gravity component

    # ── Adaptation LP dynamics ─────────────────────────────────────────────────
    tau  = theta_oto['tau_oto']
    dx_L = (SENS_LEFT  @ f - x_L) / tau
    dx_R = (SENS_RIGHT @ f - x_R) / tau

    return jnp.concatenate([dq_head, dx_L, dx_R])


# ── Output extraction ──────────────────────────────────────────────────────────

def otolith_outputs(x_oto):
    """Extract sensor outputs from state vector.

    Returns:
        y: (N_SENSORS,) = [y_L (3) | y_R (3)]   LP-filtered GIA per side (m/s²)
    """
    return jnp.concatenate([x_oto[_IDX_L], x_oto[_IDX_R]])


def gia_estimate(y_oto):
    """3-D GIA estimate by averaging left and right outputs.

    Args:
        y_oto: (N_SENSORS,) or (T, N_SENSORS)

    Returns:
        f_hat: (3,) or (T, 3)   best GIA estimate (m/s²)
    """
    return (PINV_SENS @ y_oto.T).T if y_oto.ndim == 2 else PINV_SENS @ y_oto


# ── Simulation entry point ─────────────────────────────────────────────────────

def simulate(theta_oto, t_array, omega_array, a_array=None,
             g_world=None, initial_q_head=None,
             dt_solve=0.005, max_steps=10000):
    """Integrate otolith ODE for 6-DOF head motion.

    Args:
        theta_oto:      dict with key 'tau_oto' (s, adaptation time constant)
        t_array:        (T,) time array (s)
        omega_array:    (T, 3) or (T,) head angular velocity (deg/s).
                        If 1-D, treated as yaw only; pitch and roll = 0.
        a_array:        (T, 3) head linear acceleration (m/s²).
                        Default: zeros (pure rotation/tilt, no translation).
        g_world:        (3,) gravity vector in world frame (m/s²).
                        Default: [0, 0, −9.81] (z-down convention).
        initial_q_head: (3,) initial head orientation rotation vector (deg).
                        Default: [0, 0, 0] (upright).
        dt_solve:       Heun fixed step (s). Default 0.005 s.
        max_steps:      ODE solver step budget.

    Returns:
        y_oto:        (T, 6) sensor outputs [y_L (3) | y_R (3)] — LP-filtered GIA
        head_orient:  (T, 3) head orientation rotation vector (deg) over time
    """
    # ── Defaults ──────────────────────────────────────────────────────────────
    if g_world is None:
        g_world = jnp.array([0., 0., -G_MAGNITUDE])

    if jnp.ndim(omega_array) == 1:
        omega_3d = jnp.stack([omega_array,
                               jnp.zeros_like(omega_array),
                               jnp.zeros_like(omega_array)], axis=1)
    else:
        omega_3d = jnp.asarray(omega_array)

    if a_array is None:
        a_3d = jnp.zeros_like(omega_3d)
    else:
        a_3d = jnp.asarray(a_array)

    # ── Interpolant for 6-DOF motion ──────────────────────────────────────────
    motion_6d     = jnp.concatenate([omega_3d, a_3d], axis=1)   # (T, 6)
    motion_interp = diffrax.LinearInterpolation(ts=t_array, ys=motion_6d)

    # ── Initial state ──────────────────────────────────────────────────────────
    x0 = get_initial_state(g_world, initial_q_head)

    # ── Integrate ──────────────────────────────────────────────────────────────
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(otolith_vector_field),
        diffrax.Heun(),
        t0=t_array[0],
        t1=t_array[-1],
        dt0=dt_solve,
        y0=x0,
        args=(motion_interp, theta_oto, g_world),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=t_array),
        max_steps=max_steps,
    )

    states = solution.ys                                    # (T, 9)
    y_oto  = jax.vmap(otolith_outputs)(states)             # (T, 6)

    return y_oto, states[:, _IDX_Q]                        # outputs, head orientation
