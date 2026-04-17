"""Plant model — extraocular muscles and globe (Level 1b: 3-D linear).

Level 1b: three independent first-order LP filters, one per rotational axis
(yaw, pitch, roll).  All axes share the same time constant τ_p.  No
cross-axis coupling, no muscle geometry.  Simple and gradient-friendly.

First-order plant model: Robinson (1964 IEEE Trans Biomed Eng; 1981 Ann Rev
Neurosci). The pulse-step input architecture (NI feedthrough tau_p) that
cancels the plant lag is from Robinson (1975).

    dx_p/dt = clip_wall((motor_cmd − x_p) / τ_p)   (wall-clipped velocity)
    q_eye   = x_p                                   (state IS the bounded position)
    w_true  = dx_p                                  (consistent with q_eye)

The orbital walls are enforced by zeroing dx_p whenever x_p is at ±L and
the velocity would push it further outside.  This keeps x_p within [−L, +L]
so q_eye = x_p directly, and w_true = dx_p is always consistent with q_eye.

Binocular layout:
    N_STATES = 6: [x_p_L (3) | x_p_R (3)]  — left eye followed by right eye.
    step() operates on a single (3,) eye; the simulator calls it twice and
    concatenates.  _IDX_P_L / _IDX_P_R are convenience slices for the
    combined (6,) plant state.

State:   x_p  (3,)  eye rotation vector (deg), bounded within ±orbital_limit
Input:   motor_cmd  (3,)  pulse-step motor command from NI
Outputs: q_eye  (3,)  eye rotation vector (= x_p)
         w_true (3,)  instantaneous eye angular velocity (deg/s) (= dx_p)

Parameters:
  τ_p — plant time constant (s). Typical: 0.15 s (Robinson 1981; Goldstein
        1983 Biol Cybern).
  orbital_limit — mechanical half-range (deg). Default: 50 deg.
"""

from typing import NamedTuple

import jax.numpy as jnp


# ── Plant parameters ────────────────────────────────────────────────────────────

class PlantParams(NamedTuple):
    """Extraocular plant parameters — orbital mechanics.

    Determined by orbital anatomy and muscle physiology.  Varies with
    strabismus surgery, orbital inflammation, thyroid eye disease, etc.
    """
    tau_p:         float = 0.15    # plant TC (s); Robinson 1981, Goldstein 1983 Biol Cybern
    orbital_limit: float = 50.0   # mechanical half-range of the orbit (deg); anatomical
    k_orbital:     float = 1.0    # sigmoid steepness for orbital gate (1/deg)


# ── State layout ───────────────────────────────────────────────────────────────

N_STATES  = 6           # [x_p_L (3) | x_p_R (3)]  — binocular
N_INPUTS  = 3
N_OUTPUTS = 3   # q_eye (position, per eye)

# Index constants for the combined (6,) plant state
_IDX_P_L = slice(0, 3)  # left  eye rotation vector
_IDX_P_R = slice(3, 6)  # right eye rotation vector


def step(x_p, motor_cmd, plant_params):
    """Single ODE step: state derivative + eye position/velocity outputs.

    The orbital limit is enforced on dx_p (not the output): when x_p is at
    ±orbital_limit and the motor drive would push it further, the velocity is
    zeroed.  This keeps the state itself bounded so w_true = dx_p is always
    consistent with q_eye = x_p.

    Args:
        x_p:          (3,)  plant state = eye rotation vector (deg), ∈ [−L, +L]
        motor_cmd:    (3,)  pulse-step motor command from NI
        plant_params: PlantParams

    Returns:
        dx_p:  (3,)  wall-clipped dx_p/dt
        q_eye: (3,)  eye rotation vector (= x_p, bounded by construction)
        w_true:(3,)  instantaneous eye angular velocity (deg/s) (= dx_p)
    """
    tau_p = plant_params.tau_p
    L     = plant_params.orbital_limit

    w_raw  = (motor_cmd - x_p) / tau_p

    # Enforce orbital walls: zero velocity when at limit and pushing outward.
    # At +L: block positive velocity; at −L: block negative velocity.
    # Recovery (velocity pointing back inward) is always allowed.
    w_true = jnp.where(x_p >= L,  jnp.minimum(w_raw,   0.0), w_raw)
    w_true = jnp.where(x_p <= -L, jnp.maximum(w_true,  0.0), w_true)

    dx_p  = w_true
    q_eye = x_p   # state is bounded by the clipped dynamics
    return dx_p, q_eye, w_true
