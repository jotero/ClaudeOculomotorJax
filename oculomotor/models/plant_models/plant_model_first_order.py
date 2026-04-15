"""Plant model — extraocular muscles and globe (Level 1b: 3-D linear).

Level 1b: three independent first-order LP filters, one per rotational axis
(yaw, pitch, roll).  All axes share the same time constant τ_p.  No
cross-axis coupling, no muscle geometry.  Simple and gradient-friendly.

First-order plant model: Robinson (1964 IEEE Trans Biomed Eng; 1981 Ann Rev
Neurosci). The pulse-step input architecture (NI feedthrough tau_p) that
cancels the plant lag is from Robinson (1975).

    dx_q/dt = (motor_cmd − q) / τ_p        (position state)
    w_true  = (motor_cmd − q) / τ_p        (algebraic velocity; equals dq/dt)
    q_eye   = soft_limit(q)                (output, deg)

State:   x_p = q_eye  (3,)   eye rotation vector (deg)
Input:   motor_cmd    (3,)   pulse-step motor command from NI
Outputs: q_eye  (3,)  soft-limited eye rotation vector
         w_true (3,)  instantaneous eye angular velocity (deg/s)

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

N_STATES  = 3
N_INPUTS  = 3
N_OUTPUTS = 3   # q_eye (position)


def soft_limit(q, plant_params):
    """Soft (differentiable) saturation of eye position to ±orbital_limit (deg).

    Applied to the plant OUTPUT only — the internal state q is left unbounded
    so the ODE integrator remains unconstrained.

    Formula:  q_sat = L * tanh(q / L)
        - identity near origin: error < 1% for |q| < 30 deg
        - saturates monotonically to ±L
        - gradient = sech²(q / L) > 0 always (no gradient saturation)
    """
    L = plant_params.orbital_limit
    return L * jnp.tanh(q / L)


def step(x_p, motor_cmd, plant_params):
    """Single ODE step: state derivative + eye position/velocity outputs.

    Args:
        x_p:         (3,)  plant state = eye rotation vector (deg)
        motor_cmd:   (3,)  pulse-step motor command from NI
        plant_params: PlantParams

    Returns:
        dx_p:  (3,)  dx_p/dt = w_true
        q_eye: (3,)  eye rotation vector, soft-limited to ±orbital_limit
        w_true:(3,)  instantaneous eye angular velocity (deg/s)
    """
    tau_p = plant_params.tau_p

    w_true = (motor_cmd - x_p) / tau_p   # algebraic velocity = dq/dt
    dx_p   = w_true
    q_eye  = soft_limit(x_p, plant_params)
    return dx_p, q_eye, w_true
