"""Plant model — extraocular muscles and globe (Level 1b: 3-D linear).

Level 1b: three independent first-order LP filters, one per rotational axis
(yaw, pitch, roll).  All axes share the same time constant τ_p.  No
cross-axis coupling, no muscle geometry.  Simple and gradient-friendly.

First-order plant model: Robinson (1964 IEEE Trans Biomed Eng; 1981 Ann Rev
Neurosci). The pulse-step input architecture (NI feedthrough tau_p) that
cancels the plant lag is from Robinson (1975).

    dx_p/dt = A(θ) @ x_p + B(θ) @ u_p
    q_eye   = soft_limit(x_p)          (nonlinear output — not C @ x_p)

States:  x_p = [q_x, q_y, q_z]       (3,)  eye rotation vector (deg)
                x = yaw  (rightward +)
                y = pitch (upward +)
                z = roll  (CW from subject view +)
Input:   u_p = [cmd_x, cmd_y, cmd_z]  (3,)  pulse-step motor command from NI
Output:  q_eye                              eye rotation vector, soft-limited to
                                            ±orbital_limit via tanh

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
    tau_p:         float = 0.15   # plant TC (s); Robinson 1981, Goldstein 1983 Biol Cybern
    orbital_limit: float = 50.0   # mechanical half-range of the orbit (deg); anatomical
    k_orbital:     float = 1.0    # sigmoid steepness for orbital gate (1/deg)


N_STATES  = 3
N_INPUTS  = 3
N_OUTPUTS = 3


def soft_limit(x_p, theta):
    """Soft (differentiable) saturation of eye position to ±orbital_limit (deg).

    Applied to the plant OUTPUT only — the internal state x_p is left unbounded
    so the ODE integrator remains unconstrained.

    Formula:  q_sat = L * tanh(x_p / L)
        - identity near origin: error < 1% for |x_p| < 30 deg
        - saturates monotonically to ±L
        - gradient = sech²(x_p / L) > 0 always (no gradient saturation)
    """
    L = theta.plant.orbital_limit
    return L * jnp.tanh(x_p / L)


def step(x_p, u_p, theta):
    """Single ODE step: state derivative + eye position output.

    Args:
        x_p:   (3,)    plant state (eye rotation vector, deg)
        u_p:   (3,)    pulse-step motor command from NI
        theta: Params  model parameters

    Returns:
        dx_p:  (3,)  dx_p/dt  (= eye angular velocity, deg/s)
        q_eye: (3,)  eye rotation vector, soft-limited to ±orbital_limit
    """
    # ── System matrices ───────────────────────────────────────────────────────
    A = (-1.0 / theta.plant.tau_p) * jnp.eye(3)
    B = ( 1.0 / theta.plant.tau_p) * jnp.eye(3)
    # C = I (identity — omitted); output is soft_limit(x_p) not C @ x_p

    # ── Dynamics ──────────────────────────────────────────────────────────────
    dx_p  = A @ x_p + B @ u_p
    q_eye = soft_limit(x_p, theta)
    return dx_p, q_eye
