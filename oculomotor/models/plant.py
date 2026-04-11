"""Plant SSM — extraocular muscles and globe (Level 1b: 3-D linear).

Level 1b: three independent first-order LP filters, one per rotational axis
(yaw, pitch, roll).  All axes share the same time constant τ_p.  No
cross-axis coupling, no muscle geometry.  Simple and gradient-friendly.

First-order plant model: Robinson (1964 IEEE Trans Biomed Eng; 1981 Ann Rev
Neurosci). The pulse-step input architecture (NI feedthrough tau_p) that
cancels the plant lag is from Robinson (1975).

    dx_p/dt = A_p(θ) @ x_p + B_p(θ) @ u_p
    y_p     = C_p @ x_p

States:  x_p = [q_x, q_y, q_z]       (3,)  eye rotation vector (deg)
                x = yaw  (rightward +)
                y = pitch (upward +)
                z = roll  (CW from subject view +)
Input:   u_p = [cmd_x, cmd_y, cmd_z]  (3,)  motor command from Neural SSM
Output:  y_p = q  (eye rotation vector — clinically observable)

The rotation-vector representation is exact for the small angles encountered
in VOR and saccades (|q| < 60°).  For Level 2+ (muscle geometry, pulleys)
the readout module converts q to muscle coordinates; the plant ODE itself
remains identical.

Parameters:
  τ_p — plant time constant (s). Typical: 0.15 s, shared across axes
        (Robinson 1981; Goldstein 1983 Biol Cybern).

Future levels:
  Level 2 — Robinson torque model: τ(q,ω) = B·ω + K·q, 6-muscle activations.
  Level 3 — Add pulley offsets per muscle (position-dependent pulling directions).
"""

import jax.numpy as jnp

N_STATES  = 3


def soft_limit(x_p, theta):
    """Soft (differentiable) saturation of eye position to ±orbital_limit (deg).

    Applied to the plant OUTPUT only — the internal state x_p is left unbounded
    so the ODE integrator remains unconstrained.

    Formula:  q_sat = L * tanh(x_p / L)
        - identity near origin: error < 1% for |x_p| < 30 deg
        - saturates monotonically to ±L
        - gradient = sech²(x_p / L) > 0 always (no gradient saturation)
    """
    L = theta.get('orbital_limit', 50.0)
    return L * jnp.tanh(x_p / L)
N_INPUTS  = 3
N_OUTPUTS = 3


def get_A(theta):
    """(3, 3) state matrix — diagonal LP decay."""
    return (-1.0 / theta['tau_p']) * jnp.eye(3)


def get_B(theta):
    """(3, 3) input matrix — diagonal LP drive."""
    return (1.0 / theta['tau_p']) * jnp.eye(3)


C = jnp.eye(3)   # (3, 3) — output is the full rotation vector


def velocity(x_p, u_p, theta):
    """Eye angular velocity = d(x_p)/dt (algebraic from plant ODE).

    For the first-order plant, velocity is not an independent state but is
    fully determined by position and motor command: w_eye = (u_p − x_p)/τ_p.
    Used to compute retinal slip without adding redundant state variables.
    """
    return get_A(theta) @ x_p + get_B(theta) @ u_p


def step(x_p, u_p, theta):
    """Single ODE step: state derivative + eye position output.

    Args:
        x_p:   (3,)  plant state (eye rotation vector, deg)
        u_p:   (3,)  pulse-step motor command from NI
        theta: dict  model parameters

    Returns:
        dx:    (3,)  dx_p/dt  (= eye angular velocity)
        q_eye: (3,)  eye rotation vector C@x_p  (= x_p for this plant)
    """
    dx    = get_A(theta) @ x_p + get_B(theta) @ u_p
    q_eye = soft_limit(x_p, theta)
    return dx, q_eye
