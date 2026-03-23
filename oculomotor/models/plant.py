"""Plant SSM — extraocular muscles and globe (Level 1b: 3-D linear).

Level 1b: three independent first-order LP filters, one per rotational axis
(yaw, pitch, roll).  All axes share the same time constant τ_p.  No
cross-axis coupling, no muscle geometry.  Simple and gradient-friendly.

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
  τ_p — plant time constant (s). Typical: 0.15 s. Shared across axes.

Future levels:
  Level 2 — Robinson torque model: τ(q,ω) = B·ω + K·q, 6-muscle activations.
  Level 3 — Add pulley offsets per muscle (position-dependent pulling directions).
"""

import jax.numpy as jnp

N_STATES  = 3
N_INPUTS  = 3
N_OUTPUTS = 3


def get_A(theta):
    """(3, 3) state matrix — diagonal LP decay."""
    return (-1.0 / theta['tau_p']) * jnp.eye(3)


def get_B(theta):
    """(3, 3) input matrix — diagonal LP drive."""
    return (1.0 / theta['tau_p']) * jnp.eye(3)


C = jnp.eye(3)   # (3, 3) — output is the full rotation vector
