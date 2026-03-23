"""Plant SSM — extraocular muscles and globe (fixed biomechanics).

First-order low-pass filter on the motor command.

    dx_p/dt = A_p(θ) @ x_p + B_p(θ) @ u_p
    y_p     = C_p @ x_p

States:  x_p = [eye_pos]       (1,)   eye position, deg
Input:   u_p = [eye_pos_cmd]   (1,)   motor command from Neural SSM
Output:  y_p = eye_pos (the clinically observable signal)

Parameters: tau_p — plant time constant (s). Typical: 0.15 s.
Fixed biomechanics; not modified by learning.

Future: replace with second-order Robinson model (2 states) or a
MuJoCo/MJX biomechanical eye model.
"""

import jax.numpy as jnp

N_STATES = 1
N_INPUTS = 1
N_OUTPUTS = 1


def get_A(theta):
    """(1, 1) state matrix."""
    return jnp.array([[-1.0 / theta['tau_p']]])


def get_B(theta):
    """(1, 1) input matrix."""
    return jnp.array([[1.0 / theta['tau_p']]])


C = jnp.array([[1.0]])  # (1, 1) — output is eye_pos
