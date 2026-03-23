"""Neural Integrator SSM — brainstem (PPRF/MVN) velocity-to-position integrator.

Converts the VOR velocity command into a sustained eye-position command.
This is the *plastic* subsystem: g_vor and tau_i can be modified by
cerebellar learning.

    dx_n/dt = A_n(θ) @ x_n + B_n(θ) @ u_n
    y_n     = C_n @ x_n

States:  x_n = [eye_pos_cmd]   (1,)   motor position command (deg)
Input:   u_n = [ω̂]             (1,)   estimated head velocity from VS (deg/s)
Output:  y_n = eye_pos_cmd → feeds Plant SSM

The VOR sign inversion (eyes move opposite to head) lives in B_n via −g_vor.
The integrator leak lives in A_n via −1/τ_i.

Parameters:
  g_vor  — VOR gain (unitless). Healthy: ~1.0. Adapted up/down by cerebellum.
  tau_i  — integrator time constant (s). Healthy: >20 s (near-perfect).
            Low tau_i → gaze-evoked nystagmus (cerebellar pathology).
"""

import jax.numpy as jnp

N_STATES = 1
N_INPUTS = 1
N_OUTPUTS = 1


def get_A(theta):
    """(1, 1) state matrix — leaky integrator."""
    return jnp.array([[-1.0 / theta['tau_i']]])


def get_B(theta):
    """(1, 1) input matrix — VOR gain with sign inversion."""
    return jnp.array([[-theta['g_vor']]])


C = jnp.array([[1.0]])  # (1, 1) — position component of output


def get_D(theta):
    """(1, 1) direct feedthrough — velocity burst component of motor neuron signal.

    The full output fed to the plant is:
        y_ni = C @ x_ni  +  D @ u_ni      (pulse-step signal)

    With D = −g_vor × τ_p the plant's low-pass lag is exactly cancelled:
        eye_pos(s) = −g_vor × head_pos(s)  at all frequencies.

    Note: τ_p appears here via cross-subsystem coupling (the direct pathway
    gain must be calibrated to the plant).  As a result τ_p is no longer
    identifiable from sinusoidal VOR data when D is included — consider
    fixing it as a known parameter (like τ_c).
    """
    return jnp.array([[-theta['g_vor'] * theta['tau_p']]])
