"""Neural Integrator SSM — brainstem velocity-to-position integrator (3-D).

Converts the VOR velocity command into a sustained eye-position command.
This is the *plastic* subsystem: g_vor and tau_i can be modified by
cerebellar learning.

    dx_n/dt = A_n(θ) @ x_n + B_n(θ) @ u_n
    y_n     = C_n @ x_n + D_n(θ) @ u_n

States:  x_n = [pos_cmd_x, pos_cmd_y, pos_cmd_z]   (3,)   motor position cmd (deg)
Input:   u_n = [ω̂_x, ω̂_y, ω̂_z]                   (3,)   velocity from VS
Output:  y_n → feeds Plant SSM (pulse-step signal)

The VOR sign inversion (eyes opposite to head) lives in B_n via −g_vor.
The integrator leak lives in A_n via −1/τ_i.

Pulse-step output (full NI output to plant):
    y_ni = C @ x_ni + D(θ) @ u_ni

With D = −g_vor · τ_p · I the plant's LP lag is exactly cancelled:
    q_eye(s) = −g_vor · q_head(s)   at all VOR frequencies.

Parameters:
  g_vor  — VOR gain (unitless). Healthy: ~1.0.
  tau_i  — integrator time constant (s). Healthy: >20 s (near-perfect).
  tau_p  — plant time constant (s); appears in D for lag cancellation.
"""

import jax.numpy as jnp

N_STATES  = 3
N_INPUTS  = 3
N_OUTPUTS = 3


def get_A(theta):
    """(3, 3) state matrix — leaky integrator, diagonal."""
    return (-1.0 / theta['tau_i']) * jnp.eye(3)


def get_B(theta):
    """(3, 3) input matrix — VOR gain with sign inversion."""
    return (-theta['g_vor']) * jnp.eye(3)


C = jnp.eye(3)   # (3, 3) — position component of output


def get_D(theta):
    """(3, 3) direct feedthrough — velocity burst component of motor neuron signal.

    The full output fed to the plant is:
        y_ni = C @ x_ni  +  D(θ) @ u_ni      (pulse-step signal)

    With D = −g_vor × τ_p · I the plant's low-pass lag is exactly cancelled:
        q_eye(s) = −g_vor × q_head(s)  at all VOR frequencies.
    """
    return (-theta['g_vor'] * theta['tau_p']) * jnp.eye(3)


def step(x_ni, u_ni, u_burst, theta):
    """Single ODE step: state derivative + motor command output.

    Two distinct input channels with different gains:

        u_ni    — velocity command from VS, scaled by B = −g_vor·I
                  (VOR sign inversion lives here)
        u_burst — saccade burst command, unit gain into state, +tau_p into plant
                  (already in eye coordinates — no sign flip, no g_vor scaling)

    Args:
        x_ni:    (3,)  NI state (position command)
        u_ni:    (3,)  velocity command from VS
        u_burst: (3,)  saccade burst velocity command (deg/s); zeros if no saccade
        theta:   dict  model parameters

    Returns:
        dx:  (3,)  dx_ni/dt
        u_p: (3,)  pulse-step motor command to plant
    """
    dx  = get_A(theta) @ x_ni + get_B(theta) @ u_ni + u_burst
    u_p = C @ x_ni + get_D(theta) @ u_ni + theta['tau_p'] * u_burst
    return dx, u_p
