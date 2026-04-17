"""Neural Integrator SSM — leaky integrator of eye-velocity commands (3-D).

Converts a combined eye-velocity command into a sustained eye-position command.
The NI is agnostic to the source of the velocity: VOR, saccade, OKR, or any
combination.  Gains and sign conventions live upstream in the simulator.

Architecture: Robinson (1975) "Oculomotor control signals" in "Basic Mechanisms
of Ocular Motility", pp. 337–374. Leak time constant characterized in primate
by Cannon & Robinson (1985 Biol Cybern).

    dx_n/dt = A_n(θ) @ x_n + u_vel
    y_n     = C_n @ x_n + tau_p · u_vel

States:  x_n = [pos_cmd_x, pos_cmd_y, pos_cmd_z]   (3,)   eye position command (deg)
Input:   u_vel                                       (3,)   combined eye-velocity command
                                                            (already in eye coordinates,
                                                            sign-flipped and gain-scaled
                                                            by the caller)
Output:  y_n → pulse-step signal to Plant SSM

Pulse-step output:
    y_n = x_n  +  tau_p · u_vel

    The tau_p feedthrough exactly cancels the plant's low-pass lag so that
    eye position tracks the velocity command at all frequencies.
    tau_p is a plant parameter, not a VOR or saccade parameter — it belongs
    here because the NI must know the plant it drives.

VOR gain and sign:
    The sign inversion (eyes move opposite to head) and g_vor live in the
    simulator, applied to the VS velocity estimate before it reaches the NI:
        u_vor = −g_vor · w_est
    The NI itself is gain-free.

Parameters:
  tau_i  — integrator leak TC (s).  Healthy: >20 s (near-perfect integration);
           fitted value ~25 s in normal rhesus monkey
           (Cannon & Robinson 1985; Robinson 1975).
  tau_p  — plant TC (s); used only for the lag-cancellation feedthrough.
"""

import jax.numpy as jnp

N_STATES  = 3
N_INPUTS  = 3
N_OUTPUTS = 3


def step(x_ni, u_vel, brain_params):
    """Single ODE step: state derivative + motor command output.

    Args:
        x_ni:  (3,)  NI state (eye position command, deg)
        u_vel: (3,)  combined eye-velocity command (deg/s)
                     caller is responsible for sign flip and gain scaling
        theta: Params  model parameters

    Returns:
        dx:  (3,)  dx_ni/dt
        u_p: (3,)  pulse-step motor command to plant
    """
    # ── System matrices ───────────────────────────────────────────────────────
    A = (-1.0 / brain_params.tau_i) * jnp.eye(3)
    D = brain_params.tau_p * jnp.eye(3)
    # B = C = I (identity — omitted)

    # ── Dynamics ──────────────────────────────────────────────────────────────
    dx  = A @ x_ni + u_vel

    # Orbital anti-windup: mirror the plant wall-clip so x_ni stays bounded.
    # When x_ni is at ±orbital_limit and the command pushes further outward,
    # zero the derivative — exactly as plant_model does for dx_p.
    L   = brain_params.orbital_limit
    dx  = jnp.where(x_ni >= L,  jnp.minimum(dx, 0.0), dx)
    dx  = jnp.where(x_ni <= -L, jnp.maximum(dx, 0.0), dx)

    u_p = x_ni + D @ u_vel
    return dx, u_p
