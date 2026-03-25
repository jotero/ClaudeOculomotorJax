"""Saccade Generator SSM — burst model with resettable integrator.

Follows the same dx = A@x + B@u / y = D(u) structure as the other SSMs,
with two nonlinearities replacing the linear D:

    dx_sg/dt = A(gate_thresh, θ) @ x_sg  +  B @ u_burst
    u_burst  = D(e_motor, θ)                              [feedthrough]

    A(gate_thresh) — gated diagonal leak (see get_A)
    B = I          — burst integrates into x_reset_int with unit gain
    D(e_motor)     — nonlinear feedthrough (see input_nonlinearity):
                       1. threshold gate:  gate_thresh = σ(k·(|e|−threshold_sac))
                       2. tanh saturation: u_burst = g·tanh(|e|/e_sat)·ê·gate_thresh

State:  x_sg = x_reset_int (3,)   resettable integrator (mirrors NI during saccade)
Input:  u_sg = e_motor (3,)        = theta_target − x_p  (actual motor error)
Output: u_burst (3,)               saccade velocity command (deg/s)

──────────────────────────────────────────────────────────────────────────────
Nonlinearities
──────────────────────────────────────────────────────────────────────────────
Input nonlinearity D(e_motor, θ):
    gate_thresh = σ(k_sac · (|e_motor| − threshold_sac))   threshold gate
    ê           = e_motor / |e_motor|                       direction unit vector
    u_burst     = g_burst · tanh(|e_motor| / e_sat_sac) · ê · gate_thresh

    • gate_thresh ≈ 0 below threshold  → burst off
    • gate_thresh ≈ 1 above threshold  → burst on; magnitude set by tanh
    • tanh gives compressive nonlinearity → realistic main-sequence saturation
      (small saccades linear, large saccades approach g_burst ceiling)

State matrix A(gate_thresh, θ):
    leak = gate_thresh / tau_i  +  (1 − gate_thresh) / tau_reset_sac
    A    = −leak · I₃

    • gate_thresh ≈ 1 (saccade active):  A = −(1/tau_i)·I   ← same as real NI
    • gate_thresh ≈ 0 (between saccades): A = −(1/tau_reset)·I  ← fast reset to 0

──────────────────────────────────────────────────────────────────────────────
Output insertion (in simulator):
    dx_ni  += u_burst            unit gain → NI holds post-saccade position
    u_p    += tau_p · u_burst    velocity pulse → plant (cancels LP lag)

Parameters
──────────
    g_burst        burst ceiling (deg/s)         default 600.0
    threshold_sac  trigger threshold (deg)        default 0.5
    k_sac          sigmoid steepness (1/deg)      default 15.0
    e_sat_sac      tanh saturation amplitude (deg) default 10.0
    tau_i          NI leak TC — shared with NI (s) default 25.0
    tau_reset_sac  inter-saccade reset TC (s)      default 0.1
"""

import jax.numpy as jnp
import jax

N_STATES  = 3   # x_reset_int (3,)
N_INPUTS  = 3   # e_motor (3,)  [= theta_target − x_p]
N_OUTPUTS = 3   # u_burst (3,)


# ── Target geometry ─────────────────────────────────────────────────────────────

def target_to_angle(p_target):
    """Convert Cartesian target position to angular gaze direction (deg).

    Args:
        p_target: (3,)  [x (rightward), y (upward), z (forward/depth)]

    Returns:
        (3,)  [yaw (rightward+), pitch (upward+), roll=0]  in degrees
    """
    x, y, z = p_target[0], p_target[1], p_target[2]
    yaw   = jnp.degrees(jnp.arctan2(x, z))
    pitch = jnp.degrees(jnp.arctan2(y, z))
    return jnp.array([yaw, pitch, 0.0])


# ── Nonlinear input → output (replaces linear D) ─────────────────────────────

def input_nonlinearity(e_motor, theta):
    """Threshold gate × tanh saturation → burst command and gate scalar.

    Two nonlinearities applied to the motor error vector:
      1. Threshold gate:  gate_thresh = σ(k · (|e_motor| − threshold_sac))
      2. tanh saturation: u_burst     = g_burst · tanh(|e_motor| / e_sat_sac) · ê

    Args:
        e_motor: (3,)  motor error = theta_target − x_p
        theta:   dict  model parameters

    Returns:
        u_burst:     (3,)    burst velocity command (deg/s)
        gate_thresh: scalar  saccade gate in [0, 1]
    """
    g_burst       = theta.get('g_burst',        600.0)
    threshold_sac = theta.get('threshold_sac',    0.5)
    k_sac         = theta.get('k_sac',           15.0)
    e_sat_sac     = theta.get('e_sat_sac',       10.0)

    err_mag     = jnp.linalg.norm(e_motor)
    gate_thresh = jax.nn.sigmoid(k_sac * (err_mag - threshold_sac))

    e_dir   = e_motor / (err_mag + 1e-6)
    u_burst = g_burst * jnp.tanh(err_mag / e_sat_sac) * e_dir * gate_thresh

    return u_burst, gate_thresh


# ── State matrix (depends on gate from input nonlinearity) ───────────────────

B = jnp.eye(3)   # (3, 3) — burst integrates into x_reset_int with unit gain


def get_A(gate_thresh, theta):
    """(3, 3) state matrix — gated diagonal leak.

    Interpolates between NI leak (during saccade) and fast reset (between):
        leak = gate_thresh / tau_i  +  (1 − gate_thresh) / tau_reset_sac
        A    = −leak · I₃

    Args:
        gate_thresh: scalar  from input_nonlinearity
        theta:       dict    model parameters
    """
    tau_i     = theta.get('tau_i',         25.0)
    tau_reset = theta.get('tau_reset_sac',  0.1)
    leak      = gate_thresh / tau_i + (1.0 - gate_thresh) / tau_reset
    return -leak * jnp.eye(3)


# ── SSM step ─────────────────────────────────────────────────────────────────

def step(x_sg, u_sg, theta):
    """Single ODE step: state derivative + burst output.

    dx_sg  = A(gate_thresh, θ) @ x_sg  +  B @ u_burst
    u_burst = input_nonlinearity(e_motor, θ)

    Args:
        x_sg:  (N_STATES,)  x_reset_int (3,) — resettable integrator
        u_sg:  (N_INPUTS,)  e_motor (3,)  = theta_target − x_p
        theta: dict         model parameters

    Returns:
        dx_sg:   (N_STATES,)  state derivative
        u_burst: (3,)         saccade velocity command (deg/s)
    """
    u_burst, gate_thresh = input_nonlinearity(u_sg, theta)
    dx = get_A(gate_thresh, theta) @ x_sg + B @ u_burst
    return dx, u_burst
