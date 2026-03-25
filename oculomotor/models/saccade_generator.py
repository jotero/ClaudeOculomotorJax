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
Nonlinearities and modal gating
──────────────────────────────────────────────────────────────────────────────
Input nonlinearity (replaces linear D):
    gate_thresh  = σ(k_sac · (|e_motor| − threshold_sac))   threshold gate
    ê            = e_motor / |e_motor|                       direction unit vector
    u_burst_raw  = g_burst · tanh(|e_motor| / e_sat_sac) · ê  (no gate here)

    • gate_thresh ≈ 0 below threshold  → burst off
    • gate_thresh ≈ 1 above threshold  → burst on; magnitude set by tanh
    • tanh gives compressive nonlinearity → realistic main-sequence saturation

Modal gating in step():
    A_ni and A_reset carry only dynamics (no gate in the matrices).
    gate_thresh is multiplied in step() to blend the two modes:

        dx = gate_thresh       · (A_ni    @ x  +  B @ u_burst_raw)   [saccade mode]
           + (1−gate_thresh)   · (A_reset @ x)                        [reset mode]

        u_burst = gate_thresh · u_burst_raw                           [gated output]

    • A_ni   = −(1/tau_i)     · I₃   — NI dynamics, same as real NI
    • A_reset = −(1/tau_reset) · I₃  — fast decay to 0
    • B      = I₃                    — unit input gain

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


# ── Input nonlinearity (replaces linear D) ───────────────────────────────────

def input_nonlinearity(e_motor, theta):
    """Threshold gate + tanh saturation → raw burst magnitude and gate scalar.

    Returns the raw (ungated) burst and the gate separately so that step()
    can apply the gate exactly once to both the state equation and the output.

    Args:
        e_motor: (3,)  motor error = theta_target − x_p
        theta:   dict  model parameters

    Returns:
        u_burst_raw: (3,)    burst magnitude, gate NOT yet applied
        gate_thresh: scalar  saccade gate in [0, 1]
    """
    g_burst       = theta.get('g_burst',        600.0)
    threshold_sac = theta.get('threshold_sac',    0.5)
    k_sac         = theta.get('k_sac',           15.0)
    e_sat_sac     = theta.get('e_sat_sac',       10.0)

    err_mag     = jnp.linalg.norm(e_motor)
    gate_thresh = jax.nn.sigmoid(k_sac * (err_mag - threshold_sac))

    e_dir       = e_motor / (err_mag + 1e-6)
    u_burst_raw = g_burst * jnp.tanh(err_mag / e_sat_sac) * e_dir   # no gate here

    return u_burst_raw, gate_thresh


# ── Mode matrices (pure dynamics — no gating inside) ─────────────────────────

B = jnp.eye(3)   # (3, 3) — unit input gain (constant)


def get_A_ni(theta):
    """(3, 3) saccade-mode state matrix — NI dynamics, same as real NI."""
    return (-1.0 / theta.get('tau_i', 25.0)) * jnp.eye(3)


def get_A_reset(theta):
    """(3, 3) reset-mode state matrix — fast decay to 0 between saccades."""
    return (-1.0 / theta.get('tau_reset_sac', 0.1)) * jnp.eye(3)


# ── SSM step ─────────────────────────────────────────────────────────────────

def step(x_sg, u_sg, theta):
    """Single ODE step: state derivative + burst output.

    gate_thresh blends the two modes — applied to inputs, not baked into A:

        dx = gate       · (A_ni    @ x  +  B @ u_burst_raw)   saccade mode
           + (1−gate)   · (A_reset @ x)                        reset mode

        u_burst = gate · u_burst_raw                           gated output

    Args:
        x_sg:  (N_STATES,)  x_reset_int (3,) — resettable integrator
        u_sg:  (N_INPUTS,)  e_motor (3,)  = theta_target − x_p
        theta: dict         model parameters

    Returns:
        dx_sg:   (N_STATES,)  state derivative
        u_burst: (3,)         saccade velocity command (deg/s)
    """
    u_burst_raw, gate_thresh = input_nonlinearity(u_sg, theta)

    dx = (        gate_thresh  * (get_A_ni(theta)    @ x_sg + B @ u_burst_raw)
         + (1.0 - gate_thresh) * (get_A_reset(theta) @ x_sg))

    u_burst = gate_thresh * u_burst_raw
    return dx, u_burst
