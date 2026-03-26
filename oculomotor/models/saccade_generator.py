"""Saccade Generator SSM — Robinson (1975) local-feedback burst model.

Robinson DА (1975) "Oculomotor control signals" in "Basic Mechanisms of Ocular
Motility and Their Clinical Implications", Pergamon, pp. 337–374.
Burst neuron recordings: Fuchs, Scudder & Kaneko (1988 J Neurophysiol).
Main sequence (velocity–amplitude): Bahill, Clark & Stark (1975 Math Biosci).

Architecture
────────────
A resettable integrator (x_reset_int) mirrors the neural integrator during a
saccade.  The burst fires while the Robinson residual (retinal error minus copy)
is above threshold, and stops automatically when the copy catches up.

    dx_sg/dt = A_mode @ x_sg  +  B @ u_burst_raw   [saccade mode]
             = A_reset @ x_sg                        [reset mode, between saccades]
    u_burst  = gate_eff · u_burst_raw

State:  x_sg = x_reset_int (3,)   copy integrator — mirrors NI during saccade
Input:  u_sg = e_pos_delayed (3,) delayed retinal position error (deg)
Output: u_burst (3,)               saccade velocity command (deg/s)

──────────────────────────────────────────────────────────────────────────────
Three gates (all scalars, all differentiable)
──────────────────────────────────────────────────────────────────────────────
    gate_err  = σ(k · (|e_pos_delayed| − threshold))
        • High when retinal error is above threshold — saccade is needed.
        • Low when eye is on target — no drive, reset integrator quickly.
        • Controls the adaptive reset TC (see below).

    gate_res  = σ(k · (|e_residual| − threshold))
        • e_residual = e_pos_delayed − x_reset_int   (Robinson residual)
        • High while the copy integrator hasn't caught up yet — keep firing.
        • Drops to zero when copy matches error — burst stops.

    dir_gate  = relu(ê_error · ê_residual)
        • Suppresses burst if residual points opposite to error.
        • Prevents backward bursts when x_reset_int overshoots e_pos_delayed.

    gate_eff  = gate_err · gate_res · dir_gate   (combined gate)

──────────────────────────────────────────────────────────────────────────────
Burst nonlinearity
──────────────────────────────────────────────────────────────────────────────
    Magnitude: g_burst · tanh(|e_pos_delayed| / e_sat_sac)
        • Saturates on the full delayed error → nonlinear main sequence.
        • Robinson-feedback scale-invariance is broken: amplitude sets the speed.

    Direction: e_residual / |e_residual|
        • Residual direction drives the copy integrator to the right endpoint.

──────────────────────────────────────────────────────────────────────────────
Adaptive reset time constant
──────────────────────────────────────────────────────────────────────────────
Between saccades (gate_eff ≈ 0) x_reset_int decays toward zero.  The TC is
interpolated by gate_err:

    tau_eff = tau_reset_fast + gate_err · (tau_reset_sac − tau_reset_fast)

    gate_err ≈ 1  (error large, eye away from target):
        tau_eff ≈ tau_reset_sac (1 s) — x_reset_int stays elevated through
        plant settling (~5·tau_p ≈ 750 ms), preventing a secondary burst.

    gate_err ≈ 0  (error small, eye on target):
        tau_eff ≈ tau_reset_fast (0.1 s) — integrator resets in ~300 ms,
        ready for the next saccade.

──────────────────────────────────────────────────────────────────────────────
Output insertion (in simulator)
──────────────────────────────────────────────────────────────────────────────
    dx_ni  += u_burst            NI integrates burst → holds post-saccade position
    u_p    += tau_p · u_burst    velocity pulse → plant (cancels LP lag)

Parameters
──────────
    g_burst        burst ceiling (deg/s)              default 600.0
                   Peak saccade velocity ~600–700 deg/s for large saccades
                   (Bahill et al. 1975; Fuchs et al. 1988).
    threshold_sac  retinal-error threshold (deg)      default 0.5
                   Saccadic dead-zone ~0.5° (Steinman et al. 1967 Science;
                   Becker 1989 in "Neurobiology of Saccadic Eye Movements").
    k_sac          sigmoid steepness (1/deg)          default 15.0
    e_sat_sac      tanh saturation amplitude (deg)    default 7.0
                   Main-sequence saturation: peak velocity plateaus beyond
                   ~10–15° (Bahill et al. 1975).
    tau_i          NI leak TC — shared with NI (s)    default 25.0
    tau_reset_sac  reset TC when error is large (s)   default 1.0
    tau_reset_fast reset TC when eye is on target (s) default 0.1
                   Saccadic refractory period ~200 ms (Fischer &
                   Ramsperger 1984 Exp Brain Res).
"""

import jax.numpy as jnp
import jax

N_STATES  = 3   # x_reset_int (3,)
N_INPUTS  = 3   # e_pos_delayed (3,)
N_OUTPUTS = 3   # u_burst (3,)


# ── Input nonlinearity ────────────────────────────────────────────────────────

def input_nonlinearity(e_pos_delayed, e_residual, theta):
    """Compute burst vector and the two scalar error gates.

    Burst MAGNITUDE saturates on |e_pos_delayed| → nonlinear main sequence
    independent of feedback loop gain.
    Burst DIRECTION follows e_residual → Robinson stopping criterion.

    Args:
        e_pos_delayed: (3,)  delayed retinal position error
        e_residual:    (3,)  e_pos_delayed − x_reset_int  (Robinson residual)
        theta:         dict  model parameters

    Returns:
        u_burst_raw: (3,)    burst vector, gates NOT yet applied
        gate_err:    scalar  high when retinal error > threshold
        gate_res:    scalar  high while Robinson residual > threshold
    """
    g_burst       = theta.get('g_burst',        600.0)
    threshold_sac = theta.get('threshold_sac',    0.5)
    k_sac         = theta.get('k_sac',           15.0)
    e_sat_sac     = theta.get('e_sat_sac',        7.0)

    e_err_mag = jnp.linalg.norm(e_pos_delayed)
    e_res_mag = jnp.linalg.norm(e_residual)

    gate_err = jax.nn.sigmoid(k_sac * (e_err_mag - threshold_sac))   # error above threshold
    gate_res = jax.nn.sigmoid(k_sac * (e_res_mag - threshold_sac))   # residual above threshold

    # Magnitude from full delayed error → nonlinear main sequence
    # Direction from residual → Robinson stopping
    burst_mag   = g_burst * jnp.tanh(e_err_mag / e_sat_sac)
    e_res_dir   = e_residual / (e_res_mag + 1e-6)
    u_burst_raw = burst_mag * e_res_dir

    # Direction gate: suppress burst when residual points opposite to error.
    # Prevents backward burst when x_reset_int overshoots e_pos_delayed.
    e_norm   = e_pos_delayed / (jnp.linalg.norm(e_pos_delayed)       + 1e-6)
    res_norm = e_residual / (jnp.linalg.norm(e_residual)  + 1e-6)
    gate_dir = jax.nn.relu(jnp.dot(e_norm, res_norm))

    return u_burst_raw, gate_err, gate_res, gate_dir


# ── Mode matrices (pure dynamics — no gating inside) ─────────────────────────

B = jnp.eye(3)   # (3, 3) — unit input gain (constant)


def get_A_ni(theta):
    """(3, 3) saccade-mode state matrix — same leak as real NI."""
    return (-1.0 / theta.get('tau_i', 25.0)) * jnp.eye(3)


def get_A_reset(theta):
    """(3, 3) reset-mode state matrix — not used directly; TC is adaptive in step()."""
    return (-1.0 / theta.get('tau_reset_sac', 1.0)) * jnp.eye(3)


# ── SSM step ─────────────────────────────────────────────────────────────────

def step(x_sg, u_sg, theta):
    """Single ODE step: state derivative + burst output.

    Args:
        x_sg:  (N_STATES,)  x_reset_int (3,) — copy integrator state
        u_sg:  (N_INPUTS,)  e_pos_delayed (3,) — delayed retinal position error
        theta: dict         model parameters

    Returns:
        dx_sg:   (N_STATES,)  state derivative
        u_burst: (3,)         saccade velocity command (deg/s)
    """
    e_residual = u_sg - x_sg   # Robinson residual: delayed error minus resetable integrator

    u_burst_raw, gate_err, gate_res, gate_dir = input_nonlinearity(u_sg, e_residual, theta)


    gate_eff = gate_err * gate_res * gate_dir

    # Adaptive reset TC: slow when error is large (prevents re-trigger during
    # plant settling), fast when eye is on target (ready for next saccade).
    tau_reset = theta.get('tau_reset_sac',  1.0)
    tau_fast  = theta.get('tau_reset_fast', 0.1)
    tau_eff   = tau_fast + gate_err * (tau_reset - tau_fast)
    A_reset_eff = (-1.0 / tau_eff) * jnp.eye(3)

    dx = (        gate_eff  * (get_A_ni(theta) @ x_sg + B @ u_burst_raw)
         + (1.0 - gate_eff) * (A_reset_eff    @ x_sg))

    u_burst = gate_eff * u_burst_raw
    return dx, u_burst
