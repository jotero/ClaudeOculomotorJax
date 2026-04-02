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
    gate_err  = σ(k · (|e_pos_delayed| − threshold_sac))
        • High when retinal error exceeds the saccadic dead-zone.
        • Triggers and sustains the burst.

    gate_res  = σ(k · (|e_residual| − threshold_stop))
        • threshold_stop ≪ threshold_sac so gate_res ≈ 1 at saccade onset,
          giving full burst strength the moment the trigger fires.
        • Drops to zero when the copy integrator catches up — burst stops.

    z_ref  — refractory / OPN state  (4th element of x_sg)
        • Charges rapidly after each burst ends when the error persists.
        • Suppresses gate_err (the trigger input), not the burst output —
          analogous to omnipause neurons (OPNs) blocking excitatory burst
          neuron input during the inter-saccadic interval.
        • Decays with τ_ref (~150 ms) to set the refractory period.

    dir_gate  = relu(ê_error · ê_residual)
        • Suppresses burst if residual points opposite to error.
        • Prevents backward bursts when x_reset_int overshoots e_pos_delayed.

    gate_eff  = gate_err · gate_res · dir_gate   (combined gate)

──────────────────────────────────────────────────────────────────────────────
Burst nonlinearity  (output nonlinearity on residual — matches main sequence)
──────────────────────────────────────────────────────────────────────────────
    Magnitude: g_burst · (1 − exp(−|e_residual| / e_sat_sac))
        • Applies the main-sequence saturation to the RESIDUAL (remaining
          distance), not the full initial error.
        • At saccade onset x_sg = 0, so |e_residual| = |e_pos_delayed| and the
          peak velocity matches the empirical main sequence:
              v_peak ≈ g_burst · (1 − exp(−A / e_sat_sac))
          with g_burst ≈ 700 deg/s and e_sat_sac ≈ 7°
          (Bahill, Clark & Stark 1975; Van Opstal & Van Gisbergen 1987).
        • Velocity tapers naturally as residual → 0, producing a bell-shaped
          burst profile without extra shaping logic.

    Direction: e_residual / |e_residual|
        • Residual direction drives the copy integrator to the right endpoint.

──────────────────────────────────────────────────────────────────────────────
Adaptive reset time constant
──────────────────────────────────────────────────────────────────────────────
A refractory state z_ref (4th element of x_sg) provides hysteresis:

    dz_ref/dt = (1−z_ref) · gate_err · (1−gate_res) / τ_charge  −  z_ref / τ_ref

    The charge signal  gate_err · (1−gate_res)  is high only post-saccade:
        • gate_err ≈ 1   (error still large — moving target)
        • gate_res ≈ 0   (residual caught up — burst just ended)
    It is zero during a saccade (gate_res≈1) and zero at rest (gate_err≈0).

    While z_ref is elevated:
        gate_active = gate_eff · (1−z_ref) ≈ 0  →  burst fully suppressed
        tau_eff = tau_fast                        →  x_reset_int resets quickly

    This breaks the continuous-burst equilibrium that otherwise occurs when a
    moving target keeps the residual growing throughout the reset phase.

──────────────────────────────────────────────────────────────────────────────
Output insertion (in simulator)
──────────────────────────────────────────────────────────────────────────────
    dx_ni  += u_burst            NI integrates burst → holds post-saccade position
    u_p    += tau_p · u_burst    velocity pulse → plant (cancels LP lag)

Parameters
──────────
    g_burst        burst ceiling (deg/s)              default 700.0
                   Peak saccade velocity ~600–700 deg/s for large saccades
                   (Bahill et al. 1975; Fuchs et al. 1988).
    threshold_sac  retinal-error trigger threshold (deg)  default 0.5
                   Saccadic dead-zone ~0.5° (Steinman et al. 1967 Science;
                   Becker 1989 in "Neurobiology of Saccadic Eye Movements").
    threshold_stop residual stopping threshold (deg)  default 0.1
                   Smaller than threshold_sac so gate_res ≈ 1 at onset.
                   Saccade endpoint accuracy ~0.1–0.2° (Becker 1989).
    k_sac          sigmoid steepness (1/deg)          default 50.0
    e_sat_sac      main-sequence saturation (deg)     default 7.0
                   Peak velocity plateaus beyond ~10–15° (Bahill et al. 1975).
    tau_i          NI leak TC — shared with NI (s)    default 25.0
    tau_reset_sac  reset TC during burst (s)          default 1.0
    tau_reset_fast reset TC after burst ends (s)      default 0.1
    tau_ref        refractory period (s)              default 0.15
                   Saccadic refractory period ~150–200 ms (Fischer &
                   Ramsperger 1984 Exp Brain Res).
    tau_ref_charge refractory charge TC (s)           default 0.005
                   How fast z_ref rises after the burst ends (~5 ms).
"""

import jax.numpy as jnp
import jax

N_STATES  = 4   # x_reset_int (3,) + z_ref (1,) refractory timer
N_INPUTS  = 3   # e_pos_delayed (3,)
N_OUTPUTS = 3   # u_burst (3,)


# ── Input nonlinearity ────────────────────────────────────────────────────────

def burst_nonlinearity(e_pos_delayed, e_residual, theta):
    """Compute burst vector and the two scalar error gates.

    Burst MAGNITUDE saturates on |e_residual| → bell-shaped velocity profile.
    At saccade onset x_sg=0, so |e_residual|=|e_pos_delayed| → main sequence.
    Burst DIRECTION follows e_residual → Robinson stopping criterion.

    Args:
        e_pos_delayed: (3,)  delayed retinal position error
        e_residual:    (3,)  e_pos_delayed − x_reset_int  (Robinson residual)
        theta:         dict  model parameters

    Returns:
        u_burst_raw: (3,)    burst vector, gates NOT yet applied
        gate_err:    scalar  high when retinal error > threshold_sac
        gate_res:    scalar  high while Robinson residual > threshold_stop
    """
    g_burst        = theta.get('g_burst',         700.0)
    threshold_sac  = theta.get('threshold_sac',     0.5)
    threshold_stop = theta.get('threshold_stop',    0.1)
    k_sac          = theta.get('k_sac',            50.0)
    e_sat_sac      = theta.get('e_sat_sac',         7.0)

    e_err_mag = jnp.linalg.norm(e_pos_delayed)
    e_res_mag = jnp.linalg.norm(e_residual)

    gate_err = jax.nn.sigmoid(k_sac * (e_err_mag - threshold_sac))    # trigger: error above dead-zone
    gate_res = jax.nn.sigmoid(k_sac * (e_res_mag - threshold_stop))   # stop: residual above precision limit

    # Magnitude from RESIDUAL → natural bell-shaped velocity profile.
    # At onset (x_sg=0): e_residual = e_pos_delayed → correct main sequence.
    # Tapers to zero as copy catches up → implicit stopping, no extra shaping.
    burst_mag   = g_burst * (1.0 - jnp.exp(-e_res_mag / e_sat_sac))
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
        x_sg:  (N_STATES,)  [x_reset_int (3,) | z_ref (1,)]
                             x_reset_int — copy integrator (position)
                             z_ref       — refractory timer (0=ready, 1=refractory)
        u_sg:  (N_INPUTS,)  e_pos_delayed (3,) — delayed retinal position error
        theta: dict         model parameters

    Returns:
        dx_sg:   (N_STATES,)  state derivative
        u_burst: (3,)         saccade velocity command (deg/s)
    """
    x_pos = x_sg[:3]   # (3,) resettable position copy integrator
    z_ref = x_sg[3]    # scalar refractory state

    e_residual = u_sg - x_pos   # Robinson residual

    u_burst_raw, gate_err, gate_res, gate_dir = burst_nonlinearity(u_sg, e_residual, theta)

    # OPN gate: z_ref suppresses the error trigger (gate_err) during the refractory
    # period — analogous to omnipause neurons (OPNs) inhibiting burst neurons.
    # While z_ref ≈ 1 the error signal cannot drive a saccade, regardless of its
    # magnitude.  gate_res and gate_dir still act as stopping / direction checks.
    gate_err_eff = gate_err * (1.0 - z_ref)
    gate_active  = gate_err_eff * gate_res * gate_dir

    # Adaptive reset TC: slow while burst is active, fast otherwise.
    tau_reset = theta.get('tau_reset_sac',  1.0)
    tau_fast  = theta.get('tau_reset_fast', 0.05)
    tau_eff   = tau_fast + gate_active * (tau_reset - tau_fast)
    A_reset_eff = (-1.0 / tau_eff) * jnp.eye(3)

    dx_pos = (        gate_active  * (get_A_ni(theta) @ x_pos + B @ u_burst_raw)
             + (1.0 - gate_active) * (A_reset_eff     @ x_pos))

    u_burst = gate_active * u_burst_raw

    # Refractory dynamics
    # charge signal: gate_err*(1−gate_res)  high only post-saccade when error
    # is still large but the residual has dropped (burst just finished).
    # This is zero during saccades (gate_res≈1) and zero at rest (gate_err≈0).
    tau_ref_charge = theta.get('tau_ref_charge', 0.005)   # fast rise ~5 ms
    tau_ref        = theta.get('tau_ref',         0.15)    # refractory decay ~150 ms
    charge         = gate_err * (1.0 - gate_res)
    dz_ref         = (1.0 - z_ref) * charge / tau_ref_charge  -  z_ref / tau_ref

    dx_sg = jnp.concatenate([dx_pos, jnp.array([dz_ref])])
    return dx_sg, u_burst
