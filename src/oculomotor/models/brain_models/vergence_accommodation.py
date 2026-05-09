"""Vergence + accommodation — single SSM module: near-response system.

Two coupled controllers managing depth-of-fixation. Cross-coupled by:

  AC/A (accommodative convergence) — accommodation drives vergence:
                                     blur → acc → AC/A → u_verg
  CA/C (convergence accommodation) — vergence drives accommodation:
                                     disparity → vg → CA/C → u_acc

Both subsystems share the same fast + slow integrator pattern; vergence has
the extra x_copy slot to track the saccadic vergence (SVBN) burst, plus a
Robinson direct phasic path on top of the integrators.

State layout (relative to x_va, total 11 states):

    x_verg (9,)   _IDX_VERG  [0:9]   — x_fast(3) | x_slow(3) | x_copy(3)
    x_acc  (2,)   _IDX_ACC   [9:11]  — x_fast    | x_slow            (scalar)

Bundled inputs to step() (length N_INPUTS=8):

    defocus            (1,)   _IDX_INPUT_DEFOCUS         — delayed cyclopean defocus (D)
    target_disparity   (3,)   _IDX_INPUT_TARGET_DISP     — delayed binocular disparity (deg)
    verg_rate_tvor     (3,)   _IDX_INPUT_VERG_RATE_TVOR  — T-VOR vergence rate (deg/s)
    z_act              (1,)   _IDX_INPUT_Z_ACT           — OPN gate (SVBN trigger)

Outputs from step():

    dx_va    (11,)   ODE derivative of the combined state
    u_verg   (3,)    vergence position command (deg) → FCP per-eye split
    u_acc    scalar  total lens-plant input (D) — neural + CA/C feedforward

Sequencing (cross-couplings read prior-step state — synaptic-delay realistic):
  1. Read state  → u_neural_acc (= acc fast+slow+tonic) and x_verg_yaw.
  2. Compute AC/A and CA/C from those state values.
  3. Step vergence       (uses aca_drive, disparity, TVOR, OPN gate).
  4. Step accommodation  (uses defocus + A_cac applied to lens-plant input).

References:
    Schor & Kotulak (1986) Vision Res 26:927   — dual-controller framework
    Read et al. (2022) J Vision 22(9):4         — modern dual-loop accommodation
    Robinson (1975)                              — pulse-step / direct path
    Rashbass & Westheimer (1961) J Physiol 159:339 — fast vergence onset
    Zee et al. (1992) J Neurophysiol 68:1624    — saccadic vergence burst (SVBN)
    Schor (1979) Vision Res 19:1359             — slow tonic adapter
    Schor & Bharadwaj (2006) JN 95:3459         — lens plant TC
    Hofstetter, Hung & Semmlow                   — clinical AC/A and CA/C ratios
"""

from typing import NamedTuple

import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────────
# Module-wide constants
# ─────────────────────────────────────────────────────────────────────────────

# Unit glossary:
#   D   = optical diopters (1/m)        — defocus and accommodation
#   pd  = prism diopters (≈ 0.5729 deg) — vergence magnitude
#   deg = degrees                       — eye rotation / vergence angle
#
# AC/A ratio is in pd/D  →  × _DEG_PER_PD × accommodation [D] = deg
# CA/C ratio is in D/pd  →  vergence [deg] / _DEG_PER_PD = pd  → × CA_C [D/pd] = D
_DEG_PER_PD = 0.5729   # 1 pd = arctan(0.01) rad ≈ 0.5729 deg

# Vergence x_copy is the local-feedback efference of the saccadic vergence
# burst (Robinson-style): integrates the SVBN output and is SUBTRACTED from
# disparity so the burst self-terminates as the saccadic vergence command
# accumulates. The integration is gated by the saccade gate z_act:
#   z_act ≈ 1 (during saccade): perfect integration (no leak)
#   z_act ≈ 0 (between saccades): fast leak (TC = _TAU_COPY_RESET)
# This way x_copy holds the accumulated burst command during the saccade,
# then drains quickly afterward.
_TAU_COPY_RESET = 0.02   # s — 20 ms drain between saccades


# ─────────────────────────────────────────────────────────────────────────────
# State + input layout
# ─────────────────────────────────────────────────────────────────────────────

# Axis layout for 3-vectors in this module (matches whole-codebase convention)
_AXIS_H, _AXIS_V, _AXIS_T = 0, 1, 2   # [yaw, pitch, roll] = [H, V, T]

# Vergence sub-state slices (within x_verg = x_va[0:9]):
#   _VG_IDX_VERG  — vergence integrator (was x_verg_fast)
#   _VG_IDX_TONIC — tonic vergence integrator (was x_verg_slow)
#   _VG_IDX_COPY  — saccadic-vergence efference copy (vestigial, kept for state layout)
_VG_IDX_VERG  = slice(0, 3)
_VG_IDX_TONIC = slice(3, 6)
_VG_IDX_COPY  = slice(6, 9)

# Accommodation sub-state slices (within x_acc = x_va[9:11]) — fast/slow each scalar
_ACC_IDX_FAST = 0
_ACC_IDX_SLOW = 1

# ── State + registries ────────────────────────────────────────────────────────

class State(NamedTuple):
    """Vergence + Accommodation state — five firing-rate populations."""
    verg_fast:  jnp.ndarray   # (3,)    phasic vergence neurons   [supraoculomotor area]
    verg_tonic: jnp.ndarray   # (3,)    tonic vergence neurons    [supraoculomotor area]
    verg_copy:  jnp.ndarray   # (3,)    saccadic-vergence EC pop  [supraoculomotor area; vestigial]
    acc_fast:   jnp.ndarray   # scalar  fast accommodation pop    [NRTP / EW]
    acc_slow:   jnp.ndarray   # scalar  slow accommodation pop    [NRTP / EW]


# State == Activations: all fields are firing rates.
Activations = State


def rest_state():
    """Zero state — used for SimState initialisation."""
    return State(
        verg_fast  = jnp.zeros(3),
        verg_tonic = jnp.zeros(3),
        verg_copy  = jnp.zeros(3),
        acc_fast   = jnp.float32(0.0),
        acc_slow   = jnp.float32(0.0),
    )


def read_activations(state):
    """All VA fields are firing rates — identity projection."""
    return state

# Bundled-input layout
N_INPUTS  = 1 + 3 + 3 + 1   # 8
N_OUTPUTS = 3   # primary output for SSM convention is u_verg; auxiliaries via tuple

_IDX_INPUT_DEFOCUS        = 0
_IDX_INPUT_TARGET_DISP    = slice(1, 4)
_IDX_INPUT_VERG_RATE_TVOR = slice(4, 7)
_IDX_INPUT_Z_ACT          = 7


# ─────────────────────────────────────────────────────────────────────────────
# step() — entire near-response math, inlined
# ─────────────────────────────────────────────────────────────────────────────

def step(state, u, brain_params):
    """Single ODE step for the unified near-response (vergence + accommodation).

    Args:
        state:        va.State    bundled state (5 firing-rate fields)
        u:            (8,)  bundled input — see _IDX_INPUT_* above
        brain_params: BrainParams

    Returns:
        dstate : va.State  state derivative
        u_verg : (3,)      vergence position command (deg) → FCP per-eye split
        u_acc  : scalar    total lens-plant input (D) — neural + CA/C
    """
    x_verg_v     = state.verg_fast
    x_verg_tonic = state.verg_tonic
    x_verg_copy  = state.verg_copy
    x_acc_fast   = state.acc_fast
    x_acc_slow   = state.acc_slow

    # ── Split bundled inputs ────────────────────────────────────────────────
    defocus          = u[_IDX_INPUT_DEFOCUS]
    target_disparity = u[_IDX_INPUT_TARGET_DISP]
    verg_rate_tvor   = u[_IDX_INPUT_VERG_RATE_TVOR]
    z_act            = u[_IDX_INPUT_Z_ACT]

    # Robinson local feedback for the SVBN burst only: subtract x_verg_copy
    # (integrated burst command) from disparity so the burst self-terminates.
    # The vergence integrator (x_verg_v) below uses RAW target_disparity —
    # the visual feedback loop already attenuates target_disparity as the eye
    # converges, so the integrator doesn't need its own residual.
    burst_residual_disparity = target_disparity - x_verg_copy

    # ── Cross-couplings ─────────────────────────────────────────────────────
    # Both cross-links use TOTAL state (fast + slow / fast + tonic), not just
    # the phasic component. Clinically AC/A and CA/C ratios are measured
    # against sustained responses; using the total state lets dark conditions
    # (where each subsystem settles near its motor tonic) drive a sustained
    # cross-link contribution, separating them from monocular/binocular
    # conditions where the totals track stimulus demand.
    aca_drive = brain_params.AC_A * _DEG_PER_PD * (x_acc_fast + x_acc_slow)
    cac_drive = brain_params.CA_C * ((x_verg_v[_AXIS_H] + x_verg_tonic[_AXIS_H]) / _DEG_PER_PD)

    # ── Vergence ────────────────────────────────────────────────────────────
    # SVBN — saccade-gated saturating burst, applied per-axis (H, V, T).
    # Uses g_svbn_conv / X_svbn_conv symmetrically for both directions on each
    # axis; the _div params are legacy (Zee 1992 Table 1 has asymmetric conv/div
    # but we ignore that for now). Driven by the BURST residual (target_disparity
    # − x_verg_copy) so the burst self-terminates via x_verg_copy filling in.
    u_svbn = (z_act * jnp.sign(burst_residual_disparity) * brain_params.g_svbn_conv
              * (1.0 - jnp.exp(-jnp.abs(burst_residual_disparity) / brain_params.X_svbn_conv)))

    # x_verg_copy: integrated SVBN burst with z_act-gated leak.
    # During saccade (z_act ≈ 1): perfect integration (no leak).
    # Between saccades (z_act ≈ 0): fast leak (TC = _TAU_COPY_RESET = 20 ms).
    dx_v_copy = u_svbn - (1.0 - z_act) * x_verg_copy / _TAU_COPY_RESET

    # ── Collect all vergence drive (raw target disparity + TVOR + burst) ────
    # The vergence integrator x_verg_v sees RAW target_disparity (closed-loop
    # via visual feedback) plus the burst velocity, not the burst-residual
    # disparity. This avoids double-counting the burst contribution.
    verg_drive = target_disparity + u_svbn + verg_rate_tvor

    # Vergence integrator (leaky)
    dx_v = brain_params.K_verg * verg_drive - x_verg_v / brain_params.tau_verg

    # Direct (bypass) pathway: plant-compensation form (= NI's u_p = x + τ_p·u_vel)
    direct_path_pos = brain_params.tau_p * verg_drive

    # Cross-link: AC/A from accommodation (H-only, deg)
    tonic_setpoint = jnp.zeros(3).at[_AXIS_H].set(brain_params.tonic_verg)
    aca_vec        = jnp.zeros(3).at[_AXIS_H].set(aca_drive)

    # Tonic vergence integrator — adaptable tonic, leaky integrator with setpoint
    # tonic_verg, driven by K_verg_tonic · (vergence integrator out + direct
    # pathway + AC/A cross-link).
    verg_pathway_out = x_verg_v + direct_path_pos
    tonic_input  = brain_params.K_verg_tonic * (verg_pathway_out + aca_vec)
    dx_v_tonic   = (tonic_setpoint + tonic_input - x_verg_tonic) / brain_params.tau_verg_tonic

    # Output: direct + vergence + tonic + cross-link. x_verg_tonic carries
    # the tonic_verg setpoint baseline.
    u_verg = direct_path_pos + x_verg_v + x_verg_tonic + aca_vec

    # ── Accommodation (mirror of vergence — only defocus drive) ─────────────
    # Same Schor block-diagram form. Direct pathway scales by τ_acc_plant
    # (the lens-plant TC, analog of τ_p for the eye plant on the vergence side).
    acc_drive = defocus

    dx_a_fast = brain_params.K_acc_fast * acc_drive - x_acc_fast / brain_params.tau_acc_fast

    direct_path_pos_acc  = brain_params.tau_acc_plant * acc_drive

    fast_pathway_out_acc = x_acc_fast + direct_path_pos_acc
    tonic_input_acc      = brain_params.K_acc_slow * (fast_pathway_out_acc + cac_drive)
    dx_a_slow            = (brain_params.tonic_acc + tonic_input_acc - x_acc_slow) / brain_params.tau_acc_slow

    u_acc = direct_path_pos_acc + x_acc_fast + x_acc_slow + cac_drive

    # ── Proximal-vergence and proximal-accommodation constant injection ─────
    # Single perceived-distance parameter `proximal_d` (D). Injected as a
    # state-INDEPENDENT drive (Hung & Semmlow 1980 style — proximal vergence is
    # a constant additive contribution, not a target-attractor). This pushes
    # vergence and accommodation in the same direction regardless of current
    # state, preserving symmetric phase between conv-fixation and div-fixation
    # configurations, matching the empirical phenomenology.
    _DEG_PER_RAD = 57.295779
    proximal_d              = brain_params.proximal_d
    proximal_verg_target_H  = proximal_d * brain_params.ipd_brain * _DEG_PER_RAD
    proximal_acc_target     = proximal_d
    dx_v       = dx_v.at[_AXIS_H].add(proximal_verg_target_H / brain_params.tau_verg)
    dx_a_fast  = dx_a_fast + proximal_acc_target / brain_params.tau_acc_fast

    # ── Pack ────────────────────────────────────────────────────────────────
    dstate = State(
        verg_fast  = dx_v,
        verg_tonic = dx_v_tonic,
        verg_copy  = dx_v_copy,
        acc_fast   = dx_a_fast,
        acc_slow   = dx_a_slow,
    )
    return dstate, u_verg, u_acc


# ── Legacy flat-array adapters (deleted once brain_model migrates to BrainState) ─

N_STATES  = 9 + 2   # 11
_IDX_VERG = slice(0, 9)
_IDX_ACC  = slice(9, 11)


def from_array(x_va):
    """(11,) flat array → va.State."""
    return State(
        verg_fast  = x_va[0:3],
        verg_tonic = x_va[3:6],
        verg_copy  = x_va[6:9],
        acc_fast   = x_va[9],
        acc_slow   = x_va[10],
    )


def to_array(state):
    """va.State → (11,) flat array."""
    return jnp.concatenate([
        state.verg_fast, state.verg_tonic, state.verg_copy,
        jnp.array([state.acc_fast, state.acc_slow]),
    ])


__all__ = [
    "step", "State", "Activations", "rest_state", "read_activations",
    "from_array", "to_array",
    "N_STATES", "N_INPUTS", "N_OUTPUTS",
    "_IDX_VERG", "_IDX_ACC",
    "_IDX_INPUT_DEFOCUS", "_IDX_INPUT_TARGET_DISP", "_IDX_INPUT_VERG_RATE_TVOR",
    "_IDX_INPUT_Z_ACT",
]
