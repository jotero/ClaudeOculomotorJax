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
# accumulates. Fast reset between saccades.
_TAU_COPY_RESET = 0.2    # s


# ─────────────────────────────────────────────────────────────────────────────
# State + input layout
# ─────────────────────────────────────────────────────────────────────────────

# Axis layout for 3-vectors in this module (matches whole-codebase convention)
_AXIS_H, _AXIS_V, _AXIS_T = 0, 1, 2   # [yaw, pitch, roll] = [H, V, T]

# Vergence sub-state slices (within x_verg = x_va[0:9]) — fast/slow/copy each (3,)
_VG_IDX_FAST = slice(0, 3)
_VG_IDX_SLOW = slice(3, 6)
_VG_IDX_COPY = slice(6, 9)

# Accommodation sub-state slices (within x_acc = x_va[9:11]) — fast/slow each scalar
_ACC_IDX_FAST = 0
_ACC_IDX_SLOW = 1

# Top-level state
N_STATES  = 9 + 2   # 11
_IDX_VERG = slice(0, 9)
_IDX_ACC  = slice(9, 11)

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

def step(x_va, u, brain_params):
    """Single ODE step for the unified near-response (vergence + accommodation).

    Args:
        x_va:         (11,) bundled state — see _IDX_VERG / _IDX_ACC
        u:            (8,)  bundled input — see _IDX_INPUT_* above
        brain_params: BrainParams

    Returns:
        dx_va  : (11,) state derivative
        u_verg : (3,)  vergence position command (deg) → FCP per-eye split
        u_acc  : scalar total lens-plant input (D) — neural + CA/C
    """
    # ── Split state into parallel sub-states for both subsystems ────────────
    x_verg = x_va[_IDX_VERG]
    x_acc  = x_va[_IDX_ACC]

    x_verg_fast = x_verg[_VG_IDX_FAST]   # (3,)
    x_verg_slow = x_verg[_VG_IDX_SLOW]   # (3,)
    x_verg_copy = x_verg[_VG_IDX_COPY]   # (3,)
    x_acc_fast  = x_acc[_ACC_IDX_FAST]   # scalar
    x_acc_slow  = x_acc[_ACC_IDX_SLOW]   # scalar

    # ── Split bundled inputs ────────────────────────────────────────────────
    defocus          = u[_IDX_INPUT_DEFOCUS]
    target_disparity = u[_IDX_INPUT_TARGET_DISP]
    verg_rate_tvor   = u[_IDX_INPUT_VERG_RATE_TVOR]
    z_act            = u[_IDX_INPUT_Z_ACT]

    # Robinson local feedback for vergence: subtract the integrated saccadic
    # vergence command (x_verg_copy) from the raw disparity so both the SVBN
    # burst and the rest of the vergence pathway see the RESIDUAL TARGET
    # DISPARITY — i.e. what's left after the saccade command has been issued.
    # With τ_copy ≈ 0.2 s, x_verg_copy decays quickly between saccades so the
    # slow loop sees raw disparity once the saccade is over.
    residual_target_disparity = target_disparity - x_verg_copy

    # ── Cross-couplings (Schor 1999: PHASIC-only, no tonic contribution) ────
    # Schor 1999 p.3: "tonic accommodation and tonic convergence have been shown
    # not to stimulate accommodative vergence and vergence accommodation under
    # open-loop conditions". So both cross-links read only the fast (phasic)
    # state — once the response transfers to the tonic, the cross-link drops out.
    # AC/A: accommodation phasic (D above dark focus) → vergence drive (deg)
    aca_drive = brain_params.AC_A * _DEG_PER_PD * x_acc_fast
    # CA/C: vergence phasic (deg above tonic, H only) → lens-plant feedforward (D)
    cac_drive = brain_params.CA_C * (x_verg_fast[_AXIS_H] / _DEG_PER_PD)

    # ── Vergence ────────────────────────────────────────────────────────────
    # SVBN — saccade-gated saturating burst, applied per-axis (H, V, T).
    # Uses g_svbn_conv / X_svbn_conv symmetrically for both directions on each
    # axis; the _div params are legacy (Zee 1992 Table 1 has asymmetric conv/div
    # but we ignore that for now). Driven by the RESIDUAL disparity, so the
    # burst shrinks as x_verg_copy fills in (Robinson local feedback).
    u_svbn = (z_act * jnp.sign(residual_target_disparity) * brain_params.g_svbn_conv
              * (1.0 - jnp.exp(-jnp.abs(residual_target_disparity) / brain_params.X_svbn_conv)))

    # x_copy: integrated SVBN burst — fast reset (τ_copy = 0.2 s) so it drains
    # between saccades and only shapes the burst self-termination.
    dx_v_copy = u_svbn - x_verg_copy / _TAU_COPY_RESET

    # ── Collect all vergence drive (residual disparity + TVOR + burst) ──────
    # Schor-1999 block-diagram form: Kf and τ_p apply uniformly to the combined
    # drive vector. Units mix (disparity in deg, burst/tvor in deg/s) — that's
    # accepted: the integrator equation `dx = K·u - x/τ` is unit-agnostic when K
    # is treated as a tunable gain, and a leaky integrator can equivalently be
    # read as "integrate the rate" or "track the setpoint" depending on viewpoint.
    verg_drive = residual_target_disparity + u_svbn + verg_rate_tvor

    # Fast leaky integrator
    dx_v_fast = brain_params.K_verg_fast * verg_drive - x_verg_fast / brain_params.tau_verg_fast

    # Direct (bypass) pathway: plant-compensation form (= NI's u_p = x + τ_p·u_vel)
    direct_path_pos = brain_params.tau_p * verg_drive

    # Cross-link: AC/A from accommodation (H-only, deg)
    tonic_vec = jnp.zeros(3).at[_AXIS_H].set(brain_params.tonic_verg)
    aca_vec   = jnp.zeros(3).at[_AXIS_H].set(aca_drive)

    # Adaptable tonic — leaky integrator with setpoint tonic_verg, driven by
    # Ks · (fast integrator out + direct pathway + AC/A cross-link).
    fast_pathway_out = x_verg_fast + direct_path_pos
    tonic_input  = brain_params.K_verg_slow * (fast_pathway_out + aca_vec)
    dx_v_slow    = (tonic_vec + tonic_input - x_verg_slow) / brain_params.tau_verg_slow

    # Output: direct + fast + slow + cross-link. x_verg_slow already carries
    # the tonic_verg setpoint baseline.
    u_verg = direct_path_pos + x_verg_fast + x_verg_slow + aca_vec

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

    # ── Pack ────────────────────────────────────────────────────────────────
    dx_va = jnp.concatenate([
        dx_v_fast, dx_v_slow, dx_v_copy,
        jnp.array([dx_a_fast, dx_a_slow]),
    ])
    return dx_va, u_verg, u_acc


__all__ = [
    "step",
    "N_STATES", "N_INPUTS", "N_OUTPUTS",
    "_IDX_VERG", "_IDX_ACC",
    "_IDX_INPUT_DEFOCUS", "_IDX_INPUT_TARGET_DISP", "_IDX_INPUT_VERG_RATE_TVOR",
    "_IDX_INPUT_Z_ACT",
]
