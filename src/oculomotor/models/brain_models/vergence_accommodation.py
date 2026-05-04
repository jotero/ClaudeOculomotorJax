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

# Vergence x_copy decays slowly between saccades so accumulated burst
# contribution doesn't drift forever.
_TAU_COPY_RESET = 30.0   # s


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
    # SVBN — H-only saccade-gated saturating burst (symmetric for conv and div).
    # Uses g_svbn_conv / X_svbn_conv for both directions; the _div params are
    # legacy (Zee 1992 Table 1 has asymmetric conv/div but we ignore that for now).
    disp_h    = target_disparity[_AXIS_H]
    u_svbn_h  = (z_act * jnp.sign(disp_h) * brain_params.g_svbn_conv
                 * (1.0 - jnp.exp(-jnp.abs(disp_h) / brain_params.X_svbn_conv)))
    u_svbn    = jnp.zeros(3).at[_AXIS_H].set(u_svbn_h)    # H-only burst vector

    # x_copy: integrated SVBN burst for observability; slow decay between saccades
    dx_v_copy = u_svbn - x_verg_copy / _TAU_COPY_RESET
    
    # ── Collect all vergence drive (disparity + TVOR + burst) ───────────────
    # Disparity goes into the fast integrator AS IS (no K) and into the direct
    # bypass pathway with unit gain (Kb=1, matching NI's pure pass-through).
    # TVOR rate and SVBN burst go to BOTH the integrator and the direct pathway.

    # Fast leaky integrator (Schor 1999): Tf·dx + x = Kf·input
    # Disparity is the position-form input (gets Kf gain). Burst and TVOR are
    # rate inputs added directly.
    dx_v_fast = ((brain_params.K_verg_fast * target_disparity - x_verg_fast)
                 / brain_params.tau_verg_fast
                 + u_svbn + verg_rate_tvor)

    # Direct (bypass) pathway with plant compensation via brain.tau_p (NI form).
    # Disparity gets unit Kb (NI-matched pass-through); rate drives × τ_p.
    direct_path_pos = (target_disparity
                       + brain_params.tau_p * (u_svbn + verg_rate_tvor))   # deg
    fast_pathway_out = x_verg_fast + direct_path_pos                       # deg

    # Cross-link: AC/A from accommodation (H-only, deg)
    tonic_vec = jnp.zeros(3).at[_AXIS_H].set(brain_params.tonic_verg)
    aca_vec   = jnp.zeros(3).at[_AXIS_H].set(aca_drive)

    # Adaptable tonic (Schor 1999): leaky integrator with setpoint at tonic_verg,
    # driven by Ks · (fast pathway output + AC/A cross-link).
    tonic_input = brain_params.K_verg_slow * (fast_pathway_out + aca_vec)
    dx_v_slow = (tonic_vec + tonic_input - x_verg_slow) / brain_params.tau_verg_slow


    # Output: sum of everything — direct (phasic) + fast + slow + cross-link.
    # x_verg_slow already includes the tonic_verg setpoint baseline, so at rest
    # u_verg = 0 + 0 + tonic_vec + 0 = tonic_vec.
    u_verg = direct_path_pos + x_verg_fast + x_verg_slow + aca_vec

    # ── Accommodation (mirror of vergence — no burst, no rate inputs) ───────
    # Defocus goes into the fast integrator AS IS (no K) and into the direct
    # pathway with unit Kb.  No rate inputs to scale by τ_acc_plant.

    # Fast leaky integrator (Schor 1999): Tf·dx + x = Kf·input
    dx_a_fast = (brain_params.K_acc_fast * defocus - x_acc_fast) / brain_params.tau_acc_fast

    # Direct (bypass) pathway — defocus with unit Kb (no rate inputs)
    direct_path_pos_acc  = defocus                                       # D
    fast_pathway_out_acc = x_acc_fast + direct_path_pos_acc              # D

    # Adaptable tonic (Schor 1999): leaky integrator with setpoint at tonic_acc,
    # driven by Ks · (fast pathway output + CA/C cross-link).
    tonic_input_acc  = brain_params.K_acc_slow * (fast_pathway_out_acc + cac_drive)
    dx_a_slow        = (brain_params.tonic_acc + tonic_input_acc - x_acc_slow) / brain_params.tau_acc_slow

    # Output: sum of everything — direct (phasic) + fast + slow + cross-link.
    # x_acc_slow already includes the tonic_acc setpoint baseline.
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
