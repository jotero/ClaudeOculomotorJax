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

    # ── Cross-couplings from current state ──────────────────────────────────
    # Brain's current accommodation neural output = fast + slow + tonic (D)
    u_neural_acc = x_acc_fast + x_acc_slow + brain_params.tonic_acc
    # Current vergence yaw = tonic + slow integrator H component (deg)
    x_verg_yaw   = brain_params.tonic_verg + x_verg_slow[_AXIS_H]

    # AC/A: accommodation increment (D above dark focus) → vergence drive (deg)
    aca_drive = brain_params.AC_A * _DEG_PER_PD * (u_neural_acc - brain_params.tonic_acc)
    # CA/C: vergence increment (deg above tonic)        → lens-plant feedforward (D)
    A_cac     = brain_params.CA_C * ((x_verg_yaw - brain_params.tonic_verg) / _DEG_PER_PD)

    # ── Vergence ────────────────────────────────────────────────────────────
    # Zee (1992) SVBN — H-only saccade-gated asymmetric saturating burst.
    # Conv much stronger than div (Table 1: peak 50°/s for 10° conv vs 12°/s for 2.5° div).
    disp_h    = target_disparity[_AXIS_H]
    is_conv   = (disp_h > 0).astype(jnp.float32)
    g_eff     = is_conv * brain_params.g_svbn_conv + (1.0 - is_conv) * brain_params.g_svbn_div
    X_eff     = is_conv * brain_params.X_svbn_conv + (1.0 - is_conv) * brain_params.X_svbn_div
    u_svbn_h  = z_act * jnp.sign(disp_h) * g_eff * (1.0 - jnp.exp(-jnp.abs(disp_h) / X_eff))
    u_svbn    = jnp.zeros(3).at[_AXIS_H].set(u_svbn_h)    # H-only burst vector

    # Dual integrators — same form as accommodation, with extras:
    #   fast: + SVBN burst boosts during saccade for persistence
    #   slow: + TVOR vergence rate (open-loop integration during head translation)
    dx_v_fast = -x_verg_fast / brain_params.tau_verg_fastn + brain_params.K_verg_fast * target_disparity + u_svbn
    dx_v_slow = -x_verg_slow / brain_params.tau_verg_slow  + brain_params.K_verg_slow * target_disparity + verg_rate_tvor
    # x_copy: integrated SVBN burst for observability; slow decay between saccades
    dx_v_copy = u_svbn - x_verg_copy / _TAU_COPY_RESET

    # Output: tonic baseline + AC/A direct + integrators + Robinson direct phasic + SVBN feedthrough.
    # AC/A enters as a direct add (unity gain) so open-loop conditions don't amplify
    # it through the high-gain integrator stack and run vergence to the orbital walls.
    u_phasic  = brain_params.K_phasic_verg * target_disparity            # deg/s
    tonic_vec = jnp.zeros(3).at[_AXIS_H].set(brain_params.tonic_verg)    # H-only tonic
    aca_vec   = jnp.zeros(3).at[_AXIS_H].set(aca_drive)                  # H-only AC/A
    u_verg    = (tonic_vec + aca_vec
                 + x_verg_fast + x_verg_slow
                 + brain_params.tau_vp * (u_phasic + u_svbn))

    # ── Accommodation ───────────────────────────────────────────────────────
    # Dual integrators on residual blur. CA/C is added to the lens-plant input
    # directly (bypasses the blur controller, so x_fast/x_slow only see the
    # residual defocus from retina once x_plant has caught up).
    e_blur    = defocus
    dx_a_fast = -x_acc_fast / brain_params.tau_acc_fast + brain_params.K_acc_fast * e_blur
    dx_a_slow = -x_acc_slow / brain_params.tau_acc_slow + brain_params.K_acc_slow * e_blur
    u_acc     = (x_acc_fast + x_acc_slow + brain_params.tonic_acc) + A_cac

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
