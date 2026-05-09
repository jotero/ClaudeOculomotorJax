"""Unified brain model — state-space replacement for brain_model.step.

Drop-in replacement: same I/O signature as brain_model.step. Implements the
unified continuous-control template developed in
manuscript/unified_oculomotor_template.md.

Architecture:

    [g] Bayesian-MAP preprocessing of raw sensory input (incl. canal saturation)
    [unified template]        the matrix-form core (pursuit, bilateral NI,
                              vergence, accommodation, AC/A, CA/C, velocity
                              storage, gravity estimator, heading estimator)
                              is one set of matrices A, B, C, D, E, F, G, M_fs,
                              M_ss, T plus the bilinear correction f(x, u):

            dx_fast/dt = A·x_fast + G_fs·x_slow + B·u' + b_prox + f(x, u')
            dx_slow/dt = C·(x_slow - T) + M_ss·x_slow + D·x_fast + B_s·u'
            y_motor    = E·x_fast + G·x_slow + F·u'

    [delegated]               saccade generator, T-VOR, target memory,
                              Listing's law, FCP — still wrapped as external
                              calls; their outputs feed into u_proc as
                              "delegated drives".

State partitioning (within the 41-state unified subset):

    x_fast (34) = [ x_pu (3) | x_ni_L (3) | x_ni_R (3) | x_v (3)
                   | x_v_copy (3) | x_a_fast (1)
                   | x_vs_A (3) | x_vs_B (3)
                   | x_g (3) | x_a_lin (3) | x_rf (3)
                   | x_v_lin (3) ]
    x_slow (10) = [ x_ni_null (3) | x_v_tonic (3) | x_a_slow (1)
                   | x_vs_null (3) ]

Setpoint T (slow):
    T = [ 0_NI (3) | tonic_setpoint_V (3) | tonic_acc (1) | 0_VS (3) ]
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oculomotor.models.brain_models import brain_model
from oculomotor.models.brain_models.perception_target import (
    _TAU_TARGET_MEM_UPDATE, _TAU_TARGET_MEM_TRUST_RISE,
    _TAU_TARGET_MEM_TRUST_DECAY, _TARGET_MEM_TRUST_THRESHOLD,
)
# `_TAU_TARGET_MEM_CONSUME` was the old constant name for the activity-driven
# memory drain; perception_target.py drives drain by |ec_d_target| now.  Keep
# a placeholder constant so this file (used only by bench_compare_unified) loads.
_TAU_TARGET_MEM_CONSUME = 0.05
from oculomotor.models.brain_models import (
    final_common_pathway as fcp,
)
from oculomotor.models.brain_models.saccade_generator import burst_velocity as _burst_velocity
from oculomotor.models.brain_models.perception_self_motion import (
    G0, _B_NOMINAL, _TAU_RF_STATE,
)
from oculomotor.models.brain_models.tvor import (
    _NPC_GATE_CENTER_FRAC, _NPC_GATE_SHARPNESS_FRAC, _DISTANCE_EPSILON,
)
from oculomotor.models.sensory_models.sensory_model import PINV_SENS
from oculomotor.models.sensory_models.retina        import ypr_to_xyz, xyz_to_ypr
from oculomotor.models.plant_models.readout         import gaze_unit_vector


# ─── Constants ─────────────────────────────────────────────────────────────
_DEG_PER_PD  = 0.5729
_DEG_PER_RAD = 57.295779
_AXIS_H = 0
_TAU_COPY_RESET = 0.02


# ─── State-vector layout (62-state unified subset) ─────────────────────────
N_FAST          = 52
_F_PU           = slice( 0,  3)   # pursuit
_F_NI_L         = slice( 3,  6)   # NI left  pop
_F_NI_R         = slice( 6,  9)   # NI right pop
_F_V            = slice( 9, 12)   # vergence fast
_F_V_COPY       = slice(12, 15)   # vergence copy
_F_A_FAST       = 15              # accommodation fast (scalar)
_F_VS_A         = slice(16, 19)   # VS population A
_F_VS_B         = slice(19, 22)   # VS population B
_F_GRAV_G       = slice(22, 25)   # gravity estimate
_F_GRAV_A       = slice(25, 28)   # linear-acc estimate
_F_GRAV_RF      = slice(28, 31)   # rotational feedback (Laurens)
_F_HEAD         = slice(31, 34)   # head linear velocity v_lin
# Saccade generator (18 states) — predominantly bilinear/sigmoid-gated dynamics in f
_F_SG_E_HELD    = slice(34, 37)   # Robinson resettable integrator (held retinal error)
_F_SG_Z_OPN     = 37              # OPN membrane potential (100 = tonic, ~0 = paused)
_F_SG_Z_ACC     = 38              # rise-to-bound accumulator
_F_SG_Z_TRIG    = 39              # intermediate trigger (sigmoid-charged from z_acc)
_F_SG_X_EBN_R   = slice(40, 43)   # right EBN membrane potentials
_F_SG_X_EBN_L   = slice(43, 46)   # left  EBN membrane potentials
_F_SG_X_IBN_R   = slice(46, 49)   # right IBN membrane potentials
_F_SG_X_IBN_L   = slice(49, 52)   # left  IBN membrane potentials

N_SLOW          = 10
_S_NI_NULL      = slice( 0,  3)
_S_V_TONIC      = slice( 3,  6)
_S_A_SLOW       = 6
_S_VS_NULL      = slice( 7, 10)

# u_proc: bundled output of g(theta, sensory_out, delegated)
N_U             = 32
_U_SLIP         = slice( 0,  3)   # target slip → pursuit
_U_DISP         = slice( 3,  6)   # raw target disparity + SVBN → vergence
_U_DEFOC        = 6               # defocus → accommodation
_U_VEL_BRAIN    = slice( 7, 10)   # NI velocity drive (u_burst + omega_tvor + listing_corr)
_U_TONIC_NI     = slice(10, 13)   # NI tonic position offset (Listing's torsion target)
_U_VRATE_TVOR   = slice(13, 16)   # T-VOR vergence-rate drive into vergence
_U_CANAL        = slice(16, 22)   # canal afferents (clipped) → VS
_U_SCENE_SLIP   = slice(22, 25)   # scene retinal slip → VS (push-pull)
_U_GIA          = slice(25, 28)   # GIA / otolith → GE
_U_SCENE_LIN_VEL = slice(28, 31)  # scene translational flow → HE
_U_SCENE_VIS    = 31              # scene visibility (gates HE visual fusion)

# y_motor / readouts (16): motor commands + auxiliary state-derived signals
# that delegated subsystems consume (so the gains only ever live in E/F).
N_Y             = 16
_Y_U_PU         = slice( 0,  3)   # pursuit velocity command
_Y_NI           = slice( 3,  6)   # NI version output (motor_cmd_ni)
_Y_U_VERG       = slice( 6,  9)   # vergence output
_Y_U_ACC        = 9               # accommodation output (scalar)
_Y_W_EST        = slice(10, 13)   # head-velocity estimate (used by saccade gen)
_Y_OCR          = slice(13, 16)   # OCR vector (used by saccade gen + target selection)


# ─── Matrix bundle ─────────────────────────────────────────────────────────
class UnifiedMatrices(NamedTuple):
    """Unified-template matrices for the unified subset (continuous + self-motion)."""
    A:        jnp.ndarray   # (52, 52) fast state coupling — operates on x (raw membrane potentials)
    A_cross:  jnp.ndarray   # (52, 52) fast cross-couplings via activations — operates on a (firing rates)
    B:        jnp.ndarray   # (52, 32) fast input gain
    G_fs:     jnp.ndarray   # (52, 10) slow→fast direct coupling (no multiplication by A)
    C:        jnp.ndarray   # (10, 10) slow diagonal leak (only diagonal entries used in C·(x-T))
    M_ss:     jnp.ndarray   # (10, 10) slow→slow cross-couplings (NOT through (x-T))
    D:        jnp.ndarray   # (10, 52) slow driven by fast
    B_slow:   jnp.ndarray   # (10, 32) slow input gain
    T:        jnp.ndarray   # (10,)    slow setpoint
    E:        jnp.ndarray   # (10, 52) motor readout from fast
    G:        jnp.ndarray   # (10, 10) motor readout from slow
    F:        jnp.ndarray   # (10, 32) direct sensor feedthrough to motor
    b_prox:   jnp.ndarray   # (52,)    constant fast drive (proximal + VS resting bias + SG tonic)


# ─── Matrix builder ────────────────────────────────────────────────────────

def matrices(theta) -> UnifiedMatrices:
    """Build the unified-template matrices from BrainParams.

    Each subsystem's contribution is described inline. The matrices encode:
      A   : leak rates of fast integrators + fast→fast cross-couplings
            (pursuit→NI, VS_pop→net via -w_est into NI velocity input).
      B   : sensor / supplementary input gain into fast integrators.
      M_fs: fast leaks toward slow-state-derived target (NI and VS null shifts).
      C   : diagonal slow leak rates.
      M_ss: slow→slow cross-couplings (AC/A and CA/C).
      D   : slow integrators driven by fast states.
      B_s : direct sensor drive into slow integrators.
      T   : slow setpoint vector (motor priors).
      E,G : motor readout from fast / slow.
      F   : direct sensor feedthrough to motor (Robinson pulse).
      b_prox : constant fast drive (proximal-cue + VS resting bias).
    """
    # ── Pre-computations ────────────────────────────────────────────────────
    K_phi_p          = theta.K_phasic_pursuit
    pu_state_to_u_pu = 1.0 / (1.0 + K_phi_p)
    pu_slip_to_u_pu  = K_phi_p / (1.0 + K_phi_p)

    inv_tau_pu_eff = 1.0/theta.tau_pursuit + theta.K_pursuit/(1.0 + K_phi_p)

    inv_tau_i = 1.0 / (theta.tau_i * jnp.array(
        [1.0, theta.tau_i_pitch_frac, theta.tau_i_roll_frac]))

    inv_tau_v       = 1.0 / theta.tau_verg
    inv_tau_v_tonic = 1.0 / theta.tau_verg_tonic
    inv_tau_v_copy  = 1.0 / _TAU_COPY_RESET
    inv_tau_a_fast  = 1.0 / theta.tau_acc_fast
    inv_tau_a_slow  = 1.0 / theta.tau_acc_slow
    inv_tau_adapt   = 1.0 / theta.tau_ni_adapt

    inv_tau_vs = 1.0 / (theta.tau_vs * jnp.array(
        [1.0, theta.tau_vs_pitch_frac, theta.tau_vs_roll_frac]))
    inv_tau_vs_adapt = 1.0 / theta.tau_vs_adapt
    inv_tau_a_lin    = 1.0 / theta.tau_a_lin
    inv_tau_head     = 1.0 / theta.tau_head
    inv_tau_rf       = 1.0 / _TAU_RF_STATE

    K_v   = theta.K_verg
    K_t   = theta.K_verg_tonic
    K_af  = theta.K_acc_fast
    K_as  = theta.K_acc_slow
    tau_p = theta.tau_p
    tau_acc_plant = theta.tau_acc_plant

    K_vs_canal = theta.K_vs
    K_vs_vis   = theta.K_vis
    g_vor      = theta.g_vor
    g_vis      = theta.g_vis
    K_gd       = theta.K_gd
    K_grav_b   = theta.K_grav     # baseline (gating handled in f)
    K_lin_b    = theta.K_lin
    K_he_vis_b = theta.K_he_vis   # multiplier on scene_visible

    alpha = theta.AC_A * _DEG_PER_PD
    beta  = theta.CA_C / _DEG_PER_PD

    I3  = jnp.eye(3)
    DI3 = jnp.diag(inv_tau_i)
    half_inv_tau_i = 0.5 * inv_tau_i
    DI_VS = jnp.diag(inv_tau_vs)

    # VS per-population health gain: g_pop = b_vs / B_NOMINAL  (6,)
    g_pop  = jnp.asarray(theta.b_vs) / _B_NOMINAL
    g_pop_A, g_pop_B = g_pop[:3], g_pop[3:]

    vor_torsion_gain = jnp.array([1.0, 1.0, 0.5])

    # ── A : (34, 34) fast leak + within-fast cross-couplings ───────────────
    A = jnp.zeros((N_FAST, N_FAST))
    A = A.at[_F_PU,     _F_PU    ].set(-inv_tau_pu_eff * I3)
    A = A.at[_F_NI_L,   _F_NI_L  ].set(-DI3)
    A = A.at[_F_NI_R,   _F_NI_R  ].set(-DI3)
    A = A.at[_F_NI_L,   _F_PU    ].set( 0.5 * pu_state_to_u_pu * I3)
    A = A.at[_F_NI_R,   _F_PU    ].set(-0.5 * pu_state_to_u_pu * I3)
    A = A.at[_F_V,      _F_V     ].set(-inv_tau_v * I3)
    A = A.at[_F_V_COPY, _F_V_COPY].set(-inv_tau_v_copy * I3)
    A = A.at[_F_A_FAST, _F_A_FAST].set(-inv_tau_a_fast)

    # VS populations: -inv_tau_vs · x_pop (per-axis leak)
    A = A.at[_F_VS_A, _F_VS_A].set(-DI_VS)
    A = A.at[_F_VS_B, _F_VS_B].set(-DI_VS)

    # SG burst-neuron BASELINE leak (the "1" of opn_gain = 1 + g_opn_bn·act_opn).
    # The cerebellar gain modulation (g_opn_bn·act_opn·x_bn) lives in the inline
    # f-style correction below.
    inv_tau_bn  = 1.0 / theta.tau_bn
    inv_tau_sac = 1.0 / theta.tau_sac
    inv_tau_trig= 1.0 / theta.tau_trig
    inv_tau_acc_leak = 1.0 / theta.tau_acc_leak
    A = A.at[_F_SG_X_EBN_R, _F_SG_X_EBN_R].set(-inv_tau_bn * I3)
    A = A.at[_F_SG_X_EBN_L, _F_SG_X_EBN_L].set(-inv_tau_bn * I3)
    A = A.at[_F_SG_X_IBN_R, _F_SG_X_IBN_R].set(-inv_tau_bn * I3)
    A = A.at[_F_SG_X_IBN_L, _F_SG_X_IBN_L].set(-inv_tau_bn * I3)

    # OPN/trigger/accumulator linear self-leaks.
    # z_opn dynamics: dz_opn = (k_tonic·(100 - z_opn) - (z_opn+g_opn_pause)·z_trig - g_ibn_opn·ibn_total)/τ_sac
    #   Linear self-leak: -k_tonic_opn / τ_sac on z_opn
    #   Constant drive:   100 · k_tonic_opn / τ_sac in b_prox
    #   Linear coupling on z_trig (state, identity activation): -g_opn_pause/τ_sac in A
    #   Linear coupling on IBN activations (firing rate sums): -g_ibn_opn/τ_sac in A_cross
    A = A.at[_F_SG_Z_OPN, _F_SG_Z_OPN ].set(-theta.k_tonic_opn * inv_tau_sac)
    A = A.at[_F_SG_Z_OPN, _F_SG_Z_TRIG].set(-theta.g_opn_pause * inv_tau_sac)
    # z_trig baseline leak: -1/τ_trig
    A = A.at[_F_SG_Z_TRIG, _F_SG_Z_TRIG].set(-inv_tau_trig)
    # z_acc passive leak: -1/τ_acc_leak
    A = A.at[_F_SG_Z_ACC, _F_SG_Z_ACC].set(-inv_tau_acc_leak)

    # NI driven by -w_est · vor_torsion_gain (where w_est = x_VS_A - x_VS_B + D_VS·u)
    # Velocity drive into NI L is +0.5·u_vel; for u_vel = -w_est·gain, this gives:
    #   dx_NI_L  +=  -0.5·g · (x_VS_A - x_VS_B)  ; dx_NI_R  +=  +0.5·g · (...)
    Tg = jnp.diag(0.5 * vor_torsion_gain)
    A = A.at[_F_NI_L, _F_VS_A].set(-Tg)
    A = A.at[_F_NI_L, _F_VS_B].set( Tg)
    A = A.at[_F_NI_R, _F_VS_A].set( Tg)
    A = A.at[_F_NI_R, _F_VS_B].set(-Tg)

    # GE linear core (baseline gain): residual = gia - g_est - a_lin
    A = A.at[_F_GRAV_G, _F_GRAV_G].set(-K_grav_b * I3)
    A = A.at[_F_GRAV_G, _F_GRAV_A].set(-K_grav_b * I3)
    A = A.at[_F_GRAV_A, _F_GRAV_A].set((-K_lin_b - inv_tau_a_lin) * I3)
    A = A.at[_F_GRAV_A, _F_GRAV_G].set(-K_lin_b * I3)
    # rf state: -rf/tau_rf (linear part); bilinear cross(gia, -g_est)/G0² in f
    A = A.at[_F_GRAV_RF, _F_GRAV_RF].set(-inv_tau_rf * I3)

    # VS rotational feedback (Laurens): rf state drives VS pops linearly.
    # dx_VS_A += -K_gd · rf ;  dx_VS_B += +K_gd · rf
    A = A.at[_F_VS_A, _F_GRAV_RF].set(-K_gd * I3)
    A = A.at[_F_VS_B, _F_GRAV_RF].set( K_gd * I3)

    # OCR coupling (linear): ocr_z = -g_ocr · g_est[0]; NI null leak target shifts
    # by ocr_z on torsion axis. Per-pop contribution to fast NI:
    #   dx_NI_L[2] += half_inv_tau_i[2] · (-g_ocr · g_est[0])
    #   dx_NI_R[2] += -half_inv_tau_i[2] · (-g_ocr · g_est[0])
    A = A.at[_F_NI_L.start + 2, _F_GRAV_G.start + 0].set(-theta.g_ocr * half_inv_tau_i[2])
    A = A.at[_F_NI_R.start + 2, _F_GRAV_G.start + 0].set( theta.g_ocr * half_inv_tau_i[2])

    # HE linear core: dx_v_lin = a_lin - v_lin/tau_head + (visual gating in f)
    A = A.at[_F_HEAD, _F_HEAD  ].set(-inv_tau_head * I3)
    A = A.at[_F_HEAD, _F_GRAV_A].set(I3)

    # ── A_cross : (52, 52) cross-couplings via ACTIVATIONS (firing rates) ───
    # Operates on a_fast (not x_fast). Saccade BN dynamics receive input from
    # IBN cross-inhibition and OPN additive offset — both via activations
    # (burst_velocity for IBN; clipped firing rate for OPN). For linear-loop
    # slots where activation = identity, A_cross is zero (those couplings
    # already live in A).
    A_cross = jnp.zeros((N_FAST, N_FAST))

    # BN cross-inhibition (contralateral IBN): dx_bn += -g_ibn_bn · a_ibn / τ_bn
    A_cross = A_cross.at[_F_SG_X_EBN_R, _F_SG_X_IBN_L].set(-theta.g_ibn_bn * inv_tau_bn * I3)
    A_cross = A_cross.at[_F_SG_X_EBN_L, _F_SG_X_IBN_R].set(-theta.g_ibn_bn * inv_tau_bn * I3)
    A_cross = A_cross.at[_F_SG_X_IBN_R, _F_SG_X_IBN_L].set(-theta.g_ibn_bn * inv_tau_bn * I3)
    A_cross = A_cross.at[_F_SG_X_IBN_L, _F_SG_X_IBN_R].set(-theta.g_ibn_bn * inv_tau_bn * I3)

    # OPN→BN additive offset (via OPN activation, scalar drives 3-axis vector):
    #   dx_bn += -g_opn_bn_hold · act_opn / τ_bn  on each axis.
    opn_inh_col = -theta.g_opn_bn_hold * inv_tau_bn * jnp.ones(3)
    A_cross = A_cross.at[_F_SG_X_EBN_R, _F_SG_Z_OPN].set(opn_inh_col)
    A_cross = A_cross.at[_F_SG_X_EBN_L, _F_SG_Z_OPN].set(opn_inh_col)
    A_cross = A_cross.at[_F_SG_X_IBN_R, _F_SG_Z_OPN].set(opn_inh_col)
    A_cross = A_cross.at[_F_SG_X_IBN_L, _F_SG_Z_OPN].set(opn_inh_col)

    # IBN→OPN inhibition: dz_opn += -g_ibn_opn · ibn_total / τ_sac
    #   ibn_total = Σ a_ibn_R + Σ a_ibn_L  →  row of ones across the 3 axes.
    ibn_to_opn_row = -theta.g_ibn_opn * inv_tau_sac * jnp.ones(3)
    A_cross = A_cross.at[_F_SG_Z_OPN, _F_SG_X_IBN_R].set(ibn_to_opn_row)
    A_cross = A_cross.at[_F_SG_Z_OPN, _F_SG_X_IBN_L].set(ibn_to_opn_row)

    # ── B : (52, 32) fast input gain ────────────────────────────────────────
    B = jnp.zeros((N_FAST, N_U))

    # Pursuit driven by slip
    B = B.at[_F_PU, _U_SLIP].set((theta.K_pursuit / (1.0 + K_phi_p)) * I3)

    # NI driven by direct slip→u_pu pulse path
    B = B.at[_F_NI_L, _U_SLIP].set( 0.5 * pu_slip_to_u_pu * I3)
    B = B.at[_F_NI_R, _U_SLIP].set(-0.5 * pu_slip_to_u_pu * I3)

    # NI driven by -w_est_feedthrough (D_VS @ canal/slip) via vor_torsion_gain
    # w_est feedthrough on canal: g_vor · PINV_SENS · canal
    # NI gets -0.5·g · w_est_feedthrough, so:
    B = B.at[_F_NI_L, _U_CANAL].set(-0.5 * (vor_torsion_gain[:, None] * g_vor * PINV_SENS))
    B = B.at[_F_NI_R, _U_CANAL].set( 0.5 * (vor_torsion_gain[:, None] * g_vor * PINV_SENS))
    # w_est feedthrough on scene_slip: -g_vis · I (note negative sign!)
    B = B.at[_F_NI_L, _U_SCENE_SLIP].set( 0.5 * jnp.diag(vor_torsion_gain) * g_vis)
    B = B.at[_F_NI_R, _U_SCENE_SLIP].set(-0.5 * jnp.diag(vor_torsion_gain) * g_vis)

    # NI driven by extra delegated velocity drives (u_burst + omega_tvor + listing_vel)
    B = B.at[_F_NI_L, _U_VEL_BRAIN].set( 0.5 * I3)
    B = B.at[_F_NI_R, _U_VEL_BRAIN].set(-0.5 * I3)

    # NI driven by tonic position shift (OCR + Listing's torsion target)
    B = B.at[_F_NI_L, _U_TONIC_NI].set( jnp.diag(half_inv_tau_i))
    B = B.at[_F_NI_R, _U_TONIC_NI].set(-jnp.diag(half_inv_tau_i))

    # Vergence fast: K_v · (disparity + SVBN) + K_v · verg_rate_tvor
    B = B.at[_F_V, _U_DISP       ].set(K_v * I3)
    B = B.at[_F_V, _U_VRATE_TVOR ].set(K_v * I3)

    # Accommodation fast driven by defocus
    B = B.at[_F_A_FAST, _U_DEFOC].set(K_af)

    # VS population A: g_pop_A · K_vs · PINV_SENS · canal  - K_vis · scene_slip
    B = B.at[_F_VS_A, _U_CANAL     ].set(g_pop_A[:, None] * K_vs_canal * PINV_SENS)
    B = B.at[_F_VS_A, _U_SCENE_SLIP].set(-K_vs_vis * I3)
    # VS population B: -g_pop_B · K_vs · PINV_SENS · canal  + K_vis · scene_slip
    B = B.at[_F_VS_B, _U_CANAL     ].set(-g_pop_B[:, None] * K_vs_canal * PINV_SENS)
    B = B.at[_F_VS_B, _U_SCENE_SLIP].set( K_vs_vis * I3)

    # GE driven by GIA: K_grav (baseline) · gia → g_est ; K_lin · gia → a_lin
    B = B.at[_F_GRAV_G, _U_GIA].set(K_grav_b * I3)
    B = B.at[_F_GRAV_A, _U_GIA].set(K_lin_b  * I3)

    # HE: vestibular path (a_lin -> v_lin) is in A; visual path is bilinear in f.

    # ── G_fs : (34, 10) slow → fast direct coupling ────────────────────────
    # Direct coefficient of x_slow[j] in dx_fast[i]. Encodes "leak toward
    # slow-state-derived target" via its diagonal-block entries:
    #   NI_L → +1/(2 tau_i) · x_NI_NULL  (NI_L leaks toward +x_null/2)
    #   VS_A → +1/(2 tau_vs) · x_VS_NULL (VS_A leaks toward +x_vs_null/2)
    G_fs = jnp.zeros((N_FAST, N_SLOW))
    G_fs = G_fs.at[_F_NI_L, _S_NI_NULL].set( 0.5 * jnp.diag(inv_tau_i))
    G_fs = G_fs.at[_F_NI_R, _S_NI_NULL].set(-0.5 * jnp.diag(inv_tau_i))
    G_fs = G_fs.at[_F_VS_A, _S_VS_NULL].set( 0.5 * jnp.diag(inv_tau_vs))
    G_fs = G_fs.at[_F_VS_B, _S_VS_NULL].set(-0.5 * jnp.diag(inv_tau_vs))

    # ── C : (10, 10) slow DIAGONAL leak ────────────────────────────────────
    C = jnp.zeros((N_SLOW, N_SLOW))
    C = C.at[_S_NI_NULL, _S_NI_NULL].set(-inv_tau_adapt   * I3)
    C = C.at[_S_V_TONIC, _S_V_TONIC].set(-inv_tau_v_tonic * I3)
    C = C.at[_S_A_SLOW,  _S_A_SLOW ].set(-inv_tau_a_slow)
    C = C.at[_S_VS_NULL, _S_VS_NULL].set(-inv_tau_vs_adapt * I3)

    # ── M_ss : (10, 10) slow→slow cross-couplings (NOT through (x-T)) ──────
    M_ss = jnp.zeros((N_SLOW, N_SLOW))
    M_ss = M_ss.at[_S_V_TONIC.start + _AXIS_H, _S_A_SLOW].set(
        K_t * alpha * inv_tau_v_tonic)
    M_ss = M_ss.at[_S_A_SLOW, _S_V_TONIC.start + _AXIS_H].set(
        K_as * beta * inv_tau_a_slow)

    # ── D : (10, 34) slow driven by fast ───────────────────────────────────
    D = jnp.zeros((N_SLOW, N_FAST))
    D = D.at[_S_NI_NULL, _F_NI_L].set( inv_tau_adapt * I3)
    D = D.at[_S_NI_NULL, _F_NI_R].set(-inv_tau_adapt * I3)
    D = D.at[_S_V_TONIC, _F_V].set((K_t * inv_tau_v_tonic) * I3)
    D = D.at[_S_V_TONIC.start + _AXIS_H, _F_A_FAST].set(K_t * alpha * inv_tau_v_tonic)
    D = D.at[_S_A_SLOW, _F_A_FAST].set(K_as * inv_tau_a_slow)
    D = D.at[_S_A_SLOW, _F_V.start + _AXIS_H].set(K_as * beta * inv_tau_a_slow)
    # VS null tracks w_est = x_A - x_B  (with feedthrough in B_slow on inputs)
    D = D.at[_S_VS_NULL, _F_VS_A].set( inv_tau_vs_adapt * I3)
    D = D.at[_S_VS_NULL, _F_VS_B].set(-inv_tau_vs_adapt * I3)
    # OCR coupling on NI null: dx_NI_NULL[2] += -inv_tau_adapt · (-g_ocr · g_est[0])
    D = D.at[_S_NI_NULL.start + 2, _F_GRAV_G.start + 0].set(theta.g_ocr * inv_tau_adapt)

    # ── B_slow : (10, 32) slow input gain ───────────────────────────────────
    B_slow = jnp.zeros((N_SLOW, N_U))
    B_slow = B_slow.at[_S_NI_NULL, _U_TONIC_NI].set(-inv_tau_adapt * I3)
    B_slow = B_slow.at[_S_V_TONIC, _U_DISP      ].set((K_t * tau_p * inv_tau_v_tonic) * I3)
    B_slow = B_slow.at[_S_V_TONIC, _U_VRATE_TVOR].set((K_t * tau_p * inv_tau_v_tonic) * I3)
    B_slow = B_slow.at[_S_A_SLOW, _U_DEFOC].set(K_as * tau_acc_plant * inv_tau_a_slow)
    # VS null tracks w_est which has D_VS feedthrough on canal/slip
    B_slow = B_slow.at[_S_VS_NULL, _U_CANAL     ].set(inv_tau_vs_adapt * g_vor * PINV_SENS)
    B_slow = B_slow.at[_S_VS_NULL, _U_SCENE_SLIP].set(-inv_tau_vs_adapt * g_vis * I3)

    # ── T : (10,) slow setpoint ────────────────────────────────────────────
    T = jnp.zeros(N_SLOW)
    T = T.at[_S_V_TONIC.start + _AXIS_H].set(theta.tonic_verg)
    T = T.at[_S_A_SLOW].set(theta.tonic_acc)

    # ── b_prox : (34,) constant fast drive ─────────────────────────────────
    b_prox = jnp.zeros(N_FAST)
    proximal_verg_H = theta.proximal_d * theta.ipd_brain * _DEG_PER_RAD
    b_prox = b_prox.at[_F_V.start + _AXIS_H].set(proximal_verg_H * inv_tau_v)
    b_prox = b_prox.at[_F_A_FAST].set(theta.proximal_d * inv_tau_a_fast)
    # VS resting bias: x_pop_A leaks toward b_vs[A] + x_null/2  →  +b_vs[A]/tau_vs
    b_vs = jnp.asarray(theta.b_vs)
    b_prox = b_prox.at[_F_VS_A].set(inv_tau_vs * b_vs[:3])
    b_prox = b_prox.at[_F_VS_B].set(inv_tau_vs * b_vs[3:])
    # OPN constant tonic recovery: dz_opn includes 100 · k_tonic_opn / τ_sac
    b_prox = b_prox.at[_F_SG_Z_OPN].set(100.0 * theta.k_tonic_opn * inv_tau_sac)

    # ── E : (10, 34) motor readout from fast ────────────────────────────────
    E = jnp.zeros((N_Y, N_FAST))
    E = E.at[_Y_U_PU, _F_PU  ].set(pu_state_to_u_pu * I3)
    E = E.at[_Y_NI,   _F_NI_L].set(I3)
    E = E.at[_Y_NI,   _F_NI_R].set(-I3)
    E = E.at[_Y_NI,   _F_PU  ].set(tau_p * pu_state_to_u_pu * I3)
    # NI pulse-step: tau_p · u_vel = tau_p · (-w_est·g + u_brain + ...)
    # -w_est·g state contribution: -tau_p·g · (x_VS_A - x_VS_B)
    Tg_full = tau_p * jnp.diag(vor_torsion_gain)
    E = E.at[_Y_NI, _F_VS_A].set(-Tg_full)
    E = E.at[_Y_NI, _F_VS_B].set( Tg_full)
    E = E.at[_Y_U_VERG, _F_V    ].set(I3)
    E = E.at[_Y_U_VERG.start + _AXIS_H, _F_A_FAST].set(alpha)
    E = E.at[_Y_U_ACC, _F_A_FAST].set(1.0)
    E = E.at[_Y_U_ACC, _F_V.start + _AXIS_H].set(beta)

    # w_est = (x_VS_A - x_VS_B) + g_vor·PINV_SENS·canal - g_vis·scene_slip
    E = E.at[_Y_W_EST, _F_VS_A].set( I3)
    E = E.at[_Y_W_EST, _F_VS_B].set(-I3)
    # ocr = [0, 0, -g_ocr · g_est[0]] — only roll axis depends on x-component of g_est
    E = E.at[_Y_OCR.start + 2, _F_GRAV_G.start + 0].set(-theta.g_ocr)

    # ── G : (10, 10) motor readout from slow ────────────────────────────────
    G = jnp.zeros((N_Y, N_SLOW))
    G = G.at[_Y_U_VERG, _S_V_TONIC].set(I3)
    G = G.at[_Y_U_VERG.start + _AXIS_H, _S_A_SLOW].set(alpha)
    G = G.at[_Y_U_ACC, _S_A_SLOW].set(1.0)
    G = G.at[_Y_U_ACC, _S_V_TONIC.start + _AXIS_H].set(beta)

    # ── F : (10, 32) direct sensor feedthrough ─────────────────────────────
    F = jnp.zeros((N_Y, N_U))
    F = F.at[_Y_U_PU, _U_SLIP].set(pu_slip_to_u_pu * I3)
    F = F.at[_Y_NI, _U_SLIP    ].set(tau_p * pu_slip_to_u_pu * I3)
    F = F.at[_Y_NI, _U_VEL_BRAIN].set(tau_p * I3)
    # NI pulse-step from -w_est_feedthrough on canal/slip
    F = F.at[_Y_NI, _U_CANAL     ].set(-tau_p * (vor_torsion_gain[:, None] * g_vor * PINV_SENS))
    F = F.at[_Y_NI, _U_SCENE_SLIP].set( tau_p * jnp.diag(vor_torsion_gain) * g_vis)
    F = F.at[_Y_U_VERG, _U_DISP      ].set(tau_p * I3)
    F = F.at[_Y_U_VERG, _U_VRATE_TVOR].set(tau_p * I3)
    F = F.at[_Y_U_ACC, _U_DEFOC].set(tau_acc_plant)
    # w_est canal/slip feedthrough (sensor → head-velocity estimate readout)
    F = F.at[_Y_W_EST, _U_CANAL     ].set(g_vor * PINV_SENS)
    F = F.at[_Y_W_EST, _U_SCENE_SLIP].set(-g_vis * I3)

    return UnifiedMatrices(A=A, A_cross=A_cross, B=B, G_fs=G_fs, C=C, M_ss=M_ss, D=D,
                            B_slow=B_slow, T=T, E=E, G=G, F=F, b_prox=b_prox)


# ─── Activations : pure-on-x nonlinearities (per-state and compound) ───────
# Distinction:
#   activations  = univariate nonlinearities of a single state (or sums /
#                  scalings / single-state compositions). NO mixing with input
#                  or with another state-derived quantity.
#   f(x, x'/u)   = mixed nonlinearities — bilinear products of two states
#                  (e.g. ω × g), state × input (rf bilinear), or saturating
#                  composites that involve multiple state types.
#
# This makes the recurrence dx/dt = M·a + ... + f(...), where M operates on
# the activation vector a (mostly identity for linear loops) and f handles
# the state-state / state-input interactions.

class Activations(NamedTuple):
    """Pure-on-x readouts: per-state φ(x_i) and single-input compounds."""
    a_fast:         jnp.ndarray   # (N_FAST,) per-fast-state activation
    a_slow:         jnp.ndarray   # (N_SLOW,) per-slow-state activation
    # Saccade-generator compound activations (linear combos / clips of activations)
    u_burst:        jnp.ndarray   # a_ebn_R − a_ebn_L         — SG motor output
    ibn_total:      jnp.ndarray   # Σ a_ibn_R + Σ a_ibn_L     — total IBN drive
    ibn_norm:       jnp.ndarray   # clip(ibn_total / 2g_burst, 0, 1)
    normalized_opn: jnp.ndarray   # a_opn / 100
    # Saccade-trigger sigmoids (univariate on a single state) — pure on x
    charge_sac:     jnp.ndarray   # clip(k_acc · (z_acc - threshold_acc), 0, 1)
    mem_active:     jnp.ndarray   # sigmoid(50 · (trust - threshold))
    # Gravity normalization
    g_hat:          jnp.ndarray   # g_est / |g_est|


def activations(x_fast, x_slow, x_target_mem, theta) -> Activations:
    """Map state → activation. Identity for linear loops; nonlinear for SG/GE.

    Per-state φ_i:
      - Linear loops (pursuit, NI, vergence, accommodation, VS pop, GE
        g/a/rf, HE, all slow states): identity (a_i = x_i).
      - Saccade-generator OPN: clip to [0, 100] (membrane → tonic-firing band).
      - Saccade-generator burst neurons (EBN/IBN): saturating-exponential
        firing rate (the "main sequence" nonlinearity), via burst_velocity.

    Compound activations: u_burst, ibn_total, ibn_norm, normalized_opn are
    univariate / linear-combo functions of activations themselves (still pure
    on x — no input mixing).

    g_hat is the normalized gravity vector — a univariate operation on the
    GE state (purely on-x).
    """
    # ── Per-state activations ───────────────────────────────────────────────
    a_fast = x_fast   # baseline: identity
    a_opn   = jnp.clip(x_fast[_F_SG_Z_OPN], 0.0, 100.0)
    a_ebn_R = _burst_velocity(x_fast[_F_SG_X_EBN_R], theta)
    a_ebn_L = _burst_velocity(x_fast[_F_SG_X_EBN_L], theta)
    a_ibn_R = _burst_velocity(x_fast[_F_SG_X_IBN_R], theta)
    a_ibn_L = _burst_velocity(x_fast[_F_SG_X_IBN_L], theta)
    a_fast = a_fast.at[_F_SG_Z_OPN  ].set(a_opn)
    a_fast = a_fast.at[_F_SG_X_EBN_R].set(a_ebn_R)
    a_fast = a_fast.at[_F_SG_X_EBN_L].set(a_ebn_L)
    a_fast = a_fast.at[_F_SG_X_IBN_R].set(a_ibn_R)
    a_fast = a_fast.at[_F_SG_X_IBN_L].set(a_ibn_L)

    a_slow = x_slow   # all identity for the unified subset

    # ── Compound activations (still pure on x) ──────────────────────────────
    u_burst        = a_ebn_R - a_ebn_L
    ibn_total      = jnp.sum(a_ibn_R) + jnp.sum(a_ibn_L)
    ibn_norm       = jnp.clip(ibn_total / (2.0 * theta.g_burst), 0.0, 1.0)
    normalized_opn = a_opn / 100.0

    # ── Gravity normalization ──────────────────────────────────────────────
    g_est = x_fast[_F_GRAV_G]
    g_hat = g_est / (jnp.linalg.norm(g_est) + 1e-9)

    # ── Trigger sigmoids (pure on a single state) ──────────────────────────
    # charge_sac: clip(k_acc · (z_acc - threshold_acc), 0, 1) — drives z_trig.
    # mem_active: sigmoid(50 · (trust - threshold)) — gates target memory.
    z_acc      = x_fast[_F_SG_Z_ACC]
    trust      = x_target_mem[3]
    charge_sac = jnp.clip(theta.k_acc * (z_acc - theta.threshold_acc), 0.0, 1.0)
    mem_active = jax.nn.sigmoid(50.0 * (trust - _TARGET_MEM_TRUST_THRESHOLD))

    return Activations(
        a_fast=a_fast, a_slow=a_slow,
        u_burst=u_burst, ibn_total=ibn_total, ibn_norm=ibn_norm,
        normalized_opn=normalized_opn,
        charge_sac=charge_sac, mem_active=mem_active,
        g_hat=g_hat,
    )


# ─── g : Bayesian-MAP input preprocessing ──────────────────────────────────

def g(sensory_out, theta, u_vel_brain, u_tonic_ni, verg_rate_tvor, u_svbn):
    """Stage-1 preprocessing: bundle sensors and delegated drives into u_proc.

    For the unified subset, g is mostly identity on the components: canal
    saturation is now sensor-side (canal.step), and the remaining sensors
    pass through unchanged. Soft-threshold dead-zones (sparse-amplitude prior
    MAP) and variance-stabilising transforms belong here once their priors
    are calibrated.
    """
    del theta
    disp_total = sensory_out.target_disparity + u_svbn
    return jnp.concatenate([
        sensory_out.target_slip,             # _U_SLIP
        disp_total,                          # _U_DISP (raw + SVBN)
        jnp.array([sensory_out.defocus]),    # _U_DEFOC
        u_vel_brain,                         # _U_VEL_BRAIN  (u_burst + omega_tvor + listing_vel)
        u_tonic_ni,                          # _U_TONIC_NI   (Listing's torsion target only;
                                              #               OCR is folded into f via x_grav_g)
        verg_rate_tvor,                      # _U_VRATE_TVOR
        sensory_out.canal,                   # _U_CANAL  (already saturation-clipped sensor-side)
        sensory_out.scene_slip,              # _U_SCENE_SLIP
        sensory_out.otolith,                 # _U_GIA
        sensory_out.scene_linear_vel,        # _U_SCENE_LIN_VEL
        jnp.array([sensory_out.scene_visible]),  # _U_SCENE_VIS
    ])


# ─── f : bilinear corrections (state × state, state × input) ───────────────

def f(x_fast, x_slow, u_proc, theta, z_act, act):
    """Bilinear corrections beyond the linear matrix dynamics.

    Active terms:
      - Gravity transport:  -ω × g_est  (state×state, SO(3) Lie bracket)
      - rf bilinear:         cross(gia, -g_est) / G0² / tau_rf  (state×input)
      - Kalman gain modulation on residual:
            (K_grav_eff - K_grav) · (gia - g_est - a_lin)  in g_est
            (K_lin_eff  - K_lin ) · (gia - g_est - a_lin)  in a_lin
        where K_eff = K · sqrt(1 + |ω × g_hat|/w_canal_gate)
      - HE visual gating: K_he_vis · scene_visible · (-scene_lin_vel - v_lin)
      - OCR coupling: NI null leaks toward -g_ocr·g_est[0] on roll axis
            → bilinear shift on dx_NI_L/R via half_inv_tau_i.
    """
    del x_slow, z_act  # not used in current f terms

    g_est    = x_fast[_F_GRAV_G]
    a_lin    = x_fast[_F_GRAV_A]
    rf_state = x_fast[_F_GRAV_RF]
    v_lin    = x_fast[_F_HEAD]
    x_VS_A   = x_fast[_F_VS_A]
    x_VS_B   = x_fast[_F_VS_B]

    canal_clipped = u_proc[_U_CANAL]
    scene_slip    = u_proc[_U_SCENE_SLIP]
    gia           = u_proc[_U_GIA]
    scene_lin_vel = u_proc[_U_SCENE_LIN_VEL]
    scene_visible = u_proc[_U_SCENE_VIS]

    # ── w_est from VS state + feedthrough ───────────────────────────────────
    # (mirrors _vs_step output: net = x_A - x_B + D · u, with D = [g_vor·PINV_SENS, -g_vis·I])
    w_est_state    = x_VS_A - x_VS_B
    w_est_canal_FT = theta.g_vor * (PINV_SENS @ canal_clipped)
    w_est_slip_FT  = -theta.g_vis * scene_slip
    w_est = w_est_state + w_est_canal_FT + w_est_slip_FT

    # ── Inverse VS time constants for population (per-axis) ─────────────────
    inv_tau_vs = 1.0 / (theta.tau_vs * jnp.array(
        [1.0, theta.tau_vs_pitch_frac, theta.tau_vs_roll_frac]))

    # ── Gravity transport: -ω × g_est  ─────────────────────────────────────
    # ω in head-frame xyz (from ypr); cross product; result in head-frame xyz.
    # Both g_est and the result already live in xyz frame (head frame).
    w_rad_xyz = jnp.radians(ypr_to_xyz(w_est))
    transport = -jnp.cross(w_rad_xyz, g_est)

    # ── Kalman gain modulation: K_grav_eff = K_grav · sqrt(1 + ρ); ρ = |ω × g_hat|/w_gate ──
    # g_hat is a pure-on-x activation (gravity normalize) supplied by act.
    g_hat   = act.g_hat
    w_xyz   = ypr_to_xyz(w_est)
    rho     = jnp.linalg.norm(jnp.cross(w_xyz, g_hat)) / theta.w_canal_gate
    gate_factor = jnp.sqrt(1.0 + rho)

    K_grav_eff_minus_b = theta.K_grav * (gate_factor - 1.0)
    K_lin_eff_minus_b  = theta.K_lin  * (1.0 / gate_factor - 1.0)
    residual = gia - g_est - a_lin

    # ── rf bilinear: rf_new = cross(gia, -g_est)/G0²; the linear -rf/tau_rf is in A
    rf_new = xyz_to_ypr(jnp.cross(gia, -g_est)) / (G0 ** 2)
    drf_bilinear = rf_new / _TAU_RF_STATE

    # ── HE visual gating: K_he_vis · scene_visible · (-scene_lin_vel - v_lin)
    K_vis_eff = theta.K_he_vis * scene_visible
    he_visual = K_vis_eff * (-scene_lin_vel - v_lin)

    # VS rf feedback and OCR coupling are now linear matrix entries (in A and D),
    # not f. They were moved out of f because they're linear state-state couplings.

    # ── Assemble f vector ───────────────────────────────────────────────────
    f_vec = jnp.zeros(N_FAST)

    # GE: gravity transport + Kalman gain correction
    f_vec = f_vec.at[_F_GRAV_G].set(transport + K_grav_eff_minus_b * residual)
    # GE: a_lin gain modulation
    f_vec = f_vec.at[_F_GRAV_A].set(K_lin_eff_minus_b * residual)
    # GE: rf bilinear
    f_vec = f_vec.at[_F_GRAV_RF].set(drf_bilinear)

    # HE: visual gate
    f_vec = f_vec.at[_F_HEAD].set(he_visual)

    return f_vec


def f_slow(x_fast, x_slow, u_proc, theta, z_act):
    """Bilinear corrections on the slow-state derivative.

    Empty in the unified subset — the OCR contribution to NI null is now a
    linear state-state entry in matrix D.
    """
    del x_fast, x_slow, u_proc, theta, z_act
    return jnp.zeros(N_SLOW)


def f_motor(x_fast, x_slow, u_proc, theta):
    """Bilinear corrections on the motor readout.

    OCR enters the NI dynamics by shifting the leak target (handled in f and
    f_slow), NOT by direct feedthrough to motor_cmd_ni. The motor readout is
    purely linear in (x_fast, x_slow, u_proc) for the unified subset.
    """
    del x_fast, x_slow, u_proc, theta
    return jnp.zeros(N_Y)


# ─── State packing helpers ─────────────────────────────────────────────────

def _pack_subset(brain_state):
    """Map BrainState NT → (x_fast, x_slow) for the unified subset.

    Pursuit caveat: unified_brain treats pursuit as a single signed integrator
    (linear).  Bilateral pursuit (x_R, x_L with rectified drives) cannot be
    projected losslessly into the linear unified form; we collapse to
    NET (x_R − x_L) here and re-distribute symmetrically in scatter.
    bench_compare_unified.py will diverge from the bilateral path in lesion
    scenarios that break L/R symmetry.
    """
    sg_st = brain_state.sg
    va_st = brain_state.va
    sm_st = brain_state.sm
    pu_st = brain_state.pu
    ni_st = brain_state.ni

    x_fast = jnp.zeros(N_FAST)
    x_fast = x_fast.at[_F_PU      ].set(pu_st.R - pu_st.L)
    x_fast = x_fast.at[_F_NI_L    ].set(ni_st.L)
    x_fast = x_fast.at[_F_NI_R    ].set(ni_st.R)
    x_fast = x_fast.at[_F_V       ].set(va_st.verg_fast)
    x_fast = x_fast.at[_F_V_COPY  ].set(va_st.verg_copy)
    x_fast = x_fast.at[_F_A_FAST  ].set(va_st.acc_fast)
    # Self-motion fast states: VS_A=L pop, VS_B=R pop, GE (g, a_lin, rf), HE
    x_fast = x_fast.at[_F_VS_A    ].set(sm_st.vs_L)
    x_fast = x_fast.at[_F_VS_B    ].set(sm_st.vs_R)
    x_fast = x_fast.at[_F_GRAV_G  ].set(sm_st.g_est)
    x_fast = x_fast.at[_F_GRAV_A  ].set(sm_st.a_lin)
    x_fast = x_fast.at[_F_GRAV_RF ].set(sm_st.rf)
    x_fast = x_fast.at[_F_HEAD    ].set(sm_st.v_lin)
    x_fast = x_fast.at[_F_SG_E_HELD ].set(sg_st.e_held)
    x_fast = x_fast.at[_F_SG_Z_OPN  ].set(sg_st.z_opn)
    x_fast = x_fast.at[_F_SG_Z_ACC  ].set(sg_st.z_acc)
    x_fast = x_fast.at[_F_SG_Z_TRIG ].set(sg_st.z_trig)
    x_fast = x_fast.at[_F_SG_X_EBN_R].set(sg_st.ebn_R)
    x_fast = x_fast.at[_F_SG_X_EBN_L].set(sg_st.ebn_L)
    x_fast = x_fast.at[_F_SG_X_IBN_R].set(sg_st.ibn_R)
    x_fast = x_fast.at[_F_SG_X_IBN_L].set(sg_st.ibn_L)

    x_slow = jnp.zeros(N_SLOW)
    x_slow = x_slow.at[_S_NI_NULL ].set(ni_st.null)
    x_slow = x_slow.at[_S_V_TONIC ].set(va_st.verg_tonic)
    x_slow = x_slow.at[_S_A_SLOW  ].set(va_st.acc_slow)
    x_slow = x_slow.at[_S_VS_NULL ].set(sm_st.vs_null)
    return x_fast, x_slow


def _scatter_subset(dbrain, dx_fast, dx_slow):
    """Map (dx_fast, dx_slow) → BrainState derivative.

    Pursuit caveat (see _pack_subset): NET dx_pu split evenly across (R, L).
    """
    _dxpu_net = dx_fast[_F_PU]
    dpu = brain_model.pu.State(R= 0.5 * _dxpu_net, L=-0.5 * _dxpu_net)
    dni = brain_model.ni.State(L=dx_fast[_F_NI_L],
                                R=dx_fast[_F_NI_R],
                                null=dx_slow[_S_NI_NULL])
    dva = brain_model.va.State(
        verg_fast  = dx_fast[_F_V],
        verg_tonic = dx_slow[_S_V_TONIC],
        verg_copy  = dx_fast[_F_V_COPY],
        acc_fast   = dx_fast[_F_A_FAST],
        acc_slow   = dx_slow[_S_A_SLOW],
    )
    dsm = brain_model.sm.State(
        vs_L    = dx_fast[_F_VS_A],
        vs_R    = dx_fast[_F_VS_B],
        vs_null = dx_slow[_S_VS_NULL],
        g_est   = dx_fast[_F_GRAV_G],
        a_lin   = dx_fast[_F_GRAV_A],
        rf      = dx_fast[_F_GRAV_RF],
        v_lin   = dx_fast[_F_HEAD],
    )
    dsg = brain_model.sg.State(
        e_held = dx_fast[_F_SG_E_HELD],
        z_opn  = dx_fast[_F_SG_Z_OPN],
        z_acc  = dx_fast[_F_SG_Z_ACC],
        z_trig = dx_fast[_F_SG_Z_TRIG],
        ebn_R  = dx_fast[_F_SG_X_EBN_R],
        ebn_L  = dx_fast[_F_SG_X_EBN_L],
        ibn_R  = dx_fast[_F_SG_X_IBN_R],
        ibn_L  = dx_fast[_F_SG_X_IBN_L],
    )
    return dbrain._replace(pu=dpu, ni=dni, va=dva, sm=dsm, sg=dsg)


# ─── Step function — drop-in replacement for brain_model.step ──────────────

def step(brain_state, sensory_out, brain_params, noise_acc=0.0):
    """Single ODE step in the unified state-space form.

    Continuous loops + self-motion (VS + gravity + heading) are now handled
    by the unified matrices + bilinear f. Saccade generator, T-VOR, target
    memory, Listing's law, and FCP remain delegated.
    """
    va_st        = brain_state.va
    pt_st        = brain_state.pt
    fcp_st       = brain_state.fcp
    x_target_mem = jnp.concatenate([pt_st.mem_pos, jnp.array([pt_st.mem_trust])])
    x_ni_net     = brain_state.ni.L - brain_state.ni.R
    x_v_copy     = va_st.verg_copy
    x_acc_fast   = va_st.acc_fast

    # Pack unified subset state and build matrices early so all gains live
    # in M (the matrix bundle). Step() consumes named readouts (w_est, ocr)
    # from the matrix output, so no gain appears inline below.
    x_fast, x_slow = _pack_subset(brain_state)
    M = matrices(brain_params)

    # State-only readouts (no inputs that aren't sensors).
    a_lin_est = x_fast[_F_GRAV_A]
    v_lin     = x_fast[_F_HEAD]

    # w_est and ocr come from the matrix readout — gains live in E/F, not here.
    # Pull them from M.E·x_fast + M.F·(canal & scene_slip portion of u_proc).
    w_est = (M.E[_Y_W_EST] @ x_fast
             + M.F[_Y_W_EST, _U_CANAL]      @ sensory_out.canal
             + M.F[_Y_W_EST, _U_SCENE_SLIP] @ sensory_out.scene_slip)
    ocr   = M.E[_Y_OCR] @ x_fast

    # ── Target memory (inline) ──────────────────────────────────────────────
    x_mem      = x_target_mem[0:3]
    trust      = x_target_mem[3]
    tgt_pos_d  = sensory_out.target_pos
    tgt_vis_d  = sensory_out.target_visible

    z_act_now = 1.0 - jnp.clip(x_fast[_F_SG_Z_OPN], 0.0, 100.0) / 100.0
    inv_consume = z_act_now / _TAU_TARGET_MEM_CONSUME
    dx_mem = (tgt_vis_d * (tgt_pos_d - x_mem) / _TAU_TARGET_MEM_UPDATE
              - inv_consume * x_mem)
    inv_rise   = 1.0 / _TAU_TARGET_MEM_TRUST_RISE
    inv_decay  = 1.0 / _TAU_TARGET_MEM_TRUST_DECAY
    gain_trust = tgt_vis_d * inv_rise + (1.0 - tgt_vis_d) * inv_decay
    dx_trust   = gain_trust * (tgt_vis_d - trust)
    # Pack target-memory derivative as pt.State NT
    dpt = brain_model.pt.State(mem_pos=dx_mem, mem_trust=dx_trust)

    # tgt_pos_eff / tgt_vis_eff use act.mem_active (computed below from x_target_mem).
    # Defer their construction until after activations() is called.

    # ── Activations (pure-on-x nonlinearities — firing rates from states) ──
    # Most cross-couplings via activations (IBN→BN, OPN→BN, IBN→OPN) live in
    # M.A_cross now; the inline SG block only needs the few quantities that
    # show up in bilinear (state×state) corrections or the saccade target logic.
    act = activations(x_fast, x_slow, x_target_mem, brain_params)
    act_opn        = act.a_fast[_F_SG_Z_OPN]
    normalized_opn = act.normalized_opn
    ibn_norm       = act.ibn_norm
    u_burst        = act.u_burst
    charge_sac     = act.charge_sac
    mem_active     = act.mem_active

    # Build target-memory-gated effective target signals using mem_active activation.
    tgt_pos_eff  = tgt_vis_d * tgt_pos_d + (1.0 - tgt_vis_d) * mem_active * x_mem
    tgt_vis_eff  = jnp.maximum(tgt_vis_d, mem_active)

    # ── Saccade generator (folded in: bilinear + sigmoid + ReLU dynamics) ───
    # State extraction (raw membrane potentials / accumulators)
    e_held  = x_fast[_F_SG_E_HELD]
    z_opn   = x_fast[_F_SG_Z_OPN]
    z_acc   = x_fast[_F_SG_Z_ACC]
    z_trig  = x_fast[_F_SG_Z_TRIG]
    x_ebn_R = x_fast[_F_SG_X_EBN_R]
    x_ebn_L = x_fast[_F_SG_X_EBN_L]
    x_ibn_R = x_fast[_F_SG_X_IBN_R]
    x_ibn_L = x_fast[_F_SG_X_IBN_L]

    # ── Saccade target selection (with Listing's & OCR landing torsion) ──────
    H_landed = x_ni_net[0] + tgt_pos_eff[0] - brain_params.listing_primary[0]
    V_landed = x_ni_net[1] + tgt_pos_eff[1] - brain_params.listing_primary[1]
    T_LL_landed = -(jnp.pi / 360.0) * H_landed * V_landed
    listing_target_delta = jnp.array([0.0, 0.0,
                                       brain_params.listing_gain * T_LL_landed - x_ni_net[2]])
    ocr_delta = ocr + listing_target_delta
    e_target = jnp.clip(tgt_pos_eff + ocr_delta,
                         -brain_params.orbital_limit - x_ni_net,
                          brain_params.orbital_limit - x_ni_net)
    tau_refractory = (brain_params.threshold_acc - brain_params.acc_burst_floor) * brain_params.tau_acc
    x_ni_pred = x_ni_net - brain_params.k_center_vel * tau_refractory * w_est
    e_center  = -brain_params.alpha_reset * (x_ni_pred - ocr)
    doing_saccade     = tgt_vis_eff
    doing_quick_phase = 1.0 - tgt_vis_eff
    e_cur     = doing_saccade * e_target + doing_quick_phase * e_center
    e_cur_mag = jnp.linalg.norm(e_cur)

    # ── Trigger gates: gate_err is a compound-state sigmoid (kept inline);
    #    charge_sac comes from act (pure-on-z_acc, in activations).
    threshold_sac_eff = (doing_saccade     * brain_params.threshold_sac
                         + doing_quick_phase * brain_params.threshold_sac_qp)
    gate_err   = jax.nn.sigmoid(brain_params.k_sac * (e_cur_mag - threshold_sac_eff))

    # ── SG cerebellar / bilinear corrections (everything that's NOT linear) ─
    # Linear pieces now in the matrix (A self-leaks; A_cross IBN→BN / OPN→BN
    # / IBN→OPN; b_prox OPN tonic recovery; A z_opn←z_trig coupling).
    # What remains here:
    #   - BN drive from e_held (sign-split ReLU — not a single per-state activation)
    #   - BN cerebellar gain modulation (bilinear: act_opn × x_bn)
    #   - e_held tracking (bilinear: normalized_opn² × (e_cur - e_held))
    #   - z_trig sigmoid drive (charge_sac) and bilinear drain (z_trig × ibn_norm)
    #   - z_acc bilinear drain and gate_err·normalized_opn drive
    #   - z_opn bilinear (-z_opn · z_trig) coupling
    cereb_bn_factor = brain_params.g_opn_bn * act_opn       # cerebellar gain modulation
    dx_ebn_R_corr = (jax.nn.relu( e_held) - x_ebn_R * cereb_bn_factor) / brain_params.tau_bn
    dx_ebn_L_corr = (jax.nn.relu(-e_held) - x_ebn_L * cereb_bn_factor) / brain_params.tau_bn
    dx_ibn_R_corr = (jax.nn.relu( e_held) - x_ibn_R * cereb_bn_factor) / brain_params.tau_bn
    dx_ibn_L_corr = (jax.nn.relu(-e_held) - x_ibn_L * cereb_bn_factor) / brain_params.tau_bn

    # Robinson resettable integrator (sample-and-hold gated by OPN)
    de_held = -u_burst + normalized_opn**2 * (e_cur - e_held) / brain_params.tau_hold
    dz_trig = (charge_sac - z_trig * brain_params.g_ibn_trig * ibn_norm) / brain_params.tau_trig
    dz_acc = (gate_err * normalized_opn / brain_params.tau_acc
              - brain_params.g_acc_drain * ibn_norm
                * (z_acc - brain_params.acc_burst_floor) / brain_params.tau_burst_drain
              + noise_acc)
    dz_opn = (-z_opn * z_trig) / brain_params.tau_sac

    # ── T-VOR (folded in: inline bilinear cross-product + depth scaling) ────
    # ω_eye = -ĝ × v_lin / D · g_tvor   (state×state, with state-derived 1/D)
    # verg_rate = ipd · (ĝ · v_lin) / D² · g_tvor_verg
    # Distance from vergence yaw via ipd / (2·tan(yaw/2)); NPC gate disengages
    # T-VOR scaling near the near-point of convergence.
    aca_term = brain_params.AC_A * _DEG_PER_PD * x_acc_fast
    current_vergence_yaw = va_st.verg_fast[0] + va_st.verg_tonic[0] + aca_term
    distance_raw = brain_params.ipd_brain / (
        2.0 * jnp.tan(jnp.radians(current_vergence_yaw) * 0.5))
    d_safe = jnp.maximum(distance_raw, _DISTANCE_EPSILON)
    npc = brain_params.distance_npc
    npc_gate = jax.nn.sigmoid(
        (d_safe - _NPC_GATE_CENTER_FRAC * npc) / (_NPC_GATE_SHARPNESS_FRAC * npc))
    inv_distance = npc_gate / d_safe
    g_hat = gaze_unit_vector(x_ni_net)
    DEG_PER_RAD = 180.0 / jnp.pi

    # Integrating + direct paths combined.
    omega_int_xyz_rad = -jnp.cross(g_hat, v_lin)     * inv_distance
    omega_dir_xyz_rad = -jnp.cross(g_hat, a_lin_est) * inv_distance
    omega_tvor = (brain_params.g_tvor      * DEG_PER_RAD * xyz_to_ypr(omega_int_xyz_rad)
                  + brain_params.K_phasic_tvor * DEG_PER_RAD * xyz_to_ypr(omega_dir_xyz_rad))

    dot_gv = jnp.dot(g_hat, v_lin)
    verg_rate_rad_H = brain_params.ipd_brain * dot_gv * inv_distance * inv_distance
    verg_rate_tvor = brain_params.g_tvor_verg * DEG_PER_RAD * jnp.array([
        verg_rate_rad_H, 0.0, 0.0,
    ])

    # ── Pursuit output for Listing's smooth_vel input ──────────────────────
    K_phi_p = brain_params.K_phasic_pursuit
    pu_state_to_u_pu = 1.0 / (1.0 + K_phi_p)
    pu_slip_to_u_pu  = K_phi_p / (1.0 + K_phi_p)
    u_pursuit = (pu_state_to_u_pu * x_fast[_F_PU]
                 + pu_slip_to_u_pu * sensory_out.target_slip)

    # ── Listing's law (folded in: pure bilinear state×state) ────────────────
    # T_LL(H, V) = -(H-H₀)·(V-V₀)·π/360       (torsion set-point — quadratic)
    # T_LL_dot   = -π/360 · [(H-H₀)·V̇ + (V-V₀)·Ḣ]  (torsion velocity — bilinear)
    # cyclo_verg_rate = -l2_frac · V̇ · radians(verg)  (bilinear velocity × state)
    HALF_ANGLE = jnp.pi / 360.0
    smooth_vel_hv = (u_pursuit + omega_tvor)[:2]
    dH = x_ni_net[0] - brain_params.listing_primary[0]
    dV = x_ni_net[1] - brain_params.listing_primary[1]
    H_dot, V_dot = smooth_vel_hv[0], smooth_vel_hv[1]
    cyc_torsion_vel    = -HALF_ANGLE * (dH * V_dot + dV * H_dot)
    cyc_torsion_target = -HALF_ANGLE * dH * dV
    cyclo_verg_rate    = (-brain_params.listing_l2_frac
                          * V_dot * jnp.radians(current_vergence_yaw))

    # ── SVBN burst ─────────────────────────────────────────────────────────
    burst_residual = sensory_out.target_disparity - x_v_copy
    u_svbn = (z_act_now
              * jnp.sign(burst_residual)
              * brain_params.g_svbn_conv
              * (1.0 - jnp.exp(-jnp.abs(burst_residual) / brain_params.X_svbn_conv)))

    # ── Aggregate inputs into NI ────────────────────────────────────────────
    # u_vel_brain holds delegated drives only: u_burst + omega_tvor + listing's velocity
    # (-w_est × vor_torsion_gain is folded into the matrices via VS state coupling)
    u_vel_brain = u_burst + omega_tvor
    u_vel_brain = u_vel_brain.at[2].add(brain_params.listing_gain * cyc_torsion_vel)
    # u_tonic_ni: Listing's torsion target only (OCR is now folded into f via x_grav_g)
    u_tonic_ni = jnp.zeros(3).at[2].set(brain_params.listing_gain * cyc_torsion_target)
    verg_rate_tvor_eff = verg_rate_tvor.at[2].add(brain_params.listing_gain * cyclo_verg_rate)

    # ── Stage 1: g(u) preprocessing ─────────────────────────────────────────
    u_proc = g(sensory_out, brain_params, u_vel_brain, u_tonic_ni,
                verg_rate_tvor_eff, u_svbn)

    # ── Stage 2: linear-affine + bilinear dynamics (M built early at top) ──
    # M.A operates on x_fast (raw membrane potentials, for self-leaks).
    # M.A_cross operates on a_fast (firing rates / activations, for cross-couplings
    # via activations — IBN→BN, OPN→BN, IBN→OPN; for linear-loop slots a = x).
    dx_fast = (M.A @ x_fast
               + M.A_cross @ act.a_fast
               + M.G_fs @ x_slow
               + M.B @ u_proc
               + M.b_prox
               + f(x_fast, x_slow, u_proc, brain_params, z_act_now, act))

    dx_slow = (M.C @ (x_slow - M.T)
               + M.M_ss @ x_slow
               + M.D @ x_fast
               + M.B_slow @ u_proc
               + f_slow(x_fast, x_slow, u_proc, brain_params, z_act_now))

    # ── Vergence copy: (1-z_act) gating override ────────────────────────────
    dx_fast = dx_fast.at[_F_V_COPY].set(
        u_svbn - (1.0 - z_act_now) * x_v_copy / _TAU_COPY_RESET)

    # ── NI anti-windup: clip dx_net at ±orbital_limit (state-dep clip on dx)
    # Applied to the bilateral net (x_L - x_R); common-mode passes unchanged.
    L_lim = brain_params.orbital_limit
    x_net_now  = x_fast[_F_NI_L] - x_fast[_F_NI_R]
    dx_L_now   = dx_fast[_F_NI_L]
    dx_R_now   = dx_fast[_F_NI_R]
    dx_net_aw  = dx_L_now - dx_R_now
    dx_sum_aw  = dx_L_now + dx_R_now
    dx_net_aw  = jnp.where(x_net_now >=  L_lim, jnp.minimum(dx_net_aw, 0.0), dx_net_aw)
    dx_net_aw  = jnp.where(x_net_now <= -L_lim, jnp.maximum(dx_net_aw, 0.0), dx_net_aw)
    dx_fast = dx_fast.at[_F_NI_L].set((dx_net_aw + dx_sum_aw) / 2.0)
    dx_fast = dx_fast.at[_F_NI_R].set((dx_sum_aw - dx_net_aw) / 2.0)

    # ── Saccade-generator state derivatives ─────────────────────────────────
    # All SG slots ADD their inline corrections to the matrix-supplied baselines
    # (linear self-leaks + A_cross cross-couplings + b_prox tonic recovery).
    # Inline pieces are: bilinear cerebellar gain (act_opn × x_bn), ReLU drives
    # from e_held, sample-and-hold OPN-gated tracking, sigmoid/clip triggers,
    # and bilinear -z_opn·z_trig coupling.
    dx_fast = dx_fast.at[_F_SG_E_HELD].add(de_held)
    dx_fast = dx_fast.at[_F_SG_Z_OPN ].add(dz_opn)
    dx_fast = dx_fast.at[_F_SG_Z_ACC ].add(dz_acc)
    dx_fast = dx_fast.at[_F_SG_Z_TRIG].add(dz_trig)
    dx_fast = dx_fast.at[_F_SG_X_EBN_R].add(dx_ebn_R_corr)
    dx_fast = dx_fast.at[_F_SG_X_EBN_L].add(dx_ebn_L_corr)
    dx_fast = dx_fast.at[_F_SG_X_IBN_R].add(dx_ibn_R_corr)
    dx_fast = dx_fast.at[_F_SG_X_IBN_L].add(dx_ibn_L_corr)

    # ── Stage 4: motor readout ──────────────────────────────────────────────
    y = (M.E @ x_fast + M.G @ x_slow + M.F @ u_proc
         + f_motor(x_fast, x_slow, u_proc, brain_params))
    u_pu_vec     = y[_Y_U_PU]
    motor_cmd_ni = y[_Y_NI]
    u_verg       = y[_Y_U_VERG]
    u_acc        = y[_Y_U_ACC]

    # ── Stage 5: pack derivative into BrainState NT ─────────────────────────
    # Start from the input brain_state's structure, then apply scatter + sub-system updates
    dbrain_init = brain_model.rest_brain_state()
    dbrain      = dbrain_init._replace(pt=dpt)
    dbrain      = _scatter_subset(dbrain, dx_fast, dx_slow)

    # ── Stage 6: efference copies and FCP nerves ────────────────────────────
    ec_vel = u_burst + u_pursuit + omega_tvor
    ec_pos = x_ni_net
    ec_verg = u_verg

    dfcp, nerves = fcp.step(fcp_st, jnp.concatenate([motor_cmd_ni, u_verg]), brain_params)
    dbrain = dbrain._replace(fcp=dfcp)

    return dbrain, nerves, ec_vel, ec_pos, ec_verg, u_acc


__all__ = [
    'step', 'matrices', 'g', 'f', 'f_slow', 'f_motor',
    'UnifiedMatrices',
    'N_FAST', 'N_SLOW', 'N_U', 'N_Y',
]
