"""Vergence SSM — slow fusional drive + Zee saccade pulse + L2 cyclovergence.

Drives disconjugate eye movements to align both foveas on a binocular target.

Architecture
────────────
Three parallel channels sharing the same OPN gate (z_act from the version SG):

    1. Slow fusional drive (existing)
       Dual-range nonlinear disparity → leaky integrator x_verg.
       Handles steady fixation, smooth vergence, and slow refusion after saccades.

    2. Zee saccade pulse (Zee et al. 1992)
       When OPN pauses (z_act=1), the disparity error at that instant is latched
       into e_held_verg and a fast burst drives x_copy_verg until the residual
       e_held_verg − x_copy_verg is exhausted.  Same ballistic logic as the
       version SG; OPN resumes at the end of both pulses simultaneously.
       This makes vergence 2–3× faster during combined saccade+vergence movements.

    3. L2 cyclovergence (extended Listing's law; Mok et al. 1992; Tweed 1997)
       As horizontal vergence φ grows, each eye's Listing's plane tilts ±φ/2.
       The resulting cyclovergence demand is:
           T_cyc = listing_l2_frac · φ · (V − V₀) · π/360
       Added as a torsional component of the vergence drive.  Off by default
       (listing_l2_frac=0) until validated against binocular torsion data.

State:  x_verg_full = [x_verg(3) | e_held_verg(3) | x_copy_verg(3)]   (9 states)

    x_verg      — tonic vergence position memory (deg); positive = converged
    e_held_verg — latched 3D vergence error at saccade onset; tracks e_disp
                  when idle (τ=5 ms), frozen during saccade burst
    x_copy_verg — vergence copy integrator; accumulates burst velocity;
                  burst ends when x_copy_verg reaches e_held_verg

Input:  e_disp (3,)    binocular disparity = pos_delayed_L − pos_delayed_R (deg)
        ac_a_drive     accommodative convergence scalar (deg)
        z_act          OPN saccade gate scalar (0=idle, 1=saccade active)
        eye_hv  (2,)   current gaze [H, V] deg from NI net (for L2)

Output: u_verg (3,)    vergence position command → split ±½ in brain_model

Parameters (added to BrainParams):
    g_burst_verg    — vergence saccade pulse gain (deg/s per deg residual); default 1.6
    listing_l2_frac — L2 cyclovergence fraction; default 0 (disabled)
    D_verg          — velocity damping coefficient; dx_verg /= (1+D); default 1.0 (ξ≈0.9)

References:
    Zee DS et al. (1992) J Neurophysiol 68:1624–1641  — saccade-vergence interactions
    Mok D et al. (1992) Invest Ophthalmol Vis Sci 33:2495–2507 — L2 / Listing's plane tilt
    Tweed D (1997) J Neurophysiol 77:2467–2479  — L2 theory
    Schor CM (1979) Vision Res 19:1359–1367  — dual-range vergence model
    Rashbass C, Westheimer G (1961) J Physiol 159:361–364  — vergence TC ~160 ms
"""

import jax
import jax.numpy as jnp

from oculomotor.models.sensory_models.cyclopean_vision import velocity_saturation as _disp_sat

# Half-angle constant (same as listing.py)
HALF_ANGLE = jnp.pi / 360.0

# e_held_verg tracks disparity with this TC when idle (fast latch, matching sg e_held)
_TAU_HOLD_VERG  = 0.005   # s
# x_copy_verg decays to 0 between saccades so the next burst starts from a clean baseline
_TAU_COPY_RESET = 0.5     # s

N_STATES  = 9   # [x_verg(3) | e_held_verg(3) | x_copy_verg(3)]
N_INPUTS  = 3
N_OUTPUTS = 3

_IDX_VERG      = slice(0, 3)
_IDX_HELD_VERG = slice(3, 6)
_IDX_COPY_VERG = slice(6, 9)


def step(x_verg_full, e_disp, ac_a_drive, z_act, eye_hv, brain_params):
    """Vergence step: slow fusional + Zee saccade pulse + L2 cyclovergence.

    Args:
        x_verg_full:  (9,)   [x_verg(3) | e_held_verg(3) | x_copy_verg(3)]
        e_disp:       (3,)   binocular disparity = pos_L − pos_R (deg)
                             Gated by bino = tv_L × tv_R before call.
        ac_a_drive:   scalar  accommodative convergence drive (deg)
        z_act:        scalar  OPN gate (0=idle, 1=saccade active); from SG state
        eye_hv:       (2,)   current gaze [H, V] deg (x_ni_net[:2]) for L2
        brain_params: BrainParams

    Returns:
        dx_verg: (9,)  state derivative
        u_verg:  (3,)  vergence position command (deg) → split ±½ to L/R in brain_model
    """
    x_verg      = x_verg_full[_IDX_VERG]
    e_held_verg = x_verg_full[_IDX_HELD_VERG]
    x_copy_verg = x_verg_full[_IDX_COPY_VERG]

    z_idl = 1.0 - z_act   # 1=idle, 0=saccade

    # ── L2 cyclovergence demand ───────────────────────────────────────────────
    # Listing's plane tilts ±φ/2 per eye as horizontal vergence φ grows.
    # Net cyclovergence (L − R torsion) = φ · (V − V₀) · π/360.
    phi   = x_verg[0]   # horizontal vergence (deg) — use state to avoid algebraic loop
    dV    = eye_hv[1] - brain_params.listing_primary[1]
    T_cyc = brain_params.listing_l2_frac * phi * dV * HALF_ANGLE
    # Drive torsional component of e_disp toward the L2-demanded cyclovergence
    e_disp_aug = e_disp.at[2].add(T_cyc - x_verg[2])

    # ── Slow fusional drive ───────────────────────────────────────────────────
    e_total = e_disp_aug + jnp.array([ac_a_drive, 0.0, 0.0])
    e_fus  = _disp_sat(e_total, v_sat=brain_params.panum_h)
    # Proximal (coarse) drive: smooth per-axis saturation (cosine rolloff → 0 at 2× limit).
    # H uses prox_sat; V/T use panum_v/t. _disp_sat on a scalar = abs-based gain; no hard clip.
    # Symmetric: divergence motor failure is handled by the diplopia gate (brain_model), not here.
    e_prox = jnp.array([
        _disp_sat(e_total[0:1], v_sat=brain_params.prox_sat)[0],
        _disp_sat(e_total[1:2], v_sat=brain_params.panum_v)[0],
        _disp_sat(e_total[2:3], v_sat=brain_params.panum_t)[0],
    ])
    e_pred = brain_params.K_verg * e_fus + brain_params.K_verg_prox * e_prox

    tonic = jnp.array([brain_params.tonic_verg, 0.0, 0.0])   # V/T tonic ≈ 0
    dx_verg_slow = ((-(1.0 / brain_params.tau_verg) * (x_verg - tonic)
                    + e_pred) / (1.0 + brain_params.D_verg))
    u_verg_slow  = x_verg + brain_params.K_phasic_verg * e_fus

    # ── Zee saccade pulse ─────────────────────────────────────────────────────
    # e_held_verg: tracks disparity when idle (τ=5ms), frozen during saccade
    de_held_verg = (e_disp_aug - e_held_verg) / _TAU_HOLD_VERG * z_idl

    # Ballistic burst: fires while residual remains, stops when copy reaches held
    e_res_verg     = e_held_verg - x_copy_verg
    e_res_verg_mag = jnp.linalg.norm(e_res_verg)
    gate_res_verg  = jax.nn.sigmoid(
        brain_params.k_sac * (e_res_verg_mag - brain_params.threshold_stop))

    u_verg_burst = z_act * gate_res_verg * brain_params.g_burst_verg * e_res_verg
    # Between saccades, x_copy decays to 0 so successive saccades each get a fresh baseline;
    # without decay, conv→div saccades carry over the accumulated copy and mis-size the burst.
    dx_copy_verg = u_verg_burst - z_idl * x_copy_verg / _TAU_COPY_RESET

    # ── Combined output ───────────────────────────────────────────────────────
    u_verg  = u_verg_slow + u_verg_burst
    dx_verg = jnp.concatenate([dx_verg_slow, de_held_verg, dx_copy_verg])
    return dx_verg, u_verg
