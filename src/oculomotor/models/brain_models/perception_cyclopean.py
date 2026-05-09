"""Cyclopean perception — binocular fusion of per-eye DELAYED retinal signals
+ brain LP smoothing.

Inputs: RetinaOut_L, RetinaOut_R (already delayed by retina sharp cascade,
sensor-saturated, visibility-gated).
Outputs: cyclopean fused + brain-LP-smoothed signals (post-fusion delay block
of 43 states).

Three binocular policies:
    binocular_fusion_policy — NPC gate + dominance for target pos/disparity
    binocular_okr_policy    — visibility-weighted optic flow average

Brain LP state layout (post-fusion smoothing):
    scene_angular_vel : 1 LP × 3 = 3   (τ_motion ≈ 20 ms)
    scene_linear_vel  : 1 LP × 3 = 3
    target_pos        : N stages × 3 = 18 (multi-stage gamma, τ_brain_pos ≈ 30 ms)
    target_vel        : 1 LP × 3 = 3   (τ_target_vel ≈ 150 ms)
    target_disparity  : 1 LP × 3 = 3   (τ_disp ≈ 150 ms)
    scene_visible     : N stages × 1 = 6
    target_visible    : N stages × 1 = 6
    defocus           : 1 LP × 1 = 1   (τ_defocus ≈ 200 ms)
    Total: 43

target_disparity is computed POST-RETINA-CASCADE from per-eye DELAYED target_pos,
since sharp delay happens upstream in retina.step.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oculomotor.models.sensory_models.retina import (
    delay_cascade_step,
    _N_STAGES_OTHER,
)


# ── Brain LP state layout (post-fusion smoothing) ──────────────────────────────
# (name, N_lp_stages, n_axes)
_CYC_BRAIN_LAYOUT = [
    ('scene_angular_vel',  1,                3),  # 1-pole LP
    ('scene_linear_vel',   1,                3),  # 1-pole LP
    ('target_pos',         _N_STAGES_OTHER,  3),  # N-stage gamma (matches sharp)
    ('target_vel',         1,                3),  # 1-pole LP (long)
    ('target_disparity',   1,                3),  # 1-pole LP (V1 stereo is slow)
    ('scene_visible',      _N_STAGES_OTHER,  1),  # N-stage gamma
    ('target_visible',     _N_STAGES_OTHER,  1),  # N-stage gamma
    ('defocus',            1,                1),  # 1-pole LP
]

_CYC_BRAIN_SIZES   = {name: N * n for name, N, n in _CYC_BRAIN_LAYOUT}
_CYC_BRAIN_OFFSETS = {}
_offset = 0
for name, N, n in _CYC_BRAIN_LAYOUT:
    _CYC_BRAIN_OFFSETS[name] = _offset
    _offset += _CYC_BRAIN_SIZES[name]
N_STATES = _offset   # 43

_OFF_SCENE_ANGULAR_VEL = _CYC_BRAIN_OFFSETS['scene_angular_vel']
_OFF_SCENE_LINEAR      = _CYC_BRAIN_OFFSETS['scene_linear_vel']
_OFF_TARGET_POS        = _CYC_BRAIN_OFFSETS['target_pos']
_OFF_TARGET_VEL        = _CYC_BRAIN_OFFSETS['target_vel']
_OFF_TARGET_DISP       = _CYC_BRAIN_OFFSETS['target_disparity']
_OFF_SCENE_VIS         = _CYC_BRAIN_OFFSETS['scene_visible']
_OFF_TARGET_VIS        = _CYC_BRAIN_OFFSETS['target_visible']
_OFF_DEFOCUS           = _CYC_BRAIN_OFFSETS['defocus']

_END_SCENE_ANGULAR_VEL = _OFF_SCENE_ANGULAR_VEL + _CYC_BRAIN_SIZES['scene_angular_vel']
_END_SCENE_LINEAR      = _OFF_SCENE_LINEAR      + _CYC_BRAIN_SIZES['scene_linear_vel']
_END_TARGET_POS        = _OFF_TARGET_POS        + _CYC_BRAIN_SIZES['target_pos']
_END_TARGET_VEL        = _OFF_TARGET_VEL        + _CYC_BRAIN_SIZES['target_vel']
_END_TARGET_DISP       = _OFF_TARGET_DISP       + _CYC_BRAIN_SIZES['target_disparity']
_END_SCENE_VIS         = _OFF_SCENE_VIS         + _CYC_BRAIN_SIZES['scene_visible']
_END_TARGET_VIS        = _OFF_TARGET_VIS        + _CYC_BRAIN_SIZES['target_visible']
_END_DEFOCUS           = _OFF_DEFOCUS           + _CYC_BRAIN_SIZES['defocus']


# ── Readout matrices — read into x_cyc_brain block (size N_STATES) ─────────────

def _make_C_last_n(end_idx, n_axes):
    """Selects last n_axes elements of a signal block within x_cyc_brain."""
    C = jnp.zeros((n_axes, N_STATES))
    for i in range(n_axes):
        C = C.at[i, end_idx - n_axes + i].set(1.0)
    return C


C_slip             = _make_C_last_n(_END_SCENE_ANGULAR_VEL, 3)
C_scene_linear_vel = _make_C_last_n(_END_SCENE_LINEAR,      3)
C_pos              = _make_C_last_n(_END_TARGET_POS,        3)
C_vel              = _make_C_last_n(_END_TARGET_VEL,        3)
C_target_disp      = _make_C_last_n(_END_TARGET_DISP,       3)
C_scene_visible    = _make_C_last_n(_END_SCENE_VIS,         1)
C_target_visible   = _make_C_last_n(_END_TARGET_VIS,        1)
C_defocus          = _make_C_last_n(_END_DEFOCUS,           1)


# ── Cyclopean output bundle ────────────────────────────────────────────────────

class CyclopeanOut(NamedTuple):
    """Cyclopean fused + brain-LP-smoothed delayed signals."""
    scene_angular_vel: jnp.ndarray  # (3,) deg/s
    scene_linear_vel:  jnp.ndarray  # (3,) m/s, head frame
    target_pos:        jnp.ndarray  # (3,) deg
    target_vel:        jnp.ndarray  # (3,) deg/s
    target_disparity:  jnp.ndarray  # (3,) deg
    scene_visible:     jnp.ndarray  # scalar
    target_visible:    jnp.ndarray  # scalar
    target_fusable:    jnp.ndarray  # scalar — algebraic fusion gate (NOT cascaded)
    defocus:           jnp.ndarray  # scalar


# ── Binocular policies ─────────────────────────────────────────────────────────

def binocular_fusion_policy(target_pos_L, target_vel_L, target_vis_L,
                            target_pos_R, target_vel_R, target_vis_R,
                            ec_pos, ec_verg, brain_params):
    """NPC gate + eye-dominance weighting → blend weights for target signals."""
    _ = target_vel_L, target_vel_R, ec_pos   # reserved for future extensions

    bino_raw = target_vis_L * target_vis_R
    raw_disp = bino_raw * (target_pos_L - target_pos_R)

    total_demand_h = raw_disp[0] + ec_verg[0]
    total_demand_v = raw_disp[1] + ec_verg[1]
    total_demand_t = raw_disp[2] + ec_verg[2]
    gate_conv = jax.nn.sigmoid(100.0 * (brain_params.npc      - total_demand_h))
    gate_div  = jax.nn.sigmoid(100.0 * (brain_params.div_max  + total_demand_h ))
    gate_vert = jax.nn.sigmoid(100.0 * (brain_params.vert_max - jnp.abs(total_demand_v)))
    gate_tors = jax.nn.sigmoid(100.0 * (brain_params.tors_max - jnp.abs(total_demand_t)))
    bino_fusable    = gate_conv * gate_div * gate_vert * gate_tors
    target_fusable  = bino_raw * bino_fusable
    _equal_weight   = jnp.maximum(target_fusable, 1.0 - bino_raw)

    dom_L = 1.0 - brain_params.eye_dominant
    dom_R = brain_params.eye_dominant
    w_L = target_vis_L * (_equal_weight + (1.0 - _equal_weight) * dom_L)
    w_R = target_vis_R * (_equal_weight + (1.0 - _equal_weight) * dom_R)

    target_visible = jnp.clip(w_L + w_R, 0.0, 1.0)
    norm = jnp.maximum(w_L + w_R, 1e-6)
    return w_L / norm, w_R / norm, raw_disp, target_fusable, target_visible


def binocular_okr_policy(scene_vis_L, scene_vis_R):
    """Visibility-weighted scene fusion → per-eye scene blend weights.

    OKR has no NPC gate and no eye dominance — both eyes contribute equally
    when the scene is present. Cyclopean scene visibility = probabilistic OR.
    """
    scene_visible = 1.0 - (1.0 - scene_vis_L) * (1.0 - scene_vis_R)
    norm = jnp.maximum(scene_vis_L + scene_vis_R, 1e-6)
    return scene_vis_L / norm, scene_vis_R / norm, scene_visible


# ── Step ───────────────────────────────────────────────────────────────────────

def step(x_cyc_brain, retina_L, retina_R, ec_pos, ec_verg, brain_params):
    """Cyclopean perception step: fuse per-eye DELAYED signals + brain LP smoothing.

    Args:
        x_cyc_brain:  (43,) brain LP cascade state for cyclopean signals
        retina_L:     RetinaOut for left eye  (delayed per-eye signals)
        retina_R:     RetinaOut for right eye
        ec_pos:       (3,) version eye position (deg) — used by fusion policy
        ec_verg:      (3,) vergence command (deg) — used by NPC gate
        brain_params: BrainParams — reads npc/div_max/vert_max/tors_max,
                      eye_dominant, tau_vis_smooth_motion, tau_vis_smooth_target_vel,
                      tau_vis_smooth_disparity, tau_vis_smooth_defocus, tau_brain_pos.

    Returns:
        dx_cyc_brain: (43,) state derivative
        cyc:          CyclopeanOut bundle of delayed cyclopean signals
    """
    # ── 1. Binocular policies on DELAYED per-eye signals ──────────────────────
    w_s_L, w_s_R, scene_visible_cyc = binocular_okr_policy(
        retina_L.scene_visible, retina_R.scene_visible)
    scene_angular_cyc = w_s_L * retina_L.scene_angular_vel + w_s_R * retina_R.scene_angular_vel
    scene_linear_cyc  = w_s_L * retina_L.scene_linear_vel  + w_s_R * retina_R.scene_linear_vel

    w_L, w_R, raw_disp, target_fusable, target_visible_cyc = binocular_fusion_policy(
        retina_L.target_pos, retina_L.target_vel, retina_L.target_visible,
        retina_R.target_pos, retina_R.target_vel, retina_R.target_visible,
        ec_pos, ec_verg, brain_params)

    # Disparity drive: both eyes must contribute (bino_raw) AND fusion-gate-proxy
    # (target_fusable already includes bino_raw). Scale by 1−|w_L−w_R| so
    # disparity collapses to zero when one eye dominates (true diplopia).
    disp_visibility  = target_fusable * (1.0 - jnp.abs(w_L - w_R))
    target_disparity_cyc = raw_disp * disp_visibility

    target_pos_cyc = w_L * retina_L.target_pos + w_R * retina_R.target_pos
    target_vel_cyc = w_L * retina_L.target_vel + w_R * retina_R.target_vel

    # Defocus fusion uses RAW per-eye target visibility (not the fusion-gated
    # w_L/w_R), so a monocularly visible target still drives accommodation
    # via the seeing eye's defocus when fusion is disabled.
    defocus_visible = 1.0 - (1.0 - scene_visible_cyc) * (1.0 - target_visible_cyc)
    def_norm = retina_L.target_visible + retina_R.target_visible + 1e-6
    w_def_L  = retina_L.target_visible / def_norm
    w_def_R  = retina_R.target_visible / def_norm
    defocus_cyc = (w_def_L * retina_L.defocus + w_def_R * retina_R.defocus) * defocus_visible

    # ── 2. Brain LP cascades ──────────────────────────────────────────────────
    tau_motion     = brain_params.tau_vis_smooth_motion
    tau_target_vel = brain_params.tau_vis_smooth_target_vel
    tau_disparity  = brain_params.tau_vis_smooth_disparity
    tau_defocus    = brain_params.tau_vis_smooth_defocus
    tau_brain_pos  = brain_params.tau_brain_pos
    N              = _N_STAGES_OTHER

    x_scene_ang  = x_cyc_brain[_OFF_SCENE_ANGULAR_VEL : _END_SCENE_ANGULAR_VEL]
    x_scene_lin  = x_cyc_brain[_OFF_SCENE_LINEAR      : _END_SCENE_LINEAR]
    x_target_pos = x_cyc_brain[_OFF_TARGET_POS        : _END_TARGET_POS]
    x_target_vel = x_cyc_brain[_OFF_TARGET_VEL        : _END_TARGET_VEL]
    x_target_disp= x_cyc_brain[_OFF_TARGET_DISP       : _END_TARGET_DISP]
    x_scene_vis  = x_cyc_brain[_OFF_SCENE_VIS         : _END_SCENE_VIS]
    x_target_vis = x_cyc_brain[_OFF_TARGET_VIS        : _END_TARGET_VIS]
    x_defocus    = x_cyc_brain[_OFF_DEFOCUS           : _END_DEFOCUS]

    dx_scene_ang   = delay_cascade_step(x_scene_ang,  scene_angular_cyc,    tau_motion,     N=1)
    dx_scene_lin   = delay_cascade_step(x_scene_lin,  scene_linear_cyc,     tau_motion,     N=1)
    dx_target_pos  = delay_cascade_step(x_target_pos, target_pos_cyc,       tau_brain_pos,  N=N)
    dx_target_vel  = delay_cascade_step(x_target_vel, target_vel_cyc,       tau_target_vel, N=1)
    dx_target_disp = delay_cascade_step(x_target_disp,target_disparity_cyc, tau_disparity,  N=1)
    dx_scene_vis   = delay_cascade_step(x_scene_vis,  scene_visible_cyc,    tau_brain_pos,  N=N)
    dx_target_vis  = delay_cascade_step(x_target_vis, target_visible_cyc,   tau_brain_pos,  N=N)
    dx_defocus     = delay_cascade_step(x_defocus,    defocus_cyc,          tau_defocus,    N=1)

    dx_cyc_brain = jnp.concatenate([
        dx_scene_ang, dx_scene_lin, dx_target_pos, dx_target_vel,
        dx_target_disp, dx_scene_vis, dx_target_vis, dx_defocus,
    ])

    # ── 3. Read delayed cyclopean signals (last n_axes of each block) ────────
    cyc = CyclopeanOut(
        scene_angular_vel = x_scene_ang[-3:],
        scene_linear_vel  = x_scene_lin[-3:],
        target_pos        = x_target_pos[-3:],
        target_vel        = x_target_vel[-3:],
        target_disparity  = x_target_disp[-3:],
        scene_visible     = x_scene_vis[-1],
        target_visible    = x_target_vis[-1],
        target_fusable    = target_fusable,
        defocus           = x_defocus[-1],
    )
    return dx_cyc_brain, cyc
