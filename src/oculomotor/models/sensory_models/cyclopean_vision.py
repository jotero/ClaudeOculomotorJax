"""Cyclopean vision — binocular pre-delay fusion + single cyclopean delay cascade.

step() fuses per-eye instantaneous retinal signals into a cyclopean representation
BEFORE the visual delay cascade, then advances the single 720-state cascade.

Three binocular policies handle the three visual subsystems:
    binocular_saccade_policy  — NPC gate + dominance for position/disparity
    binocular_pursuit_policy  — fusion-gated motion visibility for velocity tracking
    binocular_okr_policy      — visibility-weighted optic flow average (no NPC gate)

State layout (720 states):
    [scene_angular_vel(120) | scene_linear_vel(120) | target_pos(120)
     | target_vel(120) | target_disparity(120)
     | scene_visible(40) | target_visible(40) | target_motion_visible(40)]
"""

import jax
import jax.numpy as jnp

from oculomotor.models.sensory_models.retina import (
    delay_cascade_step, ypr_to_xyz,
    _N_PER_SIG,
    _OFF_SCENE_LINEAR, _OFF_TARGET_POS, _OFF_TARGET_VEL, _OFF_TARGET_DISP,
    _OFF_SCENE_VIS, _OFF_TARGET_VIS, _OFF_STROBED, _OFF_TARGET_FUSABLE,
)
from oculomotor.models.plant_models.readout import rotation_matrix


# ── Velocity saturation ─────────────────────────────────────────────────────────

def velocity_saturation(v, v_sat, v_zero=None, v_offset=None):
    """Smooth velocity saturation: passes at low speed, gain ramps to zero at high speed.

    For a velocity vector v:
        |v| ≤ v_sat          → output = v           (gain = 1)
        v_sat < |v| < v_zero → output = v * gain    (cosine rolloff, 1 → 0)
        |v| ≥ v_zero         → output = 0            (gain = 0)

    The cosine rolloff keeps gain and its derivative continuous at both endpoints.
    MT/MST neurons are band-pass tuned for speed (peak ~10–40 deg/s, near-zero
    above ~100 deg/s).  NOT/AOS broader tuning peaks ~40–80 deg/s.  This function
    faithfully models that insensitivity to implausibly fast retinal motion, as
    distinct from jnp.clip which would keep driving at ±v_sat for large inputs.

    Background-shifted saturation (v_offset): when supplied, the saturation window
    is centred on v_offset (v_rel = v − v_offset), ensuring EC cancellation holds
    even when the eye is already moving (OKN fast phases, saccades during pursuit).

    Args:
        v:        (N,) velocity vector (deg/s); norm computed over the full vector
        v_sat:    saturation onset (deg/s) — gain is exactly 1 below this
        v_zero:   speed where gain reaches 0 (deg/s); default = 2 × v_sat
        v_offset: (N,) background velocity to shift the clip window (deg/s)

    Returns:
        Same shape as v, scaled by smooth gain ∈ [0, 1], plus v_offset if given.
    """
    if v_zero is None:
        v_zero = 2.0 * v_sat
    if v_offset is not None:
        v_rel = v - v_offset
    else:
        v_rel = v
    speed = jnp.linalg.norm(v_rel)
    t     = jnp.clip((speed - v_sat) / (v_zero - v_sat), 0.0, 1.0)
    gain  = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    result = v_rel * gain
    if v_offset is not None:
        result = result + v_offset
    return result


# ── Binocular policies ───────────────────────────────────────────────────────────

def binocular_saccade_policy(target_pos_L, target_vis_L, target_pos_R, target_vis_R,
                              verg_h, sensory_params):
    """NPC gate + eye-dominance weighting for position/disparity signals.

    Determines whether the two retinal images can be fused given the current vergence
    demand and motor limits.  In diplopia the dominant eye drives position; in fusion
    both eyes contribute proportionally to their visibility.

    Three-state binocular model (horizontal):
        Fused    (demand within motor range): both eyes weighted by visibility
        Diplopic (demand exceeds npc or −div_max): dominant eye only

    Vertical / torsional gates suppress fusion when those disparities are also large.

    Args:
        target_pos_L:   (3,) target direction [yaw,pitch,0] (deg) — left eye
        target_vis_L:   scalar target gate = target_present × target_in_vf ∈ [0,1]
        target_pos_R:   (3,) target direction [yaw,pitch,0] (deg) — right eye
        target_vis_R:   scalar target gate ∈ [0,1]
        verg_h:         scalar current vergence angle (deg, H) — from ec_verg[0]
        sensory_params: SensoryParams — reads npc, div_max, vert_max, tors_max, eye_dominant

    Returns:
        w_L:             scalar left-eye blend weight ∈ [0, 1]
        w_R:             scalar right-eye blend weight ∈ [0, 1]
        target_disparity:(3,) diplopia-gated vergence disparity (deg)
        target_fusable:  scalar fusion gate ∈ [0, 1] — pass to binocular_pursuit_policy
    """
    bino_raw = target_vis_L * target_vis_R
    raw_disp = bino_raw * (target_pos_L - target_pos_R)

    total_demand_h = raw_disp[0] + verg_h
    gate_conv = jax.nn.sigmoid(100.0 * (sensory_params.npc      - total_demand_h))
    gate_div  = jax.nn.sigmoid(100.0 * (total_demand_h          + sensory_params.div_max))
    gate_vert = jax.nn.sigmoid(100.0 * (sensory_params.vert_max - jnp.abs(raw_disp[1])))
    gate_tors = jax.nn.sigmoid(100.0 * (sensory_params.tors_max - jnp.abs(raw_disp[2])))
    target_fusable = gate_conv * gate_div * gate_vert * gate_tors

    dom_L = 1.0 - sensory_params.eye_dominant
    dom_R = sensory_params.eye_dominant
    w_L = target_vis_L * (target_fusable + (1.0 - target_fusable) * dom_L)
    w_R = target_vis_R * (target_fusable + (1.0 - target_fusable) * dom_R)

    return w_L, w_R, target_fusable * raw_disp, target_fusable


def binocular_pursuit_policy(target_motion_vis_L, target_motion_vis_R, target_fusable, sensory_params):
    """Apply fusion gate to strobe-gated motion visibility → per-eye velocity blend weights.

    Pursuit does not re-evaluate the NPC gate — fusion state is inherited from
    binocular_saccade_policy.  In diplopia the dominant eye drives velocity tracking;
    in fusion both eyes contribute proportionally to their (strobe-gated) motion visibility.

    Args:
        target_motion_vis_L: scalar motion gate = target_vis × (1−strobe) ∈ [0,1] — left
        target_motion_vis_R: scalar motion gate ∈ [0,1] — right
        target_fusable:      scalar fusion gate from binocular_saccade_policy ∈ [0,1]
        sensory_params:      SensoryParams — reads eye_dominant

    Returns:
        w_m_L: scalar left-eye motion blend weight ∈ [0, 1]
        w_m_R: scalar right-eye motion blend weight ∈ [0, 1]
    """
    dom_L = 1.0 - sensory_params.eye_dominant
    dom_R = sensory_params.eye_dominant
    w_m_L = target_motion_vis_L * (target_fusable + (1.0 - target_fusable) * dom_L)
    w_m_R = target_motion_vis_R * (target_fusable + (1.0 - target_fusable) * dom_R)
    return w_m_L, w_m_R


def binocular_okr_policy(scene_vis_L, scene_vis_R):
    """Visibility-weighted optic flow average → per-eye scene blend weights + cyclopean gate.

    OKR is driven by the background scene, not a foveated target.  There is no NPC gate
    and no eye dominance — both eyes contribute equally whenever the scene is present.
    Cyclopean scene visibility follows a probabilistic OR: scene_visible = 1 − (1−L)(1−R).

    Args:
        scene_vis_L: scalar scene presence gate = scene_present ∈ [0,1] — left eye
        scene_vis_R: scalar scene presence gate ∈ [0,1] — right eye

    Returns:
        w_L:          scalar left-eye scene weight (= scene_vis_L)
        w_R:          scalar right-eye scene weight (= scene_vis_R)
        scene_visible:scalar cyclopean scene gate ∈ [0,1]
    """
    scene_visible = 1.0 - (1.0 - scene_vis_L) * (1.0 - scene_vis_R)
    return scene_vis_L, scene_vis_R, scene_visible


def step(x_vis,
         scene_angular_vel_L, scene_linear_vel_L, target_pos_L, target_vel_L, scene_vis_L, target_vis_L, target_motion_vis_L,
         scene_angular_vel_R, scene_linear_vel_R, target_pos_R, target_vel_R, scene_vis_R, target_vis_R, target_motion_vis_R,
         sensory_params,
         ec_vel, ec_pos, ec_verg):
    """Fuse per-eye retinal signals, apply EC correction, then advance the cyclopean cascade.

    EC subtraction happens pre-delay: u_version (the version motor command rotated into eye
    frame) is added to the instantaneous retinal slip before the cascade.  This is
    mathematically equivalent to the old post-delay approach (delay is linear) but saves
    the two 120-state EC delay cascades entirely.

    All visibility inputs are pre-gated — this function only receives _vis signals:
        scene_vis_L/R        = scene_present            (from retina)
        target_vis_L/R       = target_present × in_vf   (from retina)
        target_motion_vis_L/R = target_vis × (1−strobe)  (from sensory_model)

    Args:
        x_vis:                  (720,)  cyclopean cascade state
        scene_angular_vel_L/R:   (3,)  rotational optic flow per eye (deg/s)
        scene_linear_vel_L/R:    (3,)  translational optic flow per eye (m/s, eye frame)
        target_pos_L/R:          (3,)  target direction [yaw,pitch,0] (deg)
        target_vel_L/R:          (3,)  target velocity on retina [yaw,pitch,roll] (deg/s)
        scene_vis_L/R:           scalar scene presence gate = scene_present ∈ [0,1]
        target_vis_L/R:          scalar target gate = target_present × target_in_vf ∈ [0,1]
        target_motion_vis_L/R:   scalar motion gate = target_vis × (1−strobe) ∈ [0,1]
        sensory_params:          SensoryParams — reads npc, div_max, vert_max, tors_max,
                                 eye_dominant, v_max_scene_vel, v_max_target_vel, tau_vis
        ec_vel:                  (3,) version velocity efference [yaw,pitch,roll] (deg/s)
                                 rotated head→eye frame internally via rotation_matrix(ypr_to_xyz(ec_pos))
        ec_pos:                  (3,) eye position efference [yaw,pitch,roll] (deg) = x_ni_net + ocr
        ec_verg:                 (3,) vergence efference [H,V,T] (deg); [0] used for NPC gate

    Returns:
        dx_vis: (720,)  cascade state derivative
    """
    # Rotate version velocity efference head→eye frame for EC correction.
    u_version = rotation_matrix(ypr_to_xyz(ec_pos)).T @ ec_vel

    # ── Scene (OKR) ───────────────────────────────────────────────────────────
    w_s_L, w_s_R, scene_visible = binocular_okr_policy(scene_vis_L, scene_vis_R)
    sv_norm           = jnp.maximum(w_s_L + w_s_R, 1e-6)
    scene_angular_cyc = (w_s_L * scene_angular_vel_L + w_s_R * scene_angular_vel_R) / sv_norm
    scene_linear_cyc  = (w_s_L * scene_linear_vel_L  + w_s_R * scene_linear_vel_R)  / sv_norm

    # EC correction pre-delay: add motor command to instantaneous slip before cascade.
    # delay(slip + u_motor) = delay(slip) + delay(u_motor) — same as old post-delay formulation.
    scene_angular_vel = velocity_saturation((scene_angular_cyc + u_version) * scene_visible, sensory_params.v_max_scene_vel)
    scene_linear_vel  = scene_linear_cyc * scene_visible

    # ── Target position / disparity (saccade) ─────────────────────────────────
    w_L, w_R, target_disparity, target_fusable = binocular_saccade_policy(
        target_pos_L, target_vis_L, target_pos_R, target_vis_R, ec_verg[0], sensory_params)

    target_visible = jnp.clip(w_L + w_R, 0.0, 1.0)
    w_norm         = jnp.maximum(w_L + w_R, 1e-6)
    target_pos     = (w_L * target_pos_L + w_R * target_pos_R) / w_norm

    # ── Target velocity (pursuit) ──────────────────────────────────────────────
    w_m_L, w_m_R = binocular_pursuit_policy(
        target_motion_vis_L, target_motion_vis_R, target_fusable, sensory_params)

    target_motion_visible = jnp.clip(w_m_L + w_m_R, 0.0, 1.0)
    u_version_notors      = u_version.at[2].set(0.0)   # torsion zeroed — retina is 2D
    w_m_norm              = jnp.maximum(w_m_L + w_m_R, 1e-6)
    target_vel_cyc        = (w_m_L * target_vel_L + w_m_R * target_vel_R) / w_m_norm
    target_slip = velocity_saturation(
        target_vel_cyc + u_version_notors * target_motion_visible,
        sensory_params.v_max_target_vel)

    # ── Advance cascade ───────────────────────────────────────────────────────
    tau_vis         = sensory_params.tau_vis
    x_scene_angular  = x_vis[                       :  _N_PER_SIG          ]
    x_scene_linear   = x_vis[_OFF_SCENE_LINEAR      : _OFF_TARGET_POS      ]
    x_target_pos     = x_vis[_OFF_TARGET_POS        : _OFF_TARGET_VEL      ]
    x_target_vel     = x_vis[_OFF_TARGET_VEL        : _OFF_TARGET_DISP     ]
    x_target_disp    = x_vis[_OFF_TARGET_DISP       : _OFF_SCENE_VIS       ]
    x_scene_vis      = x_vis[_OFF_SCENE_VIS         : _OFF_TARGET_VIS      ]
    x_target_vis     = x_vis[_OFF_TARGET_VIS        : _OFF_STROBED         ]
    x_target_motion  = x_vis[_OFF_STROBED           : _OFF_TARGET_FUSABLE  ]
    x_target_fusable = x_vis[_OFF_TARGET_FUSABLE    :                      ]

    return jnp.concatenate([
        delay_cascade_step(x_scene_angular,   scene_angular_vel,    tau_vis),
        delay_cascade_step(x_scene_linear,    scene_linear_vel,     tau_vis),
        delay_cascade_step(x_target_pos,      target_pos,           tau_vis),
        delay_cascade_step(x_target_vel,      target_slip,          tau_vis),
        delay_cascade_step(x_target_disp,     target_disparity,     tau_vis),
        delay_cascade_step(x_scene_vis,       scene_visible,        tau_vis),
        delay_cascade_step(x_target_vis,      target_visible,       tau_vis),
        delay_cascade_step(x_target_motion,   target_motion_visible, tau_vis),
        delay_cascade_step(x_target_fusable,  target_fusable,       tau_vis),
    ])
