"""Cyclopean vision — binocular pre-delay fusion + single cyclopean delay cascade.

step() fuses per-eye instantaneous retinal signals into a cyclopean representation
BEFORE the visual delay cascade, then advances the single 720-state cascade.

Fusion applies visibility weighting, diplopia suppression (NPC gate), and velocity
saturation.  This is more principled than post-delay fusion — vergence demand is
computed from instantaneous disparity (not delayed disparity), and all cascade
inputs share the same frame of reference.

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
    _OFF_SCENE_VIS, _OFF_TARGET_VIS, _OFF_STROBED,
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


def step(x_vis,
         scene_angular_vel_L, scene_linear_vel_L, target_pos_L, target_vel_L, target_in_vf_L,
         scene_angular_vel_R, scene_linear_vel_R, target_pos_R, target_vel_R, target_in_vf_R,
         scene_present_L, target_present_L,
         scene_present_R, target_present_R,
         target_strobed, sensory_params,
         ec_vel, ec_pos, ec_verg):
    """Fuse per-eye retinal signals, apply EC correction, then advance the cyclopean cascade.

    EC subtraction happens pre-delay: u_version (the version motor command rotated into eye
    frame) is added to the instantaneous retinal slip before the cascade.  This is
    mathematically equivalent to the old post-delay approach (delay is linear) but saves
    the two 120-state EC delay cascades entirely.

    Args:
        x_vis:               (720,)  cyclopean cascade state
        scene_angular_vel_L/R: (3,)  rotational optic flow per eye (deg/s)
        scene_linear_vel_L/R:  (3,)  translational optic flow per eye (m/s, eye frame)
        target_pos_L/R:        (3,)  target direction [yaw,pitch,0] (deg)
        target_vel_L/R:        (3,)  target velocity on retina [yaw,pitch,roll] (deg/s)
        target_in_vf_L/R:      scalar geometric visual-field gate ∈ [0,1]
        scene_present_L/R:     scalar 0=dark, 1=lit
        target_present_L/R:    scalar 0=occluded, 1=visible
        target_strobed:        scalar 1=stroboscopic
        sensory_params:        SensoryParams — reads npc, div_max, vert_max, tors_max,
                               eye_dominant, v_max_scene_vel, v_max_target_vel, tau_vis
        ec_vel:                (3,) version velocity efference [yaw,pitch,roll] (deg/s) = u_burst + u_pursuit
                               rotated head→eye frame internally via rotation_matrix(ypr_to_xyz(ec_pos))
        ec_pos:                (3,) eye position efference [yaw,pitch,roll] (deg) = x_ni_net + ocr
        ec_verg:               (3,) vergence efference [H,V,T] (deg) = x_verg; [0] used for NPC gate

    Returns:
        dx_vis: (720,)  cascade state derivative
    """
    # Rotate version velocity efference head→eye frame for EC correction.
    u_version = rotation_matrix(ypr_to_xyz(ec_pos)).T @ ec_vel

    # ── Scene: average by scene_present (instantaneous) ──────────────────────
    sv_L, sv_R = scene_present_L, scene_present_R
    sv_norm           = jnp.maximum(sv_L + sv_R, 1e-6)
    scene_angular_cyc = (sv_L * scene_angular_vel_L + sv_R * scene_angular_vel_R) / sv_norm
    scene_linear_cyc  = (sv_L * scene_linear_vel_L  + sv_R * scene_linear_vel_R)  / sv_norm
    scene_visible     = jnp.clip(sv_L + sv_R, 0.0, 1.0)

    # EC correction pre-delay: add motor command to instantaneous slip before cascade.
    # delay(slip + u_motor) = delay(slip) + delay(u_motor) — same as old post-delay formulation.
    scene_angular_vel = velocity_saturation((scene_angular_cyc + u_version) * scene_visible, sensory_params.v_max_scene_vel)
    scene_linear_vel  = scene_linear_cyc * scene_visible

    # ── Target: per-eye visibility + diplopia suppression ────────────────────
    tv_L     = target_present_L * target_in_vf_L
    tv_R     = target_present_R * target_in_vf_R
    bino_raw = tv_L * tv_R
    raw_disp = bino_raw * (target_pos_L - target_pos_R)

    # NPC gate: suppresses vergence and version when total demand exceeds motor limit
    # Three-state binocular model (horizontal; vertical/torsional use vert_max / tors_max):
    #   Fused    (|disp_H| < panum_h ≈ 2°):  fine fusional drive; fuse=1
    #   Fusable  (panum_h < |disp_H|, total in [−div_max, npc]): coarse proximal drive; fuse=1
    #   Diplopic (total_H > npc  or  < −div_max): vergence suppressed, dominant eye; fuse=0
    d_h            = raw_disp[0]
    total_demand_h = d_h + ec_verg[0]
    gate_conv = jax.nn.sigmoid(100.0 * (sensory_params.npc      - total_demand_h))
    gate_div  = jax.nn.sigmoid(100.0 * (total_demand_h          + sensory_params.div_max))
    gate_vert = jax.nn.sigmoid(100.0 * (sensory_params.vert_max - jnp.abs(raw_disp[1])))
    gate_tors = jax.nn.sigmoid(100.0 * (sensory_params.tors_max - jnp.abs(raw_disp[2])))
    fuse             = gate_conv * gate_div * gate_vert * gate_tors
    target_disparity = fuse * raw_disp

    dom_L  = 1.0 - sensory_params.eye_dominant
    dom_R  = sensory_params.eye_dominant
    tv_L_s = fuse * tv_L + (1.0 - fuse) * tv_L * dom_L   # diplopia-suppressed
    tv_R_s = fuse * tv_R + (1.0 - fuse) * tv_R * dom_R
    target_visible = jnp.clip(tv_L_s + tv_R_s, 0.0, 1.0)

    # Version position: dominant eye in diplopia, cyclopean in fusion
    tv_norm_s  = jnp.maximum(tv_L_s + tv_R_s, 1e-6)
    target_pos = (tv_L_s * target_pos_L + tv_R_s * target_pos_R) / tv_norm_s

    # Target velocity: binocular average, EC-corrected pre-delay, then strobe gate + saturation
    tv_norm          = jnp.maximum(tv_L + tv_R, 1e-6)
    u_version_notors = u_version.at[2].set(0.0)   # torsion zeroed — retina is 2D
    target_vel_cyc   = (tv_L * target_vel_L + tv_R * target_vel_R) / tv_norm
    target_slip = velocity_saturation(
        (target_vel_cyc + u_version_notors * target_visible) * (1.0 - target_strobed),
        sensory_params.v_max_target_vel)

    target_motion_visible = target_visible * (1.0 - target_strobed)

    # ── Advance cascade ───────────────────────────────────────────────────────
    tau_vis         = sensory_params.tau_vis
    x_scene_angular = x_vis[                  :  _N_PER_SIG      ]
    x_scene_linear  = x_vis[_OFF_SCENE_LINEAR : _OFF_TARGET_POS  ]
    x_target_pos    = x_vis[_OFF_TARGET_POS   : _OFF_TARGET_VEL  ]
    x_target_vel    = x_vis[_OFF_TARGET_VEL   : _OFF_TARGET_DISP ]
    x_target_disp   = x_vis[_OFF_TARGET_DISP  : _OFF_SCENE_VIS   ]
    x_scene_vis     = x_vis[_OFF_SCENE_VIS    : _OFF_TARGET_VIS  ]
    x_target_vis    = x_vis[_OFF_TARGET_VIS   : _OFF_STROBED     ]
    x_target_motion = x_vis[_OFF_STROBED      :                  ]

    return jnp.concatenate([
        delay_cascade_step(x_scene_angular,  scene_angular_vel,    tau_vis),
        delay_cascade_step(x_scene_linear,   scene_linear_vel,     tau_vis),
        delay_cascade_step(x_target_pos,     target_pos,           tau_vis),
        delay_cascade_step(x_target_vel,     target_slip,          tau_vis),
        delay_cascade_step(x_target_disp,    target_disparity,     tau_vis),
        delay_cascade_step(x_scene_vis,      scene_visible,        tau_vis),
        delay_cascade_step(x_target_vis,     target_visible,       tau_vis),
        delay_cascade_step(x_target_motion,  target_motion_visible, tau_vis),
    ])
