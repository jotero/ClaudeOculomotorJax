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
    delay_cascade_step, cascade_lp_step, ypr_to_xyz, xyz_to_ypr,
    N_STAGES, _N_STAGES_OTHER,
    _OFF_SCENE_ANGULAR_VEL, _END_SCENE_ANGULAR_VEL,
    _OFF_SCENE_LINEAR,      _END_SCENE_LINEAR,
    _OFF_TARGET_POS,        _END_TARGET_POS,
    _OFF_TARGET_VEL,        _END_TARGET_VEL,
    _OFF_TARGET_DISP,       _END_TARGET_DISP,
    _OFF_SCENE_VIS,         _END_SCENE_VIS,
    _OFF_TARGET_VIS,        _END_TARGET_VIS,
    _OFF_DEFOCUS,           _END_DEFOCUS,
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

def binocular_fusion_policy(target_pos_L, target_vel_L, target_vis_L,
                             target_pos_R, target_vel_R, target_vis_R,
                             ec_pos, ec_verg, sensory_params):
    """NPC gate + eye-dominance weighting → single pair of blend weights.

    A single (w_L, w_R) pair is used uniformly for target position, velocity,
    and accommodation demand.  In diplopia the dominant eye drives all signals;
    in fusion both eyes contribute proportionally to their visibility.

    Three-state binocular model (horizontal):
        Fused    (demand within motor range): both eyes weighted by visibility
        Diplopic (demand exceeds npc or −div_max): dominant eye only

    Vertical / torsional gates suppress fusion when those disparities are large.

    ec_pos and ec_verg give the full 6-D oculomotor state (version + vergence).
    target_vel_L/R and ec_pos are accepted for future use (interocular velocity
    difference, gaze-angle-dependent fusion limits, Listing's law) but not yet
    used for weight computation.

    Args:
        target_pos_L:   (3,) target direction [yaw,pitch,0] (deg) — left eye
        target_vel_L:   (3,) target retinal velocity [yaw,pitch,0] (deg/s) — left eye
        target_vis_L:   scalar target gate = target_present × target_in_vf ∈ [0,1]
        target_pos_R:   (3,) target direction [yaw,pitch,0] (deg) — right eye
        target_vel_R:   (3,) target retinal velocity [yaw,pitch,0] (deg/s) — right eye
        target_vis_R:   scalar target gate ∈ [0,1]
        ec_pos:         (3,) version eye position [yaw,pitch,roll] (deg) — for future gaze-modulated fusion
        ec_verg:        (3,) vergence [H,V,T] (deg); [0] used for NPC gate
        sensory_params: SensoryParams — reads npc, div_max, vert_max, tors_max, eye_dominant

    Returns:
        w_L:             scalar left-eye blend weight, normalised (w_L + w_R = 1 or both 0)
        w_R:             scalar right-eye blend weight, normalised
        target_disparity:(3,) diplopia-gated vergence disparity (deg)
        target_fusable:  scalar fusion gate ∈ [0, 1]
        target_visible:  scalar cyclopean target visibility ∈ [0, 1]
    """
    _ = target_vel_L, target_vel_R, ec_pos   # reserved for future extensions

    bino_raw = target_vis_L * target_vis_R
    raw_disp = bino_raw * (target_pos_L - target_pos_R)

    total_demand_h = raw_disp[0] + ec_verg[0]
    total_demand_v = raw_disp[1] + ec_verg[1]
    total_demand_t = raw_disp[2] + ec_verg[2]
    gate_conv = jax.nn.sigmoid(100.0 * (sensory_params.npc      - total_demand_h))
    gate_div  = jax.nn.sigmoid(100.0 * (sensory_params.div_max  + total_demand_h ))
    gate_vert = jax.nn.sigmoid(100.0 * (sensory_params.vert_max - jnp.abs(total_demand_v)))
    gate_tors = jax.nn.sigmoid(100.0 * (sensory_params.tors_max - jnp.abs(total_demand_t)))
    # Fusion requires both eyes to see the target — with one eye occluded there
    # is nothing to fuse. Keep two related signals:
    #   target_fusable    — true fusion (both see AND within range); 0 in monocular
    #                       case. Used downstream (perception, disparity gating).
    #   _equal_weight     — 1 if EITHER fused (binocular within limits) OR monocular
    #                       (only one eye sees). Used for the per-eye blend so that
    #                       dominance only kicks in for true diplopia (binocular but
    #                       outside fusion range), not when only one eye sees.
    bino_fusable    = gate_conv * gate_div * gate_vert * gate_tors
    target_fusable  = bino_raw * bino_fusable
    _equal_weight   = jnp.maximum(target_fusable, 1.0 - bino_raw)

    dom_L = 1.0 - sensory_params.eye_dominant
    dom_R = sensory_params.eye_dominant
    w_L = target_vis_L * (_equal_weight + (1.0 - _equal_weight) * dom_L)
    w_R = target_vis_R * (_equal_weight + (1.0 - _equal_weight) * dom_R)

    target_visible = jnp.clip(w_L + w_R, 0.0, 1.0)
    norm = jnp.maximum(w_L + w_R, 1e-6)
    # Disparity drive: gated only by both eyes seeing the target (bino_raw),
    # NOT by the fusion-limit gate. Even when eyes are out of fusion range,
    # the disparity is still the correct error signal driving vergence back
    # into range — zeroing it out here would leave vergence without any drive
    # exactly when it most needs one.  target_fusable is returned separately
    # for downstream consumers (perception, FEF) but no longer multiplies the
    # vergence drive itself.
    return w_L / norm, w_R / norm, raw_disp, target_fusable, target_visible


def binocular_okr_policy(scene_angular_vel_L, scene_linear_vel_L,
                         scene_angular_vel_R, scene_linear_vel_R,
                         scene_vis_L, scene_vis_R):
    """Visibility-weighted optic flow average → per-eye scene blend weights + cyclopean gate.

    OKR is driven by the background scene, not a foveated target.  There is no NPC gate
    and no eye dominance — both eyes contribute equally whenever the scene is present.
    Cyclopean scene visibility follows a probabilistic OR: scene_visible = 1 − (1−L)(1−R).

    Scene velocities are accepted for future use (e.g. interocular velocity difference
    as a cue to scene depth or motion parallax) but not yet processed here.

    Args:
        scene_angular_vel_L/R: (3,) rotational optic flow per eye (deg/s)
        scene_linear_vel_L/R:  (3,) translational optic flow per eye (m/s, eye frame)
        scene_vis_L:           scalar scene presence gate = scene_present ∈ [0,1] — left eye
        scene_vis_R:           scalar scene presence gate ∈ [0,1] — right eye

    Returns:
        w_L:          scalar left-eye scene weight (= scene_vis_L)
        w_R:          scalar right-eye scene weight (= scene_vis_R)
        scene_visible:scalar cyclopean scene gate ∈ [0,1]
    """
    scene_visible = 1.0 - (1.0 - scene_vis_L) * (1.0 - scene_vis_R)
    norm = jnp.maximum(scene_vis_L + scene_vis_R, 1e-6)
    return scene_vis_L / norm, scene_vis_R / norm, scene_visible


def step(x_vis,
         scene_angular_vel_L, scene_linear_vel_L, target_pos_L, target_vel_L, scene_vis_L, target_vis_L, target_motion_vis_L,
         scene_angular_vel_R, scene_linear_vel_R, target_pos_R, target_vel_R, scene_vis_R, target_vis_R, target_motion_vis_R,
         sensory_params,
         ec_vel, ec_pos, ec_verg,
         defocus_L=0.0, defocus_R=0.0):
    """Fuse per-eye retinal signals, apply EC correction, then advance the cyclopean cascade.

    EC subtraction happens pre-delay: ec_vel_eye (the version motor command rotated into eye
    frame) is added to the instantaneous retinal slip before the cascade.  This is
    mathematically equivalent to the old post-delay approach (delay is linear) but saves
    the two 120-state EC delay cascades entirely.

    All visibility inputs are pre-gated — this function only receives _vis signals:
        scene_vis_L/R        = scene_present            (from retina)
        target_vis_L/R       = target_present × in_vf   (from retina)
        target_motion_vis_L/R = target_vis × (1−strobe)  (from sensory_model)

    defocus_L/R = (acc_demand + refractive_error − x_acc_plant) per eye (diopters).
    Positive = near target closer than current focus (need more accommodation).
    Gated by defocus_visible = OR(scene_vis, target_vis) — broader than target_visible alone.

    Args:
        x_vis:                  (800,)  cyclopean cascade state
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
        ec_pos:                  (3,) eye position efference [yaw,pitch,roll] (deg) = x_ni_net
        ec_verg:                 (3,) vergence efference [H,V,T] (deg); [0] used for NPC gate
        defocus_L/R:             scalar per-eye defocus (D) = acc_demand + RE − x_plant

    Returns:
        dx_vis: (800,)  cascade state derivative (defocus delayed signal read via C_defocus)
    """
    # Rotate version velocity efference from head frame to eye frame.
    # ec_vel is in YPR space; angular velocity must be converted to XYZ before
    # applying the rotation matrix (which operates in XYZ), then converted back.
    # Skipping ypr_to_xyz / xyz_to_ypr introduces a spurious roll component that
    # grows with gaze angle (same class of bug as the VVOR frame fix in retina.py).
    R_eye_ec   = rotation_matrix(ypr_to_xyz(ec_pos))
    ec_vel_eye = xyz_to_ypr(R_eye_ec.T @ ypr_to_xyz(ec_vel))

    # ── Scene (OKR) ───────────────────────────────────────────────────────────
    w_s_L, w_s_R, scene_visible = binocular_okr_policy(
        scene_angular_vel_L, scene_linear_vel_L,
        scene_angular_vel_R, scene_linear_vel_R,
        scene_vis_L, scene_vis_R)
    scene_angular_cyc = w_s_L * scene_angular_vel_L + w_s_R * scene_angular_vel_R
    scene_linear_cyc  = w_s_L * scene_linear_vel_L  + w_s_R * scene_linear_vel_R

    # EC correction pre-delay: add motor command to instantaneous slip before cascade.
    # delay(slip + u_motor) = delay(slip) + delay(u_motor) — same as old post-delay formulation.
    scene_angular_vel = velocity_saturation((scene_angular_cyc + ec_vel_eye) * scene_visible, sensory_params.v_max_scene_vel)
    scene_linear_vel  = scene_linear_cyc * scene_visible


    # ── Target: fusion policy → single weights for pos / vel / defocus ───────────
    w_L, w_R, target_disparity, target_fusable, target_visible = binocular_fusion_policy(
        target_pos_L, target_vel_L, target_vis_L,
        target_pos_R, target_vel_R, target_vis_R,
        ec_pos, ec_verg, sensory_params)

    # Disparity is a meaningful drive only when both eyes contribute AND the
    # target is within fusion range. If either condition fails (one-eye view
    # OR demand exceeds fusion limits), the disparity drive cuts out and
    # vergence relaxes toward its tonic setpoint.
    disp_visibility  = target_fusable * (1.0 - jnp.abs(w_L - w_R))
    target_disparity = target_disparity * disp_visibility

    target_pos = w_L * target_pos_L + w_R * target_pos_R

    # Defocus gate: broader than target_visible alone — the lens responds whenever
    # EITHER the scene OR the target is visible in EITHER eye.  This prevents
    # accommodation from relaxing to infinity during brief occlusions of the target
    # (e.g. during blinks) when the background is still visible.
    defocus_visible = 1.0 - (1.0 - scene_visible) * (1.0 - target_visible)
    # Defocus weighting uses RAW per-eye target visibility (not the fusion-gated
    # w_L, w_R). This way a monocularly-visible target still drives accommodation
    # via the seeing eye's defocus, even when binocular fusion is disabled (e.g.
    # demand exceeds NPC) and w_L/w_R could otherwise collapse to zero.
    def_norm = target_vis_L + target_vis_R + 1e-6
    w_def_L  = target_vis_L / def_norm
    w_def_R  = target_vis_R / def_norm
    defocus_cyc = (w_def_L * defocus_L + w_def_R * defocus_R) * defocus_visible

    # ── Target velocity (pursuit) ──────────────────────────────────────────────
    # Strobe gate: target_motion_vis = target_vis × (1−strobe), already zero when strobed.
    # Apply same w_L/w_R weights; strobe zeroes the motion-visible blend.
    target_motion_visible = w_L * target_motion_vis_L + w_R * target_motion_vis_R
    ec_vel_eye_notors = ec_vel_eye.at[2].set(0.0)   # torsion zeroed — retina is 2D
    target_vel_cyc    = w_L * target_vel_L + w_R * target_vel_R
    target_slip = velocity_saturation(
        (target_vel_cyc + ec_vel_eye_notors) * target_motion_visible,
        sensory_params.v_max_target_vel)

    # ── Advance cascade ───────────────────────────────────────────────────────
    # target_pos uses the legacy 40-stage cascade (no LP) — saccade targeting
    # needs a sharp transport delay. All other signals use a short sharp cascade
    # (Pugh-Lamb photo-transduction model) plus a per-channel 1-pole LP for
    # neural-integration smoothing.
    tau_vis     = sensory_params.tau_vis
    tau_sharp   = sensory_params.tau_vis_sharp
    tau_motion  = sensory_params.tau_vis_smooth_motion
    tau_disp    = sensory_params.tau_vis_smooth_disparity
    tau_defocus = sensory_params.tau_vis_smooth_defocus
    N_OTHER     = _N_STAGES_OTHER

    x_scene_angular  = x_vis[_OFF_SCENE_ANGULAR_VEL : _END_SCENE_ANGULAR_VEL]
    x_scene_linear   = x_vis[_OFF_SCENE_LINEAR      : _END_SCENE_LINEAR]
    x_target_pos     = x_vis[_OFF_TARGET_POS        : _END_TARGET_POS]
    x_target_vel     = x_vis[_OFF_TARGET_VEL        : _END_TARGET_VEL]
    x_target_disp    = x_vis[_OFF_TARGET_DISP       : _END_TARGET_DISP]
    x_scene_vis_b    = x_vis[_OFF_SCENE_VIS         : _END_SCENE_VIS]
    x_target_vis_b   = x_vis[_OFF_TARGET_VIS        : _END_TARGET_VIS]
    x_defocus        = x_vis[_OFF_DEFOCUS           : _END_DEFOCUS]

    return jnp.concatenate([
        cascade_lp_step(x_scene_angular,  scene_angular_vel,     tau_sharp, tau_motion,  N_OTHER, 3, 1),
        cascade_lp_step(x_scene_linear,   scene_linear_vel,      tau_sharp, tau_motion,  N_OTHER, 3, 1),
        delay_cascade_step(x_target_pos,  target_pos,            tau_vis,                N=N_STAGES),
        cascade_lp_step(x_target_vel,     target_slip,           tau_sharp, tau_motion,  N_OTHER, 3, 1),
        # Disparity uses a 1-pole LP — V1 stereo matching is the genuinely slow
        # computation; the long exponential tail of a 1-pole captures that.
        cascade_lp_step(x_target_disp,    target_disparity,      tau_sharp, tau_disp,    N_OTHER, 3, 1),
        # Visibility flags consumed by brain: 40-stage sharp cascade (no LP) —
        # preserves brief target-flash pulses without amplitude loss.
        delay_cascade_step(x_scene_vis_b,    scene_visible,         tau_vis, N=N_STAGES),
        delay_cascade_step(x_target_vis_b,   target_visible,        tau_vis, N=N_STAGES),
        cascade_lp_step(x_defocus,        defocus_cyc,           tau_sharp, tau_defocus, N_OTHER, 1, 1),
    ])
