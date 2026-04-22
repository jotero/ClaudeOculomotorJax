"""Sensory model — thin connector wiring canal, otolith, and visual delay cascade.

Imports the canal SSM (canal.py), otolith SSM (otolith.py), and visual delay
cascade (retina.py) and aggregates them into a single combined step +
read_outputs interface.

Signal flow:
    w_head    → [Canal array]    → y_canals (6,)   afferent firing rates
    a_head,
    q_head    → [Otolith array]  → f_gia (3,)      GIA estimate → gravity_estimator
    Per-eye retinal geometry → six delay cascades per eye (see retina.py):
        scene_vel      → delayed                         → slip_L/R   (raw, gated in brain)
        target_pos     → delayed                         → pos_L/R    (raw, gated in brain)
        target_vel     → delayed                         → vel_L/R    (raw, gated in brain)
        scene_present  → delayed                         → scene_vis_L/R
        target_present × target_in_vf → delayed          → target_vis_L/R

Binocular averaging is the brain's responsibility (brain_model.py).  This module
only provides raw per-eye cascade readouts plus delayed visibility gates.

Binocular state layout:
    x_sensory = [x_c (12) | x_oto (6) | x_vis_L (440) | x_vis_R (440)]  — N_STATES = 898

    x_vis_{L,R} layout:
        [x_scene_vel(120) | x_target_pos(120) | x_target_vel(120)
         | x_scene_visible(40) | x_target_visible(40)]

Index constants (relative to x_sensory):
    _IDX_C     — canal states    (12,)
    _IDX_OTO   — otolith states   (6,)
    _IDX_VIS_L — left  visual delay cascade states (480,)
    _IDX_VIS_R — right visual delay cascade states (480,)
    _IDX_VIS   — alias for _IDX_VIS_L (backward compatibility)

SensoryOutput fields:
    Shared:
        canal:        (6,)  canal afferent rates
        f_otolith:    (3,)  LP-filtered GIA
    Per-eye raw cascade readouts (un-gated):
        slip_L/R:     (3,)  delayed scene velocity
        pos_L/R:      (3,)  delayed target position
        vel_L/R:      (3,)  delayed target velocity
    Per-eye delayed visibility gates:
        scene_vis_L/R:  scalar  delay(scene_present)
        target_vis_L/R: scalar  delay(target_present × target_in_vf)
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oculomotor.models.sensory_models import canal   as _canal
from oculomotor.models.sensory_models import otolith as _otolith
from oculomotor.models.sensory_models import retina  as _retina


# ── Sensory parameters ──────────────────────────────────────────────────────────

class SensoryParams(NamedTuple):
    """Sensory parameters — canal mechanics + otolith + visual pathway.

    These are determined by peripheral anatomy/physiology.  Fixed during
    typical patient fitting but freed for known peripheral pathology
    (e.g. canal paresis → canal_gains, drug effects → tau_vis).
    """
    # Semicircular canals — Steinhausen torsion-pendulum (Fernandez & Goldberg 1971)
    tau_c:              float       = 5.0    # cupula adaptation TC (s); HP corner ≈ 0.03 Hz
    tau_s:              float       = 0.005  # endolymph inertia TC (s); LP corner ≈ 32 Hz
    canal_gains:        jnp.ndarray = jnp.ones(6)  # (6,) per-canal scale; 1=intact, 0=paresis
    canal_floor:        float       = 80.0   # resting discharge (deg/s); inhibitory saturation point
                                             # (Goldberg & Fernandez 1971 J Neurophysiol 34:635)

    # Otolith — first-order LP adaptation (Fernandez & Goldberg 1976)
    tau_oto:            float       = 100.0  # otolith adaptation TC (s); large → near-DC pass

    # Visual pathway
    tau_vis:            float       = 0.08   # gamma-cascade mean delay (s); Lisberger & Movshon 1999
    visual_field_limit: float       = 90.0   # retinal eccentricity limit (deg); ~90° monocular field
    k_visual_field:     float       = 1.0    # sigmoid steepness for visual field gate (1/deg)

    # Sensory noise (std in output units; 0 = noiseless)
    sigma_canal:        float       = 0.0    # canal afferent noise (deg/s equiv.); ~1–3 deg/s realistic
    sigma_slip:         float       = 0.0    # retinal slip noise (deg/s); drives VS/OKR
    sigma_pos:          float       = 0.0    # retinal position noise (deg);  drives SG → microsaccades
    tau_pos_drift:      float       = 0.3    # OU drift TC (s); sets how slowly pos error wanders
    sigma_vel:          float       = 0.0    # target velocity noise (deg/s); drives pursuit

    # Binocular
    ipd:                float       = 0.064  # inter-pupillary distance (m); ~64 mm adult

# ── Re-exports for external callers ────────────────────────────────────────────

# Canal
N_CANALS          = _canal.N_CANALS        # 6
ORIENTATIONS      = _canal.ORIENTATIONS    # (6, 3)
PINV_SENS         = _canal.PINV_SENS       # (3, 6)
FLOOR             = _canal.FLOOR           # 80.0
_SOFTNESS         = _canal._SOFTNESS       # 0.5  nonlinearity sharpness
canal_nonlinearity = _canal.nonlinearity   # renamed in canal.py

# Visual delay
N_STAGES           = _retina.N_STAGES             # 40
_N_PER_SIG         = _retina._N_PER_SIG           # 120
C_slip             = _retina.C_slip               # (3, 480)  delayed scene velocity
C_pos              = _retina.C_pos                # (3, 480)  delayed target position (raw)
C_vel              = _retina.C_vel                # (3, 480)  delayed target velocity (raw)
C_scene_visible    = _retina.C_scene_visible      # (1, 480)  delayed scene_present
C_target_visible   = _retina.C_target_visible     # (1, 480)  delayed target_present × target_in_vf
C_strobed          = _retina.C_strobed            # (1, 480)  delayed target_strobed
delay_cascade_step = _retina.delay_cascade_step
delay_cascade_read = _retina.delay_cascade_read

# ── State layout ───────────────────────────────────────────────────────────────

_N_CANAL_STATES  = _canal.N_STATES          # 12
_N_OTO_STATES    = _otolith.N_STATES        #  6
_N_VIS_STATES    = _retina.N_STATES         # 480  [x_scene_vel(120)|x_target_pos(120)|x_target_vel(120)|x_scene_visible(40)|x_target_visible(40)|x_strobed(40)]
N_STATES         = _N_CANAL_STATES + _N_OTO_STATES + 2 * _N_VIS_STATES  # 12+6+480+480 = 978

# Index constants — relative to x_sensory
_IDX_C     = slice(0,
                   _N_CANAL_STATES)                                              # (12,)
_IDX_OTO   = slice(_N_CANAL_STATES,
                   _N_CANAL_STATES + _N_OTO_STATES)                             # (6,)
_IDX_VIS_L = slice(_N_CANAL_STATES + _N_OTO_STATES,
                   _N_CANAL_STATES + _N_OTO_STATES + _N_VIS_STATES)             # (480,)
_IDX_VIS_R = slice(_N_CANAL_STATES + _N_OTO_STATES + _N_VIS_STATES,
                   _N_CANAL_STATES + _N_OTO_STATES + 2 * _N_VIS_STATES)         # (480,)
_IDX_VIS   = _IDX_VIS_L   # backward-compatibility alias (left eye cascade)


# ── Bundled sensory output ──────────────────────────────────────────────────────

class SensoryOutput(NamedTuple):
    """Bundled sensory outputs — passed as a unit to brain_model.

    Per-eye raw cascade readouts are un-gated; brain_model.py applies visibility
    gates and computes binocular averages (gate × signal, then weighted mean).

    Shared:
        canal:        (6,)   canal afferent rates
        f_otolith:    (3,)   LP-filtered GIA → gravity estimator
    Per-eye raw (un-gated) delayed signals:
        slip_L/R:     (3,)   delayed scene velocity    → VS / OKR after gating
        pos_L/R:      (3,)   delayed target position   → SG after gating
        vel_L/R:      (3,)   delayed target velocity   → pursuit after gating
    Per-eye delayed visibility gates:
        scene_vis_L/R:    scalar  delay(scene_present)
        target_vis_L/R:   scalar  delay(target_present × target_in_vf)
        strobe_delayed_L/R: scalar  delay(target_strobed) — brain uses to gate EC in pursuit
    """
    canal:           jnp.ndarray   # (6,)
    f_otolith:       jnp.ndarray   # (3,)
    slip_L:          jnp.ndarray   # (3,)
    slip_R:          jnp.ndarray   # (3,)
    pos_L:           jnp.ndarray   # (3,)
    pos_R:           jnp.ndarray   # (3,)
    vel_L:           jnp.ndarray   # (3,)
    vel_R:           jnp.ndarray   # (3,)
    scene_vis_L:     jnp.ndarray   # scalar
    scene_vis_R:     jnp.ndarray   # scalar
    target_vis_L:    jnp.ndarray   # scalar
    target_vis_R:    jnp.ndarray   # scalar
    strobe_delayed_L: jnp.ndarray  # scalar — delayed target_strobed
    strobe_delayed_R: jnp.ndarray  # scalar — delayed target_strobed


def read_outputs(x_sensory, sensory_params):
    """Read all sensory outputs from the current state (pure state readout).

    Returns per-eye raw cascade readouts and delayed visibility gates.
    Binocular averaging is the brain's responsibility.

    Args:
        x_sensory:      (978,)  sensory state
        sensory_params: SensoryParams

    Returns:
        SensoryOutput with per-eye raw signals and delayed visibility gates.
    """
    x_c     = x_sensory[_IDX_C]
    x_oto   = x_sensory[_IDX_OTO]
    x_vis_L = x_sensory[_IDX_VIS_L]
    x_vis_R = x_sensory[_IDX_VIS_R]

    canal_out = _canal.nonlinearity(x_c, sensory_params.canal_gains, sensory_params.canal_floor)
    f_gia     = _otolith.PINV_SENS @ x_oto

    return SensoryOutput(
        canal        = canal_out,
        f_otolith    = f_gia,
        slip_L       = _retina.C_slip @ x_vis_L,
        slip_R       = _retina.C_slip @ x_vis_R,
        pos_L        = _retina.C_pos  @ x_vis_L,
        pos_R        = _retina.C_pos  @ x_vis_R,
        vel_L        = _retina.C_vel  @ x_vis_L,
        vel_R        = _retina.C_vel  @ x_vis_R,
        scene_vis_L      = (_retina.C_scene_visible  @ x_vis_L)[0],
        scene_vis_R      = (_retina.C_scene_visible  @ x_vis_R)[0],
        target_vis_L     = (_retina.C_target_visible @ x_vis_L)[0],
        target_vis_R     = (_retina.C_target_visible @ x_vis_R)[0],
        strobe_delayed_L = (_retina.C_strobed        @ x_vis_L)[0],
        strobe_delayed_R = (_retina.C_strobed        @ x_vis_R)[0],
    )


# ── Combined step ───────────────────────────────────────────────────────────────

def step(x_sensory, q_head, w_head, a_head, q_eye_L, w_eye_L, q_eye_R, w_eye_R,
         w_scene, v_target, p_target, scene_present_L, scene_present_R,
         target_present_L, target_present_R, target_strobed,
         sensory_params):
    """Single ODE step for the sensory subsystem (canal + otolith + visual delay).

    Computes retinal signals for each eye using IPD geometry, then advances
    the canal, otolith, and two visual delay cascades (one per eye).

    Args:
        x_sensory:        (978,)  sensory state [x_c(12)|x_oto(6)|x_vis_L(480)|x_vis_R(480)]
        q_head:           (3,)    head angular position (deg)
        w_head:           (3,)    head angular velocity (deg/s)
        a_head:           (3,)    head linear acceleration (m/s²)
        q_eye_L:          (3,)    left  eye angular position — plant state (deg)
        w_eye_L:          (3,)    left  eye angular velocity — plant derivative (deg/s)
        q_eye_R:          (3,)    right eye angular position — plant state (deg)
        w_eye_R:          (3,)    right eye angular velocity — plant derivative (deg/s)
        w_scene:          (3,)    scene angular velocity (deg/s)
        v_target:         (3,)    target angular velocity in world frame (deg/s)
        p_target:         (3,)    Cartesian target position (world frame)
        scene_present_L:  scalar  0=L eye dark, 1=L eye lit
        scene_present_R:  scalar  0=R eye dark, 1=R eye lit
        target_present_L: scalar  0=L eye covered, 1=L eye sees target — gates e_vel/e_pos/target_in_vf L
        target_present_R: scalar  0=R eye covered, 1=R eye sees target — gates e_vel/e_pos/target_in_vf R
        target_strobed:   scalar  1=stroboscopic illumination — zeros target_vel before retinal cascade;
                                  position (saccades) preserved, velocity (pursuit) absent
        sensory_params:   SensoryParams  model parameters

    Returns:
        dx_sensory: (818,)  dx_sensory/dt

    Brain-facing outputs are read via read_outputs() from the updated state.
    """
    x_c     = x_sensory[_IDX_C]
    x_oto   = x_sensory[_IDX_OTO]
    x_vis_L = x_sensory[_IDX_VIS_L]
    x_vis_R = x_sensory[_IDX_VIS_R]

    # Eye offsets in head frame — all geometry handled inside retinal_signals.
    ipd_half     = sensory_params.ipd * 0.5
    eye_off_L    = jnp.array([-ipd_half, 0.0, 0.0])   # left  eye in head frame
    eye_off_R    = jnp.array([ ipd_half, 0.0, 0.0])   # right eye in head frame

    # Retinal signals per eye (all raw — no pre-gating; gating happens at readout)
    target_pos_L, scene_vel_L, target_vel_L, target_in_vf_L = _retina.retinal_signals(
        p_target, eye_off_L, q_head, w_head, q_eye_L, w_eye_L, w_scene, v_target,
        sensory_params.visual_field_limit, sensory_params.k_visual_field)

    target_pos_R, scene_vel_R, target_vel_R, target_in_vf_R = _retina.retinal_signals(
        p_target, eye_off_R, q_head, w_head, q_eye_R, w_eye_R, w_scene, v_target,
        sensory_params.visual_field_limit, sensory_params.k_visual_field)

    # Visibility of scene and target. Both dependent of presence, target depends on field of view too.
    target_visible_L = target_present_L * target_in_vf_L
    target_visible_R = target_present_R * target_in_vf_R
    scene_visible_L = scene_present_L;
    scene_visible_R = scene_present_R;

    dx_c,   _ = _canal.step(x_c,   w_head, sensory_params)
    dx_oto, _ = _otolith.step(x_oto, jnp.concatenate([a_head, q_head]), sensory_params)

    dx_vis_L = _retina.step(
        x_vis_L, scene_vel_L, target_pos_L, target_vel_L,
        scene_visible_L, target_visible_L, target_strobed, sensory_params.tau_vis)
    dx_vis_R = _retina.step(
        x_vis_R, scene_vel_R, target_pos_R, target_vel_R,
        scene_visible_R, target_visible_R, target_strobed, sensory_params.tau_vis)

    dx_sensory = jnp.concatenate([dx_c, dx_oto, dx_vis_L, dx_vis_R])
    return dx_sensory
