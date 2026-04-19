"""Sensory model — thin connector wiring canal, otolith, and visual delay cascade.

Imports the canal SSM (canal.py), otolith SSM (otolith.py), and visual delay
cascade (retina.py) and aggregates them into a single combined step +
read_outputs interface.

Signal flow:
    w_head    → [Canal array]    → y_canals (6,)   afferent firing rates
    a_head,
    q_head    → [Otolith array]  → f_gia (3,)      GIA estimate → gravity_estimator
    Per-eye retinal geometry → six delay cascades per eye (see retina.py):
        scene_vel      → delayed → × scene_visible  → slip_delayed   (VS / OKR)
        target_pos     → delayed → × target_visible → pos_delayed    (SG error)
        target_vel     → delayed → × target_visible → vel_delayed    (pursuit)
        target_in_vf        → delayed                    → target_in_vf        (SG target select)
        scene_present  → delayed                    → scene_visible
        target_present × target_in_vf → delayed          → target_visible

Binocular state layout:
    x_sensory = [x_c (12) | x_oto (6) | x_vis_L (480) | x_vis_R (480)]  — N_STATES = 978

    x_vis_{L,R} layout:
        [x_scene_vel(120) | x_target_pos(120) | x_target_vel(120)
         | x_target_in_vf(40) | x_scene_visible(40) | x_target_visible(40)]

Index constants (relative to x_sensory):
    _IDX_C     — canal states    (12,)
    _IDX_OTO   — otolith states   (6,)
    _IDX_VIS_L — left  visual delay cascade states (480,)
    _IDX_VIS_R — right visual delay cascade states (480,)
    _IDX_VIS   — alias for _IDX_VIS_L (backward compatibility)

SensoryOutput fields:
    Brain-facing (averaged across eyes) — brain_model.py reads these unchanged:
        canal, slip_delayed, pos_delayed, target_in_vf, vel_delayed, f_otolith,
        scene_visible, target_visible
    Per-eye diagnostics:
        slip_delayed_L/R, pos_delayed_L/R, target_in_vf_L/R, vel_delayed_L/R
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
C_target_in_vf             = _retina.C_target_in_vf               # (1, 480)  delayed target_in_vf (geometric)
C_scene_visible    = _retina.C_scene_visible      # (1, 480)  delayed scene_present
C_target_visible   = _retina.C_target_visible     # (1, 480)  delayed target_present × target_in_vf
delay_cascade_step = _retina.delay_cascade_step
delay_cascade_read = _retina.delay_cascade_read

# ── State layout ───────────────────────────────────────────────────────────────

_N_CANAL_STATES  = _canal.N_STATES          # 12
_N_OTO_STATES    = _otolith.N_STATES        #  6
_N_VIS_STATES    = _retina.N_STATES         # 480  [x_scene_vel(120)|x_target_pos(120)|x_target_vel(120)|x_target_in_vf(40)|x_scene_visible(40)|x_target_visible(40)]
N_STATES         = _N_CANAL_STATES + _N_OTO_STATES + 2 * _N_VIS_STATES  # 12+6+480+480 = 978

# Index constants — relative to x_sensory
_IDX_C     = slice(0,
                   _N_CANAL_STATES)                                              # (12,)
_IDX_OTO   = slice(_N_CANAL_STATES,
                   _N_CANAL_STATES + _N_OTO_STATES)                             # (6,)
_IDX_VIS_L = slice(_N_CANAL_STATES + _N_OTO_STATES,
                   _N_CANAL_STATES + _N_OTO_STATES + _N_VIS_STATES)             # (400,)
_IDX_VIS_R = slice(_N_CANAL_STATES + _N_OTO_STATES + _N_VIS_STATES,
                   _N_CANAL_STATES + _N_OTO_STATES + 2 * _N_VIS_STATES)         # (400,)
_IDX_VIS   = _IDX_VIS_L   # backward-compatibility alias (left eye cascade)


# ── Bundled sensory output ──────────────────────────────────────────────────────

class SensoryOutput(NamedTuple):
    """Bundled sensory outputs — passed as a unit to brain_model.

    Brain-facing fields (averaged across L and R eyes) — brain_model.py reads
    these; it is unchanged by the binocular extension:
        canal:          (6,)    canal afferent rates
        slip_delayed:   (3,)    delayed retinal slip → VS / OKR  (L+R average)
        pos_delayed:    (3,)    delayed gated position error → SG  (L+R average)
        target_in_vf:        scalar  delayed visual-field gate  (L+R average)
        vel_delayed:    (3,)    delayed target velocity → pursuit  (L+R average)
        f_otolith:      (3,)    LP-filtered GIA → gravity estimator
        scene_visible:  scalar  avg delay(scene_present) across eyes
        target_visible: scalar  avg delay(target_present × target_in_vf) across eyes

    Per-eye fields (diagnostics / binocular analysis):
        slip_delayed_L/R:   (3,)   per-eye delayed retinal slip
        pos_delayed_L/R:    (3,)   per-eye delayed gated position error
        target_in_vf_L/R:   scalar per-eye delayed geometric visual-field gate
        vel_delayed_L/R:    (3,)   per-eye delayed target velocity
        scene_visible_L/R:  scalar per-eye delay(scene_present)
    """
    # Brain-facing (averaged)
    canal:          jnp.ndarray   # (6,)
    slip_delayed:   jnp.ndarray   # (3,)  L+R average
    pos_delayed:    jnp.ndarray   # (3,)  L+R average
    target_in_vf:   jnp.ndarray   # scalar  L+R average
    vel_delayed:    jnp.ndarray   # (3,)  L+R average
    f_otolith:      jnp.ndarray   # (3,)
    scene_visible:  jnp.ndarray   # scalar  avg delay(scene_present)
    target_visible: jnp.ndarray   # scalar  avg delay(target_present × target_in_vf)

    # Per-eye diagnostics
    slip_delayed_L:   jnp.ndarray   # (3,)
    slip_delayed_R:   jnp.ndarray   # (3,)
    pos_delayed_L:    jnp.ndarray   # (3,)
    pos_delayed_R:    jnp.ndarray   # (3,)
    target_in_vf_L:   jnp.ndarray   # scalar
    target_in_vf_R:   jnp.ndarray   # scalar
    vel_delayed_L:    jnp.ndarray   # (3,)
    vel_delayed_R:    jnp.ndarray   # (3,)
    scene_visible_L:  jnp.ndarray   # scalar  delay(scene_present_L)
    scene_visible_R:  jnp.ndarray   # scalar  delay(scene_present_R)


def read_outputs(x_sensory, sensory_params):
    """Read all sensory outputs from the current state (pure state readout).

    Reads canal afferents, otolith GIA, and delayed visual signals from both eye
    cascades.  Scene and target visibility are read from their own delay cascades
    (scene_visible = delay(scene_present), target_visible = delay(target_present×target_in_vf))
    and applied as gates at readout.

    Args:
        x_sensory:      (978,)  sensory state
        sensory_params: SensoryParams

    Returns:
        SensoryOutput with brain-facing averaged fields and per-eye diagnostic fields.
    """
    x_c     = x_sensory[_IDX_C]
    x_oto   = x_sensory[_IDX_OTO]
    x_vis_L = x_sensory[_IDX_VIS_L]
    x_vis_R = x_sensory[_IDX_VIS_R]

    canal_out = _canal.nonlinearity(x_c, sensory_params.canal_gains)
    f_gia     = _otolith.PINV_SENS @ x_oto   # LP-filtered GIA estimate (3,)

    # ── Per-eye readouts — gate at readout using delayed presence signals ─────────
    # Raw delayed signals (zero means signal was zero, not that it was unobserved)
    scene_vis_L   = (_retina.C_scene_visible  @ x_vis_L)[0]  # delay(scene_present)
    scene_vis_R   = (_retina.C_scene_visible  @ x_vis_R)[0]
    target_vis_L  = (_retina.C_target_visible @ x_vis_L)[0]  # delay(target_present × target_in_vf)
    target_vis_R  = (_retina.C_target_visible @ x_vis_R)[0]
    gate_L        = (_retina.C_target_in_vf           @ x_vis_L)[0]  # delay(target_in_vf) — for SG
    gate_R        = (_retina.C_target_in_vf           @ x_vis_R)[0]

    # Apply delayed presence gates: signal × observed_flag
    scene_vis     = 0.5 * (scene_vis_L + scene_vis_R)   # same input both eyes
    slip_L = scene_vis   * (_retina.C_slip @ x_vis_L)
    slip_R = scene_vis   * (_retina.C_slip @ x_vis_R)
    pos_L  = target_vis_L * (_retina.C_pos @ x_vis_L)
    pos_R  = target_vis_R * (_retina.C_pos @ x_vis_R)
    vel_L  = target_vis_L * (_retina.C_vel @ x_vis_L)
    vel_R  = target_vis_R * (_retina.C_vel @ x_vis_R)

    # Brain-facing averages — target_visible-weighted so a non-visible eye contributes nothing
    tv_sum   = target_vis_L + target_vis_R
    norm     = jnp.maximum(tv_sum, 1e-6)
    slip_avg = 0.5 * (slip_L + slip_R)
    pos_avg  = (target_vis_L * pos_L + target_vis_R * pos_R) / norm
    vel_avg  = (target_vis_L * vel_L + target_vis_R * vel_R) / norm
    gate_avg = jnp.clip(gate_L + gate_R, 0.0, 1.0)   # 1 if either eye has target in field

    return SensoryOutput(
        canal          = canal_out,
        slip_delayed   = slip_avg,
        pos_delayed    = pos_avg,
        target_in_vf        = gate_avg,
        vel_delayed    = vel_avg,
        f_otolith      = f_gia,
        scene_visible  = scene_vis,    # delay(scene_present)
        target_visible = tv_sum * 0.5, # avg delay(target_present × target_in_vf)
        slip_delayed_L = slip_L,
        slip_delayed_R = slip_R,
        pos_delayed_L  = pos_L,
        pos_delayed_R  = pos_R,
        target_in_vf_L      = gate_L,
        target_in_vf_R      = gate_R,
        vel_delayed_L   = vel_L,
        vel_delayed_R   = vel_R,
        scene_visible_L = scene_vis_L,
        scene_visible_R = scene_vis_R,
    )


# ── Combined step ───────────────────────────────────────────────────────────────

def step(x_sensory, q_head, w_head, a_head, q_eye_L, w_eye_L, q_eye_R, w_eye_R,
         w_scene, v_target, p_target, scene_present_L, scene_present_R,
         target_present_L, target_present_R,
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

    # Presence × visibility combined into target_visible per eye
    target_visible_L = target_present_L * target_in_vf_L
    target_visible_R = target_present_R * target_in_vf_R

    dx_c,   _ = _canal.step(x_c,   w_head, sensory_params)
    dx_oto, _ = _otolith.step(x_oto, jnp.concatenate([a_head, q_head]), sensory_params)

    dx_vis_L = _retina.step(
        x_vis_L, scene_vel_L, target_pos_L, target_vel_L, target_in_vf_L,
        scene_present_L, target_visible_L, sensory_params.tau_vis)
    dx_vis_R = _retina.step(
        x_vis_R, scene_vel_R, target_pos_R, target_vel_R, target_in_vf_R,
        scene_present_R, target_visible_R, sensory_params.tau_vis)

    dx_sensory = jnp.concatenate([dx_c, dx_oto, dx_vis_L, dx_vis_R])
    return dx_sensory
