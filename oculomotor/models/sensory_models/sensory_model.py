"""Sensory model — thin connector wiring canal, otolith, and visual delay cascade.

Imports the canal SSM (canal.py), otolith SSM (otolith.py), and visual delay
cascade (retina.py) and aggregates them into a single combined step +
read_outputs interface.

Signal flow:
    w_head    → [Canal array]         → y_canals (6,)   afferent firing rates
    a_head,
    q_head    → [Otolith array]       → f_gia (3,)      GIA estimate → gravity_estimator
    e_slip    → [Visual delay, sig 0] → e_slip_delayed   (for VS / OKR)
    e_pos     → [Visual delay, sig 1] → e_pos_delayed    (for saccade generator)
    e_vel     → [Visual delay, sig 2] → vel_delayed      (for smooth pursuit)

State vector  x_sensory = [x_c (12) | x_oto (6) | x_vis (360)]  — N_STATES = 378

Index constants (relative to x_sensory):
    _IDX_C   — canal states   (12,)
    _IDX_OTO — otolith states  (6,)
    _IDX_VIS — visual delay cascade states (360,)

Outputs of step():
    dx_sensory      (378,)  state derivative
    y_canals        (6,)    canal afferent firing rates
    e_slip_delayed  (3,)    delayed retinal slip   → VS / OKR
    e_pos_delayed   (3,)    delayed position error → saccade generator

Note: e_slip_delayed and e_pos_delayed are pure state readouts (C @ x_vis).
They reflect signals from tau_vis seconds ago — they do NOT depend on the
new e_slip / e_pos inputs passed in this step.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oculomotor.models.sensory_models import canal           as _canal
from oculomotor.models.sensory_models import otolith         as _otolith
from oculomotor.models.sensory_models import retina          as _retina
from oculomotor.models.sensory_models import target_selector as _ts


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

# ── Re-exports for external callers ────────────────────────────────────────────

# Canal
N_CANALS          = _canal.N_CANALS        # 6
ORIENTATIONS      = _canal.ORIENTATIONS    # (6, 3)
PINV_SENS         = _canal.PINV_SENS       # (3, 6)
FLOOR             = _canal.FLOOR           # 80.0
_SOFTNESS         = _canal._SOFTNESS       # 0.5  nonlinearity sharpness
canal_nonlinearity = _canal.nonlinearity   # renamed in canal.py

# Visual delay
N_STAGES          = _retina.N_STAGES       # 40
_N_PER_SIG        = _retina._N_PER_SIG     # 120
C_slip            = _retina.C_slip         # (3, 360)
C_pos             = _retina.C_pos          # (3, 360)
C_vel             = _retina.C_vel          # (3, 360)  target velocity channel
delay_cascade_step = _retina.delay_cascade_step
delay_cascade_read = _retina.delay_cascade_read

# ── State layout ───────────────────────────────────────────────────────────────

_N_CANAL_STATES  = _canal.N_STATES          # 12
_N_OTO_STATES    = _otolith.N_STATES        #  6
_N_VIS_STATES    = _retina.N_STATES         # 360
N_STATES         = _N_CANAL_STATES + _N_OTO_STATES + _N_VIS_STATES  # 378

# Index constants — relative to x_sensory
_IDX_C   = slice(0,
                 _N_CANAL_STATES)                                                # (12,)
_IDX_OTO = slice(_N_CANAL_STATES,
                 _N_CANAL_STATES + _N_OTO_STATES)                               # (6,)
_IDX_VIS = slice(_N_CANAL_STATES + _N_OTO_STATES,
                 _N_CANAL_STATES + _N_OTO_STATES + _N_VIS_STATES)               # (360,)


# ── Bundled sensory output ──────────────────────────────────────────────────────

class SensoryOutput(NamedTuple):
    """Bundled sensory outputs — passed as a unit to brain_model.

    Fields:
        canal:         (N_CANALS,) = (6,)  canal afferent rates (deg/s equiv.)
        slip_delayed:  (3,)  delayed retinal slip  → VS / OKR
        pos_delayed:   (3,)  delayed position error — raw, before visual-field gating
        pos_visible:   (3,)  pos_delayed gated by visual field limit
                             ≈ pos_delayed when target is in-field, → 0 when out-of-field
                             Target may be *present* (physical stimulus on) but not
                             *visible* (outside the ~90° visual field).
        e_cmd:         (3,)  motor error command for the saccade generator
                             pos_visible after orbital gate + anti-windup clip
                             (head-centered, clipped to ±orbital_limit)
        vel_delayed:    (3,)  delayed target velocity on retina → smooth pursuit
                              = v_target − w_head − w_eye  (delayed by tau_vis)
                              Zero when eye perfectly tracks the target.
        f_otolith:      (3,)  LP-filtered GIA from otolith states → gravity estimator
                              = PINV_SENS @ x_oto;  at rest upright: [+9.81, 0, 0]
        scene_present:  scalar  0=dark, 1=full visual scene present — gates OKR/VS EC
        target_present: scalar  0=no target, 1=foveal target present — gates pursuit EC
                                Set to 0 for pure OKN (full-field motion, no foveal target)
                                so that scene motion does not drive the pursuit integrator.
    """
    canal:          jnp.ndarray   # (6,)
    slip_delayed:   jnp.ndarray   # (3,)
    pos_delayed:    jnp.ndarray   # (3,)
    pos_visible:    jnp.ndarray   # (3,)
    e_cmd:          jnp.ndarray   # (3,)
    vel_delayed:    jnp.ndarray   # (3,)
    f_otolith:      jnp.ndarray   # (3,)
    scene_present:  jnp.ndarray   # scalar
    target_present: jnp.ndarray   # scalar


def read_outputs(x_sensory, x_p, scene_present, target_present,
                 sensory_params, plant_params, brain_params):
    """Read all sensory outputs from the current state (pure state readout).

    Combines canal afferents, otolith GIA, delayed visual signals, motor error
    command, and scene/target presence flags into a SensoryOutput bundle.

    Visual-field gate: suppresses pos error when target eccentricity > vf_limit.
    Centering gate: fires e_reset = −alpha_reset·x_p only when gate_vf ≈ 0
    (target outside visual field).  Anti-windup clip parks the eye at the
    orbital limit when target is visible but beyond motor range.

    Args:
        x_sensory:      (378,)  sensory state [x_c (12) | x_oto (6) | x_vis (360)]
        x_p:            (3,)    plant state — eye rotation vector (deg, head-centered)
        scene_present:  scalar  0=dark, 1=full visual scene present (OKR/VS gate)
        target_present: scalar  0=no foveal target, 1=target present (pursuit gate)
        sensory_params: SensoryParams
        plant_params:   PlantParams  (reads orbital_limit, k_orbital)
        brain_params:   BrainParams  (reads alpha_reset)

    Returns:
        SensoryOutput with fields (canal, slip_delayed, pos_delayed, pos_visible,
                                   e_cmd, vel_delayed, f_otolith, scene_present,
                                   target_present)
    """
    x_c   = x_sensory[_IDX_C]
    x_oto = x_sensory[_IDX_OTO]
    x_vis = x_sensory[_IDX_VIS]

    canal_out    = _canal.nonlinearity(x_c, sensory_params.canal_gains)
    f_gia        = _otolith.PINV_SENS @ x_oto   # LP-filtered GIA estimate (3,)
    slip_delayed = _retina.C_slip @ x_vis
    pos_delayed  = _retina.C_pos  @ x_vis
    vel_delayed  = _retina.C_vel  @ x_vis

    # Visual-field gate — suppresses pos error when target is outside ~vf_limit deg
    vf_limit    = sensory_params.visual_field_limit
    k_vf        = sensory_params.k_visual_field
    e_mag       = jnp.linalg.norm(pos_delayed) + 1e-9
    gate_vf     = 1.0 - jax.nn.sigmoid(k_vf * (e_mag - vf_limit))   # ≈1 in-field, ≈0 out
    pos_visible = gate_vf * pos_delayed

    # Motor error command — centering (out-of-field only) + anti-windup clip
    e_cmd = _ts.select(pos_visible, x_p, gate_vf, plant_params, brain_params)

    return SensoryOutput(
        canal          = canal_out,
        slip_delayed   = slip_delayed,
        pos_delayed    = pos_delayed,
        pos_visible    = pos_visible,
        e_cmd          = e_cmd,
        vel_delayed    = vel_delayed,
        f_otolith      = f_gia,
        scene_present  = scene_present,
        target_present = target_present,
    )


# ── Combined step ───────────────────────────────────────────────────────────────

def step(x_sensory, q_head, w_head, a_head, q_eye, w_eye,
         w_scene, v_target, p_target, scene_present, sensory_params):
    """Single ODE step for the sensory subsystem (canal + otolith + visual delay).

    Computes retinal signals internally from world/body state, then advances
    the canal, otolith, and visual delay cascades.

    Args:
        x_sensory:     (378,)  sensory state [x_c (12) | x_oto (6) | x_vis (360)]
        q_head:        (3,)    head angular position (deg)
        w_head:        (3,)    head angular velocity (deg/s)
        a_head:        (3,)    head linear acceleration (m/s²)
        q_eye:         (3,)    eye angular position — plant state (deg)
        w_eye:         (3,)    eye angular velocity — plant derivative (deg/s)
        w_scene:       (3,)    scene angular velocity (deg/s)
        v_target:      (3,)    target angular velocity in world frame (deg/s)
                               Used to compute foveal target velocity on retina.
                               Zero for stationary targets (no pursuit drive).
        p_target:      (3,)    Cartesian target position
        scene_present: scalar  0=dark, 1=lit — gates retinal slip
        sensory_params: SensoryParams  model parameters

    Returns:
        dx_sensory:     (378,)  dx_sensory/dt
        y_canals:       (6,)    canal afferent firing rates
        e_slip_delayed: (3,)    delayed retinal slip   (for VS / OKR)
        e_pos_delayed:  (3,)    delayed position error (for saccade generator)
    """
    x_c   = x_sensory[_IDX_C]
    x_oto = x_sensory[_IDX_OTO]
    x_vis = x_sensory[_IDX_VIS]

    e_pos, raw_slip, e_vel = _retina.retinal_signals(
        p_target, q_head, w_head, q_eye, w_eye, w_scene, v_target, scene_present)

    dx_c,   y_canals                       = _canal.step(x_c,  w_head,                           sensory_params)
    dx_oto, _                              = _otolith.step(x_oto, jnp.concatenate([a_head, q_head]), sensory_params)
    dx_vis, e_slip_delayed, e_pos_delayed  = _retina.step(x_vis, raw_slip, e_pos, e_vel,          sensory_params.tau_vis)

    dx_sensory = jnp.concatenate([dx_c, dx_oto, dx_vis])
    return dx_sensory, y_canals, e_slip_delayed, e_pos_delayed
