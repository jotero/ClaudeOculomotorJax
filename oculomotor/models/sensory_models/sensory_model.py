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

State vector  x_sensory = [x_c (12) | x_oto (6) | x_vis (400)]  — N_STATES = 418

    x_vis layout: [x_slip (120) | x_pos_vis (120) | x_vel (120) | x_gate (40)]
      x_pos_vis delays the GATED position error gate_vf · e_pos (retinal geometry)
      x_gate    delays the visual-field gate scalar gate_vf

Index constants (relative to x_sensory):
    _IDX_C   — canal states   (12,)
    _IDX_OTO — otolith states  (6,)
    _IDX_VIS — visual delay cascade states (400,)

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

# ── Re-exports for external callers ────────────────────────────────────────────

# Canal
N_CANALS          = _canal.N_CANALS        # 6
ORIENTATIONS      = _canal.ORIENTATIONS    # (6, 3)
PINV_SENS         = _canal.PINV_SENS       # (3, 6)
FLOOR             = _canal.FLOOR           # 80.0
_SOFTNESS         = _canal._SOFTNESS       # 0.5  nonlinearity sharpness
canal_nonlinearity = _canal.nonlinearity   # renamed in canal.py

# Visual delay
N_STAGES           = _retina.N_STAGES            # 40
_N_PER_SIG         = _retina._N_PER_SIG          # 120
C_slip             = _retina.C_slip              # (3, 400)
C_pos              = _retina.C_pos              # (3, 400)  gated pos (gate_vf · e_pos)
C_vel              = _retina.C_vel              # (3, 400)  target velocity channel
C_gate             = _retina.C_gate             # (1, 400)  delayed visual-field gate
delay_cascade_step = _retina.delay_cascade_step
delay_cascade_read = _retina.delay_cascade_read

# ── State layout ───────────────────────────────────────────────────────────────

_N_CANAL_STATES  = _canal.N_STATES          # 12
_N_OTO_STATES    = _otolith.N_STATES        #  6
_N_VIS_STATES    = _retina.N_STATES         # 400  [x_slip(120)|x_pos_vis(120)|x_vel(120)|x_gate(40)]
N_STATES         = _N_CANAL_STATES + _N_OTO_STATES + _N_VIS_STATES  # 418

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
        slip_delayed:  (3,)  delayed retinal slip → VS / OKR
        pos_delayed:   (3,)  delayed retinal position error (raw, deg)
                             = target_direction − head_position − eye_position, delayed by τ_vis
        gate_vf:       scalar  visual-field gate: ≈1 when target is in-field, ≈0 when out-of-field
                               computed from |pos_delayed| vs. visual_field_limit
        vel_delayed:   (3,)  delayed target velocity on retina → smooth pursuit
                             Zero when eye perfectly tracks the target.
        f_otolith:     (3,)  LP-filtered GIA from otolith states → gravity estimator
                             = PINV_SENS @ x_oto;  at rest upright: [+9.81, 0, 0]
        scene_present: scalar  0=dark, 1=full visual scene present — gates OKR/VS EC
        target_present: scalar  0=no target, 1=foveal target present — gates pursuit EC
                                Set to 0 for pure OKN (full-field motion, no foveal target)
                                so that scene motion does not drive the pursuit integrator.

    Target selection (visual-field clip + centering saccade) is handled inside the
    saccade generator using pos_delayed, gate_vf, and the neural integrator state x_ni
    as a proxy for current eye position.
    """
    canal:          jnp.ndarray   # (6,)
    slip_delayed:   jnp.ndarray   # (3,)
    pos_delayed:    jnp.ndarray   # (3,)
    gate_vf:        jnp.ndarray   # scalar
    vel_delayed:    jnp.ndarray   # (3,)
    f_otolith:      jnp.ndarray   # (3,)
    scene_present:  jnp.ndarray   # scalar
    target_present: jnp.ndarray   # scalar


def read_outputs(x_sensory, scene_present, target_present, sensory_params):
    """Read all sensory outputs from the current state (pure state readout).

    Reads canal afferents, otolith GIA, delayed visual signals, and the visual-field
    gate from the sensory state.  Target selection (clip + centering) is handled
    downstream in the saccade generator using the brain's x_ni estimate.

    Args:
        x_sensory:      (378,)  sensory state [x_c (12) | x_oto (6) | x_vis (360)]
        scene_present:  scalar  0=dark, 1=full visual scene present (OKR/VS gate)
        target_present: scalar  0=no foveal target, 1=target present (pursuit gate)
        sensory_params: SensoryParams  (reads canal_gains, vf_limit, k_visual_field)

    Returns:
        SensoryOutput with fields (canal, slip_delayed, pos_delayed, gate_vf,
                                   vel_delayed, f_otolith, scene_present, target_present)
    """
    x_c   = x_sensory[_IDX_C]
    x_oto = x_sensory[_IDX_OTO]
    x_vis = x_sensory[_IDX_VIS]

    canal_out    = _canal.nonlinearity(x_c, sensory_params.canal_gains)
    f_gia        = _otolith.PINV_SENS @ x_oto   # LP-filtered GIA estimate (3,)
    slip_delayed = _retina.C_slip @ x_vis        # delayed retinal slip
    pos_delayed  = _retina.C_pos  @ x_vis        # delayed gated pos error (gate_vf · e_pos)
    vel_delayed  = _retina.C_vel  @ x_vis        # delayed target velocity
    gate_vf      = (_retina.C_gate @ x_vis)[0]  # delayed visual-field gate (scalar)

    return SensoryOutput(
        canal          = canal_out,
        slip_delayed   = slip_delayed,
        pos_delayed    = pos_delayed,
        gate_vf        = gate_vf,
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

    e_pos, e_pos_vis, raw_slip, e_vel, gate_vf = _retina.retinal_signals(
        p_target, q_head, w_head, q_eye, w_eye, w_scene, v_target, scene_present,
        sensory_params.visual_field_limit, sensory_params.k_visual_field)

    dx_c,   y_canals                       = _canal.step(x_c,  w_head,                               sensory_params)
    dx_oto, _                              = _otolith.step(x_oto, jnp.concatenate([a_head, q_head]),  sensory_params)
    dx_vis, e_slip_delayed, e_pos_delayed  = _retina.step(x_vis, raw_slip, e_pos_vis, e_vel, gate_vf, sensory_params.tau_vis)

    dx_sensory = jnp.concatenate([dx_c, dx_oto, dx_vis])
    return dx_sensory, y_canals, e_slip_delayed, e_pos_delayed
