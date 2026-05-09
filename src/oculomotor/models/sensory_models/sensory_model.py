"""Sensory model — thin connector wiring canal, otolith, and per-eye retina.

Imports the canal SSM (canal.py), otolith SSM (otolith.py), and per-eye retina
(retina.py) and aggregates them into a single combined step + read_outputs.
Binocular fusion + brain-LP smoothing live in the brain
(``brain_models.perception_cyclopean``) — they operate on already-delayed
per-eye signals and are cortical computations, not peripheral.

Signal flow:
    w_head            → [Canal array]    → y_canals (6,)   afferent firing rates
    a_head, q_head    → [Otolith array]  → f_gia (3,)      GIA → gravity estimator
    per-eye stimulus  → [retina.step] L  → RetinaOut_L (delayed per-eye signals)
                      → [retina.step] R  → RetinaOut_R
    SensoryOutput bundles canal + otolith + retina_L + retina_R for the brain.

State layout (198 states):
    x_sensory = [x_canal (12) | x_oto (6) | x_retina_L (90) | x_retina_R (90)]

Index constants (relative to x_sensory):
    _IDX_C         — canal states                   (12,)
    _IDX_OTO       — otolith states                  (6,)
    _IDX_RETINA_L  — left-eye sharp cascade states  (90,)
    _IDX_RETINA_R  — right-eye sharp cascade states (90,)
    _IDX_VIS       — full per-eye visual block     (180,)
    _IDX_VIS_L     — alias for _IDX_RETINA_L (backward compat)
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oculomotor.models.sensory_models import canal            as _canal
from oculomotor.models.sensory_models import otolith          as _otolith
from oculomotor.models.sensory_models import retina           as _retina
from oculomotor.models.sensory_models.retina import RetinaOut  # noqa: F401  (re-export)
from oculomotor.models.plant_models.readout import rotation_matrix as _rotation_matrix


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
    canal_v_max:        float       = 400.0  # excitatory canal afferent saturation (deg/s).
                                             # Sensor-side ceiling — applied in canal.step output.
                                             # Inhibitory saturation is the FLOOR (canal_floor); this is
                                             # the symmetric upper clip on afferent firing rate.
                                             # Goldberg & Fernandez 1971: ~300–600 deg/s; 400 conservative.
                                             # At typical stimulus velocities (<200 deg/s) the clip is inert.

    # Otolith — first-order LP adaptation (Fernandez & Goldberg 1976)
    tau_oto:            float       = 100.0  # otolith adaptation TC (s); large → near-DC pass

    # Visual pathway — sensor-side parameters only. The brain-side LP smoothing
    # TCs (tau_vis_smooth_*) and the binocular-fusion-policy parameters (npc /
    # div_max / vert_max / tors_max / eye_dominant) live in BrainParams since
    # they're cortical decisions; perception_cyclopean (in brain_models) reads
    # them from there.
    tau_vis_sharp:      float = 0.05   # sharp cascade mean delay (s) — photo-transduction +
                                       # axonal/synaptic transport (Pugh & Lamb 1993,
                                       # Dunn & Rieke 2006). Used by retina.step.
    visual_field_limit: float = 90.0   # retinal eccentricity limit (deg); ~90° monocular field
    k_visual_field:     float = 1.0    # sigmoid steepness for visual field gate (1/deg)

    # Sensory noise (std in output units; 0 = noiseless). All four sources are
    # Ornstein-Uhlenbeck processes — short τ approaches white noise (band-limited),
    # longer τ produces drift-like fluctuations.
    sigma_canal:        float       = 1.0    # canal afferent noise (deg/s); filtered heavily by VS/NI/plant
    tau_canal_drift:    float       = 0.005  # OU TC for canal noise (s); essentially band-limited white
    sigma_slip:         float       = 0.0    # retinal slip noise (deg/s); drives VS/OKR (off by default)
    tau_slip_drift:     float       = 0.005  # OU TC for slip noise (s)
    sigma_pos:          float       = 0.2    # retinal position drift (deg); triggers microsaccades
    tau_pos_drift:      float       = 0.2    # OU TC for retinal-pos drift (s); inter-microsaccade interval
    sigma_vel:          float       = 1.0    # target velocity noise (deg/s); drives pursuit jitter
    tau_vel_drift:      float       = 0.005  # OU TC for retinal-vel noise (s); essentially band-limited white

    # Binocular geometry
    ipd:                float       = 0.064  # inter-pupillary distance (m); ~64 mm adult

    # Sensor-side velocity saturation (applied per-eye in retina.step before sharp cascade).
    # Mirrors the speed tuning of MT/MST (target) and NOT/AOS (scene) neurons. Must match
    # the v_max_pursuit / v_max_okr values in BrainParams so that the EC correction (clipped
    # to the same ceiling) exactly cancels what made it through the retina cascade.
    v_max_target_vel:   float       = 40.0   # MT/MST speed ceiling (deg/s)
    v_max_scene_vel:    float       = 80.0   # NOT/AOS speed ceiling (deg/s)

# ── Re-exports for external callers ────────────────────────────────────────────

# Canal
N_CANALS          = _canal.N_CANALS        # 6
ORIENTATIONS      = _canal.ORIENTATIONS    # (6, 3)
PINV_SENS         = _canal.PINV_SENS       # (3, 6)
FLOOR             = _canal.FLOOR           # 80.0
_SOFTNESS         = _canal._SOFTNESS       # 0.5  nonlinearity sharpness
canal_nonlinearity = _canal.nonlinearity   # renamed in canal.py

# Visual delay — readout helpers live in perception_cyclopean (in brain_models).
# External code reading delayed cyclopean signals should import them directly:
#     from oculomotor.models.brain_models.perception_cyclopean import C_slip, ...
#     from oculomotor.models.brain_models.brain_model           import _IDX_CYC_BRAIN
#     scene_slip_d = states.brain[:, _IDX_CYC_BRAIN] @ C_slip.T
N_STAGES           = _retina.N_STAGES             # 40 (legacy constant)
_N_PER_SIG         = _retina._N_PER_SIG           # 120 (legacy)
delay_cascade_step = _retina.delay_cascade_step

# ── State layout ───────────────────────────────────────────────────────────────
# Per-eye retina sharp cascades only (90 each). The cyclopean brain LP block
# now lives in brain state (perception_cyclopean is in brain_models).

_N_CANAL_STATES  = _canal.N_STATES                # 12
_N_OTO_STATES    = _otolith.N_STATES              #  6
_N_RETINA_PER_EYE= _retina.N_STATES_PER_EYE       # 90
_N_VIS_STATES    = 2 * _N_RETINA_PER_EYE          # 90+90 = 180
N_STATES         = _N_CANAL_STATES + _N_OTO_STATES + _N_VIS_STATES  # 12+6+180 = 198


# ── State NamedTuple ──────────────────────────────────────────────────────────

class State(NamedTuple):
    """Top-level sensory state — canal + otolith + per-eye retina."""
    canal:    _canal.State    # (12,)  bandpass two-stage SSM
    otolith:  _otolith.State  #  (6,)  bilateral LP adaptation
    retina_L: _retina.State   # (90,)  per-eye sharp cascade — left eye
    retina_R: _retina.State   # (90,)  per-eye sharp cascade — right eye


def rest_state():
    """Initial sensory state (otolith starts settled to gravity, others zero)."""
    return State(
        canal    = _canal.rest_state(),
        otolith  = _otolith.rest_state(),
        retina_L = _retina.rest_state(),
        retina_R = _retina.rest_state(),
    )


# Legacy flat-array adapters retained for any external callers that still hold
# (T,) trajectory arrays predating the SimState NT migration.
def to_array(state):
    """sensory_model.State → (198,) flat array."""
    return jnp.concatenate([
        _canal.to_array(state.canal),
        _otolith.to_array(state.otolith),
        _retina.to_array(state.retina_L),
        _retina.to_array(state.retina_R),
    ])


# ── Bundled sensory output ──────────────────────────────────────────────────────

class SensoryOutput(NamedTuple):
    """Bundled sensory outputs — passed as a unit to brain_model.

    All visual signals are cyclopean (binocularly fused before delay).

    Shared:
        canal:            (6,)   canal afferent rates
        otolith:          (3,)   instantaneous GIA in head frame (m/s²) → gravity estimator
    Cyclopean delayed signals:
        scene_slip:       (3,)   delayed scene angular velocity [yaw,pitch,roll] deg/s → VS / OKR
        scene_linear_vel: (3,)   delayed scene linear velocity [x,y,z] m/s, eye frame → looming
        target_pos:       (3,)   delayed target position   → SG after gating
        target_slip:      (3,)   delayed target velocity   → pursuit after gating
        target_disparity: (3,)   delayed vergence disparity (diplopia-gated) → vergence
        scene_visible:    scalar delayed cyclopean scene presence gate
        target_visible:   scalar delayed cyclopean target presence gate
        target_motion_visible: scalar delayed pursuit gate = delay(target_visible × (1−strobe))
        defocus:          float  delayed cyclopean defocus (D) = delay(acc_demand + RE − x_plant)
                                 Positive = near target closer than current accommodation.
                                 Gated by defocus_visible = OR(scene_vis, target_vis).
    """
    canal:    jnp.ndarray            # (6,)  canal afferent rates
    otolith:  jnp.ndarray            # (3,)  instantaneous GIA (m/s², head frame)
    retina_L: RetinaOut              # delayed per-eye signals — left eye
    retina_R: RetinaOut              # delayed per-eye signals — right eye


def read_outputs(state, sensory_params, q_head, a_head):
    """Read all sensory outputs from the current state (pure state readout).

    Args:
        state:          sensory_model.State
        sensory_params: SensoryParams
        q_head:         (3,)    head rotation vector [yaw,pitch,roll] (deg) — for GIA
        a_head:         (3,)    head linear acceleration (m/s², world frame) — for GIA

    Returns:
        SensoryOutput with delayed per-eye signals.
    """
    # Concatenate (x1, x2) to apply canal nonlinearity (reads x2 internally).
    canal_flat = jnp.concatenate([state.canal.x1, state.canal.x2])
    canal_out  = _canal.nonlinearity(canal_flat,
                                      sensory_params.canal_gains,
                                      sensory_params.canal_floor)
    canal_out = jnp.clip(canal_out, -sensory_params.canal_v_max, sensory_params.canal_v_max)

    # Instantaneous GIA in head frame — same formula as otolith.step().
    q_xyz = jnp.array([-q_head[1], q_head[0], q_head[2]])
    R     = _rotation_matrix(q_xyz)
    f_gia = R.T @ _otolith.G_WORLD + R.T @ a_head   # GIA in head frame (m/s²)

    return SensoryOutput(
        canal    = canal_out,
        otolith  = f_gia,
        retina_L = _retina.read_outputs(state.retina_L),
        retina_R = _retina.read_outputs(state.retina_R),
    )


# ── Combined step ───────────────────────────────────────────────────────────────

def step(state,
         # ── Head kinematics ───────────────────────────────────────────────────
         q_head, w_head, x_head, v_head, a_head,
         # ── Eye kinematics (prism-shifted by ODE before this call) ────────────
         q_eye_L, w_eye_L, q_eye_R, w_eye_R,
         # ── Scene stimulus (per eye) ──────────────────────────────────────────
         q_scene_L, w_scene_L, x_scene_L, v_scene_L,
         q_scene_R, w_scene_R, x_scene_R, v_scene_R,
         # ── Target stimulus (per eye) ─────────────────────────────────────────
         p_target_L, dp_dt_L,
         p_target_R, dp_dt_R,
         # ── Defocus (per eye; = acc_demand + refractive_error − x_acc_plant) ──
         defocus_L, defocus_R,
         # ── Visibility flags ──────────────────────────────────────────────────
         scene_present_L, scene_present_R,
         target_present_L, target_present_R, target_strobed,
         # ── Efference copy (from brain) — unused here, kept for API stability ─
         ec_vel, ec_pos, ec_verg,
         # ── Parameters ───────────────────────────────────────────────────────
         sensory_params):
    """Single ODE step for the sensory subsystem (canal + otolith + per-eye retina).

    Args:
        state: sensory_model.State (canal + otolith + retina_L + retina_R)

    Returns:
        dstate: sensory_model.State  state derivative
    """
    ipd_half  = sensory_params.ipd * 0.5
    eye_off_L = jnp.array([-ipd_half, 0.0, 0.0])
    eye_off_R = jnp.array([ ipd_half, 0.0, 0.0])

    dcanal,   _ = _canal.step(state.canal,     w_head, sensory_params)
    dotolith, _ = _otolith.step(state.otolith, jnp.concatenate([a_head, q_head]), sensory_params)

    # Per-eye retina cascades (cyclopean fusion happens in brain).
    # ec_vel / ec_pos / ec_verg are no longer used here.
    _ = ec_vel, ec_pos, ec_verg
    dretina_L, _ = _retina.step(
        state.retina_L, eye_off_L, q_head, w_head, x_head, v_head,
        q_eye_L, w_eye_L, w_scene_L, v_scene_L, p_target_L, dp_dt_L,
        defocus_L, scene_present_L, target_present_L, target_strobed,
        sensory_params)
    dretina_R, _ = _retina.step(
        state.retina_R, eye_off_R, q_head, w_head, x_head, v_head,
        q_eye_R, w_eye_R, w_scene_R, v_scene_R, p_target_R, dp_dt_R,
        defocus_R, scene_present_R, target_present_R, target_strobed,
        sensory_params)

    return State(canal=dcanal, otolith=dotolith,
                  retina_L=dretina_L, retina_R=dretina_R)
