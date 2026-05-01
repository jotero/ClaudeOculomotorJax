"""Sensory model — thin connector wiring canal, otolith, and visual delay cascade.

Imports the canal SSM (canal.py), otolith SSM (otolith.py), and visual delay
cascade (retina.py / cyclopean_vision.py) and aggregates them into a single
combined step + read_outputs interface.

Signal flow:
    w_head    → [Canal array]    → y_canals (6,)   afferent firing rates
    a_head,
    q_head    → [Otolith array]  → f_gia (3,)      GIA estimate → gravity_estimator
    Per-eye retinal geometry → cyclopean pre-delay fusion → single delay cascade (720 states):
        scene_angular_vel  → delayed → scene_slip         (3,)  → VS / OKR
        scene_linear_vel   → delayed → scene_linear_vel   (3,)  → looming
        target_pos         → delayed → target_pos         (3,)  → SG after gating
        target_vel         → delayed → target_slip        (3,)  → pursuit after gating
        target_disparity   → delayed → target_disparity   (3,)  → vergence
        scene_visible      → delayed → scene_visible      scalar
        target_visible     → delayed → target_visible     scalar
        target_motion_vis  → delayed → target_motion_visible  scalar

Binocular fusion happens BEFORE the delay cascade (cyclopean_vision.pre_delay_fusion).
The brain receives fully cyclopean signals — no per-eye readouts needed.

Single-cascade state layout:
    x_sensory = [x_c (12) | x_oto (6) | x_vis (720)]  — N_STATES = 738

    x_vis layout:
        [scene_angular_vel(120) | scene_linear_vel(120) | target_pos(120)
         | target_vel(120) | target_disparity(120)
         | scene_visible(40) | target_visible(40) | strobed(40)]

Index constants (relative to x_sensory):
    _IDX_C     — canal states      (12,)
    _IDX_OTO   — otolith states     (6,)
    _IDX_VIS   — cyclopean visual delay cascade states (720,)
    _IDX_VIS_L — alias for _IDX_VIS (backward compatibility)

SensoryOutput fields:
    Shared:
        canal:            (6,)   canal afferent rates
        otolith:          (3,)   instantaneous GIA in head frame (m/s²)
    Cyclopean delayed signals:
        scene_slip:       (3,)   delayed scene angular velocity [yaw,pitch,roll] deg/s → VS/OKR
        scene_linear_vel: (3,)   delayed scene linear velocity [x,y,z] m/s → looming
        target_pos:       (3,)   delayed target position → SG
        target_slip:      (3,)   delayed target velocity → pursuit
        target_disparity: (3,)   delayed vergence disparity (diplopia-gated) → vergence
        scene_visible:    scalar delayed cyclopean scene presence gate
        target_visible:   scalar delayed cyclopean target presence gate
        target_motion_visible: scalar delayed pursuit gate = delay(target_visible × (1−strobe))
        acc_demand:       float  accommodation demand (1/m)
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oculomotor.models.sensory_models import canal            as _canal
from oculomotor.models.sensory_models import otolith          as _otolith
from oculomotor.models.sensory_models import retina           as _retina
from oculomotor.models.sensory_models import cyclopean_vision as _cv
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

    # Otolith — first-order LP adaptation (Fernandez & Goldberg 1976)
    tau_oto:            float       = 100.0  # otolith adaptation TC (s); large → near-DC pass

    # Visual pathway
    tau_vis:            float       = 0.08   # gamma-cascade mean delay (s); Lisberger & Movshon 1999
    visual_field_limit: float       = 90.0   # retinal eccentricity limit (deg); ~90° monocular field
    k_visual_field:     float       = 1.0    # sigmoid steepness for visual field gate (1/deg)

    # Sensory noise (std in output units; 0 = noiseless)
    sigma_canal:        float       = 0.5    # canal afferent noise (deg/s equiv.); ~1–3 deg/s realistic
    sigma_slip:         float       = 0.0    # retinal slip noise (deg/s); drives VS/OKR
    sigma_pos:          float       = 0.2    # retinal position noise (deg);  drives SG → microsaccades
    tau_pos_drift:      float       = 0.3    # OU drift TC (s); sets how slowly pos error wanders
    sigma_vel:          float       = 0.2    # target velocity noise (deg/s); drives pursuit

    # Binocular geometry
    ipd:                float       = 0.064  # inter-pupillary distance (m); ~64 mm adult

    # Binocular fusion and motor limits (used by pre_delay_fusion for diplopia suppression)
    # ── Motor limits (absolute vergence angle — eye position space) ─────────────────────────
    #    Diplopia gate closes when total vergence demand exceeds these.
    #    Horizontal is asymmetric (large convergence range, small divergence range).
    #    Vertical and torsional are symmetric.
    npc:                float       = 50.0   # near point of convergence (deg); convergence motor limit
                                             # 50° ≈ NPC 7 cm (IPD=64 mm); physiological range ~40–55° for young adults
    div_max:            float       = 6.0    # maximum divergence (deg); ~6° for young adults
    vert_max:           float       = 5.0    # maximum vertical vergence ±(deg); ~3–5° clinical range
    tors_max:           float       = 8.0    # maximum cyclovergence ±(deg); ~5–8° max
    eye_dominant:       float       = 1.0    # 1.0 = right dominant, 0.0 = left dominant

    # Pre-delay velocity saturation (applied before visual cascade to suppress spikes)
    # Mirrors the speed tuning of MT/MST (target) and NOT/AOS (scene) neurons.
    # Must match the v_max_pursuit / v_max_okr values in BrainParams so that the
    # EC correction (which is clipped to the same ceiling) exactly cancels what
    # made it through the visual cascade.
    v_max_target_vel:   float       = 40.0   # MT/MST speed ceiling (deg/s); clips target_vel before cascade
    v_max_scene_vel:    float       = 80.0   # NOT/AOS speed ceiling (deg/s); clips scene_vel before cascade

# ── Re-exports for external callers ────────────────────────────────────────────

# Canal
N_CANALS          = _canal.N_CANALS        # 6
ORIENTATIONS      = _canal.ORIENTATIONS    # (6, 3)
PINV_SENS         = _canal.PINV_SENS       # (3, 6)
FLOOR             = _canal.FLOOR           # 80.0
_SOFTNESS         = _canal._SOFTNESS       # 0.5  nonlinearity sharpness
canal_nonlinearity = _canal.nonlinearity   # renamed in canal.py

# Visual delay — readout matrices for the single 800-state cyclopean cascade
N_STAGES           = _retina.N_STAGES             # 40
_N_PER_SIG         = _retina._N_PER_SIG           # 120
C_slip             = _retina.C_slip               # (3, 800)  delayed scene angular velocity
C_scene_linear_vel = _retina.C_scene_linear_vel   # (3, 800)  delayed scene linear velocity (m/s, eye xyz)
C_pos              = _retina.C_pos                # (3, 800)  delayed target position (raw)
C_vel              = _retina.C_vel                # (3, 800)  delayed target velocity (raw)
C_target_disp      = _retina.C_target_disp        # (3, 800)  delayed target disparity (deg)
C_scene_visible    = _retina.C_scene_visible      # (1, 800)  delayed scene_present
C_target_visible   = _retina.C_target_visible     # (1, 800)  delayed target_present × target_in_vf
C_target_motion_visible = _retina.C_target_motion_visible  # (1, 800)  delayed pursuit gate
C_defocus          = _retina.C_defocus            # (1, 800)  delayed defocus (D)
delay_cascade_step = _retina.delay_cascade_step
delay_cascade_read = _retina.delay_cascade_read

# ── State layout ───────────────────────────────────────────────────────────────

_N_CANAL_STATES  = _canal.N_STATES          # 12
_N_OTO_STATES    = _otolith.N_STATES        #  6
_N_VIS_STATES    = _retina.N_STATES         # 800  single cyclopean cascade (incl. defocus)
N_STATES         = _N_CANAL_STATES + _N_OTO_STATES + _N_VIS_STATES  # 12+6+800 = 818

# Index constants — relative to x_sensory
_IDX_C     = slice(0,
                   _N_CANAL_STATES)                                              # (12,)
_IDX_OTO   = slice(_N_CANAL_STATES,
                   _N_CANAL_STATES + _N_OTO_STATES)                             # (6,)
_IDX_VIS   = slice(_N_CANAL_STATES + _N_OTO_STATES,
                   _N_CANAL_STATES + _N_OTO_STATES + _N_VIS_STATES)             # (720,)
_IDX_VIS_L = _IDX_VIS   # backward-compat alias (some scripts still use this name)
# _IDX_VIS_R intentionally removed — single cyclopean cascade


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
    canal:           jnp.ndarray   # (6,)
    otolith:         jnp.ndarray   # (3,)
    scene_slip:      jnp.ndarray   # (3,)
    scene_linear_vel: jnp.ndarray  # (3,)  scene linear velocity, eye-frame xyz (m/s)
    target_pos:      jnp.ndarray   # (3,)
    target_slip:     jnp.ndarray   # (3,)
    target_disparity: jnp.ndarray  # (3,)
    scene_visible:   jnp.ndarray   # scalar
    target_visible:  jnp.ndarray   # scalar
    target_motion_visible: jnp.ndarray  # scalar
    defocus:         float = 0.0   # delayed cyclopean defocus (diopters)


def read_outputs(x_sensory, sensory_params, q_head, a_head):
    """Read all sensory outputs from the current state (pure state readout).

    Returns cyclopean delayed cascade readouts — all binocularly fused.
    acc_demand_L/R are 0.0 here; the ODE layer passes them to sensory_model.step()
    which computes the fusion-weighted acc_demand_cyc via cyclopean_vision.step().

    Args:
        x_sensory:      (738,)  sensory state
        sensory_params: SensoryParams
        q_head:         (3,)    head rotation vector [yaw,pitch,roll] (deg) — for GIA
        a_head:         (3,)    head linear acceleration (m/s², world frame) — for GIA

    Returns:
        SensoryOutput with cyclopean delayed signals.
    """
    x_c   = x_sensory[_IDX_C]
    x_vis = x_sensory[_IDX_VIS]

    canal_out = _canal.nonlinearity(x_c, sensory_params.canal_gains, sensory_params.canal_floor)

    # Instantaneous GIA in head frame — same formula as otolith.step().
    # x_oto (LP state, tau=100s) is NOT used here: it hasn't adapted after
    # a short-duration tilt, so using it would pull the gravity estimator
    # back toward upright (exactly the drift bug we're fixing).
    # x_oto is retained in the state purely for somatogravic illusion modelling.
    q_xyz = jnp.array([-q_head[1], q_head[0], q_head[2]])
    R     = _rotation_matrix(q_xyz)
    f_gia = R.T @ _otolith.G_WORLD + R.T @ a_head   # GIA in head frame (m/s²)

    return SensoryOutput(
        canal            = canal_out,
        otolith          = f_gia,
        scene_slip       = _retina.C_slip             @ x_vis,
        scene_linear_vel = _retina.C_scene_linear_vel @ x_vis,
        target_pos       = _retina.C_pos              @ x_vis,
        target_slip      = _retina.C_vel              @ x_vis,
        target_disparity = _retina.C_target_disp      @ x_vis,
        scene_visible    = (_retina.C_scene_visible    @ x_vis)[0],
        target_visible   = (_retina.C_target_visible   @ x_vis)[0],
        target_motion_visible = (_retina.C_target_motion_visible @ x_vis)[0],
        defocus          = (_retina.C_defocus          @ x_vis)[0],
    )


# ── Combined step ───────────────────────────────────────────────────────────────

def step(x_sensory,
         # ── Head kinematics ───────────────────────────────────────────────────
         q_head, w_head, x_head, a_head,
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
         # ── Efference copy (from brain) ───────────────────────────────────────
         ec_vel, ec_pos, ec_verg,
         # ── Parameters ───────────────────────────────────────────────────────
         sensory_params):
    """Single ODE step for the sensory subsystem (canal + otolith + visual delay).

    Projects each eye's retina via world_to_retina, fuses per-eye signals into a
    cyclopean representation via binocular_fusion_policy, then advances the canal,
    otolith, and single cyclopean visual delay cascade.

    All per-eye scene/target inputs are pre-transformed by the ODE layer before
    this call — optical interventions (prisms, stereo displays) are invisible here.

    World frame is LEFT-HANDED: x=right, y=up, z=forward  (x × y = −z).

    Returns:
        dx_sensory: (818,)  dx_sensory/dt
    """
    x_c   = x_sensory[_IDX_C]
    x_oto = x_sensory[_IDX_OTO]
    x_vis = x_sensory[_IDX_VIS]

    ipd_half  = sensory_params.ipd * 0.5
    eye_off_L = jnp.array([-ipd_half, 0.0, 0.0])
    eye_off_R = jnp.array([ ipd_half, 0.0, 0.0])

    dx_c,   _ = _canal.step(x_c,   w_head, sensory_params)
    dx_oto, _ = _otolith.step(x_oto, jnp.concatenate([a_head, q_head]), sensory_params)

    # World → retina projection (one call per eye; geometry + visibility live in retina.py).
    # All per-eye inputs (scene, target, eye pose) are pre-transformed by the ODE layer —
    # prisms, stereo displays, and covers are invisible to this function.
    target_pos_L, scene_angular_vel_L, scene_linear_vel_L, target_vel_L, scene_vis_L, target_vis_L = \
        _retina.world_to_retina(
            p_target_L, eye_off_L, q_head, w_head, x_head,
            q_eye_L, w_eye_L, w_scene_L, v_scene_L, dp_dt_L,
            scene_present_L, target_present_L,
            sensory_params.visual_field_limit, sensory_params.k_visual_field)

    target_pos_R, scene_angular_vel_R, scene_linear_vel_R, target_vel_R, scene_vis_R, target_vis_R = \
        _retina.world_to_retina(
            p_target_R, eye_off_R, q_head, w_head, x_head,
            q_eye_R, w_eye_R, w_scene_R, v_scene_R, dp_dt_R,
            scene_present_R, target_present_R,
            sensory_params.visual_field_limit, sensory_params.k_visual_field)

    # Strobe gate: applied here so cyclopean fusion only receives _visible signals
    target_motion_vis_L = target_vis_L * (1.0 - target_strobed)
    target_motion_vis_R = target_vis_R * (1.0 - target_strobed)

    # Pre-delay cyclopean fusion + cascade advance (single call)
    dx_vis = _cv.step(
        x_vis,
        scene_angular_vel_L, scene_linear_vel_L, target_pos_L, target_vel_L, scene_vis_L, target_vis_L, target_motion_vis_L,
        scene_angular_vel_R, scene_linear_vel_R, target_pos_R, target_vel_R, scene_vis_R, target_vis_R, target_motion_vis_R,
        sensory_params, ec_vel, ec_pos, ec_verg,
        defocus_L, defocus_R)

    return jnp.concatenate([dx_c, dx_oto, dx_vis])
