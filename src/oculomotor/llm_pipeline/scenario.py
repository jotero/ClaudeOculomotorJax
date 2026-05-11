"""SimulationScenario schema — structured representation of a natural-language scenario.

Schema overview
---------------
    SimulationScenario
    ├── description:  str                     free-text label
    ├── head:         list[BodySegment]        head 6-DOF kinematics in world frame
    ├── target:       list[BodySegment]        target 3-D world position + velocity
    ├── scene:        list[BodySegment]        scene / visual-background motion
    ├── visual:       list[VisualFlagsSegment] scene_present / target_present flags
    ├── patient:      Patient                  parameter overrides vs. healthy defaults
    └── plot:         PlotConfig               which panels to show

Stimulus channels
-----------------
Each channel is a list of segments concatenated in time.
Total simulation duration = max(sum of durations across all four channels).

BodySegment
    Full 6-DOF kinematics: rotation (yaw/pitch/roll, deg) + translation (x/y/z, m).
    Within each segment position is a polynomial:
        pos(t) = pos₀ + vel₀·t + ½·acc·t²        (except sinusoid / impulse profiles)
        vel(t) = vel₀ + acc·t

    Initial conditions: None = continue from previous segment's final state;
                        value = jump to that value at the segment boundary.

Coordinate conventions
-----------------------
    Rotation  — yaw:   + rightward (horizontal VOR axis)
                pitch: + upward
                roll:  + CCW from behind
    Translation — x: + rightward (m)
                  y: + upward    (m)
                  z: + forward / depth (m)
    Gravity is along −y in the world frame (y_acc ≈ −9.81 m/s² felt by otoliths).

Per-body semantics
-------------------
    head  : rot_* → semicircular canals (angular velocity).
             lin_* → otoliths (linear acceleration; future).
    target: Specified as 3-D world position in metres.
             lin_x_0/y_0/z_0 — Cartesian position (m). Omit = carry forward.
             lin_x_vel/y_vel/z_vel — velocity (m/s).
             lin_z_0 = depth (default 1 m). To place at angle θ:
             lin_x_0 = tan(θ_deg × π/180) × lin_z_0.
             Runner projects 3-D target–head vector to retinal angles.
    scene : rot_yaw_vel → OKR / velocity-storage drive (angular velocity).
             lin_* → future linear-vection stimulation.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ── 6-DOF body segment ─────────────────────────────────────────────────────────

class BodySegment(BaseModel):
    """One contiguous segment of 6-DOF kinematics for a rigid body.

    Rotation (deg / deg·s⁻¹ / deg·s⁻²):  yaw, pitch, roll
    Translation (m / m·s⁻¹ / m·s⁻²):     x (right), y (up), z (forward/depth)

    Profile
    -------
    'constant'  polynomial:      pos(t) = pos₀ + vel₀·t + ½·acc·t²
    'sinusoid'  oscillation:     rot_vel(t) = amplitude · sin(2π·freq·t)
                                 rot_pos(t) = pos₀ + amplitude/(2π·f) · (1 − cos(2π·f·t))
                                 amplitude = rot_*_vel field; starts at zero velocity.
    'impulse'   vHIT trapezoid:  rises to rot_*_vel in ramp_dur_s, then falls, then coasts.

    The profile applies to ALL rotational DOF that have a non-zero velocity.
    Translational DOF always use the polynomial profile.

    Initial conditions
    ------------------
    Fields ending in `_0` (pos) or as `_vel` (velocity): None = carry from previous
    segment's final state; explicit value = jump/reset at segment start.
    """

    duration_s: float = Field(gt=0, le=120, description="Duration of this segment (s).")

    rot_profile: Literal['constant', 'sinusoid', 'impulse'] = Field(
        default='constant',
        description=(
            "'constant' — polynomial (vel₀·t + ½acc·t²). "
            "'sinusoid' — vel(t)=A·sin(2πft), amplitude = rot_*_vel. "
            "'impulse'  — brief trapezoidal pulse; peak = rot_*_vel, rise/fall = ramp_dur_s."
        )
    )

    # ── Rotational DOF ─────────────────────────────────────────────────────────
    rot_yaw_0:   Optional[float] = Field(default=None, description="Initial yaw angle (deg). None = continue from prev.")
    rot_pitch_0: Optional[float] = Field(default=None, description="Initial pitch angle (deg). None = continue.")
    rot_roll_0:  Optional[float] = Field(default=None, description="Initial roll angle (deg). None = continue.")

    rot_yaw_vel:   Optional[float] = Field(default=None,
        description="Yaw angular velocity (deg/s). For sinusoid/impulse = amplitude. None = continue from prev.")
    rot_pitch_vel: Optional[float] = Field(default=None, description="Pitch angular velocity (deg/s). None = continue.")
    rot_roll_vel:  Optional[float] = Field(default=None, description="Roll angular velocity (deg/s). None = continue.")

    rot_yaw_acc:   float = Field(default=0.0, description="Yaw angular acceleration (deg/s²). Used with profile='constant'.")
    rot_pitch_acc: float = Field(default=0.0, description="Pitch angular acceleration (deg/s²).")
    rot_roll_acc:  float = Field(default=0.0, description="Roll angular acceleration (deg/s²).")

    # ── Translational DOF ──────────────────────────────────────────────────────
    lin_x_0: Optional[float] = Field(default=None, description="Initial x position (m, rightward +). None = continue.")
    lin_y_0: Optional[float] = Field(default=None, description="Initial y position (m, upward +). None = continue.")
    lin_z_0: Optional[float] = Field(default=None,
        description="Initial z position / depth (m, forward +). For target: viewing distance (default 1 m). None = continue.")

    lin_x_vel: Optional[float] = Field(default=None, description="x velocity (m/s). None = continue.")
    lin_y_vel: Optional[float] = Field(default=None, description="y velocity (m/s). None = continue.")
    lin_z_vel: Optional[float] = Field(default=None, description="z velocity (m/s). None = continue.")

    lin_x_acc: float = Field(default=0.0, description="x acceleration (m/s²).")
    lin_y_acc: float = Field(default=0.0, description="y acceleration (m/s²). Use -9.81 for gravity along -y.")
    lin_z_acc: float = Field(default=0.0, description="z acceleration (m/s²).")

    # ── Profile parameters ─────────────────────────────────────────────────────
    frequency_hz: float = Field(default=1.0, gt=0, le=20,
        description="Sinusoid frequency (Hz). Only used with rot_profile='sinusoid'. Typical 0.1–2 Hz.")
    ramp_dur_s: float = Field(default=0.02, gt=0,
        description="Rise and fall time for impulse profile (s). 0.02 s = realistic vHIT.")

    @field_validator('rot_yaw_vel', 'rot_pitch_vel', 'rot_roll_vel', mode='before')
    @classmethod
    def _check_rot_vel(cls, v):
        if v is not None and abs(v) > 1200:
            raise ValueError(f'|rot_vel|={abs(v)} > 1200 deg/s is physiologically unrealistic')
        return v

    @model_validator(mode='after')
    def _check_impulse_geometry(self):
        if self.rot_profile == 'impulse' and 2 * self.ramp_dur_s >= self.duration_s:
            raise ValueError(
                f'impulse: 2×ramp_dur_s ({2*self.ramp_dur_s:.3f} s) must be < '
                f'duration_s ({self.duration_s} s). Increase duration_s.')
        return self


# ── Visual flags ───────────────────────────────────────────────────────────────

class VisualFlagsSegment(BaseModel):
    """Scene / target visibility flags for one time segment.

    scene_present  = is the room lit?  True → OKR and visual stabilisation active.
    target_present = is there a discrete foveal target?  True → pursuit and saccades active.
                     Shorthand: sets BOTH eyes simultaneously.
    target_present_L / target_present_R = per-eye override (cover test).
                     None (default) = inherit from target_present.
                     False = that eye is covered (eye patch) → vergence leaks toward tonic_verg.

    Scene MOTION is specified via scene BodySegment rot_yaw_vel — NOT here.

    Common combinations
    -------------------
    VOR in the dark:       scene_present=False, target_present=False
    VOR fixating in dark:  scene_present=False, target_present=True  (HIT)
    VVOR / saccades:       scene_present=True,  target_present=True  (default)
    OKN drum, no dot:      scene_present=True,  target_present=False
    Smooth pursuit:        scene_present=True,  target_present=True
    Cover test (R covered):scene_present=True,  target_present=True,
                           target_present_L=True, target_present_R=False
    Stroboscopic pursuit:  scene_present=True,  target_present=True, target_strobed=True
                           (position visible → saccades; velocity absent → no pursuit drive)
    """
    duration_s:       float          = Field(gt=0, le=120, description="Duration of this segment (s).")
    scene_present:    bool           = Field(default=True,  description="Both-eye shorthand: True = lit room → OKR active for both eyes.")
    scene_present_L:  Optional[bool] = Field(default=None,  description="L-eye override. None = inherit scene_present. False = L eye in darkness.")
    scene_present_R:  Optional[bool] = Field(default=None,  description="R-eye override. None = inherit scene_present. False = R eye in darkness.")
    target_present:   bool           = Field(default=True,  description="Both-eye shorthand: True = target visible for both eyes.")
    target_present_L: Optional[bool] = Field(default=None,  description="L-eye override. None = inherit target_present. False = cover L eye.")
    target_present_R: Optional[bool] = Field(default=None,  description="R-eye override. None = inherit target_present. False = cover R eye.")
    target_strobed:   bool           = Field(default=False, description="Stroboscopic illumination: True = position signal present but velocity signal absent. Blocks pursuit drive while preserving saccadic targeting.")


# ── Patient (unchanged) ────────────────────────────────────────────────────────

# ── Patient — auto-generated from docs/parameters_schema.yaml ──────────────────
# The Patient Pydantic class (and its fields, defaults, descriptions, range
# validators) is derived at import time from the YAML schema.  Adding/removing
# the `disorders:` key on a YAML entry adds/removes the corresponding LLM-
# tunable parameter.  See patient_builder.py for the build logic.
from oculomotor.llm_pipeline.patient_builder import Patient

# Augment the auto-generated docstring with clinical guidance the LLM should
# always have in mind (this gets surfaced via Pydantic's class-level metadata).
Patient.__doc__ = """Model parameter overrides relative to healthy defaults.

Only specify parameters that differ from the healthy default.  Leave all
others at their defaults.  Auto-generated from docs/parameters_schema.yaml.

Gaze-evoked nystagmus (GEN) — IMPORTANT
----------------------------------------
GEN requires a leaky neural integrator so gaze positions drift centripetally.
Two ways to make the NI leaky:
  - K_cereb_fl = 0  → the floccular Cannon-Robinson leak-cancellation is lost;
                      NI leaks at its intrinsic tau_i (~25 s).  For pronounced
                      GEN, ALSO shorten tau_i (3–8 s).
  - tau_i = 3–8 s   → directly shortens the NI TC (brainstem NPH lesion).

But in a lit room with a target, smooth pursuit compensates the drift → no
nystagmus visible.  To see GEN in the light you must ALSO impair pursuit:
    K_cereb_pu = 0    (or K_pursuit = 0.1–0.3)
With healthy pursuit (K_pursuit ≥ 1, K_cereb_pu = 1) the drift is masked even
with a very leaky NI.

In the dark (scene_present=False, target_present=False) pursuit and OKR are
absent, so GEN is always visible with a leaky NI regardless of pursuit gain.

Quick reference — typical cerebellar patient for GEN:
    K_cereb_fl=0, tau_i=4, K_cereb_pu=0  (or K_pursuit=0.2), tau_vs=5, K_vs=0.05
"""


# ── PlotConfig (unchanged) ─────────────────────────────────────────────────────

class PlotConfig(BaseModel):
    """Which signal panels to include in the figure."""

    panels: list[Literal[
        'eye_position', 'eye_velocity', 'head_velocity',
        'gaze_error', 'retinal_error', 'canal_afferents',
        'velocity_storage', 'neural_integrator',
        'saccade_burst', 'pursuit_drive', 'refractory',
        'vergence',
        # Cerebellar diagnostic panels
        'cerebellum_pursuit', 'cerebellum_vor',
        # Stimulus panels
        'target_position', 'target_velocity', 'scene_velocity', 'visual_flags',
    ]] = Field(
        description=(
            "Ordered list of panels. Minimal sets:\n"
            "  VOR / HIT:    ['head_velocity', 'eye_velocity', 'eye_position']\n"
            "  OKN / OKAN:   ['scene_velocity', 'visual_flags', 'eye_velocity', 'eye_position', 'velocity_storage']\n"
            "  Saccades:     ['target_position', 'eye_position', 'eye_velocity', 'saccade_burst', 'refractory']\n"
            "  Pursuit:      ['eye_position', 'eye_velocity', 'pursuit_drive', 'cerebellum_pursuit']\n"
            "  Vergence / cover test: ['eye_position', 'vergence', 'neural_integrator']\n"
            "  Cerebellar lesions (flocculus/paraflocculus/nodulus): add 'cerebellum_pursuit' and/or 'cerebellum_vor'\n"
            "  Full cascade: all panels\n"
            "Panel meanings:\n"
            "  cerebellum_pursuit — VPF pursuit EC correction (vpf_drive), pursuit saccadic-suppression gate, pred_err\n"
            "  cerebellum_vor     — flocculus NI leak-cancellation (fl_drive), OKR EC correction (fl_okr_drive),"
            " nodulus-uvula gravity dumping (nu_drive), scene saccadic-suppression gate"
        )
    )
    title: str = Field(default='', description="Figure title (auto-set from description if empty).")


# ── Top-level schema ───────────────────────────────────────────────────────────

class SimulationScenario(BaseModel):
    """Complete oculomotor simulation scenario.

    All four channels (head, target, scene, visual) are lists of segments
    concatenated in time.  Total duration = max of the four channel sums.

    Target shorthand
    ----------------
    For convenience the LLM can use angular notation on target segments:
        rot_yaw_0   = target angle (deg)  → lin_x_0 = tan(angle) × z_depth
        rot_yaw_vel = angular velocity (deg/s) → lin_x_vel ≈ ang_vel × π/180 × z_depth
    z_depth defaults to lin_z_0 of the segment, or 1.0 m if unset.
    The runner ALWAYS uses the 3-D Cartesian path for retinal projection.

    ## Segment recipes

    Saccade 20° right (2 s):
        head:   [{duration_s: 2}]
        target: [{duration_s: 0.3, lin_z_0: 1.0, lin_x_0: 0.0},
                 {duration_s: 1.7, lin_x_0: 0.364}]         ← tan(20°) × 1 m
        scene:  [{duration_s: 2}]
        visual: [{duration_s: 2}]                            ← defaults: both present

    Rightward vHIT (2.5 s):
        head:   [{duration_s: 2.5, rot_yaw_vel: 200, rot_profile: "impulse"}]
        target: [{duration_s: 2.5, lin_z_0: 1.0}]           ← stationary straight ahead
        scene:  [{duration_s: 2.5}]
        visual: [{duration_s: 2.5, scene_present: false, target_present: true}]

    OKN 30 deg/s (20 s) + OKAN (40 s):
        head:   [{duration_s: 60}]
        target: [{duration_s: 60, lin_z_0: 1.0}]
        scene:  [{duration_s: 20, rot_yaw_vel: 30},
                 {duration_s: 40}]                           ← scene stops, OKAN persists
        visual: [{duration_s: 60, scene_present: true, target_present: false}]

    Pursuit 20 deg/s at 1 m, onset 0.3 s:
        head:   [{duration_s: 5}]
        target: [{duration_s: 0.3, lin_z_0: 1.0},
                 {duration_s: 4.7, rot_yaw_vel: 20}]         ← shorthand: lin_x_vel≈0.349 m/s
        scene:  [{duration_s: 5}]
        visual: [{duration_s: 5}]

    VOR in the dark (5 s rotation + 15 s coast):
        head:   [{duration_s: 5, rot_yaw_vel: 60},
                 {duration_s: 15, rot_yaw_vel: 0}]
        target: [{duration_s: 20, lin_z_0: 1.0}]
        scene:  [{duration_s: 20}]
        visual: [{duration_s: 20, scene_present: false, target_present: false}]

    tVOR (head translates laterally, target stationary 1 m ahead):
        head:   [{duration_s: 3, lin_x_vel: 0.1}]            ← head moves right at 0.1 m/s
        target: [{duration_s: 3, lin_x_0: 0.0, lin_z_0: 1.0}] ← target fixed in world
        scene:  [{duration_s: 3}]
        visual: [{duration_s: 3, scene_present: false, target_present: true}]

    Cover test (esophoria patient, R eye covered 3–18 s):
        head:   [{duration_s: 25}]
        target: [{duration_s: 25, lin_z_0: 2.0}]             ← target at 2 m
        scene:  [{duration_s: 25}]
        visual: [{duration_s: 3},                             ← both eyes open
                 {duration_s: 15, target_present_L: true, target_present_R: false},  ← cover R
                 {duration_s: 7}]                             ← uncover, re-fusion
        patient: {tonic_verg: 8.0}                            ← elevated tonic drive = esophoric
        plot: {panels: ['eye_position', 'vergence']}

    Gaze-evoked nystagmus (lit room — requires BOTH leaky NI AND bad pursuit):
        head:   [{duration_s: 10}]
        target: [{duration_s: 10, lin_z_0: 1.0}]             ← target eccentrically (e.g. 20°)
        scene:  [{duration_s: 10}]
        visual: [{duration_s: 10}]                            ← lit room with target
        patient: {tau_i: 4.0, K_pursuit: 0.2}                ← BOTH must be impaired

    Gaze-evoked nystagmus (dark — leaky NI alone is sufficient):
        visual: [{duration_s: 10, scene_present: false, target_present: false}]
        patient: {tau_i: 4.0}                                 ← pursuit irrelevant in dark

    ## CN lesion recipes (9-positions or saccade trajectory)

    CN VI nerve palsy (right LR only):
        patient: {g_nerve: [1,1,1,1,1,1, 0,1,1,1,1,1]}      ← index 6 = LR_R = 0
        plot: {panels: ['eye_position', 'eye_velocity']}

    CN VI nucleus palsy (right abducens nucleus → right LR only):
        patient: {g_nucleus: [1,0,1,1,1,1,1,1,1,1,1,1]}      ← index 1 = ABN_R = 0
        Note: in this model ABN nucleus lesion only affects ipsilateral LR (no MLF
        modelled at this level). For a full horizontal gaze palsy with conjugate
        MR involvement, model with both CN VI nerve AND right CN3_MR_R zeroed.

    CN III nerve palsy (right):
        patient: {g_nerve: [1,1,1,1,1,1, 1,0,0,0,1,0]}       ← MR_R(7), SR_R(8), IR_R(9), IO_R(11) = 0
        Right eye will sit lateral (LR unopposed) and slightly depressed (SO intact).

    CN IV nerve palsy (right SO palsy):
        patient: {g_nerve: [1,1,1,1,1,1, 1,1,1,1,0,1]}       ← index 10 = SO_R = 0
        Right hypertropia worst in left gaze and left head tilt.

    Left INO (right eye adducts normally, LEFT eye adduction lag on rightward gaze):
        patient: {g_mlf_L: 0.0}
        Use rightward saccade stimulus. Left eye yaw will be slow/absent.
        Convergence is preserved (CN3_MR_L direct vergence drive intact).

    Right INO (left eye adducts normally, RIGHT eye adduction lag on leftward gaze):
        patient: {g_mlf_R: 0.0}

    Bilateral INO / BIMLF:
        patient: {g_mlf_L: 0.0, g_mlf_R: 0.0}

    Partial CN VI nerve palsy (incomplete, recovering):
        patient: {g_nerve: [1,1,1,1,1,1, 0.3,1,1,1,1,1]}     ← LR_R at 30%

    Horizontal gaze palsy (CN VI nucleus + conjugate MR, simulated):
        patient: {g_nucleus: [1,0,1,1,1,1,1,1,1,1,1,1],
                  g_nerve:   [1,1,1,1,1,1, 1,0,1,1,1,1]}      ← ABN_R=0, MR_R nerve=0
    """

    description: str = Field(description="One-sentence plain-English description (used as figure title).")
    narrative: str = Field(
        default="",
        description=(
            "2–4 sentence plain-English explanation of how the clinical case is modelled: "
            "which parameters were changed from healthy defaults, what each change represents "
            "physiologically, and what the expected output looks like. "
            "Written for a clinician reader — no variable names, no jargon beyond standard "
            "vestibular/oculomotor terminology. "
            "Example: 'Left vestibular neuritis is modelled by setting the three left-ear "
            "canal gains to zero (indices 0–2: horizontal, anterior, posterior). The right "
            "vestibular nucleus fires unopposed at rest, driving a spontaneous rightward "
            "nystagmus. During leftward head impulses no compensatory signal arrives, producing "
            "corrective catch-up saccades; rightward impulses remain intact.'"
        )
    )

    head: list[BodySegment] = Field(
        min_length=1,
        description="Piecewise head 6-DOF motion. rot_* → canals; lin_* → otoliths (future)."
    )
    target: list[BodySegment] = Field(
        min_length=1,
        description=(
            "Piecewise target kinematics in 3-D world coordinates (metres). "
            "lin_x_0/y_0/z_0 = position (m), lin_x_vel/y_vel/z_vel = velocity (m/s). "
            "lin_z_0 = viewing distance (default 1 m). "
            "To place at angle θ: lin_x_0 = tan(θ_deg × π/180) × lin_z_0."
        )
    )
    scene: list[BodySegment] = Field(
        min_length=1,
        description=(
            "Piecewise scene / visual-background motion. "
            "rot_yaw_vel = scene angular velocity driving OKR (deg/s). "
            "All zeros = stationary lit room."
        )
    )
    visual: list[VisualFlagsSegment] = Field(
        min_length=1,
        description=(
            "Piecewise scene_present / target_present flags. "
            "Both default True (lit room + target visible). "
            "Use scene_present=False for darkness; target_present=False for OKN drum (no fixation dot)."
        )
    )
    patient: Patient = Field(default_factory=Patient)
    plot: PlotConfig

    @property
    def duration_s(self) -> float:
        """Derived total duration = max channel sum."""
        return max(
            sum(s.duration_s for s in self.head),
            sum(s.duration_s for s in self.target),
            sum(s.duration_s for s in self.scene),
            sum(s.duration_s for s in self.visual),
        )

    @model_validator(mode='after')
    def _check_total_duration(self):
        dur = self.duration_s
        if dur < 0.5:
            raise ValueError(f'Total duration {dur:.2f} s is too short (minimum 0.5 s)')
        if dur > 120:
            raise ValueError(f'Total duration {dur:.2f} s exceeds the 120 s limit')
        return self


# ── Comparison schema ─────────────────────────────────────────────────────────

class SimulationComparison(BaseModel):
    """2–4 scenarios overlaid on the same figure.

    All scenarios should share identical stimulus (head, target, scene, visual)
    and differ only in patient parameters.
    """

    title: str = Field(description="Figure title, e.g. 'Healthy vs Left Neuritis — vHIT'.")
    narrative: str = Field(
        default="",
        description=(
            "2–4 sentence explanation of what differs between the compared conditions, "
            "what parameters encode each condition, and what the reader should look for in "
            "the overlay. Plain English, clinician-facing, no variable names."
        )
    )
    panels: list[Literal[
        'eye_position', 'eye_velocity', 'head_velocity', 'gaze_error',
        'retinal_error', 'canal_afferents', 'velocity_storage',
        'neural_integrator', 'saccade_burst', 'pursuit_drive', 'refractory',
        'vergence',
        'target_position', 'target_velocity', 'scene_velocity', 'visual_flags',
    ]] = Field(description="Panels applied to all scenarios in the comparison.")
    scenarios: list[SimulationScenario] = Field(
        min_length=2, max_length=4,
        description=(
            "2–4 scenarios. Same stimulus, differ only in patient. "
            "Each scenario.description becomes its legend label (keep ≤5 words)."
        )
    )

    @model_validator(mode='after')
    def _check_durations_match(self):
        durations = [s.duration_s for s in self.scenarios]
        if max(durations) - min(durations) > 0.5:
            raise ValueError(
                f'All scenarios should have approximately equal duration. '
                f'Got: {[round(d, 2) for d in durations]}')
        return self


# ── JSON schema export ─────────────────────────────────────────────────────────

def json_schema() -> dict:
    return SimulationScenario.model_json_schema()


def comparison_json_schema() -> dict:
    return SimulationComparison.model_json_schema()
