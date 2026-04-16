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
             Use lin_x/y/z_0 for exact Cartesian placement.
             Use rot_yaw_0 / rot_pitch_0 as angular shorthand: auto-converted
             to lin_x/y = tan(angle_deg) × z_depth.  lin_z_0 default = 1 m.
             Use rot_yaw_vel / rot_pitch_vel as angular-velocity shorthand
             (converted to lin_x/y_vel = ang_vel × π/180 × z_depth).
             Runner always projects 3-D target–head vector to retinal angles.
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

    Scene MOTION is specified via scene BodySegment rot_yaw_vel — NOT here.

    Common combinations
    -------------------
    VOR in the dark:       scene_present=False, target_present=False
    VOR fixating in dark:  scene_present=False, target_present=True  (HIT)
    VVOR / saccades:       scene_present=True,  target_present=True  (default)
    OKN drum, no dot:      scene_present=True,  target_present=False
    Smooth pursuit:        scene_present=True,  target_present=True
    """
    duration_s:     float = Field(gt=0, le=120, description="Duration of this segment (s).")
    scene_present:  bool  = Field(default=True,  description="True = lit room → OKR active.")
    target_present: bool  = Field(default=True,  description="True = discrete target visible → pursuit/saccades active.")


# ── Patient (unchanged) ────────────────────────────────────────────────────────

class Patient(BaseModel):
    """Model parameter overrides relative to healthy defaults.

    Only specify parameters that differ from the healthy default.
    Leave all others at their defaults.
    """

    canal_gains: Annotated[list[float], Field(min_length=6, max_length=6)] = Field(
        default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        description=(
            "Per-canal sensitivity scale [0=complete paresis, 1=fully intact]. "
            "Canal order: [L_HC, L_AC, L_PC, R_HC, R_AC, R_PC]. "
            "Left vestibular neuritis → [0,0,0,1,1,1]; "
            "right neuritis → [1,1,1,0,0,0]; bilateral loss → [0,0,0,0,0,0]."
        )
    )
    tau_vs: float = Field(default=20.0, description="Velocity storage TC (s). Healthy ~20 s. Nodulus lesion → 1–3 s.")
    K_vs:   float = Field(default=0.1,  description="Canal→VS gain (1/s). Healthy ~0.1. Reduce with tau_vs for nodulus lesions.")
    K_vis:  float = Field(default=1.0,  description="Visual (OKR) gain into VS (1/s). Healthy ~1.0. 0 = no OKR.")
    g_vis:  float = Field(default=0.3,  description="Visual feedthrough gain. Healthy ~0.3. Fast OKR onset.")
    tau_i:  float = Field(default=25.0, description="Neural integrator TC (s). Healthy ≥20 s. Short (2–8 s) → gaze-evoked nystagmus. Cerebellar lesions.")
    g_burst: float = Field(default=700.0, description="Saccade burst ceiling (deg/s). Healthy ~700. 0 = complete palsy. 200–400 = slow saccades (PSP, SCA).")
    K_pursuit:        float = Field(default=2.0,  description="Pursuit integrator gain (1/s). Reduce for pursuit deficit (MT/MST, Parkinson's → 0.2–0.5).")
    K_phasic_pursuit: float = Field(default=5.0,  description="Pursuit direct feedthrough gain. Healthy ~5. Provides fast pursuit onset.")
    tau_pursuit:      float = Field(default=40.0, description="Pursuit integrator TC (s). Healthy ~40 s. Short (5–15 s) → poor pursuit maintenance.")
    K_grav: float = Field(default=0.5, description="Otolith gravity estimation gain. Relevant for tilt / OVAR.")

    @field_validator('canal_gains')
    @classmethod
    def _check_canal_gains(cls, v):
        for i, g in enumerate(v):
            if not (0.0 <= g <= 1.0):
                raise ValueError(f'canal_gains[{i}]={g} out of range [0, 1]')
        return v

    @field_validator('tau_vs', 'tau_i', 'tau_pursuit')
    @classmethod
    def _check_positive_tc(cls, v, info):
        if v <= 0:
            raise ValueError(f'{info.field_name} must be > 0')
        return v

    @field_validator('g_burst')
    @classmethod
    def _check_burst(cls, v):
        if v < 0:
            raise ValueError('g_burst must be ≥ 0 (use 0 to disable saccades)')
        if v > 1000:
            raise ValueError(f'g_burst={v} is physiologically unrealistic (max ~700)')
        return v

    @field_validator('K_vs', 'K_vis', 'K_pursuit', 'K_phasic_pursuit', 'K_grav')
    @classmethod
    def _check_nonneg_gains(cls, v, info):
        if v < 0:
            raise ValueError(f'{info.field_name} must be ≥ 0')
        return v


# ── PlotConfig (unchanged) ─────────────────────────────────────────────────────

class PlotConfig(BaseModel):
    """Which signal panels to include in the figure."""

    panels: list[Literal[
        'eye_position', 'eye_velocity', 'head_velocity',
        'gaze_error', 'retinal_error', 'canal_afferents',
        'velocity_storage', 'neural_integrator',
        'saccade_burst', 'pursuit_drive', 'refractory',
    ]] = Field(
        description=(
            "Ordered list of panels. Minimal sets:\n"
            "  VOR / HIT:    ['head_velocity', 'eye_velocity', 'eye_position']\n"
            "  OKN / OKAN:   ['head_velocity', 'eye_velocity', 'eye_position', 'velocity_storage']\n"
            "  Saccades:     ['eye_position', 'eye_velocity', 'saccade_burst', 'refractory']\n"
            "  Pursuit:      ['eye_position', 'eye_velocity', 'pursuit_drive']\n"
            "  Full cascade: all panels"
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
        target: [{duration_s: 0.3, lin_z_0: 1.0},
                 {duration_s: 1.7, rot_yaw_0: 20}]          ← shorthand for lin_x_0=0.364
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
    """

    description: str = Field(description="One-sentence plain-English description (used as figure title).")

    head: list[BodySegment] = Field(
        min_length=1,
        description="Piecewise head 6-DOF motion. rot_* → canals; lin_* → otoliths (future)."
    )
    target: list[BodySegment] = Field(
        min_length=1,
        description=(
            "Piecewise target kinematics. Specify in 3-D world coordinates (lin_*) in metres, "
            "or use rot_yaw_0/rot_yaw_vel as angular shorthand (auto-converted to Cartesian). "
            "lin_z_0 = viewing distance (default 1 m)."
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
    panels: list[Literal[
        'eye_position', 'eye_velocity', 'head_velocity', 'gaze_error',
        'retinal_error', 'canal_afferents', 'velocity_storage',
        'neural_integrator', 'saccade_burst', 'pursuit_drive', 'refractory',
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
