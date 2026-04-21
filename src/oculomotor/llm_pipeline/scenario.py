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
                     False = that eye is covered (eye patch) → vergence leaks toward phoria.

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
    """
    duration_s:       float          = Field(gt=0, le=120, description="Duration of this segment (s).")
    scene_present:    bool           = Field(default=True,  description="Both-eye shorthand: True = lit room → OKR active for both eyes.")
    scene_present_L:  Optional[bool] = Field(default=None,  description="L-eye override. None = inherit scene_present. False = L eye in darkness.")
    scene_present_R:  Optional[bool] = Field(default=None,  description="R-eye override. None = inherit scene_present. False = R eye in darkness.")
    target_present:   bool           = Field(default=True,  description="Both-eye shorthand: True = target visible for both eyes.")
    target_present_L: Optional[bool] = Field(default=None,  description="L-eye override. None = inherit target_present. False = cover L eye.")
    target_present_R: Optional[bool] = Field(default=None,  description="R-eye override. None = inherit target_present. False = cover R eye.")


# ── Patient (unchanged) ────────────────────────────────────────────────────────

class Patient(BaseModel):
    """Model parameter overrides relative to healthy defaults.

    Only specify parameters that differ from the healthy default.
    Leave all others at their defaults.

    Gaze-evoked nystagmus (GEN) — IMPORTANT
    ----------------------------------------
    GEN requires a leaky neural integrator (tau_i small) so gaze positions
    drift centripetally.  But in a lit room with a target, smooth pursuit
    compensates that drift → no nystagmus is visible.

    To simulate GEN you must therefore impair BOTH:
        tau_i    = 3–8 s    (leaky NI; cerebellar/brainstem lesion)
        K_pursuit = 0.1–0.3  (severely impaired pursuit; cerebellar)
    With K_pursuit ≥ 1 pursuit compensates the NI drift and GEN disappears,
    even when tau_i is very short.

    In the dark (scene_present=False, target_present=False) pursuit and OKR
    are absent, so GEN is always visible with a leaky NI regardless of K_pursuit.

    Quick reference — typical cerebellar patient for GEN:
        tau_i=4, K_pursuit=0.2, tau_vs=5, K_vs=0.05
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
    b_vs_L: float = Field(
        default=100.0,
        description=(
            "Left vestibular nucleus bias and responsiveness (deg/s). Healthy 100. "
            "Controls both resting firing rate and canal drive gain for the left VN. "
            "70 = vestibular neuritis (intrinsic activity survives, afferent drive lost). "
            "0  = VN infarct (population completely silent, no canal drive). "
            "Use alongside canal_gains=0 for neuritis; b_vs_L=0 alone for VN infarct."
        )
    )
    b_vs_R: float = Field(
        default=100.0,
        description=(
            "Right vestibular nucleus bias and responsiveness (deg/s). Healthy 100. "
            "Same scale as b_vs_L. 70 = right neuritis; 0 = right VN infarct."
        )
    )
    tau_vs: float = Field(default=20.0, description="Velocity storage TC (s). Healthy 20 s. Nodulus/uvula lesion → 1–3 s. OKN/OKAN decay TC.")
    K_vs:   float = Field(default=0.1,  description="Canal→VS charging gain (1/s). Healthy 0.1. Reduce with tau_vs for nodulus lesions.")
    K_vis:  float = Field(default=0.1,  description="Visual→VS charging gain (1/s). Healthy 0.1. OKR/OKAN drive. 0 = no OKR.")
    g_vis:  float = Field(default=0.6,  description="Direct visual feedthrough gain (Raphan 1979). Healthy 0.6. Fast OKR onset component. OKR inner-loop stable when g_vis < 1.")
    tau_i:  float = Field(default=25.0,
        description=(
            "Neural integrator TC (s). Healthy 25 s. "
            "Short (2–8 s) → centripetal drift → gaze-evoked nystagmus (GEN). "
            "GEN visible in lit room only if pursuit also impaired (K_pursuit ≤ 0.3). "
            "Cerebellar lesion: pair with K_pursuit=0.1–0.3."
        ))
    g_burst: float = Field(default=700.0, description="Saccade burst ceiling (deg/s). Healthy 700. 0 = complete palsy. 200–400 = slow saccades (PSP, SCA).")
    K_pursuit:        float = Field(default=4.0,
        description=(
            "Pursuit integration gain (1/s). Healthy 4.0. Rise TC ≈ 1/K_pursuit. "
            "Cerebellar/MT lesion → 0.1–0.5 (severe pursuit deficit). "
            "Reduce alongside tau_i to see GEN in a lit room."
        ))
    K_phasic_pursuit: float = Field(default=5.0,  description="Pursuit direct feedthrough gain. Healthy 5.0. Controls fast pursuit onset velocity step.")
    tau_pursuit:      float = Field(default=40.0, description="Pursuit leak TC (s). Healthy 40 s → ~97% gain at 1 Hz. Short (5–15 s) → poor pursuit maintenance.")
    K_grav: float = Field(default=0.5, description="Otolith gravity correction gain. Healthy 0.5. Relevant for tilt / OVAR / off-vertical axis rotation.")

    # Adaptation time constants
    tau_vs_adapt: float = Field(
        default=600.0,
        description=(
            "VS null-adaptation TC (s). Default 600 s → negligible in short demos. "
            "Reduce to 30–60 s to model PAN (periodic alternating nystagmus) — "
            "slow oscillatory null-point drift due to VS adaptation."
        ))
    tau_ni_adapt: float = Field(
        default=20.0,
        description=(
            "NI null-adaptation TC (s). Default 20 s → rebound nystagmus after sustained eccentric gaze. "
            "Longer (>60 s) → weaker rebound; shorter (5–10 s) → strong rebound after brief gaze deviation. "
            "Cerebellar/brainstem lesion may impair or abolish rebound nystagmus."
        ))

    # Vergence — binocular disparity-driven convergence / divergence
    K_verg:        float              = Field(default=4.0,
        description="Vergence integrator gain (1/s). Healthy ~4. Reduce for convergence insufficiency.")
    K_phasic_verg: float              = Field(default=1.0,
        description="Vergence direct feedthrough (dim'less). Healthy ~1. Controls fast vergence onset.")
    tau_verg:      float              = Field(default=25.0,
        description="Vergence position leak TC (s). Healthy >20 s. Short → vergence doesn't hold.")
    phoria: Annotated[list[float], Field(min_length=3, max_length=3)] = Field(
        default=[0.0, 0.0, 0.0],
        description=(
            "Resting vergence angle [H, V, torsion] (deg) when binocular fusion is absent. "
            "phoria[0] > 0 = esophoria (over-convergence tendency); "
            "phoria[0] < 0 = exophoria (divergence tendency). "
            "Healthy = [0,0,0] (orthophoria). "
            "Cover test reveals phoria: cover one eye, vergence leaks toward phoria."
        )
    )

    # ── Motor nucleus gains (Stage 1) ──────────────────────────────────────────
    g_nucleus: Annotated[list[float], Field(min_length=12, max_length=12)] = Field(
        default=[1.0] * 12,
        description=(
            "Per-nucleus gains [0=complete lesion, 1=intact]. 12 values in order:\n"
            "  [0] ABN_L   — left  abducens nucleus  (CN VI, drives LR_L)\n"
            "  [1] ABN_R   — right abducens nucleus  (CN VI, drives LR_R)\n"
            "  [2] CN4_L   — left  trochlear nucleus (CN IV, drives contralateral SO_R)\n"
            "  [3] CN4_R   — right trochlear nucleus (CN IV, drives contralateral SO_L)\n"
            "  [4] CN3_MR_L — left  CN III medial rectus subnucleus (drives MR_L)\n"
            "  [5] CN3_MR_R — right CN III medial rectus subnucleus (drives MR_R)\n"
            "  [6] CN3_SR_L — left  CN III superior rectus subnucleus\n"
            "  [7] CN3_SR_R — right CN III superior rectus subnucleus\n"
            "  [8] CN3_IR_L — left  CN III inferior rectus subnucleus\n"
            "  [9] CN3_IR_R — right CN III inferior rectus subnucleus\n"
            " [10] CN3_IO_L — left  CN III inferior oblique subnucleus\n"
            " [11] CN3_IO_R — right CN III inferior oblique subnucleus\n"
            "Examples:\n"
            "  CN VI nucleus palsy (R): index 1 → 0  → right LR paralysed\n"
            "  CN IV nucleus palsy (R): index 3 → 0  → left SO paralysed (contralateral projection)\n"
            "  CN III nucleus palsy (R): indices 5,7,9,11 → 0  → right MR/SR/IR/IO paralysed"
        )
    )

    # ── Per-nerve (fascicular / peripheral) gains (Stage 2) ────────────────────
    g_nerve: Annotated[list[float], Field(min_length=12, max_length=12)] = Field(
        default=[1.0] * 12,
        description=(
            "Per-nerve gains [0=complete palsy, 1=intact]. 12 values in order:\n"
            "Left-eye nerves (indices 0–5):\n"
            "  [0] LR_L  — left lateral rectus  (CN VI nerve)\n"
            "  [1] MR_L  — left medial rectus   (CN III nerve)\n"
            "  [2] SR_L  — left superior rectus (CN III nerve)\n"
            "  [3] IR_L  — left inferior rectus (CN III nerve)\n"
            "  [4] SO_L  — left superior oblique (CN IV nerve)\n"
            "  [5] IO_L  — left inferior oblique (CN III nerve)\n"
            "Right-eye nerves (indices 6–11):\n"
            "  [6] LR_R  — right lateral rectus  (CN VI nerve)\n"
            "  [7] MR_R  — right medial rectus   (CN III nerve)\n"
            "  [8] SR_R  — right superior rectus (CN III nerve)\n"
            "  [9] IR_R  — right inferior rectus (CN III nerve)\n"
            " [10] SO_R  — right superior oblique (CN IV nerve)\n"
            " [11] IO_R  — right inferior oblique (CN III nerve)\n"
            "Examples:\n"
            "  CN VI nerve palsy (R):  index 6 → 0   → right LR only\n"
            "  CN III nerve palsy (R): indices 7,8,9,11 → 0  → right eye fixed lateral/down\n"
            "  CN IV nerve palsy (R):  index 10 → 0  → right SO: hypertropia in adduction\n"
            "  Partial palsy (e.g. recovering CN VI): index 6 → 0.4"
        )
    )

    # ── MLF (medial longitudinal fasciculus) integrity ─────────────────────────
    g_mlf_ver_L: float = Field(
        default=1.0,
        description=(
            "Version-yaw gain for the left MR subnucleus (CN3_MR_L). "
            "Models the MLF internuclear pathway from right ABN → left MR. "
            "0 = left INO: left eye adduction is absent on rightward conjugate gaze; "
            "convergence (vergence component) is preserved. "
            "Partial values (0.3–0.7) = incomplete INO / recovering INO. "
            "Healthy = 1.0."
        )
    )
    g_mlf_ver_R: float = Field(
        default=1.0,
        description=(
            "Version-yaw gain for the right MR subnucleus (CN3_MR_R). "
            "Models the MLF internuclear pathway from left ABN → right MR. "
            "0 = right INO: right eye adduction absent on leftward conjugate gaze; "
            "convergence preserved. "
            "Set both g_mlf_ver_L=0 and g_mlf_ver_R=0 for bilateral INO (BIMLF). "
            "Healthy = 1.0."
        )
    )

    @field_validator('canal_gains')
    @classmethod
    def _check_canal_gains(cls, v):
        for i, g in enumerate(v):
            if not (0.0 <= g <= 1.0):
                raise ValueError(f'canal_gains[{i}]={g} out of range [0, 1]')
        return v

    @field_validator('g_nucleus', 'g_nerve')
    @classmethod
    def _check_nerve_gains(cls, v, info):
        for i, g in enumerate(v):
            if not (0.0 <= g <= 1.0):
                raise ValueError(f'{info.field_name}[{i}]={g} out of range [0, 1]')
        return v

    @field_validator('g_mlf_ver_L', 'g_mlf_ver_R')
    @classmethod
    def _check_mlf(cls, v, info):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f'{info.field_name}={v} out of range [0, 1]')
        return v

    @field_validator('tau_vs', 'tau_i', 'tau_pursuit', 'tau_verg', 'tau_vs_adapt', 'tau_ni_adapt')
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

    @field_validator('K_vs', 'K_vis', 'K_pursuit', 'K_phasic_pursuit', 'K_grav',
                     'K_verg', 'K_phasic_verg')
    @classmethod
    def _check_nonneg_gains(cls, v, info):
        if v < 0:
            raise ValueError(f'{info.field_name} must be ≥ 0')
        return v

    @field_validator('phoria')
    @classmethod
    def _check_phoria(cls, v):
        for i, p in enumerate(v):
            if abs(p) > 30:
                raise ValueError(f'phoria[{i}]={p} deg is extreme (>30 deg); typical range ±15 deg')
        return v


# ── PlotConfig (unchanged) ─────────────────────────────────────────────────────

class PlotConfig(BaseModel):
    """Which signal panels to include in the figure."""

    panels: list[Literal[
        'eye_position', 'eye_velocity', 'head_velocity',
        'gaze_error', 'retinal_error', 'canal_afferents',
        'velocity_storage', 'neural_integrator',
        'saccade_burst', 'pursuit_drive', 'refractory',
        'vergence',
        'target_position', 'target_velocity', 'scene_velocity', 'visual_flags',
    ]] = Field(
        description=(
            "Ordered list of panels. Minimal sets:\n"
            "  VOR / HIT:    ['head_velocity', 'eye_velocity', 'eye_position']\n"
            "  OKN / OKAN:   ['scene_velocity', 'visual_flags', 'eye_velocity', 'eye_position', 'velocity_storage']\n"
            "  Saccades:     ['target_position', 'eye_position', 'eye_velocity', 'saccade_burst', 'refractory']\n"
            "  Pursuit:      ['eye_position', 'eye_velocity', 'pursuit_drive']\n"
            "  Vergence / cover test: ['eye_position', 'vergence', 'neural_integrator']\n"
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
        patient: {phoria: [8, 0, 0]}                          ← 8° esophoria
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
        patient: {g_mlf_ver_L: 0.0}
        Use rightward saccade stimulus. Left eye yaw will be slow/absent.
        Convergence is preserved (vrg component intact).

    Right INO (left eye adducts normally, RIGHT eye adduction lag on leftward gaze):
        patient: {g_mlf_ver_R: 0.0}

    Bilateral INO / BIMLF:
        patient: {g_mlf_ver_L: 0.0, g_mlf_ver_R: 0.0}

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
