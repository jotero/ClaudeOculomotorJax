"""SimulationScenario schema — structured representation of a natural-language scenario.

An LLM converts a plain-English description into a ``SimulationScenario`` JSON object.
The runner (``oculomotor/runner.py``) converts that object into stimulus arrays + model
parameters and runs the simulation.

Schema overview
---------------
    SimulationScenario
    ├── description:  str                   free-text label
    ├── head:         list[HeadSegment]     piecewise head motion
    ├── target:       list[TargetSegment]   piecewise target kinematics
    ├── visual:       list[VisualSegment]   piecewise scene/lighting conditions
    ├── patient:      Patient               model parameter overrides vs. healthy
    └── plot:         PlotConfig            which panels to show

All three channels (head, target, visual) are piecewise — each segment specifies
what happens during a contiguous time window.  The runner concatenates them in order.
If channels have different total durations the shorter ones are held at their last
value (or padded with zeros for head velocity).

Total simulation duration = max(sum head durations, sum target durations, sum visual durations).
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Head segments ──────────────────────────────────────────────────────────────

class HeadSegment(BaseModel):
    """One contiguous segment of head motion.

    Profiles
    --------
    'constant' — constant angular velocity (use velocity_deg_s=0 for still)
    'sinusoid' — sinusoidal: v(t) = velocity_deg_s * sin(2π * frequency_hz * t)
    'impulse'  — brief high-velocity pulse (vHIT): rises and falls in ramp_dur_s,
                 then coasts at 0 for the remainder of duration_s
    """

    duration_s: float = Field(
        gt=0, le=120,
        description="Duration of this segment (s). Must be > 0."
    )
    velocity_deg_s: float = Field(
        default=0.0,
        description=(
            "Velocity or amplitude (deg/s). "
            "For 'constant': constant angular velocity (0 = still). "
            "For 'sinusoid': peak amplitude. "
            "For 'impulse': peak velocity (150–300 deg/s for vHIT). "
            "Sign = direction: positive rightward/upward, negative leftward/downward."
        )
    )
    profile: Literal['constant', 'sinusoid', 'impulse'] = Field(
        default='constant',
        description=(
            "'constant' — constant velocity (0 = stationary). "
            "'sinusoid' — oscillation at frequency_hz. "
            "'impulse'  — brief vHIT-like pulse (rise/fall in ramp_dur_s)."
        )
    )
    frequency_hz: float = Field(
        default=1.0,
        description="Sinusoid frequency (Hz). Only used when profile='sinusoid'. Typical 0.1–2 Hz."
    )
    ramp_dur_s: float = Field(
        default=0.02,
        description=(
            "Rise and fall duration for impulse profile (s). "
            "Default 0.02 s (20 ms) is realistic for a clinical vHIT. "
            "Only used when profile='impulse'."
        )
    )
    axis: Literal['yaw', 'pitch', 'roll'] = Field(
        default='yaw',
        description="Rotation axis. 'yaw'=horizontal, 'pitch'=vertical, 'roll'=torsional."
    )

    @field_validator('velocity_deg_s')
    @classmethod
    def _check_amplitude(cls, v):
        if abs(v) > 1000:
            raise ValueError(f'|velocity_deg_s|={abs(v)} is physiologically unrealistic (max ~300 deg/s for vHIT)')
        return v

    @field_validator('frequency_hz')
    @classmethod
    def _check_freq(cls, v):
        if v <= 0 or v > 20:
            raise ValueError('frequency_hz must be in (0, 20] Hz')
        return v

    @model_validator(mode='after')
    def _check_impulse(self):
        if self.profile == 'impulse':
            if self.ramp_dur_s <= 0:
                raise ValueError('ramp_dur_s must be > 0 for impulse profile')
            if 2 * self.ramp_dur_s >= self.duration_s:
                raise ValueError(
                    f'impulse ramp ({2*self.ramp_dur_s:.3f} s rise+fall) must be < '
                    f'segment duration ({self.duration_s} s). Increase duration_s.')
        return self


# ── Target segments ────────────────────────────────────────────────────────────

class TargetSegment(BaseModel):
    """One contiguous segment of target kinematics.

    The target position accumulates continuously across segments.
    A position jump (e.g. saccade target) is specified with position_yaw_deg /
    position_pitch_deg — set these only when you want an instantaneous jump at the
    start of the segment.  Leave them as None to continue from the previous position.

    Examples
    --------
    Stationary target at 0°:
        {duration_s: 1.0}   # all defaults

    Jump to 20° right at t=0.2 s:
        [{duration_s: 0.2},
         {duration_s: 1.8, position_yaw_deg: 20.0}]

    Pursuit ramp at 20 deg/s starting at t=0.3 s:
        [{duration_s: 0.3},
         {duration_s: 3.7, velocity_yaw_deg_s: 20.0}]
    """

    duration_s: float = Field(gt=0, le=120, description="Duration of this segment (s).")
    position_yaw_deg: Optional[float] = Field(
        default=None,
        description=(
            "If set: target jumps to this absolute yaw position (deg) at the start of this segment. "
            "If None: position continues from previous segment. "
            "Right = positive."
        )
    )
    position_pitch_deg: Optional[float] = Field(
        default=None,
        description=(
            "If set: target jumps to this absolute pitch position (deg) at segment start. "
            "Up = positive."
        )
    )
    velocity_yaw_deg_s: float = Field(
        default=0.0,
        description="Target angular velocity in yaw (deg/s). Positive = rightward."
    )
    velocity_pitch_deg_s: float = Field(
        default=0.0,
        description="Target angular velocity in pitch (deg/s). Positive = upward."
    )


# ── Visual / scene segments ────────────────────────────────────────────────────

class VisualSegment(BaseModel):
    """Scene and target-visibility conditions for one time segment.

    Two independent flags govern visual input:

    scene_present  — is the room lit?  (controls OKR / visual stabilisation)
    target_present — is there a discrete fixation / pursuit target?  (controls pursuit integrator)

    Common combinations
    -------------------
    VOR in the dark:        scene_present=false, target_present=false
    VOR fixating in dark:   scene_present=false, target_present=true
    VVOR (lit room):        scene_present=true,  target_present=true
    OKN (drum, no dot):     scene_present=true,  target_present=false, scene_velocity_deg_s=30
    Saccades lit room:      scene_present=true,  target_present=true
    Smooth pursuit:         scene_present=true,  target_present=true

    For OKN + OKAN use two segments:
        [{duration_s: 20, scene_present: true, target_present: false, scene_velocity_deg_s: 30},
         {duration_s: 40, scene_present: true, target_present: false, scene_velocity_deg_s: 0}]
    """

    duration_s: float = Field(gt=0, le=120, description="Duration of this segment (s).")
    scene_present: bool = Field(
        default=True,
        description=(
            "True = lit room → drives OKR and visual stabilisation. "
            "False = complete darkness. "
            "VOR-in-dark / HIT: False. Saccades / pursuit / OKN: True."
        )
    )
    target_present: bool = Field(
        default=True,
        description=(
            "True = a discrete foveal target exists → activates smooth pursuit + saccade system. "
            "False = no discrete target (OKN drum only, VOR-in-dark). "
            "OKN: False (no fixation point). Saccades / pursuit / VVOR / HIT: True."
        )
    )
    scene_velocity_deg_s: float = Field(
        default=0.0,
        description=(
            "Angular velocity of the full visual scene (deg/s). Non-zero only for OKN. "
            "Requires scene_present=True. "
            "Positive = rightward (for yaw axis)."
        )
    )
    scene_axis: Literal['yaw', 'pitch', 'roll'] = Field(
        default='yaw',
        description="Axis of scene motion. 'yaw' = horizontal OKN, 'pitch' = vertical OKN."
    )

    @model_validator(mode='after')
    def _check_scene_vel(self):
        if self.scene_velocity_deg_s != 0.0 and not self.scene_present:
            raise ValueError(
                'scene_velocity_deg_s is non-zero but scene_present=False. '
                'Set scene_present=True for OKN/OKR stimulation.')
        return self


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
            "Canal order: [L_HC, L_AC, L_PC, R_HC, R_AC, R_PC] "
            "(L/R = left/right; HC = horizontal, AC = anterior, PC = posterior). "
            "Push-pull pairs: L_HC+R_HC (yaw), L_AC+R_PC (LARP), L_PC+R_AC (RALP). "
            "Examples: "
            "left vestibular neuritis → [0,0,0,1,1,1]; "
            "right neuritis → [1,1,1,0,0,0]; "
            "bilateral loss → [0,0,0,0,0,0]."
        )
    )
    tau_vs: float = Field(
        default=20.0,
        description=(
            "Velocity storage time constant (s). Healthy ~20 s. "
            "Extends effective VOR TC beyond mechanical canal TC (~5 s). "
            "Controls OKAN duration. Nodulus/uvula lesion → 1–3 s."
        )
    )
    K_vs: float = Field(
        default=0.1,
        description=(
            "Canal→velocity-storage charging gain (1/s). Healthy ~0.1. "
            "Reduce along with tau_vs for nodulus lesions."
        )
    )
    K_vis: float = Field(
        default=1.0,
        description=(
            "Visual (OKR) charging gain into velocity storage (1/s). Healthy ~1.0. "
            "0 = no OKR (visual loss or darkness). "
            "OKN steady-state gain ≈ K_vis·tau_vs / (1 + K_vis·tau_vs)."
        )
    )
    g_vis: float = Field(
        default=0.3,
        description=(
            "Visual feedthrough gain (dimensionless). Healthy ~0.3. "
            "Provides fast direct-pathway OKR onset before integrator charges."
        )
    )
    tau_i: float = Field(
        default=25.0,
        description=(
            "Neural integrator leak time constant (s). Healthy ≥ 20 s. "
            "Short tau_i → gaze-evoked nystagmus (GEN): drift at eccentric gaze. "
            "Cerebellar lesions (flocculus, SCA6, alcohol) → 2–8 s."
        )
    )
    g_burst: float = Field(
        default=700.0,
        description=(
            "Excitatory burst neuron ceiling — peak saccade velocity (deg/s). "
            "Healthy ~700. 0 = complete saccadic palsy. "
            "200–400 = slow saccades (PSP, SCA, drugs)."
        )
    )
    K_pursuit: float = Field(
        default=2.0,
        description=(
            "Pursuit integrator gain (1/s). Healthy ~2. "
            "Rise TC ≈ (1 + K_phasic_pursuit) / K_pursuit. "
            "Reduce for pursuit deficits (MT/MST, Parkinson's → 0.2–0.5)."
        )
    )
    K_phasic_pursuit: float = Field(
        default=5.0,
        description=(
            "Pursuit direct feedthrough gain. Healthy ~5. "
            "Provides fast pursuit onset. High → quick initial eye acceleration."
        )
    )
    tau_pursuit: float = Field(
        default=40.0,
        description=(
            "Pursuit integrator leak TC (s). Healthy ~40 s. "
            "Short → pursuit velocity decays during sustained motion. "
            "Cerebellar → 5–15 s; Parkinson's → 10–20 s."
        )
    )
    K_grav: float = Field(
        default=0.5,
        description=(
            "Otolith-driven gravity estimation gain. Relevant for tilt / OVAR. "
            "Utricular damage → reduce toward 0."
        )
    )

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
            raise ValueError(f'{info.field_name} must be > 0 (got {v})')
        return v

    @field_validator('g_burst')
    @classmethod
    def _check_burst(cls, v):
        if v < 0:
            raise ValueError(f'g_burst must be ≥ 0 (use 0 to disable saccades)')
        if v > 1000:
            raise ValueError(f'g_burst={v} is physiologically unrealistic (max ~700)')
        return v

    @field_validator('K_vs', 'K_vis', 'K_pursuit', 'K_phasic_pursuit', 'K_grav')
    @classmethod
    def _check_nonneg_gains(cls, v, info):
        if v < 0:
            raise ValueError(f'{info.field_name} must be ≥ 0 (got {v})')
        return v


# ── PlotConfig (unchanged) ─────────────────────────────────────────────────────

class PlotConfig(BaseModel):
    """Which signal panels to include in the figure."""

    panels: list[Literal[
        'eye_position',
        'eye_velocity',
        'head_velocity',
        'gaze_error',
        'retinal_error',
        'canal_afferents',
        'velocity_storage',
        'neural_integrator',
        'saccade_burst',
        'pursuit_drive',
        'refractory',
    ]] = Field(
        description=(
            "Ordered list of panels to show. Choose the minimal set that tells the story:\n"
            "  VOR / HIT:         ['head_velocity', 'eye_velocity', 'eye_position']\n"
            "  VOR + VS:          ['head_velocity', 'eye_velocity', 'velocity_storage']\n"
            "  OKN / OKAN:        ['head_velocity', 'eye_velocity', 'eye_position', 'velocity_storage']\n"
            "  Saccades:          ['eye_position', 'eye_velocity', 'saccade_burst', 'refractory']\n"
            "  Pursuit:           ['eye_position', 'eye_velocity', 'pursuit_drive']\n"
            "  Full cascade:      all panels"
        )
    )
    title: str = Field(
        default='',
        description="Figure title (auto-set from description if empty)."
    )


# ── Top-level schema ───────────────────────────────────────────────────────────

class SimulationScenario(BaseModel):
    """Complete specification of an oculomotor simulation scenario.

    Each stimulus channel (head, target, visual) is a list of segments that
    are concatenated in time order.  This allows arbitrary stimulus sequences:
    alternating HITs, OKN+OKAN, pursuit with changing velocity, gap paradigms, etc.

    The simulation duration is derived automatically as the maximum of the total
    durations across all three channels.

    ## Quick segment examples

    Single rightward saccade to 20°:
        head:   [{duration_s: 2, velocity_deg_s: 0}]
        target: [{duration_s: 0.3}, {duration_s: 1.7, position_yaw_deg: 20}]
        visual: [{duration_s: 2, scene_present: true, target_present: true}]

    Rightward vHIT:
        head:   [{duration_s: 2.5, velocity_deg_s: 200, profile: "impulse"}]
        target: [{duration_s: 2.5, position_yaw_deg: 0}]
        visual: [{duration_s: 2.5, scene_present: false, target_present: true}]

    OKN 20 s + OKAN 40 s:
        head:   [{duration_s: 60, velocity_deg_s: 0}]
        target: [{duration_s: 60}]
        visual: [{duration_s: 20, scene_present: true, target_present: false, scene_velocity_deg_s: 30},
                 {duration_s: 40, scene_present: true, target_present: false}]

    Smooth pursuit 20 deg/s starting at 0.3 s:
        head:   [{duration_s: 5, velocity_deg_s: 0}]
        target: [{duration_s: 0.3}, {duration_s: 4.7, velocity_yaw_deg_s: 20}]
        visual: [{duration_s: 5, scene_present: true, target_present: true}]

    Alternating left+right HITs:
        head:   [{duration_s: 3, velocity_deg_s: 200, profile: "impulse"},
                 {duration_s: 3, velocity_deg_s: -200, profile: "impulse"}]
        target: [{duration_s: 6, position_yaw_deg: 0}]
        visual: [{duration_s: 6, scene_present: false, target_present: true}]

    VOR in the dark (step rotation):
        head:   [{duration_s: 5, velocity_deg_s: 60, profile: "constant"},
                 {duration_s: 15, velocity_deg_s: 0}]
        target: [{duration_s: 20}]
        visual: [{duration_s: 20, scene_present: false, target_present: false}]
    """

    description: str = Field(
        description="One-sentence plain-English description of the scenario (used as figure title)."
    )
    head: list[HeadSegment] = Field(
        min_length=1,
        description=(
            "Piecewise head motion. Segments are concatenated in order. "
            "Use [{duration_s: T, velocity_deg_s: 0}] for a stationary head."
        )
    )
    target: list[TargetSegment] = Field(
        min_length=1,
        description=(
            "Piecewise target kinematics. Segments are concatenated in order. "
            "Position accumulates continuously; set position_yaw_deg to jump at a segment boundary."
        )
    )
    visual: list[VisualSegment] = Field(
        min_length=1,
        description=(
            "Piecewise scene / lighting conditions. Segments are concatenated in order. "
            "Each segment specifies scene_present, target_present, and optional scene motion."
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
    """A set of 2–4 scenarios to overlay on the same figure.

    All scenarios should share the same stimulus (head, target, visual segments)
    and differ only in patient parameters.

    The shared panels list applies to all scenarios.
    """

    title: str = Field(
        description="Figure title, e.g. 'Healthy vs Left Neuritis — Head Impulse Test'."
    )
    panels: list[Literal[
        'eye_position', 'eye_velocity', 'head_velocity', 'gaze_error',
        'retinal_error', 'canal_afferents', 'velocity_storage',
        'neural_integrator', 'saccade_burst', 'pursuit_drive', 'refractory',
    ]] = Field(
        description=(
            "Panels to show — same choices as SimulationScenario.plot.panels. "
            "Applied to all scenarios in the comparison."
        )
    )
    scenarios: list[SimulationScenario] = Field(
        min_length=2,
        max_length=4,
        description=(
            "2–4 scenarios to compare. "
            "All must have identical stimulus (head, target, visual) and differ only in patient. "
            "Each scenario.description becomes its legend label — keep it short (≤5 words)."
        )
    )

    @model_validator(mode='after')
    def _check_durations_match(self):
        durations = [s.duration_s for s in self.scenarios]
        if max(durations) - min(durations) > 0.5:
            raise ValueError(
                f'All scenarios in a comparison should have approximately the same total duration. '
                f'Got: {[round(d, 2) for d in durations]}')
        return self


# ── JSON schema export (for LLM tool call) ────────────────────────────────────

def json_schema() -> dict:
    """Return the full JSON schema for use in an LLM tool_use call."""
    return SimulationScenario.model_json_schema()


def comparison_json_schema() -> dict:
    """Return JSON schema for SimulationComparison."""
    return SimulationComparison.model_json_schema()
