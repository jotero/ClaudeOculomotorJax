"""SimulationScenario schema — structured representation of a natural-language scenario.

An LLM converts a plain-English description (e.g. "healthy subject, 20 deg saccade
left") into a ``SimulationScenario`` JSON object.  The runner (``oculomotor/runner.py``)
converts that object into stimulus arrays + model parameters and runs the simulation.

Schema overview
---------------
    SimulationScenario
    ├── description:   str           free-text label shown on the figure
    ├── duration_s:    float         total simulation duration
    ├── head_motion:   HeadMotion    what the head does
    ├── target:        Target        what the visual target does
    ├── visual:        Visual        scene / lighting conditions
    ├── patient:       Patient       model parameter overrides vs. healthy defaults
    └── plot:          PlotConfig    which panels to show

Clinical condition → parameter table (for LLM system prompt)
-------------------------------------------------------------
Healthy:
    canal_gains=[1]*6, tau_vs=20, K_vs=0.1, K_vis=1.0, g_vis=0.3,
    tau_i=25, g_burst=700, K_pursuit=2, K_phasic_pursuit=5, tau_pursuit=40

Unilateral vestibular neuritis / labyrinthectomy (left):
    canal_gains=[0, 0, 0, 1, 1, 1]  # all left canals dead, right intact
    Canal order: [L_HC, L_AC, L_PC, R_HC, R_AC, R_PC]

Unilateral vestibular neuritis (right):
    canal_gains=[1, 1, 1, 0, 0, 0]

Bilateral vestibular loss (e.g. gentamicin ototoxicity):
    canal_gains=[0]*6, K_grav=0.0  # also affects otoliths

Cerebellar gaze-holding deficit (flocculus/SCA/alcohol):
    tau_i=2.0–5.0  # gaze-evoked nystagmus; worse with larger eccentricity

Velocity storage loss (nodulus/uvula lesion):
    tau_vs=1.0, K_vs=0.001  # no OKAN, short VOR TC

Saccade palsy — complete (bilateral PPRF or locked-in):
    g_burst=0.0

Slow saccades (PSP, SCA, drug):
    g_burst=200.0–400.0

Pursuit deficit (MT/MST or cerebellar):
    K_pursuit=0.3, K_phasic_pursuit=0.5, tau_pursuit=8.0

Canal paresis (single canal, e.g. BPPV canal):
    Set the appropriate canal_gains element to 0.0–0.5

Internuclear ophthalmoplegia (INO) — not directly modelled (requires binocular plant).
BPPV — brief positional vertigo; use pitch head impulse + partial canal paresis.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Sub-schemas ────────────────────────────────────────────────────────────────

class HeadMotion(BaseModel):
    """Description of how the head moves.

    Constraints:
      - amplitude_deg_s ≥ 0
      - rotate_dur_s > 0 (when type != 'none')
      - coast_dur_s ≥ 0
      - For 'impulse': rotate_dur_s ≤ 0.05 s (realistic vHIT rise time)
    """

    type: Literal['none', 'step', 'sinusoid', 'impulse'] = Field(
        description=(
            "'none'     — stationary head\n"
            "'step'     — constant angular velocity for rotate_dur, then coast\n"
            "'sinusoid' — sinusoidal oscillation at frequency_hz\n"
            "'impulse'  — brief high-velocity pulse (vHIT / head-impulse test)"
        )
    )
    amplitude_deg_s: float = Field(
        default=0.0,
        description=(
            "For 'step'/'impulse': peak angular velocity (deg/s). "
            "For 'sinusoid': peak velocity amplitude (deg/s). "
            "Typical ranges: step/OKN 10–60, HIT 150–300, sinusoid 10–50."
        )
    )
    rotate_dur_s: float = Field(
        default=1.0,
        description=(
            "Duration of active rotation phase (s). "
            "For 'step': constant-velocity phase. "
            "For 'impulse': rise/fall duration (keep ≤0.05 s for realistic HIT)."
        )
    )
    coast_dur_s: float = Field(
        default=1.0,
        description="Duration of zero-velocity coast after rotation (s)."
    )
    frequency_hz: float = Field(
        default=1.0,
        description="Oscillation frequency for 'sinusoid' type (Hz). Typical 0.1–2 Hz."
    )
    axis: Literal['yaw', 'pitch', 'roll'] = Field(
        default='yaw',
        description=(
            "Axis of head rotation. "
            "'yaw' = horizontal (tests horizontal canals). "
            "'pitch' = vertical. "
            "'roll' = torsional."
        )
    )

    @field_validator('amplitude_deg_s')
    @classmethod
    def _check_amplitude(cls, v):
        if v < 0:
            raise ValueError('amplitude_deg_s must be ≥ 0')
        return v

    @field_validator('rotate_dur_s')
    @classmethod
    def _check_rotate_dur(cls, v):
        if v < 0:
            raise ValueError('rotate_dur_s must be ≥ 0')
        return v

    @field_validator('coast_dur_s')
    @classmethod
    def _check_coast_dur(cls, v):
        if v < 0:
            raise ValueError('coast_dur_s must be ≥ 0')
        return v

    @field_validator('frequency_hz')
    @classmethod
    def _check_freq(cls, v):
        if v <= 0:
            raise ValueError('frequency_hz must be > 0')
        if v > 20:
            raise ValueError('frequency_hz > 20 Hz is outside the physiological range')
        return v

    @model_validator(mode='after')
    def _check_impulse_duration(self):
        if self.type == 'impulse' and self.rotate_dur_s > 0.05:
            raise ValueError(
                f'For head impulse (vHIT), rotate_dur_s must be ≤ 0.05 s '
                f'(got {self.rotate_dur_s} s). This is the rise time of the impulse.')
        return self


class TargetStep(BaseModel):
    """A single target position step."""
    t_s: float = Field(description="Time of step (s).")
    yaw_deg: float = Field(default=0.0, description="Target yaw after step (deg). Right positive.")
    pitch_deg: float = Field(default=0.0, description="Target pitch after step (deg). Up positive.")


class Target(BaseModel):
    """Description of the visual target."""

    type: Literal['stationary', 'steps', 'ramp'] = Field(
        description=(
            "'stationary' — fixed target (use for VOR, OKN, fixation)\n"
            "'steps'      — one or more instantaneous position jumps (use for saccades)\n"
            "'ramp'       — target moves at constant velocity after onset (use for pursuit)"
        )
    )
    initial_yaw_deg: float = Field(
        default=0.0,
        description="Initial target yaw position (deg). Right positive."
    )
    initial_pitch_deg: float = Field(
        default=0.0,
        description="Initial target pitch position (deg). Up positive."
    )
    steps: list[TargetStep] = Field(
        default_factory=list,
        description=(
            "List of position steps (for type='steps'). "
            "Example: [{'t_s': 0.2, 'yaw_deg': 15.0, 'pitch_deg': 0.0}]"
        )
    )
    ramp_velocity_deg_s: float = Field(
        default=0.0,
        description="Target velocity after ramp onset (deg/s). For type='ramp'."
    )
    ramp_onset_s: float = Field(
        default=0.2,
        description="Time when ramp starts (s). For type='ramp'."
    )
    ramp_axis: Literal['yaw', 'pitch'] = Field(
        default='yaw',
        description="Axis of ramp motion. For type='ramp'."
    )


class Visual(BaseModel):
    """Lighting and scene conditions.

    Two independent flags govern visual input:

    scene_present — is the room/background lit?
        Controls OKR and visual stabilisation (VS).
        True  = lit room (subject sees the walls, background, surroundings).
        False = complete darkness (no visual reference at all).

    target_present — is there a specific foveal fixation/pursuit target?
        Controls the smooth pursuit integrator.
        True  = a small fixation dot or moving target is visible.
        False = no discrete target (e.g. full-field OKN drum with no fixation point,
                or VOR in the dark where the subject just stares into darkness).

    Common combinations
    -------------------
    VOR in the dark:
        scene_present=False, target_present=False
        (no visual information at all — pure vestibular)

    VOR fixating a target in the dark (e.g. laser dot on wall):
        scene_present=False, target_present=True
        (suppresses VOR if target moves with head; tests visual-vestibular interaction)

    VVOR — VOR in a lit stationary room:
        scene_present=True, target_present=True
        (OKR corrects VOR slip; gaze stays stable)

    OKN — full-field optokinetic drum, no fixation target:
        scene_present=True, target_present=False, scene_velocity_deg_s=30
        (drives OKR/VS via slip; pursuit NOT driven; nystagmus with saccades)

    Saccades in a lit room to a stationary target:
        scene_present=True, target_present=True, no scene_velocity

    Smooth pursuit of a moving target in a lit room:
        scene_present=True, target_present=True, use target.type='ramp'

    Smooth pursuit in darkness (no OKR, only pursuit):
        scene_present=False, target_present=True, use target.type='ramp'
    """

    scene_present: bool = Field(
        default=True,
        description=(
            "True = lit room / visual background visible → drives OKR and visual stabilisation. "
            "False = complete darkness → no visual input whatsoever. "
            "VOR-in-dark: False. VVOR / OKN / saccades: True."
        )
    )
    target_present: bool = Field(
        default=True,
        description=(
            "True = a discrete foveal target exists (fixation dot, pursuit target) → "
            "activates smooth pursuit integrator. "
            "False = no foveal target (OKN drum only, VOR-in-dark, staring into space). "
            "OKN paradigm: False (no fixation point — subject watches the full-field drum). "
            "Saccades / pursuit: True."
        )
    )
    scene_velocity_deg_s: float = Field(
        default=0.0,
        description=(
            "Angular velocity of the full visual scene (deg/s). "
            "Non-zero only for OKN / optokinetic stimulation. "
            "Positive = scene moves in the positive axis direction. "
            "Requires scene_present=True to have any effect. "
            "Scene moves at this speed for scene_on_dur_s seconds, then stops "
            "(scene stays lit → OKAN driven by velocity storage)."
        )
    )
    scene_on_dur_s: float = Field(
        default=0.0,
        description=(
            "How long the scene moves (s). Only relevant when scene_velocity_deg_s ≠ 0. "
            "After scene_on_dur_s the scene stops moving but stays lit. "
            "Set equal to duration_s to keep moving for the whole trial. "
            "Must be ≤ duration_s."
        )
    )
    scene_axis: Literal['yaw', 'pitch', 'roll'] = Field(
        default='yaw',
        description="Axis of scene motion ('yaw' for horizontal OKN, 'pitch' for vertical OKN)."
    )


class Patient(BaseModel):
    """Model parameter overrides relative to healthy defaults.

    Only specify parameters that differ from the healthy default.
    Leave all others at their defaults.
    """

    # ── Semicircular canals ────────────────────────────────────────────────────
    canal_gains: Annotated[list[float], Field(min_length=6, max_length=6)] = Field(
        default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        description=(
            "Per-canal sensitivity scale [0=complete paresis, 1=fully intact]. "
            "Canal order: [L_HC, L_AC, L_PC, R_HC, R_AC, R_PC] "
            "(L/R = left/right; HC = horizontal, AC = anterior, PC = posterior). "
            "Determines the VOR gain and directionality for each plane of rotation. "
            "Push-pull pairs: L_HC+R_HC (yaw), L_AC+R_PC (LARP), L_PC+R_AC (RALP). "
            "Partial paresis (0.3–0.7) → reduced VOR gain in that canal's plane. "
            "Examples: "
            "left vestibular neuritis → [0,0,0,1,1,1]; "
            "right neuritis → [1,1,1,0,0,0]; "
            "bilateral loss → [0,0,0,0,0,0]; "
            "isolated left HC paresis → [0,1,1,1,1,1]; "
            "superior canal dehiscence affects AC → reduce L_AC or R_AC."
        )
    )

    # ── Velocity storage ───────────────────────────────────────────────────────
    tau_vs: float = Field(
        default=20.0,
        description=(
            "Velocity storage time constant (s). Healthy primate ~20 s. "
            "Extends the effective VOR time constant beyond the mechanical canal TC (~5 s). "
            "Controls how long OKAN (optokinetic after-nystagmus) persists after scene off. "
            "Lesions of the nodulus/uvula (cerebellar) → 1–3 s (short VS, no OKAN). "
            "Upbeat/downbeat nystagmus patients often have reduced tau_vs."
        )
    )
    K_vs: float = Field(
        default=0.1,
        description=(
            "Canal-to-velocity-storage charging gain (1/s). Healthy ~0.1. "
            "Controls how quickly canal signals charge the velocity storage integrator. "
            "Lower → weaker VOR time-constant extension; higher → faster charging but may oscillate. "
            "Reduce along with tau_vs for nodulus lesions."
        )
    )
    K_vis: float = Field(
        default=1.0,
        description=(
            "Visual (OKR) charging gain into velocity storage (1/s). Healthy ~1.0. "
            "Controls the steady-state OKN gain and OKAN buildup speed. "
            "0.0 = no optokinetic response (complete visual loss or complete darkness adaptation). "
            "Reduced in patients with poor visual acuity or cortical visual loss. "
            "OKN steady-state gain ≈ K_vis·tau_vs / (1 + K_vis·tau_vs) → ~0.95 at healthy values."
        )
    )
    g_vis: float = Field(
        default=0.3,
        description=(
            "Visual feedthrough gain (dimensionless). Healthy ~0.3. "
            "Provides a fast direct-pathway OKR onset before the integrator charges. "
            "Reduce for patients with slow OKR onset (e.g. cortical processing delays)."
        )
    )

    # ── Neural integrator ──────────────────────────────────────────────────────
    tau_i: float = Field(
        default=25.0,
        description=(
            "Neural integrator (NI) leak time constant (s). Healthy ≥ 20 s. "
            "The NI holds gaze position in the orbit — a leaky integrator drifts back to centre. "
            "Short tau_i → gaze-evoked nystagmus (GEN): eyes drift centripetally from eccentric gaze, "
            "corrective saccades beat toward the eccentric target. "
            "Cerebellar lesions (flocculus, paraflocculus, SCA6, alcohol intoxication) → 2–8 s. "
            "Congenital nystagmus or INS may also show reduced tau_i. "
            "Alexander's law: GEN worse in the direction away from a peripheral lesion."
        )
    )

    # ── Saccade generator ──────────────────────────────────────────────────────
    g_burst: float = Field(
        default=700.0,
        description=(
            "Excitatory burst neuron (EBN) ceiling — sets peak saccade velocity (deg/s). "
            "Healthy ~700 deg/s peak for large saccades (main sequence saturates ~600–700). "
            "0 = complete saccadic palsy (bilateral PPRF/IBN lesion, locked-in syndrome). "
            "200–400 = slow saccades (unilateral PPRF lesion, progressive supranuclear palsy, "
            "spinocerebellar ataxia, drug effects, fatigue). "
            "Saccade peak velocity follows: v_peak ≈ g_burst · (1 − exp(−amplitude / e_sat))."
        )
    )

    # ── Smooth pursuit ─────────────────────────────────────────────────────────
    K_pursuit: float = Field(
        default=2.0,
        description=(
            "Pursuit integrator charging gain (1/s). Healthy ~2. "
            "Controls how quickly pursuit velocity builds up after target motion onset. "
            "Rise time constant ≈ (1 + K_phasic_pursuit) / K_pursuit. "
            "Reduce for patients with impaired pursuit initiation "
            "(MT/MST lesion, cerebellar lesion, Parkinson's disease → ~0.2–0.5). "
            "Very low → pursuit never catches up; patient relies entirely on catch-up saccades."
        )
    )
    K_phasic_pursuit: float = Field(
        default=5.0,
        description=(
            "Pursuit direct feedthrough gain (dimensionless). Healthy ~5. "
            "Provides fast pursuit onset via a direct (phasic) pathway — "
            "fraction of target velocity immediately commanded = K_phasic / (1 + K_phasic). "
            "High K_phasic → fast initial eye acceleration toward target. "
            "Reduce alongside K_pursuit for pursuit deficits. "
            "Set both to ~0 to eliminate smooth pursuit entirely."
        )
    )
    tau_pursuit: float = Field(
        default=40.0,
        description=(
            "Pursuit integrator leak time constant (s). Healthy ~40 s. "
            "Determines how well pursuit is maintained during sustained tracking. "
            "Short tau_pursuit → pursuit velocity decays during sustained motion → "
            "patient falls behind target and needs repeated catch-up saccades. "
            "Reduce for patients with poor pursuit maintenance "
            "(cerebellar lesion → 5–15 s; Parkinson's → 10–20 s)."
        )
    )

    # ── Otolith / gravity estimation ───────────────────────────────────────────
    K_grav: float = Field(
        default=0.5,
        description=(
            "Otolith-driven gravity estimation gain (1/s). TC = 1/K_grav ≈ 2 s. "
            "Controls how quickly the internal gravity estimate adapts to otolith input. "
            "Relevant for tilt, off-vertical axis rotation (OVAR), and somatogravic illusions. "
            "Otolith loss (utricular damage) → reduce K_grav toward 0. "
            "Currently has limited effect unless head tilt or linear acceleration is in the stimulus."
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
            raise ValueError(f'g_burst must be ≥ 0 (got {v}); use 0 to disable saccades')
        if v > 1000:
            raise ValueError(f'g_burst={v} is physiologically unrealistic (max ~700 deg/s)')
        return v

    @field_validator('K_vs', 'K_vis', 'K_pursuit', 'K_phasic_pursuit', 'K_grav')
    @classmethod
    def _check_nonneg_gains(cls, v, info):
        if v < 0:
            raise ValueError(f'{info.field_name} must be ≥ 0 (got {v})')
        return v


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

    The LLM fills in this schema from a natural-language description.
    The runner converts it to stimulus arrays and model parameters and
    runs the simulation.
    """

    description: str = Field(
        description="One-sentence plain-English description of the scenario."
    )
    duration_s: float = Field(
        description=(
            "Total simulation duration (s). Must be between 0.5 and 120 s. "
            "Typical values: saccade 1–2 s, VOR 5–10 s, OKAN 30–60 s, pursuit 3–5 s, "
            "HIT 2–3 s, repeated HIT 10–20 s. Do not exceed 120 s."
        )
    )
    head_motion: HeadMotion
    target: Target
    visual: Visual
    patient: Patient = Field(default_factory=Patient)
    plot: PlotConfig

    @field_validator('duration_s')
    @classmethod
    def _check_duration(cls, v):
        if v < 0.5:
            raise ValueError(f'duration_s={v} is too short (minimum 0.5 s)')
        if v > 120:
            raise ValueError(
                f'duration_s={v} exceeds the 120 s limit. '
                'Use a shorter duration or break into multiple scenarios.')
        return v

    @model_validator(mode='after')
    def _cross_validate(self):
        hm = self.head_motion
        vis = self.visual

        # Head motion must fit inside the total duration
        if hm.type == 'step':
            active = hm.rotate_dur_s + hm.coast_dur_s
            if active > self.duration_s:
                # Auto-trim rather than error — runner handles this
                pass

        # scene_on_dur must not exceed total duration
        if vis.scene_on_dur_s > self.duration_s:
            raise ValueError(
                f'visual.scene_on_dur_s={vis.scene_on_dur_s} exceeds '
                f'duration_s={self.duration_s}')

        # scene_velocity without scene_present is a likely mistake
        if vis.scene_velocity_deg_s != 0.0 and not vis.scene_present:
            raise ValueError(
                'visual.scene_velocity_deg_s is non-zero but scene_present=False. '
                'Set scene_present=True for OKN/OKR stimulation.')

        # target steps must be within duration
        if self.target.type == 'steps':
            for step in self.target.steps:
                if step.t_s >= self.duration_s:
                    raise ValueError(
                        f'Target step at t_s={step.t_s} s is at or after '
                        f'duration_s={self.duration_s} s — the step would never occur.')
                if step.t_s < 0:
                    raise ValueError(f'Target step t_s={step.t_s} must be ≥ 0')

        # ramp onset must be within duration
        if self.target.type == 'ramp' and self.target.ramp_onset_s >= self.duration_s:
            raise ValueError(
                f'target.ramp_onset_s={self.target.ramp_onset_s} must be < duration_s={self.duration_s}')

        return self


# ── JSON schema export (for LLM tool call) ────────────────────────────────────

def json_schema() -> dict:
    """Return the full JSON schema for use in an LLM tool_use call."""
    return SimulationScenario.model_json_schema()
