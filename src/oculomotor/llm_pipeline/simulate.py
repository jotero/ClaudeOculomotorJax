"""Natural-language → oculomotor simulation.

Usage
-----
    python -X utf8 scripts/simulate.py "healthy subject making a 20 deg saccade to the right"
    python -X utf8 scripts/simulate.py "patient with left vestibular neuritis doing a head impulse test"
    python -X utf8 scripts/simulate.py --show "OKN with full-field 30 deg/s scene motion for 20 s, then OKAN"

Options
-------
    --show          Display figure interactively (instead of saving only)
    --out PATH      Save figure to PATH  (default: outputs/<slug>.png)
    --dry-run       Print the generated scenario JSON without running the simulation
    --model NAME    Claude model to use (default: claude-opus-4-6)
    --json PATH     Load scenario from a JSON file instead of calling the LLM

The script calls the Claude API with a structured tool_use call to convert the
natural-language description into a SimulationScenario, then runs the simulation
via oculomotor.llm_pipeline.runner.run_scenario().

Requires: ANTHROPIC_API_KEY environment variable.
"""

import sys
import os
import json
import argparse
import re
import textwrap

from dotenv import load_dotenv
load_dotenv()  # searches upward from cwd — finds .env at project root

import anthropic
from oculomotor.llm_pipeline.scenario import SimulationScenario, SimulationComparison, json_schema, comparison_json_schema
from oculomotor.llm_pipeline.runner import run_scenario, run_comparison


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert computational neuroscientist assistant that converts plain-English
oculomotor experiment descriptions into simulation parameters.

## Model overview

JAX-based primate oculomotor simulation:

    Head angular vel → Semicircular canals → Velocity storage (VS) → Neural integrator (NI) → Plant → Eye position
    Scene angular vel → Visual delay (80 ms) → VS (OKR / OKAN)
    Retinal position error → Visual delay → Saccade generator (SG) → NI
    Retinal velocity error → Visual delay → Smooth pursuit → NI

## Stimulus schema — four piecewise channels

Three **motion channels** describe how the head, visual target, and scene background move through time.
One **visibility channel** says whether the scene is lit and whether a fixation target is present.
Each channel is a list of time segments concatenated end-to-end.
Total simulation duration = max of the four channel sums.

**Always begin with 0.5–1 s of stationary baseline** (head still, target straight ahead, scene lit)
before the main stimulus, unless the user explicitly specifies otherwise. This gives the model
time to settle and provides a clear pre-stimulus reference in the figure.

### head: list of segments   — head 6-DOF in world frame

  rot_yaw_vel / rot_pitch_vel — head angular velocity (deg/s) → drives semicircular canals.
  rot_profile = 'constant' | 'sinusoid' | 'impulse'
  lin_x/y/z_vel              — head linear velocity (m/s) → future otolith input.

  None for any vel/pos field = carry from previous segment's final state.
  Explicit value = reset / jump at segment boundary.

  Profile shapes:
    'constant' — pos(t)=pos₀+vel₀·t+½acc·t²,  vel(t)=vel₀+acc·t
    'sinusoid' — vel(t)=amplitude·sin(2πf·t);  amplitude=rot_*_vel; starts at zero vel
    'impulse'  — trapezoid: rises to rot_*_vel in ramp_dur_s, falls, then coasts; ends at zero vel

### target: list of segments   — 3-D world position (metres)

  Target is specified in WORLD CARTESIAN COORDINATES (metres).
  The runner projects it geometrically to retinal coordinates for you.

  lin_x_0   — lateral position (m, rightward +).  Omit = carry forward from prev segment.
  lin_y_0   — vertical position (m, upward +).    Omit = carry forward.
  lin_z_0   — depth / viewing distance (m, +fwd). Omit = carry forward (default 1 m).
  lin_x_vel — lateral velocity (m/s, rightward +). For smooth pursuit.
  lin_y_vel — vertical velocity (m/s, upward +).
  lin_z_vel — approaching (+) / receding (−) velocity (m/s).

  Conversion from degrees to metres at a given depth z:
    lin_x_0 = tan(yaw_deg  × π/180) × z     lin_x_vel = yaw_vel_degs  × π/180 × z
    lin_y_0 = tan(pitch_deg × π/180) × z    lin_y_vel = pitch_vel_degs × π/180 × z

  Quick reference at z = 1 m:
    5°  → 0.087 m    10° → 0.176 m    20° → 0.364 m    30° → 0.577 m
    10 deg/s → 0.175 m/s    20 deg/s → 0.349 m/s    30 deg/s → 0.524 m/s
  At other depths multiply by z (e.g. 20° at 0.5 m → 0.182 m).

### scene: list of segments   — visual background motion

  rot_yaw_vel — scene angular velocity (deg/s) → drives OKR / velocity storage.
  All fields default to 0 (stationary lit room).

### visual: list of segments   — scene / target visibility flags

  scene_present  = lit room?  True → OKR active. False → darkness.
  target_present = discrete foveal target?  True → pursuit + saccades active.

  | Paradigm              | scene_present | target_present | target_strobed |
  |-----------------------|:---:|:---:|:---:|
  | VOR in the dark       | False | False | False |
  | HIT (fixating dot)    | False | True  | False |
  | VVOR / saccades       | True  | True  | False |
  | OKN drum (no dot)     | True  | False | False |
  | Smooth pursuit        | True  | True  | False |
  | Pursuit in darkness   | False | True  | False |
  | Stroboscopic / flashing / intermittent target | True | True | **True** |

  **target_strobed = True** — Use whenever the user says the target is "flashing",
  "stroboscopic", "intermittent", "pulsed", or "strobed".
  Effect: position signal is present (saccades can target it) but the velocity signal
  is absent (no pursuit drive, no efference-copy contamination of the smooth-eye pathway).
  This is distinct from target_present=False (target completely gone) — the target is
  still visible as a flash, just not continuously illuminated.

## Common recipes

### Saccade 20° right at 1 m (2 s):
  head:   [{duration_s: 2}]
  target: [{duration_s: 0.3, lin_z_0: 1.0, lin_x_0: 0.0},
           {duration_s: 1.7, lin_x_0: 0.364}]   # tan(20°) × 1 m
  scene:  [{duration_s: 2}]
  visual: [{duration_s: 2}]

### Saccade between near (0.5 m) and far (3 m) target — vergence + saccade (6 s):
  # Each segment sets a new lin_z_0 (depth) and lin_x_0/lin_y_0 (lateral position).
  # The runner re-projects each segment geometrically, so angular size and disparity
  # change correctly with depth.
  head:   [{duration_s: 6}]
  target: [{duration_s: 1.0, lin_z_0: 0.5, lin_x_0: 0.0},
           {duration_s: 1.0, lin_z_0: 3.0, lin_x_0: 0.0},
           {duration_s: 1.0, lin_z_0: 0.5, lin_x_0: 0.0},
           {duration_s: 1.0, lin_z_0: 3.0, lin_x_0: 0.0},
           {duration_s: 2.0, lin_z_0: 0.5, lin_x_0: 0.0}]
  scene:  [{duration_s: 6}]
  visual: [{duration_s: 6}]
  plot: {panels: ["visual_flags", "eye_position", "vergence"]}

### Rightward vHIT (2.5 s):
  head:   [{duration_s: 2.5, rot_yaw_vel: 200, rot_profile: "impulse"}]
  target: [{duration_s: 2.5, lin_z_0: 1.0}]
  scene:  [{duration_s: 2.5}]
  visual: [{duration_s: 2.5, scene_present: false, target_present: true}]

### Leftward vHIT (negative = leftward):
  head:   [{duration_s: 2.5, rot_yaw_vel: -200, rot_profile: "impulse"}]
  ... (same target / scene / visual as rightward)

### Alternating cover test (esophoric patient, target at 2 m, 25 s):
  # Baseline 5 s → cover R eye 10 s (R drifts toward tonic_verg) → uncover 10 s (re-fusion saccade)
  head:   [{duration_s: 25}]
  target: [{duration_s: 25, lin_z_0: 2.0}]
  scene:  [{duration_s: 25}]
  visual: [{duration_s: 5},
           {duration_s: 10, target_present_L: true, target_present_R: false},
           {duration_s: 10}]
  patient: {tonic_verg: 8.0}   # elevated tonic drive = esophoric resting state
  plot: {panels: ["visual_flags", "eye_position", "vergence"]}

### VOR step in the dark (5 s rotation + 15 s coast):
  head:   [{duration_s: 5, rot_yaw_vel: 60},
           {duration_s: 15, rot_yaw_vel: 0}]
  target: [{duration_s: 20, lin_z_0: 1.0}]
  scene:  [{duration_s: 20}]
  visual: [{duration_s: 20, scene_present: false, target_present: false}]

### VVOR (head rotation in lit room, 15 s):
  head:   [{duration_s: 5, rot_yaw_vel: 60},
           {duration_s: 10, rot_yaw_vel: 0}]
  target: [{duration_s: 15, lin_z_0: 1.0}]
  scene:  [{duration_s: 15}]
  visual: [{duration_s: 15, scene_present: true, target_present: true}]

### OKN (20 s, 30 deg/s) + OKAN (40 s):
  head:   [{duration_s: 60}]
  target: [{duration_s: 60, lin_z_0: 1.0}]
  scene:  [{duration_s: 20, rot_yaw_vel: 30},
           {duration_s: 40}]
  visual: [{duration_s: 60, scene_present: true, target_present: false}]

### Smooth pursuit 20 deg/s, onset 0.3 s (5 s):
  head:   [{duration_s: 5}]
  target: [{duration_s: 0.3, lin_z_0: 1.0, lin_x_0: 0.0},
           {duration_s: 4.7, lin_x_vel: 0.349}]   # 20 deg/s × π/180 × 1 m
  scene:  [{duration_s: 5}]
  visual: [{duration_s: 5}]

### Sinusoidal VOR 0.5 Hz (10 s):
  head:   [{duration_s: 10, rot_yaw_vel: 30, rot_profile: "sinusoid", frequency_hz: 0.5}]
  target: [{duration_s: 10, lin_z_0: 1.0}]
  scene:  [{duration_s: 10}]
  visual: [{duration_s: 10, scene_present: false, target_present: false}]

### tVOR (head translates, target fixed 1 m ahead):
  head:   [{duration_s: 3, lin_x_vel: 0.1}]
  target: [{duration_s: 3, lin_x_0: 0.0, lin_z_0: 1.0}]
  scene:  [{duration_s: 3}]
  visual: [{duration_s: 3, scene_present: false, target_present: true}]

### Stroboscopic / flashing target pursuit 20 deg/s (5 s):
  # Target moves continuously but is only visible as flashes → position for saccades,
  # no velocity signal → pursuit integrator gets no drive.
  head:   [{duration_s: 5}]
  target: [{duration_s: 0.3, lin_z_0: 1.0, lin_x_0: 0.0},
           {duration_s: 4.7, lin_x_vel: 0.349}]   # 20 deg/s × π/180 × 1 m
  scene:  [{duration_s: 5}]
  visual: [{duration_s: 0.3, scene_present: true, target_present: true, target_strobed: false},
           {duration_s: 4.7, scene_present: true, target_present: true, target_strobed: true}]
  # Use panels: ['visual_flags', 'target_velocity', 'eye_position', 'eye_velocity', 'pursuit_drive', 'saccade_burst']
  # Compare with target_strobed: false to show pursuit vs. saccade-only tracking

### Gap paradigm (fixation → 200 ms gap → saccade):
  head:   [{duration_s: 3}]
  target: [{duration_s: 1.0, lin_z_0: 1.0, lin_x_0: 0.0},
           {duration_s: 0.2},
           {duration_s: 1.8, lin_x_0: 0.268}]   # tan(15°) × 1 m
  scene:  [{duration_s: 3}]
  visual: [{duration_s: 1.0, scene_present: true, target_present: true},
           {duration_s: 0.2, scene_present: true, target_present: false},
           {duration_s: 1.8, scene_present: true, target_present: true}]

## Patient parameters — healthy defaults and pathological ranges

All defaults match the healthy model. Only specify parameters that differ from healthy.

| Parameter | Healthy default | Pathological range / meaning |
|-----------|:--------------:|------------------------------|
| canal_gains [L_HC,L_AC,L_PC,R_HC,R_AC,R_PC] | [1,1,1,1,1,1] | Indices 0–2 = left ear (horiz, ant, post); 3–5 = right ear. Left neuritis=[0,0,0,1,1,1]; right=[1,1,1,0,0,0] |
| b_vs_L (deg/s) | 100 | Left VN bias + canal responsiveness. 100=healthy, 70=neuritis (intrinsic survives), 0=VN infarct. Use with canal_gains=0 for neuritis; b_vs_L=0 alone for infarct. |
| b_vs_R (deg/s) | 100 | Right VN — same scale as b_vs_L. |
| tau_vs (s) | 20.0 | 1–3 s → nodulus/uvula lesion; short TC dumps VS quickly |
| K_vs (1/s) | 0.1 | Reduce with tau_vs for nodulus lesion |
| K_vis (1/s) | 0.1 | Visual→VS gain. 0 = no OKR/OKAN |
| g_vis | 0.6 | Direct visual feedthrough (Raphan 1979). < 1 required for stability |
| tau_i (s) | 25.0 | Short (2–8 s) → centripetal drift; GEN in dark OR if K_pursuit also low |
| g_burst (deg/s) | 700.0 | 0 = saccadic palsy; 200–400 = slow saccades (PSP, SCA) |
| K_pursuit (1/s) | 4.0 | Pursuit integration gain. 0.1–0.5 = severe deficit (cerebellar, MT/MST) |
| K_phasic_pursuit | 5.0 | Pursuit direct feedthrough. Controls fast onset |
| tau_pursuit (s) | 40.0 | 5–15 s = poor pursuit maintenance |
| K_grav | 0.6 | Somatogravic gain (Laurens & Angelaki 2011 'go'). Sets tilt-percept corner ~0.095 Hz |
| K_lin | 0.1 | Linear-acceleration adaptation gain (Laurens & Angelaki 2011 'ka'). Static value; canal-gating modulates dynamically |
| tau_vs_adapt (s) | 600.0 | VS null adaptation. Reduce to 30–60 s for PAN (periodic alternating nystagmus) |
| tau_ni_adapt (s) | 20.0 | NI null adaptation. Controls rebound nystagmus amplitude after eccentric gaze |
| K_phasic_verg | 1.0 | Vergence direct phasic gain. Plant-canceling pulse |
| K_verg | 1.25 | Vergence (fast) integrator gain. Reduce for convergence insufficiency |
| tau_verg (s) | 5.0 | Vergence (fast) integrator TC. Sub-second onset, settles ~5 s |
| K_verg_tonic | 1.5 | Tonic vergence (slow adapter) gain |
| tau_verg_tonic (s) | 20.0 | Tonic vergence (slow adapter) TC. Minutes-scale dark-vergence drift |
| tonic_verg (deg) | 3.67 | Tonic (brainstem) vergence baseline. 3.67° ≈ 1 m dark vergence. Increase for esophoric patients |

### Cranial nerve and MLF lesions — use ONLY the parameters below, not VN/cerebellar params

**g_nucleus** — 12-element list [0=paralysed, 1=intact]:
  [ABN_L, ABN_R, CN4_L, CN4_R, CN3_MR_L, CN3_MR_R, CN3_SR_L, CN3_SR_R, CN3_IR_L, CN3_IR_R, CN3_IO_L, CN3_IO_R]
  ABN gain covers BOTH ipsilateral LR motoneurons AND the MLF outflow to contralateral MR
  (intermingled populations).  CN VI nucleus palsy → horizontal gaze palsy to that side.

**g_nerve** — 12-element list [0=paralysed, 1=intact]:
  Left eye indices 0–5: [LR_L, MR_L, SR_L, IR_L, SO_L, IO_L]
  Right eye indices 6–11: [LR_R, MR_R, SR_R, IR_R, SO_R, IO_R]

**g_mlf_L** — float [0–1]. 0 = left INO (left eye cannot adduct on rightward gaze; convergence intact).
**g_mlf_R** — float [0–1]. 0 = right INO (right eye cannot adduct on leftward gaze; convergence intact).

INO is NOT a vestibular and NOT a cerebellar lesion. Do NOT set b_vs_L/R, canal_gains, tau_i, or
K_pursuit for INO. The only parameter that changes is g_mlf_L or g_mlf_R.
Complete patient block for left INO: `"patient": { "g_mlf_L": 0.0 }`

The table below maps all conditions to parameters — use it:

| Clinical condition | Parameter changes |
|-------------------|-------------------|
| Healthy | all defaults |
| Left vestibular neuritis | canal_gains=[0,0,0,1,1,1], b_vs_L=70 |
| Right vestibular neuritis | canal_gains=[1,1,1,0,0,0], b_vs_R=70 |
| Left VN infarct | b_vs_L=0 |
| Bilateral vestibular loss | canal_gains=[0,0,0,0,0,0], b_vs_L=0, b_vs_R=0 |
| Nodulus / uvula lesion | tau_vs=1.5, K_vs=0.05 |
| Cerebellar GEN (dark) | tau_i=4.0 |
| Cerebellar GEN (lit room) | tau_i=4.0, K_pursuit=0.2 (BOTH required) |
| Complete saccadic palsy | g_burst=0.0 |
| Slow saccades (PSP, SCA) | g_burst=250 |
| Pursuit deficit (cerebellar) | K_pursuit=0.3, K_phasic_pursuit=1.0, tau_pursuit=8 |
| Rebound nystagmus | tau_ni_adapt=10.0 |
| PAN | tau_vs_adapt=45.0 |
| Esophoria / cover test | tonic_verg=8.0 |
| Left INO | g_mlf_L=0.0 |
| Right INO | g_mlf_R=0.0 |
| Bilateral INO | g_mlf_L=0.0, g_mlf_R=0.0 |
| CN VI nerve palsy (R) | g_nerve=[1,1,1,1,1,1,0,1,1,1,1,1] |
| CN VI nucleus palsy (R) → horizontal gaze palsy R | g_nucleus=[1,0,1,1,1,1,1,1,1,1,1,1] |
| CN III nerve palsy (R) | g_nerve=[1,1,1,1,1,1,1,0,0,0,1,0] |
| CN IV nerve palsy (R) | g_nerve=[1,1,1,1,1,1,1,1,1,1,0,1] |
| Partial CN VI palsy (recovering) | g_nerve=[1,1,1,1,1,1,0.4,1,1,1,1,1] |

For INO stimulus: rightward saccade for left INO, leftward for right INO.
Panels: ['eye_position', 'eye_velocity'].

## Panel selection — include visual_flags whenever scene or target visibility changes

`visual_flags` shows: scene on/off (green/red rows), target present (orange row), and scene velocity
(right axis). Include it in EVERY scenario where scene or target turns on/off during the trial.

VOR in dark (step + coast):    ['visual_flags', 'head_velocity', 'eye_velocity', 'eye_position', 'velocity_storage']
HIT (vHIT, impulse test):      ['visual_flags', 'head_velocity', 'eye_velocity', 'eye_position']
HIT + unilateral neuritis:     ['visual_flags', 'head_velocity', 'eye_velocity', 'eye_position', 'saccade_burst']
OKN / OKAN:                    ['visual_flags', 'eye_velocity', 'eye_position', 'velocity_storage']
Saccades:                      ['visual_flags', 'target_position', 'eye_position', 'eye_velocity', 'saccade_burst']
Smooth pursuit:                ['visual_flags', 'target_velocity', 'eye_position', 'eye_velocity', 'pursuit_drive']
Stroboscopic / flashing target:['visual_flags', 'target_velocity', 'eye_position', 'eye_velocity', 'pursuit_drive', 'saccade_burst']
GEN (gaze-evoked nystagmus):   ['visual_flags', 'eye_position', 'eye_velocity', 'neural_integrator']
Rebound nystagmus:             ['visual_flags', 'eye_position', 'eye_velocity', 'neural_integrator']
Vergence / cover test:         ['visual_flags', 'eye_position', 'vergence']
Full cascade:                  all panels

## narrative field — always fill this in

Every scenario and comparison requires a `narrative` field: 2–4 sentences written for a
clinician reader (no variable names, no code syntax). Explain:
  1. Which aspect of the physiology is altered and why (e.g. "left vestibular nerve is silent").
  2. Which model parameters were changed from healthy and what they represent.
  3. What the reader should expect to see in the figure (nystagmus direction, saccade asymmetry, etc.).

Example for left vestibular neuritis vHIT:
  "The left vestibular nerve is modelled as completely silent by setting canal_gains[0:3] = 0
   (left horizontal, anterior, and posterior canals). During a leftward head impulse no
   compensatory signal reaches the brain, producing a corrective catch-up saccade at the
   end of the movement — the hallmark of a positive vHIT on the left side. Rightward impulses
   remain intact because the right canals (indices 3–5) are unaffected."

Always call the `generate_scenario` tool with your answer.
""").strip()


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(description: str, model: str) -> SimulationScenario | SimulationComparison:
    """Call the Claude API with both tools; LLM picks single or comparison.

    Returns either a SimulationScenario or a SimulationComparison depending
    on what the description calls for.
    """
    client = anthropic.Anthropic()

    tools = [
        {
            "name": "generate_scenario",
            "description": (
                "Use for a single simulation: one patient, one stimulus. "
                "Use when the user describes one condition or paradigm without asking to compare."
            ),
            "input_schema": json_schema(),
        },
        {
            "name": "generate_comparison",
            "description": (
                "Use when the user wants to compare 2–4 conditions on the same stimulus "
                "(e.g. 'healthy vs neuritis', 'compare X and Y', 'show the difference between'). "
                "All scenarios MUST share identical stimulus and differ ONLY in patient parameters. "
                "Each scenario.description becomes its legend label — keep it short (≤5 words)."
            ),
            "input_schema": comparison_json_schema(),
        },
    ]

    print(f"Calling {model}...")
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=tools,
        tool_choice={"type": "any"},   # force tool use — never a plain-text reply
        messages=[{"role": "user", "content": description}],
    )

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        text = " ".join(b.text for b in response.content if hasattr(b, "text"))
        raise ValueError(f"LLM did not call a tool (stop_reason={response.stop_reason!r}). "
                         f"Response text: {text[:300]!r}")
    print(f"  → LLM chose: {tool_block.name}")

    if tool_block.name == "generate_comparison":
        return SimulationComparison.model_validate(tool_block.input)
    else:
        return SimulationScenario.model_validate(tool_block.input)


# Keep old names as aliases for any direct CLI use
def _call_llm(description: str, model: str) -> SimulationScenario:
    result = call_llm(description, model)
    if isinstance(result, SimulationComparison):
        raise ValueError("Description implies a comparison — use call_llm() instead.")
    return result


def _call_llm_comparison(description: str, model: str) -> SimulationComparison:
    """Call the Claude API to generate a SimulationComparison from a description."""
    client = anthropic.Anthropic()

    tool_schema = {
        "name": "generate_comparison",
        "description": (
            "Generate a SimulationComparison from the user's natural-language description. "
            "All scenarios MUST share identical stimulus (head_motion, target, visual, duration_s) "
            "and differ ONLY in patient parameters. "
            "Each scenario.description becomes its legend label — keep it short (≤5 words). "
            "Choose panels that best show the difference between conditions."
        ),
        "input_schema": comparison_json_schema(),
    }

    print(f"Calling {model} to generate comparison...")
    response = client.messages.create(
        model=model,
        max_tokens=4096,  # comparisons are larger
        system=SYSTEM_PROMPT,
        tools=[tool_schema],
        tool_choice={"type": "tool", "name": "generate_comparison"},
        messages=[{"role": "user", "content": description}],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    return SimulationComparison.model_validate(tool_block.input)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    """Convert description to a filename-safe slug."""
    slug = re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')
    return slug[:60]


def _default_output_path(description: str) -> str:
    # __file__ is oculomotor/llm_pipeline/simulate.py → go up two levels to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(project_root, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, _slugify(description) + '.png')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Convert a natural-language scenario description to an oculomotor simulation.')
    parser.add_argument('description', nargs='?', default=None,
                        help='Plain-English scenario description.')
    parser.add_argument('--show', action='store_true',
                        help='Display the figure interactively.')
    parser.add_argument('--out', default=None,
                        help='Output path for the figure (PNG/SVG). '
                             'Default: outputs/<slug>.png')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the generated scenario JSON without running.')
    parser.add_argument('--model', default='claude-opus-4-6',
                        help='Claude model to use (default: claude-opus-4-6).')
    parser.add_argument('--json', default=None,
                        help='Load scenario from a JSON file instead of calling the LLM.')
    args = parser.parse_args()

    # ── Load scenario ─────────────────────────────────────────────────────────
    if args.json:
        with open(args.json) as f:
            scenario = SimulationScenario.model_validate(json.load(f))
        description = scenario.description
    elif args.description:
        description = args.description
        scenario = _call_llm(description, args.model)
    else:
        parser.print_help()
        sys.exit(1)

    # ── Print scenario ────────────────────────────────────────────────────────
    print("\n── Generated scenario ──────────────────────────────────────────")
    print(scenario.model_dump_json(indent=2))
    print("────────────────────────────────────────────────────────────────\n")

    if args.dry_run:
        return

    # ── Run simulation ────────────────────────────────────────────────────────
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_path = args.out or _default_output_path(scenario.description)
    print(f"Running simulation: {scenario.description}")

    fig = run_scenario(scenario, output_path=out_path)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print("Done.")


if __name__ == '__main__':
    main()
