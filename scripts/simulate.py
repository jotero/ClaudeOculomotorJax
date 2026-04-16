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
via oculomotor.runner.run_scenario().

Requires: ANTHROPIC_API_KEY environment variable.
"""

import sys
import os
import json
import argparse
import re
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import anthropic
from oculomotor.scenario import SimulationScenario, SimulationComparison, json_schema, comparison_json_schema
from oculomotor.runner import run_scenario, run_comparison


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

The runner converts the 3-D target world position to retinal angles via:
    p_target = (target_pos − head_pos) / depth     [tan(yaw), tan(pitch), 1]
    v_target = d/dt(arctan(rel/depth))              [deg/s; includes head-translation effect]

## Stimulus schema — four piecewise channels

Each channel is a list of **BodySegment** (or VisualFlagsSegment) objects concatenated in time.
Total simulation duration = max of the four channel sums.

### head: list[BodySegment]   — head 6-DOF in world frame

  rot_yaw_vel / rot_pitch_vel — head angular velocity (deg/s) → drives semicircular canals.
  rot_profile = 'constant' | 'sinusoid' | 'impulse'
  lin_x/y/z_vel              — head linear velocity (m/s) → future otolith input.

  None for any vel/pos field = carry from previous segment's final state.
  Explicit value = reset / jump at segment boundary.

  Profile shapes:
    'constant' — pos(t)=pos₀+vel₀·t+½acc·t²,  vel(t)=vel₀+acc·t
    'sinusoid' — vel(t)=amplitude·sin(2πf·t);  amplitude=rot_*_vel; starts at zero vel
    'impulse'  — trapezoid: rises to rot_*_vel in ramp_dur_s, falls, then coasts; ends at zero vel

### target: list[BodySegment]  — 3-D world position (metres)

  Target is specified in WORLD CARTESIAN COORDINATES (metres), not angles.
  The runner projects it geometrically to retinal coordinates for you.

  lin_x_0    — lateral position (m, rightward +).  Default: 0 m (straight ahead).
  lin_y_0    — vertical position (m, upward +).    Default: 0 m.
  lin_z_0    — depth / viewing distance (m, +fwd). Default: 1 m.
  lin_x_vel  — lateral velocity (m/s).  Pursuit drive ≈ lin_x_vel / lin_z_0 (deg/s).
  lin_z_vel  — approaching/receding target velocity (m/s).

  ANGULAR SHORTHAND (auto-converted at z=lin_z_0 or 1 m):
    rot_yaw_0    = target angle (deg)  →  lin_x_0 = tan(rot_yaw_0°) × z
    rot_yaw_vel  = angular vel (deg/s) →  lin_x_vel ≈ rot_yaw_vel × π/180 × z
    (same for pitch / y)

  Conversion table at z=1 m:
    10°  → lin_x_0 ≈ 0.176 m       10 deg/s  → lin_x_vel ≈ 0.175 m/s
    20°  → lin_x_0 ≈ 0.364 m       20 deg/s  → lin_x_vel ≈ 0.349 m/s
    30°  → lin_x_0 ≈ 0.577 m       30 deg/s  → lin_x_vel ≈ 0.524 m/s
    None (omit) = position/velocity continues from previous segment

### scene: list[BodySegment]  — visual background motion

  rot_yaw_vel — scene angular velocity (deg/s) → drives OKR / velocity storage.
  All fields default to 0 (stationary lit room).

### visual: list[VisualFlagsSegment]  — scene / target visibility

  scene_present  = lit room?  True → OKR active. False → darkness.
  target_present = discrete foveal target?  True → pursuit + saccades active.

  | Paradigm              | scene_present | target_present |
  |-----------------------|:---:|:---:|
  | VOR in the dark       | False | False |
  | HIT (fixating dot)    | False | True  |
  | VVOR / saccades       | True  | True  |
  | OKN drum (no dot)     | True  | False |
  | Smooth pursuit        | True  | True  |
  | Pursuit in darkness   | False | True  |

## Common recipes

### Saccade 20° right (2 s):
  head:   [{duration_s: 2}]
  target: [{duration_s: 0.3, lin_z_0: 1.0},
           {duration_s: 1.7, rot_yaw_0: 20}]
  scene:  [{duration_s: 2}]
  visual: [{duration_s: 2}]

### Rightward vHIT (2.5 s):
  head:   [{duration_s: 2.5, rot_yaw_vel: 200, rot_profile: "impulse"}]
  target: [{duration_s: 2.5, lin_z_0: 1.0}]
  scene:  [{duration_s: 2.5}]
  visual: [{duration_s: 2.5, scene_present: false, target_present: true}]

### Leftward vHIT (negative = leftward):
  head:   [{duration_s: 2.5, rot_yaw_vel: -200, rot_profile: "impulse"}]
  ... (same target / scene / visual as rightward)

### Alternating left + right HITs (6 s):
  head:   [{duration_s: 3, rot_yaw_vel: 200, rot_profile: "impulse"},
           {duration_s: 3, rot_yaw_vel: -200, rot_profile: "impulse"}]
  target: [{duration_s: 6, lin_z_0: 1.0}]
  scene:  [{duration_s: 6}]
  visual: [{duration_s: 6, scene_present: false, target_present: true}]

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
  target: [{duration_s: 0.3, lin_z_0: 1.0},
           {duration_s: 4.7, rot_yaw_vel: 20}]
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

### Gap paradigm (fixation → 200 ms gap → saccade):
  head:   [{duration_s: 3}]
  target: [{duration_s: 1.0, lin_z_0: 1.0, lin_x_0: 0.0},
           {duration_s: 0.2},
           {duration_s: 1.8, rot_yaw_0: 15}]
  scene:  [{duration_s: 3}]
  visual: [{duration_s: 1.0, scene_present: true, target_present: true},
           {duration_s: 0.2, scene_present: true, target_present: false},
           {duration_s: 1.8, scene_present: true, target_present: true}]

## Patient parameters (overrides from healthy defaults)

canal_gains [L_HC, L_AC, L_PC, R_HC, R_AC, R_PC]: per-canal sensitivity 0–1.
tau_vs (s):  velocity storage TC. Healthy ~20 s. Nodulus lesion → 1–3 s.
K_vs (1/s):  canal→VS gain. Healthy 0.1.
K_vis (1/s): visual→VS gain. Healthy 1.0. 0 = no OKR.
tau_i (s):   neural integrator TC. Healthy ≥20 s. Short (2–5 s) → gaze-evoked nystagmus.
g_burst:     saccade burst ceiling (deg/s). Healthy 700. 0 = complete palsy.
K_pursuit / K_phasic_pursuit: pursuit gains. Healthy 2 / 5.
tau_pursuit (s): pursuit maintenance TC. Healthy 40 s. Short (5–15 s) = poor maintenance.

| Condition | Parameter changes |
|-----------|-------------------|
| Healthy | all defaults |
| Left vestibular neuritis | canal_gains=[0,0,0,1,1,1] |
| Right vestibular neuritis | canal_gains=[1,1,1,0,0,0] |
| Bilateral vestibular loss | canal_gains=[0,0,0,0,0,0] |
| Cerebellar GEN (SCA, alcohol) | tau_i=2.0–5.0 |
| Nodulus / uvula lesion | tau_vs=1.5, K_vs=0.001 |
| Complete saccadic palsy | g_burst=0.0 |
| Slow saccades (PSP, SCA) | g_burst=200–350 |
| Pursuit deficit (MT/MST, Parkinson's) | K_pursuit=0.3, K_phasic_pursuit=0.5, tau_pursuit=8 |

## Panel selection

VOR / HIT:       ['head_velocity', 'eye_velocity', 'eye_position']
HIT + neuritis:  ['head_velocity', 'eye_velocity', 'eye_position', 'saccade_burst']
OKN / OKAN:      ['head_velocity', 'eye_velocity', 'eye_position', 'velocity_storage']
Saccades:        ['eye_position', 'eye_velocity', 'saccade_burst', 'refractory']
Pursuit:         ['eye_position', 'eye_velocity', 'pursuit_drive']
Full cascade:    all panels

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
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": description}],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
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
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
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
