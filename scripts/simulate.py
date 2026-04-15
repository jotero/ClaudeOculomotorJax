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

This is a JAX-based simulation of the primate oculomotor system:

    Head velocity → Semicircular canals → Velocity storage (VS) → Neural integrator (NI) → Plant → Eye position
    Retinal slip  → Visual delay (80 ms) → VS (OKR/OKAN)
    Retinal error → Visual delay (80 ms) → Saccade generator (SG) → NI
    Target vel.   → Visual delay (80 ms) → Smooth pursuit → NI

Key subsystems:
- **Canals**: 6 semicircular canals (L/R × HC/AC/PC). canal_gains=[0..1] per canal.
  Canal order: [L_HC, L_AC, L_PC, R_HC, R_AC, R_PC]
- **Velocity storage**: extends VOR TC from ~5 s to ~20 s. tau_vs controls OKAN TC.
- **Neural integrator**: holds gaze position. Short tau_i → gaze-evoked nystagmus.
- **Saccade generator**: Robinson burst model. g_burst=700 normal, 0=palsy.
- **Smooth pursuit**: leaky integrator. K_pursuit controls rise speed.

## Stimulus schema — piecewise segments

Each stimulus channel is a list of segments concatenated in time order.
Total duration = max(sum of head durations, sum of target durations, sum of visual durations).

### head: list[HeadSegment]

Each segment: {duration_s, velocity_deg_s, profile, frequency_hz, ramp_dur_s, axis}

  profile = 'constant'  → constant velocity for duration_s (velocity_deg_s=0 = stationary)
  profile = 'sinusoid'  → v(t) = velocity_deg_s * sin(2π * frequency_hz * t)
  profile = 'impulse'   → brief pulse: rises to velocity_deg_s in ramp_dur_s (default 0.02 s),
                          falls back, then coasts; total = duration_s

  Sign convention: positive = rightward / upward.
  Typical impulse amplitude: 150–300 deg/s; duration_s ≥ 2*ramp_dur_s + 0.1 s for coast.

### target: list[TargetSegment]

Each segment: {duration_s, position_yaw_deg, position_pitch_deg, velocity_yaw_deg_s, velocity_pitch_deg_s}

  position_yaw_deg   — if set: target JUMPS to this absolute position at segment start (saccade target).
                       if null/omitted: position continues smoothly from previous segment.
  velocity_yaw_deg_s — constant target velocity during this segment (deg/s). Integrates position.

  Use a position jump + zero velocity for saccade targets.
  Use null position + non-zero velocity for pursuit ramps.

### visual: list[VisualSegment]

Each segment: {duration_s, scene_present, target_present, scene_velocity_deg_s, scene_axis}

  scene_present  = is the room lit?  True → OKR active. False → complete darkness.
  target_present = is there a discrete foveal target?  True → pursuit + saccades active.
                   False → OKN drum only / VOR in dark.
  scene_velocity_deg_s = full-field scene velocity for OKN. Requires scene_present=True.

  CRITICAL: scene_velocity_deg_s ≠ 0 requires scene_present=True.
  CRITICAL: OKN → target_present=False (no fixation dot, just the drum).
  CRITICAL: VOR-in-dark → both scene_present=False AND target_present=False.

| Paradigm | scene_present | target_present | scene_velocity_deg_s |
|----------|:---:|:---:|:---:|
| VOR in the dark | False | False | 0 |
| VOR fixating laser dot (dark) | False | True | 0 |
| VVOR (lit stationary room) | True | True | 0 |
| OKN drum, no fixation dot | True | False | 30 (or desired) |
| Saccades in lit room | True | True | 0 |
| Smooth pursuit (lit room) | True | True | 0 |
| Smooth pursuit in darkness | False | True | 0 |

## Common stimulus recipes (copy and adapt)

### Single saccade 20° right (2 s total):
  head:   [{duration_s: 2, velocity_deg_s: 0, profile: "constant"}]
  target: [{duration_s: 0.3}, {duration_s: 1.7, position_yaw_deg: 20}]
  visual: [{duration_s: 2, scene_present: true, target_present: true}]

### Rightward head impulse test / vHIT (2.5 s):
  head:   [{duration_s: 2.5, velocity_deg_s: 200, profile: "impulse", axis: "yaw"}]
  target: [{duration_s: 2.5, position_yaw_deg: 0}]
  visual: [{duration_s: 2.5, scene_present: false, target_present: true}]

### Leftward vHIT (negative amplitude):
  head:   [{duration_s: 2.5, velocity_deg_s: -200, profile: "impulse", axis: "yaw"}]
  target: [{duration_s: 2.5, position_yaw_deg: 0}]
  visual: [{duration_s: 2.5, scene_present: false, target_present: true}]

### Alternating left+right HITs (6 s):
  head:   [{duration_s: 3, velocity_deg_s: 200, profile: "impulse"},
           {duration_s: 3, velocity_deg_s: -200, profile: "impulse"}]
  target: [{duration_s: 6, position_yaw_deg: 0}]
  visual: [{duration_s: 6, scene_present: false, target_present: true}]

### VOR step in the dark (20 s):
  head:   [{duration_s: 5, velocity_deg_s: 60, profile: "constant"},
           {duration_s: 15, velocity_deg_s: 0}]
  target: [{duration_s: 20}]
  visual: [{duration_s: 20, scene_present: false, target_present: false}]

### VVOR — VOR with lit stationary room (15 s):
  head:   [{duration_s: 5, velocity_deg_s: 60, profile: "constant"},
           {duration_s: 10, velocity_deg_s: 0}]
  target: [{duration_s: 15}]
  visual: [{duration_s: 15, scene_present: true, target_present: true}]

### OKN (20 s) + OKAN (40 s):
  head:   [{duration_s: 60, velocity_deg_s: 0}]
  target: [{duration_s: 60}]
  visual: [{duration_s: 20, scene_present: true, target_present: false, scene_velocity_deg_s: 30},
           {duration_s: 40, scene_present: true, target_present: false}]

### Smooth pursuit 20 deg/s, onset at 0.3 s (5 s total):
  head:   [{duration_s: 5, velocity_deg_s: 0}]
  target: [{duration_s: 0.3}, {duration_s: 4.7, velocity_yaw_deg_s: 20}]
  visual: [{duration_s: 5, scene_present: true, target_present: true}]

### Gap paradigm (fixation → gap → saccade):
  head:   [{duration_s: 3, velocity_deg_s: 0}]
  target: [{duration_s: 1.0, position_yaw_deg: 0},
           {duration_s: 0.2, position_yaw_deg: null},          ← gap: no target jump, same pos
           {duration_s: 1.8, position_yaw_deg: 15}]
  visual: [{duration_s: 1.0, scene_present: true, target_present: true},
           {duration_s: 0.2, scene_present: true, target_present: false},  ← gap period
           {duration_s: 1.8, scene_present: true, target_present: true}]

### Saccade series — 5 targets 10° apart (5 s):
  head:   [{duration_s: 5, velocity_deg_s: 0}]
  target: [{duration_s: 0.5, position_yaw_deg: 0},
           {duration_s: 0.8, position_yaw_deg: 10},
           {duration_s: 0.8, position_yaw_deg: 20},
           {duration_s: 0.8, position_yaw_deg: 30},
           {duration_s: 0.8, position_yaw_deg: 20},
           {duration_s: 0.3, position_yaw_deg: 0}]
  visual: [{duration_s: 5, scene_present: true, target_present: true}]

### Sinusoidal head rotation 0.5 Hz (10 s):
  head:   [{duration_s: 10, velocity_deg_s: 30, profile: "sinusoid", frequency_hz: 0.5}]
  target: [{duration_s: 10}]
  visual: [{duration_s: 10, scene_present: false, target_present: false}]

## Patient parameters (overrides from healthy defaults)

canal_gains [L_HC, L_AC, L_PC, R_HC, R_AC, R_PC]: per-canal sensitivity 0–1.
    Push-pull pairs: L_HC+R_HC (yaw), L_AC+R_PC (LARP), L_PC+R_AC (RALP).

tau_vs (s): velocity storage TC. Healthy ~20 s. Short (~1–2 s) after nodulus/uvula lesion.
K_vs (1/s): canal→VS gain. Healthy 0.1. Reduce with tau_vs for nodulus lesions.
K_vis (1/s): visual→VS gain. Healthy 1.0. 0 = no OKR.
tau_i (s): neural integrator TC. Healthy ≥20 s. Short (2–5 s) → gaze-evoked nystagmus.
g_burst (deg/s): saccade burst ceiling. Healthy 700. 0 = complete palsy.
K_pursuit / K_phasic_pursuit: pursuit gains. Healthy 2 / 5. Reduce for pursuit deficit.
tau_pursuit (s): pursuit maintenance TC. Healthy 40 s. Short (5–15 s) = poor maintenance.

## Clinical condition → parameters

| Condition | Parameter changes |
|-----------|-------------------|
| Healthy | all defaults |
| Left vestibular neuritis / labyrinthectomy | canal_gains=[0,0,0,1,1,1] |
| Right vestibular neuritis | canal_gains=[1,1,1,0,0,0] |
| Bilateral vestibular loss (ototoxicity) | canal_gains=[0,0,0,0,0,0] |
| Cerebellar GEN (SCA, flocculus, alcohol) | tau_i=2.0–5.0 |
| Nodulus/uvula lesion (short VS) | tau_vs=1.5, K_vs=0.001 |
| Complete saccadic palsy (PPRF/locked-in) | g_burst=0.0 |
| Slow saccades (PSP, SCA, drugs) | g_burst=200–350 |
| Pursuit deficit (MT/MST, Parkinson's) | K_pursuit=0.3, K_phasic_pursuit=0.5, tau_pursuit=8 |
| Visual loss (no OKR) | K_vis=0.0, g_vis=0.0 |

For conditions not listed: reason from the parameter meanings above.

## Panel selection guidelines

Choose the minimal set of panels that tells the story:
- VOR: ['head_velocity', 'eye_velocity', 'eye_position']
- VOR + velocity storage: ['head_velocity', 'eye_velocity', 'velocity_storage', 'eye_position']
- OKN / OKAN: ['head_velocity', 'eye_velocity', 'eye_position', 'velocity_storage']
- Saccades: ['eye_position', 'eye_velocity', 'saccade_burst', 'refractory']
- Pursuit: ['eye_position', 'eye_velocity', 'pursuit_drive']
- HIT (healthy): ['head_velocity', 'eye_velocity', 'eye_position']
- HIT (vestibular neuritis): ['head_velocity', 'eye_velocity', 'eye_position', 'saccade_burst']
- Full cascade: all panels

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
