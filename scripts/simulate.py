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
from oculomotor.scenario import SimulationScenario, json_schema
from oculomotor.runner import run_scenario


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
  Canal order in canal_gains: [L_HC, L_AC, L_PC, R_HC, R_AC, R_PC]
- **Velocity storage**: extends VOR time constant from ~5 s to ~20 s. tau_vs controls OKAN TC.
- **Neural integrator**: holds gaze position. Short tau_i → gaze-evoked nystagmus.
- **Saccade generator**: Robinson burst model. g_burst=700 normal, 0=palsy.
- **Smooth pursuit**: leaky integrator. K_pursuit controls rise speed.

## Parameter meanings (set these to simulate pathology)

canal_gains [L_HC, L_AC, L_PC, R_HC, R_AC, R_PC]: per-canal sensitivity 0–1.
    Push-pull pairs: L_HC+R_HC (yaw VOR), L_AC+R_PC (LARP), L_PC+R_AC (RALP).
    Reduce to 0 for complete paresis, 0.3–0.6 for partial.

tau_vs (s): velocity storage TC. Healthy ~20 s. Short (~1–2 s) after nodulus/uvula lesion.
    Controls OKAN duration and VOR TC extension.

K_vs (1/s): canal→VS gain. Healthy 0.1. Reduce with tau_vs for nodulus lesions.

K_vis (1/s): visual→VS gain. Healthy 1.0. 0 = no OKR. Reduce for visual loss.

tau_i (s): neural integrator TC. Healthy ≥20 s. Short (2–5 s) → gaze-evoked nystagmus.
    Cerebellar lesions (flocculus, SCA, alcohol). Alexander's law applies.

g_burst (deg/s): saccade burst ceiling. Healthy 700. 0 = complete palsy. 200–400 = slow saccades.

K_pursuit / K_phasic_pursuit: pursuit gains. Healthy 2 / 5. Reduce for pursuit deficit.
tau_pursuit (s): pursuit maintenance TC. Healthy 40 s. Short (5–15 s) = poor maintenance.

## Clinical condition → parameters

| Condition | Parameter changes |
|-----------|-------------------|
| Healthy | all defaults |
| Left vestibular neuritis / labyrinthectomy | canal_gains=[0,0,0,1,1,1] |
| Right vestibular neuritis | canal_gains=[1,1,1,0,0,0] |
| Bilateral vestibular loss (ototoxicity) | canal_gains=[0,0,0,0,0,0] |
| Left superior canal dehiscence | canal_gains=[1,0,1,1,1,1] |
| Cerebellar GEN (SCA, flocculus, alcohol) | tau_i=2.0–5.0 |
| Nodulus/uvula lesion (short VS) | tau_vs=1.5, K_vs=0.001 |
| Complete saccadic palsy (PPRF/locked-in) | g_burst=0.0 |
| Slow saccades (PSP, SCA, drugs) | g_burst=200–350 |
| Pursuit deficit (MT/MST, Parkinson's) | K_pursuit=0.3, K_phasic_pursuit=0.5, tau_pursuit=8 |
| Visual loss (no OKR) | K_vis=0.0, g_vis=0.0 |

For conditions not listed: reason from the parameter meanings above.
e.g. "progressive supranuclear palsy" → slow saccades (g_burst↓) + gaze-holding deficit (tau_i↓).
e.g. "Wernicke's encephalopathy" → pursuit deficit + gaze-holding deficit.

## Visual scene vs target — CRITICAL DISTINCTION

These two flags are independent:

  scene_present  = is the room/background lit?  (controls OKR, visual stabilisation)
  target_present = is there a discrete fixation/pursuit target?  (controls pursuit integrator)

| Paradigm | scene_present | target_present | scene_velocity_deg_s |
|----------|---------------|----------------|----------------------|
| VOR in the dark | False | False | 0 |
| VOR fixating a laser dot in darkness | False | True | 0 |
| VVOR (lit stationary room) | True | True | 0 |
| OKN drum, no fixation target | True | False | 30 (or desired) |
| Saccades in a lit room | True | True | 0 |
| Smooth pursuit (lit room) | True | True | 0 (target moves via target.type='ramp') |
| Smooth pursuit in darkness | False | True | 0 |

NEVER set scene_velocity_deg_s != 0 with scene_present=False.
For OKN: target_present MUST be False (no fixation point — full-field drum only).
For VOR-in-dark: BOTH must be False.

## Stimulus conventions

- Positive yaw = rightward, positive pitch = upward.
- For head impulse test (vHIT): type='impulse', amplitude 150–250 deg/s, rotate_dur_s=0.02.
  scene_present=False (dark room or at least no full-field OKR), target_present=True (patient fixates a dot).

## Duration guidelines

| Paradigm | Typical duration |
|----------|-----------------|
| Single saccade | 1–2 s |
| Saccade series | 3–5 s |
| VOR step | 10–20 s |
| OKAN | 40–60 s |
| Pursuit | 3–5 s |
| HIT | 2–3 s |

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

def _call_llm(description: str, model: str) -> SimulationScenario:
    """Call the Claude API to convert a description to a SimulationScenario."""
    client = anthropic.Anthropic()

    tool_schema = {
        "name": "generate_scenario",
        "description": (
            "Generate a SimulationScenario from the user's natural-language description. "
            "Fill every field. Use clinical knowledge to set patient parameters correctly."
        ),
        "input_schema": json_schema(),
    }

    print(f"Calling {model} to generate scenario...")
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=[tool_schema],
        tool_choice={"type": "tool", "name": "generate_scenario"},
        messages=[{"role": "user", "content": description}],
    )

    # Extract tool_use block
    tool_block = next(b for b in response.content if b.type == "tool_use")
    raw = tool_block.input

    return SimulationScenario.model_validate(raw)


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
