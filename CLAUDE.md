# ClaudeOculomotorJax — Project Context for Claude

## Running scripts

Always use `-X utf8` to avoid Windows cp1252 encoding errors (Greek letters in print statements crash otherwise):

```bash
"d:/OneDrive/UC Berkeley/OMlab - JOM/Code/ClaudeOculomotorJax/.venv/Scripts/python.exe" -X utf8 scripts/demo_vor.py
```

Or from PowerShell:

```powershell
& "d:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\.venv\Scripts\python.exe" -X utf8 scripts\demo_vor.py
```

## Stimulus conventions

Keep all pathways active by default. Only disable a pathway (e.g. `g_burst=0`, `K_vis=0`, `scene_present=0`) when the demo is explicitly testing what happens *without* that pathway — and add a comment explaining why.

In particular:
- **Saccade demos**: keep `scene_present=1` (the animal sees the target in a lit room). Do not zero out OKR gains.
- **VOR demos**: keep saccades on (`g_burst=700`) unless the demo is specifically about the canal→VS cascade without fast phases.
- **OKR demos**: keep saccades on so the nystagmus sawtooth is visible.

When a panel looks wrong, diagnose via simulation before disabling pathways. Autoscale artifacts (matplotlib scaling noise to a salient-looking signal) are a common false alarm — enforce a minimum y-axis range for velocity panels.

## What this is

A JAX-based simulation of the primate oculomotor system. The goal is a differentiable, biophysically grounded model of how the brain controls eye movements — suitable for fitting to experimental data.

## Architecture

Signal flow: `Head velocity → Canal Array → Velocity Storage → Neural Integrator → Plant → Eye position`

Visual loop: `Retinal slip → Visual delay cascade → Velocity Storage (OKR/OKAN)`

Saccade loop: `Retinal position error → Visual delay → Saccade Generator (Robinson local-feedback) → NI + Plant`

Pursuit loop: `Retinal velocity error → Visual delay → Pursuit integrator (Smith predictor) → NI + Plant`

### Folder structure (`oculomotor/`)

```
oculomotor/
├── __init__.py                        __version__ from git describe --tags --always --dirty
├── models/
│   ├── sensory_models/
│   │   ├── canal.py                   Canal array SSM (Steinhausen, 6 canals, 12 states)
│   │   ├── otolith.py                 Otolith SSM (bilateral LP adaptation, 6 states)
│   │   ├── retina.py                  Retinal geometry + visual delay cascade (400 states)
│   │   │                              Delays: slip(120) | pos_vis(120) | vel(120) | gate_vf(40)
│   │   └── sensory_model.py           Connector: canal + otolith + retina → SensoryOutput (418 states)
│   ├── brain_models/
│   │   ├── velocity_storage.py        VS leaky integrator; PINV_SENS mixing + OKR (3 states)
│   │   ├── neural_integrator.py       NI leaky integrator (Robinson 1975) (3 states)
│   │   ├── saccade_generator.py       Robinson local-feedback burst + OPN gate (9 states)
│   │   │                              Target selection (clip + centering) is internal to SG
│   │   ├── efference_copy.py          Motor command delay cascade for slip cancellation (120 states)
│   │   ├── gravity_estimator.py       Cross-product gravity transport + otolith correction (3 states)
│   │   ├── pursuit.py                 Smooth pursuit leaky integrator + Smith predictor (3 states)
│   │   └── brain_model.py             Connector: VS + NI + SG + EC + GE + pursuit → motor_cmd (141 states)
│   ├── plant_models/
│   │   ├── plant_model_first_order.py First-order plant (Robinson 1964) (3 states)
│   │   └── readout.py                 Eye position readout utilities + rotation_matrix()
│   └── llm_pipeline/                  Natural-language → simulation pipeline
│       ├── scenario.py                Pydantic schema (SimulationScenario, Patient, …)
│       ├── runner.py                  Stimulus builder + simulator wiring + figure generator
│       └── simulate.py                CLI entry point + Claude API call (call_llm / main)
└── sim/
    ├── simulator.py                   ODE wiring + simulate() entry point
    └── stimuli.py                     Centralized stimulus generators (build_body_arrays, …)
```

### scripts/

```
scripts/
├── simulate.py          Thin shim → oculomotor.llm_pipeline.simulate.main()
├── server.py            FastAPI web server — LLM pipeline + logging + feedback + data download
├── demo_vor.py          VOR / VVOR / OKN diagnostic demos
├── demo_saccade.py      Saccade main sequence + cascade demos
├── demo_pursuit.py      Smooth pursuit + catch-up saccade demos
└── demo_fixation.py     Fixational eye movements — noise source comparison
```

### State structure (562 total)

The ODE state is a `SimState` NamedTuple with three groups:

```
SimState(
    sensory (418):  [x_c (12) | x_oto (6) | x_vis (400)]
                     canal      otolith     visual delay cascade
                     _IDX_C     _IDX_OTO    _IDX_VIS

    brain   (141):  [x_vs (3) | x_ni (3) | x_sg (9) | x_ec (120) | x_grav (3) | x_pursuit (3)]
                     vel-store   NI          sacc-gen   EC delay     gravity est   pursuit mem
                     _IDX_VS     _IDX_NI     _IDX_SG    _IDX_EC      _IDX_GRAV     _IDX_PURSUIT

    plant     (3):  x_p — eye rotation vector (deg); directly observable
)
```

`x_sg` sub-layout: `[x_copy(3) | z_ref(1) | e_held(3) | z_sac(1) | z_acc(1)]`

`x_vis` sub-layout: `[x_slip(120) | x_pos_vis(120) | x_vel(120) | x_gate(40)]`

### Params structure

Parameters are nested NamedTuples — not dicts. Access via attribute path:

```python
class SensoryParams(NamedTuple):
    tau_c, tau_s, canal_gains, tau_oto,
    tau_vis, visual_field_limit, k_visual_field,
    sigma_canal, sigma_slip, sigma_pos, sigma_vel, tau_pos_drift

class PlantParams(NamedTuple):
    tau_p

class BrainParams(NamedTuple):
    tau_vs, K_vs, K_vis, g_vis,
    tau_i, tau_p, tau_vis,
    g_burst, e_sat_sac, k_sac, threshold_sac, ...
    K_pursuit, K_phasic_pursuit, tau_pursuit,
    K_grav, K_gd, g_ocr, orbital_limit, alpha_reset

class Params(NamedTuple):
    sensory: SensoryParams
    plant:   PlantParams
    brain:   BrainParams
```

Use `with_sensory(params, sigma_canal=2.0)` / `with_brain(params, tau_vs=15.0)` / `with_plant(params, tau_p=0.2)` to create modified copies.

### Solver

`diffrax.Heun()` fixed step, `dt = 0.001 s`. Must satisfy `dt < 2 * tau_stage_vis = 0.004 s`.

### Sensory noise

Four independent noise sources, all zero by default. Pre-generated as arrays before `diffeqsolve` and passed as `LinearInterpolation` inputs — ODE remains pure and differentiable.

```python
params = with_sensory(PARAMS_DEFAULT,
    sigma_canal    = 2.0,   # canal afferent noise (deg/s); filtered heavily by VS/NI/plant
    sigma_slip     = 0.0,   # retinal slip noise (deg/s); drives VS/OKR
    sigma_pos      = 0.3,   # retinal position drift (deg); OU process → microsaccades
    tau_pos_drift  = 0.3,   # OU time constant (s); controls inter-microsaccade interval
    sigma_vel      = 5.0,   # retinal velocity noise (deg/s); drives pursuit integrator
)
states = simulate(params, t, ..., key=jax.random.PRNGKey(42))
```

`sigma_pos` uses an Ornstein-Uhlenbeck process (not white noise) so drift accumulates slowly,
crosses the SG threshold occasionally, and triggers sparse corrective microsaccades.
White noise on `pos_delayed` would fire the SG continuously.

### Versioning

`oculomotor.__version__` is derived from `git describe --tags --always --dirty` at import time.
No manual version bumping required — tag a release with `git tag v1.0` and it appears automatically.
The version string is logged with every server simulation call.

### What "correct behavior" looks like

Each behavior has a corresponding demo script and output figure.

1. **VOR in the dark** — eye velocity ≈ −head velocity; gain ~0.9–1.0. Canal adaptation TC (~5 s) causes the VOR to decay during sustained rotation; velocity storage extends the effective TC to ~15–20 s.
   - Demo: `scripts/demo_vor.py` → `outputs/vor_dark.png`

2. **Velocity storage / TC extension** — during constant-velocity rotation in the dark, eye velocity decays with TC ~15–20 s (not the canal TC of ~5 s). VVOR: in a stationary lit world, OKR corrects VOR slip as the canal adapts — gaze stays stable throughout.
   - Demo: `scripts/demo_vor.py` → `outputs/vvor.png`

3. **OKN + OKAN** — during full-field visual motion, steady-state OKN gain ≈ 1. After scene off, OKAN persists with TC ~20 s (`tau_vs`). With saccades on, eye shows sawtooth nystagmus.
   - Demo: `scripts/demo_vor.py` → `outputs/okr.png`

4. **Saccades — main sequence + refractory period** — peak velocity follows `v_peak ≈ 700·(1−exp(−A/7))`, saturating ~600–700 deg/s. Robust intersaccadic interval (~150–200 ms). Oblique saccades straight with synchronized components.
   - Demo: `scripts/demo_saccade.py` → `outputs/saccade_summary.png`

5. **Smooth pursuit** — foveal target tracking via MT/MST velocity pathway. Pursuit integrator + Smith predictor (efference copy cancels saccadic contamination). Catch-up saccades fire when position error exceeds threshold during ramp pursuit.
   - Demo: `scripts/demo_pursuit.py` → `outputs/smooth_pursuit.png`

6. **Saccades during head movement** — corrective saccades fire periodically as VOR slip accumulates; staircase toward target.
   - Demo: `scripts/demo_saccade.py` → `outputs/vor_saccade_cascade.png`

7. **Efference copy** — burst commands must not contaminate VS/OKR. VS state identical with/without saccades during OKN.
   - Demo: `scripts/demo_efference.py` → `outputs/efference_*.png`

8. **Fixational eye movements** — canal noise filtered by VS/NI/plant; retinal position OU drift produces sparse corrective microsaccades; retinal velocity noise drives pursuit-like slow drift.
   - Demo: `scripts/demo_fixation.py` → `outputs/fixation.png`

## Current status (2026-04-16)

- **Working well**: VOR, VVOR, OKN/OKAN, saccades (main sequence, refractory, oblique), smooth pursuit (velocity-driven), efference copy slip cancellation, otolith LP adaptation, sensory noise system, fixational eye movements.
- **Pending improvement**: Pursuit position sensitivity (`K_pursuit_pos` — see future work).
- **Not yet debugged**: Gravity estimator (`gravity_estimator.py`) — implemented but behavior not verified. Will be debugged together with vergence, since T-VOR is strongly vergence-dependent.
- **Next focus**: Binocularity and vergence — see future work section.

## Not yet implemented / pending (future work)

- **Pursuit position sensitivity** — pursuit should be weakly driven by `pos_delayed` (retinal position error) in addition to `vel_delayed`, to correct steady-state position offsets. Add `K_pursuit_pos` gain term in `pursuit.step()`: `e_combined += K_pursuit_pos * pos_delayed`.

- **Binocularity / vergence** — currently monocular. Next major development area. Requires separate L/R eye plants, vergence angle state, and disparity-driven vergence controller.

- **Gravity estimator + T-VOR** — `gravity_estimator.py` is implemented but not validated. Debugging planned alongside vergence since translational VOR requires vergence angle to compute the correct compensatory eye movement (near targets need larger compensation than far targets).

- **Listing's law** — torsional constraints not enforced.

- **Multiple plant models** — see design note below.

- **Multiple brain models** — see design note below.

### Design note: swappable plants and brain models

The simulator is designed so that `plant_models/` and `brain_models/` can be swapped without touching sensory machinery. The integration point is the **motor command interface** between brain and plant.

**Motor command format (current):**

The brain outputs `motor_cmd: (3,)` — the Robinson pulse-step sum in rotation-vector space (yaw/pitch/roll, deg/s equivalent). This is the NI output `x_ni + tau_p * u_vel`, which combines the tonic position hold and phasic burst feedthrough into a single vector. The plant does not need to know the decomposition.

Units: motoneuron firing rate equivalent. The plant converts to forces/torques internally.

**Plant interface contract:**

```python
def step(x_p, motor_cmd, plant_params) -> (dx_p, w_p, w_eye):
    # x_p:      (3,)  plant state (eye rotation vector, deg)
    # motor_cmd:(3,)  pulse-step motor command from NI
    # w_eye:    (3,)  instantaneous eye velocity → retina / feedback (algebraic)
```

Any plant implementing this contract (first-order, second-order, MJX-backed) is a drop-in replacement in `simulator.py`.

**Brain model interface contract:**

```python
def step(x_brain, sensory_out: SensoryOutput, brain_params) -> (dx_brain, motor_cmd):
    # sensory_out fields: canal(6), slip_delayed(3), pos_delayed(3), gate_vf(scalar),
    #                     vel_delayed(3), f_otolith(3), scene_present, target_present
```

Different brain architectures (Raphan-Cohen, Kalman, RL policy) swap in here. The sensory model and plant remain unchanged.

## SSM module convention

Every subsystem is a **state-space model (SSM)** with a uniform interface. This is the contract — new modules must follow it.

### Equations

```
dx/dt = A(θ) @ x  +  B(θ) @ u      # state derivative
y     = C     @ x  +  D(θ) @ u      # output (feedthrough allowed)
```

### Module structure

Each module exposes:

| Symbol | Type | When θ-dependent |
|--------|------|-----------------|
| `N_STATES`, `N_INPUTS`, `N_OUTPUTS` | `int` constants | never |
| `step(x, u, theta)` → `(dx, y)` | pure function | — |
| Module-level constants (e.g. `C_slip`, `PINV_SENS`) | only when used externally | — |

### `step()` contract

```python
def step(x, u, theta):
    A = ...   # build from theta inside step
    B = ...
    dx = A @ x + B @ u
    y  = C @ x + D @ u   # C, D omitted if identity or zero
    return dx, y
```

- **A, B, C, D are local variables inside `step()`** — not separate module-level functions.
- Identity matrices (B=I, C=I, D=I) are omitted — just use `x` directly and note `# B = I`.
- **Pure function** — no side effects, no global state. Compatible with `jax.jit` and `jax.grad`.
- Returns `(dx, y)` always — ODE integrator uses `dx`; simulator uses `y` to wire modules.
- Input/output shapes and units must be documented in the module docstring.
- **`theta` is a `Params` NamedTuple** — access via `theta.sensory.tau_c`, `theta.brain.tau_vs`, etc. Never treat it as a dict.
- Module-level constants are kept only when used by external code (e.g. `retina.C_slip`, `retina.C_gate`, `canal.PINV_SENS`).

### Nonlinear extensions

Some modules have nonlinearities that wrap the linear ABCD core:

- **Canal** (`canal.py`): `nonlinearity(x_c, gains)` applies smooth push-pull rectification to the `x2` (inertia state) to get afferent firing rates. The linear `A @ x + B @ u` drives the state derivative; only the output is nonlinear. Re-exported as `canal_nonlinearity` from `sensory_model.py`.
- **Saccade generator**: gates (`gate_err`, `gate_res`, `gate_dir`) and adaptive reset TC layered on top of linear SSM core. Target selection (orbital clip + centering saccade) is handled internally using `x_ni` as a proxy for eye position and `gate_vf` to detect out-of-field targets.
- **Visual delay** (`retina.py`): fixed companion-form `A` (40-stage cascade) with module-level `C_slip` / `C_pos` / `C_vel` / `C_gate` readout matrices — kept at module level because external code reads them directly. `gate_vf` is computed in `retinal_signals()` (retinal geometry, not a brain decision) and delayed by its own 40-stage scalar cascade so the SG can distinguish fixation from out-of-field.

### Connector modules

`sensory_model.py` and `brain_model.py` are **connector modules** — they import their sub-SSMs, own the combined state layout and index constants, and expose a single `step()` + output-read interface. They do not implement physics themselves.

### Example: Neural Integrator (simplest case)

```python
N_STATES = N_INPUTS = N_OUTPUTS = 3

def step(x_ni, u_vel, theta):
    A = (-1/theta.brain.tau_i) * jnp.eye(3)
    D = theta.brain.tau_p * jnp.eye(3)
    # B = C = I (identity — omitted)
    dx  = A @ x_ni + u_vel
    u_p = x_ni + D @ u_vel
    return dx, u_p
```

### Wiring in the simulator

`ODE_ocular_motor` in `sim/simulator.py` calls each module's `step()` in signal-flow order, passing outputs of one as inputs to the next. The global state is a `SimState` NamedTuple — each field is sliced by pre-computed index constants (`_IDX_C`, `_IDX_OTO`, `_IDX_VIS`, `_IDX_VS`, etc.).

Evaluation order within one ODE step:
1. `sensory_model.read_outputs()` — slice delayed signals from sensory state
2. Apply sensory noise (canal, slip, pos OU drift, vel) to `sensory_out`
3. `brain_model.step()` — VS + NI + SG + EC + GE + pursuit → motor_cmd
4. `plant_model.step()` — motor_cmd → dx_plant; w_eye = dx_plant (algebraic, no lag)
5. `sensory_model.step()` — canal + otolith + visual delay driven by w_eye (must follow plant)

## LLM simulation pipeline

`oculomotor/llm_pipeline/simulate.py` converts a plain-English scenario description into a simulation
and figure using the Claude API.  `scripts/simulate.py` is a thin shim that calls into it.

### Usage

```bash
# Requires ANTHROPIC_API_KEY to be set in the environment
python -X utf8 scripts/simulate.py "healthy subject making a 20 deg saccade to the right"
python -X utf8 scripts/simulate.py "patient with left vestibular neuritis doing a head impulse test"
python -X utf8 scripts/simulate.py --dry-run "OKN: 30 deg/s full-field scene motion for 20 s then OKAN"
python -X utf8 scripts/simulate.py --json path/to/scenario.json   # skip LLM, load JSON directly
python -X utf8 scripts/simulate.py --show "..."                   # display figure interactively
```

### Web server

```bash
python -X utf8 scripts/server.py          # http://localhost:8000
python -X utf8 scripts/server.py --host 0.0.0.0 --port 8000
```

Features:
- LLM-driven single simulation or comparison
- **CSV log**: `outputs/simulation_log.csv` — timestamp, run_id, version, prompt, mode, title, figure_file, looks_correct, feedback
- **Feedback UI**: checkbox ("looks correct") + comment box + disclaimer; `POST /feedback`
- **Data download**: `GET /download/{run_id}` — CSV of t, eye pos/vel (yaw/pitch/roll), head/scene/target velocities
- Figures saved to `outputs/server_figures/<run_id>.png`

### Architecture

```
User description (str)
    ↓  Claude API (tool_use, forced schema)
SimulationScenario  (oculomotor/llm_pipeline/scenario.py — Pydantic)
    ├── HeadMotion    → oculomotor/sim/stimuli.py → head_vel_array (T, 3)
    ├── Target        → oculomotor/sim/stimuli.py → p_target_array, v_target_array
    ├── Visual        → oculomotor/sim/stimuli.py → v_scene_array, scene/target_present arrays
    └── Patient       → with_brain() / with_sensory() → Params NamedTuple
    ↓  oculomotor/llm_pipeline/runner.py
simulate(params, t, **stim_kw, return_states=True)
    ↓
matplotlib Figure  →  outputs/<slug>.png
```

### Adding new stimulus types

Add a generator to `oculomotor/sim/stimuli.py` following the existing pattern (returns
`t_array`, plus the relevant arrays). Then add the new `type` literal to the
appropriate sub-schema in `oculomotor/llm_pipeline/scenario.py` and handle it in
`oculomotor/llm_pipeline/runner.py:_build_stimulus()`.

### Adding new plot panels

Add a new `Literal` value to `PlotConfig.panels` in `oculomotor/llm_pipeline/scenario.py` and a
corresponding `elif panel_name == '...'` branch in `oculomotor/llm_pipeline/runner.py:_draw_panel()`.

### API key

Set `ANTHROPIC_API_KEY` in your shell or `.env` before running `simulate.py`.
The model defaults to `claude-opus-4-6`; use `--model claude-sonnet-4-6` for faster/cheaper calls.

## Tech stack

- **JAX** — core framework, autodiff, `jit`, `vmap`
- **Diffrax** — ODE integration within JAX
- **Optax** — gradient-based optimization
- **Matplotlib** — diagnostics and plotting
- **Pydantic** — SimulationScenario schema and validation
- **FastAPI + uvicorn** — web server
- **Anthropic Python SDK** — LLM scenario generation

## Fitting approach (future work)

**Current priority: get the fixed-parameter model to simulate correct behavior across all paradigms. Fitting comes later.**

The long-term goal is to fit this model to patient eye movement data — recovering parameters like `tau_vs`, `K_vs`, `canal_gains`, `tau_i` that characterize specific vestibular or cerebellar pathologies.

Planned approach when ready:
- Validate parameter recovery on synthetic data first (simulate with known θ, fit from perturbed init, check recovery)
- **Loss**: MSE between model-predicted and observed eye position/velocity, summed over stimulus conditions
- **Optimizer**: `optax.adam`, typical lr ~1e-3
- **Reparameterization**: `softplus` for positive TCs, `sigmoid` for bounded gains — ensures constraints without clamping
- **Gradients**: flow through `diffrax.diffeqsolve` via reverse-mode autodiff (already differentiable by construction)
- **Diagnostic plots**: loss curve, parameter trajectories vs. step, Bode plot (gain + phase vs. frequency), time-domain overlay (predicted vs. observed), residuals

## Conventions

- All angles in **degrees**, angular velocity in **deg/s**
- Eye position = `x_p` (plant state, 3D rotation vector)
- 3D axes: `[yaw, pitch, roll]`
- Head velocity input can be 1D (horizontal only) or 3D
- Gravity / specific force axis convention: **x = up** (matches `canal.py` and `gravity_estimator.py`); at rest upright, `g_head = [9.81, 0, 0]` m/s²
- `scene_present`: scalar in [0,1] — is the visual scene physically on? (external input)
- `target_present`: scalar in [0,1] — is there a foveal target? Gates pursuit integrator; set to 0 for pure OKN
- `gate_vf`: scalar in [0,1] — delayed visual-field gate (from retinal geometry); distinguishes fixation from out-of-field in the SG
- `pos_delayed`: (3,) delayed gated position error `gate_vf · e_pos` — zero means fixating OR target out of field; use `gate_vf` to disambiguate
