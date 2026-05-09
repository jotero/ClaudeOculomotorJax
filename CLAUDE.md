# ClaudeOculomotorJax — Project Context for Claude

## Shared analysis utilities — no redundant helpers

`oculomotor/analysis.py` is the single source of truth for post-hoc signal extraction and plotting
helpers. **Never redefine these locally in a demo script or notebook cell:**

| Function | What it gives you |
|---|---|
| `vs_net(states)` | VS net signal x_L − x_R, (T, 3), deg/s |
| `ni_net(states)` | NI net signal x_L − x_R, (T, 3), deg |
| `vs_null(states)` | VS null-adaptation state, (T, 3) |
| `ni_null(states)` | NI null-adaptation state, (T, 3) |
| `extract_burst(states, theta)` | u_burst via vmap, (T, 3) |
| `extract_sg(states, theta)` | All SG sub-states dict |
| `extract_canal(states)` | Canal yaw estimate, (T,) |
| `extract_spv(t, ev, burst)` | Slow-phase velocity via burst mask |
| `fit_tc(t, y, t_start, t_end)` | Exponential TC fit |
| `ax_fmt(ax, ylabel, xlabel, ylim)` | Standard axes formatting |

When you need a yaw-only scalar in a notebook, use a thin one-liner wrapper:
```python
def vs_net_yaw(states): return _vs_net3(states)[:, 0]
```
Do **not** reimplement the logic.

## Running scripts

Always use `-X utf8` to avoid Windows cp1252 encoding errors (Greek letters in print statements crash otherwise):

```bash
"d:/OneDrive/UC Berkeley/OMlab - JOM/Code/ClaudeOculomotorJax/.venv/Scripts/python.exe" -X utf8 scripts/bench_vor_okr.py
```

Or from PowerShell:

```powershell
& "d:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\.venv\Scripts\python.exe" -X utf8 scripts\bench_vor_okr.py
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

Visual pathway: `Per-eye retinal slip → retina.step (sharp gamma cascade) → perception_cyclopean (binocular fusion + brain LP) → VS / pursuit / SG / vergence`

Saccade loop: `Retinal position error → retina + cyclopean → SG (Robinson local-feedback) → NI + Plant`

Pursuit loop: `Retinal velocity error → retina + cyclopean → Pursuit integrator (Smith predictor) → NI + Plant`

### Folder structure (`oculomotor/`)

```
oculomotor/
├── __init__.py                        __version__ from git describe --tags --always --dirty
├── models/
│   ├── sensory_models/                Peripheral sensors only — canal, otolith, retina geometry
│   │   ├── canal.py                   Canal array SSM (Steinhausen, 6 canals, 12 states)
│   │   ├── otolith.py                 Otolith SSM (bilateral LP adaptation, 6 states)
│   │   ├── retina.py                  Geometry (world_to_retina) + sensor saturation +
│   │   │                              per-eye sharp gamma cascade (90 states/eye).
│   │   │                              retina.step → RetinaOut (delayed per-eye signals).
│   │   └── sensory_model.py           Connector: canal + otolith + retina_L + retina_R →
│   │                                   SensoryOutput {canal, otolith, retina_L, retina_R}
│   │                                   (198 states)
│   ├── brain_models/                  Cortical computations — operate on already-delayed signals
│   │   ├── perception_self_motion.py  VS + GE + HE unified observer (Laurens & Angelaki 2017)
│   │   │                              + post-delay scene EC sub + magnitude/directional gates
│   │   ├── perception_target.py       Target EC sub + gates + working memory (FEF/dlPFC layer)
│   │   ├── perception_cyclopean.py    Binocular fusion (NPC gate, OKR, dominance) on delayed
│   │   │                              per-eye signals + brain LP smoothing (43 states).
│   │   │                              perception_cyclopean.step → CyclopeanOut.
│   │   ├── neural_integrator.py       NI leaky integrator + bilateral push-pull + null adapt
│   │   ├── saccade_generator.py       Robinson local-feedback burst + OPN gate
│   │   ├── pursuit.py                 Smooth pursuit leaky integrator + Smith predictor
│   │   ├── tvor.py                    Translational VOR (vestibular + visual fusion)
│   │   ├── vergence_accommodation.py  Vergence + accommodation (AC/A and CA/C cross-links)
│   │   ├── listing.py                 Listing's-law torsion corrections (smooth pathways)
│   │   ├── final_common_pathway.py    Nucleus + nerve gain stages → 12 muscle activations
│   │   ├── efference_copy.py          (legacy) — kept for backward compat; current EC handled
│   │   │                              via cascade_lp_step blocks inside brain_model.py
│   │   └── brain_model.py             Connector: cyclopean → target/self_motion perception →
│   │                                   pursuit + SG + TVOR + NI + vergence/acc → motor_cmd
│   │                                   (151 states)
│   ├── plant_models/
│   │   ├── plant_model_first_order.py First-order plant (Robinson 1964) (3 states/eye)
│   │   └── readout.py                 Eye position readout + rotation_matrix()
│   └── llm_pipeline/                  Natural-language → simulation pipeline
│       ├── scenario.py                Pydantic schema (SimulationScenario, Patient, …)
│       ├── runner.py                  Stimulus builder + simulator wiring + figure generator
│       └── simulate.py                CLI entry point + Claude API call
└── sim/
    ├── simulator.py                   ODE wiring + simulate() entry point
    └── stimuli.py                     Centralized stimulus generators
```

### scripts/

```
scripts/
├── simulate.py             Thin shim → oculomotor.llm_pipeline.simulate.main()
├── server.py               FastAPI web server — LLM pipeline + logging + feedback + download
├── bench_vor_okr.py        VOR / VVOR / OKN bench (Raphan Fig. 9 + cascade plots)
├── bench_saccades.py       Saccade main sequence, oblique, refractoriness, cascade
├── bench_pursuit.py        Smooth pursuit velocity range, sinusoidal, cascade
├── bench_fixation.py       Fixational eye movements
├── bench_vergence.py       Vergence step + sustained
├── bench_accommodation.py  Accommodation step responses
├── bench_gravity.py        OCR / tilt suppression / OVAR
├── bench_listing.py        Listing's-law torsion checks
├── bench_tvor.py           Translational VOR
├── bench_clinical_*.py     Clinical paradigms (cerebellum, CN palsies, NI/VS, vergence, vestibular, saccades)
├── diag_vergence*.py       Vergence diagnostics
└── _bench_dt.py            Numerical-stability sweep (private)
```

### State structure (binocular)

The ODE state is a `SimState` NamedTuple with three groups (sizes after the
per-eye-retina + cyclopean-fusion-in-brain refactor):

```
SimState(
    sensory (198):  [x_c (12) | x_oto (6) | x_retina_L (90) | x_retina_R (90)]
                     canal      otolith     per-eye sharp     per-eye sharp
                     _IDX_C     _IDX_OTO    _IDX_RETINA_L     _IDX_RETINA_R
                                            (_IDX_VIS_L = _IDX_RETINA_L,  backward compat)
                                            (_IDX_VIS   = full retina_L+R block)

    brain   (151):  [self_motion (21) | NI (9) | SG (18) | pursuit (3) | va (11) |
                      target_mem (4) | cyc_brain (43) | ec_scene (21) | ec_target (21)]
                     _IDX_SELF_MOTION   = VS + GE + HE   (perception_self_motion)
                     _IDX_NI            = bilateral NI + null
                     _IDX_SG            = saccade generator
                     _IDX_PURSUIT       = pursuit velocity memory
                     _IDX_VA            = vergence + accommodation
                     _IDX_TARGET_MEM    = perception_target memory (x_mem(3) + trust(1))
                     _IDX_CYC_BRAIN     = perception_cyclopean brain LP block
                     _IDX_EC_VEL_SCENE  = scene EC cascade (matches scene cascade shape)
                     _IDX_EC_VEL_TARGET = target EC cascade (matches target_vel shape)

    plant     (6):  [x_p_L (3) | x_p_R (3)]  — left/right eye rotation vectors (deg)
                     _IDX_P_L    _IDX_P_R
)
```

`x_retina_<L|R>` sub-layout (per eye, sharp gamma cascade only — N=6 stages each):
`[scene_angular_vel(18) | scene_linear_vel(18) | target_pos(18) | target_vel(18) |
  scene_visible(6) | target_visible(6) | defocus(6)]`

`x_cyc_brain` sub-layout (post-fusion brain LP smoothing):
`[scene_angular_vel(3 LP) | scene_linear_vel(3 LP) | target_pos(18 N-stage) |
  target_vel(3 LP) | target_disparity(3 LP) | scene_visible(6 N-stage) |
  target_visible(6 N-stage) | defocus(1 LP)]`

Cyclopean delayed signals are read via `perception_cyclopean.C_*` matrices on
`brain[:, _IDX_CYC_BRAIN]` (NOT on sensory state — they live in brain now).

### Params structure

Parameters are nested NamedTuples — not dicts. Access via attribute path:

```python
class SensoryParams(NamedTuple):
    # Sensor-side only — peripheral physiology.
    tau_c, tau_s, canal_gains, canal_floor, canal_v_max, tau_oto,         # canals + otolith
    tau_vis_sharp,                                                        # retina sharp delay
    v_max_target_vel, v_max_scene_vel,                                    # MT/MST + NOT/AOS ceilings
    visual_field_limit, k_visual_field,                                   # eccentricity gate
    sigma_canal, sigma_slip, sigma_pos, sigma_vel, tau_*_drift,           # noise (OU)
    ipd                                                                   # binocular geometry

class PlantParams(NamedTuple):
    tau_p

class BrainParams(NamedTuple):
    # Cortical + brainstem parameters.
    tau_vs, tau_vs_pitch_frac, tau_vs_roll_frac, K_vs, K_vis, g_vis, b_vs, tau_vs_adapt,  # VS
    tau_i, tau_p, tau_vis, b_ni, tau_ni_adapt,                    # NI
    tau_vis_sharp, tau_vis_smooth_motion, tau_vis_smooth_target_vel,
    tau_vis_smooth_disparity, tau_vis_smooth_defocus, tau_brain_pos,    # brain-side LP TCs
    npc, div_max, vert_max, tors_max, eye_dominant,               # binocular fusion policy
    v_crit_ec_gate, n_ec_gate, alpha_ec_dir, bias_ec_dir,         # post-delay EC gate
    g_burst, e_sat_sac, k_sac, threshold_sac, ...                 # saccade generator
    K_pursuit, K_phasic_pursuit, tau_pursuit,                     # pursuit
    K_grav, K_gd, g_ocr, orbital_limit, alpha_reset, ...          # gravity / OCR / SG misc

class Params(NamedTuple):
    sensory: SensoryParams
    plant:   PlantParams
    brain:   BrainParams
```

Use `with_sensory(params, sigma_canal=2.0)` / `with_brain(params, tau_vs=15.0)` / `with_plant(params, tau_p=0.2)` to create modified copies.

### Solver

`diffrax.Heun()` fixed step, `dt = 0.001 s`. Must satisfy `dt < 2 * tau_stage_vis = 0.004 s`.

### Sensory noise

Four independent noise sources, **non-zero by default** (so a vanilla `simulate(PARAMS_DEFAULT, ...)` already produces realistic fixational drift, microsaccades, and pursuit jitter). All four are Ornstein-Uhlenbeck processes — small τ approaches band-limited white noise, longer τ produces drift-like fluctuations. Defaults from `SensoryParams`:

| σ param      | default     | τ param            | default | what it drives |
|--------------|-------------|--------------------|---------|----------------|
| `sigma_canal`  | 1.0 deg/s | `tau_canal_drift`  | 0.005 s | canal afferent noise; filtered by VS/NI/plant |
| `sigma_slip`   | 0.0 deg/s | `tau_slip_drift`   | 0.005 s | retinal slip noise (off by default); VS/OKR |
| `sigma_pos`    | 0.2 deg   | `tau_pos_drift`    | 0.2 s   | retinal position drift; **triggers microsaccades** |
| `sigma_vel`    | 1.0 deg/s | `tau_vel_drift`    | 0.005 s | retinal velocity noise; pursuit integrator |

`SG_acc` accumulator diffusion (`sigma_acc=0.2`, in `BrainParams`) adds RT variability to saccade triggering.

Noise is pre-generated as arrays before `diffeqsolve` and passed as `LinearInterpolation` inputs — ODE remains pure and differentiable.

```python
params = with_sensory(PARAMS_DEFAULT,
    sigma_canal    = 2.0,   # crank up canal noise
    sigma_pos      = 0.0,   # disable microsaccades for clean cascade traces
    ...
)
states = simulate(params, t, ..., key=jax.random.PRNGKey(42))
```

`sigma_pos` uses an Ornstein-Uhlenbeck process (not white noise) so drift accumulates slowly,
crosses the SG threshold occasionally, and triggers sparse corrective microsaccades.
White noise on `pos_delayed` would fire the SG continuously.

**Debug benches that need noiseless traces** (e.g. cascade figures, symmetric-vergence triggers, anything where you need exact bilateral cancellation) must explicitly disable the relevant noise sources via `with_sensory(...sigma_canal=0, sigma_pos=0, sigma_vel=0)` and `with_brain(...sigma_acc=0)`. By convention these go in the bench's top-level `PARAMS_*` constant alongside any other overrides, so the figure footer (params overrides line) makes them visible.

### Versioning

`oculomotor.__version__` is derived from `git describe --tags --always --dirty` at import time.
No manual version bumping required — tag a release with `git tag v1.0` and it appears automatically.
The version string is logged with every server simulation call.

### What "correct behavior" looks like

Each behavior has a corresponding demo script and output figure.

1. **VOR in the dark** — eye velocity ≈ −head velocity; gain ~0.9–1.0. Canal adaptation TC (~5 s) causes the VOR to decay during sustained rotation; velocity storage extends the effective TC to ~15–20 s.
   - Demo: `scripts/bench_vor_okr.py` → `outputs/vor_dark.png`

2. **Velocity storage / TC extension** — during constant-velocity rotation in the dark, eye velocity decays with TC ~15–20 s (not the canal TC of ~5 s). VVOR: in a stationary lit world, OKR corrects VOR slip as the canal adapts — gaze stays stable throughout.
   - Demo: `scripts/bench_vor_okr.py` → `outputs/vvor.png`

3. **OKN + OKAN** — during full-field visual motion, steady-state OKN gain ≈ 1. After scene off, OKAN persists with TC ~20 s (`tau_vs`). With saccades on, eye shows sawtooth nystagmus.
   - Demo: `scripts/bench_vor_okr.py` → `outputs/okr.png`

4. **Saccades — main sequence + refractory period** — peak velocity follows `v_peak ≈ 700·(1−exp(−A/7))`, saturating ~600–700 deg/s. Robust intersaccadic interval (~150–200 ms). Oblique saccades straight with synchronized components.
   - Demo: `scripts/bench_saccades.py` → `outputs/saccade_summary.png`

5. **Smooth pursuit** — foveal target tracking via MT/MST velocity pathway. Pursuit integrator + Smith predictor (efference copy cancels saccadic contamination). Catch-up saccades fire when position error exceeds threshold during ramp pursuit.
   - Demo: `scripts/bench_pursuit.py` → `outputs/smooth_pursuit.png`

6. **Saccades during head movement** — corrective saccades fire periodically as VOR slip accumulates; staircase toward target.
   - Demo: `scripts/bench_saccades.py` → `outputs/vor_saccade_cascade.png`

7. **Efference copy** — burst commands must not contaminate VS/OKR. Verified inside the VOR/OKR cascade plot in `bench_vor_okr.py` and the saccade cascade in `bench_saccades.py`.

8. **Fixational eye movements** — canal noise filtered by VS/NI/plant; retinal position OU drift produces sparse corrective microsaccades; retinal velocity noise drives pursuit-like slow drift.
   - Demo: `scripts/bench_fixation.py` → `outputs/fixation.png`

## Current status (2026-05-08)

- **Working well**: VOR, VVOR, OKN/OKAN, saccades (main sequence, refractory, oblique), smooth pursuit (velocity-driven), efference copy slip cancellation, otolith LP adaptation, sensory noise system, fixational eye movements, OCR (ocular counter-rolling).
- **Recent change (2026-05-08)**: visual pathway refactored — per-eye sharp gamma cascade now lives in `retina.step` (90 states/eye), and binocular fusion + brain LP smoothing moved into `brain_models/perception_cyclopean.py` (43 cyclopean brain LP states). `SensoryOutput` now bundles per-eye `RetinaOut`s; brain consumes a `CyclopeanOut` produced inside `brain_model.step`. Brain-side LP TCs and fusion-policy params (npc, div_max, vert_max, tors_max, eye_dominant, tau_vis_smooth_*) migrated from `SensoryParams` to `BrainParams`. Sensory state shrank 818→198, brain grew 108→151. Behavior preserved within ~1% on bench numbers.
- **Recent change (2026-04-28)**: VS time constants now per-axis via scalar + two fractions. `tau_vs` (yaw, 20 s) is unchanged — all existing lesion code still works. New params: `tau_vs_pitch_frac=0.4` (→ 8 s pitch) and `tau_vs_roll_frac=0.15` (→ 3 s roll). Driven by physiological evidence (Raphan 1979, Dai 1991, Angelaki 1995). Roll decay at 3 s fixes torsion overshoot in OCR benchmarks. `with_brain(p, tau_vs=X)` still works — all axes scale together.
- **Recent change (2026-04-18)**: NI expanded to bilateral push-pull architecture (9 states: x_L, x_R, x_null). Null adaptation added to both NI (`tau_ni_adapt=20s`) and VS (`tau_vs_adapt=600s`). Net output `x_L − x_R` identical to old scalar `x_ni` in healthy symmetric case. Models rebound nystagmus (NI) and extended OKAN / velocity storage adaptation (VS). Brain: 147→156 states; model total: 971→980. New BrainParams: `b_ni=0`, `tau_ni_adapt=20s`, `tau_vs_adapt=600s`.
- **Recent change (prior session)**: VS expanded to bilateral push-pull architecture (6→9 states including x_null). Net output `x_L − x_R` identical to old scalar `x_vs` in healthy symmetric case — all existing demos/notebooks work unchanged (use `vs_net()` helper for extraction). New `b_vs` parameter (default 100 deg/s) sets VN resting bias.
- **New notebook**: `notebooks/integrator_disorders.ipynb` — gaze-evoked nystagmus, rebound nystagmus (NI null adaptation), Bruns nystagmus, VS null / extended OKAN, PAN placeholder.
- **Pending improvement**: Pursuit position sensitivity (`K_pursuit_pos` — see future work).
- **Not yet debugged**: Gravity estimator (`gravity_estimator.py`) — partially validated (OCR benchmarks pass); torsion drift during static tilt being investigated. T-VOR debugging deferred until vergence is implemented.
- **Next focus**: Complete OCR torsion debugging; then binocularity and vergence.

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
    # sensory_out fields: canal(6), slip_delayed(3), pos_delayed(3), target_in_vf(scalar),
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
- Module-level constants are kept only when used by external code (e.g. `retina.C_slip`, `retina.C_target_in_vf`, `canal.PINV_SENS`).

### Nonlinear extensions

Some modules have nonlinearities that wrap the linear ABCD core:

- **Canal** (`canal.py`): `nonlinearity(x_c, gains)` applies smooth push-pull rectification to the `x2` (inertia state) to get afferent firing rates. The linear `A @ x + B @ u` drives the state derivative; only the output is nonlinear. Re-exported as `canal_nonlinearity` from `sensory_model.py`.
- **Saccade generator**: gates (`gate_err`, `gate_res`, `gate_dir`) and adaptive reset TC layered on top of linear SSM core. Target selection (orbital clip + centering saccade) is handled internally using `x_ni` as a proxy for eye position and `target_in_vf` to detect out-of-field targets.
- **Visual delay** (`retina.py` + `perception_cyclopean.py`): two-stage. (a) Per-eye sharp gamma cascade in `retina.step` (N=6 stages × τ_retina), with `velocity_saturation` and visibility gating done before cascade input. (b) Post-fusion brain LP smoothing in `perception_cyclopean.step` (channel-specific TCs: motion, target_vel, disparity, defocus, plus N-stage gamma for target_pos / visibility). The brain's `C_slip` / `C_pos` / `C_vel` / `C_target_disp` / `C_target_visible` / etc. readout matrices live in `perception_cyclopean` and read into `brain[:, _IDX_CYC_BRAIN]`.

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
1. `sensory_model.read_outputs()` — exposes canal, otolith, and per-eye `RetinaOut`
   from sensory state (sharp-cascade-delayed signals).
2. Apply sensory noise (canal + per-eye slip / target_vel / target_pos OU drift)
   to `sensory_out.retina_L` and `sensory_out.retina_R`.
3. `brain_model.step()`:
   - `perception_cyclopean.step` — fuse per-eye delayed signals + brain LP
     smoothing → `CyclopeanOut`.
   - `perception_target.step` — target EC sub + magnitude/directional gates +
     working memory (consumes `cyc.target_*`).
   - `perception_self_motion.step` — VS + GE + HE (consumes `cyc.scene_*`).
   - Pursuit, SG, T-VOR, NI, vergence/accommodation, final common pathway.
   - Post-delay EC cascades (`x_ec_scene`, `x_ec_target`) advanced at end.
4. `plant_model.step()` — motor_cmd → dx_plant; w_eye = dx_plant.
5. `sensory_model.step()` — canal + otolith + per-eye `retina.step` driven by
   the freshly-updated eye state (must follow plant).

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
- **World frame is LEFT-HANDED**: x=right, y=up, z=forward (x × y = −z)
- **Angular vectors** `[yaw, pitch, roll]` are NOT in xyz order — use `ypr_to_xyz` / `xyz_to_ypr` (from `retina.py` for JAX, `kinematics.py` for numpy) before/after rotation-matrix ops:
  - `ypr_to_xyz([yaw, pitch, roll]) = [−pitch, yaw, roll]`
  - yaw (idx 0): rotation about +y — left-hand: forward→right (rightward turn)
  - pitch (idx 1): rotation about −x — left-hand: forward→up (look up)
  - roll (idx 2): rotation about +z — left-hand: right→up
- Head velocity input can be 1D (horizontal only) or 3D
- Gravity / specific force axis convention: **x = up** (matches `canal.py` and `gravity_estimator.py`); at rest upright, `g_head = [9.81, 0, 0]` m/s²
- `scene_present`: scalar in [0,1] — is the visual scene physically on? (external input)
- `target_present`: scalar in [0,1] — is there a foveal target? Gates pursuit integrator; set to 0 for pure OKN
- `target_in_vf`: scalar in [0,1] — delayed visual-field gate (from retinal geometry); distinguishes fixation from out-of-field in the SG
- `pos_delayed`: (3,) delayed gated position error `target_in_vf · e_pos` — zero means fixating OR target out of field; use `target_in_vf` to disambiguate
