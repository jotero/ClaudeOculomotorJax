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

### Folder structure (`oculomotor/`)

```
oculomotor/
├── params.py                          Params NamedTuple (PhysParams + BrainParams)
├── models/
│   ├── sensory_models/
│   │   ├── canal.py                   Canal array SSM (Steinhausen, 6 canals, 12 states)
│   │   ├── retina.py                  Retinal geometry + visual delay cascade (240 states)
│   │   └── sensory_model.py           Connector: canal + retina → SensoryOutput (252 states)
│   ├── brain_models/
│   │   ├── velocity_storage.py        VS leaky integrator; PINV_SENS mixing + OKR (3 states)
│   │   ├── neural_integrator.py       NI leaky integrator (Robinson 1975) (3 states)
│   │   ├── saccade_generator.py       Robinson local-feedback burst + OPN gate (9 states)
│   │   ├── efference_copy.py          Burst delay cascade for slip cancellation (120 states)
│   │   ├── target_selector.py         Orbital gate → motor error command e_cmd
│   │   └── brain_model.py             Connector: VS + NI + SG + EC → u_p (135 states)
│   └── plant_models/
│       ├── plant_model_first_order.py First-order plant (Robinson 1964) (3 states)
│       └── readout.py                 Eye position readout utilities
└── sim/
    └── simulator.py                   ODE wiring + simulate() entry point
```

### State structure (390 total)

The ODE state is a `SimState` NamedTuple with three groups:

```
SimState(
    sensory (252):  [x_c (12) | x_vis (240)]
                     canal       visual delay cascade
    brain   (135):  [x_vs (3) | x_ni (3) | x_sg (9) | x_ec (120)]
                     vel-store   NI          sacc-gen   EC delay
    plant     (3):  x_p — eye rotation vector (deg); directly observable
)
```

`x_sg` sub-layout: `[x_copy(3) | z_ref(1) | e_held(3) | z_sac(1) | z_acc(1)]`

### Params structure

Parameters are nested NamedTuples — not dicts. Access via attribute path:

```python
class PhysParams(NamedTuple):
    tau_c, tau_s, canal_gains, tau_p, tau_vis, ...

class BrainParams(NamedTuple):
    tau_vs, K_vs, K_vis, g_vis, tau_i, ...

class Params(NamedTuple):
    phys: PhysParams
    brain: BrainParams
```

Use `with_phys(params, tau_p=0.2)` / `with_brain(params, tau_vs=15.0)` to create modified copies.

### Solver

`diffrax.Heun()` fixed step, `dt = 0.001 s`. Must satisfy `dt < 2 * tau_stage_vis = 0.004 s`.

### What "correct behavior" looks like

Each behavior has a corresponding demo script and output figure.

1. **VOR in the dark** — eye velocity ≈ −head velocity; gain ~0.9–1.0. Canal adaptation TC (~5 s) causes the VOR to decay during sustained rotation; velocity storage extends the effective TC to ~15–20 s.
   - Demo: `scripts/demo_vor.py` → `outputs/vor_dark.png`

2. **Velocity storage / TC extension** — during constant-velocity rotation in the dark, eye velocity decays with TC ~15–20 s (not the canal TC of ~5 s). VVOR: in a stationary lit world, OKR corrects VOR slip as the canal adapts — gaze stays stable throughout.
   - Demo: `scripts/demo_vor.py` → `outputs/vor_dark.png` (VS state panel), `outputs/vvor.png`

3. **OKN + OKAN** — during full-field visual motion, steady-state OKN gain ≈ 1. After scene off, OKAN persists with TC ~20 s (`tau_vs`). With saccades on, eye shows sawtooth nystagmus.
   - Demo: `scripts/demo_vor.py` → `outputs/okr.png`

4. **Saccades — main sequence + refractory period** — peak velocity follows `v_peak ≈ 700·(1−exp(−A/7))`, saturating ~600–700 deg/s. Robust intersaccadic interval (~150–200 ms). Oblique saccades straight with synchronized components.
   - Demo: `scripts/demo_saccade.py` → `outputs/saccade_summary.png`

5. **Saccades during pursuit / ramp target (staircase)** — catch-up saccade staircase as target moves continuously. No smooth pursuit pathway yet.
   - Demo: `scripts/demo_pursuit.py` → `outputs/smooth_pursuit.png`

6. **Saccades during head movement** — corrective saccades fire periodically as VOR slip accumulates; staircase toward target.
   - Demo: `scripts/demo_saccade.py` → `outputs/vor_saccade_cascade.png`

7. **Efference copy** — burst commands must not contaminate VS/OKR. VS state identical with/without saccades during OKN.
   - Demo: `scripts/demo_efference.py` → `outputs/efference_*.png`

## Not yet implemented (future work)

- **Smooth pursuit** — foveal target tracking (MT/MST pathway). Requires a separate pursuit drive distinct from full-field OKN.
- **Binocularity / vergence** — currently monocular.
- **Listing's law** — torsional constraints not enforced.
- **Otolith + head orientation** — `otolith.py` partially implemented but not connected. Requires gravity estimation, somatogravic feedback, tilt-translation disambiguation.
- **Multiple plant models** — see design note below.
- **Multiple brain models** — see design note below.

### Design note: swappable plants and brain models

The simulator is designed so that `plant_models/` and `brain_models/` can be swapped without touching sensory machinery. The integration point is the **motor command interface** between brain and plant.

**Motor command format (current and future):**

The brain should output a `MotorCommand` NamedTuple:

```python
class MotorCommand(NamedTuple):
    step:  jnp.ndarray   # (3,) tonic NI drive  — eye position hold
    pulse: jnp.ndarray   # (3,) phasic burst    — saccade acceleration
```

Units: motoneuron firing rate equivalent (normalized or deg/s). The plant converts these to forces/torques internally — the brain does not need to know muscle geometry.

**Why separate step and pulse?**
- Robinson's first-order plant combines them as `u_p = step + pulse` — valid because inertia is negligible (overdamped, I→0 means force ∝ velocity directly).
- A second-order plant (MJX/muscle-level) applies them to different motoneuron pools with different dynamics — tonic vs phasic muscle fibers.
- Keeping them split in the interface costs nothing for the first-order plant and enables the upgrade.

**Plant interface contract:**

```python
def step(x_p, cmd: MotorCommand, theta) -> (dx_p, q_eye):
    # x_p:  (N,)    plant state (eye rotation or muscle activations)
    # cmd:  MotorCommand  — step + pulse, both (3,) in rotation-vector space
    # q_eye: (3,)   observed eye rotation (output to retina / feedback)
```

Any plant implementing this contract (first-order, second-order, MJX-backed) is a drop-in replacement in `simulator.py`.

**Brain model interface contract:**

```python
def step(x_brain, sensory_out: SensoryOutput, e_cmd, scene_present, theta)
    -> (dx_brain, motor_cmd: MotorCommand, u_burst):
```

Different brain architectures (Raphan-Cohen, Kalman, RL policy) swap in here. The sensory model and plant remain unchanged.

## Current status

Saccades to static targets work well. Saccades to moving targets / during pursuit are the active area of work (as of 2026-04-14).

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
- **`theta` is a `Params` NamedTuple** — access via `theta.phys.tau_c`, `theta.brain.tau_vs`, etc. Never treat it as a dict.
- Module-level constants are kept only when used by external code (e.g. `retina.C_slip`, `canal.PINV_SENS`).

### Nonlinear extensions

Some modules have nonlinearities that wrap the linear ABCD core:

- **Canal** (`canal.py`): `nonlinearity(x_c, gains)` applies smooth push-pull rectification to the `x2` (inertia state) to get afferent firing rates. The linear `A @ x + B @ u` drives the state derivative; only the output is nonlinear. Re-exported as `canal_nonlinearity` from `sensory_model.py`.
- **Saccade generator**: gates (`gate_err`, `gate_res`, `gate_dir`) and adaptive reset TC layered on top of linear SSM core.
- **Visual delay** (`retina.py`): fixed companion-form `A` (40-stage cascade) with module-level `C_slip` / `C_pos` readout — kept at module level because external code reads them directly.

### Connector modules

`sensory_model.py` and `brain_model.py` are **connector modules** — they import their sub-SSMs, own the combined state layout and index constants, and expose a single `step()` + output-read interface. They do not implement physics themselves.

### Example: Neural Integrator (simplest case)

```python
N_STATES = N_INPUTS = N_OUTPUTS = 3

def step(x_ni, u_vel, theta):
    A = (-1/theta.brain.tau_i) * jnp.eye(3)
    D = theta.phys.tau_p * jnp.eye(3)
    # B = C = I (identity — omitted)
    dx  = A @ x_ni + u_vel
    u_p = x_ni + D @ u_vel
    return dx, u_p
```

### Wiring in the simulator

`ODE_ocular_motor` in `sim/simulator.py` calls each module's `step()` in signal-flow order, passing outputs of one as inputs to the next. The global state is a `SimState` NamedTuple — each field is sliced by pre-computed index constants (`_IDX_C`, `_IDX_VIS`, `_IDX_VS`, etc.).

## Tech stack

- **JAX** — core framework, autodiff, `jit`, `vmap`
- **Diffrax** — ODE integration within JAX
- **Optax** — gradient-based optimization
- **Matplotlib** — diagnostics and plotting

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
- `scene_present`: scalar in [0,1] — is the visual scene physically on? (external input)
- `pos_visible`: (3,) — position error after visual-field gate — target may be present but outside ~90° field
