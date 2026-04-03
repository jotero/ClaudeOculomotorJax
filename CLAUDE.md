# ClaudeOculomotorJax — Project Context for Claude

## What this is

A JAX-based simulation of the primate oculomotor system. The goal is a differentiable, biophysically grounded model of how the brain controls eye movements — suitable for fitting to experimental data.

## Architecture

Signal flow: `Head velocity → Canal Array → Velocity Storage → Neural Integrator → Plant → Eye position`

Visual loop: `Retinal slip → Visual delay cascade → Velocity Storage (OKR/OKAN)`

Saccade loop: `Retinal position error → Visual delay → Saccade Generator (Robinson local-feedback) → NI + Plant`

### Key modules (`oculomotor/models/`)

| Module | Role |
|--------|------|
| `canal.py` | Semicircular canal array (Steinhausen torsion-pendulum, 6 canals) |
| `velocity_storage.py` | VS leaky integrator; owns canal PINV_SENS mixing + visual gains |
| `neural_integrator.py` | NI leaky integrator (Robinson 1975) |
| `plant.py` | First-order ocular plant (Robinson 1964) |
| `retina.py` | Converts Cartesian target → retinal angle |
| `visual_delay.py` | 40-stage cascade delay (120 slip + 120 pos states) |
| `saccade_generator.py` | Robinson local-feedback burst model + refractory OPN gate |
| `efference_copy.py` | NI+plant copy driven by burst only (for e_slip cancellation) |
| `ocular_motor_simulator.py` | Top-level ODE + `simulate()` entry point |

### State vector (270 total)

`[x_c (12) | x_vs (3) | x_ni (3) | x_p (3) | x_vis (240) | x_sg (4) | x_ec (6)]`

### Solver

`diffrax.Heun()` fixed step, `dt = 0.001 s`. Must satisfy `dt < 2 * tau_stage_vis = 0.004 s`.

### What "correct behavior" looks like

Each behavior has a corresponding demo script and output figure.

1. **VOR in the dark** — eye velocity ≈ −head velocity; gain ~0.9–1.0. Canal adaptation TC (~5 s) causes the VOR to decay during sustained rotation; velocity storage extends the effective TC to ~15–20 s.
   - Demo: `scripts/demo_vor.py` → `outputs/vor_dark.png` (8-panel signal cascade: head → canal → VS → NI → plant; overlays with/without VS)

2. **Velocity storage / TC extension** — during constant-velocity rotation in the dark, eye velocity decays with TC ~15–20 s (not the canal TC of ~5 s). VS charges from canal input and sustains the response beyond cupula adaptation. VVOR: in a stationary lit world, OKR corrects the VOR slip that accumulates as the canal adapts — gaze stays stable throughout.
   - Demo: `scripts/demo_vor.py` → `outputs/vor_dark.png` (VS state panel), `outputs/vvor.png` (dark vs lit gaze error)

3. **OKN + OKAN** — during full-field visual motion, steady-state OKN gain ≈ 1 (eyes track scene velocity). After scene off, OKAN persists with TC ~20 s (`tau_vs`). Fast OKR onset via `g_vis` feedthrough; sustained OKAN via `K_vis` charging `x_vs`. With saccades on, eye shows sawtooth nystagmus (slow phase + resetting fast phases).
   - Demo: `scripts/demo_vor.py` → `outputs/okr.png` (6-panel cascade: scene vel → retinal slip → delayed slip → VS state → visual drive → eye vel)

4. **Saccades — main sequence + refractory period** — peak velocity vs. amplitude follows the nonlinear main sequence: `v_peak ≈ 700 · (1 − exp(−A/7))`, saturating ~600–700 deg/s for large saccades. Critically: robust intersaccadic interval (~150–200 ms) — the system must not re-trigger immediately after a saccade even if a large error remains. Oblique saccades should be straight (not curved) with synchronized H and V components.
   - Demo: `scripts/demo_saccade.py` → `outputs/saccade_main_sequence.png`, `outputs/saccade_sequence.png`, `outputs/saccade_oblique.png`

5. **Saccades during pursuit / ramp target (staircase)** — when target moves continuously, saccades should form a staircase: each saccade lands on or near the moving target, then the system waits out the refractory period (~150 ms), then saccades again as error grows. No smooth pursuit pathway exists yet, so catch-up saccades are the only mechanism. This is the **current problem**.
   - Demo: `scripts/demo_smooth_pursuit.py` → `outputs/smooth_pursuit.png`

6. **Saccades during head movement** — same staircase requirement as pursuit, but driven by VOR failure to perfectly compensate head velocity. As head rotates, residual retinal error accumulates; corrective saccades should fire periodically (respecting the refractory period) and bring gaze back toward the target each time. Between saccades, VOR provides partial compensation. This is closely related to the pursuit problem and likely shares the same root cause.
   - Demo: `scripts/demo_saccade.py` → `outputs/saccade_vor.png` (head 60°/s step, corrective saccades to straight-ahead target)

7. **Efference copy** — saccade bursts must NOT contaminate the OKR/VS pathway. The plant copy (`x_pc`) should: (a) stay at zero when there is no burst, (b) track burst-driven eye velocity during a saccade and decay back with TC = `tau_p` after the burst ends, (c) keep the delayed slip (`e_slip_delayed`) clean during OKN nystagmus — VS state should be the same whether saccades are on or off (only scene motion drives it). Without efference copy, each fast phase of OKN would spuriously charge VS.
   - Demo: `scripts/demo_efference_copy.py` → `outputs/efference_plant_copy.png`, `outputs/efference_cancellation.png`, `outputs/efference_okn_debug.png`

## Not yet implemented (future work)

- **Smooth pursuit** — foveal target tracking driven by retinal slip of a *small* target (distinct from full-field OKN). Requires a separate pursuit pathway (MT/MST → pursuit motor command) that is not yet in the model.
- **Binocularity / vergence** — currently monocular. Vergence angle, ACA ratio (accommodative convergence), and binocular disparity not modeled.
- **Listing's law** — torsional constraints on 3D eye position not enforced.
- **More complex plant** — current plant is first-order (Robinson 1964). Future: two-muscle-group plant, orbital mechanics, pulley mechanics (Demer et al.).
- **Otolith + head orientation estimation** — `otolith.py` is partially implemented but not connected to the main simulator. Full integration requires: otolith linear acceleration sensing, internal estimate of head orientation (gravity vector), somatogravic feedback (tilt-translation ambiguity resolution), and rotational-translational interaction (centripetal terms). This feeds into VS and the overall spatial orientation estimate.

## Current status

Saccades to static targets work well. Saccades to moving targets / during pursuit are broken (as of 2026-04-02). The saccade trigger/refractory logic is the active area of work.

## Key parameters to know

```python
'g_burst':       700.0   # burst ceiling (deg/s)
'threshold_sac':   0.5   # saccade dead-zone (deg)
'tau_ref':         0.15  # refractory period (s)
'tau_reset_fast':  0.1   # reset TC after burst (s)
```

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
| `get_A(theta)` | function → `(N_STATES, N_STATES)` | always (at minimum leaks are θ-dependent) |
| `get_B(theta)` or `B` | function or module constant | if θ-dependent: function; if fixed: `B = jnp.eye(...)` |
| `C` | module constant `(N_OUTPUTS, N_STATES)` | almost never — usually identity |
| `get_D(theta)` or absent | function → `(N_OUTPUTS, N_INPUTS)` | if θ-dependent: function; if no feedthrough: omit |
| `step(x, u, theta)` → `(dx, y)` | pure function | — |

### `step()` contract

```python
def step(x, u, theta):
    dx = get_A(theta) @ x + get_B(theta) @ u   # or B @ u if fixed
    y  = C @ x + get_D(theta) @ u              # or C @ x if no feedthrough
    return dx, y
```

- **Pure function** — no side effects, no global state. Compatible with `jax.jit` and `jax.grad`.
- Returns `(dx, y)` always — the ODE integrator uses `dx`; the simulator uses `y` to wire modules together.
- Input/output shapes and units must be documented in the module docstring.
- `theta` is always a `dict`; use `.get('key', default)` for optional parameters.

### Nonlinear extensions

Some modules have nonlinearities that wrap the linear ABCD core:

- **Canal**: `canal_nonlinearity(x_c, gains)` applies the smooth half-wave rectification to `x2` (inertia state) to get afferent firing rates. The linear `A @ x + B @ u` still drives the state derivative; only the *output* is nonlinear.
- **Saccade generator**: gates (`gate_err`, `gate_res`, `gate_dir`) and an adaptive reset TC are layered on top. The ABCD structure is still present — `A_ni` (saccade mode) and `A_reset` (inter-saccade mode) are separate matrices; gating blends between them via a differentiable soft switch.
- **Visual delay**: uses a fixed companion-form `A` (40-stage cascade) with a fixed `C_slip` / `C_pos` readout; `B` is identity (input drives the first stage directly).

### Example: Neural Integrator (simplest case)

```python
N_STATES = N_INPUTS = N_OUTPUTS = 3

def get_A(theta): return (-1/theta['tau_i']) * jnp.eye(3)
B = jnp.eye(3)
C = jnp.eye(3)
def get_D(theta): return theta['tau_p'] * jnp.eye(3)

def step(x_ni, u_vel, theta):
    dx  = get_A(theta) @ x_ni + u_vel          # B = I, omitted
    u_p = C @ x_ni + get_D(theta) @ u_vel
    return dx, u_p
```

### Wiring in the simulator

`ODE_ocular_motor` calls each module's `step()` in signal-flow order, passing outputs of one as inputs to the next. The global state vector is a flat concatenation of all module states — sliced by pre-computed index constants (`_IDX_C`, `_IDX_VS`, etc.).

## Tech stack

- **JAX** — core framework, autodiff, `jit`, `vmap`
- **Diffrax** — ODE integration within JAX
- **Optax** — gradient-based optimization
- **Matplotlib** — diagnostics and plotting

## Fitting approach

- **Loss**: MSE between model-predicted and observed eye position/velocity, summed over stimulus conditions
- **Optimizer**: `optax.adam`, typical lr ~1e-3
- **Reparameterization**: use `softplus` for positive TCs (`tau_c`, `tau_i`, `tau_p`, `tau_vs`), `sigmoid` for bounded gains — ensures constraints without clamping
- **Gradients**: flow through `diffrax.diffeqsolve` via reverse-mode autodiff

## Standard stimulus protocols

- **Sinusoidal rotation** (0.05–2.0 Hz, amplitude ~30 deg/s) — probes canal TC at low freq, VOR gain at mid freq, plant at high freq; use for Bode plots
- **Velocity step / head impulse** — exposes integrator leak; eye drifts back after head stops if NI is leaky
- **Constant velocity in dark** — exposes velocity storage TC (~20 s decay)
- **OKN**: constant-velocity scene, then lights out — measures OKN gain and OKAN TC
- **Saccade to static target** — main sequence, latency, endpoint accuracy
- **Saccade to moving target** — staircase behavior, intersaccadic interval

## Diagnostic plots to generate

1. Loss curve vs. optimization step
2. Parameter trajectories vs. step (with ground truth as dashed line)
3. Bode plot — gain and phase vs. frequency, model vs. data overlaid
4. Time-domain overlay — predicted vs. observed eye position per condition
5. Residuals — predicted minus observed, check for systematic misfit

## Conventions

- All angles in **degrees**, angular velocity in **deg/s**
- Eye position = `x_p` (plant state, 3D rotation vector)
- 3D axes: `[yaw, pitch, roll]`
- Head velocity input can be 1D (horizontal only) or 3D
