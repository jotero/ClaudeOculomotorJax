# VOR + Neural Integrator Model — Implementation Spec

## Overview

Build a minimal differentiable model of the vestibulo-ocular reflex (VOR) in JAX. The model takes head velocity as input and produces eye position as output, passing through: semicircular canal dynamics → VOR gain → neural integrator → eye plant. All parameters are fittable via gradient descent. The first milestone is parameter recovery from synthetic data.

## Tech Stack

- **JAX** (core framework, autodiff, jit, vmap)
- **Diffrax** (ODE integration within JAX)
- **Optax** (gradient-based optimisation)
- **Matplotlib** (diagnostics and plotting)

Install: `pip install jax diffrax optax matplotlib`

## Repo Structure

```
oculomotor/
  models/
    __init__.py
    canal.py          # Semicircular canal transfer function
    integrator.py     # Neural (velocity-to-position) integrator
    plant.py          # Eye plant dynamics
    vor.py            # Full VOR model composing the above
  sim/
    __init__.py
    stimuli.py        # Head motion stimulus generation
    synthetic.py      # Generate synthetic eye movement data
  fitting/
    __init__.py
    loss.py           # Loss functions
    optimize.py       # Parameter recovery / fitting pipeline
  tests/
    test_param_recovery.py   # End-to-end parameter recovery test
  scripts/
    run_recovery.py          # Main script: generate data, fit, plot
```

## Mathematical Model

### Signal Flow

```
head_velocity(t) → [Canal] → canal_signal(t) → [×VOR gain] → eye_velocity_command(t)
                                                                        ↓
                                            eye_position(t) ← [Plant] ← [Neural Integrator] → eye_position_command(t)
```

### 1. Semicircular Canal (canal.py)

The canals act as a high-pass filter on head velocity, approximated as:

```
τ_c · d(canal)/dt = -canal + head_velocity
```

where `canal` is the canal afferent signal. This is a first-order high-pass with time constant τ_c. At frequencies well above 1/τ_c the canal output faithfully encodes head velocity; at very low frequencies it decays.

**State variable:** `canal(t)` (scalar, deg/s equivalent)

**Parameter:**

| Name  | Symbol | Typical | Range       | Unit |
|-------|--------|---------|-------------|------|
| Canal time constant | τ_c | 5.0 | [2.0, 10.0] | s |

### 2. VOR Gain (applied in vor.py)

A simple multiplicative gain on the canal signal, with sign inversion (VOR moves eyes opposite to head):

```
eye_velocity_command = -g_vor × canal
```

**Parameter:**

| Name     | Symbol | Typical | Range       | Unit     |
|----------|--------|---------|-------------|----------|
| VOR gain | g_vor  | 1.0     | [0.3, 1.5]  | unitless |

In healthy subjects g_vor ≈ 1.0. Bilateral vestibular loss → low gain. Cerebellar lesions can produce gain miscalibration.

### 3. Neural Integrator (integrator.py)

The neural integrator converts the velocity command into a position command. A perfect integrator holds eye position indefinitely; a leaky integrator drifts back toward centre:

```
τ_i · d(eye_pos_cmd)/dt = -eye_pos_cmd + eye_velocity_command × τ_i
```

Rearranged:

```
d(eye_pos_cmd)/dt = -eye_pos_cmd / τ_i + eye_velocity_command
```

When τ_i → ∞, this is a perfect integrator (d/dt = eye_velocity_command). When τ_i is finite and small, the integrator leaks — the signature of gaze-evoked nystagmus.

**State variable:** `eye_pos_cmd(t)` (scalar, deg)

**Parameter:**

| Name                     | Symbol | Typical | Range        | Unit |
|--------------------------|--------|---------|--------------|------|
| Integrator time constant | τ_i    | 25.0    | [3.0, 100.0] | s    |

Healthy: τ_i > 20s (effectively perfect for short tasks). Cerebellar patients: τ_i can be as low as 3–5s.

### 4. Eye Plant (plant.py)

The oculomotor plant (extraocular muscles + globe) is modelled as a first-order low-pass filter:

```
τ_p · d(eye_pos)/dt = -eye_pos + eye_pos_cmd
```

This smooths the motor command. In a more complete model this would be second-order (Robinson's model), but first-order is sufficient for VOR fitting at moderate frequencies.

**State variable:** `eye_pos(t)` (scalar, deg)

**Parameter:**

| Name              | Symbol | Typical | Range        | Unit |
|-------------------|--------|---------|--------------|------|
| Plant time constant | τ_p  | 0.15    | [0.05, 0.5]  | s    |

### 5. Full System (vor.py)

The combined ODE state vector is:

```
x = [canal, eye_pos_cmd, eye_pos]   (3 state variables)
```

The parameter vector θ is:

```
θ = [τ_c, g_vor, τ_i, τ_p]   (4 parameters)
```

The dynamics function `f(t, x, θ, head_vel(t))` returns `dx/dt`:

```python
dx0/dt = (-x[0] + head_vel(t)) / τ_c                    # canal
dx1/dt = -x[1] / τ_i + (-g_vor * x[0])                  # integrator
dx2/dt = (-x[2] + x[1]) / τ_p                            # plant
```

### Output

The observable is `eye_pos(t)` = `x[2]`. In a clinical setting this is measured by video-oculography.

## Stimulus Protocol (stimuli.py)

### Sinusoidal Rotation

For parameter recovery, use sinusoidal head velocity at multiple frequencies:

```python
frequencies = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]   # Hz
amplitude = 30.0   # deg/s peak head velocity
duration = 20.0     # seconds per frequency (enough cycles for steady-state)
```

Each frequency probes a different aspect:
- Low freq (0.05 Hz): exposes canal time constant (gain drops, phase leads)
- Mid freq (0.2–0.5 Hz): exposes VOR gain cleanly
- High freq (2 Hz): exposes plant dynamics

### Step (Head Impulse)

A velocity step (quick head rotation, then hold still):

```python
head_vel(t) = amplitude   if t < step_duration
              0           otherwise
```

This directly exposes the integrator leak: after the head stops, a leaky integrator causes the eye to drift back (gaze-evoked nystagmus pattern).

## Synthetic Data Generation (synthetic.py)

1. Pick ground-truth parameters θ_true (e.g., the "typical" values above)
2. For each stimulus, integrate the ODE using Diffrax (use `diffrax.Tsit5` or `diffrax.Dopri5`)
3. Sample eye_pos at a realistic sample rate (e.g., 500 Hz, typical of video-oculography)
4. Add Gaussian observation noise: `eye_pos_observed = eye_pos + N(0, σ_obs)` with σ_obs ≈ 0.3 deg

Return: `(t, head_vel, eye_pos_observed)` for each stimulus condition.

## Loss Function (loss.py)

Mean squared error between model-predicted and observed eye position, summed over all stimulus conditions:

```python
def loss(θ, stimuli, observations):
    total = 0.0
    for (t, head_vel, eye_obs) in zip(stimuli, observations):
        eye_pred = simulate(θ, t, head_vel)  # run ODE with Diffrax
        total += jnp.mean((eye_pred - eye_obs) ** 2)
    return total / len(stimuli)
```

Gradients via `jax.grad(loss)`. The ODE solve (Diffrax) supports reverse-mode autodiff through the solver.

## Optimisation (optimize.py)

```python
import optax

optimizer = optax.adam(learning_rate=1e-3)

# Work in unconstrained space using softplus / sigmoid reparameterisation:
#   τ_c = softplus(φ_c)       → ensures τ_c > 0
#   g_vor = sigmoid(φ_g) * 2  → ensures g_vor ∈ (0, 2)
#   τ_i = softplus(φ_i)       → ensures τ_i > 0
#   τ_p = softplus(φ_p)       → ensures τ_p > 0

# Initialise φ from perturbed values (NOT at the true parameters)
# Run for N steps, log loss, track parameter trajectories
```

## Parameter Recovery Test (test_param_recovery.py)

The critical first test:

1. Generate synthetic data from θ_true = [5.0, 1.0, 25.0, 0.15]
2. Initialise θ_init at perturbed values, e.g., [3.0, 0.7, 10.0, 0.3]
3. Run optimisation for ~2000 steps
4. **Pass criterion:** all recovered parameters within 10% of ground truth

Run this from multiple random initialisations to check for local minima.

## Diagnostics Script (scripts/run_recovery.py)

This script should produce the following plots:

1. **Loss curve** — loss vs. optimisation step (should decrease smoothly)
2. **Parameter trajectories** — each parameter vs. step, with ground-truth as dashed horizontal line
3. **Bode plot** — gain and phase vs. frequency for both ground-truth and fitted model, overlaid. This is the key clinical summary.
4. **Time-domain overlay** — for each stimulus condition, plot observed and predicted eye position on the same axes
5. **Residuals** — predicted minus observed, to check for systematic misfit

## Implementation Notes

- Use `jax.jit` on the loss function and the simulate function.
- Use `jax.vmap` if batching over multiple stimulus conditions simultaneously.
- For the ODE solver, use `diffrax.diffeqsolve` with `diffrax.Tsit5()` solver and `diffrax.PIDController` stepsize controller (rtol=1e-5, atol=1e-7).
- The head velocity input is a time-varying external signal. Pass it to the ODE vector field as an interpolated function using `diffrax.LinearInterpolation` over a discrete time series.
- Keep the dt for saving output at 1/500 s (matching the simulated sample rate).
- Use `jax.tree_util` for parameter pytrees if you want clean nesting later.
- All modules should be pure functions compatible with `jit` and `grad` — no side effects, no global state.

## Future Extensions (do NOT implement yet)

These are noted here to inform architectural decisions — keep the code modular enough to add:

- **Saccade generator**: pulse-step controller on top of the integrator
- **Smooth pursuit**: velocity-matching feedback loop
- **Antisaccade controller**: cortical inhibition layer
- **Second-order plant**: Robinson's model for more realistic high-frequency dynamics
- **MuJoCo/MJX plant**: replace the transfer-function plant with a biomechanical model
- **Perceptual model**: visual acuity prediction from retinal slip
- **Multiple patients**: vmap over a batch of parameter vectors
- **Simulation-based inference**: replace gradient descent with amortised neural posterior estimation for full Bayesian parameter posteriors

## First Session Deliverable

At the end of the first Claude Code session, I should be able to run:

```bash
python scripts/run_recovery.py
```

and see:
- Successful parameter recovery (printed values close to ground truth)
- Five diagnostic plots saved to `outputs/`
- Clean, modular code I can extend in subsequent sessions
