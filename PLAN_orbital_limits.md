# Plan: Orbital Limits + Target Selector + Dark Nystagmus Trigger

**Goal:** Add eye position limits to the plant, a target selector that combines
visual and orbital-reset error signals, and wire both into the simulator.
The saccade generator itself does not change — it remains transparent to the
source of the motor error command.

**N_TOTAL stays at 270.** The target selector is a pure feedthrough, no states.

---

## Files that change

| File | Change |
|------|--------|
| `oculomotor/models/plant.py` | Add `soft_limit()`, update `step()` output |
| `oculomotor/models/target_selector.py` | **New file** — pure function, no states |
| `oculomotor/models/ocular_motor_simulator.py` | Import, add parameters, rewire SG input |
| `oculomotor/models/efference_copy.py` | Add deferred Option 2 comment only |

**Files that do NOT change:** `saccade_generator.py`, `neural_integrator.py`,
`velocity_storage.py`, `canal.py`, `visual_delay.py`, `retina.py`.

---

## New parameters for THETA_DEFAULT

```python
# ── Orbital limits ────────────────────────────────────────────────────────────
'orbital_limit':  50.0,   # half-range of orbital soft clamp (deg)
                          # plant output = L * tanh(x_p / L), bounded to ±L
'k_orbital':       0.1,   # sigmoid steepness for reset gate in target selector (1/deg)
                          # wider value → smoother transition around orbital_limit

# ── Target selector ───────────────────────────────────────────────────────────
'alpha_reset':     1.0,   # orbital reset gain; e_reset = -alpha_reset * x_p
                          # only active when |x_p| approaches orbital_limit
```

---

## Step 1 — `plant.py`: soft saturation on output

Add a module-level function before `step()`:

```python
def soft_limit(x_p, theta):
    """Soft (differentiable) saturation of eye position to ±orbital_limit (deg).

    Applied to the plant OUTPUT only — the internal state x_p is left unbounded
    so the ODE integrator remains unconstrained.

    Formula:  q_sat = L * tanh(x_p / L)
        - identity near origin: tanh(z) ≈ z for |z| << 1 (error < 1% for |x_p| < 30 deg)
        - saturates monotonically to ±L
        - gradient = sech²(x_p / L) > 0 always — no gradient saturation
    """
    L = theta.get('orbital_limit', 50.0)
    return L * jnp.tanh(x_p / L)
```

Update `step()` to use it:

```python
def step(x_p, u_p, theta):
    dx    = get_A(theta) @ x_p + get_B(theta) @ u_p
    q_eye = soft_limit(x_p, theta)   # bounded reported position
    return dx, q_eye
```

**Test:** For `|x_p| < 30°`, output is < 1% different from old linear plant.
For `x_p = 60°`, output ≈ 48.2° (tanh(1.2) * 50). Run `demo_vor.py` — should
be visually identical to before (VOR eye position stays well within ±30°).

---

## Step 2 — `target_selector.py`: new pure-function module

```python
"""Target selector — upstream feedthrough for the saccade generator.

No internal states.  Pure function, JAX/jit/grad compatible.
Does NOT follow the SSM step() convention — it is a feedthrough, not a module.

Signal flow:
    e_pos_delayed  ──(visual mode)──→ ┐
                                      ├─→ blend ─→ clip ─→ e_cmd ─→ SG
    -alpha_reset * x_p  (reset mode) ─→ ┘

Mode selection (per axis, smooth):
    gate_reset = sigmoid(k_orbital * (|x_p| - orbital_limit))
        gate ≈ 0:  visual mode  (eye well within limits)
        gate ≈ 1:  reset mode   (eye near or past limit)

    e_cmd = (1 - gate_reset) * e_visual + gate_reset * e_reset

Anti-windup clip (per axis):
    e_cmd_clipped = clip(e_cmd, -limit - x_p, +limit - x_p)
    Ensures the commanded landing position (x_p + e_cmd) stays within ±limit.
    jnp.clip gradient is 0 at the boundary — acceptable for fitting.

Parameters
──────────
    orbital_limit  (deg)    half-range of orbital limit, default 50.0
    k_orbital      (1/deg)  sigmoid steepness for reset gate, default 0.1
    alpha_reset    (-)      reset gain, default 1.0
"""

import jax
import jax.numpy as jnp


def select(e_pos_delayed, x_p, theta):
    """Compute motor error command for the saccade generator.

    Args:
        e_pos_delayed : (3,)  delayed retinal position error (deg)
        x_p           : (3,)  raw (unsaturated) plant state (deg)
        theta         : dict

    Returns:
        e_cmd : (3,)  motor error command (deg) — clipped to orbital range
    """
    limit       = theta.get('orbital_limit', 50.0)
    k           = theta.get('k_orbital',      0.1)
    alpha_reset = theta.get('alpha_reset',    1.0)

    # Per-axis reset gate: 0 = visual, 1 = orbital reset
    gate_reset = jax.nn.sigmoid(k * (jnp.abs(x_p) - limit))   # (3,)

    e_visual = e_pos_delayed
    e_reset  = -alpha_reset * x_p

    e_cmd = (1.0 - gate_reset) * e_visual + gate_reset * e_reset

    # Clip so commanded landing stays within ±limit
    e_cmd_clipped = jnp.clip(e_cmd, -limit - x_p, limit - x_p)

    return e_cmd_clipped
```

**Test (standalone):**
```python
# Visual mode: eye at center, 5° target
e_cmd = ts.select([5, 0, 0], [2, 0, 0], theta)   # expect ≈ [5, 0, 0]

# Clip: eye at 48°, 5° visual error → would land at 53° → clip to 2°
e_cmd = ts.select([5, 0, 0], [48, 0, 0], theta)  # expect ≈ [2, 0, 0]

# Reset mode: eye at 55° → e_reset = -55° → clipped back toward center
e_cmd = ts.select([5, 0, 0], [55, 0, 0], theta)  # expect negative (toward center)
```

---

## Step 3 — `ocular_motor_simulator.py`: import + rewire

**3a — Add import** (alongside other model imports):
```python
from oculomotor.models import target_selector as ts
```

**3b — Add parameters to THETA_DEFAULT** (after the `k_acc` block):
```python
    # ── Orbital limits + target selector ─────────────────────────────────────
    'orbital_limit':  50.0,
    'k_orbital':       0.1,
    'alpha_reset':     1.0,
```

**3c — Rewire in ODE_ocular_motor**, just before the SG step call:
```python
# OLD:
dx_sg, u_burst = sg.step(x_sg, e_pos_delayed, theta)

# NEW:
e_cmd          = ts.select(e_pos_delayed, x_p, theta)
dx_sg, u_burst = sg.step(x_sg, e_cmd, theta)
```

Note: `x_p` is the raw plant state slice, already computed in the ODE RHS.
`e_pos_delayed` is the cascade output, also already computed. No new quantities.

**3d — Retina computation stays unchanged:**
`e_pos = retina.target_to_angle(p_target) - q_head - x_p` uses raw `x_p`.
This is correct — we want actual geometric retinal error, not the saturated one.

**3e — Plant step call:** `plant.step()` now returns `soft_limit(x_p)` as its
output. Verify the simulator uses the plant output as the eye position readout
and doesn't double-apply the saturation anywhere.

**Test:** Run all existing demos. For normal stimuli (|x_p| < 30°) output
should be visually identical. The plant tanh error at 30° is < 1%.

---

## Step 4 — `efference_copy.py`: Option 2 deferred comment

Add a comment block near the top of the module docstring:

```
Option 2 (DEFERRED): efference copy anti-windup
────────────────────────────────────────────────
If the target selector's clip fails to prevent x_p from exceeding
orbital_limit (e.g. due to NI drift or overshooting orbital reset saccades),
the efference copy plant state x_pc will also exceed the limit. Since the real
plant reports soft_limit(x_p) but the efference copy tracks the unbounded x_pc,
the slip cancellation identity w_burst_pred ≡ dx_pc breaks slightly.

Fix: apply plant.soft_limit() to x_pc in ec.step() so both saturate identically.
Implement only if simulations show e_slip artifacts during orbital saturation.
The algebraic cancellation identity must be re-derived if this is applied.
```

---

## Implementation order

1. `plant.py` — add `soft_limit`, update `step()` → run `demo_vor.py` to verify no regression
2. `target_selector.py` — write new file → unit-test standalone
3. `ocular_motor_simulator.py` — import, parameters, rewire → run all demos
4. `efference_copy.py` — add deferred comment
5. Write a targeted test: saccade to 80° target, verify landing clips to ≤50°
6. Write a dark rotation test: head rotation, no scene, verify fast phases fire
   periodically with orbital reset driving them back toward center

---

## Option 2 note (deferred)

If x_p saturates past orbital_limit despite the target selector clip, the
efference copy's x_pc will diverge from soft_limit(x_p). Fix: apply
`plant.soft_limit(x_pc, theta)` inside `ec.step()` so both predict the same
saturated position. Deferred because: (a) the clip in target_selector should
prevent this in normal operation, and (b) applying saturation to the efference
copy requires re-deriving the slip cancellation identity. Implement only if
e_slip artifacts appear during orbital saturation in simulations.
