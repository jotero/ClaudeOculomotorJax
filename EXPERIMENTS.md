# Experiment Log

Each entry: hypothesis → result → status. Most recent first.

---

## 2026-04-02 — Saccade trigger during pursuit [ACTIVE]

**Problem:** Saccades work well to a static target but fail when the target is moving (e.g., pursuit + saccade). Likely the trigger/refractory logic misbehaves when `e_pos_delayed` is nonzero due to ongoing pursuit motion.

**Hypothesis:** The refractory state `z_ref` or the reset dynamics cause the saccade generator to either fire continuously or suppress itself incorrectly when the error signal has a sustained non-zero component from pursuit.

**What's been tried:**
- Added refractory OPN gate (`z_ref`) to prevent re-triggering — helped for static targets
- Adaptive reset TC (`tau_reset_sac` slow during burst, `tau_reset_fast` after) — helped for static targets
- Direction gate (`gate_dir = relu(ê_error · ê_residual)`) to prevent backward bursts

**Status:** In progress as of latest commit `fef0b3d`

---

## ~2026-03 — Efference copy for slip cancellation

**Hypothesis:** Burst-driven eye movement contaminates retinal slip signal, causing OKR to incorrectly respond to saccade velocity. Need to subtract burst contribution from `e_slip`.

**Approach:** Added `efference_copy.py` — mirrors NI+plant response to `u_burst` only. Computes `w_burst_pred = (x_ni_pc − x_pc)/tau_p + u_burst` to cancel burst from slip.

**Result:** Efference copy works. `e_slip` is now properly gated. Saccades and efference copy work but OKN was broken at that point.

**Status:** Resolved — merged into main architecture (commit `678e7ad`)

---

## ~2026-03 — OKN / OKR

**Hypothesis:** Visual pathway gains (`K_vis`, `g_vis`) were not set correctly; OKAN not sustained.

**Approach:** Iterated on VS visual gain architecture. VS owns both canal PINV_SENS mixing and visual gains. Two components: `K_vis` (state gain → sustains OKAN) + `g_vis` (direct feedthrough → fast OKR onset).

**Result:** Velocity storage good (commit `d35b1b1`). OKR improving (commit `2ac0c6f`).

**Status:** Resolved — stable at current parameter values

---

## ~2026-02 — Saccades (basic)

**Hypothesis:** Robinson local-feedback burst model can be implemented as a differentiable ODE.

**Approach:** `saccade_generator.py` with resettable integrator `x_reset_int`. Three gates: `gate_err`, `gate_res`, `gate_dir`. Burst nonlinearity on residual for main sequence.

**Result:** Saccades look good for static targets (commits `1fe2a62`, `8150230`). Main sequence matches data. Latency ~150 ms with visual delay.

**Status:** Works for static targets. Moving target problem is current focus.

---

## ~2026-02 — 3D extension

**Hypothesis:** Extend from 2D horizontal-only to full 3D (yaw/pitch/roll).

**Result:** Done (commit `0dee6b1`). All axes handled; 1D input still supported (padded to 3D).

**Status:** Complete

---

## Template for new entries

```
## YYYY-MM-DD — [Title]

**Problem/Goal:**

**Hypothesis:**

**Approach:**

**Result:**

**Status:** [In progress | Resolved | Abandoned | Pending]
```
