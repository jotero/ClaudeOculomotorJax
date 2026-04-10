# Experiment Log

Each entry: hypothesis → result → status. Most recent first.

---

## 2026-04-02 — Demo review: issues found

**VOR dark / VVOR oscillations:** Not a model bug — saccades are enabled (`g_burst=700`) so the saccade generator fires corrective fast phases as VOR drives the eye away from straight-ahead. The oscillations ARE valid nystagmus. Fix: disable saccades (`g_burst=0`) in VOR/VVOR demos which are meant to show the canal→VS→NI cascade cleanly.

**VVOR residual gaze error (~5°):** Caused by the saccade oscillations contaminating the gaze error signal. Will resolve once saccades are disabled in that demo.

**Saccade stopping (smooth pursuit, OKN debug, saccade VOR) — root cause identified:** During the refractory period between saccades, `x_reset_int` decays toward zero via `A_reset`. For a moving target this is wrong: when the next saccade fires, the copy integrator has partially reset so it misjudges the residual → wrong burst amplitude → oscillation or undershoot. Saccade VOR shows the extreme: after the first burst, refractory locks out subsequent saccades while motor error grows to 90°+.

**Main sequence slightly below reference at small amplitudes:** `e_sat_sac=7°` makes the exponential saturation compress slowly; small saccades fall below the `700*(1-exp(-A/7))` reference. May need a lower `e_sat_sac` or verify `g_burst` scaling.

**Status:** VOR/VVOR demo fix is trivial. Saccade stopping is the core problem (see entry below).

---

## 2026-04-10 — Reliable saccade triggering: lessons learned [RESOLVED]

**Problem:** Catch-up saccades to moving targets produced a continuous ~5–10 deg/s drift instead of a clean staircase. Static-target saccades worked fine. The system would either fire a single saccade and drift, or fire a continuous burst that never turned off.

**Root cause (the continuous-burst fixed point):**
When the target moves during a saccade, the live residual `e_pos_delayed − x_copy` never decreases to zero because the target is always ahead. This kept `gate_res ≈ 1`, so the OPN charge signal stayed near zero, z_ref stabilized at ~0.10–0.20, and the burst never stopped. Every fixing attempt that worked on the residual signal itself was attacking a symptom.

**Lesson 1 — The residual must be ballistic (sample-and-hold)**

The correct fix is architectural: the burst must run against a *frozen* copy of the error at saccade onset, not against the live visual signal. Added `e_held` (3 states): tracks `e_pos_delayed` between saccades, freezes when `z_sac=1`. The Robinson residual is now `e_held − x_copy`, which decreases monotonically to zero as `x_copy` integrates toward the held target, regardless of where the target has moved during the saccade.

This eliminates every moving-target fixed point at the source.

**Lesson 2 — Refractory charging must not use gate_opn**

Original charge formula: `charge = gate_err × gate_opn × (1−gate_res)`. The problem: as z_ref rises, gate_opn falls, which cuts the charge, which prevents z_ref from rising further — a self-defeating loop. z_ref stabilizes at whatever value makes `gate_opn × charge = z_ref/τ_ref`, typically ~0.20, never reaching the release threshold.

Fix: `charge = z_sac × (1−gate_res)`. Drive charge from the saccade latch itself (which is definitively 0 or 1), not from a soft gate on z_ref. This breaks the loop entirely: once `gate_res → 0` at burst end, charge = z_sac = 1, and z_ref charges unconditionally to near 1.

**Lesson 3 — z_sac release threshold must be >> OPN gate threshold**

`threshold_sac_release = 0.4`, `threshold_ref = 0.1`. If both were 0.1, the equation `dz_sac = 0` gives `z_sac = gate_opn` at steady state — a circular relabeling of the same fixed point. The high release threshold means z_sac only releases *after* z_ref has fully charged, which only happens *after* the burst has truly stopped.

**Lesson 4 — A rise-to-bound accumulator solves two problems at once**

Problem 1: The 40-stage visual cascade takes ~120 ms to settle after a target step. Without any delay, `gate_err` fires as soon as cascade output crosses the 0.5° threshold (sometimes at 5–10% cascade settling for large targets), z_sac freezes e_held at that partial value, and the burst undershoots. This compresses the main sequence — peak velocity appears amplitude-independent.

Problem 2: Noise robustness — brief retinal error spikes can trigger spurious saccades.

Fix: added `z_acc` (9th state), a leaky integrator of `gate_err × gate_opn` with τ_acc = 80 ms. z_sac only fires when `z_acc > threshold_acc = 0.5`, requiring ~50–80 ms of sustained suprathreshold error. During this accumulation window, `z_sac = 0` so `e_held` keeps tracking the settling cascade. By the time z_sac fires, e_held holds 90–99% of the true target error across all amplitudes (vs. 5–74% without the accumulator).

**Final state vector:** `[x_copy(3) | z_ref(1) | e_held(3) | z_sac(1) | z_acc(1)]` — 9 states

**Result:** Clean saccade staircase for all target velocities (1–20 deg/s). Main sequence follows the nonlinear reference (700*(1−exp(−A/7))). Refractory period ~150–180 ms. All existing demos (VOR, OKR, efference copy) unaffected.

**Status:** Resolved — 2026-04-10

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
