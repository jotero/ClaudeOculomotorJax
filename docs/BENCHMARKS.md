# OculomotorJax — Benchmark Specification

Each section lists the expected behaviors and numerical targets.
Literature citations give the primary source for each criterion.
Scripts in `docs/` produce two figure types per section:
- **behavior** — stimulus + eye output only (paper-style replication)
- **cascade** — internal signal chain (debugging / mechanistic)

---

## 1. Saccades (`docs/saccades.py`)

### 1a. Main Sequence
**Expected:**
- Peak velocity follows the hyperbolic main sequence:
  `v_peak ≈ 700 × (1 − exp(−A / 7))` deg/s
- Amplitude range 0.5–20°; saturates near 600–700 deg/s for large saccades
- Duration increases with amplitude (roughly linear for A > 5°)
- **Pass criterion:** All data points within ±20% of the reference curve

**Citations:** Bahill et al. (1975) *Science*; Robinson (1975) *J Neurophysiol*

### 1b. Oblique Saccades
**Expected:**
- Oblique saccades travel in a straight line in 2D eye-position space
- Horizontal and vertical components are synchronised (begin and end together)
- The slower component stretches to match the faster — no "L-shaped" path

**Citations:** Smit et al. (1990) *J Neurophysiol*; van Gisbergen et al. (1985)

### 1c. Saccade Refractoriness (Double-Step Paradigm)
**Expected:**
- Target steps to 10° (first saccade begins); then steps to 20° at varying ISI (0–400 ms after first step)
- At ISI < ~100 ms: second saccade cannot fire (refractory) — eye goes directly to updated target
- At ISI > ~150 ms: both saccades occur separately; ISI is preserved
- **Pass criterion:** no double-firing within 100 ms of first saccade offset

**Citations:** Becker & Jürgens (1979) *Vision Res*; Carpenter (1988) *Movements of the Eyes*

### 1d. Saccade Signal Cascade (Internal, cascade figure)
**Expected signal flow:**
1. Target step → visual cascade (40-stage delay, ~120 ms)
2. Cascade output → z_acc rises (accumulator), crosses threshold → z_sac latches
3. e_held freezes at cascade value; e_res = e_held − x_copy drives burst
4. x_copy integrates toward e_held; burst decays → e_res → 0
5. z_ref charges (refractory), then z_sac releases
6. Plant: eye velocity ≈ burst waveform (1st-order, tau_p = 0.005 s)

---

## 2. VOR / OKR (`docs/vor_okr.py`)

### 2a. Raphan Fig. 9 Replication
Three conditions with 30 deg/s stimulus, each 30 s on + 50 s coast/dark.

**VOR in dark (panels A/B):**
- Slow-phase velocity (SPV) ≈ −30 deg/s at onset (compensatory)
- SPV decays during sustained rotation: fast component (canal TC ~5 s), then slow (VS TC ~20 s)
- Post-rotatory nystagmus after rotation ends; SPV decays to 0 with TC ≈ 15–25 s
- **Pass criterion:** post-rotatory TC fit within 10–30 s

**OKN + OKAN (panels C/D):**
- During scene motion: SPV builds to ≈ +30 deg/s (gain ≈ 1.0)
- Sawtooth nystagmus visible in raw eye position (fast phases every 1–3 s)
- After scene off (OKAN): SPV decays exponentially with TC ≈ 15–25 s
- **Pass criterion:** steady-state OKN gain > 0.75; OKAN TC within 10–30 s

**VVOR (panels E/F):**
- During rotation in lit stationary scene: SPV ≈ −head velocity throughout
- VOR gain remains ≈ 1.0 despite canal adaptation (OKR compensates)
- Post-rotation: SPV decays quickly once rotation stops and scene is still present
- **Pass criterion:** mean VVOR gain during rotation > 0.85

**Citations:** Raphan, Matsuo & Cohen (1979) *Exp Brain Res* 35:229–248

### 2b. OKN Nystagmus Zoom
**Expected:** Clear sawtooth waveform in first 10 s of OKN; slow phases ~30 deg/s,
fast phases resetting to near-center periodically.

### 2c. VOR Time-Constant Comparison
**Expected:**
- Without VS (tau_vs ≈ 0): SPV decays with canal TC ~5 s
- With VS (tau_vs = 20 s): SPV decay extended to ~20 s
- TC ratio (VS / no-VS) ≈ 4×

**Citations:** Cohen, Matsuo & Raphan (1977) *J Neurophysiol*

### 2d. VOR/OKR Signal Cascade (Internal, cascade figure)
**Expected signal flow (VOR):** head → canal afferents → VS → NI → burst (fast phases)
→ plant → eye
**Expected signal flow (OKR):** scene → retinal slip → visual delay → VS → NI → plant

---

## 3. Tilt / Translation (`docs/tilt_translation.py`)

### 3a. OVAR (Off-Vertical Axis Rotation) — *placeholder*
**Expected:** During rotation around a tilted axis, otolith drives periodic modulation
of nystagmus SPV. Canal component constant; otolith adds sinusoidal bias.

**Citations:** Raphan et al. (1981); Angelaki & Hess (1994)

### 3b. VOR Tilt Suppression — *placeholder*
**Expected:** Static head tilt activates the otolith-ocular reflex (ocular counterroll),
rotating the eyes opposite to the head tilt. In primates this is partially suppressed (~10–15%
counterroll per degree of tilt). Requires gravity estimator + otolith model working correctly.

---

## 4. Smooth Pursuit (`docs/pursuit.py`)

### 4a. Velocity Range (step-ramp)
**Expected:**
- Ramp at 5 deg/s: pursuit tracks within ±1 deg (no catch-up saccade needed)
- Ramp at 10 deg/s: small initial lag, one catch-up saccade, then tracking
- Ramp at 20 deg/s: two or more catch-up saccades, eventual tracking
- Ramp at 40 deg/s: pursuit saturates, multiple catch-up saccades throughout
- **Pass criterion:** steady-state gain (t > 1.5 s) > 0.8 at 5–10 deg/s

**Citations:** Lisberger & Westbrook (1985) *J Neurosci*; Rashbass (1961)

### 4b. Sinusoidal Pursuit (Bode)
**Expected:**
- 0.3 Hz, 20 deg/s peak: gain ≈ 0.8–1.0, phase lag < 30°
- Gain decreases and phase lag increases with frequency above ~0.5 Hz
- Smooth pursuit, not saccadic — no obvious fast phases at low frequency

### 4c. Pursuit Signal Cascade (Internal, cascade figure)
**Expected:** velocity error → pursuit integrator → NI → plant;
efference copy cancels saccadic contamination of retinal velocity signal

---

## 5. Vergence (`docs/vergence.py`)

### 5a. Symmetric Vergence — *placeholder*
**Expected:** Both eyes converge symmetrically on a near target (< 40 cm).
Vergence TC ≈ 150–200 ms. No version change.

### 5b. Asymmetric Vergence — *placeholder*
**Expected:** Target moves laterally with simultaneous depth change.
Version component (conjugate) separates from vergence component.

**Citations:** Mays (1984); Cumming & Judge (1986)

---

## 6. Fixation (`docs/fixation.py`)

### 6a. Noise Source Comparison
**Expected:**
- **Noiseless:** eye stays at 0° throughout
- **Canal noise (σ = 3 deg/s):** slow drift, rare corrective microsaccades
- **Retinal pos noise (σ = 0.3°):** OU drift accumulates, microsaccades fire
  when position error crosses threshold (~0.5–1°)
- **Retinal vel noise (σ = 5 deg/s):** smooth-pursuit-like drift

**Citations:** Rolfs (2009) *Neurosci Biobehav Rev* 33:1597–1627

---

## Numerical Pass/Fail Summary

| Benchmark | Target | Pass Criterion |
|---|---|---|
| Main sequence peak vel (20°) | 700×(1−e^{−20/7}) ≈ 660 deg/s | 550–750 deg/s |
| Post-rotatory VOR TC | 15–25 s | 10–30 s |
| OKN steady-state gain | ≈ 1.0 | > 0.75 |
| OKAN TC | 15–25 s | 10–30 s |
| VVOR gain during rotation | ≈ 1.0 | > 0.85 |
| Pursuit gain (5 deg/s, SS) | ≈ 1.0 | > 0.80 |
| Saccade ISI (refractory) | > 150 ms | > 100 ms |
