# Benchmark Reference Papers

## 1. VOR / Velocity Storage (`bench_vor_okr.py`)

- **Raphan, Matsuo & Cohen (1979)** *Exp Brain Res* 35:229–248 — primary target; Fig. 9; VOR, OKN, OKAN; yaw TC ~20 s; `g_vis=0.6`
- **Cohen, Matsuo & Raphan (1977)** *J Neurophysiol* — yaw VS TC (~20 s monkey)
- **Dai et al. (1991)** — roll TC ~2–5 s
- **Hess & Dieringer (1991)** / **Angelaki & Henn (2000)** — pitch TC ~5–10 s

---

## 2. Saccades (`bench_saccades.py`, `bench_clinical_saccades.py`)

- **Bahill et al. (1975)** *Science* — main sequence (`v_peak` vs amplitude)
- **Robinson (1975)** *J Neurophysiol* — local-feedback burst model; NI pulse-step
- **Becker & Jürgens (1979)** *Vision Res* — double-step refractoriness
- **Hepp et al. (1989)** *Exp Brain Res* 75:551–564 — OPN role
- **Zee & Robinson (1979)** *Ann Neurol* 5:207 — ocular flutter

---

## 3. Neural Integrator (`bench_clinical_ni_vs.py`, `bench_clinical_cerebellum.py`)

- **Cannon & Robinson (1985)** *Biol Cybern* — NI leak TC; healthy `tau_i > 20 s`
- **Zee et al. (1980)** *Brain* — gaze-evoked nystagmus; rebound nystagmus
- **Cohen et al. (1992)** *J Neurophysiol* — periodic alternating nystagmus (PAN)
- **Zee et al. (1987)** *J Neurophysiol* — VS not contaminated by saccades

---

## 4. Gravity Estimator (`bench_gravity.py`)

- **Laurens & Angelaki (2011)** *Exp Brain Res* 210:407–422 — primary target; Figs 5 & 6; `K_grav=0.6`, `K_gd=0.5`
- **Howard & Templeton (1966)** — OCR gain (~10° at 90°; `g_ocr` calibration)
- **Fernandez & Goldberg (1976)** — otolith LP adaptation dynamics

---

## 5. Smooth Pursuit (`bench_pursuit.py`)

- **Lisberger & Westbrook (1985)** *J Neurosci* — pursuit integrator model
- **Rashbass (1961)** — step-ramp; catch-up saccade threshold
- **Lisberger (1988)** — Smith predictor / efference copy in pursuit
- **Lisberger & Movshon (1999)** — visual delay `tau_vis ≈ 80 ms`

---

## 6. Listing's Law (`bench_listing.py`)

- **Tweed, Haslwanter & Fetter (1998)** *IOVS* 39:1500 — Listing's plane during saccades
- **van Rijn & van den Berg (1993)** *Exp Brain Res* — Listing's law during pursuit/vergence

---

## 7. Canals / Sensory (`canal.py`, `sensory_model.py`)

- **Goldberg & Fernandez (1971)** *J Neurophysiol* 34:635 — afferent resting discharge (~80 °/s; `FLOOR`)

---

## 8. Plant (`plant_model_first_order.py`)

- **Robinson (1964)** *IEEE Trans Biomed Eng* — first-order plant model; `tau_p ≈ 0.15 s`

---

## 9. Vergence (`bench_vergence.py`)

- **Rashbass & Westheimer (1961)** *J Physiol* 159:361 — vergence TC ~160 ms
- **Cumming & Judge (1986)** *J Physiol* — vergence dynamics

---

## 10. Fixation / Microsaccades (`bench_fixation.py`)

- **Rolfs (2009)** *Neurosci Biobehav Rev* 33:1597 — microsaccade statistics, OU drift model

---

## 11. Clinical Vestibular (`bench_clinical_vestibular.py`)

- **Halmagyi & Curthoys (1988)** — head impulse test (vHIT)
- **Alexander (1912)** *Pflügers Arch* — Alexander's law
- **Leigh & Zee (2015)** *Neurology of Eye Movements*, 5th ed. — clinical reference

---

## 12. Clinical INO / CN Palsies (`bench_clinical_cn_palsies.py`)

- **Bhidayasiri et al. (2000)** *Brain* 123:1241 — INO pathophysiology
- **Zee et al. (1992)** *Ann Neurol* 32:756 — BIMLF syndrome
