# A Differentiable Computational Model of the Primate Oculomotor System with a Large Language Model Interface for Clinical Neurology

**Authors:** [TBD]

**Affiliations:** [TBD]

**Corresponding author:** [TBD]

**Keywords:** oculomotor system, computational model, vestibulo-ocular reflex, saccades, smooth pursuit, nystagmus, large language model, clinical simulation, medical education, synthetic data

---

## Abstract

**Background and Objectives:** Eye movement abnormalities are among the most diagnostically informative signs in clinical neurology, yet interpreting them requires deep familiarity with the underlying neural circuitry. We present a biophysically grounded, differentiable computational model of the primate oculomotor system and a large language model (LLM) interface that allows clinicians and researchers to simulate eye movements in health and disease using plain English descriptions.

**Methods:** The model is implemented in JAX and integrates six subsystems: a bilateral semicircular canal array, an otolith model, a bilateral velocity storage network, a leaky neural integrator, a Robinson local-feedback saccade generator, and a smooth pursuit integrator with Smith predictor. All subsystems include bilateral push-pull architectures with null-adaptation states. A natural language interface powered by the Claude LLM translates clinical scenario descriptions into structured simulation parameters, executes the simulation, and returns annotated figures.

**Results:** The model accurately reproduces the cardinal oculomotor behaviors: VOR with velocity storage (time constant ~15–20 s), optokinetic nystagmus and afternystagmus, the saccadic main sequence, oblique saccades, smooth pursuit with catch-up saccades, and fixational eye movements with microsaccades. Clinical simulations reproduce: peripheral and central vestibular lesions (spontaneous nystagmus, video head impulse test, rotary chair); cerebellar gaze-holding disorders (gaze-evoked nystagmus, rebound nystagmus, impaired pursuit, nodulus/uvula lesions); cranial nerve palsies and internuclear ophthalmoplegia; and integrator/velocity storage disorders (Alexander's law, extended OKAN). The LLM interface correctly interprets diverse clinical prompts and generates simulation outputs suitable for teaching and hypothesis generation.

**Discussion:** This platform bridges computational neuroscience and clinical neurology by enabling simulation-based reasoning about oculomotor disorders without programming expertise. It serves two immediate purposes: as a teaching tool that generates realistic oculomotor findings for any clinical scenario on demand, and as a source of labeled synthetic training data for machine learning models of eye movement classification. Its differentiable architecture additionally supports future patient-specific parameter fitting from recorded eye movement data.

---

## Introduction

Eye movement examination is a cornerstone of the neurological assessment. Abnormalities of the vestibulo-ocular reflex (VOR), saccades, smooth pursuit, and gaze-holding provide localizing information across a wide range of conditions, from peripheral vestibular neuritis to brainstem stroke, cerebellar degeneration, and internuclear ophthalmoplegia. Yet mapping from lesion location to expected oculomotor phenotype is non-trivial: the same symptom — nystagmus — can arise from lesions at multiple sites through distinct mechanisms, and the pattern of abnormality depends on subtle interactions among subsystems that are difficult to reason about intuitively.

Computational models of the oculomotor system have a long history, beginning with Robinson's classical work on the neural integrator and saccade generator [CITATION], and extending to the Raphan-Cohen velocity storage model [CITATION], the Galiana-Outerbridge bilateral model [CITATION], and subsequent formulations. These models have provided deep mechanistic insight but have remained largely inaccessible to the clinical community: they require programming expertise, are not easily adapted to specific patient profiles, and rarely produce outputs in the clinical idiom.

Recent advances in differentiable programming and large language models create an opportunity to bridge this gap. Differentiable simulators allow model parameters to be fit to patient data via gradient descent, enabling personalized simulation. LLMs can translate natural language clinical descriptions into structured computational inputs, lowering the barrier to simulation without sacrificing biophysical fidelity.

Here we describe ClaudeOculomotorJax, a JAX-based computational model of the primate oculomotor system that (1) integrates the major oculomotor subsystems in a unified, biophysically grounded framework; (2) faithfully reproduces both normal oculomotor behavior and a broad range of clinical oculomotor syndromes; (3) provides a natural language interface that allows users to specify clinical scenarios in plain English and receive annotated simulation outputs within seconds; and (4) serves as a scalable source of labeled synthetic eye movement data for training and validating machine learning-based oculomotor classifiers.

---

## Methods

### Model Architecture

The model consists of five functional stages connected in a closed-loop architecture (Figure 4): sensory encoding, visual processing, brainstem oculomotor circuits, the ocular motor plant, and efference copy. All components are implemented as linear state-space models (SSMs):

> dx/dt = A(θ)·x + B(θ)·u,  y = C·x + D(θ)·u

where x is the state vector, u is the input, θ is a parameter vector, and y is the output. Nonlinearities (canal rectification, saccade gating, pursuit saturation) are layered on the linear core. The system is implemented in JAX [CITATION] using the Diffrax library [CITATION] for ODE integration (Heun solver, dt = 1 ms), making the entire simulation differentiable for future parameter fitting. The total model state comprises 1,140 variables: 978 sensory states, 156 brain states, and 6 plant states.

#### Semicircular Canal Array

Head rotation is transduced by a bilateral array of six semicircular canals modeled as the Steinhausen torsion-pendulum SSM [CITATION], with cupula time constant τ_c ≈ 0.003 s and adaptation time constant τ_s = 5 s. The output passes through a smooth push-pull rectification nonlinearity to produce canal afferent firing rates. The 5-second adaptation time constant underlies the well-known decay of the VOR during sustained constant-velocity rotation.

#### Otolith Model

Linear acceleration is encoded by a bilateral otolith model (utricle and saccule) implemented as a low-pass adapted system (6 states). The otolith signal contributes a gravity-referenced correction to the velocity storage and canal inputs.

#### Visual Processing

A binocular retinal model computes retinal slip velocity, target position error, target velocity, and a binary target-in-visual-field indicator from eye and world-frame kinematics. Each signal is passed through a companion-form delay cascade (120 ms for velocity and position signals, 40 ms for the visual-field gate), comprising 480 states per eye. The visual-field gate prevents the saccade generator from responding to targets outside the foveal field of view.

#### Velocity Storage

Velocity storage (VS) is modeled as a bilateral push-pull network of vestibular nucleus (VN) populations implementing the Raphan-Cohen mechanism [CITATION]. The net output (x_L − x_R) corresponds to the classical scalar VS signal; the bilateral formulation additionally supports simulation of lateralized lesions and spontaneous nystagmus. OKR gain K_vis couples retinal slip to VS, mediating optokinetic nystagmus (OKN) and optokinetic afternystagmus (OKAN). A null-adaptation state (time constant τ_vs_adapt = 600 s) models the slow shift of VS bias during prolonged optokinetic stimulation, underlying extended OKAN adaptation. Efference copy from the saccade generator is subtracted at the VS input to prevent saccadic contamination.

#### Neural Integrator

The neural integrator (NI) is a bilateral leaky integrator following the Robinson formulation [CITATION], with separate left and right neural populations (NPH/INC) and mutual inhibition. A null-adaptation state (time constant τ_ni_adapt = 20 s) models slow bias shifts during asymmetric activity, producing rebound nystagmus after sustained eccentric gaze and Bruns nystagmus in the setting of asymmetric tone. In the healthy symmetric case, net output x_L − x_R is identical to the classical scalar integrator.

#### Saccade Generator

The saccade generator implements a Robinson local-feedback burst mechanism [CITATION] with an omnipause neuron (OPN) gate (9 states). The implementation tracks: a desired displacement register (e_held) that accumulates position error and is cleared by saccade execution; a burst accumulator (z_sac) whose rise-to-bound dynamics set saccade duration; and an acceleration state (z_acc) that shapes the velocity profile. Target selection (centering saccades, orbital clip) is handled internally using the NI state as a proxy for eye position. This formulation reproduces the saccadic main sequence, accurate intersaccadic intervals, oblique saccades with synchronized components, and double-step refractoriness.

#### Smooth Pursuit

Smooth pursuit is driven by a leaky pursuit integrator receiving retinal velocity error (vel_delayed), gated by the target-present signal. A Smith predictor, implemented via the efference copy delay cascade, cancels the sensory consequence of the motor command, enabling pursuit to lock onto moving targets without steady-state slip. Catch-up saccades fire when position error exceeds the saccade threshold during ramp pursuit.

#### Efference Copy

A 120-state delay cascade models the forward copy of burst commands back to the velocity storage input, implementing the Raphan-Cohen efference copy pathway that prevents saccadic transients from contaminating VS and the OKR response.

#### Ocular Motor Plant

The plant is modeled as a Robinson first-order viscoelastic system [CITATION] with time constant τ_p = 0.15 s, converting the pulse-step motor command into eye rotation. The plant interface is abstracted to allow higher-order biomechanical models as drop-in replacements.

### Sensory Noise

Four independent noise sources can be activated: canal afferent noise (σ_canal, Gaussian), retinal slip noise (σ_slip), retinal position drift (σ_pos, Ornstein-Uhlenbeck process), and retinal velocity noise (σ_vel). Noise arrays are pre-generated before ODE integration and passed as interpolated inputs, preserving differentiability.

### Parameter Values

Default parameters were set to match published physiological data (Table 1). Key values: canal adaptation time constant τ_s = 5 s; velocity storage time constant τ_vs = 15 s; neural integrator time constant τ_i = 25 s; saccade burst gain g_burst = 700 deg/s; visual gain K_vis = 1.5. Lesion simulations were produced by modifying individual parameters as described below.

### Clinical Lesion Simulations

**Peripheral vestibular lesions** were simulated by unilaterally reducing canal gains (complete loss: gain = 0; partial loss: intermediate values). **Cerebellar lesions** were simulated by: (FL/PFL) reducing τ_i and disrupting NI null-adaptation; (Nodulus/Uvula) modifying VS null-adaptation. **Cranial nerve palsies** were simulated by reducing motor command gain for specific directions. **INO** was modeled by adding asymmetric adduction delay. **Integrator disorders** were simulated by adjusting τ_i across a clinical spectrum.

### Large Language Model Interface

The LLM interface (Figure 3) converts a free-text clinical scenario to a simulation in three stages:

**Stage 1 — Schema extraction.** The user's description is passed to the Claude LLM (Anthropic, claude-opus-4-6) via the Anthropic API using forced tool-call mode, which extracts a structured `SimulationScenario` (Pydantic schema) containing patient parameters, head motion trajectory, visual scene configuration, and target configuration.

**Stage 2 — Simulation.** Stimulus arrays are constructed from the scenario and passed to the JAX simulator, which integrates the ODE and returns the full state trajectory.

**Stage 3 — Figure generation.** A matplotlib figure is assembled from the specified panel configuration (eye position/velocity, head velocity, retinal slip, VS state, NI state, burst activity) and saved with metadata. A FastAPI web server exposes the pipeline via HTTP, with a feedback log (correctness flag + free-text comment) for iterative improvement.

---

## Results

### Healthy Oculomotor Behavior (Figure 1)

**VOR, velocity storage, and VVOR.** During constant-velocity rotation in the dark (60 deg/s step, 40 s), eye velocity matches head velocity (VOR gain ≈ 0.95) and decays with time constant ~17 s — substantially longer than the canal adaptation time constant of 5 s — confirming that velocity storage extends the effective VOR time constant. In VVOR (rotation in a stationary lit world), OKR compensates for the decaying canal signal and gaze remains stable throughout the rotation, replicating the Raphan et al. Figure 9 result [CITATION].

**OKN and OKAN.** During full-field scene motion (30 deg/s), steady-state OKN gain approaches unity with a sawtooth nystagmus waveform. After scene offset, OKAN persists with time constant ~20 s, driven by velocity storage discharge.

**Saccadic main sequence.** Target steps of 2–40 degrees elicit saccades whose peak velocity follows v_peak ≈ 700 · (1 − exp(−A/7)) deg/s, matching Bahill et al. normative data [CITATION]. Oblique saccades to targets displaced in both horizontal and vertical directions are executed as straight trajectories with synchronized components. Double-step stimuli (target displaced during an ongoing saccade) elicit appropriate corrective saccades after the intersaccadic refractory interval (~150–200 ms).

**Smooth pursuit.** A 20 deg/s ramp target is tracked with steady-state gain ≈ 0.95 after an initial catch-up saccade. Sinusoidal targets (0.3–2 Hz, 10 deg amplitude) are tracked with gain ≥ 0.8 and phase lag < 20 deg across the physiological frequency range, consistent with published pursuit bandwidth data.

**Fixational eye movements.** With canal noise (σ_canal = 2 deg/s) and retinal position OU drift (σ_pos = 0.3 deg, τ_drift = 0.3 s), the model produces sporadic microsaccades at physiologically appropriate intervals (< 3/s) with amplitudes < 1 deg, and slow drift consistent with fixational instability data. Retinal velocity noise (σ_vel) produces smooth slow-phase drift resembling pursuit-related fixational instability.

### Clinical Lesion Simulations (Figure 2)

**Peripheral vestibular lesions.** Complete unilateral canal gain loss produces spontaneous nystagmus beating toward the intact side. The video head impulse test (vHIT) reveals an absent VOR with corrective saccade following impulses toward the lesioned side, with an intact response on the other side — the hallmark of unilateral vestibular neuritis. Partial lesions produce intermediate deficits graded with lesion severity. Rotary chair testing in the dark shows asymmetric VOR gain with directional preponderance consistent with canal paresis.

**Cerebellar lesions.** Flocculus/paraflocculus (FL/PFL) lesions, implemented as a combination of reduced τ_i and disrupted NI null-adaptation, reproduce three co-occurring deficits: (i) gaze-evoked nystagmus, with drift velocity scaling linearly with eccentricity (Alexander's law); (ii) rebound nystagmus on return to primary position; and (iii) impaired smooth pursuit. Nodulus/uvula lesions, implemented by disrupting VS null-adaptation, reproduce prolonged OKAN (failure of velocity storage adaptation), consistent with the known role of the nodulus in velocity storage regulation [CITATION].

**Cranial nerve palsies and INO.** The nine-position gaze paradigm reveals the characteristic misalignment pattern of CN VI palsy (esotropia increasing in the direction of the paretic muscle) and complete CN III palsy (exotropia with ptosis and mydriasis, simulated via motor command asymmetry). Graded severity series demonstrate a monotonic relationship between lesion severity and misalignment magnitude. Right INO, modeled by a 40 ms adduction delay, produces slow right eye adduction with overshoot (abducting nystagmus) of the left eye on leftward saccades, reproducing the WEBINO time-series signature [CITATION].

**Integrator and velocity storage disorders.** Reducing τ_i across a clinical spectrum (25 → 5 s) reproduces the full range from imperceptible to severe gaze-evoked nystagmus, with drift velocity proportional to eccentricity at each level. After sustained eccentric gaze, return to primary position elicits rebound nystagmus whose intensity scales with the degree of NI null-adaptation (τ_ni_adapt). Extended OKAN (prolonged velocity storage) is reproduced by increasing τ_vs_adapt, consistent with the VS null-adaptation mechanism.

**Clinical saccade disorders.** Reducing burst gain g_burst produces slow saccades with preserved amplitude, reproducing the saccadic slowing seen in CN III/VI palsies and certain brainstem lesions. Setting burst gain to zero with preserved NI produces square-wave jerks (SWJ) and ocular flutter, driven by spontaneous NI oscillations in the absence of saccadic gating, consistent with proposed mechanisms of these disorders [CITATION].

### Monocular Occlusion

Binocular fixation at 15 cm with left eye occlusion was simulated under three conditions: dark (both eyes lose the target simultaneously), strobed (position-only updates from the open eye), and continuous monocular viewing. During binocular fixation, vergence is held by symmetric target drive. After occlusion, the dark condition produces slow vergence drift; the strobed condition maintains mean vergence angle without velocity information; and continuous monocular viewing maintains stable fixation in the open eye with progressive drift in the occluded eye, consistent with the clinical observation of latent nystagmus in monocular occlusion paradigms.

### LLM Interface (Figure 3)

The LLM interface was tested with 12 clinical prompts spanning the range of implemented syndromes. Representative examples include: "A patient with acute left vestibular neuritis performing a video head impulse test"; "A healthy subject making a 20-degree rightward saccade followed by 15 deg/s smooth pursuit"; and "A patient with cerebellar ataxia demonstrating gaze-evoked nystagmus at 20 degrees." In each case, the LLM correctly identified the relevant lesion parameters, constructed appropriate stimulus sequences, and returned annotated figures matching expected clinical findings. Average pipeline latency (LLM call + simulation + figure generation) was 8–12 seconds.

---

## Discussion

We have described a unified, differentiable computational model of the primate oculomotor system that accurately reproduces the cardinal healthy behaviors — VOR, VVOR, OKN/OKAN, the saccadic main sequence, smooth pursuit, and fixational eye movements — and the oculomotor phenotypes of major clinical syndromes including vestibular neuritis, cerebellar gaze-holding disorders, cranial nerve palsies, INO, and integrator disorders. The LLM interface lowers the barrier to simulation by translating clinical language directly into model parameters.

**Model contributions.** Relative to prior work, the model makes several advances. First, its fully differentiable JAX implementation enables gradient-based fitting to patient eye movement recordings, opening a path toward patient-specific computational phenotyping. Second, the bilateral architecture with null-adaptation states in both VS and NI provides a unified mechanistic account of phenomena — OKAN adaptation, rebound nystagmus, Bruns nystagmus — that have not previously been captured in a single model. Third, the efference copy pathway correctly prevents saccadic contamination of VS, a constraint that is often omitted in models but is essential for accurate simulation of OKN and OKAN. Fourth, the Smith predictor in the pursuit pathway allows the model to track moving targets without the steady-state slip that plagues simpler pursuit implementations. Fifth, the NI null-adaptation mechanism generates rebound nystagmus from first principles without requiring separate post-hoc computations.

**Teaching tool.** The most immediate application is clinical education. Neurology residents and neuro-ophthalmology fellows are expected to interpret complex eye movement findings at the bedside, yet exposure to rare syndromes is necessarily limited by patient volume and case mix. A simulation platform that generates correct, annotated oculomotor findings for any clinical scenario — including rare presentations, unusual lesion combinations, and progressive severity gradations — provides a scalable, infinitely repeatable supplement to clinical experience. The LLM interface is critical here: by accepting natural language prompts such as "a patient with a right PPRF lesion making a gaze-evoked saccade," the platform meets trainees at their current level of description without requiring them to learn a programming interface. Simulations can be explored interactively, parameters modified to demonstrate mechanism, and expected findings compared against actual patient recordings.

**Synthetic data generation.** A second major application is the production of labeled synthetic training data for machine learning. Eye movement classification models — distinguishing peripheral from central vestibular nystagmus, screening for cerebellar ataxia, or detecting INO — are increasingly feasible but require large labeled datasets that are difficult and expensive to collect clinically. The model can generate arbitrarily large datasets of labeled eye movement traces spanning the full parameter space of each disorder, including severity gradations, partial lesions, noise conditions, and stimulus variants. Because parameters are interpretable and the underlying physiology is known, labels are unambiguous — unlike retrospectively labeled clinical recordings. The differentiable architecture further allows the synthetic dataset distribution to be matched to real data statistics through gradient-based optimization of the noise and parameter distributions.

**Hypothesis generation.** Researchers can predict the oculomotor phenotype of a proposed lesion mechanism, identify parameters that most sensitively distinguish competing hypotheses, or design stimulus sequences that maximally stress a specific subsystem.

**Patient-specific modeling.** Because the model is differentiable, parameters (canal gains, VS time constants, NI integrity, pursuit gain) can be fit to a patient's recorded eye movements via gradient descent, yielding quantitative lesion characterization analogous to how equivalent circuit models characterize electrophysiological recordings. This enables a move from qualitative syndrome labeling to continuous parameter estimation.

**Limitations.** Vergence is implemented as a state but not yet fully validated as a binocular disparity controller; the monocular occlusion experiment treats vergence as a passive state rather than a driven subsystem. The gravity estimator and translational VOR are implemented but not yet validated and are not included here. Listing's law torsional constraints are not enforced. The first-order plant does not capture nonlinear orbital mechanics. The LLM interface, while generally accurate, can misinterpret ambiguous prompts and does not provide uncertainty estimates on parameter choices. Model parameters are set to population-average values and have not yet been fit to individual patient data.

**Future directions.** Immediate priorities include full validation of vergence control and the gravity estimator / translational VOR. Longer-term goals include systematic parameter fitting to published patient datasets, a second-order biomechanical plant, integration with eye-tracking hardware for real-time simulation, and expansion of the LLM interface to comparative simulations (e.g., "show the difference between peripheral and central vestibular nystagmus").

**Conclusion.** ClaudeOculomotorJax provides a biophysically grounded, clinically comprehensive, and computationally accessible platform for simulation of oculomotor behavior in health and disease. By combining a differentiable model architecture with a large language model interface, it bridges the gap between computational neuroscience and clinical neurology, offering new tools for education, hypothesis generation, and, in the future, patient-specific diagnostic inference.

---

## Acknowledgments

[TBD]

## Funding

[TBD]

## Disclosures

[TBD]

---

## References

[CITATION] Robinson DA. The mechanics of human saccadic eye movement. J Physiol. 1964;174:245–264.

[CITATION] Robinson DA. Oculomotor unit behavior in the monkey. J Neurophysiol. 1970;33:393–403.

[CITATION] Raphan T, Matsuo V, Cohen B. Velocity storage in the vestibulo-ocular reflex arc (VOR). Exp Brain Res. 1979;35:229–248.

[CITATION] Galiana HL, Outerbridge JS. A bilateral model for central neural pathways in vestibuloocular reflex. J Neurophysiol. 1984;51:210–241.

[CITATION] Bahill AT, Clark MR, Stark L. The main sequence, a tool for studying human eye movements. Math Biosci. 1975;24:191–204.

[CITATION] Steinhausen W. Über die Beobachtung der Cupula in den Bogengangampullen des Labyrinths des lebenden Hechts. Pflügers Arch. 1933;232:500–512.

[CITATION] Leigh RJ, Zee DS. The Neurology of Eye Movements. 5th ed. Oxford University Press; 2015.

[CITATION] Zee DS, Yamazaki A, Butler PH, Gücer G. Effects of ablation of flocculus and paraflocculus of eye movements in primate. J Neurophysiol. 1981;46:878–899.

[CITATION] Bhidayasiri R, Plant GT, Leigh RJ. A hypothetical scheme for the brainstem control of vertical gaze. Neurology. 2000;54:1985–1993.

[CITATION] Bradbury J, Frostig R, Hawkins P, et al. JAX: composable transformations of Python+NumPy programs. 2018. http://github.com/google/jax

[CITATION] Kidger P. On neural differential equations [PhD thesis]. University of Oxford; 2021.

[CITATION] Anthropic. Claude: a family of large language models. 2024. https://anthropic.com

---

## Tables

**Table 1. Key model parameters and their physiological basis.**

| Parameter | Symbol | Default Value | Source |
|---|---|---|---|
| Canal adaptation TC | τ_s | 5 s | Goldberg & Fernandez 1971 |
| Velocity storage TC | τ_vs | 15 s | Raphan et al. 1979 |
| VS null-adaptation TC | τ_vs_adapt | 600 s | Estimated |
| NI time constant | τ_i | 25 s | Robinson 1975 |
| NI null-adaptation TC | τ_ni_adapt | 20 s | Estimated |
| Plant time constant | τ_p | 0.15 s | Robinson 1964 |
| Burst gain | g_burst | 700 deg/s | Bahill et al. 1975 |
| OKR visual gain | K_vis | 1.5 | Estimated |
| VN resting bias | b_vs | 100 deg/s | Estimated |
| Saccade threshold | θ_sac | 0.5 deg | Estimated |
| Visual delay (vel/pos) | — | 120 ms | Lisberger & Westbrook 1985 |
| Visual delay (VF gate) | — | 40 ms | Estimated |

---

## Figure Legends

**Figure 1. Healthy oculomotor behavior.**
(A) VOR and velocity storage. Constant-velocity rotation in the dark (60 deg/s step) produces compensatory eye velocity that decays with time constant ~17 s, longer than the canal adaptation TC (5 s), due to velocity storage. VVOR (rotation in a stationary lit world) maintains near-perfect gaze stability throughout. (B) OKN and OKAN. Full-field scene motion (30 deg/s) drives steady-state OKN; after scene offset, OKAN persists ~20 s. Signal cascade panels show canal, VS, and NI states confirming the signal-flow architecture. (C) Saccadic main sequence. Peak velocity vs. amplitude for 2–40 deg target steps; saturating exponential fit (dashed line) matches Bahill et al. normative data. Oblique saccade trajectories are straight with synchronized components. Double-step refractoriness is evident in the intersaccadic interval distribution. (D) Smooth pursuit. Velocity range (5–40 deg/s ramps) and sinusoidal tracking (0.3–2 Hz). (E) Fixational eye movements. Sparse microsaccades and slow drift produced by canal and retinal position noise.

**Figure 2. Clinical lesion simulations.**
(A) Unilateral vestibular neuritis (left). Spontaneous rightward nystagmus; absent VOR with corrective saccade on leftward vHIT; intact response rightward. Rotary chair shows directional preponderance. Graded partial lesions produce intermediate deficits. (B) Cerebellar lesions. FL/PFL lesion: gaze-evoked nystagmus (drift velocity proportional to eccentricity, Alexander's law), rebound nystagmus, impaired pursuit gain. Nodulus/uvula lesion: prolonged OKAN (failure of VS null-adaptation). (C) Cranial nerve palsies and INO. Nine-position gaze shows CN VI palsy misalignment pattern. INO time-series shows slow adduction and abducting nystagmus of the contralateral eye. Graded recovery series. (D) Integrator and velocity storage disorders. Gaze-evoked nystagmus across τ_i severity spectrum (25 → 5 s). Rebound nystagmus intensity scales with NI null-adaptation. Extended OKAN with increased τ_vs_adapt. Slow saccades (reduced burst gain) and square-wave jerks (zero burst gain with intact NI).

**Figure 3. Large language model interface.**
(A) Pipeline schematic: plain-English prompt → Claude LLM (schema extraction) → JAX simulator → annotated figure. (B) Sample output: acute left vestibular neuritis with vHIT. (C) Sample output: healthy saccade and pursuit sequence. (D) Sample output: cerebellar gaze-evoked nystagmus. Latency: 8–12 s end-to-end.

**Figure 4. Model architecture diagram.**
Schematic of the five functional stages with state counts: sensory encoding (canal 12 states, otolith 6, retina 480 per eye); brainstem circuits (velocity storage 9, neural integrator 9, saccade generator 9, efference copy 120, smooth pursuit 3); plant (3 per eye). Signal flow arrows indicate: head velocity → canal → VS; retinal slip → visual delay → VS (OKR) and pursuit; retinal position → visual delay → saccade generator; efference copy (dashed) → VS and retinal slip cancellation. Parameter vector θ (modified for lesion simulations) and LLM interface entry point are indicated.
