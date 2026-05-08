# A unified Bayesian template for continuous oculomotor servo loops

*Draft note*

## Abstract

We argue that the oculomotor system has only **two** architectures, not eight. One is the saccadic decision system: an evidence accumulator that crosses a bound and triggers a discrete burst of motor activity. The other is a continuous-control template shared identically by every other loop — VOR, OKR, smooth pursuit, vergence, accommodation, T-VOR, and the gaze-holding integrator. The continuous template has the same dynamic equations, the same forward-model prediction-error form $K \cdot (\mathrm{sensor} - M(\mathrm{motor})) / \tau_{\mathrm{fast}}$, the same integrator pair (fast phasic + slow tonic-with-setpoint), the same soft-threshold form, and the same kind of perceptual-prior injection in every loop. What differs across loops are only the parameter values: the loop gain $K$, the forward model $M(\cdot)$ (trivial or EC-based), the receptor noise variance, the signal bandwidth, the plant lag, the cross-link calibrations, and the motor and perceptual priors. The differences are quantitative; the architecture is one. Saccades alone require an additional structural element — the decision step — because they emit a discrete commit rather than a continuous output.

## 1. Two architectures, not eight

The oculomotor literature describes eight servo loops in eight block diagrams: VOR with Raphan-Cohen velocity storage, OKR with optokinetic gain, pursuit with Lisberger's Smith predictor, saccades with Robinson's local-feedback burst, vergence with Schor's dual integrator, accommodation with the lens plant, T-VOR with distance scaling, neural integrator with bilateral push-pull. Each block diagram looks different.

We claim there are really only two:

1. **Saccades**: decision-driven, discrete commit. An evidence accumulator (the rise-to-bound z_acc machinery) integrates retinal position error; when it crosses a threshold, a burst of motor commands fires the eye to a new position. The burst self-terminates via local feedback and the system returns to fixation.

2. **All other loops**: continuous-control servo. A noisy sensory measurement is transformed (variance-stabilising step where needed), passed through a push-pull rectified readout, low-pass filtered at the SNR-optimal cutoff, fed into a fast phasic integrator and a slow tonic integrator with motor setpoint, and emitted to the plant. The closed-loop visual feedback completes the loop.

The structural difference between (1) and (2) is the decision step. Everything else is shared.

## 2. The continuous-control template

Every continuous loop has the same dynamic structure:

```
[1] Sensor signal              x(t) = s(t) + n(t)              (s = signal, n = noise)
[2] Variance-stabilising        u(t) = ψ(x(t))                  (ψ = √, log, or identity)
[3] Push-pull MAP readout       ŝ(t) = ReLU(u−θ) − ReLU(−u−θ)   (θ = σ²/τ_prior)
[4] Temporal LP                 ŝ̄(t) = LP(ŝ; τ_LP)             (τ_LP from Wiener)
[5] Prediction error            e(t) = ŝ̄(t) − e_efference(t)   (closed-loop deviation)
[6] Fast integrator             dx_fast/dt = K e − x_fast/τ_fast + p_perceptual / τ_fast
[7] Slow integrator             dx_slow/dt = (x_setpoint + K_a · pathway − x_slow) / τ_slow
[8] Output                       u_motor = direct + x_fast + x_slow + cross-links
[9] Plant                        dx_plant/dt = (u_motor − x_plant) / τ_plant
[10] Loop closure                x_plant → world → sensor at [1]
```

This is the *same* equation in every loop. What changes is the values of the symbols:

- $\psi$ is identity for Gaussian-noise sensors, $\sqrt{x}$ for Poisson photon noise, $\log|x|$ for multiplicative noise.
- $\theta$ is the noise-to-prior ratio; bigger noise widens the dead-zone.
- $\tau_{\mathrm{LP}}$ is set by the SNR-crossover frequency of the channel.
- $\tau_{\mathrm{fast}}$ and $\tau_{\mathrm{slow}}$ are channel-specific time constants.
- $K$ is the loop gain, set by stability requirements.
- $x_{\mathrm{setpoint}}$ is the motor prior (tonic baseline).
- $p_{\mathrm{perceptual}}$ is the cognitive/contextual perceptual prior (proximal cue, expectation).
- Cross-links carry information between loops (AC/A and CA/C between vergence and accommodation; efference copy across all).

The form of every step is shared. Only the values of the parameters differ.

## 3. Same template, different parameters

Each continuous loop is the template with its own parameter set:

| Loop | Receptor | $\psi(x)$ | $\tau_{\mathrm{LP}}$ | $\tau_{\mathrm{fast}}$ | $\tau_{\mathrm{slow}}$ | Cross-links | Plant $\tau$ |
|---|---|---|---|---|---|---|---|
| VOR | canal afferent | id | $\sim 50$ ms | VS $\sim 20$ s | NI $\sim 25$ s | NI for hold | $\sim 200$ ms |
| OKR | retinal slip | id | $\sim 50$ ms | VS $\sim 20$ s | NI $\sim 25$ s | EC slip cancellation | $\sim 200$ ms |
| Pursuit | retinal velocity | id | $\sim 150$ ms | pursuit $\sim 5$ s | leak $\sim 40$ s | Smith predictor (EC) | $\sim 200$ ms |
| Vergence | disparity | id | $\sim 150$ ms | fast $\sim 5$ s | tonic $\sim 20$ s | AC/A from acc | $\sim 200$ ms |
| Accommodation | defocus | id | $\sim 50$ ms | fast $\sim 5$ s | tonic $\sim 20$ s | CA/C from verg | lens $\sim 150$ ms |
| T-VOR | otolith $\to$ heading | id | $\sim 2$ s | heading $\sim 2$ s | none | distance scaling via vergence | $\sim 200$ ms |
| Neural integrator | motor command | id | none | none | gaze $\sim 25$ s | bilateral push-pull, null adapt | $\sim 200$ ms |

The phototransduction cascade has $\psi = \sqrt{x}$ — this is upstream of the retinal-slip / disparity / defocus signals that downstream loops consume.

## 4. Saccades: the structural exception

Saccades follow steps [1] through [4] of the template (sensor, MAP readout, LP), but then diverge:

```
[5'] Drift-diffusion accumulator:
         da/dt = |ŝ̄(t)| − leak · a(t)

[6'] Threshold cross:
         when a(t) > θ_decision: trigger burst

[7'] Burst generator (Robinson local feedback):
         u_burst(t) = ramp toward target, self-terminates via x_copy efference
```

This is a fundamentally different action: a discrete commit rather than a continuous response. The decision step (drift-diffusion to bound) is what saccades have that the continuous loops do not. The burst itself is then a discrete event that returns the system to its continuous fixation state.

Why is saccades the only loop with a decision? Because the goal of saccades is itself discrete — *direction-of-gaze* steps to a new value when a new target appears, a categorical event. Vergence/accommodation/pursuit can all respond proportionally to their inputs; saccades cannot, because eye position is a single quantity that must commit to one location at a time.

## 5. What differs between loops, and why

Each parameter in the template is set by physical or physiological constraints, not by free choice:

**Receptor noise distribution.** Photon shot noise is Poisson at low light. Retinal velocity (motion) has multiplicative variance scaling. Vestibular afferents have approximately Gaussian additive noise. Neural firing rates have variance that scales with mean rate. Different loops process signals from different sensors with different noise models, so $\psi$ and the downstream LP $\tau$ differ.

**Signal bandwidth.** Disparity changes slowly (V1 stereo correspondence is genuinely slow). Defocus changes can be fast (blur is detected near-instantly). Vestibular slip can change rapidly. The Wiener-optimal $\tau_{\mathrm{LP}}$ is dictated by where signal-power and noise-power cross.

**Plant lag.** The eye plant has $\tau \sim 200$ ms. The lens plant has $\tau \sim 150$ ms. Both are biophysical constants.

**Loop gain $K$.** Set by stability margin: $K \tau_{\mathrm{fast}}$ controls closed-loop response speed, but high values destabilise. Each loop's $K$ is calibrated to operate in a stable regime.

**Cross-links.** AC/A and CA/C are physiologically calibrated by long-term operation; their loop gain $L = K_t \alpha \beta K_s$ controls cross-coupling amplification. Smith-predictor in pursuit is the efference-copy strength.

**Tonic and perceptual priors.** Subject-specific. Tonic vergence and tonic accommodation are motor priors set by long-term operating range. Perceptual priors (proximal cue, expectation) are cognitive/contextual.

In every case the parameter is a physical or physiological measurement, not a free knob. The model has many fewer effective degrees of freedom than a per-loop block diagram count would suggest.

## 6. The prediction-error structure

A subtle but unifying feature: every loop's fast integrator is driven by *prediction error*, not raw sensor signal. We claim that **every continuous loop is driven by**

$$
K \cdot e(t) / \tau_{\mathrm{fast}}, \qquad
e(t) = \mathrm{sensor}(t) - M(\mathrm{motor}(t))
$$

where $M(\cdot)$ is the brain's forward-model prediction of the sensor given current motor state, and $K$ is the loop gain. The architecture is the same; what differs across loops is the forward model $M$ (trivial or non-trivial), the sensor noise, the bandwidth, and the gain.

Two extremes of the forward model:

**Trivial forward model: $M(\mathrm{motor}) = 0$.** The goal is "sensor reads zero". The prediction error reduces to the raw sensor reading. This is the case for closed-loop sensors that already represent deviation from the goal:
- *Vergence*: disparity = (target binocular position − eye binocular position) is itself the deviation from the goal "fused on target".
- *Accommodation*: defocus = (demand − lens power) is the deviation from the goal "in focus".
- *Saccades*: retinal position error is the deviation from "target on fovea".

**Non-trivial forward model: $M(\mathrm{motor}) = $ predicted sensor from EC.** The forward model uses an efference copy of the motor command to predict the sensor's expected reading. The prediction error is the actual sensor minus this prediction:
- *OKR / Velocity storage*: $M(\mathrm{eye\,vel}) = -\mathrm{eye\,vel}$ on retina; the residual is world flow.
- *Pursuit (Smith predictor)*: cancels the predicted contribution of eye motion to retinal velocity.
- *VOR (implicitly)*: the canal predicts the upcoming slip; the motor command sets eye velocity to null that prediction.

In every case the input to the integrator is a forward-model prediction error. The model is trivial when the sensor *is* already the error (vergence, accommodation, saccades); the model is the EC pathway when self-motion confounds the raw sensor (OKR, pursuit, VOR).

This is a stronger unification than "all loops use prediction errors" because it identifies *the same equation* in every loop: integrator input = $K \cdot (\mathrm{sensor} - M(\mathrm{motor})) / \tau_{\mathrm{fast}}$. The differences across loops collapse to two parameters: the form of $M$ and the value of $K$.

## 7. Implications

**Many fewer free parameters.** Because the template's parameters are dictated by noise statistics and signal bandwidth, the model has many fewer free knobs than a per-loop block-diagram count would suggest. Fitting a clinical patient becomes a small-parameter problem.

**Each loop's "personality" is its noise statistics.** Pursuit is sluggish because motion has multiplicative noise and a slow MT/MST decoder. Disparity is sloppy because V1 stereo computation is genuinely slow. VOR is fast because canal afferents are clean. Eight loops, eight noise statistics, one architecture.

**Pathology has a unified description.** A "neural integrator failure" affects step [7] for any loop with a slow integrator. A "sensor-noise pathology" widens $\theta$ and slows responses uniformly. The template gives a mechanistic vocabulary for oculomotor disorders without requiring a separate framework for each loop.

**The decision/continuous split is principled.** Saccades are decisions (commit-to-target events) and require the additional drift-diffusion step. All other loops are continuous controllers and don't. The two architectures correspond to two distinct types of motor goal.

## 8. State-space form of the unified template

The unified template is a linear-affine system on a partitioned state, supplemented by a small set of bilinear corrections at well-defined interfaces. Let

- $x_{\text{fast}} \in \mathbb{R}^{n_f}$  — fast integrators (per-loop pursuit memory, NI populations, vergence fast, accommodation fast, …)
- $x_{\text{slow}} \in \mathbb{R}^{n_s}$  — slow tonic adapters (NI null, vergence tonic, accommodation slow)
- $u \in \mathbb{R}^{n_u}$  — raw sensory input (slip, disparity, defocus) plus delegated drives (efference-copy-canceled velocity, OCR/Listing's tonic, T-VOR rate)
- $y \in \mathbb{R}^{n_y}$  — motor readout (pursuit command, NI version, vergence command, accommodation command)

**The dynamics:**

$$
\boxed{
\begin{aligned}
\dot{x}_{\text{fast}} &= A\,(x_{\text{fast}} - M_{fs}\,x_{\text{slow}}) + B\,g(u) + b_{\text{prox}} + f(x, g(u)) \\
\dot{x}_{\text{slow}} &= C\,(x_{\text{slow}} - T) + M_{ss}\,x_{\text{slow}} + D\,x_{\text{fast}} + B_s\,g(u) \\
y                     &= E\,x_{\text{fast}} + G\,x_{\text{slow}} + F\,g(u)
\end{aligned}
}
$$

**The matrices and their physiological meaning:**

| Matrix | Role | Set by |
|---|---|---|
| $A$ | leak rates of fast integrators (incl. fast→fast cross-couplings, e.g., pursuit→NI) | spectral content of natural-input prior |
| $B$ | sensor / supplementary drive into fast integrators | Kalman gain (signal-to-noise ratio) |
| $M_{fs}$ | "what slow state each fast state leaks toward" (e.g., NI L leaks toward $x_{\text{null}}/2$) | bilateral architecture of the integrator |
| $C$ | diagonal slow leak rates | slow timescale of the prior |
| $M_{ss}$ | slow→slow cross-couplings (AC/A and CA/C) | natural co-variance of disparity and defocus (cue combination) |
| $D$ | slow integrators driven by fast states (Schor cascade) | dual-integrator architecture |
| $B_s$ | direct sensor drive into slow integrators (closed-loop direct paths) | Schor direct-path gains |
| $T$ | setpoint vector for slow integrators | motor prior (tonic vergence, tonic accommodation) |
| $E$ | linear readout of motor command from fast states | pulse-step structure (fast = pulse) |
| $G$ | linear readout of motor command from slow states | pulse-step structure (slow = step) |
| $F$ | direct sensor feedthrough to motor (Robinson pulse / direct path) | $\tau_p$ for vergence/NI; $\tau_{\text{plant}}$ for accommodation |
| $b_{\text{prox}}$ | proximal-cue constant injection on fast | Hung-Semmlow perceived-distance prior |

**The functions $g$ and $f$:**

- $g(u)$ — **Bayesian-MAP preprocessing** of the raw sensory input. Implements the soft-threshold dead-zone (sparse-amplitude prior, $\theta = \sigma^2/\tau_{\text{prior}}$), variance-stabilising transforms ($\sqrt{\,\cdot\,}$ for Poisson, $\log|\,\cdot\,|$ for multiplicative noise), and perspective-to-diopter conversion (so the rational $1/z$ depth scaling becomes linear). For the continuous-loop subset, $g$ is currently identity on slip / disparity / defocus.

- $f(x, g(u))$ — **bilinear corrections** from the symmetric outer product of $[1; \hat x; \hat u]$. Empty for the continuous-loop subset; populated with antisymmetric cross-product entries for SO(3) kinematics ($\omega \times r$ in plant rotation, gravity transport in VS), with depth-modulated entries for T-VOR ($v_{\text{trans}}\cdot D$), and with sigmoid-gated entries for saccade burst dynamics ($\sigma(z_{\text{acc}}) \times \text{burst}$).

**One equation, one architecture.** The dynamics for vergence, accommodation, NI, and pursuit are *one* state-space system whose blocks correspond to those subsystems. Cross-couplings (AC/A, CA/C, pursuit→NI) appear as off-diagonal entries in $A$, $D$, $M_{ss}$. Saccades fit by adding decision-side entries (accumulator $z_{\text{acc}}$ as a state, sigmoid gates as bilinear $f$ terms).

### 8.1 Reference implementation

The module `oculomotor.models.brain_models.unified_brain` implements this exactly. It exposes:

```python
matrices(theta) -> UnifiedMatrices(A, B, M_fs, C, M_ss, D, B_slow, T, E, G, F, b_prox)
g(sensory_out, theta, ...)    # Bayesian preprocessing → u'
f(x_fast, x_slow, u', theta, z_act)   # bilinear corrections (empty for subset)
step(x_brain, sensory_out, brain_params, noise_acc=0.0)
```

The `step` function is a drop-in replacement for `brain_model.step` (same I/O signature). Subsystems outside the linear-template subset (self-motion / VS / gravity, saccade generator, T-VOR, target memory, Listing's law) are delegated to existing modules; the continuous-loop core (pursuit, NI, vergence, accommodation, AC/A and CA/C couplings) is implemented entirely via the matrices above.

Comparison on `bench_compare_unified.py` (5 deg/s pursuit ramp, 10 deg saccade step) shows the unified module reproduces `brain_model.step` to floating-point precision (eye-position max abs diff $\sim 10^{-6}$ deg).

### 8.2 What the matrices encode for each loop

For each oculomotor loop, the corresponding matrix entries are:

**Pursuit** (3 fast states, no slow):
- $A_{\text{pu,pu}} = -[1/\tau_{\text{pu}} + K_{\text{pu}}/(1+K_\phi)]\,I_3$ (Smith-folded leak)
- $B_{\text{pu, slip}} = K_{\text{pu}}/(1+K_\phi)\,I_3$ (Kalman gain)
- $E_{\text{pu, pu}} = 1/(1+K_\phi)\,I_3$, $F_{\text{pu, slip}} = K_\phi/(1+K_\phi)\,I_3$ (pulse-step Smith readout)

**Bilateral neural integrator** (6 fast + 3 slow):
- Diagonal leaks $-1/\tau_i\,I_3$ on $L$ and $R$ populations.
- $M_{fs}$: $L$ leaks toward $+x_{\text{null}}/2$, $R$ toward $-x_{\text{null}}/2$.
- $D$: $\dot x_{\text{null}} = (x_L - x_R - x_{\text{null}})/\tau_{\text{adapt}}$.
- $E$: net = $x_L - x_R$, pulse via $\tau_p\cdot u_{\text{vel}}$.

**Vergence** (Schor dual integrator, 6 fast + 3 slow):
- Fast $A_{\text{v,v}} = -1/\tau_v\,I_3$, $B_{\text{v,disp}} = K_v\,I_3$.
- Slow leak $-1/\tau_t$, $D_{\text{v\,tonic, v}} = K_t/\tau_t\,I_3$, $B_s$ direct path $K_t\tau_p/\tau_t$.
- Setpoint $T_{\text{v\,tonic}}[H] = \text{tonic\_verg}$.
- AC/A: $D_{\text{v\,tonic}[H], a\,fast} = K_t\alpha/\tau_t$, $M_{ss\,[v\,tonic[H], a\,slow]} = K_t\alpha/\tau_t$.

**Accommodation** (Schor dual integrator, 1 fast + 1 slow):
- Mirror of vergence with single scalar each.
- CA/C: $D_{\text{a\,slow}, v[H]} = K_s\beta/\tau_s$, $M_{ss\,[a\,slow, v\,tonic[H]]} = K_s\beta/\tau_s$.

The cross-coupling pattern is symmetric and expresses the natural input statistics: disparity and defocus co-vary in natural environments, and the matrices encode that co-variance as Bayesian-optimal cue combination at integrator level.

## 9. Conclusion

The oculomotor system has one architecture for continuous control and one architecture for discrete decision-driven action. The seven continuous loops differ only in their parameter values, all of which are dictated by noise statistics, signal bandwidth, and physiological constants. Saccades alone require an additional structural element — the decision step — because their motor goal is discrete.

The matrices $A, B, C, D, E, F, G, M_{fs}, M_{ss}, T$ make this concrete. Each loop is a sub-block of one linear system on $(x_{\text{fast}}, x_{\text{slow}})$. Saccades add accumulator states whose threshold-crossing fires a bilinear gating in $f(x, u)$, but the underlying dynamics are still on the same state vector. Sensory and motor interfaces own the residual nonlinearities: rectifications, perspective division, push-pull splits.

This claim is a corollary of the inverted causal arrow argued in the companion note on push-pull rectification: if biology selects substrates that perform Bayesian inference cheaply, and if Bayesian inference under uncertainty has a canonical structure (rectifier $\to$ soft threshold $\to$ Wiener LP $\to$ leaky integrators), then we should expect every Bayesian-inference-driven motor loop to share that structure. The fact that the seven oculomotor servo loops do is direct evidence that the inversion holds.

## References

- Bogacz, R., Brown, E., Moehlis, J., Holmes, P., Cohen, J. D. (2006). The physics of optimal decision making: a formal analysis of models of performance in two-alternative forced-choice tasks. *Psychological Review* 113, 700–765.
- Lisberger, S. G., Westbrook, L. E. (1985). Properties of visual inputs that initiate horizontal smooth pursuit eye movements in monkeys. *Journal of Neuroscience* 5, 1662–1673.
- Raphan, T., Matsuo, V., Cohen, B. (1979). Velocity storage in the vestibulo-ocular reflex arc (VOR). *Experimental Brain Research* 35, 229–248.
- Robinson, D. A. (1975). Oculomotor control signals. In *Basic Mechanisms of Ocular Motility and their Clinical Implications*, ed. G. Lennerstrand and P. Bach-y-Rita, 337–374. Pergamon Press.
- Schor, C. M. (1986). Adaptive regulation of accommodative vergence and vergence accommodation. *American Journal of Optometry and Physiological Optics* 63, 587–609.
- Wiener, N. (1949). *Extrapolation, Interpolation, and Smoothing of Stationary Time Series.* MIT Press.
