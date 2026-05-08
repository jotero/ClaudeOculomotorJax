# Saccade triggers and drains as Bayesian-optimal SPRT (a Kalman gain on a decision)

*Draft note*

## Abstract

The continuous-control template (companion note: `unified_oculomotor_template.md`) gives every smooth oculomotor loop the same Bayesian shape: linear-affine dynamics, Kalman-derived gains, a sparse-prior dead zone, and a two-timescale spectral decomposition. Saccades sit outside that template because they are *decision-driven* rather than continuous. Here we argue that the saccade generator's trigger machinery — the rise-to-bound accumulator $z_{\text{acc}}$, the OPN tonic recovery, the IBN-driven drains, the `gate_err` sigmoid — is *not* a separate kind of computation but the same Bayesian inference applied to a different problem: deciding *when to act* rather than *what to track*. The accumulator is a sequential probability ratio test (Wald 1947, Bogacz et al. 2006); its rates are Kalman gains in disguise. Same template, different problem class.

## 1. The decision problem

At every moment a fixating animal must decide between two latent states:

- $H_0$ — *fixation is good*: target is on the fovea, no action needed.
- $H_1$ — *fixation is broken*: target has drifted off, fire a saccade.

The brain's evidence about which hypothesis is true is the retinal position error magnitude $|e_{\text{cur}}|$. Under $H_0$ the error is small (driven by sensor noise, microsaccadic drift, fixation jitter). Under $H_1$ the error is large (driven by genuine target offset). The optimal decision rule under iid evidence is Wald's *Sequential Probability Ratio Test*:

$$
S(t) = \int_0^t \log\frac{P(e\mid H_1)}{P(e\mid H_0)} \, d\tau,
\qquad
\text{commit when}\ S(t) > \theta_{\text{decision}}.
$$

The accumulator $z_{\text{acc}}$ in the saccade generator is exactly $S(t)$: a running log-likelihood ratio that integrates per-step evidence about whether fixation is broken.

## 2. Mapping the SG components onto SPRT / Bayesian dynamics

| SG component | Bayesian / SPRT role |
|---|---|
| $z_{\text{acc}}$ | running posterior log-likelihood for "saccade needed" |
| `gate_err = σ(|e| − θ)` | per-step evidence: thresholded innovation on position error |
| Drift rate $\sim \text{gate\_err}\cdot \text{normalized\_opn}/\tau_{\text{acc}}$ | gain on evidence accumulation — the SPRT analog of the Kalman gain |
| $\tau_{\text{acc}}$ | inverse Kalman gain — long $\tau$ ⇒ conservative ⇒ need strong/sustained evidence |
| Threshold $\theta_{\text{acc}}$ | Wald's decision boundary, set by loss-ratio $C_{\text{wait}}/C_{\text{error}}$ |
| Threshold cross | optimal stopping rule — commit and emit action |
| Drain on action ($g_{\text{acc\_drain}}\cdot \text{ibn\_norm}\cdot z_{\text{acc}}$) | Bayesian *reset on commit* — posterior collapses after action, start fresh |
| Passive leak ($-z_{\text{acc}}/\tau_{\text{acc\_leak}}$) | forgetting / process noise — old evidence is downweighted because the world's state is non-stationary |
| `z_opn` tonic recovery ($k_{\text{tonic}}\cdot (100-z_{\text{opn}})$) | prior pull: default is "OPN active, no saccade" — posterior relaxes to prior absent contrary evidence |
| `z_trig` smooth onset | low-pass on the commit signal — same role as Kalman update gain |
| OPN/IBN Schmitt latch | bistable commit logic — once committed, stay committed until action completes |

Every "drain" or "rate" in the trigger machinery has a Bayesian interpretation as either (a) a gain on evidence, (b) a decay of stale evidence, or (c) a reset of posterior after action.

## 3. The Kalman-gain analogy made precise

In a Kalman filter, the gain on observations is

$$
K = \frac{\Sigma_{\text{prior}}}{\Sigma_{\text{prior}} + \Sigma_{\text{meas}}}
$$

— it balances confidence in the prior against confidence in the measurement. High prior uncertainty ⇒ high gain ⇒ trust the new measurement. Low prior uncertainty ⇒ low gain ⇒ stick with the prior.

In the SPRT formulation, the analogous quantity is the drift rate of the accumulator. If $\sigma_e^2$ is the noise on position error and $\Delta\mu$ is the expected jump under $H_1$, the optimal drift rate is

$$
\mu_{\text{drift}} = \frac{\Delta\mu}{\sigma_e^2}.
$$

So $\tau_{\text{acc}}^{-1} \propto \Delta\mu / \sigma_e^2$ — exactly the same "signal-to-noise" balance as the Kalman gain. The numerator is "how much does the evidence change under $H_1$" (signal); the denominator is "how noisy is the per-step evidence" (measurement variance).

In short: **the accumulator's drift-rate gain *is* a Kalman gain on the binary decision $\{H_0, H_1\}$**.

## 4. Drains and the same `1 + g·activation` factoring

The decay/drain rates in SG follow the same `1 + g·activation` pattern as the cerebellar gain modulations on continuous loops:

- $z_{\text{trig}}$ drain rate: $1 + g_{\text{ibn\_trig}}\cdot \text{ibn\_norm}$ — baseline 1 (recover after evidence settles), *plus* IBN-active drain when burst is firing.
- BN clamp rate: $1 + g_{\text{opn\_bn}}\cdot \text{act\_opn}$ — baseline 1 (membrane self-decay), *plus* OPN clamp during fixation.

In each case the "1" is the brainstem / prior-driven baseline, and the "$g \cdot \text{activation}$" is a state-dependent gain modulation that the cerebellum/vermis can plausibly learn or adapt.

The Kalman-gain modulation in the gravity estimator —

$$
K_{\text{grav,eff}} = K_{\text{grav}} \cdot \sqrt{1 + |\omega \times \hat g|/w_{\text{gate}}}
$$

— follows the same recipe. Baseline gain × (1 + state-dependent uncertainty correction).

This means saccade trigger drains, BN gain modulation, and Kalman gain modulation are all *the same kind of object*: state-dependent gains with a baseline + correction structure, where the correction tracks current uncertainty / commitment state.

## 5. Why this matters

**Same Bayesian template across the system.** The continuous loops are Kalman filters on continuous state. The saccade generator is a Kalman/SPRT-equivalent on a binary decision. Both compute posterior estimates given noisy evidence; both modulate gains by uncertainty; both use the prior to fall back when evidence is weak. The "saccades are special" caveat from the unified-template note refers to the *output* (discrete commit) rather than to the *machinery* (Bayesian inference with state-dependent gains). The machinery is shared.

**Cerebellar adaptation has a unified target.** The cerebellum modulates state-dependent gains everywhere: VOR gain (continuous loop), Kalman gain on gravity (continuous loop), saccade drift rate (decision loop). All of them are `K_baseline · (1 + activation·correction)` factorings. A unified theory of cerebellar learning would update the *correction* term across all of these — the same kind of plasticity rule, applied at the same kind of slot, on three different kinds of computation.

**The trigger thresholds become parameters of an optimal decision rule, not arbitrary knobs.** Wald's analysis sets the threshold from the loss ratio:
$$\theta_{\text{decision}} \approx \log\frac{C_{\text{wait}}}{C_{\text{error}}}.$$
Behaviorally, lowering the saccade threshold (faster triggering) is exactly what we'd expect when the cost of waiting goes up (target moving fast, time pressure) or the cost of error goes down (small position errors are tolerable). This matches the empirical pattern: saccade rate increases with reward urgency, decreases under high precision demands.

**Uncertainty-adaptive triggers.** The natural extension is to make $\tau_{\text{acc}}$ (and the drain rates) themselves uncertainty-dependent. For example, a recent target flash creates a spike in *posterior uncertainty* about whether the eye is still on target — the gain on evidence should rise (shorter $\tau_{\text{acc}}$). A long stable fixation grows confidence — gain falls. This would make the accumulator a *state-dependent SPRT*, with the same shape `K_baseline · uncertainty_factor` we already have for `K_grav · √(1+ρ)` on continuous gravity tracking.

## 6. Evidence prediction: the same template, one level up

The previous sections cast the SG's *trigger machinery* — the drift gain on $z_{\text{acc}}$, the drains, the OPN recovery — as Kalman/SPRT operations on a binary decision. But the *input* to that machinery is itself a noisy, delayed signal, and the same Bayesian framework can clean it up.

**The second-bump artifact.** Currently the SG's evidence is `pos_delayed` — the retinal target error after the ~80 ms visual cascade. This produces a structural artifact: every saccade is followed ~80 ms later by a second bump in `pos_delayed`, the visual-delay echo of the pre-saccade error finally arriving through the cascade. By the time the bump appears the eye has already corrected for it; the SG is being asked to ignore a signal that no longer reflects the world. The refractory parameters absorb this — `acc_burst_floor` $= -0.5$ pushes $z_{\text{acc}}$ deep enough below threshold that the post-saccade echo cannot re-trigger, giving an ISI floor of ~270 ms (`threshold_acc − acc_burst_floor`)$\cdot \tau_{\text{acc}}$. The trigger machinery is doing double duty: legitimate Bayesian reset on commit, *plus* an empirical pad sized to outlast a delay echo.

**The Kalman fix is the same recipe one frame up.** Maintain a target-position prior in head frame, $\hat p_{\text{target,head}}$. The predicted retinal error at any moment is

$$
\hat e_{\text{ret}}(t) = \hat p_{\text{target,head}} - \hat p_{\text{eye,head}}(t),
$$

computed instantaneously from the integrator state — un-delayed. Update the prior via the delayed innovation

$$
\nu = e_{\text{ret,delayed}} - \big(\hat p_{\text{target,head}} - \hat p_{\text{eye,head}}(t - \tau_{\text{vis}})\big),
$$

gated by a saccade-aware $K(|w_{\text{eye}}|)$ — high during fixation/pursuit, low during the burst — exactly the structure already proposed for VS slip cancellation. After landing, $\hat p_{\text{target,head}}$ stays put while $\hat p_{\text{eye,head}}$ jumps to the new value, so $\hat e_{\text{ret}}$ reads ~0 immediately. The delayed observation arrives ~80 ms later and confirms (innovation $\approx 0$); no second bump appears in the SG's evidence stream because there is nothing to suppress.

**Refractoriness reverts to its biophysical role.** With the echo removed at the source, the trigger parameters stop carrying the artifact-suppression load. `acc_burst_floor` can move toward 0, ISI shortens toward observed primate values (180–200 ms), and corrective saccades fire on real residual error rather than on delayed echoes. The quick-phase guard `threshold_sac_qp`, currently sized to absorb stale OKN slip, becomes structural rather than load-bearing.

**The unifying observation.** Section 3 mapped the Kalman gain onto the SG's *decision rate* — the drift gain on $z_{\text{acc}}$. Position prediction maps the Kalman gain onto the SG's *evidence input* — the gain on the innovation that drives `gate_err`. Both layers are Kalman gains tuned to current uncertainty, both gated by saccade execution state, both falling out of the same Bayesian template. The refractory period exists where it must (biophysics: OPN recovery, BN repolarization), but no longer compensates for a missing predictor.

## 7. Concrete computational claim

Every "rate" or "gain" in the saccade generator is, in the limit of a properly calibrated Bayesian decision, a function of two quantities only:

1. The current *prior confidence* in the appropriate hypothesis ($H_0$ for the accumulator, "OPN active" for $z_{\text{opn}}$, "burst running" for $z_{\text{trig}}$).
2. The current *evidence rate* — how fast new evidence is arriving.

Under that calibration:

- $\tau_{\text{acc}}$ = inverse drift-rate gain = $\sigma_e^2 / \Delta\mu$
- $\tau_{\text{acc\_leak}}$ = forgetting timescale = inverse of process-noise rate on $H_0$
- $\tau_{\text{burst\_drain}}$ = post-commit reset timescale = effectively 0 (Bayes posterior collapse)
- $\tau_{\text{sac}}$ for OPN recovery = inverse prior strength on "OPN active" (how long until prior dominates again)
- $\tau_{\text{trig}}$ = update timescale on the trigger commit signal

The values of these timescales — empirical in `BrainParams` — would be predicted from sensor noise statistics, target dynamics, and a behavioral loss function rather than fit by hand. Saccade kinetics become a *consequence* of the Bayesian decision problem rather than an independent set of empirical parameters.

## 8. Consequences for the manuscript framework

This note completes the unification:

- **Continuous loops**: Kalman filter on continuous state. (Sect. on `unified_oculomotor_template.md`.)
- **Saccades — execution**: bilinear control-affine, push-pull rectification, main-sequence saturation. (Same manuscript.)
- **Saccades — triggering (decision rate)**: SPRT / drift-diffusion on a binary decision, with Kalman-equivalent gains and `1 + g·activation` modulations. (Sects. 1–5.)
- **Saccades — triggering (evidence)**: Kalman filter with saccade-gated innovation, head-frame prior, delayed observation. Removes the post-saccade delay echo at source, freeing the refractory parameters from artifact-suppression duty. (Sect. 6.)

All four are instances of the same general principle: *Bayesian inference under uncertainty, with state-dependent gains that the cerebellum can adapt*. The differences are problem-class differences (state estimation vs. decision under uncertainty) and output-class differences (continuous policy vs. discrete commit), not differences in the underlying machinery.

## References

- Wald, A. (1947). *Sequential Analysis.* Wiley. (SPRT and optimal stopping.)
- Ratcliff, R., & McKoon, G. (2008). The diffusion decision model: theory and data for two-choice decision tasks. *Neural Computation* 20, 873–922.
- Bogacz, R., Brown, E., Moehlis, J., Holmes, P., Cohen, J. D. (2006). The physics of optimal decision making: a formal analysis of models of performance in two-alternative forced-choice tasks. *Psychological Review* 113, 700–765.
- Carpenter, R. H. S., & Williams, M. L. L. (1995). Neural computation of log likelihood in control of saccadic eye movements. *Nature* 377, 59–62.
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Trans. ASME J. Basic Engineering* 82, 35–45.
- Robinson, D. A. (1975). Oculomotor control signals. In *Basic Mechanisms of Ocular Motility and their Clinical Implications*, ed. G. Lennerstrand and P. Bach-y-Rita, 337–374. Pergamon.
- Smith, O. J. M. (1957). Closer control of loops with dead time. *Chemical Engineering Progress* 53, 217–219. (Smith predictor — delay-aware feedback.)
- Wolpert, D. M., Ghahramani, Z., & Jordan, M. I. (1995). An internal model for sensorimotor integration. *Science* 269, 1880–1882.
