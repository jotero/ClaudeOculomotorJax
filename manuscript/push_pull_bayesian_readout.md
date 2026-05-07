# Push-pull rectification as Bayesian MAP inference

*Draft note*

## Abstract

The push-pull architecture — a signed quantity encoded as the difference of two rectified non-negative channels with thresholds — natively performs Bayesian MAP inference. With additive Gaussian noise on the input, the difference of rectified channels is the soft-thresholding estimator, exactly the MAP estimate under a sparse Laplace prior, with the rectifier offset equal to the noise-to-prior ratio. Inference is not an additional computation imposed on the architecture; it falls out of the subtraction. Non-Gaussian noise distributions are handled by upstream variance-stabilising transformations (square-root, logarithmic), explaining why other nonlinearities accompany the rectified-pair motif throughout the brain. Temporal integration is the natural dual under a smoothness prior, giving Wiener and Kalman estimation. We argue that the causal arrow between biology and computation runs the other way around than usually stated: the requirement of inference under uncertainty selects the substrate, rather than biophysics constraining a separately-derived computation.

## 1. Push-pull as a substrate for inference

A signed scalar quantity $s$ cannot be encoded by a single neuron whose firing rate is non-negative. The brain encodes it instead as the difference of two non-negative populations,

$$
s_{\mathrm{encoded}} \;=\; \mathrm{ReLU}(x - \theta) \;-\; \mathrm{ReLU}(-x - \theta),
$$

where $x$ is the input drive and $\theta$ is the firing threshold (resting-rate offset). This motif appears throughout the nervous system: V1 simple cells with on/off receptive fields, vestibular nuclei with bilateral push-pull encoding, neural integrators with left/right populations, dopaminergic reward-prediction error neurons. It is conventionally explained as a biophysical workaround for the non-negativity of firing rates.

We claim it is more than that. Read as an estimator of $s$ from a noisy observation $x$, the push-pull difference is the maximum-a-posteriori estimate under a particular prior and noise model. The architecture does not merely encode the signal — it infers it.

## 2. Push-pull rectification is MAP under sparse-Laplace prior + Gaussian noise

Consider a single scalar observation $x$ of an unknown signal $s$, corrupted by additive Gaussian noise:

$$
x \;=\; s + n, \qquad n \sim \mathcal{N}(0,\,\sigma^2).
$$

Suppose the system holds the prior belief that $s$ is usually zero, with occasional excursions decaying in probability with magnitude — a Laplace prior:

$$
p(s) \;\propto\; \exp\!\left(-\,\frac{|s|}{\tau}\right).
$$

The Laplace prior is the natural sparse prior. It expresses the assumption that "no signal" is the default state and signals of any non-zero magnitude are surprising, with surprise growing linearly in $|s|$ rather than quadratically as in a Gaussian.

The posterior is proportional to prior $\times$ likelihood:

$$
\log p(s \mid x) \;=\; -\,\frac{|s|}{\tau} \;-\; \frac{(x - s)^2}{2\sigma^2} \;+\; \mathrm{const}.
$$

The MAP estimate maximises this expression. Differentiating with respect to $s$,

$$
0 \;=\; -\,\frac{\mathrm{sign}(s)}{\tau} \;+\; \frac{x - s}{\sigma^2}
\quad\Longrightarrow\quad
s \;=\; x \;-\; \mathrm{sign}(s)\cdot\frac{\sigma^2}{\tau}.
$$

Solving, we obtain the **soft-thresholding function**:

$$
s_{\mathrm{MAP}}(x) \;=\; \mathrm{sign}(x)\cdot\max\!\bigl(|x| - \theta,\; 0\bigr),
\qquad \theta \;=\; \frac{\sigma^2}{\tau}.
$$

Equivalently, written as a difference of two rectified channels,

$$
s_{\mathrm{MAP}}(x) \;=\; \mathrm{ReLU}(x - \theta) \;-\; \mathrm{ReLU}(-x - \theta).
$$

This is exactly the push-pull architecture from Section 1. The two pictures — encoding bidirectional signed firing rates, and computing the MAP estimate of a sparse signal under noise — produce the same equation. The Bayesian view fixes the threshold's value at $\theta = \sigma^2/\tau$. Larger noise widens the threshold (more shrinkage); a sparser prior (smaller $\tau$) widens it further.

This identity is established in statistics and signal processing (Donoho 1995; Tibshirani 1996) and underlies the sparse-coding interpretation of V1 simple cells (Olshausen and Field 1996).

## 3. Variance-stabilising transformations explain the non-ReLU nonlinearities

The derivation in Section 2 assumes additive Gaussian noise, and the resulting nonlinearity is rectified linear (with threshold). Real biological noise is often not Gaussian: photon arrivals are Poisson, spike-count variability scales with mean firing rate, contrast-dependent gain in V1 generates noise with variance proportional to mean firing. Yet the rectified-pair motif persists across all these regimes, often preceded or followed by a different nonlinearity — square-root, logarithmic, power-law, or saturating. Why?

Because a single upstream nonlinearity can convert any of these noise distributions into approximately additive Gaussian, after which the standard push-pull soft-thresholding readout of Section 2 is again Bayesian-optimal. This is the **variance-stabilising transformation** principle.

For Poisson noise, the transformation $x \mapsto \sqrt{x}$ (or Anscombe's $\sqrt{x + 3/8}$) makes the variance approximately constant in the transformed space. For multiplicative noise with constant coefficient of variation, the transformation $x \mapsto \log|x|$ does the same. Once the transformed signal has additive Gaussian noise, push-pull rectification on the transformed coordinates implements MAP — and the transformation itself is just a feedforward nonlinearity at the input.

This explains, normatively, why neural circuits combine rectifiers with other nonlinearities. Photoreceptor cascades are approximately square-root (Pugh and Lamb 1993): stabilising Poisson photon noise. Auditory cochlear processing is approximately logarithmic: stabilising multiplicative loudness noise. Cortical firing-rate populations often span several orders of magnitude in a log-linear way: stabilising signal-dependent spike variability. None of these are noise-rejection devices in their own right; they are *coordinate changes* that make the downstream rectified-population readout Bayesian-optimal in its noise model.

The architectural consequence is that any non-ReLU nonlinearity in a sensory pathway is a candidate variance-stabilising transformation. Identifying which transformation, and therefore which noise distribution it presumes, becomes a way to read the implicit noise model the brain has assumed at each stage.

## 4. Temporal integration: the bandwidth–variance trade-off

The push-pull readout in Section 2 produces an instantaneous MAP estimate from a single noisy observation. With a noisy input the estimate is itself noisy — the rectifiers pass noise above threshold, and the difference of rectified channels inherits the input's variance. Temporal integration reduces this variance by averaging the rectified output over time.

The trade-off is exactly bias against variance, expressed along the time axis. Long integration averages over more samples, reducing variance as $1/T$, but also smooths fast changes in the signal: the estimate lags the true $s(t)$. Short integration tracks fast changes but accepts higher variance per estimate. The optimal compromise depends on what the signal *looks* like spectrally compared to the noise.

For a signal with power spectrum $S_{\mathrm{signal}}(\omega)$ corrupted by noise with spectrum $S_{\mathrm{noise}}(\omega)$, the best linear estimator is the **Wiener filter**:

$$
\hat{H}(\omega) \;=\; \frac{S_{\mathrm{signal}}(\omega)}{S_{\mathrm{signal}}(\omega) + S_{\mathrm{noise}}(\omega)}.
$$

It passes frequencies where signal dominates the noise ($S_{\mathrm{signal}}/S_{\mathrm{noise}} \gg 1$) and attenuates frequencies where noise dominates. For low-pass signals with white noise, this collapses to a leaky integrator with cutoff at the SNR-crossover frequency. The integration time constant $\tau \approx 1/(2\pi f_c)$ is set by the signal's bandwidth — not a free knob. The Kalman filter generalises this to non-stationary signals with state-space dynamics; it sets the integration constant adaptively from the prior's predicted state and the measurement variance.

Plugged in immediately downstream of the push-pull readout, this gives the brain's standard inference circuit: **rectified populations $\to$ leaky integrator**. The rectifiers do amplitude MAP per instant (Section 2); the integrator's leak constant performs Wiener-style variance reduction along time. The amplitude axis is feedforward, the time axis is recurrent (the integrator is a self-loop), and together they bound the MAP estimate against both the noise floor and the signal bandwidth. The sequential probability ratio test and drift-diffusion model (Ratcliff 1978; Bogacz et al. 2006) are the case where the integrator output is read against a decision threshold rather than a continuous signal.

## 5. The inverted causal arrow

It is conventional to motivate the push-pull architecture from biophysics: neurons cannot fire at negative rates, so bidirectional signals must be encoded through pairs of rectified channels. The inferential interpretation is then offered as a happy coincidence — biology forced rectifiers, and rectifiers happen to do useful Bayesian computation.

We propose the inverse causation. The functional requirement is inferential: a system that extracts signals from noisy evidence must compute something close to a posterior estimate. The form of that estimate is dictated by the prior and noise model — soft-thresholding for sparse signals in additive Gaussian noise, with non-Gaussian noise distributions handled by upstream variance-stabilising transformations that map them back to the Gaussian case. Evolution found neural substrates that implement these readouts cheaply: rectified populations for soft-thresholding, square-root and logarithmic compressions for variance stabilisation, leaky integrators for temporal Wiener filtering. The biophysical constraints are not antecedent puzzles whose solution happens to be Bayesian; they are the substrates that *can* be Bayesian, which is why they were selected. Optimal inference and biological implementation coincide because the implementations evolved precisely to satisfy the inference requirement.

The push-pull architecture is the canonical example. Two rectified channels with thresholds are not a workaround for non-negative firing rates; they are an inference circuit. The non-negativity is what makes the rectifier biophysically realisable, but the *reason* the brain encodes signed quantities this way — rather than by, say, allowing single neurons to span a wider dynamic range — is that the architecture happens to compute the MAP estimate the brain needs anyway.

## 6. Conclusion

Push-pull rectification implements MAP inference under the simplest non-trivial prior — sparseness — and the simplest non-trivial noise model — additive Gaussian. The threshold has a normative interpretation as the noise-to-prior ratio. Non-Gaussian noise distributions are handled by an upstream variance-stabilising transformation — square-root for Poisson, logarithmic for multiplicative — after which the standard rectified readout applies in the transformed coordinates. Temporal estimation is the natural dual under a smoothness prior, implemented by leaky integrators (Wiener) or recurrent state-space filters (Kalman).

The framing inverts the usual causal arrow between biology and function. It is not that biology imposes constraints under which the brain manages, after the fact, to do something useful. Inference under uncertainty requires a particular form of estimator, and the brain has selected, over evolutionary time, the substrates that implement those estimators cheaply. Rectified populations are favoured because they soft-threshold; square-root and logarithmic compressions are favoured because they stabilise variance; leaky integrators are favoured because they Wiener-filter slow signals against fast noise. The biology and the computation are not separate problems coincidentally aligning; they are the same problem, with the biology supplying the implementation that the computation demands. Choose the prior and the noise model, and the architecture, threshold, and integration time fall out together.

## References

Bogacz, R., Brown, E., Moehlis, J., Holmes, P., and Cohen, J. D. (2006). The physics of optimal decision making: a formal analysis of models of performance in two-alternative forced-choice tasks. *Psychological Review* 113, 700–765.

Donoho, D. L. (1995). De-noising by soft-thresholding. *IEEE Transactions on Information Theory* 41, 613–627.

Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering* 82, 35–45.

Olshausen, B. A. and Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature* 381, 607–609.

Ratcliff, R. (1978). A theory of memory retrieval. *Psychological Review* 85, 59–108.

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society B* 58, 267–288.

Wiener, N. (1949). *Extrapolation, Interpolation, and Smoothing of Stationary Time Series.* MIT Press.
