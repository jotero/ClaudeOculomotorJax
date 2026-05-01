# The gaze-holding integrator imposes a Hering prior that bounds strabismus compensation

*Draft manuscript*

## Abstract

Comitant strabismus is conventionally framed as either a peripheral plant abnormality or a central deficit of binocular fusion. We argue for a third framing, complementary to both: that a substantial component of the residual binocular misalignment that survives long-standing adaptation is a signature of structural priors in the recurrent connectivity of the binocular neural integrator. The integrator's continuous-attractor architecture imposes a specific manifold of held postural states on motoneuron activity space, and this manifold geometry, rather than any feedforward yoking rule, is the operative structural prior on binocular alignment. Under symmetric plant geometry, the natural manifold structure — a low-dimensional sheet decomposing into conjugate and vergence directions — is exactly what binocular alignment requires. Under any asymmetric or gaze-dependent plant, including the asymmetries introduced by extraocular muscle weakness, pulley heterotopia, or surgical reroutings, the alignment-correct manifold differs from the natural one, and reshaping the integrator's connectivity to track the correct manifold incurs both a homeostatic regularization cost and a stability cost. Adaptation finds a compromise on this trade-off, and the residual deviation has a characteristic joint signature: persistent alignment error coupled with subclinical gaze-holding abnormalities. This framing reinterprets several empirically puzzling observations — spread of comitance, the partial efficacy of disconjugate adaptation, the co-occurrence of strabismus with subtle integrator pathology, the developmental sensitivity of binocular alignment — as predictable consequences of a single mechanism. We formalize the framework as a constrained optimization on the integrator's recurrent weight matrix and identify the predicted residuals analytically for the linearized regime.

## 1. Introduction

The binocular oculomotor plant is a 12-input, 6-output device, and its biological controller has access to vastly more degrees of freedom than the alignment problem strictly requires. Yet binocular misalignment is common, and adaptation often fails to fully correct it even given decades of consistent error feedback. The mismatch between abundant peripheral controllability and incomplete adaptive correction is the puzzle this paper is concerned with.

The traditional framing of this puzzle invokes Hering's law of equal innervation, the proposition that the brain issues a unified conjugate command plus a unified vergence command, and that this yoked architecture limits what adaptation can install. A century of clinical thinking, and the standard interpretation of the Hess screen, builds on this premise. But the strict-Hering account has been steadily eroded by experimental work showing that the system has more than two adaptive degrees of freedom: dichoptic saccade adaptation produces opposite-direction corrections in the two eyes, premotor neurons in the paramedian pontine reticular formation encode monocular rather than purely conjugate commands, and saccade-vergence interactions reveal mixed rather than purely additive control. If the brain were strictly Hering, none of this should occur.

We propose that the structural prior responsible for residual misalignment lives at a different level of the system entirely: not in a feedforward yoking rule on the command pathway, but in the recurrent dynamics of the neural integrator. The integrator is the postural controller. Its connectivity defines the manifold of stable held states the system can occupy, and this manifold has a specific low-dimensional geometry that emerges from the requirement of accurate gaze-holding under metabolic and stability constraints. That geometry is the binocular alignment prior. It is more constraining than feedforward yoking would be, because reshaping it to compensate an asymmetric plant simultaneously imposes regularization and stability costs that adaptation must trade off against the benefit of corrected alignment. The residual after adaptation saturates is therefore not a failure of plasticity per se but the unique optimum of a multi-criterion objective with no jointly satisfying solution.

The remainder of the paper develops this proposal. Section 2 reviews the Hering picture and its empirical revisions. Section 3 develops the integrator-as-postural-controller view, drawing on the continuous attractor framework. Section 4 formalizes a linearized binocular integrator model, identifies the held-state manifold, and characterizes the achievable set of binocular postures. Section 5 introduces plant asymmetry and derives the residual deviation under a homeostatic learning objective with a stability constraint. Section 6 lays out predicted clinical signatures, and Section 7 discusses scope and limitations.

## 2. From Hering's law to its empirical revisions

Hering's 1868 proposal that the two eyes are innervated as a single organ shaped a century of oculomotor thought. In its strong form, every binocular eye movement is generated by the sum of a conjugate command, applied identically to corresponding muscles of the two eyes, and a vergence command, applied antisymmetrically. The neural correlate this implies is a controller in which descending signals split into exactly two channels, distributed by fixed, anatomically determined yoking ratios.

The strong form has been falsified at multiple levels. Zhou and King (1998) showed that PPRF neurons previously assumed to encode conjugate velocity commands actually encode monocular saccadic commands, with separate populations driving each eye. The Mays group has documented saccade-vergence burst neurons whose activity does not factor into independent conjugate and vergence components. Maiello, Harrison, and Bex (2016) used a dichoptic adaptation paradigm to show that the saccadic system can install opposite-direction corrections in the two eyes, demonstrating an adaptive degree of freedom that strict Hering forbids. Schor's phoria adaptation work, and the Maxwell/Schor extensions to context-dependent vergence adaptation, similarly establish that the brain has more adaptive parameters available than yoking allows.

A weaker form of the Hering claim survives: that the brain's controller class includes Hering-like decompositions as natural axes, even if it does not strictly enforce them. The Hybrid Binocular Control framework (Ramat and colleagues) makes this explicit, combining a yoked conjugate subsystem with two slower monocular controllers. The empirical evidence, taken together, supports this softer view. Binocular control is approximately Hering for fast transient movements and increasingly disconjugate for slower postural and adaptive components.

This empirical landscape exposes the question we want to address. If the controller class is richer than strict Hering, then strict Hering is the wrong place to look for the structural prior that limits adaptation. The Hering decomposition is a natural basis but not a hard constraint. So where does the actual constraint live, and why does it produce residual deviations with a characteristic clinical structure?

Our answer: it lives in the integrator. The integrator is what enforces stable held postures, and its connectivity must satisfy stability constraints that the feedforward command pathway does not. The strong form of Hering is approximately what the integrator's natural connectivity geometry produces, and the limits on adaptive correction are limits on how far the integrator's connectivity can be reshaped without compromising the gaze-holding function it exists to perform.

## 3. The integrator as postural controller

The neural integrator for horizontal eye position resides in the nucleus prepositus hypoglossi and adjacent medial vestibular nucleus; the vertical-torsional integrator is in the interstitial nucleus of Cajal. Functionally, the integrator transforms transient velocity commands into the tonic motoneuron firing rates required to hold the eye against orbital elastic forces. Mathematically, it implements an approximate temporal integration whose long time constant arises from recurrent excitation tuned to produce a near-zero eigenvalue in the dynamics matrix.

The continuous-attractor framework, introduced for oculomotor integration by Seung (1996) and developed by Cannon and Robinson, makes this precise. A network with appropriately tuned recurrent connectivity has a continuous one-parameter family of stable fixed points — a line attractor. Drive along the attractor shifts the network to a new fixed point; perturbations orthogonal to the attractor decay back. The motoneuron readout of the attractor state is the held eye position. Multistable attractors of higher dimension extend the same idea to multiple integrated channels: a 3D attractor manifold supports the held states of a single eye in 3D, a 6D manifold supports binocular held states.

The integrator is therefore not a passive intermediary between command and motoneurons. It is the postural controller, and the geometry of its attractor manifold is the geometry of the achievable set of held postures. Whatever the descending command does transiently, the only states the system can dwell in are those on the manifold.

This recasts the binocular alignment problem. Aligned binocular fixation requires that for every gaze direction, the integrator can settle into a state whose motoneuron readout produces a binocularly aligned eye configuration. The set of aligned binocular held states is a 6D submanifold of the 12D joint motoneuron rate space (one rate per muscle), parameterized by version and vergence target. Whether the integrator can occupy these states depends on whether the integrator's attractor manifold, projected through the plant, contains them. Adaptation can reshape the integrator's connectivity, and therefore the manifold, but only within the limits imposed by the integrator's other functional requirements: stability of held states, low metabolic cost, and accurate integration of velocity commands into position.

The homeostatic learning framework developed in our prior work captures these requirements as a constrained optimization. The recurrent weight matrix is regularized toward minimal recurrent excursion under behaviorally relevant input statistics, while a data fidelity term enforces accurate integration. We propose that binocular alignment fits into this framework as an additional fidelity term, demanding that the manifold's projection through the plant include the binocularly aligned configurations. The natural alignment of the three terms — minimal recurrence, accurate integration, aligned held states — under a symmetric plant produces the empirically observed near-Hering decomposition. Their misalignment under an asymmetric plant produces the residual.

## 4. A linear theoretical framework

Let $r \in \mathbb{R}^N$ be the activity of integrator neurons projecting to extraocular motoneurons, with $N$ large. The integrator obeys

$$\tau \dot{r} = -r + W r + B_c c(t)$$

where $W$ is the recurrent connectivity, $c(t)$ is the descending command, and $B_c$ is the input projection. Let $A = W - I$. Held states are zeros of $\dot{r}$, given by the kernel of $A$ when $c \equiv 0$. For graded held states across a continuous range of fixations, $A$ must have a near-zero subspace of dimension equal to the number of postural degrees of freedom — six for binocular control.

The motoneuron firing rates derive from $r$ by a fixed projection, $m = M r$, with $M$ a $12 \times N$ matrix. The eye configuration produced by motoneuron rates $m$ is determined by the plant, $\theta = P m$, with $P$ a $6 \times 12$ matrix encoding the muscle pulling-direction Jacobian and tonic length-tension properties. Composing, the held binocular eye configuration achievable from integrator state $r$ is $\theta = P M r$.

The achievable set of held binocular configurations is therefore

$$\mathcal{H} = \{ P M r : r \in \ker A \}.$$

The structural prior on binocular alignment is the geometry of $\mathcal{H}$ as determined by the structure of $\ker A$. The dimension of $\mathcal{H}$ is at most $\dim(\ker A)$, and equals it generically when $PM$ has full rank on $\ker A$. We assume this throughout: $\dim \ker A = 6$ and the projection is full rank, so $\mathcal{H}$ is a 6D linear subspace of the 12D motoneuron-rate-image $\mathbb{R}^6$ of plant outputs.

Under symmetric plant geometry — left and right plants related by mirror reflection — the standard Hering decomposition emerges naturally. Decompose $\ker A = K_v \oplus K_d$, where $K_v$ is a 3D conjugate subspace, in which neurons projecting to yoked muscle pairs co-fire, and $K_d$ is a 3D disjunctive subspace, in which they antifire. Under symmetry, $P M$ maps $K_v$ to the 3D conjugate eye configuration subspace and $K_d$ to the 3D vergence subspace, and these two subspaces are exactly the standard binocular alignment basis. Adaptation can independently scale these channels and add tonic offsets — corresponding to phoria adjustments in the language of the clinical literature — within the natural Hering basis without restructuring $\ker A$ at all. This is why concomitant deviations of small magnitude adapt smoothly: they fit inside the homeostatic optimum.

Now suppose the plant is asymmetric: $P$ no longer has the symmetric block structure that aligns the natural Hering kernel with the binocular alignment basis. The set of motoneuron rate patterns producing binocularly aligned eye configurations is now a different 6D subspace $\mathcal{R}^* \subset \mathbb{R}^N$, related to but rotated away from the natural Hering kernel. For perfect alignment, we would need $\ker A = \mathcal{R}^*$. The question becomes whether adaptation can reshape $A$ to install this kernel, and at what cost.

## 5. Symmetric versus asymmetric plants and the stability tax

Reshaping $\ker A$ to align with $\mathcal{R}^*$ amounts to perturbing the recurrent weight matrix $W$ such that the new dynamics matrix $A' = W' - I$ has its near-zero eigenvalue subspace along $\mathcal{R}^*$ rather than the natural Hering directions. Two costs constrain how far this reshaping can go.

The first is the homeostatic regularization cost. Following our prior framework, the learning objective penalizes recurrent excursion via a term proportional to $\text{tr}(A C A^T)$, where $C$ is the covariance of integrator activity under natural behavior. This term favors weight matrices that are small in directions that are heavily exercised. Reshaping the kernel to include disconjugate-rate directions — directions in which yoked pairs fire unequally — requires nonzero entries in $W$ along those directions. If natural binocular behavior has the disconjugate-rate directions of $\mathcal{R}^*$ at low variance under $C$, then installing them in $\ker A$ requires the regularizer to absorb a cost it would prefer not to pay.

The second is the stability cost. The integrator's gaze-holding function requires that the eigenvalues along $\ker A$ be precisely zero. Generic perturbations to $W$ that move the kernel direction also tend to push some eigenvalues slightly positive eigenvalues produce centrifugal drift (runaway), negative eigenvalues produce centripetal drift (gaze-evoked nystagmus, the common clinical finding). Maintaining zero eigenvalues exactly along a non-natural kernel requires fine-tuning of $W$ in directions that the homeostatic regularizer pushes toward zero. The set of weight matrices with both the right kernel direction and exactly-zero eigenvalues along it is a lower-dimensional submanifold of weight space, and adaptation operating on a regularized objective will not generically converge to it.

The full adaptive optimum is the solution of a constrained minimization,

$$\min_W \; \alpha \, \mathcal{E}_{\text{align}}(W; P) + \beta \, \mathcal{E}_{\text{integrate}}(W) + \gamma \, \text{tr}(W C W^T) \quad \text{s.t. stability of} \; \ker(W-I),$$

where $\mathcal{E}_{\text{align}}$ measures binocular misalignment averaged over fixations, $\mathcal{E}_{\text{integrate}}$ measures integration accuracy, and the trace term is the homeostatic regularizer. Under a symmetric plant, all four terms are jointly minimized at the same $W$: the natural Hering integrator. Under an asymmetric plant, no single $W$ jointly minimizes them, and the optimum is a compromise.

The structure of this compromise determines what the residual looks like. Linearizing around the symmetric optimum and treating plant asymmetry as a small perturbation $\delta P$, the residual misalignment scales linearly with $\delta P$, while the additional homeostatic and stability costs scale quadratically with the weight perturbation $\delta W$ that the system installs. The optimum sets $\delta W$ proportional to $\delta P$ with a coefficient determined by the regularization weights. Critically, the residual misalignment and the residual stability cost are coupled: reducing one increases the other. So the predicted clinical fingerprint of an asymmetric plant under chronic adaptation is not pure misalignment, and not pure gaze-holding pathology, but a specific linear combination of the two whose ratio is a property of the regularization landscape.

This is the stability tax. The brain cannot install the correct manifold geometry for free. Every degree of disconjugate-rate kernel structure carries a near-zero-eigenvalue perturbation that manifests as subclinical drift. The empirically observed coexistence of strabismus with subtle integrator-like signs — gaze-evoked nystagmus, post-saccadic drift, rebound nystagmus — is a direct prediction of this framework, not a coincidence to be explained by independent pathologies.

## 6. Predictions and clinical correspondence

Several otherwise-puzzling clinical observations follow naturally from the framework.

Spread of comitance — the gradual conversion of an initially incomitant deviation into a comitant one over months — is, in this view, not a failure of adaptation but the homeostatic regularizer's preferred solution. Faced with an asymmetric plant, the system can either reshape the kernel to compensate (high homeostatic cost, residual stability tax) or shift the entire manifold so that all gaze directions are equally affected (low homeostatic cost because the kernel structure is preserved, residual is uniform misalignment). The latter is energetically preferred and is what is observed clinically.

The developmental sensitivity of binocular alignment follows from the plasticity profile of the integrator. The integrator's basic kernel structure is set during early development, when the relevant behavioral statistics that shape $C$ are still being established. Plant asymmetries present during this period can be absorbed into the natural kernel as it forms; asymmetries arising after the kernel consolidates require perturbation of an already-stabilized structure and pay the full stability tax. This predicts that infantile strabismus and adult-onset palsies should produce qualitatively different residuals: developmental cases consolidate into kernels that are intrinsically miscalibrated but stable, while acquired cases install perturbed kernels that always carry residual stability costs. Existing clinical observations of stronger gaze-holding abnormalities in acquired than developmental strabismus are consistent with this distinction, though the comparison has not been quantified systematically.

Disconjugate adaptation paradigms (Maiello et al. 2016; Bucci & Kapoula 1997) probe the system's ability to install non-Hering structure on a short timescale. Our framework predicts that such adaptation is possible because the kernel of $A$ is locally plastic, but that the magnitude of achievable disconjugate adaptation is bounded by the regularization weights. Strabismic subjects, whose kernels have already been perturbed chronically, should show reduced range of additional disconjugate adaptation, consistent with Bucci and Kapoula's findings.

Vertical and torsional alignment should be more refractory to adaptation than horizontal. This follows because the relevant adaptive error signals (vertical and cyclo-disparity) are sparser and weaker than horizontal disparity, so the data fidelity term has less leverage relative to the regularizer. Clinically, vertical phorias are smaller in magnitude but harder to fully resolve, and torsional alignment is famously poorly adapted — both consistent with the framework.

The framework also predicts a class of experiments not yet performed. If adaptation is forced via prism or post-surgical realignment, the recovered alignment should come bundled with measurable integrator-level changes — small drifts, time-constant shifts, altered gaze-holding statistics. The signature should be predictable from the magnitude of the imposed alignment shift. To our knowledge no study has measured high-precision integrator function before and after surgical alignment in adult strabismus; this would be a direct test.

## 7. Discussion

We have argued that the structural prior limiting adaptive compensation of binocular misalignment lives in the recurrent connectivity of the neural integrator, not in any feedforward yoking rule. The integrator's continuous-attractor architecture imposes a manifold of held postural states whose geometry is set by the joint requirements of stability, integration accuracy, and homeostatic minimal-recurrence. Under symmetric plants this manifold's natural geometry is approximately Hering, which is why the Hering description has explanatory force. Under asymmetric plants the alignment-correct manifold differs from the natural one, and adaptation cannot install it without paying coupled regularization and stability costs. The residual deviation has a characteristic joint signature in alignment and gaze-holding that distinguishes it from peripheral plant abnormalities.

The framing has two virtues over the strict-Hering account. First, it is consistent with the empirical evidence that the brain has more than two adaptive channels, because the constraint is on the integrator's manifold geometry rather than on the feedforward command pathway. Second, it makes coupled predictions about alignment and gaze-holding that the feedforward account does not, opening clear empirical tests.

Several limitations should be flagged. The linearized treatment in Section 5 is valid only for small plant perturbations; severe pulley dislocations or muscle palsies require a nonlinear extension where the manifold geometry varies with gaze. Listing's-law-related torsional structure and the half-angle rule introduce additional gaze-dependent constraints that interact nontrivially with the integrator's attractor structure; we have treated these as outside the scope of the present analysis. The relationship between motoneuron rate patterns and integrator activity, which we have collapsed into the projection $M$, is in reality an active read-out involving brainstem premotor nuclei whose own dynamics may further constrain the achievable set. And the behavioral covariance $C$ that drives the homeostatic regularizer is itself shaped by the available integrator structure, producing a self-referential loop whose fixed-point analysis we have not undertaken here.

Despite these limitations, the central proposal stands as a tractable theoretical claim with concrete predictions: the locus of structural priors on binocular alignment is the integrator, the predicted residual signature is coupled misalignment and gaze-holding pathology, and the developmental versus acquired distinction follows from when in the integrator's plasticity timeline the asymmetry arrived. Whether this framework explains a substantial fraction of the variance in clinical strabismus, or only a particular slice of it, is an empirical question that the predictions in Section 6 are designed to address.

## References

(To be completed; key citations to integrate)

Cannon, S. C., & Robinson, D. A. (1987). Loss of the neural integrator of the oculomotor system from brainstem lesions in monkey. *J. Neurophysiol.*

Seung, H. S. (1996). How the brain keeps the eyes still. *PNAS.*

Goldman, M. S., Compte, A., & Wang, X.-J. (Continuous attractor models of the integrator.)

Schor, C. M., & colleagues. (Phoria adaptation, fast and slow components.)

Maxwell, J. S., & Schor, C. M. (Context-dependent vergence adaptation.)

King, W. M. (2011). Binocular coordination of eye movements — Hering's law of equal innervation or uniocular control? *Vision Research.*

Zhou, W., & King, W. M. (1998). Premotor commands encode monocular eye movements. *Nature.*

Mays, L. E., Porter, J. D., Gamlin, P. D., & colleagues. (Saccade-vergence burst neurons.)

Maiello, G., Harrison, W. J., & Bex, P. J. (2016). Monocular and binocular contributions to oculomotor plasticity. *Scientific Reports.*

Bucci, M. P., & Kapoula, Z. (1997). Deficiency of adaptive control of the binocular coordination of saccades in strabismus. *Vision Research.*

Tweed, D. (1997). Visual-motor optimization in binocular control. *Vision Research.*

Demer, J. L. (Pulley heterotopias and their oculomotor consequences.)

Quaia, C., & Optican, L. M. (Saccadic control models.)

Ramat, S., & colleagues. (Hybrid Binocular Control model.)

Todorov, E., & Jordan, M. I. (2002). Optimal feedback control as a theory of motor coordination. *Nature Neuroscience.*
