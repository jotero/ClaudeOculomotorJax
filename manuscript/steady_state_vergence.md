# Steady-state vergence in the proximal/tonic/AC-A model

*Draft note*

## 1. Model summary

The vergence–accommodation system has six relevant coupled states: a fast vergence integrator $x_v$, a slow tonic-vergence integrator $x_t$, a fast accommodation integrator $x_a$, a slow tonic-accommodation integrator $x_s$, plus the cross-link drives $aca$ (accommodation $\to$ vergence) and $cac$ (vergence $\to$ accommodation).

Continuous-time dynamics (Schor 1986 dual-integrator + Hung-Semmlow proximal injection):

$$
\begin{aligned}
\dot{x}_v   &= K_{\mathrm{verg}}\,(\mathrm{disparity}) - x_v / \tau_{\mathrm{verg}} + p_{\mathrm{verg}} / \tau_{\mathrm{verg}}, \\
\dot{x}_t   &= \bigl( v_{\mathrm{tonic}} + K_t\,(x_v + aca) - x_t \bigr) / \tau_t, \\
\dot{x}_a   &= K_{\mathrm{acc}}\,(\mathrm{defocus}) - x_a / \tau_{\mathrm{acc}} + p_{\mathrm{acc}} / \tau_{\mathrm{acc}}, \\
\dot{x}_s   &= \bigl( a_{\mathrm{tonic}} + K_s\,(x_a + cac) - x_s \bigr) / \tau_s.
\end{aligned}
$$

The cross-links use the *total* state of each subsystem (rather than only the phasic component):

$$
aca = \alpha\,(x_a + x_s), \qquad
cac = \beta\,(x_v + x_t),
$$

with $\alpha = \mathrm{AC/A} \cdot 0.5729$ deg/D and $\beta = \mathrm{CA/C} / 0.5729$ D/deg.

The proximal cue is a **single perceived-distance parameter** $d$ (in diopters). It injects a state-independent constant drive into both fast integrators:

$$
p_{\mathrm{verg}} = d \cdot \mathrm{IPD} \cdot \tfrac{180}{\pi} \equiv d \cdot \mathrm{IPD}_f, \qquad
p_{\mathrm{acc}} = d.
$$

The full vergence and accommodation outputs are

$$
V = x_v + x_t + aca, \qquad
A = x_a + x_s + cac.
$$

## 2. Steady-state notation

Set all time derivatives to zero. Define the cross-coupling loop gain

$$
L \;=\; K_t \cdot \alpha \cdot \beta \cdot K_s.
$$

For typical values ($K_t = K_s = 1.5$, $\mathrm{AC/A} = 2$ pd/D, $\mathrm{CA/C} = 0.08$ D/pd) we have $\alpha \approx 1.146$, $\beta \approx 0.140$, $L \approx 0.36$, and the cross-coupling amplification $1/(1-L) \approx 1.56$.

## 3. Steady-state outputs

The fast integrators settle at

$$
x_v^{\ast} = K_{\mathrm{verg}} \tau_{\mathrm{verg}} \cdot \mathrm{disparity} + p_{\mathrm{verg}}, \qquad
x_a^{\ast} = K_{\mathrm{acc}} \tau_{\mathrm{acc}} \cdot \mathrm{defocus} + p_{\mathrm{acc}}.
$$

The slow integrators settle at their setpoints plus an adaptation term:

$$
x_t^{\ast} = v_{\mathrm{tonic}} + K_t\,(x_v^{\ast} + aca^{\ast}), \qquad
x_s^{\ast} = a_{\mathrm{tonic}} + K_s\,(x_a^{\ast} + cac^{\ast}).
$$

After solving the coupled $aca$–$cac$ equations,

$$
\boxed{
V^{\ast} \;=\; \frac{1}{1-L} \,\Bigl[\, x_v^{\ast}\,(1 + K_t) + v_{\mathrm{tonic}} + \alpha\,(1 + K_t)\,\bigl( x_a^{\ast}\,(1 + K_s) + a_{\mathrm{tonic}} \bigr) \,\Bigr]
}
$$

$$
A^{\ast} \;=\; \frac{1}{1-L} \,\Bigl[\, x_a^{\ast}\,(1 + K_s) + a_{\mathrm{tonic}} + \beta\,(1 + K_s)\,\bigl( x_v^{\ast}\,(1 + K_t) + v_{\mathrm{tonic}} \bigr) \,\Bigr].
$$

Each box is **linear** in the four scalars $\{p_{\mathrm{acc}},\,p_{\mathrm{verg}},\,a_{\mathrm{tonic}},\,v_{\mathrm{tonic}}\}$ plus the (closed-loop) values of $\mathrm{disparity}$ and $\mathrm{defocus}$.

## 4. Three reference conditions

Define the gain factor $G = (1+K_t)/(1-L)$. With $K_t = 1.5$, $L = 0.36$: $G \approx 3.91$.

### Case A: dark, no visual input

$\mathrm{disparity} = 0$, $\mathrm{defocus} = 0$, so $x_v^{\ast} = p_{\mathrm{verg}} = d \cdot \mathrm{IPD}_f$ and $x_a^{\ast} = p_{\mathrm{acc}} = d$.

$$
V_{\mathrm{dark}} \;=\; G \cdot d \cdot \mathrm{IPD}_f \;+\; \frac{v_{\mathrm{tonic}}}{1-L} \;+\; G \alpha \, \bigl( d (1 + K_s) + a_{\mathrm{tonic}} \bigr).
$$

### Case B: monocular fixation at 40 cm, lens-corrected so eye accommodates to demand

Defocus closed-loop drives $A^{\ast} = 1/0.4 = 2.5$ D. Disparity $= 0$ (only one eye sees). The proximal contribution to $x_v$ stays as $d \cdot \mathrm{IPD}_f$. Total accommodation contributes through ACA at the new closed-loop level:

$$
V_{\mathrm{40\,cm}} \;=\; V_{\mathrm{dark}} \;+\; G\alpha\, \bigl( A^{\ast}_{\mathrm{40}} - x_a^{\ast}\,(1 + K_s) - a_{\mathrm{tonic}} \bigr) \cdot \tfrac{1}{1+K_s},
$$

or more compactly,

$$
V_{\mathrm{mono}}(d_{\mathrm{tgt}}) \;=\; G \cdot d \cdot \mathrm{IPD}_f \;+\; \frac{v_{\mathrm{tonic}}}{1-L} \;+\; G \alpha \cdot A^{\ast}(d_{\mathrm{tgt}}),
$$

where $A^{\ast}(d_{\mathrm{tgt}}) \approx 1/d_{\mathrm{tgt}}$ (closed-loop accommodation tracks demand). For $d_{\mathrm{tgt}} = 0.4$ m, $A^{\ast} = 2.5$ D.

### Case C: monocular fixation at 4 m

Same formula with $A^{\ast}(4) = 0.25$ D:

$$
V_{\mathrm{4\,m}} \;=\; G \cdot d \cdot \mathrm{IPD}_f \;+\; \frac{v_{\mathrm{tonic}}}{1-L} \;+\; G \alpha \cdot 0.25.
$$

## 5. Numerical example

Default values: $\mathrm{AC/A} = 2$, $\mathrm{CA/C} = 0.08$, $K_t = K_s = 1.5$, IPD $= 64$ mm, $v_{\mathrm{tonic}} = 5°$, $a_{\mathrm{tonic}} = 1$ D, $d = 1$ D.

Plugging in: $\mathrm{IPD}_f = 3.67$, $\alpha = 1.146$, $L = 0.36$, $G = 3.91$, $1/(1-L) = 1.56$.

| Condition | $V^{\ast}$ contribution from each term |
|---|---|
| proximal $G \cdot 1 \cdot 3.67$ | $14.4°$ |
| tonic $v_{\mathrm{tonic}}/(1-L)$ | $7.8°$ |
| $G \alpha \cdot a_{\mathrm{tonic}}$ | $4.5°$ |
| $G \alpha \cdot d \cdot (1+K_s)$ | $11.2°$ |
| **Dark $V^{\ast}$** | **$\approx 37.9°$** |
| Monocular 40 cm: + $G \alpha \cdot 2.5$ | $+ 11.2°$ → $V \approx 49.1°$ |
| Monocular 4 m: + $G \alpha \cdot 0.25$ | $+ 1.1°$ → $V \approx 39.0°$ |

The dark-vs-monocular separation is set by accommodation: $V_{\mathrm{40\,cm}} - V_{\mathrm{dark}} = G \alpha \cdot 2.5 \approx 11°$.

## 6. Practical implications

1. **Linear superposition of effects.** The contribution of each term ($p_{\mathrm{verg}}$, $p_{\mathrm{acc}}$, $v_{\mathrm{tonic}}$, $a_{\mathrm{tonic}}$, $A^{\ast}$) to $V^{\ast}$ is independent. Tuning one shifts the total by an exact algebraic amount.
2. **Cross-coupling amplification $1/(1-L)$.** Loop gain $L = K_t \alpha \beta K_s$ shows up as an overall multiplicative amplification on every term. Set $\mathrm{AC/A}$ and $\mathrm{CA/C}$ to control it.
3. **Proximal contributes both directly (via $p_{\mathrm{verg}}$) and through accommodation (via $\alpha \cdot p_{\mathrm{acc}}$).** A pure increase in $d$ raises vergence by $G \cdot \mathrm{IPD}_f + G \alpha (1 + K_s) \approx 9.5°/$D — large.
4. **Tonic vergence has only a $1/(1-L) \approx 1.56$ amplification**; tonic accommodation has an $G\alpha \approx 4.5$/D amplification (an extra factor $\alpha (1+K_t) \approx 2.9$). Tonic accommodation is therefore a more "leveraged" knob in this architecture.

## References

- Schor, C. M. (1986). The relationship between fusional vergence eye movements and fixation disparity. *Vision Research* 19, 1359–1367.
- Hung, G. K. and Semmlow, J. L. (1980). Static behavior of accommodation and vergence: computer simulation of an interactive dual-feedback system. *IEEE Trans. Biomedical Engineering* 27, 439–447.
- Maddox, E. E. (1893). *The Clinical Use of Prisms.* Bristol: J. Wright.
