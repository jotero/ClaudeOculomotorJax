# OVAR / tilt-suppression / T-VOR diagnosis — notes for next session

## TL;DR

The "OVAR backwards modulation" is not a bug — it's a structural tension between K_gd
and the modulation source. We can't currently get both correct OVAR baseline AND correct
OVAR modulation with the same K_gd value.

## Findings

### 1. K_gd is the OVAR modulation killer at high tilts

Reverting K_gd from 2.86 (current) → 0.0 (b861fd9 "ovar and tilt suppression both good!"
default) restores the expected `modulation amplitude ∝ sin(tilt)`:

| tilt | K_gd=2.86 (current) | K_gd=0 (old default) |
|---|---|---|
| 10° pk-pk SPV | 54 | 55 |
| 30° pk-pk SPV | 60 | 75 |
| 60° pk-pk SPV | 57 | 103 |
| 90° pk-pk SPV | 54 | 111 |

`rf = (gia × g_est) / G0²` magnitude scales with sin(tilt). K_gd × rf damps VS, more
strongly at high tilts. So K_gd's damping kills high-tilt modulation while pumping
mean baseline up.

### 2. But K_gd also creates the OVAR sustained-nystagmus baseline

| tilt | K_gd=2.86 mean SPV | K_gd=0 mean SPV |
|---|---|---|
| 10° | -25 | -25 |
| 30° | -33 | -26 |
| 60° | -44 | -26 |
| 90° | -47 | -25 |

Without K_gd, the brain doesn't know it's tilted → no extra yaw drive → flat baseline.
With K_gd, the rotational feedback adds a tilt-dependent push to VS → baseline grows
correctly.

### 3. What creates "modulation" at K_gd=0?

It's **T-VOR coupling**, not VS. Chain:
- gia oscillates in head frame (gravity rotates as head spins)
- residual = gia - g_est - a_lin oscillates
- a_lin captures the oscillation (K_lin × residual)
- v_lin integrates a_lin
- T-VOR omega = -g_hat × v_lin / D oscillates
- Adds to NI velocity → eye velocity oscillates

So K_gd=0 modulation has nothing to do with the L&A 2011 OVAR-modulation mechanism.

### 4. The structural tension

L&A 2011 / 2017 OVAR shows BOTH:
- Sustained nystagmus baseline ∝ sin(tilt) — from K_gd
- Modulation amplitude ∝ sin(tilt) — also expected from rf

But in our model, K_gd kills modulation while creating baseline. Two interpretations:
- (a) K_gd's rf-damping is too aggressive at high tilts. A non-linear or saturating
      gain might preserve modulation while still creating baseline.
- (b) The L&A modulation actually comes from a DIFFERENT mechanism (e.g., otolith
      phasic afferents) that we don't model. K_gd does the baseline correctly;
      modulation needs a separate mechanism.

I lean toward (b) since rf damping by definition reduces oscillation, not amplifies it.

### 5. Tilt-suppression depends on the canal gate

The earlier tilt-suppression result (v_lin = 0.006 m/s) was with `w_canal_gate=5`.
Currently the gate is disabled (1e6) and tilt-suppression is back to v_lin = 0.32 m/s
across all K_gd values. **The gate matters for tilt-suppression even though it didn't
help OVAR.**

| config | tilt v_lin (m/s) |
|---|---|
| gate=5, K_gd=2.86 | **0.006** |
| gate=∞, K_gd=2.86 | 0.32 |
| gate=∞, K_gd=0.0 | 0.44 |

### 6. VOR post-rot TC vs OKAN TC mismatch

| metric | current | target |
|---|---|---|
| VOR post-rot τ | 35–93s | 15–20s |
| OKAN τ | 19.9s | ~20s |
| VVOR yaw gain | 0.86 | >0.85 |

OKAN matches; VOR post-rot is too long. The mismatch is from null-adaptation
(`tau_vs_adapt=600s`) tail interacting differently with canal-driven VS charging
vs slip-driven VS charging. Tau_vs alone can't fix without breaking OKAN.

## Recommended next-session plan

In priority order:

### P1: Decide the K_gd vs OVAR-modulation tension
Either:
- (a) Accept K_gd=2.86 with flat modulation, document the gap
- (b) Add phasic otolith afferent → would create real OVAR modulation independent of K_gd
- (c) Try K_gd=0.5–1.0 as a compromise (some baseline, some modulation)

### P2: Re-enable canal gate
Set `w_canal_gate = 5.0` again — needed for tilt-suppression. Doesn't break OVAR
(verified above: gate=5 vs gate=∞ gives same OVAR pattern).

### P3: VOR post-rot TC investigation
The fitted TC of 35–93s is dominated by the slow null-adaptation tail. Either:
- Shorten `tau_vs_adapt` so the slow component decays in VOR test window — but check it doesn't break PAN modeling
- Or fit only the fast portion (first ~τ_vs of decay) so the slow tail doesn't bias

### P4: T-VOR DARK gain gap
Laurens 2011 architecture+values gives 0.08, P&T 1991 reports 0.5–0.7. Not
fixable by parameters. Needs:
- Phasic otolith (improves transient, won't fix plateau gap)
- OR a longer-TC translational integrator (tau_head currently 2s, P&T plateau is 1.6s)
- OR accept the gap and document

## Current parameter state (as of writing)

```python
K_grav       = 0.6     # Laurens 2011 "go"
K_lin        = 0.1     # Laurens 2011
tau_a_lin    = 1.5     # Laurens 2013 translation prior
w_canal_gate = 1e6     # disabled — RE-ENABLE TO 5.0 for tilt-suppression
K_gd         = 2.86    # Laurens 2011 — creates baseline, kills modulation
tau_vs       = 20.0    # Cohen 1977
tau_vs_adapt = 600.0   # default — extends VOR post-rot TC
```
