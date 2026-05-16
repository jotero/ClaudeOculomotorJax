# Plant compensation — analytical derivation

The brainstem-to-eye motor chain in this model is

```
NI_net  ──[ pulse-step ]──>  motor_cmd  ──[ effective plant ]──>  x_p (eye)
```

where the *pulse-step* adds velocity (and possibly higher-derivative)
feedforward to the integrator's position output, so that the cascade of
low-pass dynamics between `motor_cmd` and the eye is inverted in the
forward direction.  The goal: choose the pulse-step so that — in linear
theory — `x_p(t) ≡ NI_net(t)`.  Write the velocity command into the NI as
`u_vel = d(NI_net)/dt`.

## Case 1: First-order plant (the classic Robinson 1964 pulse-step)

```
plant(s)  =  1 / (1 + s·τ_p)
```

For `x_p = NI_net`:

```
motor_cmd  =  NI_net · (1 + s·τ_p)
           =  NI_net + τ_p · s·NI_net
           =  NI_net + τ_p · u_vel
```

Two terms: **step** (`NI_net`) + **pulse** (`τ_p · u_vel`).  This is the
classic Robinson pulse-step.

## Case 2: Two cascaded first-order LPs (e.g. MN-LP × plant-LP)

```
plant(s)  =  1 / [(1 + s·τ_mn)(1 + s·τ_p)]
```

For `x_p = NI_net`:

```
motor_cmd  =  NI_net · (1 + s·τ_mn)(1 + s·τ_p)
           =  NI_net · [1  +  (τ_p + τ_mn)·s  +  τ_p·τ_mn·s²]
           =  NI_net  +  (τ_p + τ_mn) · u_vel  +  τ_p·τ_mn · u_vel'
```

Three terms now:

1. **Step** — `NI_net`.
2. **Velocity feedforward** — `(τ_p + τ_mn) · u_vel`. The coefficient is
   the *sum* of the two TCs.
3. **Acceleration feedforward** — `τ_p · τ_mn · u_vel'`. The coefficient
   is the *product* of the two TCs.

### Realising the derivative `u_vel'`

Direct numerical differentiation is noise-prone. Use a fast lead filter:
maintain a state `u_lp = LP_{τ_fast}[u_vel]`; then

```
u_vel'  ≈  (u_vel − u_lp) / τ_fast
```

In the limit `τ_fast → 0` this recovers the ideal derivative.  In a
discrete integrator the tightest practical choice is `τ_fast ≈ dt`, which
leaves the inverse with a residual pole at `1/τ_fast` — the eye lags
NI_net by ~`τ_fast` instead of zero.

## General case: N cascaded first-order LPs at τ₁, …, τₙ

```
plant(s)   =  1 / ∏ᵢ (1 + s·τᵢ)
motor_cmd  =  NI_net · ∏ᵢ (1 + s·τᵢ)
           =  NI_net · [1  +  e₁·s  +  e₂·s²  +  …  +  eₙ·sⁿ]
```

The coefficient `e_k` is the *k-th elementary symmetric polynomial* of the
τᵢ:

| order | coefficient | physical meaning |
|------:|---|---|
| e₁ | Σ τᵢ | velocity feedforward |
| e₂ | Σ_{i<j} τᵢ·τⱼ | acceleration feedforward |
| e₃ | Σ_{i<j<k} τᵢ·τⱼ·τₖ | jerk feedforward |
| ⋮ | ⋮ | ⋮ |
| eₙ | ∏ τᵢ | n-th derivative feedforward |

Each higher-order term costs one more numerical derivative, with rapidly
diminishing returns and growing noise sensitivity.

## Application to this model: conjugate yaw

The actual motor chain in `simulator.py` is

```
motor_cmd_ni
  →  FCP  ( M_NUCLEUS encode  →  ×2  →  rect. floor / NERVE_MAX clip
           →  MN-LP τ_mn  →  MLF cross-projection
           →  M_NERVE_PROJ  →  nerve clip  →  nerves )
  →  plant LP τ_p  →  eye
```

For **conjugate yaw** the effective MN LP is per-eye asymmetric: the right
eye's LR motoneuron is **1-stage** (ABN → LR direct, one `τ_mn`), while
the left eye's MR motoneuron is **2-stage** (AIN → MLF → CN3_MR, two
cascaded `τ_mn`).  The conjugate eye position is the average:

```
½·M¹ + ½·M²  =  (1 + s·τ_mn/2) / (1 + s·τ_mn)²
```

Inverting and series-expanding,

```
motor_cmd  =  NI_net · (1 + s·τ_p) · (1 + s·τ_mn)² / (1 + s·τ_mn/2)
           =  NI_net  +  (τ_p + 1.5·τ_mn) · u_vel
                      +  (1.5·τ_p·τ_mn + 0.25·τ_mn²) · u_vel'
                      +  O(s³)
```

So for conjugate yaw, the velocity feedforward uses `τ_mn_eff = 1.5·τ_mn`
(half 1-stage + half 2-stage) and the acceleration coefficient is the
same `τ_p · τ_mn_eff` to leading order — the `0.25·τ_mn²` correction is
~0.5 % of the leading term and is dropped.

For **pitch and roll** there is no MLF (the CN3/CN4 motoneurons take the
premotor drive directly), so `τ_mn_eff = τ_mn` for those axes.

Per-axis vector:
```
τ_mn_eff  =  τ_mn · [1.5, 1.0, 1.0]
```

## What linear compensation misses: the antagonist rectification floor

The derivations above assume the effective plant is a clean cascade of
LPs.  The real FCP has a *rectification floor*: motoneurons can't fire
negatively (muscles can only pull, not push).  The `×2` factor in the FCP
encode (`premotor = 2 · M_NUCLEUS @ cmd`) makes the steady-state gain
right — the agonist alone carries the full push-pull amplitude while the
antagonist sits at the floor — but the transient dynamics are
*asymmetric*: only the agonist sees the MN_LP dynamics; the antagonist
is pinned at zero (no LP, no derivative response).

A *linear* feedforward inverse cannot compensate this asymmetry.  The
acceleration term `τ_p·τ_mn_eff · u_vel'` is *positive* at burst onset
(drives the agonist up sharper than the LP would on its own — good) but
*negative* at burst offset (would drive the antagonist up to brake the
eye — but the antagonist is floored, so the negative impulse only acts
by *dropping the agonist below its tonic baseline*, briefly flooring it
at zero).  The result: the eye decelerates *too much* at burst-end and
lands behind NI_net.

Empirically, for a 40° conjugate saccade (NERVE_MAX = 350, gate 0.3/1.5):

| pulse-step | peak vel | eye − NI @ pk vel | eye − NI @ burst-end | max\|u_pursuit\| post-saccade |
|---|---:|---:|---:|---:|
| classic Robinson | 694 | −5.19° | **−0.08°** | **0.71 deg/s** |
| + 1st-order MN comp `(τ_p+τ_mn_eff)·u_vel` | 723 | −4.33° | +1.42° | 3.00 |
| + full 2nd-order `... + τ_p·τ_mn_eff·u_vel'` | 714 | −2.71° | −1.63° | 2.26 |

**Classic Robinson is the best operating point.**  It reaches NI cleanly
by burst-end (~0° gap) and produces the lowest post-saccadic spurious
pursuit drive (~0.7 deg/s).  Adding linear MN-LP compensation reduces
the *peak-velocity* eye-vs-NI lag (~5° → ~3°) but *worsens* the
burst-end landing and the post-saccadic drive — the floor asymmetry bites
harder than the linear lag helps.

### Why classic Robinson works as well as it does

`τ_mn ≈ 5 ms` is fast compared to the burst duration (~50–80 ms).  By
burst-end both LPs have had ≫ `τ_mn` to settle to their current input.
The peak-velocity lag (~5°) is a transient that decays within
`~3·τ_mn = 15 ms` once the burst ends, so the eye lands on NI cleanly
even without any MN-LP-specific feedforward.  Higher-order compensation
would only matter if the burst were comparable to `τ_mn` in
duration — which it is not.

### What does work

The MN-LP residue *and* the FCP nonlinear asymmetries are best handled
**downstream of the NI**, in the cerebellum's forward model.  A
`fcp.step` copy + per-eye plant LP inside the cerebellum tracks the
*actual* eye position (including the floor-clip asymmetry) to ~0.1°, so
the cerebellar EC matches the retinal slip to ~0.3 deg/s — well below
the ~0.7 deg/s residual that the NI alone leaves.  The trade is 20
states in the cerebellum for the FCP+plant copy, against ~3 deg/s of
residual post-saccadic spurious pursuit drive.

## Summary

- Compensating a first-order plant needs a position term + velocity
  feedforward at coefficient `τ_p` (classic Robinson).
- Compensating an N-cascaded plant needs N velocity-derivative terms,
  with coefficients given by the elementary symmetric polynomials of
  the TCs.
- For MN-LP × plant-LP (our case), that's a 3-term controller: position
  + velocity at `(τ_p + τ_mn_eff)` + acceleration at `τ_p·τ_mn_eff`.
- For conjugate yaw, `τ_mn_eff = 1.5·τ_mn` (MLF asymmetry averaged);
  for pitch/roll, `τ_mn_eff = τ_mn`.
- The acceleration feedforward needs a numerical derivative of `u_vel`,
  approximated via a fast lead filter.
- In practice — because the FCP has a rectification floor that breaks
  linearity at the burst edges — adding the higher-order terms hurts
  more than it helps.  Classic Robinson (1st-order compensation) is the
  empirical optimum at the NI; the MN-LP residue and the FCP nonlinear
  asymmetries are properly handled inside the cerebellum's forward
  model rather than the brainstem controller.
