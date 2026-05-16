"""Final common pathway — version+vergence  →  motor neurons  →  output nerves.

Anatomically faithful chain:

    version_vergence (6,)
        ↓ M_NUCLEUS                  (signed nucleus drives)
        ↓ × g_nucleus                (nuclear lesion: multiplicative cell-count gain)
        ↓ MLF tract (axon)           (AIN MN → contralateral CN3_MR MN; cap = g_mlf · NERVE_MAX)
        ↓ M_NERVE_PROJ × 2           (nucleus → MN target firing rate)
        ↓ MN intrinsic f-I curve     (biophysical max NERVE_MAX, not a lesion knob)
    motor neurons (12 dynamic states, x_mn)
        ↓ tau_mn LP dynamics         (~5 ms membrane TC)
        ↓ axonal conduction clip     (cap = g_nerve · NERVE_MAX, frequency-selective)
    output nerves (12,) → extraocular muscles

The MLF is a motor-neuron-to-motor-neuron connection: abducens internuclear
neurons (AIN) fire as ordinary MNs (intrinsic NERVE_MAX), their axons enter
the contralateral MLF tract (conduction-capped), and they synapse onto CN3_MR
motoneurons in the oculomotor nucleus.  CN3_MR motoneurons sum vergence drive
(direct from supraoculomotor area) with MLF input → fire → drive MR muscle.

Lesion semantics:

    g_nucleus (12,)  Multiplicative gain on signed nucleus drive.
                     Models cell loss in a nucleus (% of cells surviving).
                     Burst AND tonic both attenuated proportionally.
                     ABN gain shared with co-located AIN (intermingled populations).

    g_mlf_L/R (2,)   Conduction cap on the MLF axon tract (axon-level clip).
                     Models demyelination / conduction block in the MLF.
                     Frequency-selective: tonic AIN drive (small) gets through;
                     saccadic burst (large) is capped → slow adducting saccades.
                     Vergence preserved (delivered via CN3_MR direct, bypasses MLF).

    g_nerve (12,)    Conduction cap on the cranial-nerve axon (axon-level clip).
                     Models nerve demyelination / fascicular lesion.
                     Frequency-selective: fixation hold preserved, burst clipped
                     → limited-motility ophthalmoplegia.  At g_nerve=0 the axon
                     transmits nothing → complete muscle paralysis.

Nerve / MN ordering (12,):  [LR_L, MR_L, SR_L, IR_L, SO_L, IO_L,
                              LR_R, MR_R, SR_R, IR_R, SO_R, IO_R]

Parameters (in BrainParams):
    g_nucleus (12,)  nuclear gains [0, 1]. Default: all ones.
    g_mlf_L/R        MLF axon conduction cap fractions [0, 1]. Default: 1.
    g_nerve   (12,)  cranial-nerve axon conduction cap fractions [0, 1]. Default: all ones.
    tau_mn           MN membrane LP TC (s). Default 0.005.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oculomotor.models.plant_models.muscle_geometry import (
    M_NUCLEUS, M_NERVE_PROJ,
    G_NUCLEUS_DEFAULT, G_NERVE_DEFAULT,
    AIN_L, AIN_R, MR_L, MR_R, CN3_MR_L, CN3_MR_R,
)

__all__ = ['G_NUCLEUS_DEFAULT', 'G_NERVE_DEFAULT', 'N_STATES', 'step', 'rest_state',
           'Activations', 'read_activations']

# State count: 14 motor neurons in nucleus order — 12 muscle-MNs that project
# via cranial nerves to extraocular muscles, plus 2 abducens internuclear
# neurons (AIN_L, AIN_R) whose axons enter the MLF tract and synapse onto
# contralateral CN3_MR motoneurons (no cranial-nerve output).
N_STATES = 14

# Biophysical maximum firing rate (deg/s equivalent), used for the premotor
# f-I curve, MLF axon conduction cap, and cranial-nerve axon conduction cap.
# Set ABOVE the healthy peak agonist-MN drive so a healthy big saccade is not
# throttled by its own cell-body f-I ceiling.  The ×2 reciprocal-compensation
# factor in the premotor encode means a big burst (motor_cmd ≈ 130–160 deg/s
# mid-burst on a 40–60° saccade) pushes the agonist MN toward 2·motor_cmd ≈
# 270–320; a 250-cap clipped that and pinned the peak eye velocity (a 40°,
# 50° and 60° saccade all topped out at the same ~607 deg/s) and added an
# undershoot/glissade.  350 leaves headroom up to ~60–65° while staying low
# enough that MLF / cranial-nerve conduction-cap lesions in the clinically
# meaningful range (g_mlf, g_nerve in [0.3, ~0.85] → cap ≤ ~300 < the ~320
# healthy adducting-MN drive) still engage the clip → graded INO / partial-
# palsy slowing.
_NERVE_MAX = 350.0

def _smooth_clip(z, g_max):
    """Smooth one-sided ceiling clip at g_max — no rectification floor.

    output = z − softplus(z − g_max):
        - z << g_max:  ≈ z        (linear regime, passes negatives through)
        - z >> g_max:  ≈ g_max    (saturation)
        - g_max → 0:   ≈ min(z, 0)  (lesion attenuates positive drive only)
    Originally this was a two-sided clip into [0, g_max] (rectification floor
    at 0 from muscles only being able to pull, not push) — but the floor
    introduced asymmetric dynamics that no linear pulse-step compensation
    could invert, leaking into post-saccadic spurious pursuit drive.  Removing
    the floor lets the antagonist motoneurons fire negatively (i.e., the
    muscles can push as well as pull, in this model abstraction), which makes
    the effective plant a clean cascade of two LPs that linear compensation
    can invert cleanly.  See docs/plant_compensation.md.
    """
    return z - jax.nn.softplus(z - g_max)


# ── State + registries ────────────────────────────────────────────────────────

class State(NamedTuple):
    """FCP state — 14 motor neuron membrane states.

    Each entry is a membrane-potential-like accumulator integrated by the LP
    dynamics in step().  `mn` is NOT a firing rate — the cell-body f-I curve
    that rectifies it to [0, NERVE_MAX] lives in `read_activations`.

    AIN_L and AIN_R are abducens internuclear neurons whose axons enter the
    MLF and synapse on contralateral CN3_MR motoneurons (no extraocular
    muscle output).  Their firing rates show up in the Activations registry
    alongside the 12 muscle MNs.
    """
    mn: jnp.ndarray   # (14,) [LR_L,LR_R,CN4_L,CN4_R,MR_L,MR_R,SR_L,SR_R,IR_L,IR_R,IO_L,IO_R,AIN_L,AIN_R]


class Activations(NamedTuple):
    """FCP firing rates — cell-body f-I curve applied."""
    mn: jnp.ndarray   # (14,) firing rates (clipped to [0, NERVE_MAX])


def zero_state():
    """All-zero state — useful as a NT-PyTree shape template."""
    return State(mn=jnp.zeros(14))


def read_activations(state):
    """Project MN membrane state → firing rates via the cell-body f-I curve.

    `state.mn` is a membrane-potential-like accumulator integrated by the LP
    dynamics in step().  The cell-body f-I curve maps it to firing rate by
    rectifying (≥0) and capping at NERVE_MAX.  Downstream axonal effects
    (MLF cap, nerve cap) are additional and applied in step().
    """
    return Activations(mn=_smooth_clip(state.mn, _NERVE_MAX))


def step(activations, premotor_activity, brain_params):
    """Single ODE step: premotor activity → motor neurons → output nerves.

        dx_mn  =  (premotor_in + mlf  −  rates) / tau_mn

    Activation-driven architecture:
      • The cell-body f-I curve clip lives in `read_activations` — applied
        once at the brain level via `brain_model.read_activations`.  The
        caller passes the resulting `activations.mn` (firing rates) here.
      • Premotor synaptic input is clipped at NERVE_MAX (cell-body f-I curve
        at the synapse) so the membrane integrator stays bounded — under
        normal operation `state.mn` ≈ `activations.mn`.
      • Cross-projections (MLF: AIN → contralateral CN3_MR) and the nerve
        output are both driven by ACTIVATIONS (firing rates), not raw state.
        Axons carry spike rates, not membrane potential.

    Two-stage clip:
        1. Cell body  (in read_activations)         → cap at NERVE_MAX
        2. Axonal     (in step, on MLF / nerves)    → cap at g_mlf · NERVE_MAX
                                                       or g_nerve · NERVE_MAX
    Healthy gains (g=1) make the axonal cap = NERVE_MAX (no extra effect);
    lesion (g<1) brings the cap below NERVE_MAX and starts limiting.

    Args:
        activations:        fcp.Activations  cell-body firing rates (14,) —
                             supplied by the brain-wide activations registry
        premotor_activity:  (6,)             [version (3,) | vergence (3,)] — NI +
                             saccade burst + pursuit + vergence integrator output
        brain_params:       BrainParams (g_nucleus, g_nerve, g_mlf_L/R, tau_mn)

    Returns:
        dstate: fcp.State  state derivative (membrane integrator)
        nerves: (12,)      axonal firing rates to extraocular muscles
                            [LR_L, MR_L, SR_L, IR_L, SO_L, IO_L,
                             LR_R, MR_R, SR_R, IR_R, SO_R, IO_R]
    """
    rates = activations.mn   # cell-body firing rates from brain registry

    # Premotor synaptic input: linear projection of the upstream brain command
    # via M_NUCLEUS, scaled by g_nucleus (cell-loss gain; AIN_L/R inherit
    # ABN_L/R gain — intermingled populations).  No ×2 here: the antagonist
    # is now allowed to fire negatively (push-pull symmetric), so both sides
    # carry the command and `M_PLANT_EYE @ nerves` round-trips to motor_cmd
    # without a reciprocal-compensation factor.  Ceiling-clipped at NERVE_MAX.
    g_nuc12  = brain_params.g_nucleus
    g_nuc14  = jnp.concatenate([g_nuc12, g_nuc12[:2]])
    premotor = _smooth_clip(g_nuc14 * (M_NUCLEUS @ premotor_activity),
                             _NERVE_MAX)                                          # (14,)

    # MLF: AIN firing rates feed contralateral CN3_MR through the MLF tract.
    # MLF axon conduction cap (g_mlf_L/R · NERVE_MAX) is frequency-selective.
    # Driven by AIN ACTIVATIONS.
    mlf = jnp.zeros(N_STATES) \
        .at[CN3_MR_L].set(_smooth_clip(rates[AIN_R], brain_params.g_mlf_L * _NERVE_MAX)) \
        .at[CN3_MR_R].set(_smooth_clip(rates[AIN_L], brain_params.g_mlf_R * _NERVE_MAX))

    # MN dynamics: linear LP integrating premotor + MLF inputs; leak uses
    # the current firing rate (= membrane state under normal operation since
    # premotor is clipped at NERVE_MAX → integrator stays bounded).
    dx_mn = (premotor + mlf - rates) / brain_params.tau_mn

    # Nerves: muscle-MN ACTIVATIONS + tonic baseline, projected to nerve order
    # via M_NERVE_PROJ (AIN→MR stripped — AIN reaches MR through MLF, not
    # directly).  The per-nucleus tonic baseline is scaled by the same
    # g_nucleus that scales dynamic firing, so a partial nuclear lesion
    # (e.g. g_nucleus[ABN_R]=0.5) reduces both tonic AND active drive equally
    # — half-lesion produces both slowed saccades AND tonic strabismus.
    # Uniform symmetric baseline (default) is invisible at the plant (zero-sum
    # decode columns); asymmetry — intrinsic or lesion-induced — drives drift.
    # Axon conduction cap (g_nerve · NERVE_MAX) acts at the cranial nerve.
    m_proj   = M_NERVE_PROJ \
        .at[MR_L, AIN_R].set(0.0) \
        .at[MR_R, AIN_L].set(0.0)
    r_base12 = brain_params.r_baseline
    r_base14 = jnp.concatenate([r_base12, jnp.zeros(2)])    # AIN doesn't project to nerves
    nerves   = _smooth_clip(m_proj @ (rates + g_nuc14 * r_base14),
                            brain_params.g_nerve * _NERVE_MAX)                   # (12,)
    return State(mn=dx_mn), nerves


# ── Legacy flat-array adapters (deleted once brain_model migrates to BrainState) ─

def from_array(x_mn):
    """(14,) flat array → fcp.State."""
    return State(mn=x_mn)


def to_array(state):
    """fcp.State → (14,) flat array."""
    return state.mn


def rest_state(premotor_activity, brain_params):
    """Steady-state MN firing rates for a given resting premotor command.

    At rest there is no version drive → AIN MNs silent → MLF input is zero,
    so steady state equals the rectified, capped premotor input.  Used to
    initialise x_mn so the model starts on the slow manifold (skips a
    ~5·tau_mn warmup transient at t=0).
    """
    g_nuc12 = brain_params.g_nucleus
    g_nuc14 = jnp.concatenate([g_nuc12, g_nuc12[:2]])
    return _smooth_clip(g_nuc14 * (M_NUCLEUS @ premotor_activity),
                        _NERVE_MAX)
