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

import jax
import jax.numpy as jnp

from oculomotor.models.plant_models.muscle_geometry import (
    M_NUCLEUS, M_NERVE_PROJ,
    G_NUCLEUS_DEFAULT, G_NERVE_DEFAULT,
    AIN_L, AIN_R, MR_L, MR_R, CN3_MR_L, CN3_MR_R,
)

__all__ = ['G_NUCLEUS_DEFAULT', 'G_NERVE_DEFAULT', 'N_STATES', 'step', 'rest_state']

# State count: 14 motor neurons in nucleus order — 12 muscle-MNs that project
# via cranial nerves to extraocular muscles, plus 2 abducens internuclear
# neurons (AIN_L, AIN_R) whose axons enter the MLF tract and synapse onto
# contralateral CN3_MR motoneurons (no cranial-nerve output).
N_STATES = 14

# Biophysical maximum firing rate (deg/s equivalent), used for the premotor
# f-I curve, MLF axon conduction cap, and cranial-nerve axon conduction cap.
# Calibrated so that healthy peak MN firing during a 50° saccade (~240 deg/s,
# limited by the upstream saccade-generator burst saturation) sits just below
# the cap, while partial lesion gains in the clinically meaningful range
# (g_mlf, g_nerve in [0.3, 0.85]) actually engage the clip → graded slowing
# of saccades that matches the clinical INO / partial palsy spectrum.
_NERVE_MAX = 250.0

def _smooth_clip(z, g_max):
    """Smooth one-sided clip into [0, g_max] via softplus difference.

    softplus(z) − softplus(z − g_max):
        - z ≤ 0:    ≈ 0          (rectification floor)
        - 0 < z < g_max:  ≈ z    (linear regime)
        - z > g_max: ≈ g_max     (saturation)
        - g_max = 0:  identically 0 everywhere  (full palsy → cell silent)
    Knee width ~1 deg/s, negligible at typical firing-rate scales.
    """
    return jax.nn.softplus(z) - jax.nn.softplus(z - g_max)


def step(x_mn, premotor_activity, brain_params):
    """Single ODE step: premotor activity → motor neurons → output nerves.

        dx_mn  =  (premotor + mlf  −  x_mn) / tau_mn

    MLF is a motor-neuron-to-motor-neuron connection: AIN MN axons enter the
    MLF tract (conduction cap = g_mlf · NERVE_MAX), terminate on contralateral
    CN3_MR motoneurons.  Only CN3_MR_L and CN3_MR_R receive MLF input; for
    every other MN the MLF term is zero.

    Args:
        x_mn:               (14,) MN firing rates in nucleus order
                             [ABN_L, ABN_R, CN4_L, CN4_R, CN3_MR_L, CN3_MR_R,
                              CN3_SR_L, CN3_SR_R, CN3_IR_L, CN3_IR_R,
                              CN3_IO_L, CN3_IO_R, AIN_L, AIN_R]
        premotor_activity:  (6,) [version (3,) | vergence (3,)] — output of NI +
                             saccade burst + pursuit + vergence integrators
        brain_params:       BrainParams (g_nucleus, g_nerve, g_mlf_L/R, tau_mn)

    Returns:
        dx_mn:  (14,) MN state derivative
        nerves: (12,) axonal firing rates to extraocular muscles
                 [LR_L, MR_L, SR_L, IR_L, SO_L, IO_L,
                  LR_R, MR_R, SR_R, IR_R, SO_R, IO_R]
    """
    # Premotor input arriving at each MN's synapses: linear projection of the
    # upstream brain command via M_NUCLEUS, scaled by g_nucleus (cell-loss
    # gain; AIN_L/R inherit ABN_L/R gain — intermingled populations), then
    # rectified and capped at the premotor neurons' biophysical max NERVE_MAX.
    # The ×2 doubling compensates for the antagonist sitting at the rectified
    # zero floor — agonist alone has to carry the full push-pull amplitude.
    g_nuc12  = brain_params.g_nucleus
    g_nuc14  = jnp.concatenate([g_nuc12, g_nuc12[:2]])
    premotor = _smooth_clip(g_nuc14 * (M_NUCLEUS @ premotor_activity) * 2.0,
                            _NERVE_MAX)                                          # (14,)

    # MLF input: AIN MN firing rates feed contralateral CN3_MR MNs through the
    # MLF axon tract.  Conduction cap (g_mlf · NERVE_MAX) is frequency-selective —
    # tonic AIN drive (small) gets through; saccade burst (large) is capped.
    # Zero everywhere except CN3_MR_L and CN3_MR_R.
    mlf = jnp.zeros(N_STATES) \
        .at[CN3_MR_L].set(_smooth_clip(x_mn[AIN_R], brain_params.g_mlf_L * _NERVE_MAX)) \
        .at[CN3_MR_R].set(_smooth_clip(x_mn[AIN_L], brain_params.g_mlf_R * _NERVE_MAX))

    # MN dynamics: linear LP integrating premotor + MLF inputs.
    dx_mn = (premotor + mlf - x_mn) / brain_params.tau_mn

    # Output nerves: muscle-MN firing rates permuted to nerve order via
    # M_NERVE_PROJ (with AIN→MR stripped — AIN reaches MR through MLF, not
    # directly).  The per-nucleus tonic baseline is scaled by the same
    # g_nucleus that scales dynamic firing, so a partial nuclear lesion
    # (e.g. g_nucleus[ABN_R]=0.5) reduces the surviving population's tone
    # AND its active response equally — half-lesion produces both slowed
    # saccades AND tonic strabismus from the now-asymmetric baseline.
    # Uniform symmetric baseline (default) is invisible at the plant
    # (zero-sum decode columns); asymmetry — whether intrinsic or induced
    # by lesion gains — produces tonic eye drift.
    # Axon conduction cap (g_nerve · NERVE_MAX) acts on the output.
    m_proj   = M_NERVE_PROJ \
        .at[MR_L, AIN_R].set(0.0) \
        .at[MR_R, AIN_L].set(0.0)
    r_base12 = brain_params.r_baseline
    r_base14 = jnp.concatenate([r_base12, jnp.zeros(2)])    # AIN doesn't project to nerves
    nerves   = _smooth_clip(m_proj @ (x_mn + g_nuc14 * r_base14),
                            brain_params.g_nerve * _NERVE_MAX)                   # (12,)
    return dx_mn, nerves


def rest_state(premotor_activity, brain_params):
    """Steady-state MN firing rates for a given resting premotor command.

    At rest there is no version drive → AIN MNs silent → MLF input is zero,
    so steady state equals the rectified, capped premotor input.  Used to
    initialise x_mn so the model starts on the slow manifold (skips a
    ~5·tau_mn warmup transient at t=0).
    """
    g_nuc12 = brain_params.g_nucleus
    g_nuc14 = jnp.concatenate([g_nuc12, g_nuc12[:2]])
    return _smooth_clip(g_nuc14 * (M_NUCLEUS @ premotor_activity) * 2.0,
                        _NERVE_MAX)
