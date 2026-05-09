"""Final common pathway — motor neuron states with smooth-saturating activation.

Implements the lower motor neuron chain from brainstem nuclei to extraocular muscles:

    Stage 1 — nucleus encode (algebraic, signed):
        [version (3,) | vergence (3,)]  →  nuclei (14,)  via M_NUCLEUS
        Includes ABN motoneurons + AIN (abducens internuclear neurons) + CN3
        + CN4 sub-populations.  g_nucleus (14,) scales each nucleus output
        (nuclear lesion gain).

    Stage 2 — nerve drive + smooth clip (algebraic):
        nuclei (14,)  →  nerve_drive (12,)  via M_NERVE_PROJ × 2
        MR motoneurons receive convergent input from CN3_MR (vergence) and
        contralateral AIN (version, via MLF).  g_mlf_L/R scale the AIN→MR
        synaptic weights in M_NERVE_PROJ — INO is a one-line lesion now.

        Smooth clip into [0, g_nerve · NERVE_MAX] via
            softplus(x) - softplus(x - g_ceil)
        zero floor at left, exponential saturation at right, ~1-unit knee width.
        Gradients are continuous everywhere — no relu/min kinks.

    Stage 3 — MN low-pass (dynamic, 12 states):
            dx_mn / dt = (drive_clipped - x_mn) / tau_mn
            nerves     = x_mn
        x_mn is the per-nerve motor-neuron firing rate (deg/s equivalent).
        tau_mn ~5 ms (oculomotor MN membrane TC; Robinson 1981; Sylvestre &
        Cullen 1999). Adds a small lag between brainstem command and muscle
        drive — well below plant TC (~150–200 ms), so behaviorally invisible
        in healthy subjects, but a clean hook for MN-level pathologies and
        smooth gradients near the activation kinks.

Nerve order (12,):  [LR_L, MR_L, SR_L, IR_L, SO_L, IO_L,
                      LR_R, MR_R, SR_R, IR_R, SO_R, IO_R]

g_nerve models nerve / MLF conduction block (frequency-selective, via ceiling):
    g_nerve = 1.0  →  ceiling = NERVE_MAX (1000 deg/s equiv.) — above normal
                       burst peak → transparent in health
    g_nerve < 1.0  →  burst is clipped → slow saccades; tonic stays below ceiling
                       → fixation preserved (INO, CN palsy, partial demyelination)

g_nucleus models nuclear / fascicular lesions (gain reduction at all frequencies).

INO is now a clean lesion at the MLF synapse on the MR motoneuron pool:
    g_mlf_L = 0  →  left  MLF cut: AIN_R → MR_L blocked → left MR fails to
                     adduct on rightward gaze (vergence preserved via CN3_MR_L)
    g_mlf_R = 0  →  right MLF cut: AIN_L → MR_R blocked → right MR fails to
                     adduct on leftward  gaze (vergence preserved via CN3_MR_R)

Parameters (in BrainParams):
    g_nucleus (12,)  nuclear gains [0, 1].  One gain per anatomical nucleus;
                     AIN_L/R share the gain of ABN_L/R since motoneurons and
                     internuclear neurons are intermingled and any real lesion
                     hits both. Default: all ones.
    g_nerve   (12,)  nerve ceiling fractions [0, 1]. Default: all ones.
    g_mlf_L/R        per-side MLF synaptic gain on AIN→MR connection.
    tau_mn           MN membrane TC (s). Default 0.005.
"""

import jax
import jax.numpy as jnp

from oculomotor.models.plant_models.muscle_geometry import (
    M_NUCLEUS, M_NERVE_PROJ,
    G_NUCLEUS_DEFAULT, G_NERVE_DEFAULT,
    AIN_L, AIN_R, MR_L, MR_R,
)

__all__ = ['G_NUCLEUS_DEFAULT', 'G_NERVE_DEFAULT', 'N_STATES', 'step', 'rest_state']

# State count: per-nerve MN firing rate.
N_STATES = 12

# Nerve firing rate ceiling (deg/s equivalent).
# Above the normal burst peak (~700 deg/s) → g_nerve=1 is transparent in health.
_NERVE_MAX = 1000.0


def _smooth_clip(x, g):
    """Smooth one-sided clip into [0, g].

    softplus(x) - softplus(x - g) ≈ clip(x, 0, g) with ~1-unit knee at each end.
    For g >> 1 (ceiling ~1000) the knee is negligible relative to the range.
    """
    return jax.nn.softplus(x) - jax.nn.softplus(x - g)


def _nerve_drive(version_vergence, brain_params):
    """Algebraic stages 1+2: signed brain command → smooth-clipped nerve drive.

    g_mlf_L/R inject directly into the AIN→MR entries of M_NERVE_PROJ:
        MR_L gets contralateral AIN_R input scaled by g_mlf_L (left  MLF)
        MR_R gets contralateral AIN_L input scaled by g_mlf_R (right MLF)
    INO is now a one-line synaptic lesion, not a column-scaling hack on M_NUCLEUS.
    """
    # MLF synaptic gains live on the AIN→MR entries of the projection matrix.
    m_proj = M_NERVE_PROJ \
        .at[MR_L, AIN_R].set(brain_params.g_mlf_L) \
        .at[MR_R, AIN_L].set(brain_params.g_mlf_R)

    # g_nucleus is (12,); expand to (14,) by sharing ABN_L/R gain with AIN_L/R
    # (motoneurons and internuclear neurons are anatomically intermingled —
    # any real abducens nucleus lesion hits both populations together).
    g_nuc12 = brain_params.g_nucleus
    g_nuc14 = jnp.concatenate([g_nuc12, g_nuc12[:2]])    # AIN_L ← ABN_L gain, AIN_R ← ABN_R gain

    nuclei = g_nuc14 * (M_NUCLEUS @ version_vergence)                            # (14,)

    # Project to nerves and apply smooth one-sided clip into [0, g_nerve·NERVE_MAX].
    nerve_signed = (m_proj @ nuclei) * 2.0                                       # (12,)
    g_ceil       = brain_params.g_nerve * _NERVE_MAX                             # (12,)
    return _smooth_clip(nerve_signed, g_ceil)


def step(x_mn, version_vergence, brain_params):
    """Single ODE step: brain command → MN low-pass → nerve activations.

    Args:
        x_mn:             (12,) MN firing rates [LR_L..IO_L, LR_R..IO_R]
        version_vergence: (6,) [version (3,) | vergence (3,)]
        brain_params:     BrainParams (reads g_nucleus, g_nerve, g_mlf_L/R, tau_mn)

    Returns:
        dx_mn:  (12,) state derivative
        nerves: (12,) per-muscle nerve activations (= x_mn)
    """
    drive_clipped = _nerve_drive(version_vergence, brain_params)
    dx_mn  = (drive_clipped - x_mn) / brain_params.tau_mn
    nerves = x_mn
    return dx_mn, nerves


def rest_state(version_vergence, brain_params):
    """Steady-state MN firing rates for a given resting brain command.

    Used to initialise x_mn so the model starts on the slow manifold
    (otherwise there is a ~5·tau_mn warmup transient at t=0).
    """
    return _nerve_drive(version_vergence, brain_params)
