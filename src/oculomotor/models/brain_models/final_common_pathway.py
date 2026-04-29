"""Final common pathway — two-stage motor nucleus encode → cranial nerve activations.

Implements the lower motor neuron chain from brainstem nuclei to extraocular muscles:

    Stage 1 — nucleus encode:
        [version (3,) | vergence (3,)]  →  nuclei (12,)  via M_NUCLEUS
        g_nucleus (12,) scales each nucleus output (nuclear lesion gain).
        Non-negative relu + ×2 converts signed drive to firing rate equivalent.

    Stage 2 — nerve project + clip:
        nuclei (12,)  →  nerves (12,)  via M_NERVE_PROJ
        nerves clipped at g_nerve × _NERVE_MAX per nerve.

Nerve clipping models conduction block at high firing rates (demyelination, compression):
    g_nerve = 1.0  →  ceiling = _NERVE_MAX (1000 deg/s equiv.) — above normal burst peak
                       → transparent in health
    g_nerve < 1.0  →  burst is clipped → slow saccades; tonic stays below ceiling
                       → fixation preserved (INO, CN palsy, partial demyelination)

g_nucleus models nuclear / fascicular lesions (gain reduction at all frequencies).
g_nerve models nerve / MLF conduction block (frequency-selective, via ceiling).

Nerve order (12,):  [LR_L, MR_L, SR_L, IR_L, SO_L, IO_L,
                      LR_R, MR_R, SR_R, IR_R, SO_R, IO_R]

INO examples:
    g_nerve[1] ↓  →  left  MR adducting saccades slow (right INO: left MLF cut)
    g_nerve[7] ↓  →  right MR adducting saccades slow (left  INO: right MLF cut)

Parameters (in BrainParams):
    g_nucleus (12,)  nuclear gains [0, 1]. Default: all ones.
    g_nerve   (12,)  nerve ceiling fractions [0, 1]. Default: all ones.
    g_mlf_ver_L/R    version_yaw gain for CN3_MR (INO modelling).
"""

import jax.numpy as jnp

from oculomotor.models.plant_models.muscle_geometry import (
    M_NUCLEUS, M_NERVE_PROJ,
    G_NUCLEUS_DEFAULT, G_NERVE_DEFAULT,
    CN3_MR_L, CN3_MR_R,
)

# Re-export defaults so BrainParams can import from here without touching muscle_geometry directly.
__all__ = ['G_NUCLEUS_DEFAULT', 'G_NERVE_DEFAULT', 'step']

# Nerve firing rate ceiling (deg/s equivalent).
# Above the normal burst peak (~700 deg/s) → g_nerve=1 is transparent in health.
_NERVE_MAX = 1000.0


def step(version_vergence, brain_params):
    """Two-stage encode: version+vergence → nerve activations.

    Args:
        version_vergence: (6,) [version (3,) | vergence (3,)]  — motor_cmd_ni + u_verg
        brain_params:     BrainParams  (reads g_nucleus, g_nerve, g_mlf_ver_L/R)

    Returns:
        nerves: (12,)  per-muscle nerve activations [LR_L..IO_L, LR_R..IO_R]
    """
    # INO: scale version_yaw (col 0) of CN3_MR rows before nucleus projection.
    m_nuc = M_NUCLEUS \
        .at[CN3_MR_L, 0].mul(brain_params.g_mlf_ver_L) \
        .at[CN3_MR_R, 0].mul(brain_params.g_mlf_ver_R)

    # Stage 1 — nucleus encode: signed drive → non-negative firing rates.
    nuclei_raw = m_nuc @ version_vergence                                       # (12,) signed
    nuclei     = jnp.diag(brain_params.g_nucleus) @ (2 * jnp.maximum(nuclei_raw, 0.0))  # (12,) ≥0

    # Stage 2 — nerve project + conduction-block clip.
    nerves = jnp.minimum(M_NERVE_PROJ @ nuclei, brain_params.g_nerve * _NERVE_MAX)  # (12,)

    return nerves
