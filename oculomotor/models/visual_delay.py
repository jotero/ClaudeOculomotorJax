"""Backward-compatible re-export shim.

Visual delay cascade has moved to oculomotor.sensory_model.
This module re-exports the public constants so existing code continues to work.

New code should import from oculomotor.sensory_model directly.
"""

from oculomotor.sensory_model import (   # noqa: F401
    N_STAGES,
    C_slip,
    C_pos,
    _N_PER_SIG,
    _A_STRUCT,
    _B_STRUCT_SIG,
)

# N_STATES for the visual delay portion only (used by demo scripts via _IDX_VIS)
from oculomotor.sensory_model import _N_VIS_STATES as N_STATES   # noqa: F401
