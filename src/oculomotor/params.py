"""Public parameter surface — single import for tools that need all three Params classes.

The Params NamedTuples themselves stay co-located with their model code
(BrainParams in brain_model.py, etc.); this module just re-exports them so
documentation generators, validators, and CLI tools can load the whole set
with one import:

    from oculomotor.params import BrainParams, SensoryParams, PlantParams

For runtime parameter overrides, prefer the existing helpers in
``oculomotor.sim.simulator``: ``with_brain``, ``with_sensory``, ``with_plant``.
"""

from oculomotor.models.brain_models.brain_model           import BrainParams
from oculomotor.models.sensory_models.sensory_model       import SensoryParams
from oculomotor.models.plant_models.plant_model_first_order import PlantParams

__all__ = ["BrainParams", "SensoryParams", "PlantParams"]
