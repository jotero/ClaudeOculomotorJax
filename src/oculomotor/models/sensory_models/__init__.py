"""Peripheral sensor submodules — canal, otolith, retina, perception_cyclopean.

Direct submodule imports:

    from oculomotor.models.sensory_models import (
        canal, otolith, retina, perception_cyclopean, sensory_model,
    )

The connector ``sensory_model`` wires them together and owns SensoryParams.
"""

from oculomotor.models.sensory_models import (
    canal,
    otolith,
    retina,
    perception_cyclopean,
    sensory_model,
)

__all__ = ["canal", "otolith", "retina", "perception_cyclopean", "sensory_model"]
