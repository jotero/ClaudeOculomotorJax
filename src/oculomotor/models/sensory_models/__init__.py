"""Peripheral sensor submodules — canal, otolith, retina, cyclopean vision.

Direct submodule imports:

    from oculomotor.models.sensory_models import (
        canal, otolith, retina, cyclopean_vision, sensory_model,
    )

The connector ``sensory_model`` wires them together and owns SensoryParams.
"""

from oculomotor.models.sensory_models import (
    canal,
    otolith,
    retina,
    cyclopean_vision,
    sensory_model,
)

__all__ = ["canal", "otolith", "retina", "cyclopean_vision", "sensory_model"]
