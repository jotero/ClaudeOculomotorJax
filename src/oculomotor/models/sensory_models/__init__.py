"""Peripheral sensor submodules — canal, otolith, retina.

Direct submodule imports:

    from oculomotor.models.sensory_models import (
        canal, otolith, retina, sensory_model,
    )

The connector ``sensory_model`` wires them together and owns SensoryParams.
``perception_cyclopean`` (binocular fusion + brain-LP smoothing) lives in
brain_models because it operates on already-delayed signals coming back
from the retina cascade — it's a cortical computation, not peripheral.
"""

from oculomotor.models.sensory_models import (
    canal,
    otolith,
    retina,
    sensory_model,
)

__all__ = ["canal", "otolith", "retina", "sensory_model"]
