"""Central brain submodules — every module exposes the SSM ``step()`` interface.

Direct submodule imports are preferred for clarity:

    from oculomotor.models.brain_models import (
        velocity_storage, neural_integrator, saccade_generator,
        gravity_estimator, heading_estimator, pursuit, vergence,
        tvor, accommodation, listing, efference_copy, final_common_pathway,
    )

The connector module ``brain_model`` wires them together and owns the combined
state layout, parameters (BrainParams), and per-step ODE evaluation order.
"""

from oculomotor.models.brain_models import (
    velocity_storage,
    neural_integrator,
    saccade_generator,
    gravity_estimator,
    heading_estimator,
    pursuit,
    vergence,
    tvor,
    accommodation,
    listing,
    efference_copy,
    final_common_pathway,
    brain_model,
)

__all__ = [
    "velocity_storage",
    "neural_integrator",
    "saccade_generator",
    "gravity_estimator",
    "heading_estimator",
    "pursuit",
    "vergence",
    "tvor",
    "accommodation",
    "listing",
    "efference_copy",
    "final_common_pathway",
    "brain_model",
]
