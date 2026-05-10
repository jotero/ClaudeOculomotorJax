"""Central brain submodules — every leaf module exposes the SSM ``step()`` interface.

Subpackages group physiologically-coupled modules:

  perception_self_motion   — VS, gravity_estimator, heading_estimator
                             (Laurens & Angelaki self-motion observer)
  vergence_accommodation   — vergence + accommodation (AC/A and CA/C cross-links)

Top-level leaf modules (no subpackage):

  neural_integrator, saccade_generator, pursuit, tvor,
  listing, cerebellum, final_common_pathway, brain_model

The connector module ``brain_model`` wires everything together and owns the
combined state layout, BrainParams, and per-step ODE evaluation order.
"""

from oculomotor.models.brain_models import (
    perception_self_motion,
    perception_target,
    perception_cyclopean,
    vergence_accommodation,
    neural_integrator,
    saccade_generator,
    pursuit,
    tvor,
    listing,
    cerebellum,
    final_common_pathway,
    brain_model,
)

__all__ = [
    "perception_self_motion",
    "perception_target",
    "perception_cyclopean",
    "vergence_accommodation",
    "neural_integrator",
    "saccade_generator",
    "pursuit",
    "tvor",
    "listing",
    "cerebellum",
    "final_common_pathway",
    "brain_model",
]
