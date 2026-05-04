"""Central brain submodules — every leaf module exposes the SSM ``step()`` interface.

Subpackages group physiologically-coupled modules:

  self_motion              — VS, gravity_estimator, heading_estimator
                             (Laurens & Angelaki self-motion observer)
  vergence_accommodation   — vergence + accommodation (AC/A and CA/C cross-links)

Top-level leaf modules (no subpackage):

  neural_integrator, saccade_generator, pursuit, tvor,
  listing, efference_copy, final_common_pathway, brain_model

The connector module ``brain_model`` wires everything together and owns the
combined state layout, BrainParams, and per-step ODE evaluation order.
"""

from oculomotor.models.brain_models import (
    self_motion,
    vergence_accommodation,
    neural_integrator,
    saccade_generator,
    pursuit,
    tvor,
    listing,
    efference_copy,
    final_common_pathway,
    brain_model,
)

__all__ = [
    "self_motion",
    "vergence_accommodation",
    "neural_integrator",
    "saccade_generator",
    "pursuit",
    "tvor",
    "listing",
    "efference_copy",
    "final_common_pathway",
    "brain_model",
]
