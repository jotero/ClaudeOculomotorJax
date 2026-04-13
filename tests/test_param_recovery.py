"""End-to-end parameter recovery test.

Pass criterion: all recovered parameters within 10% of ground truth,
starting from perturbed initial conditions.
"""

import pytest

from oculomotor.sim.synthetic import generate_dataset
from oculomotor.sim.simulator import THETA_DEFAULT
from oculomotor.fitting.optimize import fit

THETA_INIT = {
    'tau_c': 3.0,
    'tau_i': 10.0,
    'tau_p': 0.30,
}

TOLERANCE = 0.10  # 10% relative error


def _relative_error(recovered, true):
    return abs(recovered - true) / abs(true)


def test_parameter_recovery():
    stimuli, observations = generate_dataset(theta=THETA_DEFAULT, seed=42)

    theta_fit, _ = fit(
        stimuli, observations,
        theta_init=THETA_INIT,
        method='lbfgs',
        print_every=0,
    )

    for key in ('tau_c', 'tau_i', 'tau_p'):
        err = _relative_error(float(theta_fit[key]), THETA_DEFAULT[key])
        print(f"  {key}: true={THETA_DEFAULT[key]:.4f}  fit={float(theta_fit[key]):.4f}  err={err:.2%}")
        assert err <= TOLERANCE, (
            f"{key}: recovered {float(theta_fit[key]):.4f} vs true {THETA_DEFAULT[key]:.4f} "
            f"({err:.2%} > {TOLERANCE:.0%})"
        )
