"""Generate synthetic eye movement data from ground-truth parameters."""

import jax
import jax.numpy as jnp

from oculomotor.sim.simulator import simulate, THETA_DEFAULT
from oculomotor.sim.stimulus import make_all_stimuli

SIGMA_OBS = 0.3  # deg, observation noise std


def generate_dataset(theta=None, sigma=SIGMA_OBS, seed=0, frequencies=None):
    """Generate synthetic observations for all stimulus conditions.

    Args:
        theta: parameter dict (defaults to THETA_DEFAULT)
        sigma: observation noise standard deviation (deg)
        seed: JAX random seed
        frequencies: list of sinusoidal frequencies in Hz (None = use defaults)

    Returns:
        stimuli: list of Stimulus objects
        observations: list of noisy eye_pos arrays
    """
    if theta is None:
        theta = THETA_DEFAULT

    stimuli = make_all_stimuli(frequencies=frequencies)
    key = jax.random.PRNGKey(seed)
    observations = []

    for stim in stimuli:
        eye_rot_clean = simulate(theta, stim)
        eye_pos_clean = eye_rot_clean[:, 0]   # horizontal component only
        key, subkey = jax.random.split(key)
        noise = sigma * jax.random.normal(subkey, shape=eye_pos_clean.shape)
        observations.append(eye_pos_clean + noise)

    return stimuli, observations
