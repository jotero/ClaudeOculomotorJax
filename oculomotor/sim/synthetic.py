"""Generate synthetic eye movement data from ground-truth parameters."""

import jax
import jax.numpy as jnp

from oculomotor.models.vor import simulate
from oculomotor.sim.stimuli import make_all_stimuli

THETA_TRUE = {
    'tau_c':  5.0,    # canal adaptation time constant (s); HP corner ≈ 0.03 Hz
    'tau_s':  0.005,  # canal inertia time constant (s); LP corner ≈ 32 Hz
    'g_vor':  1.0,    # VOR gain (unitless)
    'tau_i':  25.0,   # neural integrator time constant (s)
    'tau_p':  0.15,   # plant time constant (s)
    'tau_vs': 50.0,   # VS prior time constant (s); τ_eff = 1/(1/τ_vs + K_vs) ≈ 20 s
    'K_vs':   0.03,   # VS Kalman gain (1/s)
}

SIGMA_OBS = 0.3  # deg, observation noise std


def generate_dataset(theta=None, sigma=SIGMA_OBS, seed=0, frequencies=None):
    """Generate synthetic observations for all stimulus conditions.

    Args:
        theta: parameter dict (defaults to THETA_TRUE)
        sigma: observation noise standard deviation (deg)
        seed: JAX random seed
        frequencies: list of sinusoidal frequencies in Hz (None = use defaults)

    Returns:
        stimuli: list of (t, head_vel) tuples
        observations: list of noisy eye_pos arrays
    """
    if theta is None:
        theta = THETA_TRUE

    stimuli = make_all_stimuli(frequencies=frequencies)
    key = jax.random.PRNGKey(seed)
    observations = []

    for t, head_vel in stimuli:
        eye_pos_clean = simulate(theta, t, head_vel)
        key, subkey = jax.random.split(key)
        noise = sigma * jax.random.normal(subkey, shape=eye_pos_clean.shape)
        observations.append(eye_pos_clean + noise)

    return stimuli, observations
