"""Head motion stimulus generation."""

import jax.numpy as jnp


FREQUENCIES_HZ = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
AMPLITUDE_DEG_S = 30.0
DURATION_S = 20.0
SAMPLE_RATE_HZ = 100.0


def sinusoidal_stimulus(frequency_hz, amplitude=AMPLITUDE_DEG_S,
                        duration=DURATION_S, sample_rate=SAMPLE_RATE_HZ):
    """Sinusoidal head velocity stimulus.

    Args:
        frequency_hz: stimulus frequency (Hz)
        amplitude: peak head velocity (deg/s)
        duration: duration (s)
        sample_rate: samples per second (Hz)

    Returns:
        t: time array, shape (T,)
        head_vel: head velocity array, shape (T,)
    """
    t = jnp.arange(0.0, duration, 1.0 / sample_rate)
    head_vel = amplitude * jnp.sin(2.0 * jnp.pi * frequency_hz * t)
    return t, head_vel


def step_stimulus(amplitude=AMPLITUDE_DEG_S, step_duration=1.0,
                  total_duration=40.0, sample_rate=SAMPLE_RATE_HZ):
    """Step (head impulse) stimulus: constant velocity then zero.

    Args:
        amplitude: head velocity during the step (deg/s)
        step_duration: duration of the velocity step (s)
        total_duration: total trial duration (s). Default 40 s — needed to
            observe ≥2 time constants of velocity storage (τ_eff ≈ 20 s).
        sample_rate: samples per second (Hz)

    Returns:
        t: time array, shape (T,)
        head_vel: head velocity array, shape (T,)
    """
    t = jnp.arange(0.0, total_duration, 1.0 / sample_rate)
    head_vel = jnp.where(t < step_duration, amplitude, 0.0)
    return t, head_vel


def make_all_stimuli(frequencies=None, sample_rate=SAMPLE_RATE_HZ):
    """Return list of (t, head_vel) tuples for all sinusoidal frequencies plus a step.

    Args:
        frequencies: list of frequencies in Hz (default FREQUENCIES_HZ)
        sample_rate: samples per second

    Returns:
        list of (t, head_vel) tuples
    """
    if frequencies is None:
        frequencies = FREQUENCIES_HZ
    stimuli = [sinusoidal_stimulus(f, sample_rate=sample_rate) for f in frequencies]
    stimuli.append(step_stimulus(sample_rate=sample_rate))
    return stimuli
