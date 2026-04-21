"""Ocular counter-roll (OCR) — torsional eye response to head tilt.

Stateless module: no ODE state, pure algebraic output.

OCR = g_ocr * sin(roll_tilt_angle)

where sin(roll_tilt_angle) = g_hat[1] / |g_hat|  (peaks at ±1 for 90° tilt).
g_ocr is the peak OCR amplitude in degrees; healthy value ~10°.

Only the roll (torsional) component is produced.  Pitch is excluded:
vertical eye position is saccade-controlled, and the pitch-gravity coupling
(Listing's law) is not yet implemented.

Input
-----
g_hat  (3,)   specific-force vector in head frame, +x upright  (m/s²)
brain_params  BrainParams NamedTuple — uses brain_params.g_ocr

Output
------
ocr    (3,)   torsional setpoint [yaw=0, pitch=0, roll] in degrees
              CW head tilt (g_hat[1] < 0) → negative roll (CCW eye) ✓
"""

import jax.numpy as jnp

N_STATES  = 0
N_INPUTS  = 3   # g_hat
N_OUTPUTS = 3   # ocr position setpoint (deg)


def compute(g_hat, brain_params):
    """Return OCR torsional setpoint (deg) from gravity estimate."""
    g_norm = jnp.linalg.norm(g_hat) + 1e-9
    return jnp.array([
        0.0,
        0.0,
        brain_params.g_ocr * g_hat[1] / g_norm,
    ])
