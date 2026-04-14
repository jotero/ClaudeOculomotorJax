"""Retinal geometry — target position to angular gaze direction.

The actual retinal error signals (e_pos, e_slip) are computed inline in the
simulator so the subtractions remain visible in the signal-flow context.
"""

import jax.numpy as jnp


def target_to_angle(p_target):
    """Convert Cartesian target position to angular gaze direction (deg).

    Args:
        p_target: (3,)  [x (rightward), y (upward), z (forward/depth)]

    Returns:
        (3,)  [yaw (rightward+), pitch (upward+), roll=0]  in degrees
    """
    x, y, z = p_target[0], p_target[1], p_target[2]
    yaw   = jnp.degrees(jnp.arctan2(x, z))
    pitch = jnp.degrees(jnp.arctan2(y, z))
    return jnp.array([yaw, pitch, 0.0])
