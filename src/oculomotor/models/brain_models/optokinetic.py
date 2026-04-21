"""Optokinetic module — scene-gated, EC-corrected visual slip for velocity storage.

Stateless module: no ODE state, pure algebraic output.

Models the NOT (nucleus of the optic tract) / AOS (accessory optic system) pathway
that drives the optokinetic response (OKR) and feeds velocity storage (VS).

Signal flow:
    slip      (3,)  raw retinal slip (scene velocity − eye velocity, deg/s)
    motor_ec  (3,)  efference copy of motor command — cancels self-generated slip
    scene_visible   scalar gate [0, 1]; 0 in darkness → OKAN decay without OKR

Processing:
    1. EC correction:  slip + motor_ec  ≈ external scene velocity (≈ 0 during fixation)
    2. Scene gate:     × scene_visible  → zero in darkness
    3. Velocity clip:  ± v_max_okr     → NOT/AOS saturation; OKR gain falls off
                                          at high speeds (> ~60–80 deg/s)

Output:
    e_slip_corrected  (3,)  visual drive to VS (deg/s), scene-gated and clipped

Parameters:
    brain_params.v_max_okr  — NOT/AOS velocity saturation (deg/s); default 80.
"""

import jax.numpy as jnp

N_STATES  = 0
N_INPUTS  = 3   # slip (retinal velocity error)
N_OUTPUTS = 3   # e_slip_corrected → VS visual input


def compute(slip, motor_ec, scene_visible, brain_params):
    """Return scene-gated, EC-corrected, clipped visual slip for VS.

    Args:
        slip:          (3,)  raw retinal slip (deg/s)
        motor_ec:      (3,)  efference copy — cancels self-generated motion
        scene_visible: scalar  scene gate [0, 1]
        brain_params:  BrainParams  (reads v_max_okr)

    Returns:
        e_slip_corrected: (3,)  visual drive to VS (deg/s)
    """
    # NOT/AOS velocity saturation applied separately to scene slip and EC,
    # then gated by scene visibility.  clip(e) + clip(ec) keeps each signal
    # within the sensory range independently before they combine.
    return scene_visible * (
        jnp.clip(slip,     -brain_params.v_max_okr, brain_params.v_max_okr)
        + jnp.clip(motor_ec, -brain_params.v_max_okr, brain_params.v_max_okr)
    )
