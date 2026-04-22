"""Optokinetic module — scene-gated, EC-corrected visual slip for velocity storage.

Stateless module: no ODE state, pure algebraic output.

Models the NOT (nucleus of the optic tract) / AOS (accessory optic system) pathway
that drives the optokinetic response (OKR) and feeds velocity storage (VS).

Signal flow:
    slip      (3,)  retinal slip already scene-gated at cascade input (scene_vel × scene_present)
    motor_ec  (3,)  efference copy pre-gated by scene_visible before calling compute()

Processing:
    1. EC correction:  slip + motor_ec  ≈ external scene velocity (≈ 0 during fixation)
    2. Velocity clip:  ± v_max_okr     → NOT/AOS saturation; OKR gain falls off
                                          at high speeds (> ~60–80 deg/s)

Note: scene gating is handled upstream — slip is zeroed by the retinal cascade when
scene_present=0; motor_ec is multiplied by scene_visible at the call site in brain_model.

Output:
    e_slip_corrected  (3,)  visual drive to VS (deg/s), scene-gated and clipped

Parameters:
    brain_params.v_max_okr  — NOT/AOS velocity saturation (deg/s); default 80.
"""

import jax.numpy as jnp

N_STATES  = 0
N_INPUTS  = 3   # slip (retinal velocity error)
N_OUTPUTS = 3   # e_slip_corrected → VS visual input


def compute(slip, motor_ec, brain_params):
    """Return scene-gated, EC-corrected, clipped visual slip for VS.

    Args:
        slip:          (3,)  raw retinal slip (deg/s)
        motor_ec:      (3,)  efference copy — cancels self-generated motion
        brain_params:  BrainParams  (reads v_max_okr)

    Returns:
        e_slip_corrected: (3,)  visual drive to VS (deg/s)
    """
    # NOT/AOS velocity saturation applied separately to scene slip and EC,
    # then gated by scene visibility.  clip(e) + clip(ec) keeps each signal
    # within the sensory range independently before they combine.
    return (jnp.clip(slip, -brain_params.v_max_okr, brain_params.v_max_okr)
        + jnp.clip(motor_ec, -brain_params.v_max_okr, brain_params.v_max_okr)
    )
