"""Shared neural nonlinearities used across sensory and brain models.

Each function is a pure JAX function — no side effects, compatible with jit/grad.
"""

import jax.numpy as jnp


def velocity_saturation(v, v_sat, v_zero=None):
    """Smooth velocity saturation: passes at low speed, gain ramps to zero at high speed.

    For a velocity vector v:
        |v| ≤ v_sat          → output = v           (gain = 1)
        v_sat < |v| < v_zero → output = v * gain    (cosine rolloff, 1 → 0)
        |v| ≥ v_zero         → output = 0            (gain = 0)

    The cosine rolloff keeps gain and its derivative continuous at both endpoints.

    Contrast with jnp.clip: clipping keeps output at ±v_sat for large inputs,
    which lets a step-function target spike (e.g. 9000 deg/s central-difference
    artifact) drive the pursuit integrator at full saturation velocity.  This
    function returns zero instead, faithfully modelling MT/MST insensitivity to
    implausibly fast retinal motion.

    Speed tuning background
    -----------------------
    MT/MST neurons are band-pass tuned for speed, not low-pass.  The population
    response peaks around 10–40 deg/s and falls off sharply above ~80–100 deg/s.
    Pursuit consequently saturates: human gain ≈ 1 below ~30 deg/s, dropping to
    ~0.5 at 60 deg/s and near zero above ~100 deg/s.

    NOT/AOS neurons driving OKR have broader tuning, peaking ~40–80 deg/s and
    falling off above ~160 deg/s.

    References
    ----------
    Maunsell JHR & Van Essen DC (1983) J Neurophysiol 49:1127-1147
        — MT speed tuning in macaque; peak ~30 deg/s, ~50 % at 8 and 100 deg/s.
    Lisberger SG et al. (1987) Annu Rev Neurosci 10:97-129
        — Review of pursuit velocity range and MT contribution.
    Priebe NJ & Lisberger SG (2004) J Neurosci 24:4907-4926
        — Population speed coding in MT; pursuit gain × speed relationship.
    Buettner U et al. (1976) Brain Res 108:359-377
        — OKN gain as a function of stimulus velocity; falls above 80 deg/s.

    Args:
        v:      (N,) velocity vector (deg/s); norm computed over the full vector
        v_sat:  saturation onset (deg/s) — gain is exactly 1 below this
        v_zero: speed where gain reaches 0 (deg/s); default = 2 × v_sat

    Returns:
        Same shape as v, scaled by smooth gain ∈ [0, 1].

    Example (pursuit):
        velocity_saturation(target_slip, v_sat=40.0)   # v_zero defaults to 80.0
    Example (OKR/NOT):
        velocity_saturation(scene_slip, v_sat=80.0, v_zero=160.0)
    """
    if v_zero is None:
        v_zero = 2.0 * v_sat
    speed = jnp.linalg.norm(v)
    t     = jnp.clip((speed - v_sat) / (v_zero - v_sat), 0.0, 1.0)
    gain  = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    return v * gain
