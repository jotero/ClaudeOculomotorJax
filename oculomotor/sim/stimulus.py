"""Stimulus — structured container for oculomotor simulator inputs.

A Stimulus holds all time-varying inputs needed by the ocular motor simulator:
    t        : time array (s)
    omega    : head angular velocity (deg/s)          — drives canals / VOR
    a_lin    : head linear acceleration (m/s²)        — drives otoliths
    v_scene  : visual scene angular velocity (deg/s)  — drives OKR

All arrays are (T, 3).  Use the factory functions below to build standard
paradigms; construct directly for arbitrary combinations.

Paradigm quick-reference
────────────────────────
    rotation_step(...)        sustained rotation then stop (VS / post-rotatory)
    sinusoidal_rotation(...)  sinusoidal VOR (Bode plot)
    hit(...)                  head impulse test (HIT)
    okr_step(...)             full-field scene step → optokinetic nystagmus (OKN)
    okr_sinusoidal(...)       sinusoidal scene motion → sinusoidal OKR
    combined(...)             superimpose any two stimuli (e.g. VVOR = VOR + OKR)
"""

import jax.numpy as jnp


# ── Stimulus container ─────────────────────────────────────────────────────────

class Stimulus:
    """Container for a complete 6-DOF + visual oculomotor stimulus.

    Attributes:
        t:       (T,) time (s)
        omega:   (T, 3) head angular velocity (deg/s)  [yaw, pitch, roll]
        a_lin:   (T, 3) head linear acceleration (m/s²)
        v_scene: (T, 3) visual scene angular velocity (deg/s); zeros = dark
    """

    def __init__(self, t, omega, a_lin=None, v_scene=None):
        self.t = jnp.asarray(t)
        T = len(self.t)

        if jnp.ndim(omega) == 1:
            # 1-D → treat as horizontal (yaw) only
            self.omega = jnp.stack([omega,
                                    jnp.zeros(T),
                                    jnp.zeros(T)], axis=1)
        else:
            self.omega = jnp.asarray(omega)

        self.a_lin   = jnp.zeros((T, 3)) if a_lin   is None else jnp.asarray(a_lin)
        self.v_scene = jnp.zeros((T, 3)) if v_scene is None else jnp.asarray(v_scene)

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def duration(self):
        """Total duration (s)."""
        return float(self.t[-1] - self.t[0])

    @property
    def dt(self):
        """Sample interval (s)."""
        return float(self.t[1] - self.t[0])

    @property
    def n_samples(self):
        """Number of time samples."""
        return len(self.t)

    def __repr__(self):
        return (f"Stimulus(T={self.n_samples}, dt={self.dt:.4f}s, "
                f"dur={self.duration:.1f}s, "
                f"dark={jnp.all(self.v_scene == 0).item()})")


# ── Factory functions ──────────────────────────────────────────────────────────

def rotation_step(v_deg_s=60.0, rotate_dur=15.0, coast_dur=60.0,
                  sample_rate=200.0):
    """Sustained constant-velocity rotation then sudden stop.

    Used to study velocity storage and post-rotatory nystagmus.

    Args:
        v_deg_s:     head velocity during rotation (deg/s)
        rotate_dur:  duration of rotation (s)
        coast_dur:   duration after stop (s) — observe VS decay
        sample_rate: samples/s

    Returns:
        Stimulus (horizontal rotation, dark)
    """
    total = rotate_dur + coast_dur
    t     = jnp.arange(0.0, total, 1.0 / sample_rate)
    omega = jnp.where(t < rotate_dur, v_deg_s, 0.0)
    return Stimulus(t, omega=omega)


def sinusoidal_rotation(freq_hz=0.1, amplitude=30.0, duration=20.0,
                        sample_rate=200.0):
    """Sinusoidal horizontal head rotation.

    Standard stimulus for VOR Bode plots (gain and phase vs. frequency).

    Args:
        freq_hz:     rotation frequency (Hz)
        amplitude:   peak head velocity (deg/s)
        duration:    total duration (s)
        sample_rate: samples/s

    Returns:
        Stimulus (horizontal sinusoidal rotation, dark)
    """
    t     = jnp.arange(0.0, duration, 1.0 / sample_rate)
    omega = amplitude * jnp.sin(2.0 * jnp.pi * freq_hz * t)
    return Stimulus(t, omega=omega)


def hit(direction=1.0, v_peak=200.0, duration=0.15,
        total_time=4.0, sample_rate=200.0):
    """Haversine head-velocity impulse (smooth HIT stimulus).

    Args:
        direction:   +1 (rightward) or −1 (leftward)
        v_peak:      peak head velocity (deg/s)
        duration:    impulse duration (s)
        total_time:  total trial duration (s)
        sample_rate: samples/s

    Returns:
        Stimulus (horizontal impulse, dark)
    """
    t      = jnp.arange(0.0, total_time, 1.0 / sample_rate)
    within = (t >= 0.0) & (t <= duration)
    pulse  = jnp.where(within, jnp.sin(jnp.pi * t / duration) ** 2, 0.0)
    return Stimulus(t, omega=direction * v_peak * pulse)


def okr_step(v_scene_deg_s=30.0, duration=30.0, axis=0,
             sample_rate=200.0):
    """Full-field visual scene moving at constant velocity (step onset).

    Head is stationary. Scene motion drives optokinetic nystagmus (OKN).
    OKN slow phase builds up with VS time constant; fast phase (saccades)
    is not modelled here.

    Args:
        v_scene_deg_s: scene angular velocity (deg/s); positive = rightward
        duration:      total duration (s)
        axis:          scene motion axis (0=yaw, 1=pitch, 2=roll)
        sample_rate:   samples/s

    Returns:
        Stimulus (stationary head, moving scene)
    """
    t       = jnp.arange(0.0, duration, 1.0 / sample_rate)
    T       = len(t)
    v_scene = jnp.zeros((T, 3)).at[:, axis].set(v_scene_deg_s)
    return Stimulus(t, omega=jnp.zeros(T), v_scene=v_scene)


def okr_sinusoidal(freq_hz=0.1, amplitude=30.0, duration=60.0,
                   axis=0, sample_rate=200.0):
    """Sinusoidal full-field scene motion. Head stationary.

    Used to characterise OKR gain and phase as a function of frequency.

    Args:
        freq_hz:   scene oscillation frequency (Hz)
        amplitude: peak scene velocity (deg/s)
        duration:  total duration (s)
        axis:      scene motion axis (0=yaw, 1=pitch, 2=roll)
        sample_rate: samples/s

    Returns:
        Stimulus (stationary head, oscillating scene)
    """
    t       = jnp.arange(0.0, duration, 1.0 / sample_rate)
    T       = len(t)
    v_sc    = amplitude * jnp.sin(2.0 * jnp.pi * freq_hz * t)
    v_scene = jnp.zeros((T, 3)).at[:, axis].set(v_sc)
    return Stimulus(t, omega=jnp.zeros(T), v_scene=v_scene)


def combined(stim_a, stim_b):
    """Superimpose two stimuli (must share the same time array).

    Example: VVOR = rotation_step(...) + okr_step(...) on same t grid.

    Args:
        stim_a, stim_b: Stimulus objects with identical t arrays

    Returns:
        Stimulus with summed omega, a_lin, v_scene
    """
    return Stimulus(
        t       = stim_a.t,
        omega   = stim_a.omega   + stim_b.omega,
        a_lin   = stim_a.a_lin   + stim_b.a_lin,
        v_scene = stim_a.v_scene + stim_b.v_scene,
    )
