"""Stimulus generators for oculomotor simulations.

All functions return numpy / JAX arrays ready to pass directly to
``oculomotor.sim.simulator.simulate()``.

Conventions
-----------
- Angles in **degrees**, angular velocities in **deg/s**.
- 3-D axes: [yaw, pitch, roll].
- Time arrays start at 0 with step ``dt`` (default 0.001 s).
- ``p_target`` uses stereographic coordinates: [tan(yaw_rad), tan(pitch_rad), 1.0].
  Use ``angle_to_cartesian()`` for conversion.
- ``v_target`` is target angular velocity in world frame (deg/s), 3-D.

Quick reference
---------------
    t, hv = rotation_step(30.0, rotate_dur=1.5, coast_dur=1.5)
    t, pt, vt = target_step([(0.2, 10.0, 0.0)])
    t, pt, vt = target_ramp(velocity_deg_s=20.0, t_start=0.2, duration=3.0)
    t, vs, sp = scene_motion(velocity_deg_s=30.0, on_dur=20.0, total_dur=40.0)
"""

import numpy as np
import jax.numpy as jnp


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_time(duration: float, dt: float = 0.001) -> np.ndarray:
    """Return a time array [0, duration) with step dt."""
    return np.arange(0.0, duration, dt, dtype=np.float32)


def angle_to_cartesian(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    """Convert (yaw, pitch) in degrees to stereographic target position.

    Returns (3,) array [tan(yaw_rad), tan(pitch_rad), 1.0] — the format
    expected by ``simulate(p_target_array=...)``.
    """
    return np.array([
        np.tan(np.radians(yaw_deg)),
        np.tan(np.radians(pitch_deg)),
        1.0,
    ], dtype=np.float32)


def _pad3(arr1d: np.ndarray, axis: str = 'yaw') -> np.ndarray:
    """Pad a (T,) array to (T, 3) along the specified axis."""
    T = len(arr1d)
    out = np.zeros((T, 3), dtype=np.float32)
    idx = {'yaw': 0, 'pitch': 1, 'roll': 2}[axis]
    out[:, idx] = arr1d
    return out


# ── Head motion ────────────────────────────────────────────────────────────────

def rotation_step(
    velocity_deg_s: float,
    rotate_dur: float,
    coast_dur: float = 0.0,
    dt: float = 0.001,
    axis: str = 'yaw',
) -> tuple[np.ndarray, np.ndarray]:
    """Step-velocity head rotation: constant velocity then coast.

    Args:
        velocity_deg_s: Head angular velocity during rotation phase (deg/s).
        rotate_dur:     Duration of constant-velocity rotation (s).
        coast_dur:      Duration of zero-velocity coast after rotation (s).
        dt:             Time step (s).
        axis:           'yaw', 'pitch', or 'roll'.

    Returns:
        t_array:        (T,)    time array (s)
        head_vel_array: (T, 3)  head angular velocity (deg/s)
    """
    total = rotate_dur + coast_dur
    t = make_time(total, dt)
    hv1d = np.where(t < rotate_dur, velocity_deg_s, 0.0).astype(np.float32)
    return t, _pad3(hv1d, axis)


def rotation_sinusoid(
    amplitude_deg_s: float,
    frequency_hz: float,
    duration: float,
    dt: float = 0.001,
    axis: str = 'yaw',
) -> tuple[np.ndarray, np.ndarray]:
    """Sinusoidal head rotation — e.g. for rotational VOR gain/phase measurement.

    Args:
        amplitude_deg_s: Peak velocity (deg/s).
        frequency_hz:    Frequency (Hz).
        duration:        Total duration (s).
        dt:              Time step (s).
        axis:            'yaw', 'pitch', or 'roll'.

    Returns:
        t_array:        (T,)    time array (s)
        head_vel_array: (T, 3)  head angular velocity (deg/s)
    """
    t = make_time(duration, dt)
    hv1d = (amplitude_deg_s * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)
    return t, _pad3(hv1d, axis)


def head_impulse(
    amplitude_deg_s: float,
    ramp_dur: float = 0.02,
    total_dur: float = 2.0,
    dt: float = 0.001,
    axis: str = 'yaw',
) -> tuple[np.ndarray, np.ndarray]:
    """Head impulse test (HIT): rapid brief head velocity pulse.

    Models the clinical video head impulse test (vHIT).  Head velocity rises
    linearly to ``amplitude_deg_s`` over ``ramp_dur`` then decays to zero
    (trapezoidal approximation).

    Args:
        amplitude_deg_s: Peak head velocity (deg/s); typical HIT ≈ 150–300 deg/s.
        ramp_dur:        Rise and fall time (s); default 20 ms.
        total_dur:       Total trial duration including coast (s).
        dt:              Time step (s).
        axis:            Direction of head impulse ('yaw', 'pitch', 'roll').

    Returns:
        t_array:        (T,)    time array (s)
        head_vel_array: (T, 3)  head angular velocity (deg/s)
    """
    t = make_time(total_dur, dt)
    hv1d = np.zeros(len(t), dtype=np.float32)
    rise   = t < ramp_dur
    fall   = (t >= ramp_dur) & (t < 2 * ramp_dur)
    hv1d[rise] = amplitude_deg_s * t[rise] / ramp_dur
    hv1d[fall] = amplitude_deg_s * (1.0 - (t[fall] - ramp_dur) / ramp_dur)
    return t, _pad3(hv1d, axis)


def rotation_none(duration: float, dt: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """Stationary head — no head motion for ``duration`` seconds."""
    t = make_time(duration, dt)
    return t, np.zeros((len(t), 3), dtype=np.float32)


# ── Target motion ──────────────────────────────────────────────────────────────

def target_stationary(
    duration: float,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    dt: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stationary fixation target.

    Returns:
        t_array:        (T,)    time array (s)
        p_target_array: (T, 3)  Cartesian target positions
        v_target_array: (T, 3)  target velocities (all zero)
    """
    t = make_time(duration, dt)
    T = len(t)
    pos = angle_to_cartesian(yaw_deg, pitch_deg)
    p_target = np.tile(pos, (T, 1))
    v_target = np.zeros((T, 3), dtype=np.float32)
    return t, p_target, v_target


def target_steps(
    jumps: list[tuple[float, float, float]],
    duration: float,
    dt: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sequence of target position steps.

    Args:
        jumps:    List of (t_jump, yaw_deg, pitch_deg) tuples.  Target jumps
                  instantaneously to (yaw, pitch) at time t_jump and holds
                  until the next jump.  First segment is straight-ahead (0°, 0°)
                  until the first jump.
        duration: Total trial duration (s).
        dt:       Time step (s).

    Returns:
        t_array:        (T,)    time array (s)
        p_target_array: (T, 3)  Cartesian target positions
        v_target_array: (T, 3)  target velocities (all zero for step targets)
    """
    t = make_time(duration, dt)
    T = len(t)
    p_target = np.zeros((T, 3), dtype=np.float32)
    p_target[:, 2] = 1.0  # default: straight ahead

    # Fill forward
    current_pos = angle_to_cartesian(0.0, 0.0)
    sorted_jumps = sorted(jumps, key=lambda x: x[0])
    jump_idx = 0
    for i, ti in enumerate(t):
        while jump_idx < len(sorted_jumps) and ti >= sorted_jumps[jump_idx][0]:
            _, yaw, pitch = sorted_jumps[jump_idx]
            current_pos = angle_to_cartesian(yaw, pitch)
            jump_idx += 1
        p_target[i] = current_pos

    v_target = np.zeros((T, 3), dtype=np.float32)
    return t, p_target, v_target


def target_ramp(
    velocity_deg_s: float,
    t_start: float = 0.2,
    duration: float = 3.0,
    dt: float = 0.001,
    axis: str = 'yaw',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ramp target: stationary until t_start, then moves at constant velocity.

    Simulates a pursuit stimulus.

    Args:
        velocity_deg_s: Target angular velocity after onset (deg/s).
        t_start:        Time when target starts moving (s).
        duration:       Total trial duration (s).
        dt:             Time step (s).
        axis:           'yaw' or 'pitch'.

    Returns:
        t_array:        (T,)    time array (s)
        p_target_array: (T, 3)  Cartesian target positions
        v_target_array: (T, 3)  target angular velocity (deg/s)
    """
    t = make_time(duration, dt)
    T = len(t)
    target_deg = np.where(t >= t_start, velocity_deg_s * (t - t_start), 0.0)

    ax_idx = {'yaw': 0, 'pitch': 1, 'roll': 2}[axis]
    p_target = np.zeros((T, 3), dtype=np.float32)
    p_target[:, 2] = 1.0
    p_target[:, ax_idx] = np.tan(np.radians(target_deg)).astype(np.float32)

    v_target = np.zeros((T, 3), dtype=np.float32)
    v_target[t >= t_start, ax_idx] = float(velocity_deg_s)

    return t, p_target, v_target


# ── Scene / visual field ───────────────────────────────────────────────────────

def scene_motion(
    velocity_deg_s: float,
    on_dur: float,
    total_dur: float,
    dt: float = 0.001,
    axis: str = 'yaw',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-field visual scene motion — drives OKN / OKR.

    Scene moves at constant velocity for ``on_dur`` seconds then stops
    (but scene_present stays on).  After scene stops, OKAN persists driven
    by velocity storage.

    Args:
        velocity_deg_s: Scene angular velocity during motion phase (deg/s).
        on_dur:         Duration of scene motion (s).
        total_dur:      Total trial duration; should be > on_dur for OKAN (s).
        dt:             Time step (s).
        axis:           'yaw', 'pitch', or 'roll'.

    Returns:
        t_array:             (T,)    time array (s)
        v_scene_array:       (T, 3)  scene angular velocity (deg/s)
        scene_present_array: (T,)    scene visibility flag [0, 1]
    """
    t = make_time(total_dur, dt)
    vs1d = np.where(t < on_dur, velocity_deg_s, 0.0).astype(np.float32)
    v_scene = _pad3(vs1d, axis)
    scene_present = np.ones(len(t), dtype=np.float32)
    return t, v_scene, scene_present


def scene_dark(duration: float, dt: float = 0.001) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """No visual scene — dark room.

    Returns:
        t_array:             (T,)    time array (s)
        v_scene_array:       (T, 3)  zeros
        scene_present_array: (T,)    zeros
    """
    t = make_time(duration, dt)
    T = len(t)
    return t, np.zeros((T, 3), dtype=np.float32), np.zeros(T, dtype=np.float32)


def scene_stationary(duration: float, dt: float = 0.001) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lit stationary scene — present but not moving.

    Returns:
        t_array:             (T,)    time array (s)
        v_scene_array:       (T, 3)  zeros
        scene_present_array: (T,)    ones
    """
    t = make_time(duration, dt)
    T = len(t)
    return t, np.zeros((T, 3), dtype=np.float32), np.ones(T, dtype=np.float32)


# ── Compound builders ──────────────────────────────────────────────────────────

def combine(t_ref: np.ndarray, **arrays) -> dict:
    """Assert all arrays match t_ref length; return as dict for simulate().

    Usage::
        stim = combine(t,
            head_vel_array=hv,
            p_target_array=pt,
            v_target_array=vt,
            scene_present_array=sp,
            target_present_array=tp,
        )
        states = simulate(params, t, **stim, return_states=True)
    """
    T = len(t_ref)
    out = {}
    for k, v in arrays.items():
        if v is None:
            continue
        arr = np.asarray(v)
        assert arr.shape[0] == T, (
            f"stimuli.combine: '{k}' has length {arr.shape[0]}, expected {T}"
        )
        out[k] = jnp.array(arr)
    return out
