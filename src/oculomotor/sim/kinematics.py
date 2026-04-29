"""Kinematic trajectory builder for oculomotor simulations.

World frame is LEFT-HANDED: x=right, y=up, z=forward  (x × y = −z).
See retina.ypr_to_xyz / xyz_to_ypr for the [yaw,pitch,roll] ↔ xyz axis mapping.

KinematicTrajectory holds a 6-DOF trajectory (orientation + position) with all
three time derivatives pre-computed.  TargetTrajectory holds a 3-DOF position
trajectory with its Cartesian velocity (used in the ODE to compute the angular
velocity of the target direction via the cross-product formula).

Builder priority for each DOF (applied independently to rot and lin):
    If pos given  → vel = central_diff(pos),  acc = central_diff(vel)
    If vel given  → pos = cumtrapz(vel, pos_0), acc = central_diff(vel)
    If acc given  → vel = cumtrapz(acc), pos = cumtrapz(vel, pos_0)
    If nothing    → zeros

When building from segments, the segment-specified acceleration is preserved
directly (not re-derived from position) to maintain piecewise-constant steps.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class KinematicTrajectory:
    """6-DOF kinematic trajectory: orientation + position with all derivatives.

    World frame: x=right, y=up, z=forward (left-handed; x × y = −z).

    Rotation uses [yaw, pitch, roll] convention:
        yaw   = rotation around y-axis (rightward positive)
        pitch = rotation around x-axis (upward positive)
        roll  = rotation around z-axis (clockwise positive from subject's view)

    Fields
    ------
    t:       (T,) float32  time (s)
    rot_pos: (T,3) float32  rotation vector [yaw, pitch, roll]  deg
    rot_vel: (T,3) float32  angular velocity                    deg/s
    lin_pos: (T,3) float32  linear position  [x, y, z]          m
    lin_vel: (T,3) float32  linear velocity                     m/s
    lin_acc: (T,3) float32  linear acceleration                 m/s²
    """
    t:       np.ndarray
    rot_pos: np.ndarray
    rot_vel: np.ndarray
    lin_pos: np.ndarray
    lin_vel: np.ndarray
    lin_acc: np.ndarray


@dataclass
class TargetTrajectory:
    """3-DOF target position trajectory with Cartesian velocity.

    World frame: x=right, y=up, z=forward (left-handed; x × y = −z).

    lin_vel is the Cartesian derivative of lin_pos — always derived from
    lin_pos by central difference so the two are mutually consistent.
    It is used in the ODE to compute the angular velocity of the target
    direction:
        v_target_ypr = xyz_to_ypr( cross(lin_pos, lin_vel) / |lin_pos|² )  [deg/s]

    Fields
    ------
    t:       (T,)   float32  time (s)
    lin_pos: (T,3)  float32  Cartesian position  [x, y, z]  m
    lin_vel: (T,3)  float32  Cartesian velocity             m/s
    """
    t:       np.ndarray
    lin_pos: np.ndarray
    lin_vel: np.ndarray


# ── Numerical helpers ──────────────────────────────────────────────────────────

def _central_diff(arr: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Second-order central-difference derivative; forward/backward at endpoints."""
    arr = np.asarray(arr, dtype=np.float64)
    t   = np.asarray(t,   dtype=np.float64)
    out = np.empty_like(arr)
    dt_c = t[2:] - t[:-2]          # (T-2,) denominator for central diff
    if arr.ndim == 1:
        out[1:-1] = (arr[2:] - arr[:-2]) / dt_c
        out[0]    = (arr[1]  - arr[0])   / (t[1] - t[0])
        out[-1]   = (arr[-1] - arr[-2])  / (t[-1] - t[-2])
    else:
        out[1:-1] = (arr[2:] - arr[:-2]) / dt_c[:, None]
        out[0]    = (arr[1]  - arr[0])   / (t[1] - t[0])
        out[-1]   = (arr[-1] - arr[-2])  / (t[-1] - t[-2])
    return out


def _cumtrapz(arr: np.ndarray, t: np.ndarray, x0=None) -> np.ndarray:
    """Cumulative trapezoidal integral; x0 sets the initial value (default 0)."""
    arr = np.asarray(arr, dtype=np.float64)
    t   = np.asarray(t,   dtype=np.float64)
    dt  = np.diff(t)                # (T-1,)
    if arr.ndim == 1:
        increments = 0.5 * (arr[:-1] + arr[1:]) * dt
        out = np.empty_like(arr)
        out[0] = float(x0) if x0 is not None else 0.0
        out[1:] = out[0] + np.cumsum(increments)
    else:
        n          = arr.shape[1]
        increments = 0.5 * (arr[:-1] + arr[1:]) * dt[:, None]
        out = np.empty_like(arr)
        out[0] = np.asarray(x0, dtype=np.float64) if x0 is not None else np.zeros(n)
        out[1:] = out[0] + np.cumsum(increments, axis=0)
    return out


def _pad3(arr1d: np.ndarray, axis: str = 'yaw') -> np.ndarray:
    """Pad a (T,) array to (T, 3) along the specified [yaw, pitch, roll] axis."""
    T   = len(arr1d)
    out = np.zeros((T, 3), dtype=np.float32)
    idx = {'yaw': 0, 'pitch': 1, 'roll': 2}[axis]
    out[:, idx] = arr1d
    return out


def _angle_to_3d(yaw_deg: np.ndarray, pitch_deg: np.ndarray,
                 distance_m: float = 1.0) -> np.ndarray:
    """Convert angular target position to 3D Cartesian (x=right, y=up, z=fwd).

    Projects onto the z=distance_m plane (tangent projection):
        x = distance_m * tan(yaw_rad)
        y = distance_m * tan(pitch_rad)
        z = distance_m
    """
    x = distance_m * np.tan(np.radians(yaw_deg))
    y = distance_m * np.tan(np.radians(pitch_deg))
    z = np.full_like(x, float(distance_m), dtype=np.float64)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


# ── Rotation ↔ YPR helpers (numpy; consistent with retina.py ypr_to_xyz / xyz_to_ypr) ──

def ypr_to_xyz(ypr_deg: np.ndarray) -> np.ndarray:
    """[yaw,pitch,roll] deg → xyz rotation-axis vector rad (left-handed world frame)."""
    r = np.radians(ypr_deg)
    return np.array([-r[1], r[0], r[2]])    # [-pitch, yaw, roll]


def xyz_to_ypr(xyz_rad: np.ndarray) -> np.ndarray:
    """xyz rotation-axis vector rad → [yaw,pitch,roll] deg (left-handed world frame)."""
    return np.degrees([xyz_rad[1], -xyz_rad[0], xyz_rad[2]])   # [y, -x, z]


def ypr_batch_to_xyz(ypr_deg: np.ndarray) -> np.ndarray:
    """(N,3) [yaw,pitch,roll] deg → (N,3) xyz rotation-axis rad."""
    r = np.radians(ypr_deg)
    return np.stack([-r[:, 1], r[:, 0], r[:, 2]], axis=1)


def xyz_batch_to_ypr(xyz_rad: np.ndarray) -> np.ndarray:
    """(N,3) xyz rotation-axis rad → (N,3) [yaw,pitch,roll] deg."""
    return np.degrees(np.stack([xyz_rad[:, 1], -xyz_rad[:, 0], xyz_rad[:, 2]], axis=1))


# ── Rotation integration and differentiation (scipy-backed) ────────────────────

def _integrate_rotation(vel_ypr_degs: np.ndarray, t: np.ndarray,
                         pos_0_ypr_deg=None) -> np.ndarray:
    """Integrate [yaw,pitch,roll] deg/s → cumulative orientation (deg).

    Uses scipy Rotation composition for correctness at any rotation magnitude,
    including simultaneous multi-axis rotations (OVAR, tilts, etc.).

    Convention: R(q) maps head→world (world ← head), body-frame (intrinsic) angular velocity.
    Each step: R_new = R_old · dR_body  (right-compose body-frame increment).

    Intrinsic means velocities are in the moving head frame — canals, gyroscopes,
    and IMUs all measure intrinsic angular velocity.  For single-axis rotations
    starting from upright (R_old = I), intrinsic = extrinsic, so VOR/saccade/
    pursuit demos are unaffected.  OVAR with a pre-tilted head requires intrinsic:
    rot_pos_0=[0,0,tilt_deg] + rot_vel=[Ω,0,0] → rotation around the tilted body
    yaw axis, making gravity sweep sinusoidally in the head frame.

    Args:
        vel_ypr_degs:  (T, 3) angular velocity [yaw,pitch,roll] deg/s  (body frame)
        t:             (T,)   time array (s)
        pos_0_ypr_deg: (3,) or None  initial orientation (deg); default zeros

    Returns:
        (T, 3) float32  cumulative orientation [yaw,pitch,roll] deg
    """
    vel = np.asarray(vel_ypr_degs, np.float64)
    T   = len(t)
    dt  = np.diff(t.astype(np.float64))

    if pos_0_ypr_deg is not None:
        r = Rotation.from_rotvec(ypr_to_xyz(np.asarray(pos_0_ypr_deg, np.float64)))
    else:
        r = Rotation.identity()

    pos = np.empty((T, 3), np.float64)
    for i in range(T):
        pos[i] = xyz_to_ypr(r.as_rotvec())
        if i < T - 1:
            dr = Rotation.from_rotvec(ypr_to_xyz(vel[i]) * dt[i])
            r  = r * dr   # body-frame: right-compose increment
    return pos.astype(np.float32)


def _differentiate_rotation(pos_ypr_deg: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Differentiate [yaw,pitch,roll] deg orientation → body-frame angular velocity (deg/s).

    Uses Lie-group central differences (log of incremental rotation) for
    correctness at any rotation magnitude.

    Interior:   ω_i = log(R_{i-1}^{-1} · R_{i+1}) / (t_{i+1} − t_{i-1})
    Endpoints:  forward / backward 1st-order.

    Returns body-frame (intrinsic) angular velocity to match _integrate_rotation.

    Args:
        pos_ypr_deg: (T, 3)  orientation [yaw,pitch,roll] deg
        t:           (T,)    time array (s)

    Returns:
        (T, 3) float32  body-frame angular velocity [yaw,pitch,roll] deg/s
    """
    T    = len(pos_ypr_deg)
    t64  = t.astype(np.float64)
    pos  = np.asarray(pos_ypr_deg, np.float64)

    rv_xyz = ypr_batch_to_xyz(pos)           # (T, 3) xyz rad
    rs     = Rotation.from_rotvec(rv_xyz)     # batch of T rotations

    vel = np.empty((T, 3), np.float64)

    # Interior (central difference in Lie group)
    for i in range(1, T - 1):
        dt2  = t64[i + 1] - t64[i - 1]
        dr   = rs[i - 1].inv() * rs[i + 1]   # body-frame: R_{i-1}^{-1} · R_{i+1}
        vel[i] = xyz_to_ypr(dr.as_rotvec() / dt2)

    # Endpoints (first-order)
    vel[0]  = xyz_to_ypr((rs[0].inv() * rs[1]) .as_rotvec() / (t64[1]  - t64[0]))
    vel[-1] = xyz_to_ypr((rs[-2].inv() * rs[-1]).as_rotvec() / (t64[-1] - t64[-2]))

    return vel.astype(np.float32)


# ── Core DOF resolvers ──────────────────────────────────────────────────────────

def _resolve_rot_dof(pos, vel, acc, t: np.ndarray, pos_0
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve (pos, vel, acc) for 3-D rotation via scipy Rotation."""
    T = len(t)
    if pos is not None:
        pos = np.asarray(pos, np.float32)
        vel = _differentiate_rotation(pos, t)
        acc = _central_diff(vel, t)
    elif vel is not None:
        vel = np.asarray(vel, np.float32)
        p0  = np.asarray(pos_0, np.float64) if pos_0 is not None else np.zeros(3)
        pos = _integrate_rotation(vel, t, pos_0_ypr_deg=p0)
        acc = _central_diff(vel, t) if acc is None else np.asarray(acc, np.float32)
    elif acc is not None:
        acc = np.asarray(acc, np.float32)
        vel = _cumtrapz(acc, t).astype(np.float32)
        p0  = np.asarray(pos_0, np.float64) if pos_0 is not None else np.zeros(3)
        pos = _integrate_rotation(vel, t, pos_0_ypr_deg=p0)
    else:
        z = np.zeros((T, 3), np.float32)
        return z, z, z
    return pos.astype(np.float32), vel.astype(np.float32), acc.astype(np.float32)


def _resolve_lin_dof(pos, vel, acc, t: np.ndarray, pos_0
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve (pos, vel, acc) for 3-D translation (component-wise, Euclidean)."""
    T = len(t)
    if pos is not None:
        pos = np.asarray(pos, np.float64)
        vel = _central_diff(pos, t)
        acc = _central_diff(vel, t)
    elif vel is not None:
        vel = np.asarray(vel, np.float64)
        p0  = np.asarray(pos_0, np.float64) if pos_0 is not None else np.zeros(vel.shape[-1] if vel.ndim > 1 else 1)
        pos = _cumtrapz(vel, t, x0=p0)
        acc = _central_diff(vel, t) if acc is None else np.asarray(acc, np.float64)
    elif acc is not None:
        acc = np.asarray(acc, np.float64)
        vel = _cumtrapz(acc, t)
        p0  = np.asarray(pos_0, np.float64) if pos_0 is not None else np.zeros(acc.shape[-1] if acc.ndim > 1 else 1)
        pos = _cumtrapz(vel, t, x0=p0)
    else:
        z = np.zeros((T, 3), np.float64)
        return z.astype(np.float32), z.astype(np.float32), z.astype(np.float32)
    return pos.astype(np.float32), vel.astype(np.float32), acc.astype(np.float32)


def build_kinematics(
    t,
    *,
    rot_pos=None, rot_vel=None, rot_acc=None,
    lin_pos=None, lin_vel=None, lin_acc=None,
    rot_pos_0=None, lin_pos_0=None,
) -> KinematicTrajectory:
    """Build a KinematicTrajectory from any combination of pos / vel / acc.

    Priority per DOF (rot and lin resolved independently):
        pos given → vel = diff(pos), acc = diff(vel)
        vel given (no pos) → pos = cumtrapz(vel, rot_pos_0), acc = diff(vel)
        acc given (no pos, no vel) → vel = cumtrapz(acc), pos = cumtrapz(vel)
        nothing given → zeros

    If both pos and vel/acc are provided, position takes precedence and the
    others are re-derived from it to guarantee internal consistency.

    Args:
        t:         (T,)   time array (s)
        rot_pos:   (T,3) or None  rotation vector [yaw,pitch,roll]  deg
        rot_vel:   (T,3) or None  angular velocity                   deg/s
        rot_acc:   (T,3) or None  angular acceleration               deg/s²
        lin_pos:   (T,3) or None  linear position  [x,y,z]          m
        lin_vel:   (T,3) or None  linear velocity                   m/s
        lin_acc:   (T,3) or None  linear acceleration               m/s²
        rot_pos_0: (3,) or None   initial rotation when integrating rot_vel/acc
        lin_pos_0: (3,) or None   initial position when integrating lin_vel/acc

    Returns:
        KinematicTrajectory with all five fields populated.
    """
    t = np.asarray(t, dtype=np.float64)
    rpos, rvel, racc = _resolve_rot_dof(rot_pos, rot_vel, rot_acc, t, rot_pos_0)
    lpos, lvel, lacc = _resolve_lin_dof(lin_pos, lin_vel, lin_acc, t, lin_pos_0)
    return KinematicTrajectory(
        t=t.astype(np.float32),
        rot_pos=rpos, rot_vel=rvel,
        lin_pos=lpos, lin_vel=lvel, lin_acc=lacc,
    )


def build_target(
    t,
    *,
    yaw_deg=None,
    pitch_deg=None,
    vel_yaw_deg_s=None,
    vel_pitch_deg_s=None,
    distance_m: float = 1.0,
    lin_pos=None,
    lin_vel=None,
) -> TargetTrajectory:
    """Build a TargetTrajectory from angular or Cartesian specification.

    Priority:
        lin_pos given → lin_vel = central_diff(lin_pos)  unless lin_vel is provided
        yaw_deg / pitch_deg given → convert to Cartesian via tan-projection,
                                    then derive lin_vel
        vel_yaw_deg_s / vel_pitch_deg_s given (no pos) → integrate to angle,
                                    then convert to Cartesian

    Args:
        t:               (T,) time array (s)
        yaw_deg:         (T,) or scalar  target yaw angle (deg)
        pitch_deg:       (T,) or scalar  target pitch angle (deg)
        vel_yaw_deg_s:   (T,) or scalar  target yaw angular velocity (deg/s)
        vel_pitch_deg_s: (T,) or scalar  target pitch angular velocity (deg/s)
        distance_m:      target depth (m), default 1.0
        lin_pos:         (T,3) Cartesian override (skips angular spec)
        lin_vel:         (T,3) explicit Cartesian velocity override. When provided,
                         skips central-difference derivation from lin_pos.
                         Use this for step-function targets (e.g. saccade target
                         that appears at a new location instantaneously) to avoid
                         a large velocity spike at the step that would
                         contaminate the pursuit integrator.

    Returns:
        TargetTrajectory with lin_pos and lin_vel.
    """
    t  = np.asarray(t, dtype=np.float64)
    T  = len(t)

    if lin_pos is not None:
        pos = np.asarray(lin_pos, dtype=np.float64)
    elif yaw_deg is not None or pitch_deg is not None:
        ya  = np.broadcast_to(np.asarray(yaw_deg,   dtype=np.float64), T) if yaw_deg   is not None else np.zeros(T)
        pa  = np.broadcast_to(np.asarray(pitch_deg, dtype=np.float64), T) if pitch_deg is not None else np.zeros(T)
        pos = _angle_to_3d(ya, pa, distance_m).astype(np.float64)
    elif vel_yaw_deg_s is not None or vel_pitch_deg_s is not None:
        vy = np.broadcast_to(np.asarray(vel_yaw_deg_s,   dtype=np.float64), T) if vel_yaw_deg_s   is not None else np.zeros(T)
        vp = np.broadcast_to(np.asarray(vel_pitch_deg_s, dtype=np.float64), T) if vel_pitch_deg_s is not None else np.zeros(T)
        ya  = _cumtrapz(vy, t).flatten()
        pa  = _cumtrapz(vp, t).flatten()
        pos = _angle_to_3d(ya, pa, distance_m).astype(np.float64)
    else:
        pos = _angle_to_3d(np.zeros(T), np.zeros(T), distance_m).astype(np.float64)

    if lin_vel is not None:
        vel = np.asarray(lin_vel, dtype=np.float64)
    else:
        vel = _central_diff(pos, t)
    return TargetTrajectory(
        t=t.astype(np.float32),
        lin_pos=pos.astype(np.float32),
        lin_vel=vel.astype(np.float32),
    )


# ── Segment-based builder (replaces stimuli.build_body_arrays) ─────────────────

def build_kinematics_from_segments(
    segments,
    total_T: int,
    dt: float = 0.001,
    default_lin_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> KinematicTrajectory:
    """Build a KinematicTrajectory from a list of motion segments.

    Segments are duck-typed objects with the attributes used by the LLM
    pipeline's BodySegment schema.  No import from llm_pipeline required.

    The segment-specified acceleration is preserved directly for the linear
    DOFs (piecewise-constant values, not re-derived from position) so that
    translational step stimuli produce correct otolith impulse responses.

    Rotational acceleration is derived by central-differencing rot_vel after
    concatenating all segments.

    Args:
        segments:        list of BodySegment-like objects
        total_T:         total number of time steps
        dt:              time step (s)
        default_lin_pos: carry-forward initial linear position (x, y, z)  m

    Returns:
        KinematicTrajectory with all fields populated.
    """
    rot_carry = {ax: [0.0, 0.0] for ax in ('yaw', 'pitch', 'roll')}
    lin_carry = {
        'x': [default_lin_pos[0], 0.0],
        'y': [default_lin_pos[1], 0.0],
        'z': [default_lin_pos[2], 0.0],
    }

    rot_pos_c = [[], [], []]   # per-DOF chunk lists: [yaw, pitch, roll]
    rot_vel_c = [[], [], []]
    lin_pos_c = [[], [], []]   # per-DOF chunk lists: [x, y, z]
    lin_vel_c = [[], [], []]
    lin_acc_c = [[], [], []]

    _ROT = [
        ('yaw',   'rot_yaw_0',   'rot_yaw_vel',   'rot_yaw_acc'),
        ('pitch', 'rot_pitch_0', 'rot_pitch_vel', 'rot_pitch_acc'),
        ('roll',  'rot_roll_0',  'rot_roll_vel',  'rot_roll_acc'),
    ]
    _LIN = [
        ('x', 'lin_x_0', 'lin_x_vel', 'lin_x_acc'),
        ('y', 'lin_y_0', 'lin_y_vel', 'lin_y_acc'),
        ('z', 'lin_z_0', 'lin_z_vel', 'lin_z_acc'),
    ]

    for seg in segments:
        T   = max(1, round(getattr(seg, 'duration_s', 0.0) / dt))
        t_s = np.arange(T, dtype=np.float64) * dt
        profile = getattr(seg, 'rot_profile', 'constant')

        # ── Rotational DOFs ───────────────────────────────────────────────────
        for i, (ax, p0f, v0f, accf) in enumerate(_ROT):
            carry_pos, carry_vel = rot_carry[ax]
            p0  = getattr(seg, p0f,  None)
            v0  = getattr(seg, v0f,  None)
            acc = getattr(seg, accf, 0.0) or 0.0

            if p0 is not None: carry_pos = float(p0)
            if v0 is not None: carry_vel = float(v0)

            if profile == 'sinusoid':
                A = carry_vel
                w = 2.0 * np.pi * getattr(seg, 'frequency_hz', 0.0)
                vel = A * np.sin(w * t_s)
                pos = carry_pos + (A / w * (1.0 - np.cos(w * t_s)) if w > 0 else np.zeros(T))
                carry_pos = float(pos[-1]); carry_vel = float(vel[-1])

            elif profile == 'impulse':
                A  = carry_vel
                rd = getattr(seg, 'ramp_dur_s', 0.02)
                vel = np.zeros(T); pos = np.zeros(T)
                rise  = t_s < rd
                fall  = (t_s >= rd) & (t_s < 2.0 * rd)
                coast = t_s >= 2.0 * rd
                vel[rise] = A * t_s[rise] / rd
                vel[fall] = A * (1.0 - (t_s[fall] - rd) / rd)
                pos[rise] = carry_pos + A * t_s[rise]**2 / (2.0 * rd)
                pos_at_rd  = carry_pos + A * rd / 2.0
                tf = t_s[fall] - rd
                pos[fall]  = pos_at_rd + A * tf - A * tf**2 / (2.0 * rd)
                pos_at_end = carry_pos + A * rd
                pos[coast] = pos_at_end
                carry_pos = float(pos_at_end); carry_vel = 0.0

            else:   # 'constant' / polynomial
                pos = carry_pos + carry_vel * t_s + 0.5 * acc * t_s**2
                vel = carry_vel + acc * t_s
                carry_pos = float(pos[-1]); carry_vel = float(vel[-1])

            rot_carry[ax] = [carry_pos, carry_vel]
            rot_pos_c[i].append(pos.astype(np.float32))
            rot_vel_c[i].append(vel.astype(np.float32))

        # ── Linear DOFs (always polynomial) ──────────────────────────────────
        for i, (ax, p0f, v0f, accf) in enumerate(_LIN):
            carry_pos, carry_vel = lin_carry[ax]
            p0  = getattr(seg, p0f,  None)
            v0  = getattr(seg, v0f,  None)
            acc = getattr(seg, accf, 0.0) or 0.0

            if p0 is not None: carry_pos = float(p0)
            if v0 is not None: carry_vel = float(v0)

            pos     = carry_pos + carry_vel * t_s + 0.5 * acc * t_s**2
            vel     = carry_vel + acc * t_s
            acc_arr = np.full(T, float(acc), dtype=np.float32)

            lin_carry[ax] = [float(pos[-1]), float(vel[-1])]
            lin_pos_c[i].append(pos.astype(np.float32))
            lin_vel_c[i].append(vel.astype(np.float32))
            lin_acc_c[i].append(acc_arr)

    def _cat(chunks_per_dof) -> np.ndarray:
        cols = [np.concatenate(c) for c in chunks_per_dof]
        return np.stack(cols, axis=1)   # (segment_T, 3)

    def _fit(arr, N):
        n = len(arr)
        if n >= N: return arr[:N]
        return np.concatenate([arr, np.tile(arr[-1:], (N - n, 1))], axis=0)

    rot_pos = _fit(_cat(rot_pos_c), total_T)
    rot_vel = _fit(_cat(rot_vel_c), total_T)
    lin_pos = _fit(_cat(lin_pos_c), total_T)
    lin_vel = _fit(_cat(lin_vel_c), total_T)
    lin_acc = _fit(_cat(lin_acc_c), total_T)

    # Derive rot_acc from rot_vel via central differences
    t_arr   = np.arange(total_T, dtype=np.float64) * dt
    rot_acc = _central_diff(rot_vel.astype(np.float64), t_arr).astype(np.float32)

    t_out = (np.arange(total_T, dtype=np.float64) * dt).astype(np.float32)
    return KinematicTrajectory(
        t=t_out,
        rot_pos=rot_pos, rot_vel=rot_vel,
        lin_pos=lin_pos, lin_vel=lin_vel, lin_acc=lin_acc,
    )


# ── Head convenience wrappers ─────────────────────────────────────────────────

def make_time(duration: float, dt: float = 0.001) -> np.ndarray:
    """Return a float32 time array [0, duration) with step dt."""
    return np.arange(0.0, duration, dt, dtype=np.float32)


def head_stationary(duration: float, dt: float = 0.001) -> KinematicTrajectory:
    """Stationary head — no motion."""
    return build_kinematics(make_time(duration, dt))


def head_rotation_step(
    velocity_deg_s: float,
    rotate_dur: float,
    coast_dur: float = 0.0,
    dt: float = 0.001,
    axis: str = 'yaw',
) -> KinematicTrajectory:
    """Step-velocity head rotation: constant velocity then coast.

    Args:
        velocity_deg_s: head angular velocity during rotation (deg/s)
        rotate_dur:     duration of constant-velocity rotation (s)
        coast_dur:      duration of zero-velocity coast after rotation (s)
        dt:             time step (s)
        axis:           'yaw', 'pitch', or 'roll'

    Returns:
        KinematicTrajectory
    """
    t    = make_time(rotate_dur + coast_dur, dt)
    vel  = np.where(t < rotate_dur, velocity_deg_s, 0.0).astype(np.float32)
    return build_kinematics(t, rot_vel=_pad3(vel, axis))


def head_rotation_sinusoid(
    amplitude_deg_s: float,
    frequency_hz: float,
    duration: float,
    dt: float = 0.001,
    axis: str = 'yaw',
) -> KinematicTrajectory:
    """Sinusoidal head rotation (for VOR gain/phase measurement).

    Args:
        amplitude_deg_s: peak velocity (deg/s)
        frequency_hz:    frequency (Hz)
        duration:        total duration (s)
        dt:              time step (s)
        axis:            'yaw', 'pitch', or 'roll'
    """
    t   = make_time(duration, dt)
    vel = (amplitude_deg_s * np.sin(2.0 * np.pi * frequency_hz * t)).astype(np.float32)
    return build_kinematics(t, rot_vel=_pad3(vel, axis))


def head_impulse(
    amplitude_deg_s: float,
    ramp_dur: float = 0.02,
    total_dur: float = 2.0,
    dt: float = 0.001,
    axis: str = 'yaw',
) -> KinematicTrajectory:
    """Head impulse test (HIT): triangular velocity pulse.

    Args:
        amplitude_deg_s: peak head velocity (deg/s); typical HIT 150–300 deg/s
        ramp_dur:        rise and fall duration (s); default 20 ms
        total_dur:       total trial duration (s)
        dt:              time step (s)
        axis:            'yaw', 'pitch', or 'roll'
    """
    t    = make_time(total_dur, dt)
    vel  = np.zeros(len(t), dtype=np.float32)
    rise = t < ramp_dur
    fall = (t >= ramp_dur) & (t < 2.0 * ramp_dur)
    vel[rise] = amplitude_deg_s * t[rise] / ramp_dur
    vel[fall] = amplitude_deg_s * (1.0 - (t[fall] - ramp_dur) / ramp_dur)
    return build_kinematics(t, rot_vel=_pad3(vel, axis))


# ── Scene convenience wrappers ────────────────────────────────────────────────

def scene_dark(duration: float, dt: float = 0.001) -> KinematicTrajectory:
    """Dark scene — no motion, no light (caller sets scene_present=0)."""
    return build_kinematics(make_time(duration, dt))


def scene_stationary(duration: float, dt: float = 0.001) -> KinematicTrajectory:
    """Lit stationary scene — present but not moving."""
    return build_kinematics(make_time(duration, dt))


def scene_rotation_step(
    velocity_deg_s: float,
    on_dur: float,
    total_dur: float,
    dt: float = 0.001,
    axis: str = 'yaw',
) -> KinematicTrajectory:
    """Full-field scene rotation (OKN stimulus): constant velocity for on_dur, then stops.

    After scene stops, OKAN persists if caller keeps scene_present=1.

    Args:
        velocity_deg_s: scene angular velocity during motion (deg/s)
        on_dur:         duration of scene motion (s)
        total_dur:      total trial duration; should be > on_dur for OKAN (s)
        dt:             time step (s)
        axis:           'yaw', 'pitch', or 'roll'
    """
    t   = make_time(total_dur, dt)
    vel = np.where(t < on_dur, velocity_deg_s, 0.0).astype(np.float32)
    return build_kinematics(t, rot_vel=_pad3(vel, axis))


# ── Target convenience wrappers ───────────────────────────────────────────────

def target_stationary(
    duration: float,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    distance_m: float = 1.0,
    dt: float = 0.001,
) -> TargetTrajectory:
    """Stationary fixation target at a fixed angular position.

    Args:
        duration:    trial duration (s)
        yaw_deg:     horizontal eccentricity (deg, rightward positive)
        pitch_deg:   vertical eccentricity (deg, upward positive)
        distance_m:  target depth (m)
        dt:          time step (s)
    """
    t = make_time(duration, dt)
    return build_target(t, yaw_deg=yaw_deg, pitch_deg=pitch_deg, distance_m=distance_m)


def target_steps(
    jumps: list[tuple[float, float, float]],
    duration: float,
    distance_m: float = 1.0,
    dt: float = 0.001,
) -> TargetTrajectory:
    """Sequence of instantaneous target position steps.

    Args:
        jumps:      list of (t_jump_s, yaw_deg, pitch_deg); target is at (0,0)
                    until the first jump.  Sorted automatically.
        duration:   total trial duration (s)
        distance_m: target depth (m)
        dt:         time step (s)
    """
    t = make_time(duration, dt)
    T = len(t)
    ya = np.zeros(T, dtype=np.float64)
    pa = np.zeros(T, dtype=np.float64)
    for t_j, y, p in sorted(jumps):
        ya[t >= t_j] = y
        pa[t >= t_j] = p
    return build_target(t, yaw_deg=ya, pitch_deg=pa, distance_m=distance_m)


def target_ramp(
    velocity_deg_s: float,
    t_start: float = 0.2,
    duration: float = 3.0,
    distance_m: float = 1.0,
    dt: float = 0.001,
    axis: str = 'yaw',
) -> TargetTrajectory:
    """Ramp target: stationary until t_start, then moves at constant angular velocity.

    Simulates a pursuit stimulus.

    Args:
        velocity_deg_s: target angular velocity after onset (deg/s)
        t_start:        time when target starts moving (s)
        duration:       total trial duration (s)
        distance_m:     target depth (m)
        dt:             time step (s)
        axis:           'yaw' or 'pitch'
    """
    t   = make_time(duration, dt)
    T   = len(t)
    deg = np.where(t >= t_start, velocity_deg_s * (t - t_start), 0.0)
    ya  = deg if axis == 'yaw'   else np.zeros(T)
    pa  = deg if axis == 'pitch' else np.zeros(T)
    return build_target(t, yaw_deg=ya, pitch_deg=pa, distance_m=distance_m)
