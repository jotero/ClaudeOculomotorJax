"""Post-hoc analysis helpers for oculomotor simulations.

Shared across notebooks and scripts — import instead of copy-pasting.

Functions
---------
vs_net(states)
    Net velocity storage signal x_L − x_R, shape (T, 3).

ni_net(states)
    Net neural integrator signal x_L − x_R, shape (T, 3).

vs_null(states)
    VS null-adaptation state, shape (T, 3).

ni_null(states)
    NI null-adaptation state, shape (T, 3).

extract_canal(states)
    Canal head-velocity estimate (yaw) from SimState trajectory.

extract_burst(states, theta)
    Recompute u_burst (T, 3) via vmap over the saccade generator.

extract_sg(states, theta)
    Full SG signal extraction: x_copy, z_ref, e_held, z_sac, z_acc,
    e_res, e_pd, u_burst, x_ni.

extract_spv(t, eye_vel, burst, burst_threshold, margin_s)
    Slow-phase velocity using the burst signal to detect fast phases.

saccade_metrics(eye_pos_yaw, eye_vel_yaw, t_jump_idx, dt)
    Amplitude, peak velocity, and duration of the primary saccade.

fit_tc(t, y, t_start, t_end, label)
    Fit A·exp(−t/τ) + offset; return (tau, t_fit, y_fit).

ax_fmt(ax, ylabel, xlabel, ylim)
    Standard axes formatting (zero line, grid, labels, tick size).
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.ndimage import binary_dilation
from scipy.optimize import curve_fit

from oculomotor.sim.simulator import (
    _IDX_VS, _IDX_VS_L, _IDX_VS_R, _IDX_VS_NULL,
    _IDX_NI, _IDX_NI_L, _IDX_NI_R, _IDX_NI_NULL,
    _IDX_SG, _IDX_VIS, _IDX_VIS_L, _IDX_VIS_R,
)
from oculomotor.models.sensory_models.sensory_model import (
    N_CANALS, FLOOR, _SOFTNESS, PINV_SENS, C_pos, C_target_visible,
)
from oculomotor.models.brain_models import saccade_generator as sg_mod

try:
    from scipy.special import softplus as _sp_softplus
except ImportError:  # fallback
    def _sp_softplus(x):
        return np.log1p(np.exp(x))


# ── State extractors ──────────────────────────────────────────────────────────

def vs_net(states):
    """Net velocity storage signal: x_L − x_R, shape (T, 3), deg/s."""
    return (np.array(states.brain[:, _IDX_VS_L]) -
            np.array(states.brain[:, _IDX_VS_R]))


def ni_net(states):
    """Net neural integrator signal: x_L − x_R, shape (T, 3), deg."""
    return (np.array(states.brain[:, _IDX_NI_L]) -
            np.array(states.brain[:, _IDX_NI_R]))


def vs_null(states):
    """VS null-adaptation state, shape (T, 3), deg/s."""
    return np.array(states.brain[:, _IDX_VS_NULL])


def ni_null(states):
    """NI null-adaptation state, shape (T, 3), deg."""
    return np.array(states.brain[:, _IDX_NI_NULL])


# ── Canal ──────────────────────────────────────────────────────────────────────

def extract_canal(states):
    """Canal head-velocity estimate (yaw) from a SimState trajectory.

    Applies the same push-pull nonlinearity as canal.nonlinearity() but in
    NumPy so it works on the full trajectory at once.

    Args:
        states: SimState with .sensory of shape (T, N_SENSORY_STATES)

    Returns:
        (T,) array  yaw canal estimate (deg/s); positive = head rotating right.
    """
    x_c  = np.array(states.sensory[:, :N_CANALS * 2])   # (T, 12)
    x2   = x_c[:, N_CANALS:]                             # inertia states
    k, f = float(_SOFTNESS), float(FLOOR)
    y_c  = -f + _sp_softplus(k * (x2 + f)) / k + _sp_softplus(k * (x2 - f)) / k
    pinv = np.array(PINV_SENS)
    return (pinv @ y_c.T).T[:, 0]                        # yaw component (T,)


# ── Saccade generator ──────────────────────────────────────────────────────────

def extract_burst(states, theta):
    """Recompute u_burst (T, 3) via vmap over the saccade generator.

    Reads the delayed position error and visual-field gate from the visual
    delay cascade at each time step, then calls sg.step() to get u_burst.

    Args:
        states: SimState with .sensory and .brain, shape (T, ...)
        theta:  Params NamedTuple (reads theta.brain for SG params)

    Returns:
        (T, 3) float array  saccade burst command (yaw, pitch, roll) in deg/s.
        Slice [:, 0] for yaw only.
    """
    def _at(state):
        x_vis_L = state.sensory[_IDX_VIS_L]
        x_vis_R = state.sensory[_IDX_VIS_R]
        e_pd    = 0.5 * (C_pos  @ x_vis_L + C_pos  @ x_vis_R)   # L+R average
        gate    = 0.5 * ((C_target_visible @ x_vis_L)[0] + (C_target_visible @ x_vis_R)[0])
        x_ni_   = state.brain[_IDX_NI]
        x_ni_net = x_ni_[:3] - x_ni_[3:6]   # net: x_L − x_R (3,)
        _, u    = sg_mod.step(state.brain[_IDX_SG], e_pd, gate, x_ni_net, theta.brain)
        return u
    return np.array(jax.vmap(_at)(states))   # (T, 3)


def extract_sg(states, theta):
    """Full SG internal signal extraction.

    Returns a dict with all SG sub-states plus the recomputed burst.
    Useful for Robinson cascade diagnostic plots.

    Args:
        states: SimState trajectory (T, ...)
        theta:  Params NamedTuple

    Returns:
        dict with keys:
            x_copy  (T, 3)  internal eye-displacement copy
            z_ref   (T,)    refractory / OPN state
            e_held  (T, 3)  sample-and-hold of position error at saccade onset
            z_sac   (T,)    saccade latch state
            z_acc   (T,)    accumulator state
            e_res   (T, 3)  residual error = e_held − x_copy  (drives burst)
            e_pd    (T, 3)  delayed position error from visual cascade
            u_burst (T, 3)  recomputed saccade burst command
            x_ni    (T, 3)  neural integrator state (eye-position proxy)
    """
    x_sg    = np.array(states.brain[:, _IDX_SG])
    x_vis_L = np.array(states.sensory[:, _IDX_VIS_L])
    x_vis_R = np.array(states.sensory[:, _IDX_VIS_R])
    x_ni    = np.array(states.brain[:, _IDX_NI])

    x_copy = x_sg[:, :3]
    z_ref  = x_sg[:, 3]
    e_held = x_sg[:, 4:7]
    z_sac  = x_sg[:, 7]
    z_acc  = x_sg[:, 8]
    e_res  = e_held - x_copy
    # Use L+R average — matches what the brain's pos_delayed actually received
    e_pd   = 0.5 * (x_vis_L @ np.array(C_pos).T + x_vis_R @ np.array(C_pos).T)  # (T, 3)

    u_burst = extract_burst(states, theta)

    return dict(
        x_copy=x_copy, z_ref=z_ref, e_held=e_held,
        z_sac=z_sac,   z_acc=z_acc,  e_res=e_res,
        e_pd=e_pd,     u_burst=u_burst, x_ni=x_ni,
    )


# ── Slow-phase velocity ────────────────────────────────────────────────────────

def extract_spv(t, eye_vel, burst, burst_threshold=10.0, margin_s=0.05):
    """Slow-phase velocity using the saccade burst to detect fast phases.

    The burst signal is zero during slow phases regardless of SPV magnitude,
    making it more reliable than a velocity threshold (which misclassifies
    high-SPV slow phases as fast phases).

    Args:
        t:               (T,) time array (s)
        eye_vel:         (T,) eye angular velocity (deg/s)
        burst:           (T,) burst signal (yaw component from extract_burst)
        burst_threshold: deg/s  — burst amplitude that declares a fast phase
        margin_s:        s      — window expansion around each detected fast phase

    Returns:
        (T,) slow-phase velocity — fast phases replaced by linear interpolation.
    """
    dt       = float(t[1] - t[0])
    margin_n = max(1, int(margin_s / dt))
    is_fast  = np.abs(burst) > burst_threshold
    is_fast  = binary_dilation(is_fast, structure=np.ones(2 * margin_n + 1))
    slow     = ~is_fast
    if slow.sum() < 2:
        return eye_vel.copy()
    return np.interp(t, t[slow], eye_vel[slow])


# ── Saccade kinematics ────────────────────────────────────────────────────────

def saccade_metrics(eye_pos_yaw, eye_vel_yaw, t_jump_idx, dt):
    """Amplitude, peak velocity, and duration of the primary saccade.

    Duration is measured from the first crossing of 10% peak velocity to the
    first time velocity drops back below 10% peak velocity AFTER the peak —
    ignoring any corrective saccades that follow.

    Args:
        eye_pos_yaw: (T,) eye position yaw (deg)
        eye_vel_yaw: (T,) eye velocity yaw (deg/s)
        t_jump_idx:  int  index of the target jump in the arrays
        dt:          float  sample period (s)

    Returns:
        (amplitude_deg, peak_velocity_degs, duration_ms)
    """
    vel = eye_vel_yaw[t_jump_idx:]
    pos = eye_pos_yaw[t_jump_idx:]

    peak_v    = float(np.max(np.abs(vel)))
    threshold = 0.1 * peak_v
    above     = np.abs(vel) > threshold

    if above.any():
        start     = int(np.argmax(above))
        peak_idx  = int(np.argmax(np.abs(vel)))
        after_peak = np.abs(vel[peak_idx:]) < threshold
        end = peak_idx + int(np.argmax(after_peak)) if after_peak.any() else len(vel) - 1
        dur = (end - start) * dt * 1000      # ms
        amp = float(pos[end] - pos[0])
    else:
        dur = float('nan')
        amp = float(pos[-1] - pos[0])

    return amp, peak_v, dur


# ── Time-constant fitting ─────────────────────────────────────────────────────

def fit_tc(t, y, t_start, t_end, label=''):
    """Fit A·exp(−t/τ) + offset to a segment of a signal.

    Args:
        t:       (T,) time array (s)
        y:       (T,) signal array
        t_start: float  start of fit window (s)
        t_end:   float  end of fit window (s)
        label:   str    printed with the result (optional)

    Returns:
        (tau, t_fit, y_fit) on success, or (None, None, None) on failure.
        tau:   fitted time constant (s)
        t_fit: time array for the fit segment
        y_fit: fitted curve values
    """
    mask  = (t >= t_start) & (t <= t_end)
    t_seg = t[mask] - t_start
    y_seg = y[mask]
    try:
        p0   = (y_seg[0] - y_seg[-1], (t_end - t_start) / 3, y_seg[-1])
        popt, _ = curve_fit(
            lambda t, A, tau, c: A * np.exp(-t / tau) + c,
            t_seg, y_seg, p0=p0, maxfev=8000,
            bounds=([-np.inf, 0.1, -np.inf], [np.inf, 200, np.inf]))
        tau   = abs(popt[1])
        y_fit = popt[0] * np.exp(-t_seg / tau) + popt[2]
        if label:
            print(f'  {label}: τ = {tau:.1f} s')
        return tau, t[mask], y_fit
    except Exception as e:
        if label:
            print(f'  {label}: fit failed — {e}')
        return None, None, None


# ── Plot formatting ───────────────────────────────────────────────────────────

def ax_fmt(ax, ylabel='', xlabel='', ylim=None):
    """Standard axes formatting: zero line, grid, labels, tick size."""
    ax.axhline(0, color='gray', lw=0.4)
    ax.grid(True, alpha=0.2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=7)
