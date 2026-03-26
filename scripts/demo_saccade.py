"""Saccade generator demos.

Figures produced
────────────────
    saccade_single.png   — single saccade: signal cascade (7 panels)
    saccade_sequence.png — multiple saccades showing re-triggering
    saccade_vor.png      — combined saccade + VOR: corrective saccades during rotation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from oculomotor.models.ocular_motor_simulator import (
    simulate, THETA_DEFAULT,
    vor_vector_field, _N_TOTAL,
    _IDX_NI, _IDX_P, _IDX_SG, _IDX_VIS,
)
from oculomotor.models import retina
from oculomotor.models import saccade_generator as sg
from oculomotor.models import visual_delay
import diffrax

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# ── Shared theme ───────────────────────────────────────────────────────────────

_C = {
    'head':   '#555555',
    'eye':    '#2166ac',
    'target': '#d6604d',
    'burst':  '#f4a582',
    'ni':     '#4dac26',
    'error':  '#762a83',
    'vel':    '#1a9641',
    'reset':  '#e08214',
}

# ── Saccade theta (VOR disabled — head still) ─────────────────────────────────

THETA_SAC = {**THETA_DEFAULT,
             'g_vor':          0.0,   # no VOR — head is still
             'K_vis':          0.0,   # no visual → dark / saccade only
             'g_vis':          0.0,
             'g_burst':      600.0,   # burst ceiling (deg/s); peak vel → g_burst as amp → ∞
             'threshold_sac':  0.5,   # trigger threshold (deg)
             'k_sac':         50.0,   # sigmoid steepness (1/deg)
             'e_sat_sac':      7.0,   # tanh saturation amplitude (deg)
             'tau_reset_sac':  1.0,   # slow reset TC (s) — active when target present
             'tau_reset_fast': 0.1,   # fast reset TC (s) — kicks in after error < threshold
             }


# ── Utility: run simulator and extract intermediate signals ───────────────────

def _extract_all(theta, t_array, hv3, vs3, pt3, max_steps=100000):
    """Run simulator and extract full state + intermediate signals."""
    T             = len(t_array)
    gains_array   = jnp.ones(6)
    dt_arr        = jnp.diff(t_array)
    hp3           = jnp.concatenate([
        jnp.zeros((1, 3)),
        jnp.cumsum(0.5 * (hv3[:-1] + hv3[1:]) * dt_arr[:, None], axis=0),
    ])
    # Saccade demos: target is visible → scene present for position error pathway.
    # Visual (OKR) drive is controlled via K_vis / g_vis in theta.
    sg1               = jnp.ones(T, dtype=jnp.float32)
    hv_interp         = diffrax.LinearInterpolation(ts=t_array, ys=hv3)
    hp_interp         = diffrax.LinearInterpolation(ts=t_array, ys=hp3)
    vs_interp         = diffrax.LinearInterpolation(ts=t_array, ys=vs3)
    target_interp     = diffrax.LinearInterpolation(ts=t_array, ys=pt3)
    scene_gain_interp = diffrax.LinearInterpolation(ts=t_array, ys=sg1)
    x0                = jnp.zeros(_N_TOTAL)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(vor_vector_field),
        diffrax.Heun(),
        t0=t_array[0],
        t1=t_array[-1],
        dt0=0.001,
        y0=x0,
        args=(theta, hv_interp, hp_interp, vs_interp, target_interp,
              scene_gain_interp, gains_array),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=t_array),
        max_steps=max_steps,
    )

    ys = solution.ys   # (T, _N_TOTAL)

    def _signals_at(x, t):
        x_p           = x[_IDX_P]
        x_ni          = x[_IDX_NI]
        x_vis         = x[_IDX_VIS]
        x_reset_int   = x[_IDX_SG]          # (3,) resettable integrator
        p_t           = target_interp.evaluate(t)
        e_motor       = retina.target_to_angle(p_t) - x_p
        e_pos_delayed = visual_delay.C_pos @ x_vis   # what the SG actually sees
        _, u_burst    = sg.step(x_reset_int, e_pos_delayed, theta)  # matches ODE
        return {'q_eye': x_p, 'x_ni': x_ni, 'e_motor': e_motor,
                'e_pos_delayed': e_pos_delayed,
                'x_reset_int': x_reset_int, 'u_burst': u_burst}

    raw = jax.vmap(lambda x, t: _signals_at(x, t))(ys, t_array)
    return {k: np.array(v) for k, v in raw.items()}


# ── Demo 1: single saccade ─────────────────────────────────────────────────────

def demo_saccade_single():
    """Saccades of 4 amplitudes (2, 5, 10, 20°) — 4 columns × 4 signal rows."""
    amplitudes = [2.0, 5.0, 10.0, 20.0]
    dt     = 0.001
    T_end  = 1.0
    t_jump = 0.2
    t      = jnp.arange(0.0, T_end, dt)
    T      = len(t)
    t_np   = np.array(t)

    n_rows, n_cols = 4, len(amplitudes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.8 * n_rows),
                             sharex=True)
    fig.suptitle('Saccade Generator — Signal Cascade', fontsize=13)

    row_labels = ['Position (deg)', 'Error / Copy (deg)',
                  'Burst + Velocity (deg/s)', 'Copy vs NI (deg)']
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8)

    for ci, amp in enumerate(amplitudes):
        hv3 = jnp.zeros((T, 3))
        vs3 = jnp.zeros((T, 3))
        pt3 = jnp.zeros((T, 3))
        pt3 = pt3.at[:, 2].set(1.0)
        pt3 = pt3.at[t >= t_jump, 0].set(jnp.tan(jnp.radians(amp)))

        sig        = _extract_all(THETA_SAC, t, hv3, vs3, pt3)
        target_yaw = np.degrees(np.arctan2(np.array(pt3[:, 0]), np.array(pt3[:, 2])))
        eye_vel    = np.gradient(sig['q_eye'][:, 0], t_np)

        def _vl(ax): ax.axvline(t_jump, color='gray', lw=0.7, ls='--', alpha=0.5)

        axes[0, ci].set_title(f'{amp:.0f}°', fontsize=10)

        # Row 0: position
        axes[0, ci].plot(t_np, target_yaw,         color=_C['target'], lw=1.5, label='target')
        axes[0, ci].plot(t_np, sig['q_eye'][:, 0], color=_C['eye'],    lw=1.5, label='eye')
        axes[0, ci].plot(t_np, sig['x_ni'][:, 0],  color=_C['ni'],     lw=1.0, ls='--', label='NI cmd')
        _vl(axes[0, ci]); axes[0, ci].axhline(0, color='k', lw=0.4)
        if ci == 0: axes[0, ci].legend(fontsize=7)

        # Row 1: delayed error & resettable integrator
        axes[1, ci].plot(t_np, sig['e_motor'][:, 0],       color=_C['error'], lw=1.5, label='e_motor')
        axes[1, ci].plot(t_np, sig['e_pos_delayed'][:, 0], color=_C['error'], lw=1.0, ls='--', label='e_delayed')
        axes[1, ci].plot(t_np, sig['x_reset_int'][:, 0],   color=_C['reset'], lw=1.5, label='x_reset_int')
        _vl(axes[1, ci]); axes[1, ci].axhline(0, color='k', lw=0.4)
        if ci == 0: axes[1, ci].legend(fontsize=7)

        # Row 2: burst command + eye velocity (twin-y)
        ax2 = axes[2, ci]
        ax2.plot(t_np, sig['u_burst'][:, 0], color=_C['burst'], lw=1.5, label='u_burst')
        ax2.plot(t_np, eye_vel,               color=_C['vel'],   lw=1.2, ls='--', label='eye vel')
        _vl(ax2); ax2.axhline(0, color='k', lw=0.4)
        if ci == 0: ax2.legend(fontsize=7)

        # Row 3: copy vs NI
        axes[3, ci].plot(t_np, sig['x_reset_int'][:, 0], color=_C['reset'], lw=1.5, label='x_reset_int')
        axes[3, ci].plot(t_np, sig['x_ni'][:, 0],         color=_C['ni'],    lw=1.0, ls='--', label='x_ni')
        axes[3, ci].axhline(amp, color=_C['target'], lw=0.8, ls=':', label='target')
        _vl(axes[3, ci]); axes[3, ci].axhline(0, color='k', lw=0.4)
        if ci == 0: axes[3, ci].legend(fontsize=7)

        axes[n_rows - 1, ci].set_xlabel('Time (s)')
        for r in range(n_rows):
            axes[r, ci].set_xlim(0, T_end)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'saccade_single.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Saved {path}')


# ── Demo 2: saccade sequence ───────────────────────────────────────────────────

def demo_saccade_sequence():
    """Three target steps: 0° → 10° → 0° → 15°."""
    dt    = 0.001
    T_end = 2.5
    t     = jnp.arange(0.0, T_end, dt)
    T     = len(t)
    t_np  = np.array(t)

    hv3 = jnp.zeros((T, 3))
    vs3 = jnp.zeros((T, 3))

    jumps = [(0.5, 10.0), (1.0, 0.0), (1.6, 15.0)]
    tgt_deg = np.zeros(T)
    for t_j, ang in jumps:
        tgt_deg[t_np >= t_j] = ang

    pt3 = jnp.zeros((T, 3))
    pt3 = pt3.at[:, 2].set(1.0)
    pt3 = pt3.at[:, 0].set(jnp.array(np.tan(np.radians(tgt_deg))))

    sig = _extract_all(THETA_SAC, t, hv3, vs3, pt3)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle('Saccade Sequence — 0° → 10° → 0° → 15°', fontsize=12)

    for t_j, _ in jumps:
        for ax in axes: ax.axvline(t_j, color='gray', lw=0.8, ls='--', alpha=0.5)

    axes[0].plot(t_np, tgt_deg,               color=_C['target'], lw=1.5, label='target')
    axes[0].plot(t_np, sig['q_eye'][:, 0],    color=_C['eye'],    lw=1.5, label='eye')
    axes[0].plot(t_np, sig['x_ni'][:, 0],     color=_C['ni'],     lw=1.2, ls='--', label='x_ni')
    axes[0].set_ylabel('deg'); axes[0].set_title('Eye Position vs Target')
    axes[0].legend(fontsize=8); axes[0].axhline(0, color='k', lw=0.5)

    axes[1].plot(t_np, sig['u_burst'][:, 0],      color=_C['burst'], lw=1.5)
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_ylabel('deg/s'); axes[1].set_title('Burst Output  u_burst')

    axes[2].plot(t_np, sig['x_reset_int'][:, 0], color=_C['reset'], lw=1.5, label='x_reset_int')
    axes[2].axhline(0, color='k', lw=0.5)
    axes[2].set_ylabel('deg'); axes[2].set_title('Resettable Integrator  x_reset_int')
    axes[2].set_xlabel('Time (s)'); axes[2].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'saccade_sequence.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Saved {path}')


# ── Demo 3: saccade + VOR ──────────────────────────────────────────────────────

def demo_saccade_vor():
    """Head rotates 60°/s; corrective saccades bring eye back to target (0°)."""
    dt    = 0.001
    T_end = 3.0
    t     = jnp.arange(0.0, T_end, dt)
    T     = len(t)
    t_np  = np.array(t)

    head_vel = np.where(t_np < 1.5, 60.0, 0.0).astype(np.float32)
    hv3 = jnp.stack([jnp.array(head_vel), jnp.zeros(T), jnp.zeros(T)], axis=1)
    vs3 = jnp.zeros((T, 3))
    pt3 = jnp.zeros((T, 3)); pt3 = pt3.at[:, 2].set(1.0)   # target: straight ahead

    theta_vor_sac = {**THETA_DEFAULT,
                     'g_vor':         1.0,
                     'K_vis':         0.0,   # dark — no OKR
                     'g_vis':         0.0,
                     'g_burst':      40.0,
                     'threshold_sac': 3.0,   # larger → fewer, larger saccades
                     'k_sac':        10.0,
                     'tau_reset_sac': 0.2,
                     }
    theta_vor_only = {**theta_vor_sac, 'g_burst': 0.0}

    sig    = _extract_all(theta_vor_sac,  t, hv3, vs3, pt3)
    sig_ns = _extract_all(theta_vor_only, t, hv3, vs3, pt3)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('VOR + Corrective Saccades (Head 60°/s, Target 0°)', fontsize=12)

    head_pos = np.cumsum(head_vel) * dt
    axes[0].plot(t_np, head_pos,                   color=_C['head'],   lw=1.0, ls=':', label='head pos')
    axes[0].plot(t_np, sig_ns['q_eye'][:, 0],      color=_C['eye'],    lw=1.2, ls='--', alpha=0.6, label='VOR only')
    axes[0].plot(t_np, sig['q_eye'][:, 0],         color=_C['eye'],    lw=1.5, label='VOR + saccade')
    axes[0].axhline(0, color=_C['target'], lw=1.0, ls='--', label='target (0°)')
    axes[0].set_ylabel('deg'); axes[0].set_title('Eye Position')
    axes[0].legend(fontsize=8)

    axes[1].plot(t_np, sig['u_burst'][:, 0], color=_C['burst'], lw=1.5)
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_ylabel('deg/s'); axes[1].set_title('Saccade Burst  u_burst')

    axes[2].plot(t_np, sig['e_motor'][:, 0],  color=_C['error'], lw=1.5, label='e_motor')
    axes[2].axhline( theta_vor_sac['threshold_sac'], color='gray', lw=0.8, ls=':', label='±threshold')
    axes[2].axhline(-theta_vor_sac['threshold_sac'], color='gray', lw=0.8, ls=':')
    axes[2].axhline(0, color='k', lw=0.5)
    axes[2].set_ylabel('deg'); axes[2].set_title('Motor Error  e_motor')
    axes[2].set_xlabel('Time (s)'); axes[2].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'saccade_vor.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Saved {path}')


# ── Demo 4: main sequence ──────────────────────────────────────────────────────

def demo_main_sequence():
    """Saccades from 0.5° to 20° — peak velocity vs amplitude (main sequence)."""
    dt    = 0.001
    T_end = 0.8       # long enough for the saccade + settling
    t     = jnp.arange(0.0, T_end, dt)
    T     = len(t)
    t_jump = 0.1      # target steps here

    # Target amplitudes to test
    amplitudes_deg = np.array([0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20],
                               dtype=np.float32)

    hv3 = jnp.zeros((T, 3))
    vs3 = jnp.zeros((T, 3))

    def _run_one(amp_deg):
        """Simulate one saccade and return (measured_amplitude, peak_velocity)."""
        # Build target array: starts at 0°, jumps to amp_deg at t_jump
        x_tgt = jnp.where(t >= t_jump, jnp.tan(jnp.radians(amp_deg)), 0.0)
        pt3   = jnp.stack([x_tgt, jnp.zeros(T), jnp.ones(T)], axis=1)

        eye = simulate(THETA_SAC, t, hv3, vs3, pt3, max_steps=200000)   # (T, 3)
        eye_yaw = eye[:, 0]   # horizontal component

        # Saccade amplitude: eye position at end relative to start
        amplitude = eye_yaw[-1] - eye_yaw[0]

        # Peak velocity: max |d(eye_yaw)/dt| over the saccade window
        vel = jnp.gradient(eye_yaw, dt)
        peak_vel = jnp.max(jnp.abs(vel))

        return amplitude, peak_vel

    # Run for each amplitude
    amps_measured = []
    peak_vels     = []
    for amp in amplitudes_deg:
        a, pv = _run_one(float(amp))
        amps_measured.append(float(a))
        peak_vels.append(float(pv))

    amps_measured = np.array(amps_measured)
    peak_vels     = np.array(peak_vels)

    # Reference main sequence: empirical fit  v_peak = 700*(1 - exp(-A/7))
    A_ref = np.linspace(0, 21, 200)
    v_ref = 700.0 * (1.0 - np.exp(-A_ref / 7.0))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle('Saccade Main Sequence', fontsize=13)

    # Left: main sequence scatter
    axes[0].plot(A_ref, v_ref, color='gray', lw=1.2, ls='--', label='700(1−e^{−A/7}) ref')
    axes[0].scatter(amps_measured, peak_vels, color=_C['eye'], s=60, zorder=5)
    axes[0].set_xlabel('Amplitude (deg)')
    axes[0].set_ylabel('Peak velocity (deg/s)')
    axes[0].set_title('Peak Velocity vs Amplitude')
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(0, 21); axes[0].set_ylim(0)

    # Right: overlaid eye position traces (normalised time for each saccade)
    ax2 = axes[1]
    cmap = plt.get_cmap('plasma')
    t_np = np.array(t)
    for i, amp in enumerate(amplitudes_deg):
        x_tgt = np.where(t_np >= t_jump, np.tan(np.radians(amp)), 0.0)
        pt3_i = jnp.stack([jnp.array(x_tgt), jnp.zeros(T), jnp.ones(T)], axis=1)
        eye_i = np.array(simulate(THETA_SAC, t, hv3, vs3, pt3_i, max_steps=200000)[:, 0])
        color = cmap(i / (len(amplitudes_deg) - 1))
        ax2.plot(t_np - t_jump, eye_i, color=color, lw=1.2,
                 label=f'{amp:.0f}°' if amp in [1, 5, 10, 20] else None)

    ax2.set_xlabel('Time from target step (s)')
    ax2.set_ylabel('Eye position (deg)')
    ax2.set_title('Eye Traces (all amplitudes)')
    ax2.set_xlim(-0.05, 0.5)
    ax2.axvline(0, color='gray', lw=0.8, ls='--')
    ax2.legend(fontsize=8, loc='upper left')

    sm = plt.cm.ScalarMappable(cmap='plasma',
                                norm=plt.Normalize(amplitudes_deg[0], amplitudes_deg[-1]))
    plt.colorbar(sm, ax=ax2, label='Amplitude (deg)')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'saccade_main_sequence.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Saved {path}')


# ── Demo 5: oblique saccades ───────────────────────────────────────────────────

def demo_oblique_saccades():
    """Oblique saccades — 2-D trajectories and component velocity profiles.

    Tests:
      • Trajectory straightness (yaw vs pitch should be a straight line)
      • Component synchrony (H and V velocities should peak together)
      • Main sequence: oblique amplitudes sit on the same curve as cardinal ones
    """
    dt     = 0.001
    T_end  = 0.7
    t      = jnp.arange(0.0, T_end, dt)
    T      = len(t)
    t_np   = np.array(t)
    t_jump = 0.1

    hv3 = jnp.zeros((T, 3))
    vs3 = jnp.zeros((T, 3))

    # (yaw_deg, pitch_deg) pairs to test
    targets = [
        ( 5.0,  0.0),   # cardinal H
        ( 0.0,  5.0),   # cardinal V
        ( 5.0,  5.0),   # 45° diagonal, small
        (10.0,  5.0),   # asymmetric oblique
        ( 5.0, 10.0),   # asymmetric oblique (flipped)
        (10.0, 10.0),   # 45° diagonal, large
        (15.0,  8.0),   # large oblique
    ]

    def _run(yaw_deg, pitch_deg):
        x_tgt = np.where(t_np >= t_jump, np.tan(np.radians(yaw_deg)),   0.0)
        y_tgt = np.where(t_np >= t_jump, np.tan(np.radians(pitch_deg)), 0.0)
        pt3   = jnp.stack([jnp.array(x_tgt), jnp.array(y_tgt), jnp.ones(T)], axis=1)
        eye   = np.array(simulate(THETA_SAC, t, hv3, vs3, pt3, max_steps=200000))
        return eye   # (T, 3)

    results = [_run(y, p) for y, p in targets]

    # ── Figure layout: 3 rows ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Oblique Saccades', fontsize=13)

    # Row 1 left: 2-D gaze trajectories
    ax_traj = fig.add_subplot(3, 3, 1)
    # Row 1 middle+right: H and V position vs time for each target
    ax_pos_h = fig.add_subplot(3, 3, 2)
    ax_pos_v = fig.add_subplot(3, 3, 3)
    # Row 2: H and V velocity profiles
    ax_vel_h = fig.add_subplot(3, 3, 4)
    ax_vel_v = fig.add_subplot(3, 3, 5)
    # Row 2 right: main sequence (oblique on same axes as cardinal)
    ax_ms    = fig.add_subplot(3, 3, 6)
    # Row 3: curvature index and component timing (peak vel time H vs V)
    ax_curv  = fig.add_subplot(3, 3, 7)
    ax_sync  = fig.add_subplot(3, 3, 8)
    ax_ratio = fig.add_subplot(3, 3, 9)

    cmap   = plt.get_cmap('tab10')
    t_rel  = t_np - t_jump

    for i, ((yaw, pitch), eye) in enumerate(zip(targets, results)):
        amp   = np.sqrt(yaw**2 + pitch**2)
        color = cmap(i / len(targets))
        label = f'({yaw:.0f}°, {pitch:.0f}°)'

        vel_h = np.gradient(eye[:, 0], dt)
        vel_v = np.gradient(eye[:, 1], dt)

        # 2-D trajectory
        ax_traj.plot(eye[:, 0], eye[:, 1], color=color, lw=1.2)
        ax_traj.plot(yaw, pitch, 'x', color=color, ms=6)

        # Position vs time
        ax_pos_h.plot(t_rel, eye[:, 0], color=color, lw=1.2, label=label)
        ax_pos_v.plot(t_rel, eye[:, 1], color=color, lw=1.2)

        # Velocity vs time
        ax_vel_h.plot(t_rel, vel_h, color=color, lw=1.2)
        ax_vel_v.plot(t_rel, vel_v, color=color, lw=1.2)

        # Main sequence: oblique amplitude vs peak resultant velocity
        peak_vel = np.max(np.sqrt(vel_h**2 + vel_v**2))
        ax_ms.scatter(amp, peak_vel, color=color, s=60, zorder=5)

        # Curvature index: max perpendicular deviation / chord length
        # (0 = perfectly straight, positive = curved)
        yaw_f, pitch_f = eye[-1, 0], eye[-1, 1]
        chord = np.sqrt(yaw_f**2 + pitch_f**2)
        if chord > 0.1:
            # perpendicular distances from each point to the chord line
            nx, ny = -pitch_f / chord, yaw_f / chord  # unit normal
            perp   = np.abs(eye[:, 0] * nx + eye[:, 1] * ny)
            curv_idx = np.max(perp) / chord
        else:
            curv_idx = 0.0
        ax_curv.scatter(amp, curv_idx * 100, color=color, s=60)

        # Component synchrony: time of peak H vel vs peak V vel
        win = t_np > t_jump
        if np.abs(yaw) > 0.5:
            t_peak_h = t_rel[win][np.argmax(np.abs(vel_h[win]))]
        else:
            t_peak_h = np.nan
        if np.abs(pitch) > 0.5:
            t_peak_v = t_rel[win][np.argmax(np.abs(vel_v[win]))]
        else:
            t_peak_v = np.nan
        ax_sync.scatter(t_peak_h * 1000, t_peak_v * 1000, color=color, s=60)

        # H/V peak velocity ratio vs H/V amplitude ratio
        if np.abs(pitch) > 0.5 and np.abs(yaw) > 0.5:
            amp_ratio = np.abs(yaw) / np.abs(pitch)
            vel_ratio = np.max(np.abs(vel_h[win])) / np.max(np.abs(vel_v[win]))
            ax_ratio.scatter(amp_ratio, vel_ratio, color=color, s=60, label=label)

    # Reference main sequence
    A_ref = np.linspace(0, 22, 200)
    v_ref = 700.0 * (1.0 - np.exp(-A_ref / 7.0))
    ax_ms.plot(A_ref, v_ref, color='gray', lw=1.0, ls='--', label='ref')

    # Diagonal identity lines / formatting
    ax_traj.set_xlabel('Yaw (deg)'); ax_traj.set_ylabel('Pitch (deg)')
    ax_traj.set_title('2-D Trajectories'); ax_traj.set_aspect('equal')
    ax_traj.axhline(0, color='k', lw=0.4); ax_traj.axvline(0, color='k', lw=0.4)

    ax_pos_h.set_title('Yaw position'); ax_pos_h.set_ylabel('deg')
    ax_pos_v.set_title('Pitch position'); ax_pos_v.set_ylabel('deg')
    ax_vel_h.set_title('Yaw velocity'); ax_vel_h.set_ylabel('deg/s')
    ax_vel_v.set_title('Pitch velocity'); ax_vel_v.set_ylabel('deg/s')
    ax_pos_h.legend(fontsize=6, loc='upper left')

    ax_ms.set_xlabel('Oblique amplitude (deg)'); ax_ms.set_ylabel('Peak resultant vel (deg/s)')
    ax_ms.set_title('Main Sequence (oblique)'); ax_ms.legend(fontsize=7)

    ax_curv.set_xlabel('Oblique amplitude (deg)'); ax_curv.set_ylabel('Curvature index (%)')
    ax_curv.set_title('Trajectory Curvature'); ax_curv.axhline(0, color='k', lw=0.4)

    _d = np.linspace(0, 70, 50)
    ax_sync.plot(_d, _d, color='gray', lw=0.8, ls='--')
    ax_sync.set_xlabel('t peak Yaw vel (ms)'); ax_sync.set_ylabel('t peak Pitch vel (ms)')
    ax_sync.set_title('Component Peak Timing')

    _r = np.linspace(0, 3, 50)
    ax_ratio.plot(_r, _r, color='gray', lw=0.8, ls='--')
    ax_ratio.set_xlabel('Amp ratio H/V'); ax_ratio.set_ylabel('Peak vel ratio H/V')
    ax_ratio.set_title('Velocity Ratio vs Amp Ratio')
    ax_ratio.legend(fontsize=6)

    for ax in [ax_pos_h, ax_pos_v, ax_vel_h, ax_vel_v]:
        ax.set_xlabel('Time from step (s)')
        ax.set_xlim(-0.02, 0.5)
        ax.axvline(0, color='gray', lw=0.6, ls='--')
        ax.axhline(0, color='k', lw=0.4)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'saccade_oblique.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Saved {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    demo_saccade_single()
    demo_saccade_sequence()
    demo_saccade_vor()
    demo_main_sequence()
    demo_oblique_saccades()
    print(f'\nDone. Plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
