"""Smooth pursuit benchmarks — velocity range, sinusoidal, signal cascade.

Usage:
    python -X utf8 scripts/bench_pursuit.py
    python -X utf8 scripts/bench_pursuit.py --show
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as utils

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, with_sensory, simulate,
    _IDX_SG, _IDX_VIS, _IDX_PURSUIT,
)
from oculomotor.models.sensory_models.sensory_model import C_pos
from oculomotor.analysis import ax_fmt, extract_burst, extract_sg, ni_net

SHOW  = '--show' in sys.argv
DT    = 0.001
THETA = with_sensory(PARAMS_DEFAULT, sigma_canal=0.5, sigma_pos=0.2, sigma_vel=0.2)


def _ramp(t_np, vel, t_jump=0.2):
    T = len(t_np)
    tgt = np.where(t_np >= t_jump, vel * (t_np - t_jump), 0.0)
    pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
    pt3[:, 0] = np.tan(np.radians(tgt))
    vt3 = np.zeros((T, 3))
    vt3[:, 0] = np.where(t_np >= t_jump, float(vel), 0.0).astype(np.float32)
    return tgt, jnp.array(pt3), jnp.array(vt3)


def _run(theta, t_np, pt3, vt3=None, target_present=True, key=0):
    t = jnp.array(t_np)
    T = len(t)
    tp = jnp.ones(T) if target_present else jnp.zeros(T)
    return simulate(theta, t, p_target_array=pt3,
                    v_target_array=vt3 if vt3 is not None else jnp.zeros((T, 3)),
                    scene_present_array=jnp.ones(T), target_present_array=tp,
                    max_steps=int(len(t_np) * 1.05) + 500,
                    return_states=True, key=jax.random.PRNGKey(key))


# ── Figure 1: velocity range comparison ──────────────────────────────────────

def _velocity_range(show):
    velocities = [5.0, 10.0, 20.0, 40.0]
    T_end, T_jump = 3.0, 0.2
    t_np = np.arange(0.0, T_end, DT)

    theta_pur = THETA
    theta_nop = with_brain(THETA, K_pursuit=0.0, K_phasic_pursuit=0.0)

    n_rows, n_cols = 3, len(velocities)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 2.5 * n_rows), sharex=True)
    fig.suptitle('Smooth Pursuit — Velocity Range  (blue: pursuit on, gray: saccades only)', fontsize=11)
    row_labels = ['Position (deg)', 'Eye velocity (deg/s)', 'Pursuit drive u_pursuit (deg/s)']
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8)

    for ci, vel in enumerate(velocities):
        tgt, pt3, vt3 = _ramp(t_np, vel, T_jump)
        st_pur = _run(theta_pur, t_np, pt3, vt3, key=ci)
        st_nop = _run(theta_nop, t_np, pt3,      key=ci + 10)

        eye_pur = np.array(st_pur.plant[:, 0])
        eye_nop = np.array(st_nop.plant[:, 0])
        ev_pur  = np.gradient(eye_pur, DT)
        ev_nop  = np.gradient(eye_nop, DT)
        u_pur   = np.array(st_pur.brain[:, _IDX_PURSUIT])[:, 0]

        axes[0, ci].set_title(f'{vel:.0f} deg/s', fontsize=10)
        for ax in axes[:, ci]:
            ax.axvline(T_jump, color='gray', lw=0.5, ls='--', alpha=0.4)

        axes[0, ci].plot(t_np, tgt,     color=utils.C['target'],    lw=1.5, label='target')
        axes[0, ci].plot(t_np, eye_nop, color=utils.C['dark'],      lw=1.0, ls='--', label='no pursuit')
        axes[0, ci].plot(t_np, eye_pur, color=utils.C['eye'],       lw=1.5, label='pursuit')
        ax_fmt(axes[0, ci])
        if ci == 0: axes[0, ci].legend(fontsize=7)

        axes[1, ci].axhline(vel, color=utils.C['target'], lw=0.8, ls=':', alpha=0.7,
                            label=f'{vel} deg/s')
        axes[1, ci].plot(t_np, ev_nop, color=utils.C['dark'], lw=0.8, ls='--', label='no pursuit')
        axes[1, ci].plot(t_np, ev_pur, color=utils.C['eye'],  lw=1.2, label='pursuit')
        axes[1, ci].set_ylim(-max(vel*0.15, 3), vel * 1.35)
        ax_fmt(axes[1, ci])
        if ci == 0: axes[1, ci].legend(fontsize=7)

        axes[2, ci].axhline(vel, color=utils.C['target'], lw=0.8, ls=':', alpha=0.7)
        axes[2, ci].plot(t_np, u_pur, color=utils.C['pursuit'], lw=1.5, label='u_pursuit')
        axes[2, ci].set_ylim(-vel * 0.1, vel * 1.35)
        ax_fmt(axes[2, ci])
        axes[2, ci].set_xlabel('Time (s)', fontsize=8)
        if ci == 0: axes[2, ci].legend(fontsize=7)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'pursuit_velocity_range', show=show)
    return utils.fig_meta(path, rp,
        title='Smooth Pursuit — Velocity Range',
        description='Step-ramp target at 5, 10, 20, 40 deg/s. '
                    'Blue: smooth pursuit enabled. Gray: saccades only (pursuit off). '
                    'Rows: position, velocity, pursuit integrator state.',
        expected='At 5–10 deg/s: pursuit tracks closely, steady-state gain > 0.8. '
                 'At higher velocities: catch-up saccades + partial pursuit.',
        citation='Lisberger & Westbrook (1985) J Neurosci; Rashbass (1961)',
        fig_type='behavior')


# ── Figure 2: sinusoidal pursuit ─────────────────────────────────────────────

def _sinusoidal(show):
    """Sinusoidal target: horizontal + vertical, 3 frequencies."""
    freqs = [0.2, 0.5, 1.0]   # Hz
    AMP   = 15.0               # deg/s peak velocity
    T_END = 12.0
    t_np  = np.arange(0.0, T_END, DT)
    T     = len(t_np)

    # 2D: H sinusoid at freq f, V sinusoid at 2*freq (Lissajous-like)
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    fig.suptitle(f'Sinusoidal Pursuit — H: A·sin(2πft), V: A·sin(4πft),  A = {AMP:.0f} deg/s peak',
                 fontsize=11)
    row_labels = ['H position (deg)', 'V position (deg)', 'H+V trajectory (deg)']
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8)

    for ci, freq in enumerate(freqs):
        axes[0, ci].set_title(f'{freq} Hz', fontsize=11)
        # Position = integral of velocity
        tgt_h = -(AMP / (2 * np.pi * freq))     * np.cos(2 * np.pi * freq       * t_np)
        tgt_v = -(AMP / (2 * np.pi * 2 * freq)) * np.cos(2 * np.pi * 2 * freq * t_np)
        vel_h = AMP * np.sin(2 * np.pi * freq       * t_np)
        vel_v = AMP * np.sin(2 * np.pi * 2 * freq * t_np)

        pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
        pt3[:, 0] = np.tan(np.radians(tgt_h))
        pt3[:, 1] = np.tan(np.radians(tgt_v))
        vt3 = np.zeros((T, 3))
        vt3[:, 0] = vel_h.astype(np.float32)
        vt3[:, 1] = vel_v.astype(np.float32)

        st   = _run(THETA, t_np, jnp.array(pt3), jnp.array(vt3), key=ci + 20)
        eye  = np.array(st.plant[:, :2])

        mask = t_np > 2.0  # skip warm-up
        axes[0, ci].plot(t_np[mask], tgt_h[mask], color=utils.C['target'], lw=1.2, label='target H')
        axes[0, ci].plot(t_np[mask], eye[mask, 0], color=utils.C['eye'],   lw=1.5, label='eye H')
        ax_fmt(axes[0, ci]); axes[0, ci].legend(fontsize=7)

        axes[1, ci].plot(t_np[mask], tgt_v[mask], color=utils.C['target'], lw=1.2, ls='--', label='target V')
        axes[1, ci].plot(t_np[mask], eye[mask, 1], color=utils.C['eye'],   lw=1.5, ls='--', label='eye V')
        ax_fmt(axes[1, ci]); axes[1, ci].legend(fontsize=7)

        axes[2, ci].plot(tgt_h[mask], tgt_v[mask], color=utils.C['target'], lw=1.0, label='target')
        axes[2, ci].plot(eye[mask, 0], eye[mask, 1], color=utils.C['eye'],  lw=1.5, label='eye')
        axes[2, ci].set_xlabel('H (deg)', fontsize=8)
        axes[2, ci].set_aspect('equal'); axes[2, ci].legend(fontsize=7)
        axes[2, ci].grid(True, alpha=0.25)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'pursuit_sinusoidal', show=show)
    return utils.fig_meta(path, rp,
        title='Sinusoidal Pursuit (H + V)',
        description='Horizontal sinusoidal target at 0.2, 0.5, 1.0 Hz (peak 15 deg/s). '
                    'Vertical at 2× horizontal frequency for Lissajous trajectory. '
                    'Rows: H position, V position, 2D trajectory.',
        expected='Low freq (0.2 Hz): near-unity gain, small phase lag. '
                 'High freq (1.0 Hz): gain < 1, larger lag, catch-up saccades.',
        citation='Lisberger et al. (1981) J Neurophysiol',
        fig_type='behavior')


# ── Figure 3: pursuit signal cascade ─────────────────────────────────────────

def _cascade(show):
    """Signal cascade for 20 deg/s pursuit with catch-up saccades."""
    vel   = 20.0
    T_end = 3.0
    t_np  = np.arange(0.0, T_end, DT)
    tgt, pt3, vt3 = _ramp(t_np, vel, t_jump=0.2)

    st  = _run(THETA, t_np, pt3, vt3, key=30)
    sg  = extract_sg(st, THETA)
    eye = np.array(st.plant[:, 0])
    ev  = np.gradient(eye, DT)
    x_pur = np.array(st.brain[:, _IDX_PURSUIT])[:, 0]

    x_vis = np.array(st.sensory[:, _IDX_VIS])
    e_vel_del = (x_vis @ np.array(C_pos).T)[:, 0]   # reuse pos channel as velocity proxy

    n_rows = 7
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.5 * n_rows))
    fig.suptitle(f'Smooth Pursuit Signal Cascade — {vel:.0f} deg/s ramp\n'
                 f'(velocity-driven pursuit integrator + catch-up saccades)', fontsize=11)

    vl = dict(color='gray', lw=0.7, ls='--', alpha=0.5)
    for ax in axes:
        ax.axvline(0.2, **vl); ax_fmt(ax)

    axes[0].plot(t_np, tgt,          color=utils.C['target'],  lw=1.5, label='target pos')
    axes[0].plot(t_np, eye,          color=utils.C['eye'],     lw=1.5, label='eye pos')
    axes[0].plot(t_np, ni_net(st)[:,0], color=utils.C['ni'],   lw=0.9, ls='--', label='NI')
    axes[0].set_ylabel('Position (deg)'); axes[0].set_title('Eye + Target Position')
    axes[0].legend(fontsize=8)

    axes[1].axhline(vel, color=utils.C['target'], lw=0.8, ls=':', alpha=0.7)
    axes[1].plot(t_np, ev, color=utils.C['eye'], lw=1.2, label='eye vel')
    axes[1].set_ylabel('Velocity (deg/s)'); axes[1].set_title('Eye Velocity')
    axes[1].legend(fontsize=8)

    axes[2].plot(t_np, x_pur, color=utils.C['pursuit'], lw=1.5, label='u_pursuit (integrator memory)')
    axes[2].axhline(vel, color=utils.C['target'], lw=0.8, ls=':', alpha=0.7)
    axes[2].set_ylabel('u_pursuit (deg/s)'); axes[2].set_title('Pursuit Velocity Integrator State')
    axes[2].legend(fontsize=8)

    axes[3].plot(t_np, sg['e_pd'][:,0],   color='darkorange', lw=1.0, ls='--', label='e_delayed')
    axes[3].plot(t_np, sg['e_held'][:,0], color=utils.C['vs'], lw=1.8, label='e_held (frozen)')
    axes[3].set_ylabel('Error (deg)'); axes[3].set_title('Visual Cascade Output + Sample-Hold')
    axes[3].legend(fontsize=8)

    axes[4].plot(t_np, sg['z_acc'], color='#e08214', lw=1.5, label='z_acc')
    axes[4].plot(t_np, sg['z_sac'], color='#1b7837', lw=1.5, label='z_sac (catch-up saccade latch)')
    axes[4].axhline(THETA.brain.threshold_acc, color='#e08214', lw=0.8, ls=':')
    axes[4].set_ylim(-0.05, 1.15)
    axes[4].set_ylabel('Accumulator'); axes[4].set_title('Catch-up Saccade Trigger')
    axes[4].legend(fontsize=8)

    axes[5].plot(t_np, sg['u_burst'][:,0], color=utils.C['burst'], lw=1.5, label='burst (catch-up)')
    axes[5].set_ylabel('Burst (deg/s)'); axes[5].set_title('Saccade Burst (Catch-up Saccades)')
    axes[5].legend(fontsize=8)

    axes[6].plot(t_np, sg['z_ref'], color=utils.C['refractory'], lw=1.5, label='z_ref (refractory)')
    axes[6].set_ylim(-0.05, 1.15)
    axes[6].set_ylabel('z_ref'); axes[6].set_title('Refractory State')
    axes[6].set_xlabel('Time (s)', fontsize=9)
    axes[6].legend(fontsize=8)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'pursuit_cascade', show=show)
    return utils.fig_meta(path, rp,
        title='Smooth Pursuit Signal Cascade (Internal)',
        description='Full signal chain for 20 deg/s pursuit: position, velocity, pursuit integrator state, '
                    'visual cascade error, accumulator/latch for catch-up saccades, burst, refractory state.',
        expected='Pursuit integrator charges to target velocity; catch-up saccades fire when positional '
                 'error exceeds threshold; saccade refractory period visible between bursts.',
        citation='Lisberger & Westbrook (1985)',
        fig_type='cascade')


# ── Section entry point ────────────────────────────────────────────────────────

SECTION = dict(
    id='pursuit', title='4. Smooth Pursuit',
    description='Smooth pursuit and catch-up saccades. Tests velocity gain at multiple speeds, '
                'sinusoidal tracking, and the pursuit integrator + saccade interaction cascade.',
)


def run(show=False):
    print('\n=== Smooth Pursuit ===')
    figs = []
    print('  1/3  velocity range …')
    figs.append(_velocity_range(show))
    print('  2/3  sinusoidal pursuit …')
    figs.append(_sinusoidal(show))
    print('  3/3  signal cascade …')
    figs.append(_cascade(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
