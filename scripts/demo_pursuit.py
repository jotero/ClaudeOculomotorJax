"""Smooth pursuit + catch-up saccade demos.

Figures produced
────────────────
    smooth_pursuit.png     — pursuit vs no-pursuit comparison at 4 velocities
    pursuit_cascade.png    — 5-row signal-flow cascade: catch-up saccades during pursuit
    vor_saccade_cascade.png — 4-panel: corrective saccades during head rotation

Usage
-----
    python -X utf8 scripts/demo_pursuit.py
"""

import sys
import os

import jax.numpy as jnp
import numpy as np
import matplotlib
SHOW = '--show' in sys.argv
if not SHOW:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, simulate,
    _IDX_NI, _IDX_SG, _IDX_VIS, _IDX_PURSUIT,
)
from oculomotor.models.sensory_models.sensory_model import C_pos
from oculomotor.analysis import ax_fmt, extract_burst, ni_net

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

_C = {
    'target':    '#d6604d',
    'eye':       '#2166ac',
    'pursuit':   '#1a9850',
    'no_pursuit':'#aaaaaa',
    'ni':        '#4dac26',
    'error':     '#762a83',
    'burst':     '#f4a582',
    'vel':       '#1a9641',
    'reset':     '#e08214',
    'head':      '#555555',
}

THETA = PARAMS_DEFAULT


# ── Utilities ──────────────────────────────────────────────────────────────────

def _make_ramp(t_np, vel_degs, t_jump=0.2):
    """Ramp target: step at t_jump to constant velocity vel_degs."""
    T = len(t_np)
    target_deg = np.where(t_np >= t_jump, vel_degs * (t_np - t_jump), 0.0)
    pt3 = jnp.stack([
        jnp.array(np.tan(np.radians(target_deg))),
        jnp.zeros(T), jnp.ones(T),
    ], axis=1)
    vt3 = jnp.stack([
        jnp.array(np.where(t_np >= t_jump, float(vel_degs), 0.0).astype(np.float32)),
        jnp.zeros(T), jnp.zeros(T),
    ], axis=1)
    return target_deg, pt3, vt3


def _extract(states, theta, t_np):
    """Extract signals from full state trajectory."""
    x_p       = np.array(states.plant[:, :3])          # (T, 3) left eye
    x_ni      = ni_net(states)                          # (T, 3) net NI position
    x_pursuit = np.array(states.brain[:, _IDX_PURSUIT]) # (T, 3)
    x_sg      = np.array(states.brain[:, _IDX_SG])

    x_copy = x_sg[:, :3]
    z_ref  = x_sg[:, 3]
    e_held = x_sg[:, 4:7]
    z_sac  = x_sg[:, 7]
    z_acc  = x_sg[:, 8]

    x_vis         = np.array(states.sensory[:, _IDX_VIS])
    e_pos_delayed = x_vis @ np.array(C_pos).T
    u_burst       = extract_burst(states, theta)         # (T, 3) — fixes missing gate/x_ni args

    dt = float(t_np[1] - t_np[0])
    return dict(
        eye_pos=x_p, x_ni=x_ni, x_pursuit=x_pursuit,
        e_pos_delayed=e_pos_delayed[:, 0],
        e_held=e_held[:, 0], x_copy=x_copy[:, 0],
        z_ref=z_ref, z_sac=z_sac, z_acc=z_acc,
        u_burst_h=u_burst[:, 0], eye_vel_h=np.gradient(x_p[:, 0], dt),
    )


# ── Figure 1: smooth_pursuit.png ──────────────────────────────────────────────

def demo_smooth_pursuit():
    """Pursuit vs no-pursuit comparison — 3 rows × 4 velocities."""
    velocities = [5.0, 10.0, 20.0, 40.0]
    T_end  = 3.0
    T_jump = 0.2
    dt     = 0.001
    t      = jnp.arange(0.0, T_end, dt)
    T      = len(t)
    t_np   = np.array(t)

    theta_pur  = THETA                                                  # pursuit on (default)
    theta_nop  = with_brain(THETA, K_pursuit=0.0, K_phasic_pursuit=0.0)  # pursuit off

    n_rows, n_cols = 3, len(velocities)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 2.4 * n_rows),
                             sharex=True)
    fig.suptitle('Smooth pursuit + catch-up saccades  (blue=pursuit on, gray=saccades only)',
                 fontsize=10)

    row_labels = [
        'Position (deg)',
        'Eye velocity (deg/s)',
        'Pursuit drive  u_pursuit (deg/s)',
    ]
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=7.5)

    max_s = int(T_end / dt) + 500

    for ci, vel in enumerate(velocities):
        target_deg, pt3, vt3 = _make_ramp(t_np, vel, T_jump)

        states_pur = simulate(theta_pur, t,
                              p_target_array=pt3, v_target_array=vt3,
                              scene_present_array=jnp.ones(T),
                              target_present_array=jnp.ones(T),
                              max_steps=max_s, return_states=True)
        states_nop = simulate(theta_nop, t,
                              p_target_array=pt3,
                              scene_present_array=jnp.ones(T),
                              target_present_array=jnp.ones(T),
                              max_steps=max_s, return_states=True)

        s_pur = _extract(states_pur, theta_pur, t_np)
        s_nop = _extract(states_nop, theta_nop, t_np)
        u_pursuit = s_pur['x_pursuit'][:, 0]

        axes[0, ci].set_title(f'{vel:.0f} deg/s', fontsize=10)

        # Row 0: position
        axes[0, ci].plot(t_np, target_deg,              color=_C['target'],    lw=1.5, label='target')
        axes[0, ci].plot(t_np, s_nop['eye_pos'][:, 0],  color=_C['no_pursuit'], lw=1.0, ls='--', label='no pursuit')
        axes[0, ci].plot(t_np, s_pur['eye_pos'][:, 0],  color=_C['eye'],       lw=1.5, label='pursuit')
        axes[0, ci].axvline(T_jump, color='gray', lw=0.5, ls='--', alpha=0.4)
        ax_fmt(axes[0, ci])
        if ci == 0:
            axes[0, ci].legend(fontsize=6.5)

        # Row 1: eye velocity vs target velocity
        axes[1, ci].axhline(vel, color=_C['target'], lw=0.8, ls=':', label=f'target {vel} deg/s')
        axes[1, ci].plot(t_np, s_nop['eye_vel_h'],   color=_C['no_pursuit'], lw=0.8, ls='--', label='no pursuit')
        axes[1, ci].plot(t_np, s_pur['eye_vel_h'],   color=_C['eye'],       lw=1.2, label='pursuit')
        axes[1, ci].axvline(T_jump, color='gray', lw=0.5, ls='--', alpha=0.4)
        yabs = max(abs(vel) * 1.3, 10.0)
        axes[1, ci].set_ylim(-yabs * 0.15, yabs)
        ax_fmt(axes[1, ci])
        if ci == 0:
            axes[1, ci].legend(fontsize=6.5)

        # Row 2: pursuit velocity command
        axes[2, ci].axhline(vel, color=_C['target'], lw=0.8, ls=':', label=f'target {vel} deg/s')
        axes[2, ci].plot(t_np, u_pursuit, color=_C['pursuit'], lw=1.5, label='u_pursuit (memory)')
        axes[2, ci].axvline(T_jump, color='gray', lw=0.5, ls='--', alpha=0.4)
        axes[2, ci].set_ylim(-vel * 0.1, vel * 1.3)
        ax_fmt(axes[2, ci])
        if ci == 0:
            axes[2, ci].legend(fontsize=6.5)
        axes[2, ci].set_xlabel('Time (s)')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'smooth_pursuit.png')
    fig.savefig(path, dpi=150)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    print(f'  Saved {path}')


# ── Figure 2: pursuit_cascade.png ─────────────────────────────────────────────

def demo_pursuit_cascade():
    """6 rows × 4 ramp velocities — smooth pursuit + residual catch-up saccades."""
    velocities = [5.0, 10.0, 20.0, 40.0]
    T_end  = 3.0
    T_jump = 0.2
    dt     = 0.001
    t      = jnp.arange(0.0, T_end, dt)
    T      = len(t)
    t_np   = np.array(t)

    n_rows, n_cols = 6, len(velocities)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 2.2 * n_rows),
                             sharex=True)
    fig.suptitle('Pursuit + catch-up saccades — signal-flow cascade',
                 fontsize=10)

    row_labels = [
        'Position (deg)',
        'Error & hold (deg)',
        'Trigger: z_acc / z_sac',
        'Ballistic residual (deg)',
        'Burst + eye vel (deg/s)',
        'Refractory z_ref',
    ]
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=7.5)

    for ci, vel in enumerate(velocities):
        target_deg, pt3, vt3 = _make_ramp(t_np, vel, T_jump)
        max_s = int(T_end / dt) + 500

        states = simulate(THETA, t,
                          p_target_array=pt3, v_target_array=vt3,
                          scene_present_array=jnp.ones(T),
                          target_present_array=jnp.ones(T),
                          max_steps=max_s, return_states=True)
        s = _extract(states, THETA, t_np)

        def _vl(ax):
            ax.axvline(T_jump, color='gray', lw=0.6, ls='--', alpha=0.4)

        axes[0, ci].set_title(f'{vel:.0f} deg/s', fontsize=10)

        axes[0, ci].plot(t_np, target_deg,          color=_C['target'],  lw=1.5, label='target')
        axes[0, ci].plot(t_np, s['eye_pos'][:, 0],  color=_C['eye'],     lw=1.5, label='eye')
        axes[0, ci].plot(t_np, s['x_ni'][:, 0],     color=_C['ni'],      lw=0.8, ls='--', label='NI')
        _vl(axes[0, ci]); ax_fmt(axes[0, ci])
        if ci == 0: axes[0, ci].legend(fontsize=6)

        axes[1, ci].plot(t_np, s['e_pos_delayed'], color=_C['error'],  lw=1.0, ls='--', label='e_delayed')
        axes[1, ci].plot(t_np, s['e_held'],        color=_C['reset'],  lw=1.8, label='e_held')
        _vl(axes[1, ci]); ax_fmt(axes[1, ci])
        if ci == 0: axes[1, ci].legend(fontsize=6)

        axes[2, ci].plot(t_np, s['z_acc'], color='#e08214', lw=1.5, label='z_acc')
        axes[2, ci].plot(t_np, s['z_sac'], color='#1b7837', lw=1.5, label='z_sac')
        axes[2, ci].axhline(THETA.brain.threshold_acc, color='#e08214', lw=0.7, ls=':')
        axes[2, ci].set_ylim(-0.05, 1.15)
        _vl(axes[2, ci]); axes[2, ci].grid(True, alpha=0.2)
        if ci == 0: axes[2, ci].legend(fontsize=6)

        axes[3, ci].plot(t_np, s['e_res'] if 'e_res' in s else s['e_held'] - s['x_copy'],
                         color=_C['error'], lw=1.5, label='e_res')
        axes[3, ci].plot(t_np, s['x_copy'], color=_C['reset'], lw=1.2, ls='--', label='x_copy')
        _vl(axes[3, ci]); ax_fmt(axes[3, ci])
        if ci == 0: axes[3, ci].legend(fontsize=6)

        axes[4, ci].plot(t_np, s['u_burst_h'], color=_C['burst'], lw=1.5, label='u_burst')
        axes[4, ci].plot(t_np, s['eye_vel_h'], color=_C['vel'],   lw=1.2, ls='--', label='eye vel')
        axes[4, ci].axhline(vel, color=_C['target'], lw=0.8, ls=':', label=f'{vel} deg/s')
        _vl(axes[4, ci]); ax_fmt(axes[4, ci])
        yabs = max(np.nanmax(np.abs(s['u_burst_h'])), np.nanmax(np.abs(s['eye_vel_h'])), vel, 1.0)
        axes[4, ci].set_ylim(-yabs * 0.1, yabs * 1.1)
        if ci == 0: axes[4, ci].legend(fontsize=6)

        axes[5, ci].plot(t_np, s['z_ref'], color='#762a83', lw=1.5, label='z_ref')
        axes[5, ci].axhline(THETA.brain.threshold_sac_release, color='#762a83', lw=0.7, ls=':')
        axes[5, ci].set_ylim(-0.05, 1.15)
        _vl(axes[5, ci]); axes[5, ci].grid(True, alpha=0.2)
        axes[5, ci].set_xlabel('Time (s)')
        if ci == 0: axes[5, ci].legend(fontsize=6)

    # fix missing e_res in _extract
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'pursuit_cascade.png')
    fig.savefig(path, dpi=150)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    print(f'  Saved {path}')


# ── Figure 3: vor_saccade_cascade.png ─────────────────────────────────────────

def demo_vor_saccade():
    """Corrective saccades during constant head rotation — 4-panel cascade."""
    dt    = 0.001
    T_end = 3.0
    t     = jnp.arange(0.0, T_end, dt)
    T     = len(t)
    t_np  = np.array(t)

    head_vel_np = np.where(t_np < 1.5, 30.0, 0.0).astype(np.float32)
    hv = jnp.stack([jnp.array(head_vel_np), jnp.zeros(T), jnp.zeros(T)], axis=1)

    theta_sac    = with_brain(PARAMS_DEFAULT,
                              K_vis=0.0, g_vis=0.0,    # dark
                              K_pursuit=0.0, K_phasic_pursuit=0.0,  # dark: no pursuit
                              g_burst=40.0,
                              threshold_sac=3.0,
                              k_sac=10.0)
    theta_no_sac = with_brain(theta_sac, g_burst=0.0)

    max_s = int(T_end / dt) + 500

    states_sac = simulate(theta_sac,    t, head_vel_array=hv,
                          target_present_array=jnp.ones(T),
                          max_steps=max_s, return_states=True)
    states_ns  = simulate(theta_no_sac, t, head_vel_array=hv,
                          target_present_array=jnp.ones(T),
                          max_steps=max_s, return_states=True)

    s    = _extract(states_sac, theta_sac, t_np)
    s_ns = _extract(states_ns,  theta_no_sac, t_np)

    head_pos = np.cumsum(head_vel_np) * dt

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    fig.suptitle('VOR + Corrective Saccades  (head 30°/s for 1.5 s, target straight-ahead)',
                 fontsize=11)

    vline_kw = dict(color='k', lw=0.8, ls='--', alpha=0.4)
    for ax in axes:
        ax.axvline(1.5, **vline_kw)
        ax_fmt(ax)

    axes[0].plot(t_np, head_pos,              color=_C['head'],       lw=1.0, ls=':', label='head pos')
    axes[0].plot(t_np, s_ns['eye_pos'][:, 0], color=_C['no_pursuit'], lw=1.2, ls='--', label='VOR only')
    axes[0].plot(t_np, s['eye_pos'][:, 0],    color=_C['eye'],        lw=1.5, label='VOR + saccades')
    axes[0].axhline(0, color=_C['target'], lw=1.0, ls=':', label='target (0°)')
    axes[0].set_ylabel('Eye pos (deg)'); axes[0].set_title('Eye Position')
    axes[0].legend(fontsize=8)

    axes[1].plot(t_np, s['u_burst_h'], color=_C['burst'], lw=1.5)
    axes[1].set_ylabel('Burst (deg/s)'); axes[1].set_title('Saccade Burst  u_burst')

    axes[2].plot(t_np, s['z_ref'], color='purple', lw=1.5)
    axes[2].set_ylim(-0.05, 1.15)
    axes[2].set_ylabel('z_ref'); axes[2].set_title('Refractory state  (1 = blocked)')

    e_motor = np.degrees(np.arctan2(np.zeros(T), np.ones(T))) - s['eye_pos'][:, 0] - head_pos
    axes[3].plot(t_np, e_motor, color=_C['error'], lw=1.5, label='gaze error')
    axes[3].axhline( theta_sac.brain.threshold_sac, color='gray', lw=0.8, ls=':', label='±threshold')
    axes[3].axhline(-theta_sac.brain.threshold_sac, color='gray', lw=0.8, ls=':')
    axes[3].set_ylabel('Gaze error (deg)'); axes[3].set_title('Gaze Error  (target − head − eye)')
    axes[3].set_xlabel('Time (s)'); axes[3].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'vor_saccade_cascade.png')
    fig.savefig(path, dpi=150)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    print(f'  Saved {path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=== Pursuit Demo ===')

    print('\n1. Smooth pursuit comparison (with vs without)')
    demo_smooth_pursuit()

    print('\n2. Pursuit + catch-up saccade cascade')
    demo_pursuit_cascade()

    print('\n3. VOR + corrective saccades')
    demo_vor_saccade()

    print(f'\nDone. Plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
