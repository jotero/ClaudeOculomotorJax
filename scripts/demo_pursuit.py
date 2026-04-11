"""Pursuit / ramp-target and VOR+saccade demos.

No smooth pursuit pathway exists yet — catch-up saccades are the only mechanism.
These demos show how saccades behave with continuously moving targets.

Figures produced
────────────────
    pursuit_cascade.png    — 5 rows × 4 velocities: ramp catch-up saccades
    vor_saccade_cascade.png — 4-panel: corrective saccades during head rotation

Usage
-----
    python -X utf8 scripts/demo_pursuit.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
SHOW = '--show' in sys.argv
if not SHOW:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.models.ocular_motor_simulator import (
    THETA_DEFAULT, simulate,
    _IDX_NI, _IDX_P, _IDX_SG, _IDX_VIS,
)
from oculomotor.models import saccade_generator as sg_mod
from oculomotor.models import visual_delay

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

_C = {
    'target': '#d6604d',
    'eye':    '#2166ac',
    'ni':     '#4dac26',
    'error':  '#762a83',
    'burst':  '#f4a582',
    'vel':    '#1a9641',
    'reset':  '#e08214',
    'head':   '#555555',
    'no_sac': '#aaaaaa',
}

THETA_SAC = THETA_DEFAULT


# ── Utilities ──────────────────────────────────────────────────────────────────

def _extract(states, theta, t_np):
    """Extract saccade-relevant signals from full state trajectory."""
    x_p   = np.array(states[:, _IDX_P])
    x_ni  = np.array(states[:, _IDX_NI])
    x_vis = np.array(states[:, _IDX_VIS])
    x_sg  = np.array(states[:, _IDX_SG])

    # SG state layout: [x_copy(3) | z_ref(1) | e_held(3) | z_sac(1) | z_acc(1)]
    x_copy = x_sg[:, :3]
    z_ref  = x_sg[:, 3]
    e_held = x_sg[:, 4:7]
    z_sac  = x_sg[:, 7]
    z_acc  = x_sg[:, 8]
    e_res  = e_held - x_copy

    e_pos_delayed = (x_vis @ np.array(visual_delay.C_pos).T)

    ys_j = jnp.array(states)
    def _burst_at(x):
        e_pd = visual_delay.C_pos @ x[_IDX_VIS]
        _, u = sg_mod.step(x[_IDX_SG], e_pd, theta)
        return u
    u_burst = np.array(jax.vmap(_burst_at)(ys_j))

    dt = float(t_np[1] - t_np[0])
    return dict(
        eye_pos=x_p, x_ni=x_ni,
        e_pos_delayed=e_pos_delayed[:, 0],
        e_held=e_held[:, 0],
        x_copy=x_copy[:, 0],
        e_res=e_res[:, 0],
        z_ref=z_ref, z_sac=z_sac, z_acc=z_acc,
        u_burst_h=u_burst[:, 0],
        eye_vel_h=np.gradient(x_p[:, 0], dt),
    )


def _ax_fmt(ax):
    ax.axhline(0, color='k', lw=0.4)
    ax.grid(True, alpha=0.2)


# ── Figure 1: pursuit_cascade.png ─────────────────────────────────────────────
#
# Same 6-row signal-flow layout as saccade_cascade.png.
# Signal flow (top → bottom):
#   Ramp target                               [row 0: position]
#     → 40-stage visual cascade (120 ms)      [row 1: e_pos_delayed rising per saccade]
#     → gate_err fires → z_acc accumulates    [row 2: trigger pathway]
#     → z_acc > 0.5 → z_sac latches          [row 2: z_sac fires]
#     → e_held freezes at settled cascade val [row 1: e_held flat during saccade]
#     → e_res = e_held − x_copy drives burst  [row 3: residual closes]
#     → u_burst → NI + plant → eye velocity  [row 4: burst & eye vel]
#     → x_copy integrates toward e_held       [row 3: x_copy rising]
#     → e_res → 0 → z_ref charges            [row 5: refractory]
#     → z_ref > 0.4 → z_sac releases         [row 2: z_sac drops]
#     → z_ref decays → next saccade           [row 5: z_ref decaying]

def demo_pursuit_cascade():
    """6 rows × 4 ramp velocities — catch-up saccades to moving target."""
    velocities = [1.0, 10.0, 20.0, 60.0]   # deg/s
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
    fig.suptitle('Ramp Target — Catch-up Saccades  ·  flow: cascade → accumulate → latch/freeze → burst → copy → refractory',
                 fontsize=10)

    row_labels = [
        'Position (deg)',
        'Error & hold (deg)',
        'Trigger: z_acc / z_sac',
        'Ballistic residual (deg)',
        'Burst & eye vel (deg/s)',
        'Refractory z_ref',
    ]
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=7.5)

    for ci, vel in enumerate(velocities):
        target_deg = np.where(t_np >= T_jump, vel * (t_np - T_jump), 0.0)
        pt3 = jnp.stack([
            jnp.array(np.tan(np.radians(target_deg))),
            jnp.zeros(T), jnp.ones(T),
        ], axis=1)
        max_s = int(T_end / dt) + 500

        states = simulate(THETA_SAC, t, p_target_array=pt3,
                          scene_present_array=jnp.ones(T),
                          target_present_array=jnp.ones(T),   # target visible
                          max_steps=max_s, return_states=True)
        s = _extract(states, THETA_SAC, t_np)

        def _vl(ax):
            ax.axvline(T_jump, color='gray', lw=0.6, ls='--', alpha=0.4)

        axes[0, ci].set_title(f'{vel:.0f} deg/s', fontsize=10)

        # ── Row 0: position ────────────────────────────────────────────────────
        axes[0, ci].plot(t_np, target_deg,          color=_C['target'], lw=1.5, label='target pos')
        axes[0, ci].plot(t_np, s['eye_pos'][:, 0],  color=_C['eye'],    lw=1.5, label='eye pos')
        axes[0, ci].plot(t_np, s['x_ni'][:, 0],     color=_C['ni'],     lw=0.9, ls='--', label='NI (eye hold signal)')
        _vl(axes[0, ci]); _ax_fmt(axes[0, ci])
        if ci == 0: axes[0, ci].legend(fontsize=6.5)

        # ── Row 1: cascade output and sample-and-hold ─────────────────────────
        axes[1, ci].plot(t_np, s['e_pos_delayed'], color=_C['error'], lw=1.0, ls='--',
                         label='e_delayed  (cascade output, settling ~120 ms)')
        axes[1, ci].plot(t_np, s['e_held'],        color=_C['reset'], lw=1.8,
                         label='e_held  (tracks cascade; freezes when z_sac=1)')
        _vl(axes[1, ci]); _ax_fmt(axes[1, ci])
        if ci == 0: axes[1, ci].legend(fontsize=6.5)

        # ── Row 2: trigger pathway ─────────────────────────────────────────────
        ax2 = axes[2, ci]
        ax2.plot(t_np, s['z_acc'], color='#e08214', lw=1.5,
                 label='z_acc  (rise-to-bound accumulator)')
        ax2.plot(t_np, s['z_sac'], color='#1b7837', lw=1.5,
                 label='z_sac  (burst latch: 1=active)')
        ax2.axhline(THETA_SAC.get('threshold_acc', 0.5), color='#e08214',
                    lw=0.8, ls=':', alpha=0.7,
                    label=f'threshold_acc={THETA_SAC.get("threshold_acc", 0.5):.1f}  (z_sac fires here)')
        ax2.set_ylim(-0.05, 1.15)
        _vl(ax2); ax2.grid(True, alpha=0.2)
        if ci == 0: ax2.legend(fontsize=6.5)

        # ── Row 3: ballistic residual ──────────────────────────────────────────
        axes[3, ci].plot(t_np, s['e_res'],  color=_C['error'], lw=1.5,
                         label='e_res = e_held − x_copy  (drives burst; closes to 0)')
        axes[3, ci].plot(t_np, s['x_copy'], color=_C['reset'],  lw=1.2, ls='--',
                         label='x_copy  (internal copy; resets to 0 between saccades)')
        _vl(axes[3, ci]); _ax_fmt(axes[3, ci])
        if ci == 0: axes[3, ci].legend(fontsize=6.5)

        # ── Row 4: burst command and eye velocity ──────────────────────────────
        axes[4, ci].plot(t_np, s['u_burst_h'], color=_C['burst'], lw=1.5,
                         label='u_burst  (velocity command to NI + plant)')
        axes[4, ci].plot(t_np, s['eye_vel_h'], color=_C['vel'],   lw=1.2, ls='--',
                         label='eye velocity  (plant response)')
        axes[4, ci].axhline(vel, color=_C['target'], lw=0.8, ls=':',
                            label=f'target velocity = {vel} deg/s')
        _vl(axes[4, ci]); _ax_fmt(axes[4, ci])
        yabs = max(np.max(np.abs(s['u_burst_h'])), np.max(np.abs(s['eye_vel_h'])), 1.0)
        axes[4, ci].set_ylim(-yabs * 0.1, yabs * 1.1)
        if ci == 0: axes[4, ci].legend(fontsize=6.5)

        # ── Row 5: refractory state ────────────────────────────────────────────
        ax5 = axes[5, ci]
        ax5.plot(t_np, s['z_ref'], color='#762a83', lw=1.5,
                 label='z_ref  (OPN refractory state)')
        ax5.axhline(THETA_SAC.get('threshold_sac_release', 0.4), color='#762a83',
                    lw=0.8, ls=':', alpha=0.8,
                    label=f'release threshold={THETA_SAC.get("threshold_sac_release", 0.4):.1f}  (z_sac drops here)')
        ax5.axhline(THETA_SAC.get('threshold_ref', 0.1), color='#c2a5cf',
                    lw=0.8, ls=':', alpha=0.8,
                    label=f'OPN gate threshold={THETA_SAC.get("threshold_ref", 0.1):.1f}  (z_acc can charge again)')
        ax5.set_ylim(-0.05, 1.15)
        _vl(ax5); ax5.grid(True, alpha=0.2)
        ax5.set_xlabel('Time (s)')
        if ci == 0: ax5.legend(fontsize=6.5)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'pursuit_cascade.png')
    fig.savefig(path, dpi=150)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    print(f'  Saved {path}')


# ── Figure 2: vor_saccade_cascade.png ─────────────────────────────────────────

def demo_vor_saccade():
    """Corrective saccades during constant head rotation — 4-panel cascade."""
    dt    = 0.001
    T_end = 3.0
    t     = jnp.arange(0.0, T_end, dt)
    T     = len(t)
    t_np  = np.array(t)

    head_vel_np = np.where(t_np < 1.5, 30.0, 0.0).astype(np.float32)
    hv = jnp.stack([jnp.array(head_vel_np), jnp.zeros(T), jnp.zeros(T)], axis=1)

    theta_sac = {**THETA_DEFAULT,
                 'K_vis': 0.0, 'g_vis': 0.0,    # dark
                 'g_burst':       40.0,
                 'threshold_sac':  3.0,
                 'k_sac':         10.0,
                 'tau_reset_sac':  0.2}
    theta_no_sac = {**theta_sac, 'g_burst': 0.0}

    max_s = int(T_end / dt) + 500

    states_sac = simulate(theta_sac,    t, head_vel_array=hv,
                          target_present_array=jnp.ones(T),   # fixation target visible
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
        _ax_fmt(ax)

    axes[0].plot(t_np, head_pos,             color=_C['head'],   lw=1.0, ls=':', label='head pos')
    axes[0].plot(t_np, s_ns['eye_pos'][:, 0], color=_C['no_sac'], lw=1.2, ls='--', label='VOR only')
    axes[0].plot(t_np, s['eye_pos'][:, 0],   color=_C['eye'],    lw=1.5, label='VOR + saccades')
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
    axes[3].axhline( theta_sac['threshold_sac'], color='gray', lw=0.8, ls=':', label='±threshold')
    axes[3].axhline(-theta_sac['threshold_sac'], color='gray', lw=0.8, ls=':')
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

    print('\n1. Pursuit cascade (ramp target, catch-up saccades)')
    demo_pursuit_cascade()

    print('\n2. VOR + corrective saccades')
    demo_vor_saccade()

    print(f'\nDone. Plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
