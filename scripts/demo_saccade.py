"""Saccade generator demos.

Figures produced
────────────────
    saccade_cascade.png  — 5 rows × 3 amplitudes: signal cascade incl. z_ref
    saccade_summary.png  — main sequence + oblique sequence (merged summary)

Usage
-----
    python -X utf8 scripts/demo_saccade.py
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

from oculomotor.sim.simulator import (
    THETA_DEFAULT, simulate,
    _IDX_NI, _IDX_SG, _IDX_VIS,
)
from oculomotor.models import saccade_generator as sg_mod
from oculomotor.models.sensory_model import C_slip, C_pos
from oculomotor.models import retina

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

_C = {
    'target': '#d6604d',
    'eye':    '#2166ac',
    'ni':     '#4dac26',
    'error':  '#762a83',
    'burst':  '#f4a582',
    'vel':    '#1a9641',
    'reset':  '#e08214',
    'ref':    '#aaaaaa',
}

THETA_SAC = {**THETA_DEFAULT, 'g_burst': 700.0}


# ── Utilities ──────────────────────────────────────────────────────────────────

def _make_pt3(t_np, jumps_deg, T):
    """Build (T, 3) target array from a list of (t_jump, yaw_deg, pitch_deg)."""
    pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
    yaw_deg = np.zeros(T); pitch_deg = np.zeros(T)
    for t_j, y, p in jumps_deg:
        yaw_deg[t_np >= t_j]   = y
        pitch_deg[t_np >= t_j] = p
    pt3[:, 0] = np.tan(np.radians(yaw_deg))
    pt3[:, 1] = np.tan(np.radians(pitch_deg))
    return jnp.array(pt3), yaw_deg, pitch_deg


def _extract(states, pt3_np, theta, t_np):
    """Extract signals from full state trajectory."""
    x_p   = np.array(states.plant)
    x_ni  = np.array(states.brain[:, _IDX_NI])
    x_vis = np.array(states.sensory[:, _IDX_VIS])
    x_sg  = np.array(states.brain[:, _IDX_SG])

    # SG state layout: [x_copy(3) | z_ref(1) | e_held(3) | z_sac(1) | z_acc(1)]
    x_copy = x_sg[:, :3]
    z_ref  = x_sg[:, 3]
    e_held = x_sg[:, 4:7]
    z_sac  = x_sg[:, 7]
    z_acc  = x_sg[:, 8]
    e_res  = e_held - x_copy   # ballistic residual

    e_pos_delayed = (x_vis @ np.array(C_pos).T)  # (T, 3)

    def _burst_at(state):
        e_pd = C_pos @ state.sensory[_IDX_VIS]
        _, u = sg_mod.step(state.brain[_IDX_SG], e_pd, theta)
        return u
    u_burst = np.array(jax.vmap(_burst_at)(states))

    target_yaw   = np.degrees(np.arctan2(pt3_np[:, 0], pt3_np[:, 2]))
    target_pitch = np.degrees(np.arctan2(pt3_np[:, 1], pt3_np[:, 2]))

    dt = float(t_np[1] - t_np[0])
    return dict(
        eye_pos=x_p, x_ni=x_ni,
        e_pos_delayed=e_pos_delayed[:, 0],
        e_held=e_held[:, 0],
        x_copy=x_copy[:, 0],
        e_res=e_res[:, 0],
        z_ref=z_ref, z_sac=z_sac, z_acc=z_acc,
        u_burst_h=u_burst[:, 0],
        u_burst_v=u_burst[:, 1],
        eye_vel_h=np.gradient(x_p[:, 0], dt),
        eye_vel_v=np.gradient(x_p[:, 1], dt),
        target_yaw=target_yaw,
        target_pitch=target_pitch,
    )


def _ax_fmt(ax):
    ax.axhline(0, color='k', lw=0.4)
    ax.grid(True, alpha=0.2)


# ── Figure 1: saccade_cascade.png ─────────────────────────────────────────────
#
# Signal flow (top → bottom):
#   Target step                               [row 0: position]
#     → 40-stage visual cascade (120 ms)      [row 1: e_pos_delayed rising]
#     → gate_err fires → z_acc accumulates    [row 2: trigger pathway]
#     → z_acc > 0.5 → z_sac latches          [row 2: z_sac fires]
#     → e_held freezes at settled cascade val [row 1: e_held flat during saccade]
#     → e_res = e_held − x_copy drives burst  [row 3: residual closes]
#     → u_burst → NI + plant → eye velocity  [row 4: burst & eye vel]
#     → x_copy integrates toward e_held       [row 3: x_copy rising]
#     → e_res → 0 → z_ref charges            [row 5: refractory]
#     → z_ref > 0.4 → z_sac releases         [row 2: z_sac drops]
#     → z_ref decays → z_acc can fire again   [row 5: z_ref decaying]

def demo_saccade_cascade():
    """6 rows × 5 amplitudes: full signal-flow cascade."""
    amplitudes = [0.6, 2.0, 5.0, 10.0, 70.0]
    dt     = 0.001
    T_end  = 1.0
    t_jump = 0.2
    t      = jnp.arange(0.0, T_end, dt)
    T      = len(t)
    t_np   = np.array(t)

    n_rows, n_cols = 6, len(amplitudes)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 2.2 * n_rows),
                             sharex=True)
    fig.suptitle('Saccade Signal Cascade  ·  flow: cascade → accumulate → latch/freeze → burst → copy → refractory',
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

    for ci, amp in enumerate(amplitudes):
        pt3, tgt_yaw, _ = _make_pt3(t_np, [(t_jump, amp, 0.0)], T)
        max_s = int(T_end / dt) + 200

        states = simulate(THETA_SAC, t, p_target_array=pt3,
                          scene_present_array=jnp.ones(T),
                          max_steps=max_s, return_states=True)
        s = _extract(states, np.array(pt3), THETA_SAC, t_np)

        def _vl(ax):
            ax.axvline(t_jump, color='gray', lw=0.6, ls='--', alpha=0.4)

        axes[0, ci].set_title(f'{amp:.1f}°', fontsize=10)

        # ── Row 0: position ────────────────────────────────────────────────────
        axes[0, ci].plot(t_np, tgt_yaw,            color=_C['target'], lw=1.5, label='target pos')
        axes[0, ci].plot(t_np, s['eye_pos'][:, 0], color=_C['eye'],    lw=1.5, label='eye pos')
        axes[0, ci].plot(t_np, s['x_ni'][:, 0],    color=_C['ni'],     lw=0.9, ls='--', label='NI (eye hold signal)')
        _vl(axes[0, ci]); _ax_fmt(axes[0, ci])
        if ci == 0: axes[0, ci].legend(fontsize=6.5)

        # ── Row 1: cascade output (e_pos_delayed) and sample-and-hold (e_held) ─
        axes[1, ci].plot(t_np, s['e_pos_delayed'], color=_C['error'], lw=1.0, ls='--',
                         label='e_delayed  (cascade output, settling ~120 ms)')
        axes[1, ci].plot(t_np, s['e_held'],        color=_C['reset'], lw=1.8,
                         label='e_held  (tracks cascade; freezes when z_sac=1)')
        _vl(axes[1, ci]); _ax_fmt(axes[1, ci])
        if ci == 0: axes[1, ci].legend(fontsize=6.5)

        # ── Row 2: trigger pathway — z_acc (analog) + z_sac (binary latch) ────
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

        # ── Row 3: ballistic residual e_res = e_held − x_copy, and x_copy ─────
        axes[3, ci].plot(t_np, s['e_res'],  color=_C['error'], lw=1.5,
                         label='e_res = e_held − x_copy  (drives burst; closes to 0)')
        axes[3, ci].plot(t_np, s['x_copy'], color=_C['reset'],  lw=1.2, ls='--',
                         label='x_copy  (internal copy; resets to 0 between saccades)')
        axes[3, ci].axhline(amp, color=_C['target'], lw=0.7, ls=':', alpha=0.6,
                            label=f'target amp = {amp}°')
        _vl(axes[3, ci]); _ax_fmt(axes[3, ci])
        if ci == 0: axes[3, ci].legend(fontsize=6.5)

        # ── Row 4: burst command and eye velocity ─────────────────────────────
        axes[4, ci].plot(t_np, s['u_burst_h'], color=_C['burst'], lw=1.5,
                         label='u_burst  (velocity command to NI + plant)')
        axes[4, ci].plot(t_np, s['eye_vel_h'], color=_C['vel'],   lw=1.2, ls='--',
                         label='eye velocity  (plant response)')
        _vl(axes[4, ci]); _ax_fmt(axes[4, ci])
        yabs = max(np.max(np.abs(s['u_burst_h'])), np.max(np.abs(s['eye_vel_h'])), 1.0)
        axes[4, ci].set_ylim(-yabs * 0.1, yabs * 1.1)
        if ci == 0: axes[4, ci].legend(fontsize=6.5)

        # ── Row 5: refractory state z_ref ────────────────────────────────────
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
    path = os.path.join(OUTPUT_DIR, 'saccade_cascade.png')
    fig.savefig(path, dpi=150)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    print(f'  Saved {path}')


# ── Figure 2: saccade_summary.png ─────────────────────────────────────────────

def demo_saccade_summary():
    """Main sequence + oblique sequence — merged summary figure.

    Left column (2 panels): main sequence scatter + 2D trajectory plot.
    Right column (3 panels): H pos, V pos, burst for an oblique sequence.
    """
    dt    = 0.001
    T_end = 3.5
    t     = jnp.arange(0.0, T_end, dt)
    T     = len(t)
    t_np  = np.array(t)

    # ── Main sequence ──────────────────────────────────────────────────────────
    amplitudes_deg = np.array([0.5, 1, 2, 3, 5, 8, 10, 15, 20], dtype=np.float32)
    t_ms = jnp.arange(0.0, 0.8, dt)
    T_ms = len(t_ms)
    t_np_ms = np.array(t_ms)
    t_jump_ms = 0.1

    amps_out, peak_vels = [], []
    for amp in amplitudes_deg:
        pt3_ms = jnp.stack([
            jnp.where(t_ms >= t_jump_ms, jnp.tan(jnp.radians(float(amp))), 0.0),
            jnp.zeros(T_ms),
            jnp.ones(T_ms),
        ], axis=1)
        eye = simulate(THETA_SAC, t_ms, p_target_array=pt3_ms,
                       scene_present_array=jnp.ones(T_ms),
                       max_steps=int(0.8/dt)+200)
        eye_yaw = np.array(eye[:, 0])
        vel = np.gradient(eye_yaw, dt)
        amps_out.append(float(eye_yaw[-1] - eye_yaw[0]))
        peak_vels.append(float(np.max(np.abs(vel))))

    # ── Oblique sequence ───────────────────────────────────────────────────────
    # Sequence of jumps to different 2D targets
    oblique_jumps = [
        (0.3,  10.0,   0.0),
        (0.9,  10.0,   8.0),
        (1.6,   0.0,   8.0),
        (2.2,   0.0,   0.0),
        (2.8, -10.0,   5.0),
    ]
    pt3_obl, tgt_yaw, tgt_pitch = _make_pt3(t_np, oblique_jumps, T)
    max_s = int(T_end / dt) + 500

    states = simulate(THETA_SAC, t, p_target_array=pt3_obl,
                      scene_present_array=jnp.ones(T),
                      max_steps=max_s, return_states=True)
    s = _extract(states, np.array(pt3_obl), THETA_SAC, t_np)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Saccade Summary — Main Sequence + Oblique Sequence', fontsize=12)

    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)
    ax_ms   = fig.add_subplot(gs[0, 0])   # main sequence scatter
    ax_traj = fig.add_subplot(gs[1, 0])   # 2D trajectory
    ax_eye  = fig.add_subplot(gs[2, 0])   # eye traces (aligned to step)
    ax_ph   = fig.add_subplot(gs[0, 1:])  # H position sequence
    ax_pv   = fig.add_subplot(gs[1, 1:], sharex=ax_ph)  # V position sequence
    ax_bur  = fig.add_subplot(gs[2, 1:], sharex=ax_ph)  # burst

    # Main sequence
    A_ref = np.linspace(0, 22, 200)
    v_ref = 700.0 * (1.0 - np.exp(-A_ref / 7.0))
    ax_ms.plot(A_ref, v_ref, color=_C['ref'], lw=1.2, ls='--', label='700(1−e^{−A/7})')
    ax_ms.scatter(amps_out, peak_vels, color=_C['eye'], s=60, zorder=5)
    ax_ms.set_xlabel('Amplitude (deg)'); ax_ms.set_ylabel('Peak vel (deg/s)')
    ax_ms.set_title('Main Sequence'); ax_ms.legend(fontsize=7)
    ax_ms.set_xlim(0, 22); ax_ms.set_ylim(0)
    ax_ms.grid(True, alpha=0.2)

    # 2D trajectory (oblique sequence)
    ax_traj.plot(s['eye_pos'][:, 0], s['eye_pos'][:, 1],
                 color=_C['eye'], lw=1.2, label='eye path')
    for _, y, p in oblique_jumps:
        ax_traj.plot(y, p, 'x', color=_C['target'], ms=8, markeredgewidth=2)
    ax_traj.set_xlabel('Yaw (deg)'); ax_traj.set_ylabel('Pitch (deg)')
    ax_traj.set_title('2-D Trajectory (oblique sequence)')
    ax_traj.set_aspect('equal'); ax_traj.grid(True, alpha=0.2)
    ax_traj.axhline(0, color='k', lw=0.4); ax_traj.axvline(0, color='k', lw=0.4)

    # Aligned eye traces (H only, all single-amplitude saccades)
    cmap = plt.get_cmap('plasma')
    for i, amp in enumerate(amplitudes_deg):
        pt3_ms = jnp.stack([
            jnp.where(t_ms >= t_jump_ms, jnp.tan(jnp.radians(float(amp))), 0.0),
            jnp.zeros(T_ms), jnp.ones(T_ms),
        ], axis=1)
        eye_i = np.array(simulate(THETA_SAC, t_ms, p_target_array=pt3_ms,
                                  scene_present_array=jnp.ones(T_ms),
                                  max_steps=int(0.8/dt)+200)[:, 0])
        ax_eye.plot(t_np_ms - t_jump_ms, eye_i,
                    color=cmap(i / (len(amplitudes_deg) - 1)), lw=1.2,
                    label=f'{amp:.0f}°' if amp in [1, 5, 10, 20] else None)
    ax_eye.set_xlabel('Time from step (s)'); ax_eye.set_ylabel('Eye pos (deg)')
    ax_eye.set_title('Eye traces (all amplitudes)')
    ax_eye.set_xlim(-0.05, 0.5); ax_eye.axvline(0, color='gray', lw=0.6, ls='--')
    ax_eye.axhline(0, color='k', lw=0.4); ax_eye.legend(fontsize=7); ax_eye.grid(True, alpha=0.2)

    # Oblique sequence time panels
    for t_j, _, _ in oblique_jumps:
        ax_ph.axvline(t_j, color='gray', lw=0.6, ls='--', alpha=0.5)
        ax_pv.axvline(t_j, color='gray', lw=0.6, ls='--', alpha=0.5)
        ax_bur.axvline(t_j, color='gray', lw=0.6, ls='--', alpha=0.5)

    ax_ph.plot(t_np, tgt_yaw,            color=_C['target'], lw=1.5, label='target H')
    ax_ph.plot(t_np, s['eye_pos'][:, 0], color=_C['eye'],    lw=1.5, label='eye H')
    ax_ph.set_ylabel('Yaw (deg)'); ax_ph.set_title('Oblique Sequence — Horizontal')
    ax_ph.legend(fontsize=8); ax_ph.grid(True, alpha=0.2); ax_ph.axhline(0, color='k', lw=0.4)

    ax_pv.plot(t_np, tgt_pitch,          color=_C['target'], lw=1.5, ls='--', label='target V')
    ax_pv.plot(t_np, s['eye_pos'][:, 1], color=_C['eye'],    lw=1.5, ls='--', label='eye V')
    ax_pv.set_ylabel('Pitch (deg)'); ax_pv.set_title('Oblique Sequence — Vertical')
    ax_pv.legend(fontsize=8); ax_pv.grid(True, alpha=0.2); ax_pv.axhline(0, color='k', lw=0.4)

    ax_bur.plot(t_np, s['u_burst_h'], color=_C['burst'], lw=1.2, label='burst H')
    ax_bur.plot(t_np, s['u_burst_v'], color='steelblue', lw=1.2, ls='--', label='burst V')
    ax_bur.plot(t_np, s['z_ref'] * 200, color='purple', lw=0.8, ls=':', label='z_ref × 200')
    ax_bur.set_ylabel('Burst (deg/s) / z_ref×200'); ax_bur.set_xlabel('Time (s)')
    ax_bur.set_title('Burst + Refractory state'); ax_bur.legend(fontsize=8)
    ax_bur.grid(True, alpha=0.2); ax_bur.axhline(0, color='k', lw=0.4)

    path = os.path.join(OUTPUT_DIR, 'saccade_summary.png')
    fig.savefig(path, dpi=150)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    print(f'  Saved {path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=== Saccade Demo ===')

    print('\n1. Saccade cascade (3 amplitudes)')
    demo_saccade_cascade()

    print('\n2. Saccade summary (main sequence + oblique sequence)')
    demo_saccade_summary()

    print(f'\nDone. Plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
