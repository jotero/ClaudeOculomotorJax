"""Saccade benchmarks — main sequence, oblique, double-step refractoriness, cascade.

Usage:
    python -X utf8 scripts/bench_saccades.py
    python -X utf8 scripts/bench_saccades.py --show
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

from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, with_sensory, simulate
from oculomotor.sim.simulator import _IDX_VIS_L, _IDX_EC, _IDX_EC_OKR, _IDX_PURSUIT
from oculomotor.sim import kinematics as km
from oculomotor.analysis import ax_fmt, extract_burst, extract_sg, ni_net, vs_net
from oculomotor.models.sensory_models.sensory_model import C_vel as C_vel_sm, C_slip as C_slip_sm

DT    = 0.001
THETA = with_sensory(PARAMS_DEFAULT, sigma_canal=0.5, sigma_pos=0.2, sigma_vel=0.2)
THETA = with_brain(THETA, g_burst=700.0)
SHOW  = '--show' in sys.argv


def _vel_sat_np(v_arr, v_sat):
    """Batch numpy cosine-rolloff saturation matching retina.velocity_saturation."""
    v_zero  = 2.0 * v_sat
    speeds  = np.linalg.norm(v_arr, axis=1, keepdims=True)
    t       = np.clip((speeds - v_sat) / (v_zero - v_sat), 0.0, 1.0)
    gain    = 0.5 * (1.0 + np.cos(np.pi * t))
    return v_arr * gain


def _primary_saccade(burst_yaw, eye_yaw, t_np, t_jump, threshold=20.0):
    """Amplitude and peak velocity of the FIRST saccade after t_jump.

    Uses the burst signal (not raw velocity) to gate on/off so corrective
    saccades and post-saccadic drift are excluded.

    Returns (amplitude_deg, peak_vel_deg_s).  Falls back to full-trace
    extremes only if no burst crossing is found.
    """
    i0     = int(t_jump / DT) + 1          # first sample after step
    burst  = burst_yaw[i0:]
    eye    = eye_yaw[i0:]
    vel    = np.gradient(eye, DT)

    is_sac = np.abs(burst) > threshold
    on_idx  = np.where(np.diff(is_sac.astype(int)) > 0)[0]
    off_idx = np.where(np.diff(is_sac.astype(int)) < 0)[0]

    if len(on_idx) == 0:
        return float(eye[-1] - eye_yaw[i0 - 1]), float(np.max(np.abs(vel)))

    i_on  = on_idx[0] + 1                  # first sample inside saccade
    after = off_idx[off_idx >= on_idx[0]]
    i_off = int(after[0]) + 1 if len(after) > 0 else len(burst) - 1

    amplitude = float(eye[i_off] - eye[i_on - 1])
    peak_vel  = float(np.max(np.abs(vel[i_on:i_off + 1])))
    return amplitude, peak_vel


# ── helpers ───────────────────────────────────────────────────────────────────

def _pt3(t_np, amp_h_deg, amp_v_deg=0.0, t_jump=0.1):
    T = len(t_np)
    pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
    pt3[:, 0] = np.where(t_np >= t_jump, np.tan(np.radians(amp_h_deg)), 0.0)
    pt3[:, 1] = np.where(t_np >= t_jump, np.tan(np.radians(amp_v_deg)), 0.0)
    return jnp.array(pt3)


def _run(t_np, pt3, key=0, max_s=None):
    t  = jnp.array(t_np)
    T  = len(t)
    ms = max_s or int((t_np[-1] - t_np[0]) / DT) + 300
    return simulate(THETA, t,
                    target=km.build_target(t_np, lin_pos=np.array(pt3)),
                    scene_present_array=jnp.ones(T),
                    max_steps=ms, return_states=True,
                    key=jax.random.PRNGKey(key))


# ── Figure 1: main sequence ───────────────────────────────────────────────────

def _main_sequence(show):
    amplitudes = [0.5, 1, 2, 3, 5, 8, 10, 15, 20]
    T_end, t_jump = 0.8, 0.1
    t_np = np.arange(0.0, T_end, DT)

    amps_out, peak_vels = [], []
    traces = {}
    for i, amp in enumerate(amplitudes):
        pt3 = _pt3(t_np, amp, t_jump=t_jump)
        st  = _run(t_np, pt3, key=i)
        # version = average of left and right eye (conjugate movement hits target)
        eye   = (np.array(st.plant[:, 0]) + np.array(st.plant[:, 3])) / 2.0
        burst = extract_burst(st, THETA)[:, 0]
        a_out, v_peak = _primary_saccade(burst, eye, t_np, t_jump)
        amps_out.append(a_out)
        peak_vels.append(v_peak)
        traces[amp] = (t_np - t_jump, eye - eye[int(t_jump / DT)])

    A_ref = np.linspace(0, 22, 300)
    v_ref = 700.0 * (1.0 - np.exp(-A_ref / 7.0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Saccade Main Sequence', fontsize=12)

    axes[0].plot(A_ref, v_ref, color=utils.C['dark'], lw=1.5, ls='--',
                 label='700(1−e^{−A/7})')
    axes[0].scatter(amps_out, peak_vels, color=utils.C['eye'], s=70, zorder=5)
    for a, v in zip(amps_out, peak_vels):
        axes[0].annotate(f'{a:.0f}°', (a, v), fontsize=7,
                         xytext=(3, 3), textcoords='offset points')
    axes[0].set_xlabel('Amplitude (deg)'); axes[0].set_ylabel('Peak velocity (deg/s)')
    axes[0].set_title('Main Sequence (scatter + reference curve)')
    axes[0].legend(fontsize=9); axes[0].set_xlim(0, 22); axes[0].set_ylim(0)
    axes[0].grid(True, alpha=0.25)

    cmap = plt.get_cmap('plasma')
    for i, amp in enumerate(amplitudes):
        t_al, eye_al = traces[amp]
        axes[1].plot(t_al, eye_al, color=cmap(i / (len(amplitudes) - 1)), lw=1.3,
                     label=f'{amp:.0f}°' if amp in [1, 5, 10, 20] else None)
    axes[1].set_xlabel('Time from step (s)'); axes[1].set_ylabel('Eye position (deg)')
    axes[1].set_title('Eye Traces (aligned to target step)')
    axes[1].set_xlim(-0.05, 0.55); axes[1].axvline(0, color='gray', lw=0.7, ls='--')
    axes[1].axhline(0, color='k', lw=0.4); axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'saccade_main_sequence', show=show)
    return utils.fig_meta(path, rp,
        title='Saccade Main Sequence',
        description='Peak velocity vs amplitude scatter (left) and aligned eye traces (right) for amplitudes 0.5–20°.',
        expected='All data within ±20% of 700(1−e^{−A/7}). Peak ≈660 deg/s at 20°.',
        citation='Bahill et al. (1975) Science; Robinson (1975) J Neurophysiol',
        fig_type='behavior')


# ── Figure 2: oblique saccades ────────────────────────────────────────────────

def _oblique(show):
    T_end = 3.5
    t_np  = np.arange(0.0, T_end, DT)
    T     = len(t_np)

    jumps = [(0.3, 12.0, 0.0), (0.9, 12.0, 8.0),
             (1.6,  0.0, 8.0), (2.2,  0.0, 0.0), (2.8, -10.0, 5.0)]
    pt3   = np.zeros((T, 3)); pt3[:, 2] = 1.0
    tgt_h = np.zeros(T); tgt_v = np.zeros(T)
    for t_j, y, p in jumps:
        tgt_h[t_np >= t_j] = y; tgt_v[t_np >= t_j] = p
    pt3[:, 0] = np.tan(np.radians(tgt_h))
    pt3[:, 1] = np.tan(np.radians(tgt_v))

    st    = _run(t_np, jnp.array(pt3), max_s=int(T_end / DT) + 500)
    # version: average of left eye (plant[:, :3]) and right eye (plant[:, 3:6])
    eye_L = np.array(st.plant[:, :3])
    eye_R = np.array(st.plant[:, 3:6])
    eye   = (eye_L + eye_R) / 2.0   # (T, 3) version position

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Oblique Saccade Sequence', fontsize=12)

    axes[0].plot(eye[:, 0], eye[:, 1], color=utils.C['eye'], lw=1.2)
    for _, y, p in jumps:
        axes[0].plot(y, p, 'x', color=utils.C['target'], ms=9, markeredgewidth=2.5)
    axes[0].set_xlabel('Yaw (deg)'); axes[0].set_ylabel('Pitch (deg)')
    axes[0].set_title('2-D Trajectory (straight lines expected)')
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.25)
    axes[0].axhline(0, color='k', lw=0.4); axes[0].axvline(0, color='k', lw=0.4)

    for ax in axes[1:]:
        for t_j, _, _ in jumps:
            ax.axvline(t_j, color='gray', lw=0.6, ls='--', alpha=0.4)

    axes[1].plot(t_np, tgt_h,   color=utils.C['target'], lw=1.5, label='target H')
    axes[1].plot(t_np, eye[:,0], color=utils.C['eye'],   lw=1.5, label='eye H')
    axes[1].set_ylabel('Yaw (deg)'); axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Horizontal Component')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.25); axes[1].axhline(0, color='k', lw=0.4)

    axes[2].plot(t_np, tgt_v,   color=utils.C['target'], lw=1.5, ls='--', label='target V')
    axes[2].plot(t_np, eye[:,1], color=utils.C['eye'],   lw=1.5, ls='--', label='eye V')
    axes[2].set_ylabel('Pitch (deg)'); axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Vertical Component')
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.25); axes[2].axhline(0, color='k', lw=0.4)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'saccade_oblique', show=show)
    return utils.fig_meta(path, rp,
        title='Oblique Saccades',
        description='Sequence of saccades to 5 oblique targets. Left: 2D trajectory. Center/right: H and V components.',
        expected='Straight 2D trajectories. H and V components synchronised (end together).',
        citation='Smit et al. (1990) J Neurophysiol',
        fig_type='behavior')


# ── Figure 3: double-step refractoriness ──────────────────────────────────────

def _refractoriness(show):
    """Double-step paradigm: first step to 10°, second step to 20° at varying ISI."""
    T_end  = 0.8
    t_np   = np.arange(0.0, T_end, DT)
    T      = len(t_np)
    t1     = 0.15   # first step
    isis   = [0.05, 0.10, 0.15, 0.20, 0.35]   # s — inter-step intervals

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Saccade Double-Step Refractoriness\n'
                 'First step: 0→10° at t=0.15 s;  second step: 10→20° at varying ISI',
                 fontsize=11)

    cmap  = plt.get_cmap('viridis')
    colors = [cmap(i / (len(isis) - 1)) for i in range(len(isis))]

    isi_found = []
    for i, isi in enumerate(isis):
        t2 = t1 + isi
        pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
        pt3[:, 0] = np.where(t_np < t1, 0.0,
                    np.where(t_np < t2, np.tan(np.radians(10.0)),
                             np.tan(np.radians(20.0))))

        st  = _run(t_np, jnp.array(pt3), key=i)
        eye = (np.array(st.plant[:, 0]) + np.array(st.plant[:, 3])) / 2.0
        vel = np.gradient(eye, DT)
        bst = extract_burst(st, THETA)[:, 0]

        lbl = f'ISI={isi*1000:.0f}ms'
        axes[0].plot(t_np, eye, color=colors[i], lw=1.8, label=lbl)
        axes[1].plot(t_np, bst, color=colors[i], lw=1.5, alpha=0.8)

        # Count saccades via burst crossings
        is_sac  = np.abs(bst) > 20.0
        n_sac   = int(np.sum(np.diff(is_sac.astype(int)) > 0))
        isi_found.append(n_sac)

    # Target reference lines
    axes[0].axhline(10.0, color=utils.C['target'], lw=1.0, ls=':', alpha=0.6, label='10°')
    axes[0].axhline(20.0, color=utils.C['target'], lw=1.5, ls=':', alpha=0.6, label='20°')
    axes[0].axvline(t1,   color='gray', lw=0.8, ls='--', alpha=0.5)

    # Second step markers per ISI
    for i, isi in enumerate(isis):
        axes[0].axvline(t1 + isi, color=colors[i], lw=0.7, ls=':', alpha=0.4)

    axes[0].set_ylabel('Eye position (deg)'); axes[0].set_title('Eye Position (5 ISIs)')
    axes[0].legend(fontsize=9, loc='upper left'); axes[0].grid(True, alpha=0.25)
    axes[0].axhline(0, color='k', lw=0.4)

    axes[1].set_ylabel('Burst command (deg/s)'); axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Saccade Burst Signal (each peak = one saccade)')
    axes[1].grid(True, alpha=0.25); axes[1].axhline(0, color='k', lw=0.4)
    axes[1].axvline(t1, color='gray', lw=0.8, ls='--', alpha=0.5)

    # Annotation: number of saccades per ISI
    for i, (isi, n) in enumerate(zip(isis, isi_found)):
        axes[1].annotate(f'ISI={isi*1000:.0f}ms → {n} sac',
                         xy=(0.02, 0.92 - i * 0.12), xycoords='axes fraction',
                         fontsize=8, color=colors[i])

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'saccade_refractoriness', show=show)
    return utils.fig_meta(path, rp,
        title='Saccade Double-Step Refractoriness',
        description='Double-step paradigm: target jumps 0→10° then 10→20° after variable ISI. '
                    'Short ISIs produce a single amended saccade (refractory); longer ISIs produce two.',
        expected='ISI < 100 ms: only 1 saccade. ISI > 150 ms: 2 separate saccades.',
        citation='Becker & Jürgens (1979) Vision Res',
        fig_type='behavior')


# ── Figure 4: signal cascade ──────────────────────────────────────────────────

def _cascade(show):
    amplitudes = [1.0, 5.0, 20.0]
    T_end, t_jump = 0.9, 0.1
    t_np = np.arange(0.0, T_end, DT)
    T    = len(t_np)

    n_rows, n_cols = 8, len(amplitudes)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 2.2 * n_rows), sharex=True)
    fig.suptitle('Saccade Signal Cascade  ·  '
                 'cascade → accumulate → latch/freeze → burst → copy → refractory',
                 fontsize=11)

    row_labels = ['Eye + target pos (deg)', 'Cascade output + hold (deg)',
                  'Residual error (deg)', 'Accum / latch + refractory',
                  'Burst (deg/s) + eye velocity', 'Tgt vel + scene slip (deg/s)',
                  'EC-corrected + sat (deg/s)', 'u_pursuit + VS/OKR (deg/s)']
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8)

    for ci, amp in enumerate(amplitudes):
        pt3 = _pt3(t_np, amp, t_jump=t_jump)
        st  = _run(t_np, pt3, key=ci, max_s=int(T_end/DT)+200)
        sg  = extract_sg(st, THETA)
        eye = (np.array(st.plant[:, 0]) + np.array(st.plant[:, 3])) / 2.0
        vel = np.gradient(eye, DT)
        tgt = np.degrees(np.arctan2(np.array(pt3[:,0]), np.array(pt3[:,2])))

        axes[0, ci].set_title(f'{amp:.0f}°', fontsize=11)
        vl  = dict(color='gray', lw=0.6, ls='--', alpha=0.5)
        for ax in axes[:, ci]:
            ax.axvline(t_jump, **vl); ax.grid(True, alpha=0.15)

        axes[0, ci].plot(t_np, tgt, color=utils.C['target'], lw=1.5, label='target')
        axes[0, ci].plot(t_np, eye, color=utils.C['eye'],    lw=1.5, label='eye')
        ax_fmt(axes[0, ci])
        if ci == 0: axes[0, ci].legend(fontsize=7)

        axes[1, ci].plot(t_np, sg['e_pd'][:,0],   color='darkorange', lw=1.0, ls='--', label='e_delayed')
        axes[1, ci].plot(t_np, sg['e_held'][:,0], color=utils.C['vs'], lw=1.8, label='e_held (frozen)')
        ax_fmt(axes[1, ci])
        if ci == 0: axes[1, ci].legend(fontsize=7)

        # Row 2: residual error + copy integrator
        axes[2, ci].plot(t_np, sg['e_res'][:,0],  color=utils.C['target'], lw=1.5, label='e_res')
        axes[2, ci].plot(t_np, sg['x_copy'][:,0], color=utils.C['vs'],     lw=1.2, ls='--', label='x_copy')
        ax_fmt(axes[2, ci])
        if ci == 0: axes[2, ci].legend(fontsize=7)

        # Row 3: accumulator / latch + refractory (all 0–1 scale, same axis)
        axes[3, ci].plot(t_np, sg['z_acc'], color='#e08214', lw=1.5, label='z_acc')
        axes[3, ci].plot(t_np, sg['z_opn'] / 100, color='#1b7837', lw=1.5, label='OPN (norm)')
        axes[3, ci].plot(t_np, sg['z_ref'], color=utils.C['refractory'], lw=1.2, ls='--', label='z_ref')
        axes[3, ci].axhline(THETA.brain.threshold_acc,         color='#e08214',             lw=0.8, ls=':')
        axes[3, ci].axhline(THETA.brain.threshold_sac_release, color=utils.C['refractory'], lw=0.8, ls=':')
        axes[3, ci].axhline(THETA.brain.threshold_ref,         color='#c2a5cf',             lw=0.8, ls=':')
        axes[3, ci].set_ylim(-0.05, 1.15)
        if ci == 0: axes[3, ci].legend(fontsize=7)

        # Row 4: eye velocity first, then burst on top
        axes[4, ci].plot(t_np, vel,                color=utils.C['eye'],   lw=1.4, label='eye vel')
        axes[4, ci].plot(t_np, sg['u_burst'][:,0], color=utils.C['burst'], lw=1.5, label='burst', zorder=3)
        ax_fmt(axes[4, ci])
        if ci == 0: axes[4, ci].legend(fontsize=7)

        # ── Pursuit / OKR signal chain (rows 5–7) ────────────────────────────
        x_vis_L      = np.array(st.sensory[:, _IDX_VIS_L])          # (T, 480)
        vel_del      = x_vis_L @ np.array(C_vel_sm).T                # (T, 3) delayed target vel
        slip_del     = x_vis_L @ np.array(C_slip_sm).T               # (T, 3) delayed scene slip
        motor_ec_pu  = np.array(st.brain[:, _IDX_EC])[:, 117:]       # (T, 3) pursuit EC readout
        motor_ec_okr = np.array(st.brain[:, _IDX_EC_OKR])[:, 117:]  # (T, 3) OKR EC readout
        x_purs       = np.array(st.brain[:, _IDX_PURSUIT])           # (T, 3) pursuit memory

        # Row 5: raw delayed velocities
        axes[5, ci].plot(t_np, vel_del[:, 0],  color='darkorange', lw=1.2, label='tgt_vel')
        axes[5, ci].plot(t_np, slip_del[:, 0], color='teal',       lw=1.2, label='scene_slip')
        ax_fmt(axes[5, ci])
        if ci == 0: axes[5, ci].legend(fontsize=7)

        # Row 6: EC-corrected + saturated (pursuit path uses pursuit EC; OKR path uses OKR EC)
        tgt_ec_sat   = _vel_sat_np(vel_del  + motor_ec_pu,  THETA.brain.v_max_pursuit)
        scene_ec_sat = _vel_sat_np(slip_del + motor_ec_okr, THETA.brain.v_max_okr)
        axes[6, ci].plot(t_np, tgt_ec_sat[:, 0],   color='#7b2d8b', lw=1.2, label='tgt−EC (sat)')
        axes[6, ci].plot(t_np, scene_ec_sat[:, 0],  color='#1a7a4a', lw=1.2, label='scene−EC (sat)')
        ax_fmt(axes[6, ci])
        if ci == 0: axes[6, ci].legend(fontsize=7)

        # Row 7: pursuit output u_pursuit + VS net (OKR-driven since no head movement)
        K_ph = float(THETA.brain.K_phasic_pursuit)
        e_pred     = (tgt_ec_sat - x_purs) / (1.0 + K_ph)
        u_purs_arr = x_purs + K_ph * e_pred
        vs_out     = vs_net(st)                                  # (T, 3) VS net x_L−x_R
        axes[7, ci].plot(t_np, u_purs_arr[:, 0], color='steelblue',  lw=1.5, label='u_pursuit')
        axes[7, ci].plot(t_np, vs_out[:, 0],     color='#d45500',    lw=1.2, label='VS/OKR')
        ax_fmt(axes[7, ci])
        axes[7, ci].set_xlabel('Time (s)', fontsize=8)
        if ci == 0: axes[7, ci].legend(fontsize=7)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'saccade_cascade', show=show)
    return utils.fig_meta(path, rp,
        title='Saccade Signal Cascade (Internal)',
        description='Row-by-row signal flow for 1°, 5°, 20° saccades: position, visual cascade + hold, '
                    'accumulator/latch, residual error, burst, eye velocity, refractory state.',
        expected='e_held freezes at saccade onset; burst proportional to e_res; '
                 'z_ref locks out next saccade for ~150 ms.',
        citation='Robinson (1975) J Neurophysiol; Scudder et al. (2002)',
        fig_type='cascade')


# ── Section entry point ────────────────────────────────────────────────────────

SECTION = dict(
    id='saccades', title='1. Saccades',
    description='Saccade kinematics and internal signal cascade. Tests main sequence (amplitude–velocity), '
                'oblique trajectory linearity, double-step refractoriness, and the full Robinson cascade.',
)


def run(show=False):
    print('\n=== Saccades ===')
    figs = []
    print('  1/4  main sequence …')
    figs.append(_main_sequence(show))
    print('  2/4  oblique saccades …')
    figs.append(_oblique(show))
    print('  3/4  double-step refractoriness …')
    figs.append(_refractoriness(show))
    print('  4/4  signal cascade …')
    figs.append(_cascade(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
