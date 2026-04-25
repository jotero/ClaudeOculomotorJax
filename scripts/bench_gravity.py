"""Gravity estimator benchmarks — OCR, OVAR (Fig 5), tilt suppression (Fig 6).

Replicates key figures from Laurens & Angelaki (2011) Exp Brain Res 210:407-422.

Parameters (Laurens & Angelaki 2011):
    K_grav = 0.6  (go — somatogravic otolith correction gain)
    K_gd   = 0.5  (gravity dumping — damps VS ⊥ gravity; enables tilt suppression / OVAR)

Usage:
    python -X utf8 scripts/bench_gravity.py
    python -X utf8 scripts/bench_gravity.py --show
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as utils

import numpy as np
import jax.numpy as jnp
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, simulate, SimConfig, _IDX_GRAV,
)
from oculomotor.sim import kinematics as km
from oculomotor.analysis import ax_fmt, vs_net, fit_tc, extract_burst, extract_spv

SHOW = '--show' in sys.argv
DT   = 0.001
G0   = 9.81

K_GD   = 0.5
K_GRAV = 0.6
G_OCR  = 10.0 / 9.81   # OCR gain (deg/(m/s²)): 10° at 90° tilt, ~10% at all angles


SECTION = dict(
    id='gravity', title='4. Gravity Estimator',
    description='Canal-otolith interaction: OCR, OVAR, VOR tilt suppression, '
                'OCR vs tilt angle, somatogravic OCR frequency dependence. '
                f'Parameters: K_grav={K_GRAV}, K_gd={K_GD}, g_ocr={G_OCR} (Laurens & Angelaki 2011).',
)


def _pad3(v1d, axis):
    T = len(v1d)
    out = np.zeros((T, 3))
    out[:, {'yaw': 0, 'pitch': 1, 'roll': 2}[axis]] = v1d
    return out


def _stim_twin(ax, t, pos, vel, pos_label, vel_label, pos_color, vel_color):
    """Plot position on left y-axis and velocity on right y-axis (twin)."""
    ax.plot(t, pos, color=pos_color, lw=1.5, label=pos_label)
    ax.set_ylabel(pos_label, fontsize=8, color=pos_color)
    ax.tick_params(axis='y', labelcolor=pos_color)
    ax2 = ax.twinx()
    ax2.plot(t, vel, color=vel_color, lw=1.0, ls='--', alpha=0.8, label=vel_label)
    ax2.set_ylabel(vel_label, fontsize=8, color=vel_color)
    ax2.tick_params(axis='y', labelcolor=vel_color)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.15)
    return ax2


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ocular Counterroll (OCR)
# ─────────────────────────────────────────────────────────────────────────────

def _ocr(show):
    TILT_DEG = 30.0
    TILT_VEL = 60.0
    TILT_T   = TILT_DEG / TILT_VEL
    HOLD_T   = 8.0
    TOTAL    = TILT_T + HOLD_T
    t        = np.arange(0.0, TOTAL, DT)

    hv_roll  = np.where(t < TILT_T, TILT_VEL, 0.0)
    head_vel = _pad3(hv_roll, 'roll')
    head_pos = np.cumsum(hv_roll) * DT

    params = with_brain(PARAMS_DEFAULT, g_ocr=G_OCR, g_burst=0.0)
    st     = simulate(params, t,
                      head=km.build_kinematics(t, rot_vel=head_vel),
                      target_present_array=np.zeros(len(t)),
                      sim_config=SimConfig(warmup_s=0.0),
                      return_states=True)

    eye_roll = (np.array(st.plant[:, 2]) + np.array(st.plant[:, 5])) / 2.0
    g_est    = np.array(st.brain[:, _IDX_GRAV])
    # Positive roll = left ear down; compensatory OCR = positive (CW, top-of-eye goes right)
    # OCR = g_ocr × G0 × sin(tilt) (sin formula, g_ocr in deg/(m/s²))
    ocr_ss   = G_OCR * G0 * np.sin(np.radians(TILT_DEG))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f'Ocular Counterroll (OCR) — {TILT_DEG:.0f}° Head Roll (left ear down)  '
                 f'(g_ocr={G_OCR}, K_grav={K_GRAV})', fontsize=12, fontweight='bold')

    _stim_twin(axes[0], t, head_pos, hv_roll,
               'Head roll pos (deg)', 'Head roll vel (deg/s)',
               utils.C['head'], 'gray')
    axes[0].axvline(TILT_T, color='gray', lw=0.7, ls=':')
    axes[0].set_title('Stimulus: 30° roll (left ear down) then hold', fontsize=9)

    axes[1].plot(t, eye_roll, color=utils.C['eye'], lw=1.5, label='Eye torsion')
    axes[1].axhline(ocr_ss, color='tomato', lw=1.0, ls='--',
                    label=f'Expected OCR ≈ {ocr_ss:.1f}° (g_ocr×G0×sin(30°))')
    axes[1].axhline(0.0, color='k', lw=0.4)
    axes[1].axvline(TILT_T, color='gray', lw=0.7, ls=':')
    ax_fmt(axes[1], ylabel='Eye torsion (deg)')
    axes[1].set_ylim(-1, 7); axes[1].legend(fontsize=8)

    # g_est[1] > 0 when right ear down; for left-ear-down tilt: g_est[1] → −G₀ sin(30°)
    expected_y = -G0 * np.sin(np.radians(TILT_DEG))
    axes[2].plot(t, g_est[:, 1], color=utils.C['canal'], lw=1.2,
                 label='g_est[1] interaural (m/s²)')
    axes[2].axhline(expected_y, color='tomato', lw=0.9, ls='--',
                    label=f'Expected = −G₀ sin(30°) = {expected_y:.2f} m/s²')
    axes[2].axhline(0.0, color='k', lw=0.4)
    axes[2].axvline(TILT_T, color='gray', lw=0.7, ls=':')
    ax_fmt(axes[2], ylabel='g_est[1] (m/s²)', xlabel='Time (s)')
    axes[2].set_ylim(-12, 3); axes[2].legend(fontsize=8)
    axes[2].set_title('g_est[1] → −G₀ sin(30°) ≈ −4.9 m/s²  (TC = 1/K_grav ≈ 1.7 s); '
                      'g_est[1] < 0 = left ear down', fontsize=9)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'gravity_ocr', show=show)
    return utils.fig_meta(path, rp,
        title='Ocular Counterroll (OCR)',
        description=f'Head rolls {TILT_DEG:.0f}°; g_ocr=0.13 drives torsional eye response.',
        expected=f'Eye counter-rolls to ≈+{ocr_ss:.1f}° (CW, compensatory). '
                 f'g_est[1] → {expected_y:.1f} m/s² with TC ≈ 1/K_grav = {1/K_GRAV:.1f} s.',
        citation='Laurens & Angelaki (2011); Boff et al. (1986)',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────
# 2. OVAR — Off-Vertical Axis Rotation  (Laurens Fig 5)
# ─────────────────────────────────────────────────────────────────────────────

def _ovar(show):
    SPIN_VEL  = 60.0
    TOTAL     = 60.0
    TILTS_DEG = [10.0, 30.0, 60.0]

    params = with_brain(PARAMS_DEFAULT, K_gd=K_GD, g_burst=700.0)
    cfg    = SimConfig(warmup_s=0.0)   # start from rest (v[0]=0 required)
    t      = np.arange(0.0, TOTAL, DT)
    T      = len(t)
    period = 360.0 / SPIN_VEL

    colors = ['#1b7837', '#762a83', '#e08214']

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f'OVAR — Off-Vertical Axis Rotation  (Laurens & Angelaki 2011, Fig 5)\n'
        f'{SPIN_VEL:.0f} °/s rotation, K_gd={K_GD}, K_grav={K_GRAV}',
        fontsize=12, fontweight='bold')

    for ci, tilt_deg in enumerate(TILTS_DEG):
        alpha   = np.radians(tilt_deg)
        omega_y = SPIN_VEL * np.cos(alpha)
        omega_z = SPIN_VEL * np.sin(alpha)

        # v[0]=0 so warmup prepend is at rest
        head_vel = np.stack([np.where(t > 0, omega_y, 0.0),
                             np.zeros(T),
                             np.where(t > 0, omega_z, 0.0)], axis=1)
        head_km  = km.build_kinematics(t, rot_vel=head_vel)

        st       = simulate(params, t,
                            head=head_km,
                            scene_present_array=np.zeros(T),
                            target_present_array=np.zeros(T),
                            sim_config=cfg, return_states=True)

        eye_pos = (np.array(st.plant[:, 0]) + np.array(st.plant[:, 3])) / 2.0
        eye_vel = np.gradient(eye_pos, DT)
        burst   = np.array(extract_burst(st, params)[:, 0])
        spv     = extract_spv(t, eye_vel, burst)
        g_est   = np.array(st.brain[:, _IDX_GRAV])

        # Head orientation: yaw and roll positions (from kinematics)
        head_yaw  = head_km.rot_pos[:, 0]
        head_roll = head_km.rot_pos[:, 2]

        col = colors[ci]
        lbl = f'{tilt_deg:.0f}° tilt'
        axes[0].plot(t, spv,           color=col, lw=1.2, alpha=0.85, label=lbl)
        axes[1].plot(t, g_est[:, 1],   color=col, lw=1.2, label=lbl)
        axes[2].plot(t, vs_net(st)[:,0], color=col, lw=1.2, label=lbl)
        # Stimulus panel: head yaw and roll orientation (overlapping circles as head rotates)
        if ci == len(TILTS_DEG) - 1:   # annotate only for largest tilt for clarity
            axes[3].plot(t % period, head_yaw % 360,  color=col, lw=0.5, alpha=0.3)
        axes[3].plot(t, head_yaw,  color=col, lw=1.0, ls='-',  alpha=0.7, label=f'{tilt_deg:.0f}° yaw pos')
        axes[3].plot(t, head_roll, color=col, lw=1.0, ls='--', alpha=0.7, label=f'{tilt_deg:.0f}° roll pos')

    pm = np.arange(period, TOTAL, period)
    for ax in axes[:3]:
        for p in pm: ax.axvline(p, color='gray', lw=0.4, ls=':', alpha=0.5)
        ax.axhline(0, color='k', lw=0.4); ax.grid(True, alpha=0.15)

    axes[3].grid(True, alpha=0.15); axes[3].axhline(0, color='k', lw=0.4)
    for p in pm: axes[3].axvline(p, color='gray', lw=0.4, ls=':', alpha=0.5)

    axes[0].set_ylabel('Eye SPV (deg/s)', fontsize=9); axes[0].legend(fontsize=8)
    axes[0].set_title('SPV sinusoidally modulated at rotation period', fontsize=9)

    amp_ref = G0 * np.sin(np.radians(TILTS_DEG[-1]))
    axes[1].set_ylabel('g_est[1] interaural (m/s²)', fontsize=9); axes[1].legend(fontsize=8)
    axes[1].set_title(f'Gravity oscillates ±G₀ sin(α); largest: ±{amp_ref:.1f} m/s²', fontsize=9)
    axes[1].set_ylim(-12, 12)

    axes[2].set_ylabel('VS net yaw (deg/s)', fontsize=9); axes[2].legend(fontsize=8)
    axes[2].set_title('VS modulated by gravity dumping (K_gd)', fontsize=9)

    axes[3].set_ylabel('Head orientation (deg)', fontsize=9)
    axes[3].set_xlabel('Time (s)', fontsize=9)
    axes[3].set_title('Stimulus: head yaw pos (solid) and roll pos (dashed) — both grow linearly; '
                      'dashed period markers = one full revolution', fontsize=9)
    # Simplified legend: just show color meaning
    from matplotlib.lines import Line2D
    axes[3].legend(handles=[
        Line2D([0],[0], color=colors[i], lw=1.5, label=f'{d:.0f}° tilt')
        for i, d in enumerate(TILTS_DEG)
    ], fontsize=8)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'gravity_ovar', show=show)
    return utils.fig_meta(path, rp,
        title='OVAR — Off-Vertical Axis Rotation',
        description=f'{SPIN_VEL:.0f}°/s rotation, tilt angles {TILTS_DEG}°. '
                    'Replicates Laurens & Angelaki (2011) Fig 5.',
        expected='SPV sinusoidally modulated at rotation period. '
                 'Modulation amplitude ∝ sin(tilt). g_est oscillates ±G₀ sin(α).',
        citation='Laurens & Angelaki (2011) Exp Brain Res 210:407',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────
# 3. VOR Tilt Suppression  (Laurens Fig 6)
# ─────────────────────────────────────────────────────────────────────────────

def _tilt_suppression(show):
    ROT_VEL   = 60.0
    ROT_T     = 20.0
    COAST_T   = 60.0
    SETTLE_T  = 8.0
    TILTS_DEG = [0.0, 30.0, 60.0, 90.0]
    TILT_VEL  = 60.0

    params = with_brain(PARAMS_DEFAULT, K_gd=K_GD)
    cfg    = SimConfig(warmup_s=0.0)
    colors = ['steelblue', '#2196a8', '#e08214', '#c62e2e']

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
    fig.suptitle(
        f'VOR Tilt Suppression  (Laurens & Angelaki 2011, Fig 6)\n'
        f'{ROT_VEL:.0f}°/s yaw, {ROT_T:.0f} s;  K_gd={K_GD}, K_grav={K_GRAV}',
        fontsize=12, fontweight='bold')

    taus = {}
    stim_data = {}   # save for stimulus panel

    for ci, tilt_deg in enumerate(TILTS_DEG):
        tilt_dur    = tilt_deg / TILT_VEL if tilt_deg > 0 else 0.0
        t_rot_start = tilt_dur + SETTLE_T if tilt_deg > 0 else 0.0
        t_total     = t_rot_start + ROT_T + COAST_T
        t_arr       = np.arange(0.0, t_total, DT)
        T           = len(t_arr)

        if tilt_deg == 0.0:
            hv_roll = np.zeros(T)
            hv_yaw  = np.where(t_arr < ROT_T, ROT_VEL, 0.0)
        else:
            hv_roll = np.where(t_arr < tilt_dur, TILT_VEL, 0.0)
            hv_yaw  = np.where(
                (t_arr >= t_rot_start) & (t_arr < t_rot_start + ROT_T),
                ROT_VEL, 0.0)
        hv_3d = np.stack([hv_yaw, np.zeros(T), hv_roll], axis=1)

        head_km = km.build_kinematics(t_arr, rot_vel=hv_3d)

        st = simulate(params, t_arr,
                      head=head_km,
                      scene_present_array=np.zeros(T),
                      target_present_array=np.zeros(T),
                      sim_config=cfg, return_states=True)

        eye_pos = (np.array(st.plant[:, 0]) + np.array(st.plant[:, 3])) / 2.0
        eye_vel = np.gradient(eye_pos, DT)
        burst   = np.array(extract_burst(st, params)[:, 0])
        spv     = extract_spv(t_arr, eye_vel, burst)
        g_est_y = np.array(st.brain[:, _IDX_GRAV])[:, 1]
        t_rel   = t_arr - t_rot_start

        stim_data[tilt_deg] = {
            't_rel':   t_rel,
            'hv_yaw':  hv_yaw,
            'hv_roll': hv_roll,
            'roll_pos': head_km.rot_pos[:, 2],
        }

        tau, t_fit, y_fit = fit_tc(t_rel, spv, ROT_T, ROT_T + COAST_T)
        taus[tilt_deg] = tau
        col = colors[ci]
        lbl = f'{tilt_deg:.0f}° roll' + (f'  τ={tau:.1f} s' if tau else '')

        axes[0].plot(t_rel, spv, color=col, lw=1.2, label=lbl)
        if t_fit is not None:
            axes[0].plot(t_fit, y_fit, color=col, lw=2.5, ls=':', alpha=0.9)
        axes[1].plot(t_rel, g_est_y, color=col, lw=1.2, label=f'{tilt_deg:.0f}°')

    # ── Stimulus panel: head roll pos (left y) + head yaw vel (right y) ──────
    ax_stim = axes[2]
    ax_stim2 = ax_stim.twinx()
    for ci, tilt_deg in enumerate(TILTS_DEG):
        d = stim_data[tilt_deg]
        col = colors[ci]
        ax_stim.plot(d['t_rel'],  d['roll_pos'], color=col, lw=1.5,
                     label=f'{tilt_deg:.0f}° roll pos')
        ax_stim2.plot(d['t_rel'], d['hv_yaw'],  color=col, lw=1.0, ls='--', alpha=0.7)

    ax_stim.set_ylabel('Head roll orientation (deg)', fontsize=8, color='k')
    ax_stim2.set_ylabel('Head yaw velocity (deg/s)', fontsize=8, color='gray')
    ax_stim2.tick_params(axis='y', labelcolor='gray')
    ax_stim.set_title('Stimulus: roll orientation = pre-tilt (solid); yaw velocity = rotation epoch (dashed)',
                      fontsize=9)
    ax_stim.legend(fontsize=7, ncol=2, loc='center right')

    xlim = (-5.0, ROT_T + COAST_T)
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.axvline(0.0,   color='gray', lw=1.0, ls='-')
        ax.axvline(ROT_T, color='gray', lw=1.0, ls='--')
        ax.axhline(0.0,   color='k',   lw=0.4)
        ax.grid(True, alpha=0.15)

    axes[0].set_ylabel('SPV (deg/s)', fontsize=9); axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_title('Post-rotatory SPV decay: upright TC ≈ τ_vs; tilted TC shortened', fontsize=9)

    axes[1].set_ylabel('g_est[1] interaural (m/s²)', fontsize=9); axes[1].legend(fontsize=8, ncol=2)
    axes[1].set_ylim(-12, 3)
    axes[1].set_title('Gravity estimate: 0° → 0; 90° roll → −G₀', fontsize=9)

    axes[2].set_xlabel('Time relative to rotation onset (s)', fontsize=9)

    # Inset: TC vs tilt
    ax_ins = axes[0].inset_axes([0.72, 0.55, 0.25, 0.38])
    valid = [(α, τ) for α, τ in taus.items() if τ is not None]
    if valid:
        alphas, tau_vals = zip(*valid)
        ax_ins.plot(alphas, tau_vals, 'o-', color='k', lw=1.5, ms=5)
        ax_ins.set_xlabel('Tilt (°)', fontsize=7); ax_ins.set_ylabel('TC (s)', fontsize=7)
        ax_ins.tick_params(labelsize=6); ax_ins.set_title('TC vs tilt', fontsize=7)
        ax_ins.grid(True, alpha=0.2)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'gravity_tilt_suppression', show=show)
    return utils.fig_meta(path, rp,
        title='VOR Tilt Suppression',
        description=f'Post-rotatory decay for {TILTS_DEG}° pre-tilt. '
                    'Replicates Laurens & Angelaki (2011) Fig 6.',
        expected='TC decreases with tilt. 0°: TC ≈ τ_vs. 90°: TC ≈ τ_canal.',
        citation='Laurens & Angelaki (2011) Exp Brain Res',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────
# 4. OCR vs tilt angle
# ─────────────────────────────────────────────────────────────────────────────

def _ocr_vs_tilt(show):
    """Measure steady-state eye torsion for a range of static head roll tilts.

    Expected: torsion ≈ G_OCR × G0 × sin(tilt) (proportional to otolith signal).
    G_OCR = 10/9.81: 90° tilt → 10° torsion (~10% at all angles via sin).
    """
    TILTS_DEG = [5.0, 10.0, 20.0, 30.0, 45.0, 60.0]
    TILT_VEL  = 60.0
    HOLD_T    = 10.0

    params = with_brain(PARAMS_DEFAULT, g_ocr=G_OCR, g_burst=0.0)
    cmap   = plt.get_cmap('plasma')
    colors = [cmap(i / (len(TILTS_DEG) - 1)) for i in range(len(TILTS_DEG))]

    torsion_ss = []
    traces_t    = {}
    traces_eye  = {}
    traces_gest = {}
    traces_pos  = {}
    traces_vel  = {}

    for i, tilt_deg in enumerate(TILTS_DEG):
        tilt_dur = tilt_deg / TILT_VEL
        total    = tilt_dur + HOLD_T
        t        = np.arange(0.0, total, DT)
        hv_roll  = np.where(t < tilt_dur, TILT_VEL, 0.0)
        head_km  = km.build_kinematics(t, rot_vel=_pad3(hv_roll, 'roll'))

        st = simulate(params, t,
                      head=head_km,
                      target_present_array=np.zeros(len(t)),
                      sim_config=SimConfig(warmup_s=0.0),
                      return_states=True)

        eye_roll = (np.array(st.plant[:, 2]) + np.array(st.plant[:, 5])) / 2.0
        g_est    = np.array(st.brain[:, _IDX_GRAV])
        traces_t[tilt_deg]    = t
        traces_eye[tilt_deg]  = eye_roll
        traces_gest[tilt_deg] = g_est[:, 1]
        traces_pos[tilt_deg]  = head_km.rot_pos[:, 2]
        traces_vel[tilt_deg]  = hv_roll
        torsion_ss.append(float(eye_roll[-1]))

    # Positive roll = left ear down; compensatory OCR is positive (CW from front)
    # Expected: g_ocr × G0 × sin(tilt) (saturates at large angles like sin)
    torsion_expected = [G_OCR * G0 * np.sin(np.radians(d)) for d in TILTS_DEG]

    fig, axes = plt.subplots(4, 1, figsize=(11, 13))
    fig.suptitle(f'OCR vs Head Roll Tilt Angle  (g_ocr={G_OCR:.2f}, K_grav={K_GRAV})',
                 fontsize=12, fontweight='bold')

    ax0  = axes[0]
    ax0r = ax0.twinx()
    for i, tilt_deg in enumerate(TILTS_DEG):
        t = traces_t[tilt_deg]
        ax0.plot(t, traces_pos[tilt_deg], color=colors[i], lw=1.5, label=f'{tilt_deg:.0f}°')
        ax0r.plot(t, traces_vel[tilt_deg], color=colors[i], lw=0.8, ls='--', alpha=0.5)
    ax0.set_ylabel('Head roll position (deg)', fontsize=8)
    ax0r.set_ylabel('Head roll velocity (deg/s)', fontsize=8, color='gray')
    ax0r.tick_params(axis='y', labelcolor='gray')
    ax0.legend(fontsize=7, ncol=3, title='Tilt', title_fontsize=7)
    ax0.set_title('Stimulus: step-ramp roll tilts at 60°/s then hold', fontsize=9)
    ax0.grid(True, alpha=0.15); ax0.axhline(0, color='k', lw=0.4)
    ax0.set_xlim(0, traces_t[TILTS_DEG[-1]][-1])

    ax1 = axes[1]
    for i, tilt_deg in enumerate(TILTS_DEG):
        t_raw    = traces_t[tilt_deg]
        tilt_dur = tilt_deg / TILT_VEL
        t_hold   = t_raw - tilt_dur
        mask     = t_hold >= -tilt_dur
        ax1.plot(t_hold[mask], traces_eye[tilt_deg][mask], color=colors[i],
                 lw=1.5, label=f'{tilt_deg:.0f}°')
    ax1.axvline(0.0, color='gray', lw=1.0, ls='-')
    ax1.axhline(0.0, color='k', lw=0.4)
    ax_fmt(ax1, ylabel='Eye torsion (deg)')
    ax1.set_title('Eye counter-roll (aligned to tilt end; should settle to G_OCR×G0×sin(tilt))', fontsize=9)
    ax1.legend(fontsize=7, ncol=3)
    ax1.set_xlim(-1.5, HOLD_T)

    # g_est[1] — should settle to −G0·sin(tilt) and NOT drift thereafter
    ax_g = axes[2]
    for i, tilt_deg in enumerate(TILTS_DEG):
        t_raw    = traces_t[tilt_deg]
        tilt_dur = tilt_deg / TILT_VEL
        t_hold   = t_raw - tilt_dur
        expected_y = -G0 * np.sin(np.radians(tilt_deg))
        ax_g.plot(t_hold, traces_gest[tilt_deg], color=colors[i], lw=1.2, label=f'{tilt_deg:.0f}°')
        ax_g.axhline(expected_y, color=colors[i], lw=0.7, ls=':', alpha=0.6)
    ax_g.axvline(0.0, color='gray', lw=1.0, ls='-')
    ax_g.axhline(0.0, color='k', lw=0.4)
    ax_g.set_ylabel('g_est[1] interaural (m/s²)', fontsize=8)
    ax_g.set_title('Gravity estimate: settles to −G0·sin(tilt) and stays flat (dotted = expected)', fontsize=9)
    ax_g.legend(fontsize=7, ncol=3)
    ax_g.set_xlim(-1.5, HOLD_T)
    ax_g.set_ylim(-12, 1)
    ax_g.grid(True, alpha=0.15)
    ax_g.set_xlabel('Time rel. hold onset (s)', fontsize=8)

    ax2 = axes[3]
    ax2.scatter(TILTS_DEG, torsion_ss, color=[colors[i] for i in range(len(TILTS_DEG))],
                s=70, zorder=5)
    ax2.plot(TILTS_DEG, torsion_expected, color='tomato', lw=1.5, ls='--',
             label=f'Sin prediction: G_OCR×G0×sin(tilt)  (max {G_OCR*G0:.1f}° at 90°)')
    for i, (td, ts) in enumerate(zip(TILTS_DEG, torsion_ss)):
        ax2.annotate(f'{ts:.1f}°', (td, ts), fontsize=7,
                     xytext=(3, 4), textcoords='offset points')
    ax2.axhline(0.0, color='k', lw=0.4)
    ax2.set_xlabel('Head tilt (deg)', fontsize=9)
    ax2.set_ylabel('Steady-state torsion (deg)', fontsize=9)
    ax2.set_title('OCR main sequence: torsion vs tilt', fontsize=9)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'torsion_ocr_vs_tilt', show=show)
    return utils.fig_meta(path, rp,
        title='OCR vs Tilt Angle',
        description=f'Steady-state counter-roll for tilts {TILTS_DEG}°. g_ocr={G_OCR}.',
        expected=f'Sin torsion ≈ +{G_OCR}×G0×sin(tilt). 30° → ≈+{G_OCR*G0*np.sin(np.radians(30)):.1f}°.',
        citation='Boff, Kaufman & Thomas (1986); Tweed et al. (1994)',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────
# 5. Somatogravic OCR — frequency dependence of lateral translation
# ─────────────────────────────────────────────────────────────────────────────

def _somatogravic_frequency(show):
    """Sinusoidal left-right translation; torsion amplitude drops above ~0.1 Hz.

    Physics: GIA in head frame = gravity + lateral_acceleration.
    The gravity estimator low-pass filters GIA with TC = 1/K_grav ≈ 1.7 s.
    At low f: g_est tracks lateral GIA → perceived tilt → OCR.
    At high f: GIA fluctuates too fast for g_est → no perceived tilt → no OCR.

    Theoretical corner frequency: fc = K_grav / (2π) ≈ 0.095 Hz.
    """
    FREQS_HZ = [0.03, 0.07, 0.15, 0.3, 0.7, 1.5]
    A_ACCEL  = 2.0       # m/s² peak lateral acceleration (≈ 0.2g)
    N_CYCLES = 6
    SETTLE_S = 5.0 / K_GRAV

    params = with_brain(PARAMS_DEFAULT, g_ocr=G_OCR, g_burst=0.0)

    cmap   = plt.get_cmap('coolwarm')
    colors = [cmap(i / (len(FREQS_HZ) - 1)) for i in range(len(FREQS_HZ))]

    amp_model  = []
    SHOW_FREQS = [0.03, 0.15, 1.5]
    trace_data = {}

    for i, freq in enumerate(FREQS_HZ):
        omega   = 2.0 * np.pi * freq
        total   = SETTLE_S + N_CYCLES / freq
        t       = np.arange(0.0, total, DT)
        T       = len(t)

        a_x     = A_ACCEL * np.sin(omega * t)
        lin_acc = np.stack([a_x, np.zeros(T), np.zeros(T)], axis=1)

        st = simulate(params, t,
                      head=km.build_kinematics(t, lin_acc=lin_acc),
                      scene_present_array=np.zeros(T),
                      target_present_array=np.zeros(T),
                      return_states=True)

        eye_roll = (np.array(st.plant[:, 2]) + np.array(st.plant[:, 5])) / 2.0
        g_est    = np.array(st.brain[:, _IDX_GRAV])

        i_ss = int((total - 3.0 / freq) / DT)
        peak = float(np.max(np.abs(eye_roll[i_ss:])))
        amp_model.append(peak)

        if freq in SHOW_FREQS:
            trace_data[freq] = {
                't': t, 'a_x': a_x, 'eye_roll': eye_roll, 'g_est_1': g_est[:, 1],
                'i_ss': i_ss,
            }

    fc          = K_GRAV / (2.0 * np.pi)
    f_theory    = np.logspace(np.log10(0.01), np.log10(5.0), 200)
    gain_theory = 1.0 / np.sqrt(1.0 + (f_theory / fc) ** 2)
    torsion_dc  = G_OCR * A_ACCEL
    amp_theory  = gain_theory * torsion_dc

    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        f'Somatogravic OCR — Frequency Dependence of Lateral Translation\n'
        f'Constant {A_ACCEL:.0f} m/s² peak acceleration;  g_ocr={G_OCR},  '
        f'K_grav={K_GRAV}  (corner freq fc ≈ {fc:.3f} Hz)',
        fontsize=12, fontweight='bold')

    gs        = fig.add_gridspec(3, len(SHOW_FREQS), hspace=0.45, wspace=0.3,
                                 height_ratios=[1, 1, 1.3])
    axes_acc  = [fig.add_subplot(gs[0, j]) for j in range(len(SHOW_FREQS))]
    axes_gest = [fig.add_subplot(gs[1, j]) for j in range(len(SHOW_FREQS))]
    ax_bode   = fig.add_subplot(gs[2, :])

    show_colors = {f: cmap(FREQS_HZ.index(f) / (len(FREQS_HZ) - 1))
                   for f in SHOW_FREQS if f in FREQS_HZ}

    for j, freq in enumerate(SHOW_FREQS):
        if freq not in trace_data:
            continue
        d    = trace_data[freq]
        t    = d['t']
        i_ss = d['i_ss']
        col  = show_colors[freq]
        t_show = t[i_ss:] - t[i_ss]

        ax_acc = axes_acc[j]
        ax_acc.plot(t_show, d['a_x'][i_ss:], color=col, lw=1.2)
        ax_acc.axhline(0, color='k', lw=0.4)
        ax_acc.set_title(f'{freq:.2f} Hz', fontsize=9, fontweight='bold')
        if j == 0: ax_acc.set_ylabel('Head a_x (m/s²)', fontsize=8)
        ax_acc.grid(True, alpha=0.15)

        ax_ge = axes_gest[j]
        ax_ge.plot(t_show, d['g_est_1'][i_ss:], color=col, lw=1.2, label='g_est[1]')
        expected_ss = -d['a_x'][i_ss:] / np.sqrt(1.0 + (2 * np.pi * freq / K_GRAV) ** 2)
        ax_ge.plot(t_show, expected_ss, color='k', lw=0.8, ls='--', alpha=0.5,
                   label='LP theory')
        ax_ge.axhline(0, color='k', lw=0.4)
        if j == 0: ax_ge.set_ylabel('g_est[1] (m/s²)', fontsize=8)
        ax_ge.legend(fontsize=6); ax_ge.grid(True, alpha=0.15)
        if j == 1: ax_ge.set_title('Gravity estimate (LP-filtered GIA)', fontsize=8)

        ax_eye2 = ax_ge.twinx()
        ax_eye2.plot(t_show, d['eye_roll'][i_ss:], color='darkorange', lw=1.2, ls='-.')
        ax_eye2.set_ylabel('Eye torsion (deg)', fontsize=7, color='darkorange')
        ax_eye2.tick_params(axis='y', labelcolor='darkorange', labelsize=7)
        if j == 1:
            from matplotlib.lines import Line2D
            fig.text(0.5, 0.38, 'Orange dash-dot = eye torsion (right axis)',
                     ha='center', fontsize=7, color='darkorange')

    ax_bode.loglog(f_theory, amp_theory, color='gray', lw=2.0, ls='--',
                   label=f'LP theory: fc={fc:.3f} Hz  (TC=1/K_grav={1/K_GRAV:.1f} s)')
    ax_bode.axvline(fc, color='gray', lw=0.8, ls=':', alpha=0.7)
    ax_bode.text(fc * 1.15, amp_theory[0] * 0.6, f'fc={fc:.3f} Hz', fontsize=8, color='gray')

    for i, (freq, amp) in enumerate(zip(FREQS_HZ, amp_model)):
        col = cmap(i / (len(FREQS_HZ) - 1))
        ax_bode.scatter([freq], [amp], color=col, s=80, zorder=5)
        ax_bode.annotate(f'{freq:.2f} Hz\n{amp:.3f}°', (freq, amp),
                         fontsize=7, xytext=(5, 5), textcoords='offset points', color=col)

    ax_bode.set_xlabel('Frequency (Hz)', fontsize=9)
    ax_bode.set_ylabel('Peak torsion amplitude (deg)', fontsize=9)
    ax_bode.set_title(
        f'Somatogravic OCR amplitude vs frequency  '
        f'(A={A_ACCEL:.0f} m/s² lateral accel;  low-f → tilt percept → OCR; high-f → no tilt percept)',
        fontsize=9)
    ax_bode.legend(fontsize=8); ax_bode.grid(True, which='both', alpha=0.2)
    ax_bode.set_xlim(0.01, 5.0)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path, rp = utils.save_fig(fig, 'torsion_somatogravic_freq', show=show)
    return utils.fig_meta(path, rp,
        title='Somatogravic OCR — Frequency Dependence',
        description=f'Sinusoidal lateral translation at {FREQS_HZ} Hz, '
                    f'{A_ACCEL} m/s² peak. OCR via gravity estimator LP filter.',
        expected=f'Corner frequency fc = K_grav/(2π) ≈ {fc:.3f} Hz. '
                 f'OCR amplitude ∝ 1/√(1+(f/fc)²). '
                 f'Low f → peak ≈ {torsion_dc:.2f}°; high f → near 0°.',
        citation='Mayne (1974); Laurens & Angelaki (2011)',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────

def run(show=False):
    print('\n=== Gravity Estimator ===')
    figs = []
    print('  1/5  OCR …'); figs.append(_ocr(show))
    print('  2/5  OVAR (Fig 5) …'); figs.append(_ovar(show))
    print('  3/5  tilt suppression (Fig 6) …'); figs.append(_tilt_suppression(show))
    print('  4/5  OCR vs tilt angle …'); figs.append(_ocr_vs_tilt(show))
    print('  5/5  somatogravic OCR frequency dependence …')
    figs.append(_somatogravic_frequency(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
