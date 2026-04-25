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

# Laurens & Angelaki (2011) parameters
K_GD   = 0.5    # gravity dumping gain for OVAR and tilt suppression
K_GRAV = 0.6    # go — already the default in BrainParams


SECTION = dict(
    id='gravity', title='4. Gravity Estimator',
    description='Otolith-canal interaction following Laurens & Angelaki (2011). '
                'OCR, OVAR (Fig 5), VOR tilt suppression (Fig 6). '
                'K_grav=0.6 (go), K_gd=0.5.',
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _pad3(v1d, axis):
    T = len(v1d)
    out = np.zeros((T, 3))
    idx = {'yaw': 0, 'pitch': 1, 'roll': 2}[axis]
    out[:, idx] = v1d
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ocular Counterroll (OCR)
# ─────────────────────────────────────────────────────────────────────────────

def _ocr(show):
    """Head rolls 30°; eyes counter-roll ~13% of tilt angle."""
    TILT_DEG = 30.0
    TILT_VEL = 60.0          # °/s
    TILT_T   = TILT_DEG / TILT_VEL
    HOLD_T   = 8.0
    TOTAL    = TILT_T + HOLD_T
    t        = np.arange(0.0, TOTAL, DT)

    hv_roll  = np.where(t < TILT_T, TILT_VEL, 0.0)
    head_vel = _pad3(hv_roll, 'roll')
    head_pos = np.cumsum(hv_roll) * DT   # (T,) roll position

    params = with_brain(PARAMS_DEFAULT, g_ocr=0.13, g_burst=0.0)
    st     = simulate(params, t,
                      head=km.build_kinematics(t, rot_vel=head_vel),
                      target_present_array=np.zeros(len(t)),
                      return_states=True)

    eye_roll = (np.array(st.plant[:, 2]) + np.array(st.plant[:, 5])) / 2.0
    g_est    = np.array(st.brain[:, _IDX_GRAV])
    ocr_ss   = -0.13 * TILT_DEG

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f'Ocular Counterroll — {TILT_DEG:.0f}° Head Roll  '
                 r'(g_ocr=0.13, K_grav=0.6)',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(t, head_pos, color=utils.C['head'], lw=1.5, label='Head roll (deg)')
    ax.axhline(TILT_DEG, color='gray', lw=0.8, ls='--', label=f'Target {TILT_DEG:.0f}°')
    ax.axvline(TILT_T, color='gray', lw=0.7, ls=':')
    ax_fmt(ax, ylabel='Head roll (deg)')
    ax.set_ylim(-5, TILT_DEG + 10); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t, eye_roll, color=utils.C['eye'], lw=1.5, label='Eye torsion')
    ax.axhline(ocr_ss, color='tomato', lw=1.0, ls='--',
               label=f'Expected OCR ≈ {ocr_ss:.1f}° (13%)')
    ax.axhline(0.0, color='k', lw=0.4)
    ax.axvline(TILT_T, color='gray', lw=0.7, ls=':')
    ax_fmt(ax, ylabel='Eye roll (deg)')
    ax.set_ylim(-7, 4); ax.legend(fontsize=8)

    ax = axes[2]
    expected_y = -G0 * np.sin(np.radians(TILT_DEG))
    ax.plot(t, g_est[:, 1], color=utils.C['canal'], lw=1.2, label='g_est[1] (interaural)')
    ax.axhline(expected_y, color='tomato', lw=0.9, ls='--',
               label=f'Expected = −G₀ sin(30°) = {expected_y:.2f} m/s²')
    ax.axhline(0.0, color='k', lw=0.4)
    ax.axvline(TILT_T, color='gray', lw=0.7, ls=':')
    ax_fmt(ax, ylabel='g_est[1] (m/s²)', xlabel='Time (s)')
    ax.set_ylim(-12, 3); ax.legend(fontsize=8)
    ax.set_title('Gravity estimate should settle to −G₀ sin(30°) ≈ −4.9 m/s²', fontsize=9)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'gravity_ocr', show=show)
    return utils.fig_meta(path, rp,
        title='Ocular Counterroll (OCR)',
        description=f'Head rolls {TILT_DEG:.0f}°; g_ocr=0.13 drives torsional eye response. '
                    'Laurens & Angelaki (2011) K_grav=0.6.',
        expected=f'Eye counter-rolls to ≈{abs(ocr_ss):.1f}° after {1/K_GRAV:.1f} s TC. '
                 f'g_est[1] converges to ≈{expected_y:.1f} m/s².',
        citation='Laurens & Angelaki (2011) Exp Brain Res; Boff et al. (1986)',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────
# 2. OVAR — Off-Vertical Axis Rotation  (replicates Laurens Fig 5)
# ─────────────────────────────────────────────────────────────────────────────

def _ovar(show):
    """Sinusoidal SPV modulation during rotation around a tilted axis.

    Laurens & Angelaki Fig 5: steady-state eye velocity during OVAR has a
    sinusoidal modulation at the rotation period. Amplitude of modulation
    scales with sin(tilt_angle).

    Three tilt angles: 10°, 30°, 60°.  One rotation speed: 60 °/s.
    """
    SPIN_VEL  = 60.0    # °/s
    TOTAL     = 60.0    # s — needs multiple periods to reach steady state
    TILTS_DEG = [10.0, 30.0, 60.0]

    params = with_brain(PARAMS_DEFAULT, K_gd=K_GD, g_burst=700.0)
    cfg    = SimConfig(warmup_s=3.0)

    t   = np.arange(0.0, TOTAL, DT)
    T   = len(t)

    colors = ['#1b7837', '#762a83', '#e08214']

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f'OVAR — Off-Vertical Axis Rotation  (Laurens & Angelaki 2011, Fig 5)\n'
        f'{SPIN_VEL:.0f} °/s rotation, K_gd={K_GD}, K_grav={K_GRAV}',
        fontsize=12, fontweight='bold')

    for ci, tilt_deg in enumerate(TILTS_DEG):
        alpha    = np.radians(tilt_deg)
        omega_y  = SPIN_VEL * np.cos(alpha)   # yaw component
        omega_z  = SPIN_VEL * np.sin(alpha)   # roll component
        period   = 360.0 / SPIN_VEL           # s per revolution

        head_vel = np.stack([
            np.full(T, omega_y),
            np.zeros(T),
            np.full(T, omega_z),
        ], axis=1)

        st       = simulate(params, t,
                            head=km.build_kinematics(t, rot_vel=head_vel),
                            scene_present_array=np.zeros(T),
                            target_present_array=np.zeros(T),
                            sim_config=cfg, return_states=True)

        eye_pos  = (np.array(st.plant[:, 0]) + np.array(st.plant[:, 3])) / 2.0
        eye_vel  = np.gradient(eye_pos, DT)
        burst    = np.array(extract_burst(st, params)[:, 0])
        spv      = extract_spv(t, eye_vel, burst)
        g_est    = np.array(st.brain[:, _IDX_GRAV])

        lbl  = f'{tilt_deg:.0f}° tilt'
        col  = colors[ci]
        period_marks = np.arange(period, TOTAL, period)

        axes[0].plot(t, spv, color=col, lw=1.2, alpha=0.85, label=lbl)
        axes[1].plot(t, g_est[:, 1], color=col, lw=1.2, label=lbl)
        axes[2].plot(t, vs_net(st)[:, 0], color=col, lw=1.2, label=lbl)

    # Reference lines and period markers
    for ax in axes:
        for pm in np.arange(period, TOTAL, period):
            ax.axvline(pm, color='gray', lw=0.4, ls=':', alpha=0.5)
        ax.grid(True, alpha=0.15)

    amp_ref = G0 * np.sin(np.radians(TILTS_DEG[-1]))
    axes[0].set_ylabel('Eye SPV (deg/s)', fontsize=9)
    axes[0].set_title('SPV should be sinusoidally modulated at rotation period', fontsize=9)
    axes[0].legend(fontsize=8, loc='upper right'); axes[0].axhline(0, color='k', lw=0.4)

    axes[1].set_ylabel('g_est[1] interaural (m/s²)', fontsize=9)
    axes[1].set_title(
        f'Gravity estimate oscillates ±G₀ sin(α); '
        f'largest tilt: ±{amp_ref:.1f} m/s²', fontsize=9)
    axes[1].legend(fontsize=8, loc='upper right'); axes[1].axhline(0, color='k', lw=0.4)
    axes[1].set_ylim(-12, 12)

    axes[2].set_ylabel('VS net yaw (deg/s)', fontsize=9)
    axes[2].set_title('VS output modulated by gravity dumping (K_gd)', fontsize=9)
    axes[2].legend(fontsize=8, loc='upper right'); axes[2].axhline(0, color='k', lw=0.4)
    axes[2].set_xlabel('Time (s)', fontsize=9)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'gravity_ovar', show=show)
    return utils.fig_meta(path, rp,
        title='OVAR — Off-Vertical Axis Rotation',
        description=f'{SPIN_VEL:.0f}°/s rotation around axes tilted {TILTS_DEG}° from vertical. '
                    'Replicates Laurens & Angelaki (2011) Fig 5.',
        expected='SPV sinusoidally modulated at rotation period. '
                 'Modulation amplitude ∝ sin(tilt angle). '
                 'g_est oscillates ±G₀ sin(α). '
                 'Larger tilt → deeper SPV modulation.',
        citation='Laurens & Angelaki (2011) Exp Brain Res 210:407; '
                 'Raphan et al. (1981) J Neurophysiol',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────
# 3. VOR Tilt Suppression  (replicates Laurens Fig 6)
# ─────────────────────────────────────────────────────────────────────────────

def _tilt_suppression(show):
    """Post-rotary nystagmus TC shortened when head is tilted 90° in roll.

    Laurens & Angelaki Fig 6: gravity dumping (K_gd) selectively damps the
    velocity storage component perpendicular to gravity.  With head upright,
    yaw rotation is parallel to gravity → K_gd has no effect → TC ≈ tau_vs.
    With head 90° tilted, yaw is perpendicular to gravity → full dumping.

    Multiple tilt angles: 0°, 30°, 60°, 90° roll before rotation.
    """
    ROT_VEL   = 60.0    # °/s
    ROT_T     = 20.0    # s — long enough for VS to reach steady state
    COAST_T   = 60.0    # s post-rotatory observation window
    SETTLE_T  = 8.0     # s for gravity estimator to settle after tilt
    TILTS_DEG = [0.0, 30.0, 60.0, 90.0]

    params = with_brain(PARAMS_DEFAULT, K_gd=K_GD)
    cfg    = SimConfig(warmup_s=0.0)

    colors = ['#1b7837', '#5aae61', '#d9f0d3'[::-1], '#762a83']
    colors = ['steelblue', '#2196a8', '#e08214', '#c62e2e']

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
    fig.suptitle(
        f'VOR Tilt Suppression  (Laurens & Angelaki 2011, Fig 6)\n'
        f'{ROT_VEL:.0f}°/s yaw rotation, {ROT_T:.0f} s;  K_gd={K_GD}, K_grav={K_GRAV}',
        fontsize=12, fontweight='bold')

    taus = {}
    all_t_rel   = {}
    all_spv     = {}
    all_g_est_y = {}

    for ci, tilt_deg in enumerate(TILTS_DEG):
        alpha = np.radians(tilt_deg)

        if tilt_deg == 0.0:
            # No tilt: direct yaw rotation
            t_arr  = np.arange(0.0, ROT_T + COAST_T, DT)
            T      = len(t_arr)
            hv_yaw = np.where(t_arr < ROT_T, ROT_VEL, 0.0)
            hv_3d  = _pad3(hv_yaw, 'yaw')
            t_rot_start = 0.0
        else:
            # Pre-tilt: roll to tilt_deg at 60°/s, settle SETTLE_T, then rotate
            tilt_vel = 60.0
            tilt_dur = tilt_deg / tilt_vel
            t_total  = tilt_dur + SETTLE_T + ROT_T + COAST_T
            t_arr    = np.arange(0.0, t_total, DT)
            T        = len(t_arr)
            t_rot_start = tilt_dur + SETTLE_T

            hv_roll = np.where(t_arr < tilt_dur, tilt_vel, 0.0)
            hv_yaw  = np.where(
                (t_arr >= t_rot_start) & (t_arr < t_rot_start + ROT_T),
                ROT_VEL, 0.0)
            hv_3d   = np.stack([hv_yaw, np.zeros(T), hv_roll], axis=1)

        st = simulate(params, t_arr,
                      head=km.build_kinematics(t_arr, rot_vel=hv_3d),
                      scene_present_array=np.zeros(T),
                      target_present_array=np.zeros(T),
                      sim_config=cfg, return_states=True)

        eye_pos = (np.array(st.plant[:, 0]) + np.array(st.plant[:, 3])) / 2.0
        eye_vel = np.gradient(eye_pos, DT)
        burst   = np.array(extract_burst(st, params)[:, 0])
        spv     = extract_spv(t_arr, eye_vel, burst)
        g_est_y = np.array(st.brain[:, _IDX_GRAV])[:, 1]

        t_rel = t_arr - t_rot_start   # zero at rotation onset
        all_t_rel[tilt_deg]   = t_rel
        all_spv[tilt_deg]     = spv
        all_g_est_y[tilt_deg] = g_est_y

        tau, t_fit, y_fit = fit_tc(t_rel, spv, ROT_T, ROT_T + COAST_T)
        taus[tilt_deg] = tau
        col = colors[ci]
        lbl = f'{tilt_deg:.0f}° roll' + (f'  τ={tau:.1f} s' if tau else '')

        # Plot on axes[0] (SPV) and axes[1] (g_est)
        xlim = (-5, ROT_T + COAST_T)
        axes[0].plot(t_rel, spv, color=col, lw=1.2, label=lbl)
        if t_fit is not None:
            axes[0].plot(t_fit, y_fit, color=col, lw=2.5, ls=':', alpha=0.9)
        axes[1].plot(t_rel, g_est_y, color=col, lw=1.2, label=f'{tilt_deg:.0f}°')

    # ── Stimulus panel (axes[2]): head velocity for 0° and 90° tilt ────────
    for ci, tilt_deg in enumerate([0.0, 90.0]):
        t_rel = all_t_rel[tilt_deg]
        col   = colors[TILTS_DEG.index(tilt_deg)]
        # Reconstruct head vel from t_rel
        if tilt_deg == 0.0:
            t_arr = np.arange(0.0, ROT_T + COAST_T, DT)
            hv    = np.where(t_arr < ROT_T, ROT_VEL, 0.0)
        else:
            tilt_dur = tilt_deg / 60.0
            t_total  = tilt_dur + SETTLE_T + ROT_T + COAST_T
            t_arr    = np.arange(0.0, t_total, DT)
            t_rot_start = tilt_dur + SETTLE_T
            hv       = np.where(
                (t_arr >= t_rot_start) & (t_arr < t_rot_start + ROT_T),
                ROT_VEL, 0.0)
        t_rel2 = t_arr - (0.0 if tilt_deg == 0.0 else tilt_dur + SETTLE_T)
        axes[2].plot(t_rel2, hv, color=col, lw=1.5, label=f'{tilt_deg:.0f}° roll — head yaw vel')

    xlim = (-5.0, ROT_T + COAST_T)
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.axvline(0.0,   color='gray', lw=1.0, ls='-')
        ax.axvline(ROT_T, color='gray', lw=1.0, ls='--')
        ax.axhline(0.0,   color='k',   lw=0.4)
        ax.grid(True, alpha=0.15)

    axes[0].set_ylabel('SPV (deg/s)', fontsize=9)
    axes[0].legend(fontsize=8, ncol=2, loc='upper right')
    axes[0].set_title(
        'Post-rotatory SPV decay: upright TC ≈ τ_vs; '
        '90° roll TC shortened by gravity dumping', fontsize=9)

    axes[1].set_ylabel('g_est[1] interaural (m/s²)', fontsize=9)
    axes[1].legend(fontsize=8, ncol=2, loc='lower right')
    axes[1].set_ylim(-12, 3)
    axes[1].set_title(
        'Gravity estimate: 0° stays at 0; 90° roll → g_est[1] converges to −G₀', fontsize=9)

    axes[2].set_ylabel('Head yaw vel (deg/s)', fontsize=9)
    axes[2].set_xlabel('Time relative to rotation onset (s)', fontsize=9)
    axes[2].legend(fontsize=8)
    axes[2].set_title('Stimulus: yaw rotation (vertical bar = onset, dashed = offset)', fontsize=9)

    # Inset: TC vs tilt angle
    ax_ins = axes[0].inset_axes([0.72, 0.55, 0.25, 0.38])
    valid  = [(α, τ) for α, τ in taus.items() if τ is not None]
    if valid:
        alphas, tau_vals = zip(*valid)
        ax_ins.plot(alphas, tau_vals, 'o-', color='k', lw=1.5, ms=5)
        ax_ins.set_xlabel('Tilt (°)', fontsize=7)
        ax_ins.set_ylabel('TC (s)', fontsize=7)
        ax_ins.tick_params(labelsize=6)
        ax_ins.set_title('TC vs tilt', fontsize=7)
        ax_ins.grid(True, alpha=0.2)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'gravity_tilt_suppression', show=show)
    return utils.fig_meta(path, rp,
        title='VOR Tilt Suppression',
        description=f'Post-rotatory decay for {TILTS_DEG}° pre-tilt conditions. '
                    'Replicates Laurens & Angelaki (2011) Fig 6.',
        expected='TC decreases monotonically with tilt angle. '
                 f'0° tilt: TC ≈ tau_vs = {PARAMS_DEFAULT.brain.tau_vs:.0f} s. '
                 '90° tilt: TC ≈ tau_canal (gravity dumping drains VS). '
                 'g_est[1] converges to −G₀ for 90° tilted.',
        citation='Laurens & Angelaki (2011) Exp Brain Res; Raphan et al. (1981)',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────

SECTION = dict(
    id='gravity', title='4. Gravity Estimator',
    description='Canal-otolith interaction: OCR, OVAR, VOR tilt suppression. '
                'Parameters: K_grav=0.6, K_gd=0.5 (Laurens & Angelaki 2011).',
)


def run(show=False):
    print('\n=== Gravity Estimator ===')
    figs = []
    print('  1/3  ocular counterroll (OCR) …')
    figs.append(_ocr(show))
    print('  2/3  OVAR (Fig 5) …')
    figs.append(_ovar(show))
    print('  3/3  tilt suppression (Fig 6) …')
    figs.append(_tilt_suppression(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
