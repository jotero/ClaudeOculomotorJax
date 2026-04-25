"""Tilt / Translation benchmarks — OCR, OVAR, VOR tilt suppression.

OCR: head rolls to a static tilt, driving the otolith → gravity estimator → torsional
eye response via g_ocr parameter.

OVAR: head rotates at constant velocity around an axis tilted from vertical.  The
canal signals a constant VOR drive; the otolith sees a sinusoidally changing gravity
direction, which modulates the slow-phase velocity via the gravity-dumping term K_gd.

VOR tilt suppression: gravity dumping (K_gd) selectively reduces velocity storage
for the component of head rotation perpendicular to gravity.  A 90° head tilt
shortens the post-rotary nystagmus TC from ~tau_vs to ~tau_canal.

NOTE: gravity estimator not yet validated — these are diagnostic figures showing
model output, not confirmed physiological matches.

Usage:
    python -X utf8 scripts/bench_tilt_translation.py
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

SHOW  = '--show' in sys.argv
DT    = 0.001
G0    = 9.81   # standard gravity (m/s²)
K_GD  = 0.5   # gravity dumping gain used for OVAR and tilt-suppression demos


SECTION = dict(
    id='tilt_translation', title='3. Tilt / Translation',
    description='Otolith-driven eye movements: OCR, OVAR, and VOR tilt suppression. '
                'OCR uses g_ocr=0.13 (13% physiological gain). '
                'OVAR and tilt suppression use K_gd=0.5 (gravity dumping). '
                'NOTE: gravity estimator not yet validated.',
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ocular Counterroll (OCR)
# ─────────────────────────────────────────────────────────────────────────────

def _ocr(show):
    """Ocular counterroll: head rolls 30°, measure torsional eye response."""
    TILT_DEG  = 30.0
    TILT_VEL  = 120.0
    TILT_T    = TILT_DEG / TILT_VEL
    HOLD_T    = 5.0
    TOTAL     = TILT_T + HOLD_T
    t         = np.arange(0.0, TOTAL, DT)

    hv_roll   = np.where(t < TILT_T, TILT_VEL, 0.0)
    head_vel  = np.stack([np.zeros_like(t), np.zeros_like(t), hv_roll], axis=1)
    head_roll_pos = np.cumsum(hv_roll) * DT

    params = with_brain(PARAMS_DEFAULT, g_ocr=0.13, g_burst=0.0)
    st     = simulate(params, t,
                      head=km.build_kinematics(t, rot_vel=head_vel),
                      target_present_array=np.zeros(len(t)),
                      return_states=True)

    eye_roll   = (np.array(st.plant[:, 2]) + np.array(st.plant[:, 5])) / 2.0
    g_est      = np.array(st.brain[:, _IDX_GRAV])
    g_est_y    = g_est[:, 1]
    ocr_ss     = -0.13 * TILT_DEG

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f'Ocular Counterroll (OCR) — {TILT_DEG:.0f}° Head Roll, g_ocr=0.13',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(t, head_roll_pos, color=utils.C['head'], lw=1.4)
    ax.axvline(TILT_T, color='gray', lw=0.8, linestyle=':')
    ax.axhline(TILT_DEG, color='gray', lw=0.8, linestyle='--', label=f'{TILT_DEG:.0f}° target')
    ax_fmt(ax, ylabel='Head roll (deg)')
    ax.set_ylim(-5, TILT_DEG + 8)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(t, eye_roll, color=utils.C['eye'], lw=1.4, label='Eye torsion (roll)')
    ax.axhline(ocr_ss, color='tomato', lw=0.9, linestyle='--',
               label=f'Expected OCR ≈ {ocr_ss:.1f}° (13% × {TILT_DEG:.0f}°)')
    ax.axhline(0.0, color='gray', lw=0.4, linestyle=':')
    ax.axvline(TILT_T, color='gray', lw=0.8, linestyle=':')
    ax_fmt(ax, ylabel='Eye roll (deg)')
    ax.set_ylim(-7, 4)
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.plot(t, g_est_y, color=utils.C['canal'], lw=1.2,
            label='g_est y (interaural, m/s²)')
    ax.axhline(G0 * np.sin(np.radians(TILT_DEG)), color='tomato', lw=0.9, linestyle='--',
               label=f'G₀ sin(30°) = {G0*np.sin(np.radians(TILT_DEG)):.2f} m/s²')
    ax.axvline(TILT_T, color='gray', lw=0.8, linestyle=':')
    ax_fmt(ax, ylabel='g_est y (m/s²)', xlabel='Time (s)')
    ax.set_ylim(-12, 12)
    ax.legend(fontsize=9)
    ax.set_title('Gravity estimator should settle to G₀ sin(30°) ≈ 4.9 m/s²', fontsize=9)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'tilt_ocr', show=show)
    return utils.fig_meta(
        path, rp,
        title='Ocular Counterroll (OCR)',
        description=f'Head rolls {TILT_DEG:.0f}° and holds. '
                    'Otolith → gravity estimator → torsional OCR via g_ocr=0.13.',
        expected=f'OCR ≈ {abs(ocr_ss):.1f}° counter-rolling. '
                 f'Settling TC ≈ 2 s (1/K_grav). '
                 f'g_est_y → G₀ sin(30°) ≈ 4.9 m/s².',
        citation='Boff, Kaufman & Thomas (1986); Tweed et al. (1994)',
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. OVAR — Off-Vertical Axis Rotation
# ─────────────────────────────────────────────────────────────────────────────

def _ovar(show):
    """OVAR: sinusoidal SPV modulation during rotation around a tilted axis."""
    TILT_DEG = 20.0    # tilt angle from vertical (°)
    SPIN_VEL = 60.0    # rotation speed (°/s); period = 6 s
    TOTAL    = 40.0

    alpha   = np.radians(TILT_DEG)
    omega_x = SPIN_VEL * np.cos(alpha)   # yaw  component (°/s)
    omega_z = SPIN_VEL * np.sin(alpha)   # roll component (°/s)
    period  = 360.0 / SPIN_VEL           # 6 s

    t = np.arange(0.0, TOTAL, DT)
    T = len(t)
    head_vel = np.stack([
        np.full(T, omega_x),
        np.zeros(T),
        np.full(T, omega_z),
    ], axis=1)

    # K_gd links gravity estimate to VS dumping; no saccades; dark room; no target
    params = with_brain(PARAMS_DEFAULT, K_gd=K_GD, g_burst=0.0)
    st     = simulate(params, t,
                      head=km.build_kinematics(t, rot_vel=head_vel),
                      target_present_array=np.zeros(T),
                      return_states=True)

    eye_yaw_pos = (np.array(st.plant[:, 0]) + np.array(st.plant[:, 3])) / 2.0
    eye_vel_yaw = np.gradient(eye_yaw_pos, DT)
    vs_yaw      = vs_net(st)[:, 0]
    g_est_y     = np.array(st.brain[:, _IDX_GRAV])[:, 1]

    period_marks = np.arange(period, TOTAL + 0.01, period)
    expected_amp = G0 * np.sin(np.radians(TILT_DEG))

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f'OVAR — {TILT_DEG:.0f}° Tilt, {SPIN_VEL:.0f}°/s Rotation, K_gd={K_GD}  '
        f'(period = {period:.0f} s)',
        fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(t, eye_vel_yaw, color=utils.C['eye'], lw=0.8)
    for pm in period_marks:
        ax.axvline(pm, color='gray', lw=0.5, linestyle=':')
    ax_fmt(ax, ylabel='Eye yaw vel (deg/s)')
    ax.set_ylim(-80, 80)
    ax.set_title('Expected: sinusoidal SPV modulation at rotation period', fontsize=9)

    ax = axes[1]
    ax.plot(t, vs_yaw, color=utils.C['vs'], lw=1.2, label='VS net yaw')
    for pm in period_marks:
        ax.axvline(pm, color='gray', lw=0.5, linestyle=':')
    ax_fmt(ax, ylabel='VS net yaw (deg/s)')
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.plot(t, g_est_y, color=utils.C['canal'], lw=1.2, label='g_est y (interaural)')
    ax.axhline( expected_amp, color='tomato', lw=0.8, linestyle='--',
               label=f'±G₀ sin({TILT_DEG:.0f}°) = ±{expected_amp:.1f} m/s²')
    ax.axhline(-expected_amp, color='tomato', lw=0.8, linestyle='--')
    for pm in period_marks:
        ax.axvline(pm, color='gray', lw=0.5, linestyle=':')
    ax_fmt(ax, ylabel='g_est y (m/s²)', xlabel='Time (s)')
    ax.set_ylim(-12, 12)
    ax.legend(fontsize=9)
    ax.set_title('Gravity in head frame oscillates → drives SPV modulation via K_gd', fontsize=9)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'tilt_ovar', show=show)
    return utils.fig_meta(
        path, rp,
        title='OVAR — Off-Vertical Axis Rotation',
        description=f'Constant {SPIN_VEL:.0f}°/s rotation around axis tilted {TILT_DEG:.0f}° from vertical. '
                    f'Canal signals constant velocity; otolith sees sinusoidal gravity direction. '
                    f'K_gd={K_GD} links gravity estimate to VS dumping.',
        expected=f'Sinusoidal SPV modulation at period = {period:.0f} s. '
                 f'g_est y oscillates between ±{expected_amp:.1f} m/s². '
                 'NOTE: gravity estimator not yet validated.',
        citation='Raphan et al. (1981); Angelaki & Hess (1994) J Neurophysiol',
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. VOR Tilt Suppression
# ─────────────────────────────────────────────────────────────────────────────

def _tilt_suppression(show):
    """Gravity dumping shortens post-rotary TC when head is tilted 90° in roll.

    With head upright, yaw rotation is parallel to gravity → K_gd has no effect
    → post-rotary TC ≈ tau_vs (extended by canal feedthrough).

    With head tilted 90° in roll, yaw rotation is perpendicular to gravity → full
    gravity dumping → VS yaw component drains rapidly → post-rotary TC shortened.

    Uses slow-phase velocity (SPV) with saccades on, matching bench_vor_okr methodology.
    """
    ROT_VEL  = 60.0   # yaw rotation velocity (°/s)
    ROT_T    = 20.0   # rotation duration (s); >> tau_canal=5s so canal fully adapts
    COAST_T  = 60.0   # post-rotary coast (s); long enough to see tau_vs=20s decay
    BASE_T   = 3.0    # stationary baseline before rotation (s)
    TILT_T   = 1.0    # roll to 90° at 90°/s
    SETTLE_T = 6.0    # gravity estimator settling; 3×TC_grav ≈ 6s
    T_pre    = TILT_T + SETTLE_T

    # Saccades ON (eye stays near center → reliable SPV); dark room (no OKR contamination)
    params = with_brain(PARAMS_DEFAULT, K_gd=K_GD)
    cfg    = SimConfig(warmup_s=0.0)

    # ── Upright ───────────────────────────────────────────────────────────────
    t_u  = np.arange(0.0, BASE_T + ROT_T + COAST_T, DT)
    T_u  = len(t_u)
    hv_u = np.where((t_u >= BASE_T) & (t_u < BASE_T + ROT_T), ROT_VEL, 0.0)
    st_u = simulate(params, t_u,
                    head=km.build_kinematics(t_u, rot_vel=km._pad3(hv_u, 'yaw')),
                    scene_present_array=np.zeros(T_u),
                    target_present_array=np.zeros(T_u),
                    sim_config=cfg, return_states=True)

    ev_u    = np.gradient((np.array(st_u.plant[:, 0]) + np.array(st_u.plant[:, 3])) / 2.0, DT)
    burst_u = np.array(extract_burst(st_u, params)[:, 0])
    spv_u   = extract_spv(t_u, ev_u, burst_u)
    t_rel_u = t_u - BASE_T

    # ── 90° Roll Tilt ─────────────────────────────────────────────────────────
    t_tilt = np.arange(0.0, T_pre + ROT_T + COAST_T, DT)
    T_tilt = len(t_tilt)
    hv_roll = np.where(t_tilt < TILT_T, 90.0, 0.0)
    hv_yaw  = np.where((t_tilt >= T_pre) & (t_tilt < T_pre + ROT_T), ROT_VEL, 0.0)
    hv_3d   = np.stack([hv_yaw, np.zeros_like(t_tilt), hv_roll], axis=1)
    st_tilt = simulate(params, t_tilt,
                       head=km.build_kinematics(t_tilt, rot_vel=hv_3d),
                       scene_present_array=np.zeros(T_tilt),
                       target_present_array=np.zeros(T_tilt),
                       sim_config=cfg, return_states=True)

    ev_tilt    = np.gradient((np.array(st_tilt.plant[:, 0]) + np.array(st_tilt.plant[:, 3])) / 2.0, DT)
    burst_tilt = np.array(extract_burst(st_tilt, params)[:, 0])
    spv_tilt   = extract_spv(t_tilt, ev_tilt, burst_tilt)
    t_rel_tilt = t_tilt - T_pre

    g_est_tilt = np.array(st_tilt.brain[:, _IDX_GRAV])[:, 1]

    # ── TC fits on post-rotary SPV ────────────────────────────────────────────
    tau_u,    t_fit_u,    y_fit_u    = fit_tc(t_rel_u,    spv_u,    ROT_T, ROT_T + COAST_T,
                                               label='Upright post-rot TC')
    tau_tilt, t_fit_tilt, y_fit_tilt = fit_tc(t_rel_tilt, spv_tilt, ROT_T, ROT_T + COAST_T,
                                               label='Tilted post-rot TC')

    tau_vs_default = float(PARAMS_DEFAULT.brain.tau_vs)
    tau_c          = float(PARAMS_DEFAULT.sensory.tau_c)

    xlim = (-BASE_T, ROT_T + COAST_T)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(f'VOR Tilt Suppression: Upright vs. 90° Roll (K_gd={K_GD})',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(t_rel_u,    spv_u,    color=utils.C['eye'],    lw=1.2, label='Upright SPV')
    ax.plot(t_rel_tilt, spv_tilt, color=utils.C['target'], lw=1.2, linestyle='--',
            label='90° tilted SPV')
    if t_fit_u is not None:
        ax.plot(t_fit_u, y_fit_u, color=utils.C['eye'], lw=2.5, linestyle=':', alpha=0.9,
                label=f'τ_upright  = {tau_u:.1f} s')
    if t_fit_tilt is not None:
        ax.plot(t_fit_tilt, y_fit_tilt, color=utils.C['target'], lw=2.5, linestyle=':', alpha=0.9,
                label=f'τ_tilted = {tau_tilt:.1f} s')
    ax.axvline(0.0,   color='gray', lw=1.0, linestyle='-')
    ax.axvline(ROT_T, color='gray', lw=1.0, linestyle='--')
    ax_fmt(ax, ylabel='Slow-phase vel (deg/s)')
    ax.set_xlim(*xlim)
    ax.legend(fontsize=9, ncol=2)
    ax.set_title(
        'Upright: VOR extended by VS (τ ≈ τ_vs). '
        'Tilted: gravity ⊥ yaw axis → VS drained → shorter post-rot TC',
        fontsize=9)

    ax = axes[1]
    ax.plot(t_rel_tilt, g_est_tilt, color=utils.C['canal'], lw=1.2,
            label='g_est y — tilted condition')
    ax.axhline(-G0, color='tomato', lw=0.9, linestyle='--',
               label=f'Expected settled: −G₀ = −{G0:.1f} m/s²')
    ax.axhline(0.0, color='gray', lw=0.4, linestyle=':')
    ax.axvline(0.0,   color='gray', lw=1.0, linestyle='-')
    ax.axvline(ROT_T, color='gray', lw=1.0, linestyle='--')
    ax_fmt(ax, ylabel='g_est y (m/s²)', xlabel='Time rel. rotation onset (s)')
    ax.set_xlim(*xlim)
    ax.set_ylim(-12, 12)
    ax.legend(fontsize=9)
    ax.set_title('Gravity estimator in tilted condition: should converge to −G₀ ≈ −9.8 m/s²',
                 fontsize=9)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'tilt_vor_suppression', show=show)
    return utils.fig_meta(
        path, rp,
        title='VOR Tilt Suppression',
        description=f'Upright vs. 90° roll-tilted yaw VOR decay (K_gd={K_GD}, '
                    f'{ROT_T:.0f} s rotation at {ROT_VEL:.0f}°/s). '
                    'Gravity dumping selectively drains VS for motion perpendicular to gravity.',
        expected=f'τ_upright ≈ {tau_vs_default:.0f}–35 s (VS extends canal TC). '
                 f'τ_tilted: shortened by gravity dumping (τ_eff = τ_vs/(1+K_gd·τ_vs)). '
                 'NOTE: gravity estimator not yet validated.',
        citation='Raphan et al. (1981) J Neurophysiol; Cohen et al. (1983)',
    )


# ─────────────────────────────────────────────────────────────────────────────

def run(show=False):
    print('\n=== Tilt / Translation ===')
    figs = []
    print('  1/3  ocular counterroll (OCR) …')
    figs.append(_ocr(show))
    print('  2/3  OVAR …')
    figs.append(_ovar(show))
    print('  3/3  VOR tilt suppression …')
    figs.append(_tilt_suppression(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
