"""Listing's law benchmarks.

Verifies that the saccade generator enforces the torsional constraint:

    T_required = OCR  −  H · V · π/360   (rotation-vector deg)

Two figures:
    listing_plane  — Listing's plane: scatter of T_final vs expected for a grid
                     of oblique saccade targets, plus cross-section plot.
    listing_ocr    — OCR-driven torsional correction: after a head tilt, a
                     torsional saccade fires to bring eye torsion to OCR.

Usage:
    python -X utf8 scripts/bench_listing.py
    python -X utf8 scripts/bench_listing.py --show
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as utils

import numpy as np
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, simulate, SimConfig, _IDX_GRAV,
)
from oculomotor.sim import kinematics as km
from oculomotor.analysis import ax_fmt, extract_burst
from oculomotor.models.brain_models.listing import HALF_ANGLE

SHOW = '--show' in sys.argv
DT   = 0.001
G0   = 9.81
G_OCR = 10.0 / G0   # same calibration as bench_gravity

SECTION = dict(
    id='listing', title="5. Listing's Law",
    description="Torsional constraint T = OCR − H·V·π/360: saccade plane scatter "
                "and OCR-driven correction saccade.",
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Listing's plane — oblique saccades
# ─────────────────────────────────────────────────────────────────────────────

def _listing_plane(show):
    """Simulate saccades to a (H, V) grid; compare final torsion to theory."""

    H_VALS = np.array([-20.0, -10.0, 0.0, 10.0, 20.0, 30.0])
    V_VALS = np.array([0.0, 10.0, 20.0, 30.0])
    TOTAL  = 1.5        # 1.5 s is enough for saccade + listing correction

    params  = with_brain(PARAMS_DEFAULT, g_burst=700.0, g_ocr=0.0)
    cfg     = SimConfig(warmup_s=0.0)
    t       = np.arange(0.0, TOTAL, DT)
    T_arr   = len(t)
    head_km = km.build_kinematics(t)

    results = []   # list of (H, V, T_final, T_expected)

    # One representative trace for the time-series panel
    trace_H, trace_V = 20.0, 20.0
    trace_data = None

    for H in H_VALS:
        for V in V_VALS:
            tgt = km.build_target(t,
                                  yaw_deg=np.full(T_arr, H),
                                  pitch_deg=np.full(T_arr, V))
            st = simulate(params, t,
                          head=head_km,
                          target=tgt,
                          target_present_array=np.ones(T_arr),
                          scene_present_array=np.zeros(T_arr),
                          sim_config=cfg, return_states=True)

            eye_h    = (np.array(st.plant[:, 0]) + np.array(st.plant[:, 3])) / 2.0
            eye_v    = (np.array(st.plant[:, 1]) + np.array(st.plant[:, 4])) / 2.0
            eye_roll = (np.array(st.plant[:, 2]) + np.array(st.plant[:, 5])) / 2.0

            T_final = float(np.mean(eye_roll[-int(0.2/DT):]))
            T_exp   = float(-H * V * HALF_ANGLE)
            results.append((H, V, T_final, T_exp))

            if H == trace_H and V == trace_V:
                trace_data = dict(t=t, eye_h=eye_h, eye_v=eye_v, eye_roll=eye_roll,
                                  T_exp=T_exp, H=H, V=V)

    H_arr  = np.array([r[0] for r in results])
    V_arr  = np.array([r[1] for r in results])
    Tf_arr = np.array([r[2] for r in results])
    Te_arr = np.array([r[3] for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Listing's Law — Torsion After Oblique Saccades  (g_ocr=0)",
                 fontsize=12, fontweight='bold')

    # ── Panel 0: time trace for representative oblique saccade ─────────────
    ax0 = axes[0]
    if trace_data is not None:
        td = trace_data
        ax0.plot(td['t'], td['eye_h'],    color=utils.C['eye'],  lw=1.5, label='H')
        ax0.plot(td['t'], td['eye_v'],    color='#e08214',       lw=1.5, label='V')
        ax0.plot(td['t'], td['eye_roll'], color='#c62e2e',       lw=1.5, label='T (torsion)')
        ax0.axhline(td['T_exp'], color='#c62e2e', lw=1.0, ls='--',
                    label=f'T_expected = −H·V·π/360 = {td["T_exp"]:.2f}°')
        ax0.axhline(0.0, color='k', lw=0.4)
        ax0.axhline(float(trace_H), color=utils.C['eye'],  lw=0.7, ls=':', alpha=0.5,
                    label=f'Target H={trace_H:.0f}°')
        ax0.axhline(float(trace_V), color='#e08214', lw=0.7, ls=':', alpha=0.5,
                    label=f'Target V={trace_V:.0f}°')
    ax0.set_title(f'Saccade to ({trace_H:.0f}°, {trace_V:.0f}°): torsion settles to T_expected',
                  fontsize=9)
    ax0.set_xlabel('Time (s)', fontsize=8)
    ax0.set_ylabel('Eye position (deg)', fontsize=8)
    ax0.legend(fontsize=7, loc='center right')
    ax0.grid(True, alpha=0.15)

    # ── Panel 1: scatter T_final vs T_expected ──────────────────────────────
    ax1 = axes[1]
    scatter_colors = plt.cm.plasma(
        (V_arr - V_arr.min()) / (V_arr.max() - V_arr.min() + 1e-6))
    ax1.scatter(Te_arr, Tf_arr, c=scatter_colors, s=50, zorder=5)
    lim = max(np.abs(Te_arr).max(), np.abs(Tf_arr).max()) * 1.15 + 0.5
    ax1.plot([-lim, lim], [-lim, lim], 'k--', lw=1.0, alpha=0.7, label='Identity')
    ax1.set_xlabel('T expected = −H·V·π/360 (deg)', fontsize=8)
    ax1.set_ylabel('T measured (deg)', fontsize=8)
    ax1.set_title("Listing's plane: measured vs expected torsion", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.15)
    ax1.set_aspect('equal', 'box')
    # Colorbar proxy
    sm = plt.cm.ScalarMappable(cmap='plasma',
                                norm=plt.Normalize(V_arr.min(), V_arr.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('V (deg)', fontsize=7)

    # ── Panel 2: cross-section T vs H for fixed V values ───────────────────
    ax2 = axes[2]
    colors_v = plt.cm.plasma(np.linspace(0.15, 0.9, len(V_VALS)))
    H_line  = np.linspace(-35, 35, 100)
    for ci, V in enumerate(V_VALS):
        mask   = V_arr == V
        H_pts  = H_arr[mask]
        Tf_pts = Tf_arr[mask]
        T_line = -H_line * V * float(HALF_ANGLE)
        ax2.plot(H_line, T_line, color=colors_v[ci], lw=1.5, ls='--', alpha=0.8)
        ax2.scatter(H_pts, Tf_pts, color=colors_v[ci], s=50, zorder=5,
                    label=f'V={V:.0f}°')
    ax2.axhline(0.0, color='k', lw=0.4)
    ax2.axvline(0.0, color='k', lw=0.4)
    ax2.set_xlabel('H eye position (deg)', fontsize=8)
    ax2.set_ylabel('Torsion T (deg)', fontsize=8)
    ax2.set_title('Cross-sections of Listing\'s plane\n(slope = −V·π/360; dashed=theory)',
                  fontsize=9)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.15)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'listing_plane', show=show)
    return utils.fig_meta(path, rp,
        title="Listing's Plane",
        description='Saccades to a (H, V) grid; torsion measured at steady state. '
                    'g_ocr=0 so OCR=0 and T_required = −H·V·π/360.',
        expected='Scatter points lie on identity line; cross-sections linear with slope −V·π/360.',
        citation='Tweed, Haslwanter & Fetter (1998) IOVS; van Rijn & van den Berg (1993) EBR',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────
# 2. OCR-driven torsional correction saccade
# ─────────────────────────────────────────────────────────────────────────────

def _listing_ocr(show):
    """Head tilts 30°; OCR drives listing error → torsional saccade fires."""

    TILT_DEG  = 30.0
    TILT_VEL  = 60.0
    PRE_REST  = 0.5       # settle at rest before tilt starts
    TILT_DUR  = TILT_DEG / TILT_VEL
    HOLD_T    = 10.0
    TOTAL     = PRE_REST + TILT_DUR + HOLD_T

    t = np.arange(0.0, TOTAL, DT)
    T = len(t)

    # v[0] = 0 (pre-rest), then tilt, then hold — warmup-safe
    hv_roll  = np.where(
        t < PRE_REST, 0.0,
        np.where(t < PRE_REST + TILT_DUR, TILT_VEL, 0.0))
    head_km  = km.build_kinematics(t, rot_vel=np.stack(
        [np.zeros(T), np.zeros(T), hv_roll], axis=1))

    params_no_sac  = with_brain(PARAMS_DEFAULT, g_ocr=G_OCR, g_burst=0.0)
    params_with_sac = with_brain(PARAMS_DEFAULT, g_ocr=G_OCR, g_burst=700.0)

    common = dict(
        head=head_km,
        target_present_array=np.zeros(T),
        scene_present_array=np.zeros(T),
        sim_config=SimConfig(warmup_s=0.0),
        return_states=True,
    )

    st_no  = simulate(params_no_sac,   t, **common)
    st_sac = simulate(params_with_sac, t, **common)

    eye_roll_no  = (np.array(st_no.plant[:, 2])  + np.array(st_no.plant[:, 5]))  / 2.0
    eye_roll_sac = (np.array(st_sac.plant[:, 2]) + np.array(st_sac.plant[:, 5])) / 2.0
    g_est_sac    = np.array(st_sac.brain[:, _IDX_GRAV])

    # OCR target: G_OCR * G0 * sin(tilt)
    ocr_expected = G_OCR * G0 * np.sin(np.radians(TILT_DEG))

    t_rel = t - (PRE_REST + TILT_DUR)   # time relative to end of tilt

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(
        f'OCR-Driven Torsional Saccade  (head tilt {TILT_DEG:.0f}°, g_ocr={G_OCR:.2f})\n'
        f'Expected torsion: G_OCR × G0 × sin({TILT_DEG:.0f}°) = {ocr_expected:.2f}°',
        fontsize=12, fontweight='bold')

    head_roll_pos = head_km.rot_pos[:, 2]
    axes[0].plot(t_rel, head_roll_pos, color=utils.C['head'], lw=1.5)
    axes[0].axvline(0.0, color='gray', lw=0.8, ls=':')
    ax_fmt(axes[0], ylabel='Head roll (deg)')
    axes[0].set_title('Stimulus: 30° head roll at 60°/s then hold', fontsize=9)

    axes[1].plot(t_rel, g_est_sac[:, 1], color=utils.C['canal'], lw=1.2,
                 label='g_est[1] interaural (m/s²)')
    axes[1].axhline(-G0 * np.sin(np.radians(TILT_DEG)), color='tomato', lw=1.0, ls='--',
                    label=f'Expected g_est[1] = −G0·sin(30°) = {-G0*np.sin(np.radians(TILT_DEG)):.2f}')
    axes[1].axvline(0.0, color='gray', lw=0.8, ls=':')
    ax_fmt(axes[1], ylabel='g_est[1] (m/s²)')
    axes[1].legend(fontsize=8); axes[1].set_ylim(-12, 3)

    axes[2].plot(t_rel, eye_roll_no,  color='steelblue', lw=1.5, ls='--',
                 label='Torsion — no saccades (tonic NI drive only)')
    axes[2].plot(t_rel, eye_roll_sac, color=utils.C['eye'], lw=2.0,
                 label='Torsion — with saccades (listing correction)')
    axes[2].axhline(ocr_expected, color='tomato', lw=1.0, ls='--',
                    label=f'OCR target = {ocr_expected:.2f}°')
    axes[2].axhline(0.0, color='k', lw=0.4)
    axes[2].axvline(0.0, color='gray', lw=0.8, ls=':')
    ax_fmt(axes[2], ylabel='Eye torsion (deg)', xlabel='Time rel. tilt end (s)')
    axes[2].legend(fontsize=8)
    axes[2].set_title(
        'Torsion: tonic NI drive reaches OCR in ~τ_i=25 s (dashed); '
        'saccade corrects rapidly (solid)', fontsize=9)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'listing_ocr_saccade', show=show)
    return utils.fig_meta(path, rp,
        title="OCR Torsional Correction Saccade",
        description=f'Head tilt {TILT_DEG:.0f}°; g_ocr={G_OCR:.2f}. '
                    'Torsional listing error drives a corrective saccade.',
        expected=f'With saccades: torsion jumps to ≈{ocr_expected:.1f}° '
                 f'after accumulator delay (~80 ms × n_sacs). '
                 f'Without saccades: slow exponential rise with τ≈τ_i=25 s.',
        citation='Listing (1854); Tweed et al. (1998)',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────
# 3. Listing's plane maintenance during smooth pursuit
# ─────────────────────────────────────────────────────────────────────────────

def _listing_pursuit(show):
    """Sinusoidal H pursuit at fixed V=20°; torsion must track −H·V·π/360 continuously."""

    FREQ_HZ  = 0.5
    AMPL_DEG = 15.0
    V_DEG    = 20.0
    PRE_REST = 1.0
    TOTAL    = 10.0

    t = np.arange(0.0, TOTAL, DT)
    T = len(t)

    phase  = 2 * np.pi * FREQ_HZ * np.maximum(t - PRE_REST, 0.0)
    tgt_h  = AMPL_DEG * np.sin(phase)
    tgt_v  = np.full(T, V_DEG)

    head_km = km.build_kinematics(t)
    tgt     = km.build_target(t, yaw_deg=tgt_h, pitch_deg=tgt_v)

    params = with_brain(PARAMS_DEFAULT, g_burst=700.0, g_ocr=0.0)
    cfg    = SimConfig(warmup_s=0.0)
    common = dict(head=head_km, target=tgt,
                  target_present_array=np.ones(T),
                  scene_present_array=np.zeros(T),
                  sim_config=cfg, return_states=True)

    st = simulate(params, t, **common)

    eye_h    = (np.array(st.plant[:, 0]) + np.array(st.plant[:, 3])) / 2.0
    eye_v    = (np.array(st.plant[:, 1]) + np.array(st.plant[:, 4])) / 2.0
    eye_roll = (np.array(st.plant[:, 2]) + np.array(st.plant[:, 5])) / 2.0

    T_expected = -eye_h * V_DEG * float(HALF_ANGLE)
    T_error    = eye_roll - T_expected

    # Trim pre-rest for cleaner display
    t_disp = t - PRE_REST

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(
        f'Listing\'s Plane During Smooth Pursuit  '
        f'(H = {AMPL_DEG:.0f}° sin, V = {V_DEG:.0f}°, {FREQ_HZ:.1f} Hz)\n'
        f'T_required = −H · {V_DEG:.0f} · π/360',
        fontsize=12, fontweight='bold')

    axes[0].plot(t_disp, tgt_h, color='gray',       lw=1.0, ls='--', label='Target H')
    axes[0].plot(t_disp, eye_h, color=utils.C['eye'], lw=1.5, label='Eye H')
    axes[0].axhline(0.0, color='k', lw=0.4)
    axes[0].axvline(0.0, color='gray', lw=0.8, ls=':')
    ax_fmt(axes[0], ylabel='H position (deg)')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].set_title('Horizontal pursuit', fontsize=9)

    axes[1].plot(t_disp, T_expected, color='tomato',        lw=1.5, ls='--',
                 label='T_required = −H·V·π/360')
    axes[1].plot(t_disp, eye_roll,   color='#c62e2e',       lw=1.5,
                 label='Eye torsion (measured)')
    axes[1].axhline(0.0, color='k', lw=0.4)
    axes[1].axvline(0.0, color='gray', lw=0.8, ls=':')
    ax_fmt(axes[1], ylabel='Torsion (deg)')
    axes[1].legend(fontsize=8, loc='upper right')
    axes[1].set_title('Torsion: measured vs Listing\'s plane', fontsize=9)

    axes[2].plot(t_disp, T_error, color='#555', lw=1.2)
    axes[2].axhline(0.0, color='k', lw=0.4)
    axes[2].axvline(0.0, color='gray', lw=0.8, ls=':')
    rmse = float(np.sqrt(np.mean(T_error[t > PRE_REST + 1.0]**2)))
    axes[2].set_title(f'Listing error = T_measured − T_required  (RMS = {rmse:.3f}°)', fontsize=9)
    ax_fmt(axes[2], ylabel='Listing error (deg)', xlabel='Time rel. pursuit onset (s)')

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'listing_pursuit', show=show)
    return utils.fig_meta(path, rp,
        title="Listing's Plane During Smooth Pursuit",
        description=f'Sinusoidal H pursuit ({AMPL_DEG:.0f}° at {FREQ_HZ:.1f} Hz) at V={V_DEG:.0f}°. '
                    'vel_torsion demand added to NI keeps torsion on Listing\'s plane continuously.',
        expected=f'Eye torsion tracks T_required = −H·{V_DEG:.0f}·π/360 in real time; '
                 f'Listing error RMS < 0.5°.',
        citation='Tweed, Haslwanter & Fetter (1998) IOVS; van Rijn & van den Berg (1993) EBR',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────

def run(show=False):
    print('\n=== Listing\'s Law ===')
    figs = []
    print("  1/3  Listing's plane …");       figs.append(_listing_plane(show))
    print('  2/3  OCR correction saccade …'); figs.append(_listing_ocr(show))
    print('  3/3  Pursuit Listing maintenance …'); figs.append(_listing_pursuit(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
