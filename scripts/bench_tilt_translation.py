"""Tilt / Translation benchmarks — Ocular Counterroll (OCR) and OVAR.

OCR: head rolls to a static tilt, driving the otolith → gravity estimator → torsional
eye response.  Uses g_ocr parameter (default 0 — set to 0.13 for physiological ~13%).

OVAR: head rotates at constant velocity around an axis tilted from vertical.  The
canal signals a constant VOR drive; the otolith sees a sinusoidally changing gravity
direction, which modulates the slow-phase velocity.  Requires the gravity estimator
and K_gd (gravity dumping in VS) to be validated — currently shown as placeholder.

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

from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, simulate, _IDX_GRAV
from oculomotor.analysis import ax_fmt, vs_net

SHOW  = '--show' in sys.argv
DT    = 0.001

G0    = 9.81   # standard gravity (m/s²)


SECTION = dict(
    id='tilt_translation', title='3. Tilt / Translation',
    description='Otolith-driven eye movements: ocular counterroll (OCR) and OVAR. '
                'OCR is simulated with g_ocr=0.13 (13% physiological gain). '
                'OVAR requires the gravity estimator K_gd coupling to VS — placeholder.',
)


def _placeholder(show, name, title, description, expected, citation):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(0.5, 0.6,  title, ha='center', va='center', fontsize=16, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.42, 'NOT YET IMPLEMENTED', ha='center', va='center',
            fontsize=14, color='tomato', transform=ax.transAxes)
    ax.text(0.5, 0.28, 'Requires: gravity estimator K_gd coupling to VS (validated)',
            ha='center', va='center', fontsize=10, color='#555555', transform=ax.transAxes)
    ax.text(0.5, 0.16, f'Expected: {expected}',
            ha='center', va='center', fontsize=9, color='#333333', transform=ax.transAxes)
    ax.text(0.5, 0.06, f'Citation: {citation}',
            ha='center', va='center', fontsize=8, color='#777777',
            fontstyle='italic', transform=ax.transAxes)
    ax.set_axis_off()
    fig.tight_layout()
    path, rp = utils.save_fig(fig, name, show=show)
    return utils.fig_meta(path, rp, title=title, description=description,
                          expected=expected, citation=citation, fig_type='behavior')


def _ocr(show):
    """Ocular counterroll: head rolls 30°, measure torsional eye response."""
    TILT_DEG  = 30.0   # final head roll angle (deg)
    TILT_VEL  = 120.0  # roll velocity during tilt phase (deg/s)
    TILT_T    = TILT_DEG / TILT_VEL          # 0.25 s
    HOLD_T    = 5.0
    TOTAL     = TILT_T + HOLD_T
    t         = np.arange(0.0, TOTAL, DT)

    # Head velocity: roll [z-axis] for TILT_T, then stationary
    hv_roll   = np.where(t < TILT_T, TILT_VEL, 0.0)
    head_vel  = np.stack([np.zeros_like(t), np.zeros_like(t), hv_roll], axis=1)

    # Head position from integration (simulator will do this internally; we compute here for plotting)
    head_roll_pos = np.cumsum(hv_roll) * DT   # (T,)

    # g_ocr=0.13 (13% OCR gain — physiological range 10–15%); no saccades; dark room (no scene)
    params = with_brain(PARAMS_DEFAULT, g_ocr=0.13, g_burst=0.0)
    st     = simulate(params, t, head_vel_array=head_vel, return_states=True)

    # Eye torsion: roll = z-component (index 2) of plant state
    eye_L_roll = np.array(st.plant[:, 2])   # left  eye roll (deg)
    eye_R_roll = np.array(st.plant[:, 5])   # right eye roll (deg)
    eye_roll   = (eye_L_roll + eye_R_roll) / 2.0

    # Gravity estimate in head frame (from brain state)
    g_est = np.array(st.brain[:, _IDX_GRAV])   # (T, 3): [x_up, y_interaural, z_nos-occipit]
    g_est_y = g_est[:, 1]                       # interaural component (non-zero when rolled)

    # Expected OCR: opposite to head roll, ~13% gain → −0.13 × 30° = −3.9°
    ocr_ss = -0.13 * TILT_DEG

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f'Ocular Counterroll (OCR) — {TILT_DEG:.0f}° Head Roll, g_ocr = 0.13',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(t, head_roll_pos, color=utils.C['head'], lw=1.4, label='Head roll position')
    ax.axvline(TILT_T, color='gray', lw=0.8, linestyle=':')
    ax.axhline(TILT_DEG, color='gray', lw=0.8, linestyle='--', label=f'{TILT_DEG:.0f}° target')
    ax_fmt(ax, ylabel='Head roll (deg)')
    ax.set_ylim(-5, TILT_DEG + 8)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(t, eye_roll, color=utils.C['eye'], lw=1.4, label='Eye torsion (roll)')
    ax.axhline(ocr_ss, color='tomato', lw=0.9, linestyle='--',
               label=f'Expected OCR ≈ {ocr_ss:.1f}° (13% gain × {TILT_DEG:.0f}°)')
    ax.axhline(0.0, color='gray', lw=0.5, linestyle=':')
    ax.axvline(TILT_T, color='gray', lw=0.8, linestyle=':')
    ax_fmt(ax, ylabel='Eye torsion (deg)')
    ax.set_ylim(-7, 4)
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.plot(t, g_est_y, color=utils.C['canal'], lw=1.2, label='g_est y-component (interaural)')
    ax.axhline(G0 * np.sin(np.radians(TILT_DEG)), color='tomato', lw=0.9, linestyle='--',
               label=f'Expected g_y = G₀ sin({TILT_DEG:.0f}°) = {G0*np.sin(np.radians(TILT_DEG)):.2f} m/s²')
    ax.axvline(TILT_T, color='gray', lw=0.8, linestyle=':')
    ax_fmt(ax, ylabel='g_est y (m/s²)', xlabel='Time (s)')
    ax.set_ylim(-12, 12)
    ax.set_title('Gravity estimator should settle to G₀·sin(30°) ≈ 4.9 m/s²', fontsize=9)
    ax.legend(fontsize=9)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'tilt_ocr', show=show)
    return utils.fig_meta(
        path, rp,
        title='Ocular Counterroll (OCR)',
        description=f'Head rolls {TILT_DEG:.0f}° and holds. '
                    'Otolith → gravity estimator → torsional OCR via g_ocr parameter. '
                    'NOTE: gravity estimator not yet validated.',
        expected=f'OCR ≈ −{abs(ocr_ss):.1f}° (counter-rolling, {TILT_DEG:.0f}° tilt, 13% gain). '
                 f'Settling TC ≈ 2 s (1/K_grav). '
                 f'g_est y → G₀ sin(30°) = {G0*np.sin(np.radians(TILT_DEG)):.1f} m/s².',
        citation='Boff, Kaufman & Thomas (1986); Tweed et al. (1994)',
    )


def _ovar(show):
    return _placeholder(show, 'tilt_ovar',
        title='OVAR — Off-Vertical Axis Rotation',
        description='Head rotates at constant velocity around a tilted axis. '
                    'Canal component is constant; otolith modulates nystagmus SPV sinusoidally.',
        expected='Sinusoidal modulation of SPV with period = rotation period. '
                 'Amplitude proportional to tilt angle. '
                 'Requires K_gd (gravity → VS coupling) to be validated.',
        citation='Raphan et al. (1981); Angelaki & Hess (1994) J Neurophysiol')


def run(show=False):
    print('\n=== Tilt / Translation ===')
    figs = []
    print('  1/2  ocular counterroll (OCR) …')
    figs.append(_ocr(show))
    print('  2/2  OVAR (placeholder) …')
    figs.append(_ovar(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
