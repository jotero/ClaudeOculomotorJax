"""Vergence benchmarks — symmetric convergence, asymmetric vergence.

Both tests run actual binocular simulations using the vergence controller
implemented in brain_model.py.  Saccades are disabled in the symmetric test
to isolate vergence dynamics.

Usage:
    python -X utf8 scripts/bench_vergence.py
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

from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, simulate, _IDX_VERG
from oculomotor.sim import kinematics as km
from oculomotor.analysis import ax_fmt

SHOW = '--show' in sys.argv
DT   = 0.001
IPD  = 0.064   # m, default inter-pupillary distance


SECTION = dict(
    id='vergence', title='5. Vergence',
    description='Binocular vergence eye movements driven by disparity. '
                'Symmetric convergence isolates vergence dynamics. '
                'Asymmetric vergence combines a version saccade with a depth change.',
)


def _verg_angle_deg(depth_m):
    """Geometric vergence angle for a midline target at given depth (degrees)."""
    return 2.0 * np.degrees(np.arctan(IPD / 2.0 / depth_m))


def _symmetric(show):
    """Symmetric convergence: far (3 m) → near (0.3 m) target step at midline."""
    T_STEP = 1.0
    TOTAL  = 4.5
    t      = np.arange(0.0, TOTAL, DT)
    T      = len(t)

    p_far  = np.array([0.0, 0.0, 3.0])
    p_near = np.array([0.0, 0.0, 0.3])
    pt     = np.where((t >= T_STEP)[:, None], p_near, p_far)

    # Disable saccades to isolate vergence; lit scene so target is visible
    params = with_brain(PARAMS_DEFAULT, g_burst=0.0)
    st     = simulate(params, t,
                      target=km.build_target(t, lin_pos=pt),
                      scene_present_array=np.ones(T),
                      return_states=True)

    eye_L_yaw = np.array(st.plant[:, 0])   # left  eye yaw (deg)
    eye_R_yaw = np.array(st.plant[:, 3])   # right eye yaw (deg)
    vergence  = eye_L_yaw - eye_R_yaw      # disconjugate; positive = convergent
    version   = (eye_L_yaw + eye_R_yaw) / 2.0

    verg_far  = _verg_angle_deg(3.0)
    verg_near = _verg_angle_deg(0.3)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle('Symmetric Vergence: Far (3 m) → Near (0.3 m)', fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(t, eye_L_yaw, color=utils.C['eye'],    lw=1.2, label='Left eye yaw')
    ax.plot(t, eye_R_yaw, color=utils.C['target'], lw=1.2, label='Right eye yaw', linestyle='--')
    ax.axvline(T_STEP, color='gray', lw=0.8, linestyle=':')
    ax_fmt(ax, ylabel='Eye yaw (deg)')
    ax.set_ylim(-8, 8)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(t, vergence, color=utils.C['eye'], lw=1.4)
    ax.axhline(verg_near, color='tomato', lw=0.9, linestyle='--',
               label=f'Geo. near vergence {verg_near:.1f}°')
    ax.axhline(verg_far,  color='gray',   lw=0.9, linestyle=':',
               label=f'Geo. far vergence  {verg_far:.1f}°')
    ax.axvline(T_STEP, color='gray', lw=0.8, linestyle=':')
    ax_fmt(ax, ylabel='Vergence L−R (deg)')
    ax.set_ylim(-1, 15)
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.plot(t, version, color=utils.C['ni'], lw=1.2)
    ax.axvline(T_STEP, color='gray', lw=0.8, linestyle=':')
    ax_fmt(ax, ylabel='Version (L+R)/2 (deg)', xlabel='Time (s)')
    ax.set_ylim(-1.5, 1.5)
    ax.set_title('Version ≈ 0 expected (symmetric step, no version change)', fontsize=9)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'vergence_symmetric', show=show)
    delta = verg_near - verg_far
    return utils.fig_meta(
        path, rp,
        title='Symmetric Vergence (Convergence)',
        description='Both eyes converge symmetrically when a near target appears at midline. '
                    'Saccades disabled to isolate vergence dynamics.',
        expected=f'Δvergence ≈ {delta:.1f}° (far→near; IPD={IPD*100:.0f} mm). '
                 f'Version ≈ 0. '
                 f'Rise TC: fast phasic component + slow integration (τ ≈ 400 ms within Panum\'s area).',
        citation='Mays (1984) J Neurophysiol; Cumming & Judge (1986) J Physiol',
    )


def _asymmetric(show):
    """Asymmetric vergence: simultaneous 10° rightward version + depth change (2 m → 0.5 m)."""
    T_STEP = 1.0
    TOTAL  = 4.5
    t      = np.arange(0.0, TOTAL, DT)
    T      = len(t)

    # cyclopean angle 10° right at 0.5 m: x = tan(10°) * 0.5 ≈ 0.088 m
    x_near = float(np.tan(np.radians(10.0)) * 0.5)
    p_far  = np.array([0.0,    0.0, 2.0])
    p_near = np.array([x_near, 0.0, 0.5])
    pt     = np.where((t >= T_STEP)[:, None], p_near, p_far)

    params = PARAMS_DEFAULT
    st     = simulate(params, t,
                      target=km.build_target(t, lin_pos=pt),
                      scene_present_array=np.ones(T),
                      return_states=True)

    eye_L_yaw = np.array(st.plant[:, 0])
    eye_R_yaw = np.array(st.plant[:, 3])
    vergence  = eye_L_yaw - eye_R_yaw
    version   = (eye_L_yaw + eye_R_yaw) / 2.0

    verg_near = _verg_angle_deg(0.5)
    verg_far  = _verg_angle_deg(2.0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle('Asymmetric Vergence: Version + Vergence (2 m → 0.5 m, 10° right)',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(t, eye_L_yaw, color=utils.C['eye'],    lw=1.2, label='Left eye yaw')
    ax.plot(t, eye_R_yaw, color=utils.C['target'], lw=1.2, label='Right eye yaw', linestyle='--')
    ax.axvline(T_STEP, color='gray', lw=0.8, linestyle=':')
    ax_fmt(ax, ylabel='Eye yaw (deg)')
    ax.set_ylim(-5, 18)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(t, version, color=utils.C['ni'], lw=1.4, label='Version (conjugate)')
    ax.axhline(10.0, color='gray', lw=0.9, linestyle='--', label='Target version = 10°')
    ax.axvline(T_STEP, color='gray', lw=0.8, linestyle=':')
    ax_fmt(ax, ylabel='Version (deg)')
    ax.set_ylim(-2, 14)
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.plot(t, vergence, color=utils.C['eye'], lw=1.4, label='Vergence (disconjugate)')
    ax.axhline(verg_near, color='tomato', lw=0.9, linestyle='--',
               label=f'Geo. vergence at 0.5 m: {verg_near:.1f}°')
    ax.axhline(verg_far,  color='gray',   lw=0.9, linestyle=':',
               label=f'Geo. vergence at 2 m:   {verg_far:.1f}°')
    ax.axvline(T_STEP, color='gray', lw=0.8, linestyle=':')
    ax_fmt(ax, ylabel='Vergence L−R (deg)', xlabel='Time (s)')
    ax.set_ylim(-1, 12)
    ax.legend(fontsize=9)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'vergence_asymmetric', show=show)
    return utils.fig_meta(
        path, rp,
        title='Asymmetric Vergence',
        description='Target moves 10° rightward AND closer (2 m → 0.5 m). '
                    'Version (conjugate) and vergence (disconjugate) should separate.',
        expected=f'Version ≈ 10° (saccade). '
                 f'Vergence ≈ {verg_near:.1f}° (near). '
                 f'Components independent (saccade TC ~ 50–70 ms; vergence TC ~ 400 ms).',
        citation='Collewijn et al. (1988) J Physiol; Zee et al. (1992)',
    )


def run(show=False):
    print('\n=== Vergence ===')
    figs = []
    print('  1/2  symmetric convergence …')
    figs.append(_symmetric(show))
    print('  2/2  asymmetric vergence …')
    figs.append(_asymmetric(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
