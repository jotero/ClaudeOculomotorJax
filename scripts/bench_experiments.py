"""Experimental / exploratory benchmarks.

Currently contains:
    1. Monocular occlusion — binocular fixation maintenance under three viewing conditions.

Usage:
    python -X utf8 scripts/bench_experiments.py
    python -X utf8 scripts/bench_experiments.py --show
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
    PARAMS_DEFAULT, SimConfig, simulate,
    _IDX_PURSUIT,
)
from oculomotor.sim.kinematics import build_target
from oculomotor.analysis import extract_burst

SHOW  = '--show' in sys.argv
DT    = 0.001
THETA = PARAMS_DEFAULT   # noiseless — deterministic


# ── Monocular occlusion ────────────────────────────────────────────────────────

_T_END  = 15.0
_T_FIX  = 5.0       # binocular fixation period before occlusion onset (s)
_DIST_M = 0.15      # target distance (m) — straight ahead, 15 cm

_CFG = SimConfig(warmup_s=30.0)   # 5 × tau_verg → vergence fully settled

_COND_LABELS = {
    'dark':       'Dark (both lose target)',
    'strobed':    'Strobed (position only)',
    'continuous': 'Continuous (monocular)',
}


def _make_flags(t_np, cond, occ_eye):
    T    = len(t_np)
    ones = np.ones(T,  dtype=np.float32)
    off  = np.where(t_np >= _T_FIX, 0.0, 1.0).astype(np.float32)
    ts   = np.where(t_np >= _T_FIX, 1.0, 0.0).astype(np.float32)

    if cond == 'dark':
        return off, off, np.zeros(T, dtype=np.float32)

    if occ_eye == 'left':
        tL, tR = off, ones
    else:
        tL, tR = ones, off

    strobed = ts if cond == 'strobed' else np.zeros(T, dtype=np.float32)
    return tL, tR, strobed


def _run_cond(t_np, cond, occ_eye):
    t  = jnp.array(t_np)
    T  = len(t_np)
    pt = jnp.tile(jnp.array([0.0, 0.0, _DIST_M]), (T, 1))
    tL, tR, ts = _make_flags(t_np, cond, occ_eye)
    return simulate(
        THETA, t,
        target                 = build_target(t, lin_pos=pt),
        scene_present_array    = jnp.zeros(T),
        target_present_L_array = jnp.array(tL),
        target_present_R_array = jnp.array(tR),
        target_strobed_array   = jnp.array(ts),
        return_states          = True,
        sim_config             = _CFG,
    )


def _occlusion(show):
    t_np = np.arange(0.0, _T_END, DT, dtype=np.float32)
    conditions = ['dark', 'strobed', 'continuous']

    results = {}
    s = _run_cond(t_np, 'dark', 'left')
    results[('dark', 'left')]  = s
    results[('dark', 'right')] = s   # symmetric

    for cond in ['strobed', 'continuous']:
        for occ in ['left', 'right']:
            results[(cond, occ)] = _run_cond(t_np, cond, occ)

    N_ROWS = 5
    fig, axes = plt.subplots(N_ROWS, 3, figsize=(13, 14), sharex=True)
    fig.suptitle(
        'Monocular occlusion — binocular fixation at 15 cm, dark room\n'
        'Vertical line = occlusion onset (t = 5 s)',
        fontsize=10,
    )

    row_labels = [
        'Left eye yaw (deg)',
        'Right eye yaw (deg)',
        'Vergence  L−R (deg)',
        'Pursuit cmd (deg/s)',
        'Saccade burst (deg/s)',
    ]
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8.5)

    for ci, cond in enumerate(conditions):
        axes[0, ci].set_title(_COND_LABELS[cond], fontsize=9)
        axes[N_ROWS - 1, ci].set_xlabel('Time (s)', fontsize=8)

        for occ in ['left', 'right']:
            st = results[(cond, occ)]

            eye_L    = np.array(st.plant[:, 0])
            eye_R    = np.array(st.plant[:, 3])
            vergence = eye_L - eye_R
            pursuit  = np.array(st.brain[:, _IDX_PURSUIT])[:, 0]
            burst    = np.array(extract_burst(st, THETA))[:, 0]

            if cond == 'dark':
                color, ls, lbl = utils.C['dark'],   '-',  'both occluded'
            elif occ == 'left':
                color, ls, lbl = utils.C['eye'],    '-',  'L eye occluded'
            else:
                color, ls, lbl = utils.C['target'], '--', 'R eye occluded'

            kw = dict(color=color, lw=1.5, ls=ls, label=lbl)
            axes[0, ci].plot(t_np, eye_L,    **kw)
            axes[1, ci].plot(t_np, eye_R,    **kw)
            axes[2, ci].plot(t_np, vergence, **kw)
            axes[3, ci].plot(t_np, pursuit,  **kw)
            axes[4, ci].plot(t_np, burst,    **kw)

            if cond == 'dark':
                break

        for row in range(N_ROWS):
            ax = axes[row, ci]
            ax.axvline(_T_FIX, color='gray', lw=0.8, ls='--', alpha=0.5)
            ax.grid(True, alpha=0.15)
            ylo, yhi = ax.get_ylim()
            span = max(yhi - ylo, 3.0)
            mid  = 0.5 * (ylo + yhi)
            ax.set_ylim(mid - span / 2, mid + span / 2)
            if ci == 0:
                ax.legend(fontsize=7)

        for ci2 in range(3):
            ax = axes[4, ci2]
            ylo, yhi = ax.get_ylim()
            span = max(yhi - ylo, 20.0)
            mid  = 0.5 * (ylo + yhi)
            axes[4, ci2].set_ylim(mid - span / 2, mid + span / 2)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'occlusion', show=show,
                              figs_dir=utils.EXPT_FIGS_DIR, base_dir=utils.EXPT_DIR)
    return utils.fig_meta(
        path, rp,
        title='Monocular occlusion',
        description='Binocular fixation at 15 cm under dark, strobed, and continuous monocular viewing.',
        expected='Both eyes maintain stable vergence during binocular phase. '
                 'After occlusion: dark → slow drift, strobed → position hold without velocity, '
                 'continuous → stable monocular fixation.',
        citation='Typical clinical dissociated nystagmus / monocular occlusion paradigm',
    )


# ── Section entry point ────────────────────────────────────────────────────────

SECTION = dict(
    id='experiments', title='Experimental',
    description='Exploratory paradigms: monocular occlusion, binocular fixation maintenance.',
)


def run(show=False):
    print('\n=== Experiments ===')
    figs = []
    print('  1/1  monocular occlusion …')
    figs.append(_occlusion(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
