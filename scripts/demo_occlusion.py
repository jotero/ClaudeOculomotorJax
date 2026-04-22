"""Monocular occlusion — binocular fixation maintenance demo.

Experiment design (6 conditions = 3 types × 2 occluded eyes):

    Phase 1 (0–5 s):   binocular fixation, target present in BOTH eyes,
                        dark room (scene always absent).

    Phase 2 (5–15 s):  ONE eye always loses the target.  The other eye:
        dark:       also loses target (both in darkness)
        strobed:    target flashes — position signal only, no velocity
        continuous: target remains fully visible

    Target: straight ahead at 15 cm.
    Repeat: with left eye occluded  (right retains/strobes/goes dark)
            with right eye occluded (left  retains/strobes/goes dark)

    dark condition is symmetric → run once, shown in both groups.

Outputs:  outputs/occlusion.png

Usage:
    python -X utf8 scripts/demo_occlusion.py [--show]
"""

import os, sys
import numpy as np
import jax.numpy as jnp
import matplotlib
SHOW = '--show' in sys.argv
if not SHOW:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, SimConfig, simulate,
    _IDX_PURSUIT, _IDX_VERG,
)
from oculomotor.analysis import extract_burst

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# ── Parameters ─────────────────────────────────────────────────────────────────

PARAMS  = PARAMS_DEFAULT          # no noise — deterministic demo
CFG     = SimConfig(warmup_s=30.0)  # 5 × tau_verg(6 s) → vergence fully settled

DT     = 0.001
T_END  = 15.0
T_FIX  = 5.0      # binocular fixation period (s)
DIST_M = 0.15     # target distance (m) — straight ahead

_C_DARK = '#555555'   # both eyes dark — grey
_C_LOCC = '#2166ac'   # left  eye occluded — blue
_C_ROCC = '#d6604d'   # right eye occluded — red

COND_LABELS = {
    'dark':       'Dark (both lose target)',
    'strobed':    'Strobed (position only)',
    'continuous': 'Continuous (monocular)',
}


# ── Stimulus builder ────────────────────────────────────────────────────────────

def _make_flags(t_np, cond, occ_eye):
    """Build (target_present_L, target_present_R, target_strobed) arrays.

    cond:    'dark' | 'strobed' | 'continuous'
    occ_eye: 'left' | 'right'   — which eye loses the target at T_FIX
    """
    T    = len(t_np)
    ones = np.ones(T,  dtype=np.float32)
    off  = np.where(t_np >= T_FIX, 0.0, 1.0).astype(np.float32)
    ts   = np.where(t_np >= T_FIX, 1.0, 0.0).astype(np.float32)

    if cond == 'dark':
        return off, off, np.zeros(T, dtype=np.float32)

    if occ_eye == 'left':
        tL = off     # left eye loses target after T_FIX
        tR = ones    # right eye always keeps target
    else:
        tL = ones
        tR = off     # right eye loses target after T_FIX

    strobed = ts if cond == 'strobed' else np.zeros(T, dtype=np.float32)
    return tL, tR, strobed


def run_condition(t_np, cond, occ_eye):
    t  = jnp.array(t_np)
    T  = len(t_np)
    pt = jnp.tile(jnp.array([0.0, 0.0, DIST_M]), (T, 1))
    tL, tR, ts = _make_flags(t_np, cond, occ_eye)
    return simulate(
        PARAMS, t,
        p_target_array         = pt,
        scene_present_array    = jnp.zeros(T),
        target_present_L_array = jnp.array(tL),
        target_present_R_array = jnp.array(tR),
        target_strobed_array   = jnp.array(ts),
        return_states          = True,
        sim_config             = CFG,
    )


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dt   = DT
    t_np = np.arange(0.0, T_END, dt, dtype=np.float32)

    conditions = ['dark', 'strobed', 'continuous']

    # Run simulations (dark is symmetric — run once)
    results = {}
    print('Running simulations...')

    print('  dark ...')
    s = run_condition(t_np, 'dark', 'left')
    results[('dark', 'left')]  = s
    results[('dark', 'right')] = s  # identical by symmetry

    for cond in ['strobed', 'continuous']:
        for occ in ['left', 'right']:
            print(f'  {cond}, {occ} eye occluded ...')
            results[(cond, occ)] = run_condition(t_np, cond, occ)

    # ── Figure: 5 rows × 3 cols ─────────────────────────────────────────────
    # Row 0: left  eye yaw
    # Row 1: right eye yaw
    # Row 2: vergence angle (L yaw − R yaw)
    # Row 3: pursuit yaw command (x_pursuit)
    # Row 4: saccade burst yaw (u_burst)
    # Columns: dark | strobed | continuous
    # Per panel: L-occluded (solid blue) and R-occluded (dashed red)
    # dark: one trace (grey, both eyes lose)

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
        axes[0, ci].set_title(COND_LABELS[cond], fontsize=9)
        axes[N_ROWS - 1, ci].set_xlabel('Time (s)', fontsize=8)

        for occ in ['left', 'right']:
            st = results[(cond, occ)]

            eye_L   = np.array(st.plant[:, 0])         # left  eye yaw
            eye_R   = np.array(st.plant[:, 3])         # right eye yaw
            vergence = eye_L - eye_R                   # positive = converged
            pursuit  = np.array(st.brain[:, _IDX_PURSUIT])[:, 0]  # pursuit yaw
            burst    = np.array(extract_burst(st, PARAMS))[:, 0]   # burst yaw

            if cond == 'dark':
                color, ls, lbl = _C_DARK, '-', 'both occluded'
            elif occ == 'left':
                color, ls, lbl = _C_LOCC, '-',  'L eye occluded'
            else:
                color, ls, lbl = _C_ROCC, '--', 'R eye occluded'

            kw = dict(color=color, lw=1.5, ls=ls, label=lbl)
            axes[0, ci].plot(t_np, eye_L,    **kw)
            axes[1, ci].plot(t_np, eye_R,    **kw)
            axes[2, ci].plot(t_np, vergence, **kw)
            axes[3, ci].plot(t_np, pursuit,  **kw)
            axes[4, ci].plot(t_np, burst,    **kw)

            if cond == 'dark':
                break   # symmetric — plot only once

        # Per-column formatting
        for row in range(N_ROWS):
            ax = axes[row, ci]
            ax.axvline(T_FIX, color='gray', lw=0.8, ls='--', alpha=0.5)
            ax.grid(True, alpha=0.15)
            ylo, yhi = ax.get_ylim()
            span = max(yhi - ylo, 3.0)
            mid  = 0.5 * (ylo + yhi)
            ax.set_ylim(mid - span / 2, mid + span / 2)
            if ci == 0:
                ax.legend(fontsize=7)

        # Burst panel: enforce minimum range so noise doesn't autoscale
        for ci2 in range(3):
            ax = axes[4, ci2]
            ylo, yhi = ax.get_ylim()
            span = max(yhi - ylo, 20.0)
            mid  = 0.5 * (ylo + yhi)
            axes[4, ci2].set_ylim(mid - span / 2, mid + span / 2)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'occlusion.png')
    fig.savefig(path, dpi=150)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    print(f'\nSaved {path}')


if __name__ == '__main__':
    main()
