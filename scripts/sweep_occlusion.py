"""Parameter sweep over (AC_A, tonic_verg, tonic_acc, proximal_d) and save
occlusion-summary plots for each combination into docs/experiments/figures/sweep/.

Sweep:
    AC_A         : 2, 4
    tonic_verg   : 0, 5, 10
    tonic_acc    : 0, 2
    proximal_d   : 0, 2, 4
    Total        : 36 combinations
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import bench_experiments as be
from oculomotor.analysis import extract_spv_states


SWEEP_DIR = Path(__file__).resolve().parents[1] / 'docs' / 'experiments' / 'figures' / 'sweep'
SWEEP_DIR.mkdir(parents=True, exist_ok=True)


def _columns():
    return [
        ('R cont', 'continuous', 'left'),
        ('R puls', 'pulsed',     'left'),
        ('L cont', 'continuous', 'right'),
        ('L puls', 'pulsed',     'right'),
        ('Dark',   'dark',       'left'),
    ]


def _collect(columns_data, t_np):
    by_cond = {'continuous': [], 'pulsed': [], 'dark': []}
    for (_title, cond, occ), st in columns_data:
        eye_L = np.array(st.plant[:, 0])
        eye_R = np.array(st.plant[:, 3])
        verg = eye_L - eye_R
        spv_L = extract_spv_states(st, np.array(t_np), eye='left')[:, 0]
        spv_R = extract_spv_states(st, np.array(t_np), eye='right')[:, 0]
        verg_spv = spv_L - spv_R
        vers_spv = extract_spv_states(st, np.array(t_np), eye='version')[:, 0]
        # Within-config L/R-viewing flip for version SPV
        vers_spv_signed = -vers_spv if occ == 'left' else vers_spv
        by_cond[cond].append((verg, verg_spv, vers_spv_signed))
    out = {}
    for cond, runs in by_cond.items():
        verg_stack     = np.stack([r[0] for r in runs], axis=0)
        verg_spv_stack = np.stack([r[1] for r in runs], axis=0)
        vers_spv_stack = np.stack([r[2] for r in runs], axis=0)
        out[cond] = (verg_stack.mean(axis=0),
                     verg_spv_stack.mean(axis=0),
                     vers_spv_stack.mean(axis=0))
    return out


def _make_summary_fig(conv, div, t_np, title_str):
    t_centred = np.array(t_np) - be._T_FIX
    keep = (t_centred >= -2.0) & (t_centred <= 10.0)
    t_plot = t_centred[keep]

    verg_spv_avg = {}
    vers_spv_avg = {}
    for cond in ('continuous', 'pulsed', 'dark'):
        _, vsv_conv, vesv_conv = conv[cond]
        _, vsv_div,  vesv_div  = div[cond]
        verg_spv_avg[cond] = (0.5 * (-vsv_conv  + vsv_div))[keep]
        vers_spv_avg[cond] = (0.5 * (-vesv_conv + vesv_div))[keep]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    fig.suptitle(title_str, fontsize=10)

    p6 = {
        ('conv', 'continuous'): ('#1b7837', '-',  'conv cont'),
        ('conv', 'pulsed'):     ('#5aae61', '--', 'conv flash'),
        ('conv', 'dark'):       ('#a6dba0', ':',  'conv dark'),
        ('div',  'continuous'): ('#762a83', '-',  'div cont'),
        ('div',  'pulsed'):     ('#9970ab', '--', 'div flash'),
        ('div',  'dark'):       ('#c2a5cf', ':',  'div dark'),
    }
    p3 = {
        'continuous': ('#1b7837', '-',  'continuous'),
        'pulsed':     ('#dd8452', '--', 'flashing'),
        'dark':       ('#762a83', ':',  'dark'),
    }

    # Panel 1: vergence position, 6 traces
    for tag, data in (('conv', conv), ('div', div)):
        for cond in ('continuous', 'pulsed', 'dark'):
            color, ls, lbl = p6[(tag, cond)]
            verg, _, _ = data[cond]
            axes[0].plot(t_plot, verg[keep], color=color, ls=ls, lw=1.2, label=lbl)
    # Panel 2: vergence SPV averaged with conv flipped
    for cond in ('continuous', 'pulsed', 'dark'):
        color, ls, lbl = p3[cond]
        axes[1].plot(t_plot, verg_spv_avg[cond], color=color, ls=ls, lw=1.4, label=lbl)
    # Panel 3: version SPV averaged
    for cond in ('continuous', 'pulsed', 'dark'):
        color, ls, lbl = p3[cond]
        axes[2].plot(t_plot, vers_spv_avg[cond], color=color, ls=ls, lw=1.4, label=lbl)

    titles = ['Vergence (deg)', 'Vergence SPV avg (deg/s)', 'Version SPV avg (deg/s)']
    for ax, ylab in zip(axes, titles):
        ax.axvline(0.0, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.axhline(0,   color='gray', lw=0.5, alpha=0.4)
        ax.set_xlabel('Time after occlusion (s)', fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_xlim(-2.0, 10.0)
        ax.grid(True, alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2, loc='best')
    axes[1].legend(fontsize=8, loc='best')
    axes[2].legend(fontsize=8, loc='best')
    fig.tight_layout()
    return fig


def _run_one(theta, dark_tonic_verg, dist_m, lens_d, t_np):
    cols = []
    for title, cond, occ in _columns():
        sim = be._run_cond(t_np, cond, occ,
                           theta_base=theta, dist_m=dist_m, lens_d=lens_d,
                           dark_tonic_verg=dark_tonic_verg)
        cols.append(((title, cond, occ), sim))
    return cols


def main():
    aca_vals  = [2.0, 3.0, 4.0]
    tv_vals   = [5.0, 8.0]
    ta_vals   = [0.0, 1.0, 2.0]
    prox_vals = [0.0, 1.0]

    t_np = np.arange(0.0, be._T_END, be.DT, dtype=np.float32)

    combos = [(a, tv, ta, p) for a in aca_vals for tv in tv_vals
              for ta in ta_vals for p in prox_vals]
    total = len(combos)
    print(f'Sweep: {total} combinations')

    for i, (aca, tv, ta, prox) in enumerate(combos, 1):
        tag = f'aca{aca:.0f}_tv{tv:.0f}_ta{ta:.0f}_prox{prox:.0f}'
        print(f'[{i:2d}/{total}] {tag}', flush=True)

        theta = be.with_brain(be._THETA_BASE,
                              tonic_verg=tv, tonic_acc=ta,
                              proximal_d=prox, AC_A=aca)

        conv_cols = _run_one(theta, tv, dist_m=0.15, lens_d=-5.6, t_np=t_np)
        div_cols  = _run_one(theta, tv, dist_m=10.0, lens_d= 1.0, t_np=t_np)

        conv = _collect(conv_cols, t_np)
        div  = _collect(div_cols,  t_np)

        title = (f'AC_A={aca:.0f}, tonic_verg={tv:.0f}°, '
                 f'tonic_acc={ta:.0f} D, proximal_d={prox:.0f} D')
        fig = _make_summary_fig(conv, div, t_np, title)
        out_path = SWEEP_DIR / f'summary_{tag}.png'
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'   → {out_path.name}', flush=True)


if __name__ == '__main__':
    main()
