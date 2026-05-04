"""Clinical vergence benchmarks — cover test and phoria.

Simulates the unilateral cover test by zeroing target visibility for the left eye
via target_present_L_array, while the right eye continues to see the target.
Binocular fusion is disrupted (bino = tv_L * tv_R = 0) so vergence drifts toward
the tonic resting vergence.  The amplitude and direction of the drift = phoria angle.

Usage:
    python -X utf8 scripts/bench_clinical_vergence.py
    python -X utf8 scripts/bench_clinical_vergence.py --show
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_clinical_utils as utils

import numpy as np
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, simulate
from oculomotor.sim import kinematics as km
from oculomotor.analysis import ax_fmt

SHOW = '--show' in sys.argv
DT   = 0.001
IPD  = 0.064   # m, inter-pupillary distance

SECTION = dict(
    id='clin_vergence',
    title='E. Vergence — Cover Test & Phoria',
    description='Clinical cover test: the left eye\'s target visibility is zeroed to simulate '
                'occlusion (target_present_L = 0). Binocular fusion is disrupted; vergence drifts '
                'toward the tonic resting vergence. The amplitude and direction of drift reveal '
                'phoria. Recovery on uncovering shows the fusional vergence response. '
                'Three conditions (exophoria, normal, esophoria) × two fixation distances.',
)


# ── Geometric helper ──────────────────────────────────────────────────────────

def _geo_verg(d_m):
    """Geometric vergence for a midline target at depth d_m (degrees)."""
    return 2.0 * np.degrees(np.arctan(IPD / 2.0 / d_m))


# ── Conditions: vary tonic_verg to produce different phorias ─────────────────

D_FAR  = 3.0    # far fixation (m)
D_NEAR = 0.5    # near fixation (m)
GEO_FAR  = _geo_verg(D_FAR)    # ≈ 1.2°
GEO_NEAR = _geo_verg(D_NEAR)   # ≈ 7.4°

TONIC_DEFAULT = PARAMS_DEFAULT.brain.tonic_verg   # ≈ 3.67°

CONDITIONS = [
    # (label, tonic_verg_deg, line_color)
    ('Exophoria\n(tonic = 0°)',                            0.0,             '#d62728'),
    (f'Normal\n(tonic = {TONIC_DEFAULT:.1f}°)',            TONIC_DEFAULT,   '#2ca02c'),
    ('Esophoria\n(tonic = 9°)',                            9.0,             '#1f77b4'),
]


# ── Simulation helper ─────────────────────────────────────────────────────────

def _run_cover(t, d_m, tonic_verg, T_COVER, T_UNCOVER):
    """Cover-uncover: L eye occluded during [T_COVER, T_UNCOVER).

    Right eye target visibility stays 1 throughout — the dominant eye continues
    to drive version saccades during cover (monocular fixation).
    Binocular disparity signal drops to 0 (bino = tv_L * tv_R = 0) so vergence
    drifts toward tonic_verg.
    """
    T     = len(t)
    p_tgt = np.tile([0.0, 0.0, float(d_m)], (T, 1))

    tg_L = np.ones(T, dtype=np.float32)
    tg_L[(t >= T_COVER) & (t < T_UNCOVER)] = 0.0   # cover L eye

    params = with_brain(PARAMS_DEFAULT, tonic_verg=float(tonic_verg))
    st = simulate(
        params, t,
        target=km.build_target(t, lin_pos=p_tgt),
        target_present_L_array=tg_L,
        target_present_R_array=np.ones(T, dtype=np.float32),
        scene_present_array=np.ones(T, dtype=np.float32),
        return_states=True,
    )
    eL = np.array(st.plant[:, 0])
    eR = np.array(st.plant[:, 3])
    return eL, eR


# ── Cover test figure ─────────────────────────────────────────────────────────

def _cover_test(show):
    """Cover-uncover test — 3 rows × 3 cols.

    Rows:  far vergence (L−R) | near vergence (L−R) | near per-eye yaw
    Cols:  exophoria | normal | esophoria

    Protocol
    --------
    0 – T_COVER   : binocular fixation (target_L = 1)
    T_COVER – T_UNC: cover L eye       (target_L = 0 → binocular fusion disrupted)
    T_UNC – T_END  : uncover           (target_L = 1 → fusion and vergence recover)
    """
    T_COVER = 1.0
    T_UNC   = 4.5
    T_END   = 7.5
    t = np.arange(0.0, T_END, DT)
    T = len(t)

    # ── Simulate all conditions × distances ───────────────────────────────────
    sim = {}
    for label, tonic, _ in CONDITIONS:
        sim[label] = {}
        for d in [D_FAR, D_NEAR]:
            eL, eR = _run_cover(t, d, tonic, T_COVER, T_UNC)
            sim[label][d] = (eL, eR, eL - eR)

    NROW = 3
    NCOL = len(CONDITIONS)
    fig, axes = plt.subplots(NROW, NCOL, figsize=(14, 11), sharex=True)
    fig.suptitle('Cover Test — Phoria Detection  (L Eye Covered)',
                 fontsize=12, fontweight='bold')

    CL = utils.C['eye']
    CR = utils.C['target']

    for col, (label, tonic, color) in enumerate(CONDITIONS):
        axes[0, col].set_title(label, fontsize=9, pad=4)

        for row, (d, geo, row_lbl) in enumerate(
            [(D_FAR,  GEO_FAR,  'Far (3 m)'),
             (D_NEAR, GEO_NEAR, 'Near (0.5 m)')]):

            eL, eR, verg = sim[label][d]
            phoria = tonic - geo
            pdir   = 'eso' if phoria > 0.15 else ('exo' if phoria < -0.15 else 'ortho')
            ax = axes[row, col]

            ax.axvspan(T_COVER, T_UNC, alpha=0.13, color='gray', label='Cover L eye')
            ax.plot(t, verg, color=color, lw=1.5, label='Vergence L−R')
            ax.axhline(geo,   color='tomato', lw=0.9, ls='--',
                       label=f'Geo target {geo:.1f}°')
            ax.axhline(tonic, color='#888',   lw=0.9, ls=':',
                       label=f'Tonic {tonic:.1f}°')

            psign = '+' if phoria > 0 else ''
            ax.text(0.97, 0.05,
                    f'Phoria: {psign}{phoria:.1f}° ({pdir})',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=7.5, color='#333')

            ylo = min(verg.min(), geo, tonic) - 0.8
            yhi = max(verg.max(), geo, tonic) + 0.8
            ax.set_ylim(ylo, yhi)

            ax_fmt(ax,
                   ylabel=f'{row_lbl}\nVergence L−R (deg)' if col == 0 else '',
                   xlabel='')
            if col == 0 and row == 0:
                ax.legend(fontsize=7.5, loc='upper left')

        # ── Row 2: per-eye yaw at near (most illustrative) ────────────────────
        eL, eR, verg = sim[label][D_NEAR]
        ax = axes[2, col]
        ax.axvspan(T_COVER, T_UNC, alpha=0.13, color='gray')
        ax.plot(t, eL, color=CL, lw=1.3, label='L eye (covered)')
        ax.plot(t, eR, color=CR, lw=1.3, ls='--', label='R eye (open)')
        # per-eye geometric targets at near
        ax.axhline(+GEO_NEAR / 2, color=CL, lw=0.8, ls='-.', alpha=0.7,
                   label=f'L target +{GEO_NEAR/2:.1f}°')
        ax.axhline(-GEO_NEAR / 2, color=CR, lw=0.8, ls='-.', alpha=0.7,
                   label=f'R target −{GEO_NEAR/2:.1f}°')
        ylo = min(eL.min(), eR.min(), -GEO_NEAR / 2) - 0.5
        yhi = max(eL.max(), eR.max(), +GEO_NEAR / 2) + 0.5
        ax.set_ylim(ylo, yhi)
        ax_fmt(ax,
               ylabel='Near: Eye yaw (deg)' if col == 0 else '',
               xlabel='Time (s)')
        if col == 0:
            ax.legend(fontsize=7.5, loc='upper left')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path, rp = utils.save_fig(fig, 'cover_test_phoria', show=show, params=PARAMS_DEFAULT,
                              conditions='Lit, alternate cover test at 3 m and 0.4 m — phoria measurement (control + simulated profiles)')
    return utils.fig_meta(
        path, rp,
        title='Cover Test — Phoria Detection',
        description='3 rows × 3 cols. '
                    'L eye occluded during shaded period (target_present_L = 0). '
                    'Cols: exophoria (tonic=0°), normal (tonic≈3.7°), esophoria (tonic=9°). '
                    'Rows 0–1: vergence at far (3 m) and near (0.5 m); '
                    'Row 2: per-eye yaw at near — covered L eye drifts, open R eye holds. '
                    'Phoria angle = tonic_verg − geo_demand (positive=eso, negative=exo).',
        expected='Exophoria: vergence falls during cover (divergent drift) at both distances. '
                 'Normal: mild eso drift at far (tonic > far demand); mild exo drift at near. '
                 'Esophoria: vergence rises during cover (convergent drift). '
                 'All cases: sharp recovery on uncover as fusion re-engages. '
                 'Row 2: L eye drifts inward (eso) or outward (exo); R eye stationary.',
        citation='Sheard C (1930) Am J Optom 7:564; '
                 'Schor CM (1979) Vision Res 19:1359; '
                 'Mitchell DE (1966) Invest Ophthalmol 5:566',
    )


# ── Section runner ────────────────────────────────────────────────────────────

def run(show=False):
    print('\n=== Clinical Vergence ===')
    figs = []
    print('  1/1  cover test — phoria detection …')
    figs.append(_cover_test(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
