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
                'Asymmetric vergence combines a version saccade with a depth change. '
                'Fixation disparity benchmark measures residual vergence error vs. target distance.',
)


def _verg_angle_deg(depth_m):
    """Geometric vergence angle for a midline target at given depth (degrees)."""
    return 2.0 * np.degrees(np.arctan(IPD / 2.0 / depth_m))


def _run_sym(t, d_start, d_end, T_STEP):
    """Simulate symmetric vergence step (no saccades). Returns (eye_L, eye_R)."""
    T  = len(t)
    p0 = np.array([0.0, 0.0, float(d_start)])
    p1 = np.array([0.0, 0.0, float(d_end)])
    pt = np.where((t >= T_STEP)[:, None], p1, p0)
    params = with_brain(PARAMS_DEFAULT, g_burst=0.0)   # saccades off → isolate vergence
    st = simulate(params, t,
                  target=km.build_target(t, lin_pos=pt),
                  scene_present_array=np.ones(T),
                  return_states=True)
    return np.array(st.plant[:, 0]), np.array(st.plant[:, 3])


def _run_asym(t, d_start, d_end, ver_deg, T_STEP):
    """Simulate asymmetric vergence + version step (saccades on). Returns (eye_L, eye_R)."""
    T      = len(t)
    x_end  = float(np.tan(np.radians(ver_deg)) * d_end)
    p0     = np.array([0.0,   0.0, float(d_start)])
    p1     = np.array([x_end, 0.0, float(d_end)])
    pt     = np.where((t >= T_STEP)[:, None], p1, p0)
    st = simulate(PARAMS_DEFAULT, t,
                  target=km.build_target(t, lin_pos=pt),
                  scene_present_array=np.ones(T),
                  return_states=True)
    return np.array(st.plant[:, 0]), np.array(st.plant[:, 3])


def _vergence_bidir(show):
    """Symmetric and asymmetric vergence in both directions — 3 rows × 4 cols.

    Columns: sym-conv | sym-div | asym-conv | asym-div
    Row 0:   L and R eye yaw positions
    Row 1:   vergence (L−R)
    Row 2:   version  (L+R)/2
    """
    T_STEP = 1.0
    TOTAL  = 5.0
    t      = np.arange(0.0, TOTAL, DT)

    D_FAR, D_NEAR = 3.0, 0.3
    VER = 10.0
    D_ASYM_START_C, D_ASYM_END_C = 2.0, 0.5
    D_ASYM_START_D, D_ASYM_END_D = 0.5, 2.0

    eL_sc, eR_sc = _run_sym(t,  D_FAR,          D_NEAR,         T_STEP)
    eL_sd, eR_sd = _run_sym(t,  D_NEAR,         D_FAR,          T_STEP)
    eL_ac, eR_ac = _run_asym(t, D_ASYM_START_C, D_ASYM_END_C,   VER, T_STEP)
    eL_ad, eR_ad = _run_asym(t, D_ASYM_START_D, D_ASYM_END_D,  -VER, T_STEP)

    eL   = [eL_sc, eL_sd, eL_ac, eL_ad]
    eR   = [eR_sc, eR_sd, eR_ac, eR_ad]
    verg = [eL_sc-eR_sc, eL_sd-eR_sd, eL_ac-eR_ac, eL_ad-eR_ad]
    vers = [(eL_sc+eR_sc)/2, (eL_sd+eR_sd)/2, (eL_ac+eR_ac)/2, (eL_ad+eR_ad)/2]

    geo = {d: _verg_angle_deg(d) for d in [D_FAR, D_NEAR, D_ASYM_START_C,
                                             D_ASYM_END_C, D_ASYM_START_D, D_ASYM_END_D]}

    # Per-column: (title, verg_start, verg_target, ver_target, is_sym)
    cols_meta = [
        ('Symmetric conv\n3 m → 0.3 m',      geo[D_FAR],          geo[D_NEAR],         0.0,   True),
        ('Symmetric div\n0.3 m → 3 m',        geo[D_NEAR],         geo[D_FAR],          0.0,   True),
        ('Asymmetric R+conv\n2 m → 0.5 m',    geo[D_ASYM_START_C], geo[D_ASYM_END_C],   VER,   False),
        ('Asymmetric L+div\n0.5 m → 2 m',     geo[D_ASYM_START_D], geo[D_ASYM_END_D],  -VER,   False),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    fig.suptitle('Vergence: Symmetric and Asymmetric — Both Directions',
                 fontsize=12, fontweight='bold')

    CL = utils.C['eye']
    CR = utils.C['target']

    for col, (title, g_start, g_tgt, v_tgt, is_sym) in enumerate(cols_meta):
        axes[0, col].set_title(title, fontsize=9, pad=4)

        # ── Row 0: L and R eye yaw ────────────────────────────────────────────
        ax = axes[0, col]
        ax.plot(t, eL[col], color=CL, lw=1.3, label='L eye')
        ax.plot(t, eR[col], color=CR, lw=1.3, ls='--', label='R eye')
        ax.axvline(T_STEP, color='gray', lw=0.8, ls=':')
        lo = min(eL[col].min(), eR[col].min()) - 1.0
        hi = max(eL[col].max(), eR[col].max()) + 1.0
        ax.set_ylim(lo, hi)
        ax_fmt(ax, ylabel='Eye yaw (deg)' if col == 0 else '')
        ax.legend(fontsize=7.5)
        if is_sym:
            ax.text(0.97, 0.05, 'saccades off', transform=ax.transAxes,
                    fontsize=7, ha='right', color='#777')

        # ── Row 1: vergence ───────────────────────────────────────────────────
        ax = axes[1, col]
        ax.plot(t, verg[col], color=CL, lw=1.5)
        ax.axhline(g_tgt,   color='tomato', lw=0.9, ls='--', label=f'Target {g_tgt:.1f}°')
        ax.axhline(g_start, color='gray',   lw=0.8, ls=':',  label=f'Start  {g_start:.1f}°')
        ax.axvline(T_STEP,  color='gray',   lw=0.8, ls=':')
        lo = min(g_start, g_tgt) - 1.0
        hi = max(g_start, g_tgt) + (3.0 if not is_sym else 1.0)
        ax.set_ylim(lo, hi)
        ax_fmt(ax, ylabel='Vergence L−R (deg)' if col == 0 else '')
        ax.legend(fontsize=7.5)

        # ── Row 2: version ────────────────────────────────────────────────────
        ax = axes[2, col]
        ax.plot(t, vers[col], color=utils.C['ni'], lw=1.4)
        ax.axhline(v_tgt, color='gray', lw=0.9, ls='--',
                   label=f'Target {v_tgt:+.0f}°')
        ax.axvline(T_STEP, color='gray', lw=0.8, ls=':')
        lim = max(abs(v_tgt) + 2, 2)
        ax.set_ylim(-lim, lim)
        ax_fmt(ax, ylabel='Version (L+R)/2 (deg)' if col == 0 else '',
               xlabel='Time (s)')
        ax.legend(fontsize=7.5)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path, rp = utils.save_fig(fig, 'vergence_bidir', show=show)

    return utils.fig_meta(
        path, rp,
        title='Vergence: Symmetric + Asymmetric, Both Directions',
        description='3 rows × 4 cols. Cols: symmetric conv, symmetric div, '
                    'asymmetric right+conv, asymmetric left+div. '
                    'Row 0: L/R eye yaw. Row 1: vergence (L−R). Row 2: version (L+R)/2.',
        expected='Symmetric: L/R yaw diverge/converge equally, version ≈ 0. '
                 'Asymmetric: both eyes saccade then vergence separates them. '
                 'Rise TC ≈ 0.5–1 s. Version and vergence components are disconjugate.',
        citation='Mays (1984) J Neurophysiol; Collewijn et al. (1988) J Physiol; '
                 'Zee et al. (1992) J Neurophysiol',
    )


def _fixation_distance(show):
    """Fixation disparity vs. target distance.

    For each of several midline target distances, run a simulation long enough
    to reach vergence steady-state, then compare the model's vergence angle to
    the geometric prediction 2·arctan(IPD/2/d).  The residual is fixation
    disparity — the small systematic error that persists even with binocular
    fusion.
    """
    DISTANCES = [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]   # metres
    T_STEP  = 0.5    # far-fixation baseline
    T_TOTAL = 7.0    # long enough for vergence to settle
    T_SS    = 6.0    # average from here to T_TOTAL for steady-state

    t    = np.arange(0.0, T_TOTAL, DT)
    T    = len(t)
    p_far = np.array([0.0, 0.0, 3.0])

    geo_verg    = np.array([_verg_angle_deg(d) for d in DISTANCES])
    meas_verg   = np.zeros(len(DISTANCES))
    tc_traces   = {}

    params = with_brain(PARAMS_DEFAULT, g_burst=0.0)   # no saccades → isolate vergence

    for i, d in enumerate(DISTANCES):
        p_near = np.array([0.0, 0.0, float(d)])
        pt = np.where((t >= T_STEP)[:, None], p_near, p_far)

        st = simulate(params, t,
                      target=km.build_target(t, lin_pos=pt),
                      scene_present_array=np.ones(T),
                      return_states=True)

        eye_L_yaw = np.array(st.plant[:, 0])
        eye_R_yaw = np.array(st.plant[:, 3])
        vergence  = eye_L_yaw - eye_R_yaw

        ss_mask = t >= T_SS
        meas_verg[i] = float(vergence[ss_mask].mean())
        tc_traces[d] = vergence

    fixation_disparity = geo_verg - meas_verg   # positive = lag (under-convergence)

    # ── colours: near → warm, far → cool ────────────────────────────────────
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=min(DISTANCES), vmax=max(DISTANCES))
    colors = [cmap(1.0 - norm(d)) for d in DISTANCES]   # near = bright

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Fixation Disparity vs. Target Distance', fontsize=12, fontweight='bold')

    # Panel 0 — vergence time-courses ─────────────────────────────────────────
    ax = axes[0]
    for i, d in enumerate(DISTANCES):
        ax.plot(t, tc_traces[d], color=colors[i], lw=1.3,
                label=f'{d:.2g} m  (geo {geo_verg[i]:.1f}°)')
    ax.axvline(T_STEP, color='gray', lw=0.8, ls=':')
    ax_fmt(ax, ylabel='Vergence L−R (deg)', xlabel='Time (s)')
    ax.legend(fontsize=7.5, ncol=2, loc='upper left')
    ax.set_title('Vergence time-courses', fontsize=9)

    # Panel 1 — fixation disparity (arcmin) ──────────────────────────────────
    ax = axes[1]
    fd_arcmin = fixation_disparity * 60.0
    for i, d in enumerate(DISTANCES):
        ax.scatter(d, fd_arcmin[i], color=colors[i], zorder=5, s=60)
    ax.plot(DISTANCES, fd_arcmin, 'k-', lw=0.8, alpha=0.5)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axhspan(-6, 6, alpha=0.07, color='green', label='Normal range (±6 arcmin)')
    ax.set_xlabel('Target distance (m)')
    ax.set_ylabel('Fixation disparity (arcmin)')
    ax.set_xscale('log')
    ax.set_xticks(DISTANCES)
    ax.set_xticklabels([f'{d:.2g}' for d in DISTANCES], fontsize=8)
    ax.set_title('Residual vergence error at steady state', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    ax.text(0.02, 0.97, 'Positive = under-convergence (exo)',
            transform=ax.transAxes, fontsize=7.5, va='top', color='#555555')

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'vergence_fixation_distance', show=show)
    return utils.fig_meta(
        path, rp,
        title='Fixation Disparity vs. Distance',
        description='Steady-state vergence residual error at eight target distances (0.2–3 m). '
                    'Left: vergence time-courses (near = warm colours). '
                    'Right: fixation disparity in arcmin — the residual after fusion settles. '
                    'Positive = under-convergence (exo FD). Green band = normal ±6 arcmin.',
        expected='Fixation disparity <10 arcmin across the physiological range (Ogle 1964: 0–6 arcmin). '
                 'Near targets may show slightly larger exo FD due to reduced fusional gain. '
                 'Vergence TC ~0.5–2 s.',
        citation='Ogle, Martens & Dyer (1967); Semmlow & Hung (1983) Vision Research 23(3)',
    )


def _diplopia(show):
    """Diplopic conditions — 3 rows × 3 cols.

    Col 0: Fused   (3 m → 0.3 m, midline, saccades off)
    Col 1: Diplopic (3 m → 0.05 m, midline, saccades off)
    Col 2: Analytical fusion gates (rows 0–1) + graded steady-state scatter (row 2)

    Row 0: L eye (blue) + R eye (red) yaw; dashed/annotated = per-eye geometric target
           Fused:   targets at ±geo/2 ≈ ±6° — within plot range, shown as lines
           Diplopic: targets at ±geo/2 ≈ ±33° — off-scale, annotated with text
    Row 1: Vergence (L−R); dashed = geometric target; dotted = proximal limit
    Row 2: Version (L+R)/2 ≈ 0 (midline target — sanity check); col 2: graded scatter
    """
    T_STEP = 1.0
    TOTAL  = 6.0
    t      = np.arange(0.0, TOTAL, DT)
    T      = len(t)

    DEPTHS     = [0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05]
    D_FUSED    = 0.3     # within proximal range  → fused
    D_DIPLOPIC = 0.05    # beyond proximal range  → diplopic
    D_BASE     = 3.0     # starting fixation distance

    params = with_brain(PARAMS_DEFAULT, g_burst=0.0)   # saccades off → isolate vergence
    p_base = np.array([0.0, 0.0, float(D_BASE)])

    eL_tr, eR_tr = {}, {}
    for d in DEPTHS:
        pt = np.where((t >= T_STEP)[:, None],
                      np.array([0.0, 0.0, float(d)]), p_base)
        st = simulate(params, t,
                      target=km.build_target(t, lin_pos=pt),
                      scene_present_array=np.ones(T), return_states=True)
        eL_tr[d] = np.array(st.plant[:, 0])
        eR_tr[d] = np.array(st.plant[:, 3])

    verg_tr = {d: eL_tr[d] - eR_tr[d] for d in DEPTHS}
    vers_tr = {d: (eL_tr[d] + eR_tr[d]) / 2 for d in DEPTHS}

    geo_d    = {d: _verg_angle_deg(d) for d in DEPTHS}
    geo_base = _verg_angle_deg(D_BASE)
    ss_mask  = t >= 5.0
    meas_v   = {d: float(verg_tr[d][ss_mask].mean()) for d in DEPTHS}

    bp         = PARAMS_DEFAULT.brain
    disp_fus   = bp.panum_h              # horizontal fusion limit (Panum's, ~2°)
    disp_prox  = bp.prox_sat            # coarse drive saturation (~20°)
    verg_max   = bp.npc                 # NPC / convergence motor limit (~50°)
    disp_vert  = bp.panum_v
    disp_tors  = bp.panum_t
    dipl_abs   = geo_base + verg_max    # absolute diplopia threshold (geo_base + npc)

    CL = utils.C['eye']
    CR = utils.C['target']

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('Diplopia: Fusion Gate and Fused vs. Diplopic Vergence',
                 fontsize=12, fontweight='bold')

    def vline(ax): ax.axvline(T_STEP, color='gray', lw=0.8, ls=':')

    # ── Cols 0 & 1: fused vs. diplopic time-series ───────────────────────────
    for col, d in enumerate([D_FUSED, D_DIPLOPIC]):
        geo  = geo_d[d]
        tL   = +geo / 2    # L eye geometric target (midline → slightly rightward)
        tR   = -geo / 2    # R eye geometric target (slightly leftward)
        tag  = 'fused' if col == 0 else 'diplopic'
        ttl  = f'Fused ({d} m, Vg_geo={geo:.1f}°)' if col == 0 else \
               f'Diplopic ({d} m, Vg_geo={geo:.1f}°)'
        eL   = eL_tr[d]
        eR   = eR_tr[d]

        # Row 0: Per-eye yaw ──────────────────────────────────────────────────
        ax = axes[0, col]
        ax.plot(t, eL, color=CL, lw=1.4, label='L eye')
        ax.plot(t, eR, color=CR, lw=1.4, ls='--', label='R eye')
        vline(ax)
        ax.set_title(ttl, fontsize=9, pad=4)
        # Y-axis: auto-scale to actual eye movements + margin
        ylo = min(eL.min(), eR.min()) - 1.0
        yhi = max(eL.max(), eR.max()) + 1.0
        ax.set_ylim(ylo, yhi)
        # Show per-eye targets as lines if in range, else annotate
        if ylo <= tL <= yhi:
            ax.axhline(tL, color=CL, lw=0.9, ls='-.', alpha=0.8, label=f'L target {tL:+.1f}°')
        else:
            ax.text(0.97, 0.97, f'L target {tL:+.1f}° (↑ off-scale)',
                    transform=ax.transAxes, ha='right', va='top', fontsize=7, color=CL)
        if ylo <= tR <= yhi:
            ax.axhline(tR, color=CR, lw=0.9, ls='-.', alpha=0.8, label=f'R target {tR:+.1f}°')
        else:
            ax.text(0.97, 0.87, f'R target {tR:+.1f}° (↓ off-scale)',
                    transform=ax.transAxes, ha='right', va='top', fontsize=7, color=CR)
        if col == 0:
            ax.text(0.03, 0.05, 'saccades off', transform=ax.transAxes,
                    fontsize=7, color='#777')
        ax_fmt(ax, ylabel='Eye yaw (deg)' if col == 0 else '')
        ax.legend(fontsize=7.5)

        # Row 1: Vergence ─────────────────────────────────────────────────────
        ax = axes[1, col]
        ax.plot(t, verg_tr[d], color=CL, lw=1.5, label='Vergence L−R')
        ax.axhline(geo,      color='tomato', lw=1.0, ls='--',
                   label=f'Geo target {geo:.1f}°')
        ax.axhline(dipl_abs, color='#888',   lw=0.9, ls=':',
                   label=f'NPC limit {dipl_abs:.1f}°')
        vline(ax)
        lo_v = geo_base - 1.0
        hi_v = min(geo, dipl_abs) + 3.0
        ax.set_ylim(lo_v, hi_v)
        ax_fmt(ax, ylabel='Vergence L−R (deg)' if col == 0 else '')
        ax.legend(fontsize=7.5)

        # Row 2: Version (sanity check — should be ≈ 0 for midline target) ───
        ax = axes[2, col]
        ax.plot(t, vers_tr[d], color=utils.C['ni'], lw=1.4, label=f'{tag}')
        ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.6, label='Midline target')
        vline(ax)
        ax.set_ylim(-2, 2)
        ax_fmt(ax, ylabel='Version (L+R)/2 (deg)' if col == 0 else '',
               xlabel='Time (s)')
        ax.legend(fontsize=7.5)

    # ── Col 2: analytical gates + graded scatter ──────────────────────────────

    # Row 0: Horizontal fusion gate — 3 zones
    ax = axes[0, 2]
    d_h_arr  = np.linspace(-disp_fus - 2, verg_max + 6, 800)
    gate_c   = 1.0 / (1.0 + np.exp(-100.0 * (verg_max  - d_h_arr)))   # NPC / diplopia gate
    gate_d_  = 1.0 / (1.0 + np.exp(-100.0 * (d_h_arr   + disp_fus)))  # divergence gate
    ax.plot(d_h_arr, gate_c * gate_d_, color=utils.C['eye'], lw=2.0, label='fuse gate')
    # shade three zones
    ax.axvspan(-disp_fus - 2, -disp_fus,  alpha=0.10, color='tomato',   label='diplopic (div)')
    ax.axvspan(-disp_fus,      disp_fus,   alpha=0.15, color='#2ca02c',  label='fused (Panum\'s)')
    ax.axvspan( disp_fus,      verg_max,   alpha=0.10, color='#ff7f0e',  label='fusable')
    ax.axvspan( verg_max,      verg_max+6, alpha=0.10, color='tomato',   label='_')
    ax.axvline( verg_max,  color='tomato',    lw=1.0, ls='--', label=f'+{verg_max:.0f}° NPC')
    ax.axvline(-disp_fus,  color='steelblue', lw=1.0, ls='--', label=f'−{disp_fus:.0f}° div')
    ax.axvline( disp_fus,  color='#2ca02c',   lw=0.8, ls=':',  label=f'+{disp_fus:.0f}° Panum\'s')
    ax.axhline(0.5, color='gray', lw=0.7, ls=':', alpha=0.6)
    ax.set_xlabel('Horizontal disparity (deg)')
    ax.set_ylabel('Fusion gate (0=diplopic, 1=fusable/fused)')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=7.5, loc='lower left')
    ax.set_title('Fusion gate — 3 states', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 1: Vertical + torsional fusion gates
    ax = axes[1, 2]
    d_abs    = np.linspace(0, max(disp_vert, disp_tors) + 3, 400)
    gate_v   = 1.0 / (1.0 + np.exp(-100.0 * (disp_vert - d_abs)))
    gate_tor = 1.0 / (1.0 + np.exp(-100.0 * (disp_tors - d_abs)))
    ax.plot(d_abs, gate_v,   color=utils.C['ni'],     lw=2.0, label=f'Vert ±{disp_vert}°')
    ax.plot(d_abs, gate_tor, color=utils.C['target'], lw=2.0, ls='--',
            label=f'Tors ±{disp_tors}°')
    ax.axhline(0.5, color='gray', lw=0.7, ls=':', alpha=0.6)
    ax.set_xlabel('|Disparity| (deg)')
    ax.set_ylabel('Fusion gate (0=diplopic, 1=fused)')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8)
    ax.set_title('Vertical & torsional fusion gates', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 2: Graded steady-state scatter
    ax = axes[2, 2]
    geo_arr  = np.array([geo_d[d] for d in DEPTHS])
    meas_arr = np.array([meas_v[d] for d in DEPTHS])
    in_range = (geo_arr - geo_base) < verg_max   # green = fusable/fused; red = diplopic
    geo_crv  = np.linspace(0, max(geo_arr) + 5, 200)
    ax.plot(geo_crv, geo_crv, 'k-', lw=1.0, alpha=0.3, label='Ideal (1:1)')
    ax.axvline(dipl_abs, color='tomato', lw=1.2, ls='--',
               label=f'NPC limit {dipl_abs:.1f}°')
    for i, d in enumerate(DEPTHS):
        ax.scatter(geo_arr[i], meas_arr[i],
                   color='#2ca02c' if in_range[i] else '#d62728',
                   s=80, zorder=5)
    from matplotlib.lines import Line2D
    leg_h = [Line2D([0],[0], marker='o', color='w', markerfacecolor='#2ca02c', ms=8, label='Fused'),
             Line2D([0],[0], marker='o', color='w', markerfacecolor='#d62728', ms=8, label='Diplopic')]
    ax.legend(handles=leg_h + [ax.lines[0], ax.lines[1]], fontsize=8)
    ax.set_xlabel('Geometric vergence demand (deg)')
    ax.set_ylabel('Steady-state vergence (deg)')
    ax.set_title('Graded convergence (green=fusable/fused, red=diplopic)', fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path, rp = utils.save_fig(fig, 'vergence_diplopia', show=show)
    return utils.fig_meta(
        path, rp,
        title='Diplopia: Fusion Gate and Vergence Limits',
        description='3 rows × 3 cols. '
                    'Cols 0–1: fused (0.3 m) vs. diplopic (0.05 m) midline step, saccades off. '
                    'Row 0: L/R eye yaw with per-eye geometric target lines (or off-scale annotation). '
                    'Row 1: vergence (L−R) with geometric target and proximal limit. '
                    'Row 2: version ≈ 0 (sanity check); col 2: graded scatter. '
                    'Col 2: horizontal, vertical/torsional fusion gates + graded scatter.',
        expected='Fused: L eye reaches +geo/2, R eye reaches −geo/2; '
                 'both eye trajectories meet the geometric target dashed lines. '
                 'Diplopic: eyes converge to prox limit (≈22°); per-eye targets (±33°) are '
                 'off-scale (annotated); version stays ≈ 0 for midline target. '
                 'Graded scatter: green dots follow 1:1 line up to prox limit, '
                 'red dots plateau below it.',
        citation='Fender & Julesz (1967) J Opt Soc Am; '
                 'Schor (1979) Vision Res; '
                 'Shipley & Rawlings (1970) Vision Res',
    )


def run(show=False):
    print('\n=== Vergence ===')
    figs = []
    print('  1/3  symmetric + asymmetric (both directions) …')
    figs.append(_vergence_bidir(show))
    print('  2/3  fixation disparity vs. distance …')
    figs.append(_fixation_distance(show))
    print('  3/3  diplopia: fusion gate and fused vs. diplopic …')
    figs.append(_diplopia(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
