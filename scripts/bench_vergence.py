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

from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory, _IDX_VERG, _IDX_SG
from oculomotor.sim import kinematics as km
from oculomotor.analysis import ax_fmt

SHOW = '--show' in sys.argv
DT   = 0.001
IPD  = 0.064   # m, default inter-pupillary distance

# Vergence behavioral demos — full default model: AC/A + CA/C cross-links on,
# Zee SVBN saccadic vergence burst on, sensory noise on. The figures should
# reflect what a realistic near-response looks like end-to-end.
PARAMS_VERG = PARAMS_DEFAULT

# Debug variant — noiseless + cross-links off + Listing's off, but the SVBN
# saccadic vergence burst is left ON (g_svbn_conv/div at defaults) so the
# debug cascades show its contribution. Used by both sym and asym debug panels.
PARAMS_VERG_DEBUG = with_brain(
    with_sensory(PARAMS_DEFAULT,
                 sigma_canal=0.0, sigma_slip=0.0, sigma_pos=0.0, sigma_vel=0.0),
    AC_A=0.0, CA_C=0.0,
    g_burst_verg=0.0,
    sigma_acc=0.0,
    listing_gain=0.0,
)

# Asym debug variant — same as PARAMS_VERG_DEBUG but with AC/A active so the
# accommodation→vergence cross-link is exposed in the asymmetric debug cascade.
PARAMS_VERG_ASYM_DEBUG = with_brain(
    with_sensory(PARAMS_DEFAULT,
                 sigma_canal=0.0, sigma_slip=0.0, sigma_pos=0.0, sigma_vel=0.0),
    CA_C=0.0,
    g_burst_verg=0.0,
    sigma_acc=0.0,
    listing_gain=0.0,
)


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
    """Simulate symmetric vergence step. Returns (eye_L, eye_R)."""
    T  = len(t)
    p0 = np.array([0.0, 0.0, float(d_start)])
    p1 = np.array([0.0, 0.0, float(d_end)])
    pt = np.where((t >= T_STEP)[:, None], p1, p0)
    st = simulate(PARAMS_VERG, t,
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
    st = simulate(PARAMS_VERG, t,
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
    path, rp = utils.save_fig(fig, 'vergence_bidir', show=show, params=PARAMS_VERG,
                              conditions='Lit (scene_present=1), midline (sym) and lateral (asym) targets stepping in depth')

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

    for i, d in enumerate(DISTANCES):
        p_near = np.array([0.0, 0.0, float(d)])
        pt = np.where((t >= T_STEP)[:, None], p_near, p_far)

        st = simulate(PARAMS_VERG, t,
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
    path, rp = utils.save_fig(fig, 'vergence_fixation_distance', show=show, params=PARAMS_VERG,
                              conditions='Lit, midline target at fixed depths spanning 0.2–3 m')
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
    """Diplopic conditions — 3 rows × 2 cols.

    Col 0: Fused   (3 m → 0.3 m, midline)
    Col 1: Diplopic (3 m → 0.05 m, midline)

    Row 0: L eye (blue) + R eye (red) yaw.
           Fused: dashed target lines at ±geo/2; Diplopic: motor-limit lines ±NPC/2.
    Row 1: Vergence (L−R); dashed = geometric target; dotted = NPC limit.
    Row 2: Version (L+R)/2 ≈ 0 (midline target — sanity check).
    """
    T_STEP = 1.0
    TOTAL  = 6.0
    t      = np.arange(0.0, TOTAL, DT)
    T      = len(t)

    D_FUSED    = 0.3     # within fusable range  → fused
    D_DIPLOPIC = 0.05    # beyond NPC            → diplopic
    D_BASE     = 3.0     # starting fixation distance
    DEPTHS     = [D_FUSED, D_DIPLOPIC]

    p_base = np.array([0.0, 0.0, float(D_BASE)])

    eL_tr, eR_tr = {}, {}
    for d in DEPTHS:
        pt = np.where((t >= T_STEP)[:, None],
                      np.array([0.0, 0.0, float(d)]), p_base)
        st = simulate(PARAMS_VERG, t,
                      target=km.build_target(t, lin_pos=pt),
                      scene_present_array=np.ones(T), return_states=True)
        eL_tr[d] = np.array(st.plant[:, 0])
        eR_tr[d] = np.array(st.plant[:, 3])

    verg_tr  = {d: eL_tr[d] - eR_tr[d] for d in DEPTHS}
    vers_tr  = {d: (eL_tr[d] + eR_tr[d]) / 2 for d in DEPTHS}
    geo_d    = {d: _verg_angle_deg(d) for d in DEPTHS}
    geo_base = _verg_angle_deg(D_BASE)

    bp       = PARAMS_VERG.brain
    sp       = PARAMS_VERG.sensory

    CL = utils.C['eye']
    CR = utils.C['target']

    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
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
        # Y-axis: auto-scale; expand to include geometric targets or NPC motor limit.
        # If targets are within 2× the eye excursion range, include them (fused case).
        # If farther (diplopic case, targets ≫ actual excursion), extend to NPC limit
        # and draw motor-limit dashed lines instead — with a text note for fusion targets.
        npc_half    = sp.npc / 2
        ylo         = min(eL.min(), eR.min()) - 1.0
        yhi         = max(eL.max(), eR.max()) + 1.0
        eye_range   = max(yhi, abs(ylo))
        target_range = max(abs(tL), abs(tR))
        if target_range <= 2.0 * eye_range:
            ylo = min(ylo, tR - 1.0)
            yhi = max(yhi, tL + 1.0)
            ax.set_ylim(ylo, yhi)
            ax.axhline(tL, color=CL, lw=0.9, ls='-.', alpha=0.8, label=f'L target {tL:+.1f}°')
            ax.axhline(tR, color=CR, lw=0.9, ls='-.', alpha=0.8, label=f'R target {tR:+.1f}°')
        else:
            # Fusion targets off-scale: extend to NPC motor limit, draw motor-limit lines
            ylo = min(ylo, -npc_half - 2.0)
            yhi = max(yhi, +npc_half + 2.0)
            ax.set_ylim(ylo, yhi)
            ax.axhline(+npc_half, color=CL, lw=0.9, ls='-.', alpha=0.7,
                       label=f'L motor limit +{npc_half:.0f}°')
            ax.axhline(-npc_half, color=CR, lw=0.9, ls='-.', alpha=0.7,
                       label=f'R motor limit −{npc_half:.0f}°')
            ax.text(0.97, 0.97,
                    f'Fusion targets: L {tL:+.1f}°, R {tR:+.1f}° (off-scale)',
                    transform=ax.transAxes, ha='right', va='top', fontsize=7, color='#555')
        ax_fmt(ax, ylabel='Eye yaw (deg)' if col == 0 else '')
        ax.legend(fontsize=7.5)

        # Row 1: Vergence ─────────────────────────────────────────────────────
        npc_limit = geo_base + sp.npc   # absolute NPC in vergence angle
        ax = axes[1, col]
        ax.plot(t, verg_tr[d], color=CL, lw=1.5, label='Vergence L−R')
        ax.axhline(geo,       color='tomato', lw=1.0, ls='--',
                   label=f'Geo target {geo:.1f}°')
        ax.axhline(npc_limit, color='#888',   lw=0.9, ls=':',
                   label=f'NPC limit {npc_limit:.1f}°')
        vline(ax)
        lo_v = geo_base - 1.0
        hi_v = min(geo, npc_limit) + 3.0
        ax.set_ylim(lo_v, hi_v)
        ax_fmt(ax, ylabel='Vergence L−R (deg)' if col == 0 else '')
        ax.legend(fontsize=7.5)

        # Row 2: Version — ≈0 for fused midline; large if monocular fixation saccade fires
        ax = axes[2, col]
        ax.plot(t, vers_tr[d], color=utils.C['ni'], lw=1.4, label='Version (L+R)/2')
        ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.6, label='Midline target')
        vline(ax)
        v_hi = max(abs(vers_tr[d].max()), abs(vers_tr[d].min())) + 1.0
        ax.set_ylim(-max(v_hi, 2.0), max(v_hi, 2.0))
        ax_fmt(ax, ylabel='Version (L+R)/2 (deg)' if col == 0 else '',
               xlabel='Time (s)')
        ax.legend(fontsize=7.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path, rp = utils.save_fig(fig, 'vergence_diplopia', show=show, params=PARAMS_VERG,
                              conditions='Lit, midline target — fused (0.3 m) and diplopic (0.05 m) depth conditions')
    return utils.fig_meta(
        path, rp,
        title='Diplopia: Fusion Gate and Vergence Limits',
        description='3 rows × 2 cols. '
                    'Col 0: fused (3→0.3 m midline). '
                    'Col 1: diplopic (3→0.05 m midline). Both columns use default params (saccades on). '
                    'Row 0: L/R eye yaw; fused shows geometric target lines (±geo/2), '
                    'diplopic shows NPC motor-limit lines (±NPC/2) with off-scale annotation. '
                    'Row 1: vergence (L−R) with geometric target and NPC limit. '
                    'Row 2: version (L+R)/2 — auto-scaled; diplopic case shows version shift '
                    'from monocular fixation saccade driven by dominant eye retinal error.',
        expected='Fused: L/R eye yaw converges to ±geo/2 dashed targets; vergence reaches geo target; '
                 'version stays ≈ 0. '
                 'Diplopic: monocular fixation saccade fires (dominant R eye drives both eyes); '
                 'R eye lands on its geometric target (−geo/2); L eye deviates. '
                 'Vergence limited well below geometric target.',
        citation='Fender & Julesz (1967) J Opt Soc Am; '
                 'Schor (1979) Vision Res; '
                 'Shipley & Rawlings (1970) Vision Res',
    )


def _run_asym_full(t, d_start, d_end, ver_deg, T_STEP):
    """Simulate asymmetric vergence + version, return full SimState (DEBUG params:
    noiseless, AC/A active, CA/C off)."""
    T     = len(t)
    x_end = float(np.tan(np.radians(ver_deg)) * d_end)
    p0    = np.array([0.0,   0.0, float(d_start)])
    p1    = np.array([x_end, 0.0, float(d_end)])
    pt    = np.where((t >= T_STEP)[:, None], p1, p0)
    return simulate(PARAMS_VERG_ASYM_DEBUG, t,
                    target=km.build_target(t, lin_pos=pt),
                    scene_present_array=np.ones(T),
                    return_states=True)


def _run_sym_full(t, d_start, d_end, T_STEP):
    """Simulate symmetric (midline) vergence step, return full SimState (ASYM_DEBUG
    params: noiseless, AC/A active, CA/C off — same as asymmetric run for fair
    cross-comparison in the unified vergence cascade).

    Scene OFF (no full-field background), target ON (foveal target stepping in
    depth). This isolates the vergence loop from any OKR/scene-driven inputs.
    """
    T  = len(t)
    p0 = np.array([0.0, 0.0, float(d_start)])
    p1 = np.array([0.0, 0.0, float(d_end)])
    pt = np.where((t >= T_STEP)[:, None], p1, p0)
    return simulate(PARAMS_VERG_ASYM_DEBUG, t,
                    target=km.build_target(t, lin_pos=pt),
                    scene_present_array=np.zeros(T),     # no scene
                    target_present_array=np.ones(T),     # target visible
                    return_states=True)




def _vergence_cascade(show):
    """Vergence cascade — 8 rows × 4 cols covering symmetric and asymmetric vergence
    in both directions (sym conv, sym div, asym conv, asym div).

    Shows eye positions, vergence (L−R), version (L+R)/2, vergence velocity,
    OPN gate, internal vergence integrator states, vestigial x_verg_copy, and
    the reconstructed Zee SVBN burst — all noiseless DEBUG with AC/A on.
    """
    T_STEP = 1.0
    TOTAL  = 5.0
    t      = np.arange(0.0, TOTAL, DT)

    # Symmetric depths
    D_SYM_FAR, D_SYM_NEAR = 3.0, 0.3
    # Asymmetric depths + version
    D_ASYM_START_C, D_ASYM_END_C = 2.0, 0.5
    D_ASYM_START_D, D_ASYM_END_D = 0.5, 2.0
    VER = 10.0

    st_sym_c  = _run_sym_full(t,  D_SYM_FAR,        D_SYM_NEAR, T_STEP)
    st_sym_d  = _run_sym_full(t,  D_SYM_NEAR,       D_SYM_FAR,  T_STEP)
    st_asym_c = _run_asym_full(t, D_ASYM_START_C, D_ASYM_END_C,  VER, T_STEP)
    st_asym_d = _run_asym_full(t, D_ASYM_START_D, D_ASYM_END_D, -VER, T_STEP)
    sts = [st_sym_c, st_sym_d, st_asym_c, st_asym_d]

    eL = [np.array(st.plant[:, 0]) for st in sts]
    eR = [np.array(st.plant[:, 3]) for st in sts]

    verg     = [eL[i] - eR[i]         for i in range(4)]
    vers     = [(eL[i] + eR[i]) / 2   for i in range(4)]
    verg_vel = [np.gradient(v, DT)    for v in verg]

    # Internal vergence states: [x_verg(3) | x_verg_tonic(3) | x_verg_copy(3)]
    verg_st  = [np.array(st.brain[:, _IDX_VERG]) for st in sts]
    sg_st    = [np.array(st.brain[:, _IDX_SG])   for st in sts]

    x_verg_h       = [vs[:, 0] for vs in verg_st]
    x_verg_tonic_h = [vs[:, 3] for vs in verg_st]
    x_copy_h       = [vs[:, 6] for vs in verg_st]

    # OPN gate
    z_opn = [ss[:, 3] for ss in sg_st]
    z_act = [1.0 - np.clip(zop, 0.0, 100.0) / 100.0 for zop in z_opn]

    # Reconstructed Zee SVBN burst (rough)
    bp = PARAMS_VERG_ASYM_DEBUG.brain
    burst_h = []
    for i in range(4):
        x_verg_i = verg_st[i][:, 0]
        disp_est = x_verg_i / max(bp.K_verg * bp.tau_verg, 1e-6)
        is_conv  = (disp_est > 0).astype(float)
        g_eff    = is_conv * bp.g_svbn_conv + (1 - is_conv) * bp.g_svbn_div
        X_eff    = is_conv * bp.X_svbn_conv + (1 - is_conv) * bp.X_svbn_div
        burst_h.append(z_act[i] * np.sign(disp_est) * g_eff * (1 - np.exp(-np.abs(disp_est) / X_eff)))

    # Per-column metadata
    starts  = [D_SYM_FAR,        D_SYM_NEAR,       D_ASYM_START_C,  D_ASYM_START_D]
    ends    = [D_SYM_NEAR,       D_SYM_FAR,        D_ASYM_END_C,    D_ASYM_END_D]
    ver_tgt = [0.0,              0.0,              VER,             -VER]
    titles  = [
        f'Sym conv  ({D_SYM_FAR:.0f} m → {D_SYM_NEAR:.1f} m)',
        f'Sym div   ({D_SYM_NEAR:.1f} m → {D_SYM_FAR:.0f} m)',
        f'Asym R+conv  ({D_ASYM_START_C:.0f} m → {D_ASYM_END_C:.1f} m, +{VER:.0f}°)',
        f'Asym L+div   ({D_ASYM_START_D:.1f} m → {D_ASYM_END_D:.0f} m, −{VER:.0f}°)',
    ]
    geo_init = [_verg_angle_deg(s) for s in starts]
    geo_tgt  = [_verg_angle_deg(e) for e in ends]

    NROWS = 8
    fig, axes = plt.subplots(NROWS, 4, figsize=(20, 20), sharex=True)
    fig.suptitle('Vergence Cascade — symmetric + asymmetric, both directions  (noiseless, AC/A on, CA/C off)',
                 fontsize=12, fontweight='bold')

    CL  = utils.C['eye']
    CR  = utils.C['target']
    CBR = '#8B4513'

    def vl(ax): ax.axvline(T_STEP, color='gray', lw=0.8, ls=':')

    for col in range(4):
        axes[0, col].set_title(titles[col], fontsize=9)

        # Row 0: per-eye position
        ax = axes[0, col]
        ax.plot(t, eL[col], color=CL, lw=1.3, label='L eye')
        ax.plot(t, eR[col], color=CR, lw=1.3, ls='--', label='R eye')
        vl(ax)
        ax_fmt(ax, ylabel='Eye yaw (deg)' if col == 0 else '')
        ax.legend(fontsize=7)

        # Row 1: vergence (L-R)
        ax = axes[1, col]
        ax.plot(t, verg[col], color=CL, lw=1.5)
        ax.axhline(geo_tgt[col],  color='tomato', lw=0.9, ls='--',
                   label=f'Target {geo_tgt[col]:.1f}°')
        ax.axhline(geo_init[col], color='gray',   lw=0.8, ls=':',
                   label=f'Start  {geo_init[col]:.1f}°')
        vl(ax)
        lo = min(geo_tgt[col], geo_init[col]) - 1.0
        hi = max(float(np.max(verg[col])), geo_tgt[col], geo_init[col]) + 2.0
        ax.set_ylim(lo, hi)
        ax_fmt(ax, ylabel='Vergence L−R (deg)' if col == 0 else '')
        ax.legend(fontsize=7)

        # Row 2: version (L+R)/2
        ax = axes[2, col]
        ax.plot(t, vers[col], color=utils.C['ni'], lw=1.4)
        ax.axhline(ver_tgt[col], color='gray', lw=0.9, ls='--',
                   label=f'Target {ver_tgt[col]:+.0f}°')
        vl(ax)
        lim = abs(ver_tgt[col]) + 3.0
        ax.set_ylim(-lim, lim)
        ax_fmt(ax, ylabel='Version (L+R)/2 (deg)' if col == 0 else '')
        ax.legend(fontsize=7)

        # Row 3: vergence velocity
        ax = axes[3, col]
        ax.plot(t, verg_vel[col], color=CBR, lw=1.3, label='Verg vel')
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        vl(ax)
        peak = float(np.max(np.abs(verg_vel[col][t > T_STEP - 0.05])))
        margin = max(peak * 0.15, 2.0)
        ax.set_ylim(-peak - margin, peak + margin)
        ax_fmt(ax, ylabel='d(Verg)/dt (deg/s)' if col == 0 else '')
        ax.legend(fontsize=7)

        # Row 4: OPN gate
        ax = axes[4, col]
        ax.plot(t, z_act[col], color='#555555', lw=1.3, label='z_act (OPN)')
        ax.axhline(0.5, color='gray', lw=0.7, ls=':', alpha=0.6)
        vl(ax)
        ax.set_ylim(-0.05, 1.1)
        ax_fmt(ax, ylabel='z_act' if col == 0 else '')
        ax.legend(fontsize=7)

        # Row 5: vergence + tonic-vergence integrators
        ax = axes[5, col]
        ax.plot(t, x_verg_h[col],       color=CBR, lw=1.3, label='x_verg[H]')
        ax.plot(t, x_verg_tonic_h[col], color=CBR, lw=1.3, ls='--', label='x_verg_tonic[H]')
        ax.axhline(0, color='gray', lw=0.7, ls=':', alpha=0.5)
        vl(ax)
        ax_fmt(ax, ylabel='Verg integrators (deg)' if col == 0 else '')
        ax.legend(fontsize=7)

        # Row 6: x_verg_copy[H]
        ax = axes[6, col]
        ax.plot(t, x_copy_h[col], color='#555555', lw=1.3, label='x_verg_copy[H]')
        ax.axhline(0, color='gray', lw=0.7, ls=':', alpha=0.5)
        vl(ax)
        ax_fmt(ax, ylabel='x_verg_copy (deg)' if col == 0 else '')
        ax.legend(fontsize=7)

        # Row 7: Zee SVBN burst (reconstructed)
        ax = axes[7, col]
        ax.plot(t, burst_h[col], color=CBR, lw=1.3, label='Zee burst[H]')
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        vl(ax)
        peak_b = float(np.max(np.abs(burst_h[col])))
        margin_b = max(peak_b * 0.15, 0.5)
        ax.set_ylim(-peak_b - margin_b, peak_b + margin_b)
        ax_fmt(ax, ylabel='Verg burst (deg/s)' if col == 0 else '', xlabel='Time (s)')
        ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path, rp = utils.save_fig(fig, 'vergence_cascade', show=show, params=PARAMS_VERG_ASYM_DEBUG)
    return utils.fig_meta(
        path, rp,
        title='Vergence Cascade (4 conditions)',
        description='8 rows × 4 cols. Cols: sym conv (3→0.3 m), sym div (0.3→3 m), '
                    'asym R+conv (2→0.5 m, +10°), asym L+div (0.5→2 m, −10°). '
                    'Rows: eye position, vergence, version, vergence velocity, OPN gate, '
                    'integrators (x_verg + x_verg_tonic [H]), x_verg_copy, Zee SVBN burst.',
        expected='Symmetric cols: version stays near 0; asymmetric: version follows target. '
                 'Vergence settles at geometric target. SVBN fires only during saccades.',
        citation='Zee et al. (1992) J Neurophysiol; Collewijn et al. (1988) J Physiol; Schor (1979) Vision Res',
    )


def _depth_for_vergence(verg_deg):
    """Distance (m) that produces a given symmetric vergence angle (deg) at default IPD."""
    half = np.radians(verg_deg) / 2.0
    return IPD / 2.0 / np.tan(half)


def _main_sequence(show):
    """Vergence main sequence: peak velocity vs. amplitude, symmetric and asymmetric.

    Symmetric (no version saccade): tests slow disparity vergence speed.
    Asymmetric (with concurrent version saccade): tests Zee SVBN facilitation.
    Convergence and divergence amplitudes both probed; conv expected stronger.
    """
    AMPS_DEG = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0])
    T_STEP   = 0.5
    TOTAL    = 4.0
    t        = np.arange(0.0, TOTAL, DT)
    T        = len(t)
    VER_ASYM = 10.0   # version saccade amplitude (deg) for asymmetric

    def _peak_verg_vel(eL, eR):
        verg     = eL - eR
        verg_vel = np.gradient(verg, DT)
        # Find peak after the step, look in [T_STEP, T_STEP + 1.5s]
        idx = (t >= T_STEP) & (t <= T_STEP + 1.5)
        return float(np.max(np.abs(verg_vel[idx])))

    sym_conv_peaks = []
    sym_div_peaks  = []
    asym_conv_peaks = []
    asym_div_peaks  = []

    for amp in AMPS_DEG:
        # Symmetric convergence: tonic to demanded amp
        d_start = _depth_for_vergence(2.0)
        d_end   = _depth_for_vergence(amp)
        eL_sc, eR_sc = _run_sym(t, d_start, d_end, T_STEP)
        sym_conv_peaks.append(_peak_verg_vel(eL_sc, eR_sc))

        # Symmetric divergence
        eL_sd, eR_sd = _run_sym(t, d_end, d_start, T_STEP)
        sym_div_peaks.append(_peak_verg_vel(eL_sd, eR_sd))

        # Asymmetric convergence: same depth change + 10° version
        eL_ac, eR_ac = _run_asym(t, d_start, d_end, VER_ASYM, T_STEP)
        asym_conv_peaks.append(_peak_verg_vel(eL_ac, eR_ac))

        # Asymmetric divergence
        eL_ad, eR_ad = _run_asym(t, d_end, d_start, -VER_ASYM, T_STEP)
        asym_div_peaks.append(_peak_verg_vel(eL_ad, eR_ad))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Vergence Main Sequence — peak vergence velocity vs. amplitude',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(AMPS_DEG, sym_conv_peaks,  'o-', color=utils.C['eye'],   lw=1.5, ms=8, label='Convergence')
    ax.plot(AMPS_DEG, sym_div_peaks,   's--', color=utils.C['target'], lw=1.5, ms=8, label='Divergence')
    ax_fmt(ax, ylabel='Peak vergence velocity (deg/s)', xlabel='Vergence amplitude (deg)')
    ax.set_title('Symmetric (no version saccade) — slow vergence', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(AMPS_DEG, asym_conv_peaks, 'o-', color=utils.C['eye'],   lw=1.5, ms=8, label='Convergence (with version)')
    ax.plot(AMPS_DEG, asym_div_peaks,  's--', color=utils.C['target'], lw=1.5, ms=8, label='Divergence (with version)')
    ax_fmt(ax, ylabel='Peak vergence velocity (deg/s)', xlabel='Vergence amplitude (deg)')
    ax.set_title(f'Asymmetric (concurrent {VER_ASYM:+.0f}° version) — Zee SVBN burst', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path, rp = utils.save_fig(fig, 'vergence_main_sequence', show=show, params=PARAMS_VERG,
                              conditions='Lit, vergence amplitudes 1°–18° (main sequence, sym + asym)')
    return utils.fig_meta(
        path, rp,
        title='Vergence Main Sequence (peak velocity vs. amplitude)',
        description='Peak vergence velocity for amplitudes 2–16°, both directions, '
                    'with and without a concurrent version saccade. '
                    'Left: symmetric (slow vergence only). Right: asymmetric (Zee SVBN burst active).',
        expected='Symmetric: peak velocity grows roughly linearly with amplitude. '
                 'Asymmetric: peak velocity 2–3× higher than symmetric (Zee facilitation). '
                 'Convergence > divergence at every amplitude (asymmetric saturating gain).',
        citation='Zee et al. (1992) J Neurophysiol; Rashbass & Westheimer (1961) J Physiol; '
                 'Collewijn et al. (1988) J Physiol',
    )


def run(show=False):
    print('\n=== Vergence ===')
    figs = []
    print('  1/5  symmetric + asymmetric (both directions) …')
    figs.append(_vergence_bidir(show))
    print('  2/5  fixation disparity vs. distance …')
    figs.append(_fixation_distance(show))
    print('  3/5  diplopia: fusion gate and fused vs. diplopic …')
    figs.append(_diplopia(show))
    print('  4/5  vergence cascade (4 conditions) …')
    figs.append(_vergence_cascade(show))
    print('  5/5  main sequence (sym + asym) …')
    figs.append(_main_sequence(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
