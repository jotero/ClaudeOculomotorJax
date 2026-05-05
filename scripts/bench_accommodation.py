"""Vergence–accommodation interaction benchmarks.

Tests the AC/A and CA/C cross-links and lens-plant dynamics using
optical interventions (prisms and lenses).

Panels
------
1. Near-step binocular: target 3 m → 0.4 m; vergence and accommodation
   should both increase to match the new target depth.
2. Lens-driven accommodation (monocular +lens): +2 D lens on right eye
   forces accommodation up → AC/A drives vergence convergence even with
   no change in binocular disparity.
3. Prism-driven vergence → CA/C: base-out prism on one eye (forcing
   divergence error) drives fusional vergence → via CA/C reduces
   accommodation demand.
4. Cover test: occlude one eye; vergence drifts to tonic, accommodation
   drifts toward dark focus on the covered side.

Usage:
    python -X utf8 scripts/bench_accommodation.py
    python -X utf8 scripts/bench_accommodation.py --show
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as utils

import numpy as np
import jax
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, _IDX_VERG, with_brain
from oculomotor.models.brain_models.brain_model import _IDX_ACC
from oculomotor.sim import kinematics as km
from oculomotor.sim.stimuli import build_cover_flags
from oculomotor.analysis import ax_fmt

SHOW = '--show' in sys.argv
DT   = 0.001
IPD  = 0.064   # m, default inter-pupillary distance

SECTION = dict(
    id='accommodation',
    title='6. Vergence–Accommodation',
    description=(
        'AC/A and CA/C cross-links: accommodation driven by binocular disparity (near step), '
        'lens-forced accommodation driving vergence via AC/A, and prism-forced vergence '
        'driving accommodation via CA/C.  Plant dynamics follow Read & Schor (2022): '
        'τ_plant ≈ 0.156 s, τ_fast ≈ 2.5 s, G_fast = 8.'
    ),
)

KEY = jax.random.PRNGKey(0)


def _verg_angle_deg(depth_m):
    return 2.0 * np.degrees(np.arctan(IPD / 2.0 / depth_m))


def _acc_demand_d(depth_m):
    return 1.0 / depth_m


def _stim_ax(ax, t, stim, ylabel, step_t=None, color='steelblue', ylim=None):
    """Plot a stimulus trace on a small top axes with a step marker."""
    ax.plot(t, stim, color=color, lw=1.5, drawstyle='steps-post')
    if step_t is not None:
        ax.axvline(step_t, color='gray', lw=0.8, ls=':')
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_facecolor('#f8f8f8')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── Panel 0a: Isolated accommodation step response (AC/A = CA/C = 0) ───────────
# Tests Step 3 of the vergence/accommodation rebuild: accommodation loop alone,
# without any vergence cross-coupling.

PARAMS_ACC_ONLY = with_brain(PARAMS_DEFAULT, AC_A=0.0, CA_C=0.0)


def _isolated_step(show):
    """Defocus step from far (6 m, 0.17 D) to near (0.4 m, 2.5 D) and back.

    AC/A and CA/C disabled — accommodation loop in isolation.
    """
    T_STEP_NEAR = 1.0
    T_STEP_FAR  = 4.0
    TOTAL       = 7.0
    t = np.arange(0.0, TOTAL, DT)
    T = len(t)

    p_far  = np.array([0.0, 0.0, 6.0])
    p_near = np.array([0.0, 0.0, 0.4])
    pt = np.tile(p_far, (T, 1))
    pt[(t >= T_STEP_NEAR) & (t < T_STEP_FAR)] = p_near

    st = simulate(PARAMS_ACC_ONLY, t,
                  target=km.build_target(t, lin_pos=pt),
                  scene_present_array=np.ones(T),
                  return_states=True, key=KEY)

    acc    = np.array(st.acc_plant[:, 0])
    demand = 1.0 / np.linalg.norm(pt, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    fig.suptitle('Isolated accommodation: far → near → far step (AC/A = CA/C = 0)',
                 fontsize=11, fontweight='bold')

    ax = axes[0]
    ax.plot(t, demand, 'k--', lw=1.0, alpha=0.7, label='Demand (1/z)')
    ax.plot(t, acc,    color=utils.C['eye'], lw=1.5, label='Accommodation')
    ax.axvline(T_STEP_NEAR, color='gray', lw=0.8, ls=':')
    ax.axvline(T_STEP_FAR,  color='gray', lw=0.8, ls=':')
    ax.set_ylim(-0.3, 3.0)
    ax_fmt(ax, ylabel='Diopters (D)')
    ax.legend(fontsize=9)
    ax.set_title(f'Far (6 m, 0.17 D) → Near (0.4 m, 2.5 D) at t={T_STEP_NEAR}; back at t={T_STEP_FAR}',
                 fontsize=9)

    ax = axes[1]
    err = demand - acc
    ax.plot(t, err, color=utils.C['target'], lw=1.3, label='Defocus error (demand − acc)')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axvline(T_STEP_NEAR, color='gray', lw=0.8, ls=':')
    ax.axvline(T_STEP_FAR,  color='gray', lw=0.8, ls=':')
    ax_fmt(ax, ylabel='Defocus error (D)', xlabel='Time (s)')
    ax.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path, rp = utils.save_fig(fig, 'accommodation_isolated_step', show=show, params=PARAMS_ACC_ONLY,
                              conditions='Lit, defocus step (cross-links AC/A=0, CA/C=0 — accommodation isolated)')
    return utils.fig_meta(path, rp,
        title='Isolated accommodation step response (no AC/A, no CA/C)',
        description='Defocus step 0.17 D → 2.5 D → 0.17 D with cross-coupling disabled. '
                    'Top: demand vs. accommodation. Bottom: defocus error.',
        expected='Latency ~300 ms; rise to ~85% of demand within ~1 s; small SS '
                 'residual error ~10–15% of demand. Symmetric far→near and near→far.',
        citation='Read, Kaspiris-Rousellis et al. (2022) J Vision 22(9):4',
    )


def _isolated_amplitudes(show):
    """Step responses to multiple amplitudes (0.5–4 D) with cross-coupling disabled."""
    AMPS_D = [0.5, 1.0, 2.0, 3.0, 4.0]
    T_STEP = 0.5
    T_TOTAL = 5.0
    t = np.arange(0.0, T_TOTAL, DT)
    T = len(t)

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(AMPS_D), vmax=max(AMPS_D))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Isolated accommodation: amplitude scaling (AC/A = CA/C = 0)',
                 fontsize=11, fontweight='bold')

    ss_acc = []
    for amp in AMPS_D:
        d_near = 1.0 / amp
        p_far  = np.array([0.0, 0.0, 6.0])
        p_near = np.array([0.0, 0.0, d_near])
        pt = np.where((t >= T_STEP)[:, None], p_near, p_far)
        st = simulate(PARAMS_ACC_ONLY, t,
                      target=km.build_target(t, lin_pos=pt),
                      scene_present_array=np.ones(T),
                      return_states=True, key=KEY)
        acc = np.array(st.acc_plant[:, 0])
        color = cmap(norm(amp))
        axes[0].plot(t, acc, color=color, lw=1.4,
                     label=f'{amp:.1f} D ({d_near*100:.1f} cm)')
        axes[0].axhline(amp, color=color, lw=0.7, ls=':', alpha=0.5)
        ss_acc.append(float(acc[t > 4.0].mean()))

    axes[0].axvline(T_STEP, color='gray', lw=0.8, ls=':')
    ax_fmt(axes[0], ylabel='Accommodation (D)', xlabel='Time (s)')
    axes[0].legend(fontsize=8, loc='lower right')
    axes[0].set_title('Time-courses (dotted = demand)', fontsize=9)

    axes[1].plot([0, max(AMPS_D)], [0, max(AMPS_D)], 'k--', lw=0.8, alpha=0.5,
                 label='Unity (perfect tracking)')
    axes[1].plot(AMPS_D, ss_acc, 'o-', color=utils.C['eye'], lw=1.5, ms=8,
                 label='Model SS accommodation')
    for amp, ss in zip(AMPS_D, ss_acc):
        gain = ss / amp if amp > 0 else 1.0
        axes[1].annotate(f'g={gain:.2f}', xy=(amp, ss), xytext=(5, -10),
                         textcoords='offset points', fontsize=8, color='#555')
    axes[1].set_xlabel('Demand (D)')
    axes[1].set_ylabel('SS accommodation (D)')
    axes[1].set_title('Amplitude scaling (gain = SS / demand)', fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, max(AMPS_D) + 0.3)
    axes[1].set_ylim(0, max(AMPS_D) + 0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path, rp = utils.save_fig(fig, 'accommodation_isolated_amplitudes', show=show, params=PARAMS_ACC_ONLY,
                              conditions='Lit, defocus steps at multiple amplitudes (AC/A=0, CA/C=0)')
    return utils.fig_meta(path, rp,
        title='Isolated accommodation amplitude scaling',
        description='Step responses at 0.5, 1, 2, 3, 4 D demand with cross-coupling '
                    'disabled. Left: time-courses. Right: SS gain vs demand.',
        expected='Linear scaling across amplitudes; gain ~0.85–0.90 of demand at '
                 'all amplitudes. No saturation below 4 D. Rise TC ~1 s.',
        citation='Hung & Semmlow (1980); Read & Schor (2022)',
    )


def _isolated_dark_focus(show):
    """Sustained near work then darkness — drift back to tonic_acc."""
    T_NEAR_ON  = 0.5
    T_NEAR_OFF = 10.5
    T_TOTAL    = 50.0
    t = np.arange(0.0, T_TOTAL, DT)
    T = len(t)

    p_far  = np.array([0.0, 0.0, 6.0])
    p_near = np.array([0.0, 0.0, 0.25])    # 4 D demand
    pt = np.tile(p_far, (T, 1))
    pt[(t >= T_NEAR_ON) & (t < T_NEAR_OFF)] = p_near

    scene_present  = np.ones(T)
    target_present = np.ones(T)
    scene_present[t >= T_NEAR_OFF]  = 0.0
    target_present[t >= T_NEAR_OFF] = 0.0

    st = simulate(PARAMS_ACC_ONLY, t,
                  target=km.build_target(t, lin_pos=pt),
                  scene_present_array=scene_present,
                  target_present_array=target_present,
                  return_states=True, key=KEY)

    acc = np.array(st.acc_plant[:, 0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle('Dark-focus drift after sustained near work (AC/A = CA/C = 0)',
                 fontsize=11, fontweight='bold')
    demand_trace = np.where((t >= T_NEAR_ON) & (t < T_NEAR_OFF), 4.0,
                            np.where(t >= T_NEAR_OFF, np.nan, 1.0/6.0))
    ax.plot(t, demand_trace, 'k--', lw=1.0, alpha=0.6, label='Demand (visible)')
    ax.plot(t, acc, color=utils.C['eye'], lw=1.5, label='Accommodation')
    ax.axhline(PARAMS_ACC_ONLY.brain.tonic_acc, color=utils.C['ni'], lw=0.9, ls=':',
               label=f'tonic_acc = {PARAMS_ACC_ONLY.brain.tonic_acc:.2f} D')
    ax.axvline(T_NEAR_ON,  color='gray', lw=0.7, ls=':')
    ax.axvline(T_NEAR_OFF, color='gray', lw=0.7, ls=':')
    ax.text(T_NEAR_ON + 0.2, 0.05, 'near work (4 D)', fontsize=8, color='#555')
    ax.text(T_NEAR_OFF + 0.5, 0.05, 'darkness (open loop)', fontsize=8, color='#555')
    ax_fmt(ax, ylabel='Diopters (D)', xlabel='Time (s)')
    ax.set_ylim(-0.2, 4.5)
    ax.legend(fontsize=9, loc='upper right')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path, rp = utils.save_fig(fig, 'accommodation_isolated_dark_focus', show=show, params=PARAMS_ACC_ONLY,
                              conditions='Dark (no scene), no target — dark-focus / tonic accommodation level')
    return utils.fig_meta(path, rp,
        title='Dark-focus drift after sustained near work',
        description='10 s of sustained 4 D near work followed by darkness '
                    '(scene + target removed). Accommodation drifts back to tonic_acc.',
        expected='Drift to tonic_acc with TC ≈ tau_acc_slow (30 s). Slight overshoot '
                 'of tonic from sustained near work (slow integrator wound up).',
        citation='Schor (1979); Read & Schor (2022)',
    )


# ── Panel 1: Near-step — depth change drives vergence + accommodation ──────────

def _near_step(show):
    """Target steps from 3 m to 0.4 m at t=1 s.

    Expected:
      - Vergence (L−R yaw) increases from ~1.2° to ~9.2°  (geometric)
      - Accommodation (x_plant) rises from ~0.33 D to ~2.5 D
      - Both settle within ~2 s after step
    """
    T_STEP = 1.0
    TOTAL  = 6.0
    t      = np.arange(0.0, TOTAL, DT)
    T      = len(t)

    D_FAR  = 3.0    # m
    D_NEAR = 0.40   # m

    p0 = np.array([0.0, 0.0, D_FAR])
    p1 = np.array([0.0, 0.0, D_NEAR])
    pt = np.where((t >= T_STEP)[:, None], p1, p0)

    st = simulate(PARAMS_DEFAULT, t,
                  target=km.build_target(t, lin_pos=pt),
                  scene_present_array=np.ones(T),
                  return_states=True, key=KEY)

    eye_L   = np.array(st.plant[:, 0])   # L yaw (deg)
    eye_R   = np.array(st.plant[:, 3])   # R yaw (deg)
    vergence = eye_L - eye_R             # positive = converged
    acc_plant= np.array(st.acc_plant[:, 0])   # actual lens acc (D)

    g_far  = _verg_angle_deg(D_FAR)
    g_near = _verg_angle_deg(D_NEAR)

    # Stimulus trace: target depth over time
    depth_trace = np.where(t >= T_STEP, D_NEAR, D_FAR)

    fig = plt.figure(figsize=(8, 9), constrained_layout=True)
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.6, 2, 2, 2], figure=fig)
    ax_stim = fig.add_subplot(gs[0])
    ax0     = fig.add_subplot(gs[1], sharex=ax_stim)
    ax1     = fig.add_subplot(gs[2], sharex=ax_stim)
    ax2     = fig.add_subplot(gs[3], sharex=ax_stim)

    fig.suptitle('Near-step: 3 m → 0.4 m  (vergence + accommodation)',
                 fontsize=11, fontweight='bold')

    _stim_ax(ax_stim, t, depth_trace, 'Depth (m)', step_t=T_STEP,
             ylim=(0.0, D_FAR + 0.5))
    plt.setp(ax_stim.get_xticklabels(), visible=False)

    ax_stim.set_title(f'Stimulus: target steps {D_FAR} m → {D_NEAR} m at t = {T_STEP} s',
                      fontsize=8, loc='left', pad=2)

    ax0.plot(t, eye_L, color=utils.C['eye'],   lw=1.3, label='L eye yaw')
    ax0.plot(t, eye_R, color=utils.C['target'], lw=1.3, ls='--', label='R eye yaw')
    ax0.axvline(T_STEP, color='gray', lw=0.8, ls=':')
    ax_fmt(ax0, ylabel='Eye yaw (deg)')
    ax0.legend(fontsize=8)
    plt.setp(ax0.get_xticklabels(), visible=False)

    ax1.plot(t, vergence, color=utils.C['vs'], lw=1.5, label='Vergence L−R')
    ax1.axhline(g_near, color='tomato', lw=0.9, ls='--',
               label=f'Geometric {g_near:.1f}° @ {D_NEAR} m')
    ax1.axhline(g_far,  color='gray',   lw=0.8, ls=':',
               label=f'Geometric {g_far:.1f}°  @ {D_FAR} m')
    ax1.axvline(T_STEP, color='gray', lw=0.8, ls=':')
    ax1.set_ylim(0, g_near + 2)
    ax_fmt(ax1, ylabel='Vergence (deg)')
    ax1.legend(fontsize=8)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.plot(t, acc_plant, color=utils.C['ni'], lw=1.5, label='Lens accommodation (x_plant)')
    ax2.axhline(_acc_demand_d(D_NEAR), color='tomato', lw=0.9, ls='--',
               label=f'Target {_acc_demand_d(D_NEAR):.2f} D @ {D_NEAR} m')
    ax2.axhline(_acc_demand_d(D_FAR),  color='gray',   lw=0.8, ls=':',
               label=f'Target {_acc_demand_d(D_FAR):.2f} D @ {D_FAR} m')
    ax2.axvline(T_STEP, color='gray', lw=0.8, ls=':')
    ax2.set_ylim(-0.2, _acc_demand_d(D_NEAR) + 0.5)
    ax_fmt(ax2, ylabel='Accommodation (D)', xlabel='Time (s)')
    ax2.legend(fontsize=8)

    path, rp = utils.save_fig(fig, 'accommodation_near_step', show=show, params=PARAMS_DEFAULT,
                              conditions='Lit, target stepping near (defocus + disparity stimulus, full cross-links)')
    return utils.fig_meta(path, rp,
        title='Near-step: binocular vergence + accommodation tracking',
        description='Target steps 3 m → 0.4 m at t=1 s. Top: L/R eye yaw. '
                    'Middle: vergence (L−R). Bottom: lens accommodation x_plant.',
        expected=f'Vergence reaches ≥ {g_near*0.7:.1f}° ({g_near:.1f}° geometric) within 4 s. '
                 f'Accommodation reaches ≥ {_acc_demand_d(D_NEAR)*0.7:.2f} D (target 2.5 D) within 4 s.',
        citation='Schor CM (1979) Vision Res 19:1359; Read et al. (2022) J Vision 22(9):4',
    )


# ── Panel 2: Lens-driven AC/A — monocular plus lens converges via AC/A ────────

def _lens_aca(show):
    """Plus lens (+2 D) added to right eye at t=1 s, scene/target at 3 m.

    Expected:
      - Accommodation increases to accommodate the extra demand from the R lens
      - AC/A cross-link drives additional convergence (vergence increases)
      - Without AC/A (AC_A=0) vergence should NOT change (control)
    """
    T_STEP = 1.0
    TOTAL  = 8.0
    t      = np.arange(0.0, TOTAL, DT)
    T      = len(t)

    D_FAR = 3.0
    LENS_D = 2.0  # plus lens strength (D)
    p0 = np.array([0.0, 0.0, D_FAR])

    lens_R = np.where(t >= T_STEP, LENS_D, 0.0).astype(np.float32)
    lens_L = np.zeros(T, dtype=np.float32)

    # With AC/A (default params)
    st_aca = simulate(PARAMS_DEFAULT, t,
                      target=km.build_target(t, lin_pos=np.tile(p0, (T, 1))),
                      scene_present_array=np.ones(T),
                      lens_L_array=lens_L, lens_R_array=lens_R,
                      return_states=True, key=KEY)

    # Without AC/A (AC_A=0)
    p_no_aca = with_brain(PARAMS_DEFAULT, AC_A=0.0)
    st_noaca = simulate(p_no_aca, t,
                        target=km.build_target(t, lin_pos=np.tile(p0, (T, 1))),
                        scene_present_array=np.ones(T),
                        lens_L_array=lens_L, lens_R_array=lens_R,
                        return_states=True, key=KEY)

    def _verg(st):
        return np.array(st.plant[:, 0] - st.plant[:, 3])
    def _acc(st):
        return np.array(st.acc_plant[:, 0])

    # Stimulus trace: right-eye lens power
    lens_trace = np.where(t >= T_STEP, LENS_D, 0.0)

    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.6, 2, 2], figure=fig)
    ax_stim = fig.add_subplot(gs[0])
    ax0     = fig.add_subplot(gs[1], sharex=ax_stim)
    ax1     = fig.add_subplot(gs[2], sharex=ax_stim)

    fig.suptitle(f'Lens-driven AC/A: +{LENS_D} D right lens added at t={T_STEP} s  (target at {D_FAR} m)',
                 fontsize=10, fontweight='bold')

    _stim_ax(ax_stim, t, lens_trace, 'R lens (D)', step_t=T_STEP,
             ylim=(-0.2, LENS_D + 0.5))
    ax_stim.set_title(f'Stimulus: +{LENS_D} D plus lens on right eye at t = {T_STEP} s',
                      fontsize=8, loc='left', pad=2)
    plt.setp(ax_stim.get_xticklabels(), visible=False)

    ax0.plot(t, _acc(st_aca),   color=utils.C['vs'],  lw=1.5, label='Accommodation (with AC/A)')
    ax0.plot(t, _acc(st_noaca), color=utils.C['eye'], lw=1.3, ls='--', label='Accommodation (AC/A=0)')
    ax0.axvline(T_STEP, color='gray', lw=0.8, ls=':')
    ax0.axhline(_acc_demand_d(D_FAR),        color='gray',   lw=0.8, ls=':',
               label=f'Far demand {_acc_demand_d(D_FAR):.2f} D')
    ax0.axhline(_acc_demand_d(D_FAR)+LENS_D, color='tomato', lw=0.9, ls='--',
               label=f'R-eye demand {_acc_demand_d(D_FAR)+LENS_D:.2f} D')
    ax_fmt(ax0, ylabel='Accommodation (D)')
    ax0.legend(fontsize=8)
    plt.setp(ax0.get_xticklabels(), visible=False)

    ax1.plot(t, _verg(st_aca),   color=utils.C['vs'],  lw=1.5, label='Vergence (with AC/A)')
    ax1.plot(t, _verg(st_noaca), color=utils.C['eye'], lw=1.3, ls='--', label='Vergence (AC/A=0)')
    g0 = _verg_angle_deg(D_FAR)
    ax1.axhline(g0,  color='gray', lw=0.8, ls=':', label=f'Geometric {g0:.1f}° @ {D_FAR} m')
    ax1.axvline(T_STEP, color='gray', lw=0.8, ls=':')
    ax_fmt(ax1, ylabel='Vergence L−R (deg)', xlabel='Time (s)')
    ax1.legend(fontsize=8)

    path, rp = utils.save_fig(fig, 'accommodation_lens_aca', show=show, params=PARAMS_DEFAULT,
                              conditions='Lit, lens-only stimulus — AC/A drives vergence (compared with AC_A=0)')
    return utils.fig_meta(path, rp,
        title='Lens-driven AC/A: monocular plus lens converges eyes',
        description=f'+{LENS_D} D right eye lens added at t={T_STEP} s; target at {D_FAR} m. '
                    'Top: accommodation (with vs without AC/A). Bottom: vergence.',
        expected='Accommodation rises ≥ 1 D. With AC/A: vergence increases above geometric. '
                 'Without AC/A: vergence stays at geometric value.',
        citation='Schor CM (1979) Vision Res 19:1359; Morgan (1944)',
    )


# ── Panel 3: Prism-driven CA/C — base-out prism drives convergence → CA/C acc ──

def _prism_cac(show):
    """Base-out prism on both eyes at t=1 s, target at 0.5 m.

    Base-out prism forces an artificial exo-deviation; the fusional
    convergence response increases x_verg → CA/C drives extra
    accommodation.  Without CA/C (CA_C=0) accommodation should be unaffected.
    """
    T_STEP  = 1.0
    TOTAL   = 8.0
    t       = np.arange(0.0, TOTAL, DT)
    T       = len(t)

    D_MID   = 0.5   # m  (2 D target)
    PRISM_H = 4.0   # deg base-out (inward-rotated) = convergence demand

    p0 = np.array([0.0, 0.0, D_MID])

    # Base-out prism on right eye: displaces image nasally → eye must converge more
    # Prism [yaw, pitch, roll]: positive yaw = rightward shift of apparent target
    prism_R = np.zeros((T, 3), dtype=np.float32)
    prism_R[t >= T_STEP, 0] = -PRISM_H    # negative yaw = nasal shift → convergence demand
    prism_L = np.zeros((T, 3), dtype=np.float32)

    # With CA/C (default)
    st_cac = simulate(PARAMS_DEFAULT, t,
                      target=km.build_target(t, lin_pos=np.tile(p0, (T, 1))),
                      scene_present_array=np.ones(T),
                      prism_L_array=prism_L, prism_R_array=prism_R,
                      return_states=True, key=KEY)

    # Without CA/C
    p_no_cac = with_brain(PARAMS_DEFAULT, CA_C=0.0)
    st_nocac = simulate(p_no_cac, t,
                        target=km.build_target(t, lin_pos=np.tile(p0, (T, 1))),
                        scene_present_array=np.ones(T),
                        prism_L_array=prism_L, prism_R_array=prism_R,
                        return_states=True, key=KEY)

    def _verg(st): return np.array(st.plant[:, 0] - st.plant[:, 3])
    def _acc(st):  return np.array(st.acc_plant[:, 0])

    # Stimulus trace: prism power (right eye)
    prism_trace = np.where(t >= T_STEP, PRISM_H, 0.0)

    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.6, 2, 2], figure=fig)
    ax_stim = fig.add_subplot(gs[0])
    ax0     = fig.add_subplot(gs[1], sharex=ax_stim)
    ax1     = fig.add_subplot(gs[2], sharex=ax_stim)

    fig.suptitle(f'Prism-driven CA/C: {PRISM_H}° base-out R prism at t={T_STEP} s  (target at {D_MID} m)',
                 fontsize=10, fontweight='bold')

    _stim_ax(ax_stim, t, prism_trace, 'Prism BO\n(deg)', step_t=T_STEP,
             ylim=(-0.5, PRISM_H + 1.0))
    ax_stim.set_title(
        f'Stimulus: {PRISM_H}° base-out prism on right eye at t = {T_STEP} s  '
        f'(target fixed at {D_MID} m = {_acc_demand_d(D_MID):.1f} D)',
        fontsize=8, loc='left', pad=2)
    plt.setp(ax_stim.get_xticklabels(), visible=False)

    g0 = _verg_angle_deg(D_MID)
    ax0.plot(t, _verg(st_cac),  color=utils.C['vs'],  lw=1.5, label='Vergence (with CA/C)')
    ax0.plot(t, _verg(st_nocac),color=utils.C['eye'], lw=1.3, ls='--', label='Vergence (CA/C=0)')
    ax0.axhline(g0,        color='gray',   lw=0.8, ls=':', label=f'Geometric {g0:.1f}° @ {D_MID} m')
    ax0.axhline(g0+PRISM_H,color='tomato', lw=0.9, ls='--', label=f'With prism demand {g0+PRISM_H:.1f}°')
    ax0.axvline(T_STEP, color='gray', lw=0.8, ls=':')
    ax_fmt(ax0, ylabel='Vergence L−R (deg)')
    ax0.legend(fontsize=8)
    plt.setp(ax0.get_xticklabels(), visible=False)

    ax1.plot(t, _acc(st_cac),  color=utils.C['vs'],  lw=1.5, label='Accommodation (with CA/C)')
    ax1.plot(t, _acc(st_nocac),color=utils.C['eye'], lw=1.3, ls='--', label='Accommodation (CA/C=0)')
    ax1.axhline(_acc_demand_d(D_MID), color='gray', lw=0.8, ls=':',
               label=f'Optical demand {_acc_demand_d(D_MID):.1f} D')
    ax1.axvline(T_STEP, color='gray', lw=0.8, ls=':')
    ax_fmt(ax1, ylabel='Accommodation (D)', xlabel='Time (s)')
    ax1.legend(fontsize=8)

    path, rp = utils.save_fig(fig, 'accommodation_prism_cac', show=show, params=PARAMS_DEFAULT,
                              conditions='Lit, prism-induced disparity — CA/C drives accommodation (compared with CA_C=0)')
    return utils.fig_meta(path, rp,
        title='Prism-driven CA/C: base-out prism increases accommodation',
        description=f'{PRISM_H}° base-out R prism at t={T_STEP} s; target at {D_MID} m. '
                    'Top: vergence vs demand. Bottom: accommodation (with vs without CA/C).',
        expected='Vergence drives convergence to compensate prism. '
                 'With CA/C: accommodation rises above optical demand. '
                 'Without CA/C: accommodation stays at optical demand.',
        citation='Schor CM & Ciuffreda KJ (1983) Vergence Eye Movements',
    )


# ── Panel 4: Lens plant dynamics — step response of lens to sudden demand ──────

def _lens_step_response(show):
    """Binocular +2 D demand step at t=0.5 s (via plus lenses on both eyes).

    Shows the lens plant first-order dynamics: actual accommodation (x_plant)
    lags behind the neural command (x_fast + x_slow + tonic_acc) with τ_plant ≈ 0.156 s.
    """
    T_STEP = 0.5
    TOTAL  = 5.0
    t      = np.arange(0.0, TOTAL, DT)
    T      = len(t)

    LENS_D = 2.0
    D_FAR  = 3.0   # hold target at 3 m (0.33 D base demand)

    p0 = np.array([0.0, 0.0, D_FAR])
    lens_both = np.where(t >= T_STEP, LENS_D, 0.0).astype(np.float32)

    st = simulate(PARAMS_DEFAULT, t,
                  target=km.build_target(t, lin_pos=np.tile(p0, (T, 1))),
                  scene_present_array=np.ones(T),
                  lens_L_array=lens_both, lens_R_array=lens_both,
                  return_states=True, key=KEY)

    # Neural command: x_fast + x_slow + tonic_acc
    x_fast  = np.array(st.brain[:, _IDX_ACC.start])
    x_slow  = np.array(st.brain[:, _IDX_ACC.start + 1])
    tonic   = float(PARAMS_DEFAULT.brain.tonic_acc)
    u_neural = x_fast + x_slow + tonic

    x_plant = np.array(st.acc_plant[:, 0])
    demand_total = _acc_demand_d(D_FAR) + LENS_D

    # Stimulus trace
    lens_trace = np.where(t >= T_STEP, LENS_D, 0.0)

    fig = plt.figure(figsize=(8, 5.5), constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.6, 3], figure=fig)
    ax_stim = fig.add_subplot(gs[0])
    ax      = fig.add_subplot(gs[1], sharex=ax_stim)

    fig.suptitle(f'Accommodation plant step: bilateral +{LENS_D} D lens at t={T_STEP} s',
                 fontsize=10, fontweight='bold')

    _stim_ax(ax_stim, t, lens_trace, 'Lens ΔD\n(both eyes)', step_t=T_STEP,
             ylim=(-0.3, LENS_D + 0.5))
    ax_stim.set_title(
        f'Stimulus: bilateral +{LENS_D} D plus lens at t = {T_STEP} s  '
        f'(target fixed at {D_FAR} m = {_acc_demand_d(D_FAR):.2f} D)',
        fontsize=8, loc='left', pad=2)
    plt.setp(ax_stim.get_xticklabels(), visible=False)

    ax.plot(t, u_neural, color=utils.C['vs'],  lw=1.3, ls='--', label='Neural command (x_fast+x_slow+tonic)')
    ax.plot(t, x_plant,  color=utils.C['ni'],  lw=1.8,          label='Lens accommodation x_plant (D)')
    ax.axhline(demand_total, color='tomato', lw=0.9, ls=':',
               label=f'Target demand {demand_total:.2f} D')
    ax.axhline(tonic,        color='gray',   lw=0.8, ls=':',
               label=f'tonic_acc {tonic:.1f} D')
    ax.axvline(T_STEP, color='gray', lw=0.8, ls=':')
    ax.set_ylim(-0.5, demand_total + 1.0)
    ax_fmt(ax, ylabel='Accommodation (D)', xlabel='Time (s)')
    ax.legend(fontsize=8)

    path, rp = utils.save_fig(fig, 'accommodation_plant_step', show=show, params=PARAMS_ACC_ONLY,
                              conditions='Lit, defocus step — lens-plant (mechanical) step response')

    # Check: plant lags neural command by ~τ_plant = 0.156 s
    step_start = _acc_demand_d(D_FAR) + tonic   # approx baseline
    step_target = demand_total
    step_h = step_target - step_start
    half  = step_start + 0.5 * step_h
    after = t >= T_STEP
    cross = np.where(after & (x_plant >= half))[0]
    t50   = float(t[cross[0]] - T_STEP) if len(cross) > 0 else float('nan')

    return utils.fig_meta(path, rp,
        title='Accommodation plant step response (τ_plant ≈ 0.156 s)',
        description=f'Bilateral +{LENS_D} D lens step at t={T_STEP} s; target at {D_FAR} m. '
                    'Neural command vs actual lens accommodation x_plant.',
        expected=f'x_plant lags neural command by ~τ_plant. '
                 f'50% rise time ≈ {t50:.2f} s (τ_plant·ln2 ≈ {0.156*np.log(2):.2f} s). '
                 f'Both converge toward {demand_total:.2f} D.',
        citation='Schor & Bharadwaj (2006) J Neurophysiol; Read et al. (2022) J Vision',
    )


# ── Panel 5: Fixation disparity curve — prism and lens sweeps ─────────────────

def _fixation_disparity_curves(show):
    """Fixation disparity as a function of equal bilateral prism or equal lens.

    Classic optometric curve (Sheard / Saladin). For each prism/lens value:
      - Apply the intervention bilaterally (equal left + right)
      - Simulate long enough (~15 s after onset) to reach steady state
      - Measure residual vergence error relative to the geometric target vergence

    Fixation disparity (FD) = geometric vergence − measured vergence (L−R yaw).
    Positive FD = eso-fixation disparity (eyes more converged than ideal).
    Negative FD = exo-fixation disparity (eyes less converged than ideal).

    For base-out prisms: the system needs MORE convergence; FD is exo (negative).
    For plus lenses: the system relaxes accommodation → AC/A slightly diverges
                     → the system compensates fusionally → small FD.
    """
    D_MID   = 0.5    # m (2 D target — moderate depth for AC/A interaction)
    T_SETTLE = 12.0  # seconds to settle after intervention
    T_STEP  = 2.0
    TOTAL   = T_STEP + T_SETTLE
    t       = np.arange(0.0, TOTAL, DT)
    T       = len(t)
    i_settle = int((T_STEP + 8.0) / DT)   # read from last 4 s

    p0 = np.array([0.0, 0.0, D_MID])
    pt = np.tile(p0, (T, 1))
    g0 = _verg_angle_deg(D_MID)

    PRISM_VALS  = np.array([-4, -2, -1, 0, 1, 2, 4, 6, 8], dtype=float)  # neg=BI, pos=BO
    LENS_VALS   = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=float)  # D

    def _steady_verg(prism_h=0.0, lens_d=0.0):
        prism_arr = np.zeros((T, 3), dtype=np.float32)
        prism_arr[t >= T_STEP, 0] = float(prism_h)
        lens_arr  = np.where(t >= T_STEP, float(lens_d), 0.0).astype(np.float32)

        st = simulate(PARAMS_DEFAULT, t,
                      target=km.build_target(t, lin_pos=pt),
                      scene_present_array=np.ones(T),
                      prism_L_array=prism_arr, prism_R_array=prism_arr,
                      lens_L_array=lens_arr,   lens_R_array=lens_arr,
                      return_states=True, key=KEY)
        verg_trace = np.array(st.plant[i_settle:, 0] - st.plant[i_settle:, 3])
        return float(np.mean(verg_trace))

    fd_prism = []
    for pv in PRISM_VALS:
        vss = _steady_verg(prism_h=pv)
        fd_prism.append((g0 + pv - vss) * 60.0)   # convert deg → arcmin

    fd_lens = []
    for lv in LENS_VALS:
        vss = _steady_verg(lens_d=lv)
        fd_lens.append((g0 - vss) * 60.0)          # convert deg → arcmin

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'Fixation disparity curves  (target at {D_MID} m = {_acc_demand_d(D_MID):.1f} D)\n'
        f'Each point = steady-state residual error after ≥{T_SETTLE:.0f} s with the intervention applied',
        fontsize=10, fontweight='bold')

    ax = axes[0]
    ax.plot(PRISM_VALS, fd_prism, 'o-', color=utils.C['vs'], lw=1.8, ms=6)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axvline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('Equal bilateral prism (° — positive = base-out)', fontsize=9)
    ax.set_title('FD vs. Prism  (BO = more convergence demand)', fontsize=9)
    ax_fmt(ax, ylabel="Fixation disparity (arcmin)\n+ = eso (over-converged) / − = exo",
           xlabel='Prism (deg; BO positive)')

    ax = axes[1]
    ax.plot(LENS_VALS, fd_lens, 's-', color=utils.C['ni'], lw=1.8, ms=6)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axvline(0, color='gray', lw=0.8, ls=':')
    ax.set_title('FD vs. Lens  (plus = extra accommodation demand)', fontsize=9)
    ax_fmt(ax, ylabel='Fixation disparity (arcmin)', xlabel='Lens power (D; plus = converging)')

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path, rp = utils.save_fig(fig, 'accommodation_fixation_disparity', show=show, params=PARAMS_DEFAULT,
                              conditions='Lit, midline target at multiple distances — accommodation lag vs depth')
    return utils.fig_meta(path, rp,
        title='Fixation disparity curves: prism sweep and lens sweep',
        description=f'Fixation disparity (demand − achieved vergence, arcmin) vs. bilateral prism (left) '
                    f'and bilateral lens (right) at {D_MID} m target.',
        expected='Prism curve: monotonic, near-linear for small prism, flattening toward Panum area limits. '
                 'Lens curve: U-shaped or monotonic depending on AC/A & CA/C gains. '
                 'Both curves cross zero near zero intervention.',
        citation='Sheard (1930); Saladin (1988) Optom Vis Sci; Schor (1979) Vision Res',
    )


# ── Panel 6: Refractive error — myopia / hyperopia accommodation response ───────

def _refractive_error(show):
    """Near-step for emmetrope, +2 D hyperope (uncorrected), −2 D myope (uncorrected).

    Refractive error shifts the defocus zero point:
      - Hyperope (RE > 0): needs MORE accommodation at every distance → controller
        must work harder; increased risk of over-accommodation / esotropia.
      - Myope   (RE < 0): needs LESS accommodation; natural far point at 1/|RE| m;
        reduced stimulus for near-driven esophoria.

    Without spectacle correction, the accommodation response should differ across
    the three groups for the same target depth, as the residual defocus (acc_demand
    + RE − x_plant) differs.
    """
    T_STEP  = 1.0
    TOTAL   = 8.0
    t       = np.arange(0.0, TOTAL, DT)
    T       = len(t)

    D_FAR  = 3.0
    D_NEAR = 0.40

    p0 = np.array([0.0, 0.0, D_FAR])
    p1 = np.array([0.0, 0.0, D_NEAR])
    pt = np.where((t >= T_STEP)[:, None], p1, p0)
    tgt = km.build_target(t, lin_pos=pt)

    groups = {
        'Emmetrope (RE=0)':         with_brain(PARAMS_DEFAULT, refractive_error=0.0),
        'Hyperope +2D (uncorrected)': with_brain(PARAMS_DEFAULT, refractive_error=2.0),
        'Myope −2D (uncorrected)':    with_brain(PARAMS_DEFAULT, refractive_error=-2.0),
    }
    colors  = [utils.C['vs'], utils.C['ni'], utils.C['eye']]
    styles  = ['-', '--', '-.']

    # Stimulus trace
    depth_trace = np.where(t >= T_STEP, D_NEAR, D_FAR)

    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.6, 2, 2], figure=fig)
    ax_stim  = fig.add_subplot(gs[0])
    ax_verg  = fig.add_subplot(gs[1], sharex=ax_stim)
    ax_acc   = fig.add_subplot(gs[2], sharex=ax_stim)

    fig.suptitle(f'Refractive error: near step {D_FAR} m → {D_NEAR} m at t={T_STEP} s  (uncorrected)',
                 fontsize=10, fontweight='bold')

    _stim_ax(ax_stim, t, depth_trace, 'Depth (m)', step_t=T_STEP,
             ylim=(0.0, D_FAR + 0.5))
    ax_stim.set_title(
        f'Stimulus: target steps {D_FAR} m → {D_NEAR} m at t = {T_STEP} s  '
        f'(no spectacle correction)',
        fontsize=8, loc='left', pad=2)
    plt.setp(ax_stim.get_xticklabels(), visible=False)

    for (label, p), col, ls in zip(groups.items(), colors, styles):
        st = simulate(p, t, target=tgt, scene_present_array=np.ones(T),
                      return_states=True, key=KEY)
        verg = np.array(st.plant[:, 0] - st.plant[:, 3])
        acc  = np.array(st.acc_plant[:, 0])
        ax_verg.plot(t, verg, color=col, ls=ls, lw=1.4, label=label)
        ax_acc.plot(t, acc,  color=col, ls=ls, lw=1.4, label=label)

    g_near = _verg_angle_deg(D_NEAR)
    g_far  = _verg_angle_deg(D_FAR)
    ax_verg.axhline(g_near, color='tomato', lw=0.9, ls=':', label=f'Geometric {g_near:.1f}°@{D_NEAR}m')
    ax_verg.axhline(g_far,  color='gray',   lw=0.8, ls=':', label=f'Geometric {g_far:.1f}°@{D_FAR}m')
    ax_verg.axvline(T_STEP, color='gray', lw=0.8, ls=':')
    ax_verg.set_ylim(0, g_near + 3)
    ax_fmt(ax_verg, ylabel='Vergence (deg)')
    ax_verg.legend(fontsize=8)
    plt.setp(ax_verg.get_xticklabels(), visible=False)

    ax_acc.axhline(_acc_demand_d(D_NEAR), color='tomato', lw=0.9, ls=':',
                    label=f'Optical {_acc_demand_d(D_NEAR):.2f} D@{D_NEAR}m')
    ax_acc.axhline(_acc_demand_d(D_FAR),  color='gray',   lw=0.8, ls=':',
                    label=f'Optical {_acc_demand_d(D_FAR):.2f} D@{D_FAR}m')
    ax_acc.axvline(T_STEP, color='gray', lw=0.8, ls=':')
    ax_acc.set_ylim(-0.5, _acc_demand_d(D_NEAR) + 1.0)
    ax_fmt(ax_acc, ylabel='Accommodation (D)', xlabel='Time (s)')
    ax_acc.legend(fontsize=8)

    path, rp = utils.save_fig(fig, 'accommodation_refractive_error', show=show, params=PARAMS_DEFAULT,
                              conditions='Lit — emmetrope vs ±2D uncorrected hyperope/myope (refractive_error swept)')
    return utils.fig_meta(path, rp,
        title='Refractive error: accommodation and vergence for myope/hyperope',
        description=f'Near step {D_FAR}→{D_NEAR} m. Three groups: emmetrope, +2D hyperope, −2D myope. '
                    'Top: vergence. Bottom: lens accommodation. No spectacle correction.',
        expected='Hyperope accommodates more (must compensate RE); vergence slightly higher via AC/A. '
                 'Myope accommodates less (RE reduces effective demand); vergence slightly lower. '
                 'All groups converge vergence toward geometric value; myope may show under-accommodation.',
        citation='Duke-Elder & Abrams (1970) System of Ophthalmology; '
                 'Saw et al. (2005) Invest Ophthalmol Vis Sci',
    )


# ── Panel: Gradient AC/A and CA/C — cross-link regression panels ───────────────

def _gradient_aca_cac(show):
    """Gradient AC/A and CA/C protocols — behavioral measurements of both
    cross-links from stimulus sweeps at a fixed fixation distance.

    Both cross-links (AC/A and CA/C) are LEFT ON at defaults in both sweeps —
    the goal is to measure the closed-loop, end-to-end behavioral cross-link
    gain that a clinician would observe, not an idealized open-loop value.

    Row 1 — Gradient AC/A (lens sweep, Maddox-style):
      - Target visible only to the LEFT eye → disparity vergence open-loop.
      - Equal lenses on both eyes step the accommodation demand → AC/A
        cross-link drives vergence, measured at SS.
      - Slope of SS vergence (pd) vs lens (D) = behavioral AC/A.

    Row 2 — Gradient CA/C (prism sweep, fully binocular):
      - Both eyes see the target → disparity vergence is closed-loop, but the
        defocus loop is also closed (we lack a true pinhole/DoG mechanism).
      - Bilateral mirrored prisms (BO/BI) shift the apparent target nasally
        or temporally → fusional vergence response → CA/C cross-link drives
        accommodation, measured at SS.
      - Slope of SS accommodation (D) vs achieved SS vergence (pd) =
        behavioral CA/C. Closed defocus loop suppresses this slope below
        the parameter; the regression should still be monotonic and positive.
    """
    DEPTH    = 0.4          # m — clinical gradient AC/A typically at 40 cm
    TEND     = 8.0          # s, settle time
    T_STEP   = 1.0          # stimulus applied at t = T_STEP
    LENSES_D = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    PRISMS_D = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])   # deg, mirrored bilaterally; +ve = BO (convergence demand)
    SS_WIN   = 0.5          # s — average over the last SS_WIN seconds for SS

    t   = np.arange(0.0, TEND, DT)
    T   = len(t)
    p0  = np.array([0.0, 0.0, DEPTH])
    target = km.build_target(t, lin_pos=np.tile(p0, (T, 1)))
    scene  = np.ones(T)

    # Both cross-links left ON at defaults — measure end-to-end behavioral gain.
    params = PARAMS_DEFAULT
    deg_per_pd = 0.5729
    aca_param = float(params.brain.AC_A)   # pd/D
    cac_param = float(params.brain.CA_C)   # D/pd

    # ── ACA sweep (lens sweep, Maddox-style monocular target) ───────────────
    tgt_L = np.ones(T, dtype=np.float32)
    tgt_R = np.zeros(T, dtype=np.float32)

    aca_verg_traces, aca_acc_traces = [], []
    aca_ss_verg, aca_ss_acc = [], []
    for L in LENSES_D:
        lens_arr = np.where(t >= T_STEP, float(L), 0.0).astype(np.float32)
        st = simulate(params, t,
                      target=target,
                      scene_present_array=scene,
                      target_present_L_array=tgt_L,
                      target_present_R_array=tgt_R,
                      lens_L_array=lens_arr, lens_R_array=lens_arr,
                      return_states=True, key=KEY)
        verg = np.array(st.plant[:, 0] - st.plant[:, 3])
        acc  = np.array(st.acc_plant[:, 0])
        aca_verg_traces.append(verg)
        aca_acc_traces.append(acc)
        n_ss = int(SS_WIN / DT)
        aca_ss_verg.append(float(np.mean(verg[-n_ss:])))
        aca_ss_acc.append(float(np.mean(acc[-n_ss:])))

    aca_ss_verg = np.array(aca_ss_verg)
    aca_ss_verg_pd = aca_ss_verg / deg_per_pd
    aca_slope_pd_per_D, aca_intercept_pd = np.polyfit(LENSES_D, aca_ss_verg_pd, 1)

    # ── CAC sweep (prism sweep, full binocular) ─────────────────────────────
    # Bilateral mirrored prisms on yaw axis: prism_L = +Δ (rightward shift = nasal
    # for L eye); prism_R = −Δ (leftward shift = nasal for R eye). Δ>0 = BO bilaterally.
    cac_verg_traces, cac_acc_traces = [], []
    cac_ss_verg, cac_ss_acc = [], []
    for P in PRISMS_D:
        prism_L_arr = np.zeros((T, 3), dtype=np.float32)
        prism_R_arr = np.zeros((T, 3), dtype=np.float32)
        prism_L_arr[t >= T_STEP, 0] = +float(P)
        prism_R_arr[t >= T_STEP, 0] = -float(P)
        st = simulate(params, t,
                      target=target,
                      scene_present_array=scene,
                      prism_L_array=prism_L_arr, prism_R_array=prism_R_arr,
                      return_states=True, key=KEY)
        verg = np.array(st.plant[:, 0] - st.plant[:, 3])
        acc  = np.array(st.acc_plant[:, 0])
        cac_verg_traces.append(verg)
        cac_acc_traces.append(acc)
        n_ss = int(SS_WIN / DT)
        cac_ss_verg.append(float(np.mean(verg[-n_ss:])))
        cac_ss_acc.append(float(np.mean(acc[-n_ss:])))

    cac_ss_verg = np.array(cac_ss_verg)
    cac_ss_acc  = np.array(cac_ss_acc)
    # Behavioral CA/C: ΔAcc (D) per ΔVerg (pd) — closed-loop slope.
    cac_ss_verg_pd = cac_ss_verg / deg_per_pd
    cac_slope_D_per_pd, cac_intercept_D = np.polyfit(cac_ss_verg_pd, cac_ss_acc, 1)

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Cross-link gradient regressions — both AC/A and CA/C cross-links ON',
                 fontsize=11, fontweight='bold')

    cmap = plt.get_cmap('coolwarm')

    # ── Row 1: ACA ──────────────────────────────────────────────────────────
    ax = axes[0, 0]
    for i, (L, v) in enumerate(zip(LENSES_D, aca_verg_traces)):
        c = cmap(i / max(len(LENSES_D) - 1, 1))
        ax.plot(t, v, color=c, lw=1.2, label=f'{L:+.0f} D')
    ax.axvline(T_STEP, color='gray', lw=0.8, ls=':')
    ax.axhline(_verg_angle_deg(DEPTH), color='black', lw=0.6, ls='--',
               label=f'geometric {_verg_angle_deg(DEPTH):.1f}°')
    ax_fmt(ax, ylabel='Vergence L−R (deg)', xlabel='Time (s)')
    ax.set_title(f'AC/A — Vergence vs lens (target at {DEPTH:.2f} m, Maddox L-only)', fontsize=9)
    ax.legend(fontsize=7, loc='best', title='Lens')

    ax = axes[0, 1]
    for i, (L, a) in enumerate(zip(LENSES_D, aca_acc_traces)):
        c = cmap(i / max(len(LENSES_D) - 1, 1))
        ax.plot(t, a, color=c, lw=1.2, label=f'{L:+.0f} D')
    ax.axvline(T_STEP, color='gray', lw=0.8, ls=':')
    ax.axhline(_acc_demand_d(DEPTH), color='black', lw=0.6, ls='--',
               label=f'demand at {DEPTH:.2f} m  ({_acc_demand_d(DEPTH):.2f} D)')
    ax_fmt(ax, ylabel='Accommodation (D)', xlabel='Time (s)')
    ax.set_title('AC/A — Accommodation response', fontsize=9)
    ax.legend(fontsize=7, loc='best', title='Lens')

    ax = axes[0, 2]
    ax.scatter(LENSES_D, aca_ss_verg_pd, s=60, color='#cc3333', zorder=5)
    Lfit = np.linspace(LENSES_D.min() - 0.3, LENSES_D.max() + 0.3, 50)
    Vfit = aca_slope_pd_per_D * Lfit + aca_intercept_pd
    ax.plot(Lfit, Vfit, '-', color='#cc3333', lw=1.2,
            label=f'Behavioral AC/A = {aca_slope_pd_per_D:.2f} pd/D\n(parameter AC_A = {aca_param:.1f} pd/D)')
    ax.axhline(_verg_angle_deg(DEPTH) / deg_per_pd, color='gray', lw=0.6, ls='--',
               label=f'geometric {_verg_angle_deg(DEPTH)/deg_per_pd:.1f} pd')
    ax.set_xlabel('Lens power (D)')
    ax.set_ylabel('Steady-state vergence (pd)')
    ax.set_title(f'Gradient AC/A regression (SS over last {SS_WIN:g} s)', fontsize=9)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.2)

    # ── Row 2: CAC ──────────────────────────────────────────────────────────
    ax = axes[1, 0]
    for i, (P, v) in enumerate(zip(PRISMS_D, cac_verg_traces)):
        c = cmap(i / max(len(PRISMS_D) - 1, 1))
        ax.plot(t, v, color=c, lw=1.2, label=f'{P:+.0f}°')
    ax.axvline(T_STEP, color='gray', lw=0.8, ls=':')
    ax.axhline(_verg_angle_deg(DEPTH), color='black', lw=0.6, ls='--',
               label=f'geometric {_verg_angle_deg(DEPTH):.1f}°')
    ax_fmt(ax, ylabel='Vergence L−R (deg)', xlabel='Time (s)')
    ax.set_title(f'CA/C — Vergence vs prism (binocular, target at {DEPTH:.2f} m)', fontsize=9)
    ax.legend(fontsize=7, loc='best', title='Prism (BO+)')

    ax = axes[1, 1]
    for i, (P, a) in enumerate(zip(PRISMS_D, cac_acc_traces)):
        c = cmap(i / max(len(PRISMS_D) - 1, 1))
        ax.plot(t, a, color=c, lw=1.2, label=f'{P:+.0f}°')
    ax.axvline(T_STEP, color='gray', lw=0.8, ls=':')
    ax.axhline(_acc_demand_d(DEPTH), color='black', lw=0.6, ls='--',
               label=f'demand at {DEPTH:.2f} m  ({_acc_demand_d(DEPTH):.2f} D)')
    ax_fmt(ax, ylabel='Accommodation (D)', xlabel='Time (s)')
    ax.set_title('CA/C — Accommodation response', fontsize=9)
    ax.legend(fontsize=7, loc='best', title='Prism (BO+)')

    ax = axes[1, 2]
    ax.scatter(cac_ss_verg_pd, cac_ss_acc, s=60, color='#3366cc', zorder=5)
    Vfit = np.linspace(cac_ss_verg_pd.min() - 0.5, cac_ss_verg_pd.max() + 0.5, 50)
    Afit = cac_slope_D_per_pd * Vfit + cac_intercept_D
    ax.plot(Vfit, Afit, '-', color='#3366cc', lw=1.2,
            label=f'Behavioral CA/C = {cac_slope_D_per_pd:.3f} D/pd\n(parameter CA_C = {cac_param:.2f} D/pd)')
    ax.axhline(_acc_demand_d(DEPTH), color='gray', lw=0.6, ls='--',
               label=f'optical demand {_acc_demand_d(DEPTH):.2f} D')
    ax.axvline(_verg_angle_deg(DEPTH) / deg_per_pd, color='gray', lw=0.6, ls=':',
               label=f'geometric {_verg_angle_deg(DEPTH)/deg_per_pd:.1f} pd')
    ax.set_xlabel('Steady-state vergence (pd)')
    ax.set_ylabel('Steady-state accommodation (D)')
    ax.set_title(f'Gradient CA/C regression (SS over last {SS_WIN:g} s)', fontsize=9)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path, rp = utils.save_fig(fig, 'accommodation_gradient_aca_cac', show=show, params=params,
                              conditions=f'Lit, midline target at {DEPTH:.2f} m. '
                                         f'AC/A row: lens sweep {LENSES_D[0]:+.0f}…{LENSES_D[-1]:+.0f} D, '
                                         f'L-eye-only (Maddox). '
                                         f'CA/C row: bilateral mirrored prism sweep '
                                         f'{PRISMS_D[0]:+.0f}…{PRISMS_D[-1]:+.0f}°, full binocular. '
                                         f'Both cross-links ON.')
    return utils.fig_meta(path, rp,
        title='Gradient AC/A and CA/C — behavioral measurements vs parameters',
        description=f'Top row (AC/A): binocular lens sweep at {DEPTH:.2f} m, L-eye-only target. '
                    f'Bottom row (CA/C): bilateral mirrored prism sweep at {DEPTH:.2f} m, full '
                    f'binocular fusion. Both cross-links left at defaults — measures realistic '
                    f'closed-loop behavioral gains. Behavioral CA/C is reduced by closed defocus '
                    f'loop (model lacks true pinhole/DoG mechanism for open-loop accommodation).',
        expected=f'Behavioral AC/A < parameter AC_A ({aca_param:.1f} pd/D): phasic decays into '
                 f'tonic in closed loop, expect ~{0.5*aca_param:.1f}–{0.8*aca_param:.1f} pd/D. '
                 f'Behavioral CA/C < parameter CA_C ({cac_param:.2f} D/pd): closed defocus loop '
                 f'suppresses additional, expect monotone positive but small slope.',
        citation='Hofstetter (1948); Schor & Kotulak (1986); Sheedy & Saladin (1983); Schor (1992, 1999); Daum (1983).',
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def run(show=SHOW):
    print('\n=== Accommodation ===')
    print('  1/10 isolated step (AC/A=CA/C=0) …')
    f1  = _isolated_step(show)
    print('  2/10 isolated amplitudes …')
    f2  = _isolated_amplitudes(show)
    print('  3/10 isolated dark-focus drift …')
    f3  = _isolated_dark_focus(show)
    print('  4/10 near step (cross-coupling on) …')
    f4  = _near_step(show)
    print('  5/10 lens-driven AC/A (single +2D step) …')
    f5  = _lens_aca(show)
    print('  6/10 prism-driven CA/C …')
    f6  = _prism_cac(show)
    print('  7/10 plant step response …')
    f7  = _lens_step_response(show)
    print('  8/10 fixation disparity curves …')
    f8  = _fixation_disparity_curves(show)
    print('  9/10 refractive error …')
    f9  = _refractive_error(show)
    print(' 10/10 gradient AC/A and CA/C regression …')
    f10 = _gradient_aca_cac(show)
    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]


if __name__ == '__main__':
    figs = run(show=SHOW)
    for f in figs:
        print(f['title'])
        for c in f.get('checks', []):
            print(f'  • {c}')
