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

    path, rp = utils.save_fig(fig, 'accommodation_near_step', show=show)
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

    path, rp = utils.save_fig(fig, 'accommodation_lens_aca', show=show)
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

    path, rp = utils.save_fig(fig, 'accommodation_prism_cac', show=show)
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

    path, rp = utils.save_fig(fig, 'accommodation_plant_step', show=show)

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
    path, rp = utils.save_fig(fig, 'accommodation_fixation_disparity', show=show)
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

    path, rp = utils.save_fig(fig, 'accommodation_refractive_error', show=show)
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


# ── Main ───────────────────────────────────────────────────────────────────────

def run(show=SHOW):
    results = [
        _near_step(show),
        _lens_aca(show),
        _prism_cac(show),
        _lens_step_response(show),
        _fixation_disparity_curves(show),
        _refractive_error(show),
    ]
    return results


if __name__ == '__main__':
    figs = run(show=SHOW)
    for f in figs:
        print(f['title'])
        for c in f.get('checks', []):
            print(f'  • {c}')
