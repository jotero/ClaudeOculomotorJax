"""Clinical integrator and velocity-storage disorder benchmarks.

Gaze-evoked nystagmus, rebound nystagmus, Bruns nystagmus, periodic
alternating nystagmus, and extended OKAN from null-point adaptation.

Usage:
    python -X utf8 scripts/bench_clinical_ni_vs.py
    python -X utf8 scripts/bench_clinical_ni_vs.py --show
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_clinical_utils as utils

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, with_sensory, simulate, SimConfig,
    with_uvh,
)
from oculomotor.sim import kinematics as km
from oculomotor.analysis import (
    ax_fmt, vs_net, vs_null, ni_net, ni_null, extract_spv_states,
)

SHOW = '--show' in sys.argv
DT   = 0.001

THETA = with_sensory(PARAMS_DEFAULT,
                     sigma_canal=0.0, sigma_pos=0.0,
                     sigma_vel=0.0, sigma_slip=0.0)

SECTION = dict(
    id='clin_ni_vs', title='B. Gaze-Holding & Integrator Disorders',
    description='Neural integrator and velocity-storage pathology: '
                'gaze-evoked nystagmus (leaky NI), rebound nystagmus '
                '(NI null adaptation), Bruns nystagmus, and extended OKAN '
                '(VS null adaptation / PAN precursor).',
)

C_HEALTHY = '#2166ac'
C_GEN     = '#d6604d'
C_BRUNS   = '#762a83'
C_ADAPT   = '#1a9641'


def _sim_lit(params, t_arr, pt_3d, scene_on=True, key=0):
    """Simulate with a stationary fixation target in a lit environment.

    The target is treated as a strobed dot (no motion feedback): lin_vel=0
    suppresses the velocity spike from step changes in lin_pos, and
    target_strobed_array=1 gates the pursuit motion channel so the pursuit
    integrator is never driven by target velocity.  Position feedback
    (pos_delayed → saccade generator) is unaffected.
    """
    T   = len(t_arr)
    t   = np.asarray(t_arr)
    sp  = np.ones(T, np.float32) if scene_on else np.zeros(T, np.float32)
    lv  = np.zeros((T, 3), np.float32)   # stationary target — zero velocity
    return simulate(params, t,
                    target=km.build_target(t, lin_pos=pt_3d, lin_vel=lv),
                    target_strobed_array=np.ones(T, np.float32),
                    scene_present_array=sp,
                    target_present_array=sp,
                    max_steps=int(T * 1.1) + 1000,
                    sim_config=SimConfig(warmup_s=0.0),
                    return_states=True,
                    key=jax.random.PRNGKey(key))


def _target_step(t_arr, deg_on, deg_off, t_switch):
    """Target position array: holds deg_on before t_switch, then deg_off."""
    T = len(t_arr)
    t = np.asarray(t_arr)
    pt = np.zeros((T, 3), np.float32)
    pt[:, 0] = np.where(t < t_switch,
                        np.tan(np.radians(deg_on)),
                        np.tan(np.radians(deg_off))).astype(np.float32)
    pt[:, 2] = 1.0
    return pt


# ─────────────────────────────────────────────────────────────────────────────
# 1. Gaze-evoked nystagmus — leaky NI (flocculus / NPH lesion)
# ─────────────────────────────────────────────────────────────────────────────

def _gen(show):
    # Cerebellar/floccular lesion impairs both NI (tau_i ↓) and pursuit (K_pursuit ↓).
    # Gain-of-function: flocculus tonically inhibits NI; loss → leak.
    # Smooth pursuit also floccular → reduced K_pursuit + K_phasic_pursuit.
    THETA_GEN  = with_brain(THETA, tau_i=2.0, K_pursuit=0.5,  K_phasic_pursuit=1.0)
    THETA_MOD  = with_brain(THETA, tau_i=6.0, K_pursuit=1.5,  K_phasic_pursuit=2.5)

    DUR_ECC  = 15.0
    DUR_POST = 8.0
    t_arr    = np.arange(0.0, DUR_ECC + DUR_POST, DT)
    pt       = _target_step(t_arr, 0.0, 20.0, 0.1)
    pt_back  = _target_step(t_arr, 20.0, 0.0, DUR_ECC)

    conds = [
        ('Healthy',       THETA,     C_HEALTHY, '-'),
        ('Moderate GEN',  THETA_MOD, C_ADAPT,   '--'),
        ('Severe GEN',    THETA_GEN, C_GEN,     '-'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(
        'Gaze-Evoked Nystagmus — Leaky Neural Integrator\n'
        '(Flocculus / Paraflocculus / NPH lesion)',
        fontsize=12, fontweight='bold')

    for label, theta, col, ls in conds:
        st = _sim_lit(theta, t_arr, pt)
        ep  = np.array(st.plant[:, 0])
        ev  = np.gradient(ep, DT)
        spv = extract_spv_states(st, t_arr)[:, 0]
        ni    = ni_net(st)[:, 0]
        ni_n  = ni_null(st)[:, 0]
        axes[0].plot(t_arr, ep,  color=col, lw=1.5, ls=ls, label=f'{label} τ_i={theta.brain.tau_i:.0f}s')
        axes[1].plot(t_arr, spv, color=col, lw=1.5, ls=ls)
        axes[2].plot(t_arr, ni,  color=col, lw=1.5, ls=ls, label=label)

    axes[2].plot(t_arr, ni_null(_sim_lit(THETA_GEN, t_arr, pt))[:, 0],
                 color=C_GEN, lw=1.0, ls=':', alpha=0.7, label='GEN NI null')

    for ax in axes:
        ax.axvline(0.1,    color='gray', lw=0.8, ls='--', alpha=0.5, label='Gaze shift')
        ax.axvline(DUR_ECC, color='k',  lw=0.8, ls='--', alpha=0.5, label='Return to center')
        ax.axhline(0, color='k', lw=0.4); ax.grid(True, alpha=0.15)

    ax_fmt(axes[0], ylabel='Eye yaw (deg)')
    axes[0].set_title('Eye position — GEN: centripetal drift during eccentric hold')
    axes[0].legend(fontsize=8)
    ax_fmt(axes[1], ylabel='SPV (deg/s)', ylim=(-200, 200))
    axes[1].set_title('Slow-phase velocity — corrective fast phases maintain gaze')
    ax_fmt(axes[2], ylabel='NI state (deg)', xlabel='Time (s)')
    axes[2].set_title('NI state — leaky NI cannot hold eccentric position')
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'clin_ni_gen', show=show)
    return utils.fig_meta(path, rp,
        title='Gaze-Evoked Nystagmus (GEN)',
        description='NI leak TC reduced: healthy 25 s → moderate 6 s → severe 2 s. '
                    'Centripetal drift during eccentric gaze; corrective fast phases.',
        expected='GEN: nystagmus beats in direction of gaze (centripetal drift). '
                 'Severity scales with τ_i shortening. '
                 'No nystagmus at primary position.',
        citation='Cannon & Robinson (1985) Biol Cybern; Zee et al. (1976) J Neurophysiol',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rebound nystagmus — NI null adaptation
# ─────────────────────────────────────────────────────────────────────────────

def _rebound(show):
    # Rebound is a cerebellar sign (floccular NI null adaptation).
    # Both conditions represent cerebellar patients → reduced pursuit gain.
    # THETA_NO_NULL disables null adaptation to show what rebound requires.
    THETA_CEREB    = with_brain(THETA,          K_pursuit=0.8, K_phasic_pursuit=2.0)
    THETA_NO_NULL  = with_brain(THETA_CEREB,    tau_ni_adapt=1e6)

    HOLD_DUR = 20.0
    POST_DUR = 15.0
    t_arr    = np.arange(0.0, HOLD_DUR + POST_DUR, DT)
    pt_hold  = _target_step(t_arr, 0.0, 20.0, 0.1)
    pt_back  = np.zeros((len(t_arr), 3), np.float32)
    pt_back[:, 2] = 1.0
    pt_full  = np.where((np.asarray(t_arr) < HOLD_DUR)[:, None], pt_hold, pt_back)

    st_null   = _sim_lit(THETA_CEREB,    t_arr, pt_full)
    st_nonull = _sim_lit(THETA_NO_NULL,  t_arr, pt_full)

    spv_null   = extract_spv_states(st_null,   t_arr)[:, 0]
    spv_nonull = extract_spv_states(st_nonull, t_arr)[:, 0]

    null_at_ret = float(ni_null(st_null)[int(HOLD_DUR / DT), 0])

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(
        f'Rebound Nystagmus — NI Null Adaptation (τ_ni_adapt = {THETA_CEREB.brain.tau_ni_adapt:.0f} s)\n'
        f'(Cerebellar flocculus/paraflocculus sign)',
        fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(t_arr, np.array(st_null.plant[:,   0]), color=C_GEN,     lw=1.5, label='With null adaptation (rebound)')
    ax.plot(t_arr, np.array(st_nonull.plant[:, 0]), color='#888888', lw=1.5, ls='--', label='No null adaptation')
    ax.axhline(20, color='gray', lw=0.7, ls=':', alpha=0.5, label='Target 20°')
    ax.axvline(HOLD_DUR, color='k', lw=0.8, ls='--', alpha=0.5)
    ax_fmt(ax, ylabel='Eye yaw (deg)')
    ax.set_title('Eye position — drift toward old target after returning to center')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t_arr, spv_null,   color=C_GEN,     lw=2.0, label='SPV (with null)')
    ax.plot(t_arr, spv_nonull, color='#888888', lw=1.5, ls='--', label='SPV (no null)')
    ax.axvline(HOLD_DUR, color='k', lw=0.8, ls='--', alpha=0.5, label='Return to center')
    ax_fmt(ax, ylabel='SPV (deg/s)', ylim=(-50, 50))
    ax.set_title('SPV — positive after return = rightward drift → nystagmus beats LEFT (rebound ✓)')
    ax.legend(fontsize=8)

    ax = axes[2]
    ni    = ni_net(st_null)[:, 0]
    ni_n  = ni_null(st_null)[:, 0]
    ax.plot(t_arr, ni,  color=C_GEN,     lw=1.5, label='NI net')
    ax.plot(t_arr, ni_n, color='#e08214', lw=1.5, ls='--', label=f'NI null (at return: {null_at_ret:.1f}°)')
    ax.axvline(HOLD_DUR, color='k', lw=0.8, ls='--', alpha=0.5)
    ax_fmt(ax, ylabel='NI state (deg)', xlabel='Time (s)')
    ax.set_title('NI null drifts toward eccentric position; persists on return → rebound drive')
    ax.legend(fontsize=8)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'clin_ni_rebound', show=show)
    return utils.fig_meta(path, rp,
        title='Rebound Nystagmus',
        description=f'NI null adaptation TC = {THETA.brain.tau_ni_adapt:.0f} s. '
                    'Hold 20° for 20 s, return to center. '
                    'Null persists → rebound beats toward center.',
        expected='Brief nystagmus beating toward former eccentric position after '
                 'returning to center. Hallmark of cerebellar floccular lesion. '
                 'No rebound without null adaptation.',
        citation='Zee et al. (1980) Brain; Hood & Leech (1974) J Neurol',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bruns nystagmus — GEN + UVH + null adaptation
# ─────────────────────────────────────────────────────────────────────────────

def _bruns(show):
    # Cerebellopontine angle tumor: left peripheral UVH + cerebellar compression.
    # Cerebellar component → leaky NI + NI null adaptation + reduced pursuit.
    THETA_BRUNS = with_uvh(
        with_brain(THETA, tau_i=3.0, tau_ni_adapt=15.0,
                   K_pursuit=0.5, K_phasic_pursuit=1.0),
        side='left', canal_gain_frac=0.1)

    SEG_DUR = 12.0
    segs    = [(0.0, 0.0), (SEG_DUR, 20.0), (2 * SEG_DUR, 0.0), (3 * SEG_DUR, -20.0)]
    t_arr   = np.arange(0.0, 4 * SEG_DUR, DT)
    T       = len(t_arr)

    # Build stepped target
    tx = np.zeros(T, np.float32)
    for t_start, deg in segs:
        tx = np.where(np.asarray(t_arr) >= t_start, np.tan(np.radians(deg)), tx).astype(np.float32)
    pt = np.stack([tx, np.zeros(T, np.float32), np.ones(T, np.float32)], axis=1)

    st = _sim_lit(THETA_BRUNS, t_arr, pt)
    spv = extract_spv_states(st, t_arr)[:, 0]
    ni    = ni_net(st)[:, 0]
    ni_n  = ni_null(st)[:, 0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        'Bruns Nystagmus — CPA Tumor Model\n'
        '(Left UVH + Leaky NI τ_i=3 s + Null Adaptation τ_ni_adapt=15 s)',
        fontsize=12, fontweight='bold')

    seg_labels = ['Center\n(spontaneous)', '+20° contra\n(right)', 'Center', '−20° ipsi\n(left)']

    ax = axes[0]
    ax.plot(t_arr, np.array(st.plant[:, 0]), color=C_BRUNS, lw=1.2, label='Eye pos')
    ax.plot(t_arr, np.degrees(np.arctan(tx)), color='gray', lw=0.8, ls='--', alpha=0.7, label='Target')
    ax_fmt(ax, ylabel='Eye yaw (deg)')
    ax.set_title('Eye position — large drift during ipsilesional gaze')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t_arr, spv, color=C_BRUNS, lw=2.0, label='SPV')
    ax.fill_between(t_arr, spv, 0,
                    where=(np.asarray(t_arr) >= 3*SEG_DUR),
                    color=C_BRUNS, alpha=0.2, label='Ipsilesional segment')
    ax_fmt(ax, ylabel='SPV (deg/s)', ylim=(-80, 60))
    ax.set_title('SPV — spontaneous at rest; large ipsilesional; '
                 'small fast rebound on return')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(t_arr, ni,  color=C_BRUNS,   lw=1.5, label='NI net')
    ax.plot(t_arr, ni_n, color='#e08214', lw=1.5, ls='--', label='NI null')
    ax_fmt(ax, ylabel='NI state (deg)', xlabel='Time (s)')
    ax.set_title('NI null tracks gaze position; rebound after each shift')
    ax.legend(fontsize=8)

    for ax in axes:
        for i in range(1, 4):
            ax.axvline(i * SEG_DUR, color='k', lw=0.5, ls='--', alpha=0.3)

    for i, lbl in enumerate(seg_labels):
        axes[0].text(i * SEG_DUR + SEG_DUR * 0.45, -25, lbl,
                     ha='center', fontsize=7.5, color='gray')

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'clin_ni_bruns', show=show)
    return utils.fig_meta(path, rp,
        title='Bruns Nystagmus (CPA Tumor)',
        description='Left UVH + leaky NI (τ_i=3 s) + null adaptation (τ=15 s). '
                    'Large nystagmus ipsilesionally; small fast rebound contraversively.',
        expected='Ipsilesional gaze: large-amplitude slow nystagmus. '
                 'Contraversive gaze: small fast nystagmus. '
                 'Spontaneous nystagmus beating toward intact side at rest.',
        citation='Bruns (1902); Leigh & Zee (2015) Neurology of Eye Movements',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────
# 4. VS null adaptation — extended OKAN (PAN precursor)
# ─────────────────────────────────────────────────────────────────────────────

def _okan_extension(show):
    THETA_ADAPT = with_brain(THETA, tau_vs_adapt=60.0)

    ON_DUR  = 30.0
    OFF_DUR = 60.0
    t_arr   = np.arange(0.0, ON_DUR + OFF_DUR, DT)
    T       = len(t_arr)
    t_rel   = t_arr - ON_DUR

    v_scene       = np.zeros((T, 3), np.float32)
    v_scene[:, 0] = np.where(t_arr < ON_DUR, 30.0, 0.0).astype(np.float32)
    sp            = np.where(t_arr < ON_DUR, 1.0, 0.0).astype(np.float32)

    def _run(params):
        return simulate(params, t_arr,
                        scene=km.build_kinematics(t_arr, rot_vel=v_scene),
                        scene_present_array=sp,
                        target_present_array=np.zeros(T),
                        max_steps=int(T * 1.1) + 500,
                        sim_config=SimConfig(warmup_s=0.0),
                        return_states=True)

    st_h = _run(THETA)
    st_a = _run(THETA_ADAPT)

    spv_h = extract_spv_states(st_h, t_arr)[:, 0]
    spv_a = extract_spv_states(st_a, t_arr)[:, 0]

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(
        'VS Null Adaptation — Extended OKAN\n'
        f'(τ_vs_adapt: {THETA.brain.tau_vs_adapt:.0f} s → {THETA_ADAPT.brain.tau_vs_adapt:.0f} s)',
        fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(t_arr, spv_h, color=C_HEALTHY, lw=2.0,
            label=f'Default  τ_vs_adapt={THETA.brain.tau_vs_adapt:.0f} s')
    ax.plot(t_arr, spv_a, color=C_ADAPT,   lw=2.0,
            label=f'Adapted  τ_vs_adapt={THETA_ADAPT.brain.tau_vs_adapt:.0f} s')
    ax.axvline(ON_DUR, color='k', lw=0.8, ls='--', alpha=0.5, label='Lights off')
    ax.axhline(30, color='gray', lw=0.7, ls=':', alpha=0.4)
    ax_fmt(ax, ylabel='SPV (deg/s)')
    ax.set_title('OKN + OKAN — faster VS null extends OKAN duration')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t_arr, -vs_net(st_h)[:, 0], color=C_HEALTHY, lw=1.5, label='VS net (default)')
    ax.plot(t_arr, -vs_net(st_a)[:, 0], color=C_ADAPT,   lw=1.5, label='VS net (adapted)')
    ax.axvline(ON_DUR, color='k', lw=0.8, ls='--', alpha=0.5)
    ax_fmt(ax, ylabel='VS net yaw (deg/s)')
    ax.set_title('VS state — decay toward null (not 0) prolongs OKAN')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(t_arr, -vs_null(st_h)[:, 0], color=C_HEALTHY, lw=1.0, ls='--',
            alpha=0.5, label=f'VS null (default, τ={THETA.brain.tau_vs_adapt:.0f}s)')
    ax.plot(t_arr, -vs_null(st_a)[:, 0], color=C_ADAPT,   lw=2.0,
            label=f'VS null (adapted, τ={THETA_ADAPT.brain.tau_vs_adapt:.0f}s)')
    ax.axvline(ON_DUR, color='k', lw=0.8, ls='--', alpha=0.5)
    ax_fmt(ax, ylabel='VS null (deg/s)', xlabel='Time (s)')
    ax.set_title('VS null builds during OKN; residual after lights-off → extended OKAN')
    ax.legend(fontsize=8)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'clin_ni_okan', show=show)
    return utils.fig_meta(path, rp,
        title='Extended OKAN / VS Null Adaptation',
        description='VS null adaptation (τ_vs_adapt 600→60 s) prolongs OKAN. '
                    'PAN would require inhibitory null coupling (future work).',
        expected='OKAN duration extended when τ_vs_adapt is shorter. '
                 'VS null builds during sustained OKN, residual prolong decay. '
                 'Normal OKN gain preserved.',
        citation='Cohen et al. (1992) J Neurophysiol; Leigh et al. (1981) Neurology',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────

def run(show=False):
    print('\n=== Clinical: Gaze-Holding & Integrator Disorders ===')
    figs = []
    print('  1/4  Gaze-evoked nystagmus …')
    figs.append(_gen(show))
    print('  2/4  Rebound nystagmus …')
    figs.append(_rebound(show))
    print('  3/4  Bruns nystagmus …')
    figs.append(_bruns(show))
    print('  4/4  Extended OKAN / VS null adaptation …')
    figs.append(_okan_extension(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
