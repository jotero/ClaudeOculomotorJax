"""Clinical vestibular lesion benchmarks.

Three standardised tests applied to healthy, UVH (neuritis), and VN infarct:
  1. Spontaneous / gaze-evoked nystagmus  — dark vs strobed fixation
  2. Video head impulse test (vHIT)        — lit, raw velocity + gain
  3. Rotary chair (dark) + OKN            — SPV only

Usage:
    python -X utf8 scripts/bench_clinical_vestibular.py
    python -X utf8 scripts/bench_clinical_vestibular.py --show
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
    with_uvh, with_vn_lesion,
    _IDX_VS_L, _IDX_VS_R,
)
from oculomotor.sim import kinematics as km
from oculomotor.analysis import ax_fmt, extract_burst, vs_net, extract_spv, fit_tc

SHOW = '--show' in sys.argv
DT   = 0.001

# Suppress sensory noise for clean deterministic figures
THETA     = with_sensory(PARAMS_DEFAULT,
                         sigma_canal=0.0, sigma_pos=0.0,
                         sigma_vel=0.0,   sigma_slip=0.0)
THETA_UVH = with_uvh(THETA, side='left', canal_gain_frac=0.1)
THETA_VNL = with_vn_lesion(THETA, side='left')

SECTION = dict(
    id='clin_vestibular', title='A. Vestibular Lesions',
    description='Three standardised tests: spontaneous nystagmus, vHIT, and rotary chair/OKN.',
)

C_HEALTHY = '#2166ac'
C_UVH     = '#e08214'
C_VNL     = '#d6604d'
C_NODULUS = '#1a9641'


def _pad3(v1d, axis):
    T   = len(v1d)
    out = np.zeros((T, 3), np.float32)
    out[:, {'yaw': 0, 'pitch': 1, 'roll': 2}[axis]] = v1d
    return out


# ── Simulation helpers ──────────────────────────────────────────────────────────

def _sim_dark(params, t_arr, head_vel_3d=None, key=0):
    """Dark, no target, no scene — spontaneous nystagmus / VOR."""
    T  = len(t_arr)
    hv = head_vel_3d if head_vel_3d is not None else np.zeros((T, 3), np.float32)
    return simulate(params, np.asarray(t_arr),
                    head=km.build_kinematics(np.asarray(t_arr), rot_vel=hv),
                    scene_present_array=np.zeros(T),
                    target_present_array=np.zeros(T),
                    max_steps=int(T * 1.1) + 500,
                    sim_config=SimConfig(warmup_s=0.0),
                    return_states=True,
                    key=jax.random.PRNGKey(key))


def _sim_strobed_fixation(params, t_arr, key=0):
    """Dark + strobed fixation target (position feedback, no motion cue).

    Strobed target gates target_motion_visible → 0, so pursuit integrator
    cannot use visual velocity → fixation suppression without smooth pursuit
    contamination.  No scene (no OKR drive).
    """
    T  = len(t_arr)
    t  = np.asarray(t_arr)
    pt_3d = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (T, 1))
    lv    = np.zeros((T, 3), np.float32)
    return simulate(params, t,
                    target=km.build_target(t, lin_pos=pt_3d, lin_vel=lv),
                    target_strobed_array=np.ones(T, np.float32),
                    scene_present_array=np.zeros(T, np.float32),
                    target_present_array=np.ones(T, np.float32),
                    max_steps=int(T * 1.1) + 500,
                    sim_config=SimConfig(warmup_s=0.0),
                    return_states=True,
                    key=jax.random.PRNGKey(key))


def _sim_hit_lit(params, t_arr, head_vel_3d=None, key=0):
    """Lit vHIT: central fixation target (1 m), scene on.

    warmup_s=2.0 allows visual fixation to partially suppress spontaneous nystagmus
    before the impulse fires.  Catch-up saccades for lesioned cases fire ~120 ms
    after the impulse onset (visual delay cascade) and are visible in raw ev.
    """
    T  = len(t_arr)
    t  = np.asarray(t_arr)
    hv = head_vel_3d if head_vel_3d is not None else np.zeros((T, 3), np.float32)
    pt_3d = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (T, 1))
    lv    = np.zeros((T, 3), np.float32)
    return simulate(params, t,
                    head=km.build_kinematics(t, rot_vel=hv),
                    target=km.build_target(t, lin_pos=pt_3d, lin_vel=lv),
                    scene_present_array=np.ones(T, np.float32),
                    target_present_array=np.ones(T, np.float32),
                    max_steps=int(T * 1.1) + 500,
                    sim_config=SimConfig(warmup_s=2.0),
                    return_states=True,
                    key=jax.random.PRNGKey(key))


def _spv_from_state(st, theta, t_arr):
    ev    = np.gradient(np.array(st.plant[:, 0]), DT)
    burst = np.array(extract_burst(st, theta)[:, 0])
    return ev, extract_spv(t_arr, ev, burst)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spontaneous / gaze-evoked nystagmus
# ─────────────────────────────────────────────────────────────────────────────

def _test_spontaneous(show):
    """Dark vs strobed-fixation for healthy, UVH, VN infarct."""
    DUR = 10.0
    t   = np.arange(0.0, DUR, DT)

    conds = [('Healthy', THETA, C_HEALTHY),
             ('UVH neuritis', THETA_UVH, C_UVH),
             ('VN infarct',   THETA_VNL, C_VNL)]

    # Dark simulations
    dark_results = {}
    for label, theta, _ in conds:
        st = _sim_dark(theta, t)
        ev, spv = _spv_from_state(st, theta, t)
        dark_results[label] = dict(
            pos=np.array(st.plant[:, 0]),
            ev=ev, spv=spv,
            spv_ss=float(np.mean(spv[t > 5.0])))

    # Strobed fixation simulations
    fix_results = {}
    for label, theta, _ in conds:
        st = _sim_strobed_fixation(theta, t)
        ev, spv = _spv_from_state(st, theta, t)
        fix_results[label] = dict(
            pos=np.array(st.plant[:, 0]),
            ev=ev, spv=spv,
            spv_ss=float(np.mean(spv[t > 5.0])))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Spontaneous Nystagmus — Left UVH (Neuritis vs VN Infarct)\n'
                 'Dark vs Fixation Suppression (Strobed Target)',
                 fontsize=11, fontweight='bold')

    # Top row: dark condition
    ax = axes[0, 0]
    for label, theta, color in conds:
        ax.plot(t, dark_results[label]['pos'], color=color, lw=0.8, label=label)
    ax_fmt(ax, ylabel='Eye position (deg)')
    ax.set_title('Eye position — dark')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for label, theta, color in conds:
        r = dark_results[label]
        ax.plot(t, r['spv'], color=color, lw=1.8,
                label=f'{label}  SPV≈{r["spv_ss"]:.0f} deg/s')
    ax_fmt(ax, ylabel='SPV (deg/s)', ylim=(-120, 20))
    ax.set_title('SPV — dark  (beats RIGHT = negative = leftward drift)')
    ax.legend(fontsize=8)

    # Bottom row: strobed fixation
    ax = axes[1, 0]
    for label, theta, color in conds:
        ax.plot(t, fix_results[label]['pos'], color=color, lw=0.8, label=label)
    ax_fmt(ax, ylabel='Eye position (deg)', xlabel='Time (s)')
    ax.set_title('Eye position — strobed fixation (fixation suppression)')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for label, theta, color in conds:
        r = fix_results[label]
        ax.plot(t, r['spv'], color=color, lw=1.8,
                label=f'{label}  SPV≈{r["spv_ss"]:.0f} deg/s')
    ax_fmt(ax, ylabel='SPV (deg/s)', xlabel='Time (s)', ylim=(-120, 20))
    ax.set_title('SPV — strobed fixation  (nystagmus partially suppressed)')
    ax.legend(fontsize=8)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'clin_vest_spontaneous', show=show)
    return utils.fig_meta(path, rp,
        title='Spontaneous Nystagmus — Dark vs Fixation Suppression',
        description='Left UVH (10% canal gain) vs VN infarct (100% VN loss). '
                    'Dark: spontaneous nystagmus at full amplitude. '
                    'Strobed fixation: nystagmus suppressed by position feedback.',
        expected='Dark SPV: healthy ≈ 0; neuritis ≈ −20–30 deg/s; infarct ≈ −80–100 deg/s. '
                 'Strobed fixation: nystagmus reduced ~50–80% vs dark.',
        citation='Halmagyi & Curthoys (1988); Bense et al. (2004)',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Video head impulse test (vHIT) — lit condition
# ─────────────────────────────────────────────────────────────────────────────

def _test_vhit(show):
    """vHIT in lit condition. Raw velocity shown; gains baseline-corrected."""
    HIT_V   = 150.0
    HIT_DUR = 0.15
    PRE_S   = 0.5
    POST_S  = 0.8
    t_hit   = np.arange(0.0, PRE_S + HIT_DUR + POST_S, DT)
    i0, i1  = int(PRE_S / DT), int((PRE_S + HIT_DUR) / DT)

    def _impulse(direction):
        hv = np.zeros(len(t_hit))
        n  = i1 - i0
        hv[i0:i1] = direction * HIT_V * np.sin(np.pi * np.arange(n) / n)
        return _pad3(hv, 'yaw')

    conds_hit = [('Healthy', THETA, C_HEALTHY), ('UVH', THETA_UVH, C_UVH)]

    hit_results = {}
    for label, theta, _ in conds_hit:
        for d, dname in [(+1, 'right'), (-1, 'left')]:
            st    = _sim_hit_lit(theta, t_hit, head_vel_3d=_impulse(d), key=d)
            ev    = np.gradient(np.array(st.plant[:, 0]), DT)
            burst = np.array(extract_burst(st, theta)[:, 0])
            spv   = extract_spv(t_hit, ev, burst)
            hv    = _impulse(d)[:, 0]
            # Gain: baseline-corrected SPV peak within the impulse window.
            # Baseline computed over last 200 ms before impulse to remove tonic SPV
            # from any residual spontaneous nystagmus.
            baseline  = float(np.mean(spv[max(0, i0 - 200):i0]))
            peak_hv   = np.abs(hv[i0:i1]).max()
            peak_spv  = np.abs(spv[i0:i1] - baseline).max()
            gain = peak_spv / peak_hv if peak_hv > 0 else 0
            hit_results[(label, dname)] = dict(ev=ev, spv=spv, hv=hv, gain=gain)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Video Head Impulse Test (vHIT) — Lit Condition\n'
                 'Raw eye velocity; catch-up saccades visible in lesion panels',
                 fontsize=11, fontweight='bold')

    # Top row: rightward impulse (toward intact side for left UVH)
    ax = axes[0, 0]
    r_h = hit_results[('Healthy', 'right')]
    r_u = hit_results[('UVH',     'right')]
    ax.plot(t_hit, r_h['hv'],  color='gray',    lw=1.2, ls='--', label='Head vel')
    ax.plot(t_hit, -r_h['ev'], color=C_HEALTHY, lw=1.2, label=f'Healthy gain={r_h["gain"]:.2f}')
    ax.plot(t_hit, -r_u['ev'], color=C_UVH,     lw=1.2, label=f'UVH     gain={r_u["gain"]:.2f}')
    ax.axvline(PRE_S,           color='k', lw=0.5, ls=':', alpha=0.5)
    ax.axvline(PRE_S + HIT_DUR, color='k', lw=0.5, ls=':', alpha=0.5)
    ax_fmt(ax, ylabel='−Eye vel (deg/s)  [raw]')
    ax.set_title('Rightward impulse (toward intact side)')
    ax.legend(fontsize=8)

    # Bottom row: leftward impulse (toward lesioned side)
    ax = axes[1, 0]
    r_h2 = hit_results[('Healthy', 'left')]
    r_u2 = hit_results[('UVH',     'left')]
    ax.plot(t_hit, r_h2['hv'],  color='gray',    lw=1.2, ls='--', label='Head vel')
    ax.plot(t_hit, -r_h2['ev'], color=C_HEALTHY, lw=1.2, label=f'Healthy gain={r_h2["gain"]:.2f}')
    ax.plot(t_hit, -r_u2['ev'], color=C_UVH,     lw=1.2, label=f'UVH     gain={r_u2["gain"]:.2f}')
    ax.axvline(PRE_S,           color='k', lw=0.5, ls=':', alpha=0.5)
    ax.axvline(PRE_S + HIT_DUR, color='k', lw=0.5, ls=':', alpha=0.5)
    ax_fmt(ax, ylabel='−Eye vel (deg/s)  [raw]', xlabel='Time (s)')
    ax.set_title('Leftward impulse (toward LESIONED side) — catch-up saccade')
    ax.legend(fontsize=8)

    # Middle column: right eye during rightward, left eye during leftward
    ax = axes[0, 1]
    st_rh = _sim_hit_lit(THETA,     t_hit, head_vel_3d=_impulse(+1), key=2)
    st_ru = _sim_hit_lit(THETA_UVH, t_hit, head_vel_3d=_impulse(+1), key=2)
    ax.plot(t_hit, r_h['hv'],  color='gray', lw=1.0, ls='--', label='Head vel')
    ax.plot(t_hit, -np.gradient(np.array(st_rh.plant[:, 3]), DT),
            color=C_HEALTHY, lw=1.0, ls='--', alpha=0.7, label='Healthy R eye')
    ax.plot(t_hit, -np.gradient(np.array(st_ru.plant[:, 3]), DT),
            color=C_UVH,     lw=1.0, ls='--', alpha=0.7, label='UVH R eye')
    ax_fmt(ax, ylabel='−Right eye vel (deg/s)')
    ax.set_title('Right eye — rightward impulse (binocular)')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    st_lh = _sim_hit_lit(THETA,     t_hit, head_vel_3d=_impulse(-1), key=2)
    st_lu = _sim_hit_lit(THETA_UVH, t_hit, head_vel_3d=_impulse(-1), key=2)
    ax.plot(t_hit, r_h2['hv'], color='gray', lw=1.0, ls='--', label='Head vel')
    ax.plot(t_hit, -np.gradient(np.array(st_lh.plant[:, 3]), DT),
            color=C_HEALTHY, lw=1.0, ls='--', alpha=0.7, label='Healthy R eye')
    ax.plot(t_hit, -np.gradient(np.array(st_lu.plant[:, 3]), DT),
            color=C_UVH,     lw=1.0, ls='--', alpha=0.7, label='UVH R eye')
    ax_fmt(ax, ylabel='−Right eye vel (deg/s)', xlabel='Time (s)')
    ax.set_title('Right eye — leftward impulse (binocular)')
    ax.legend(fontsize=8)

    # Right column: gain summary bars
    ax = axes[0, 2]
    gains_r = [hit_results[('Healthy','right')]['gain'], hit_results[('UVH','right')]['gain']]
    gains_l = [hit_results[('Healthy','left')]['gain'],  hit_results[('UVH','left')]['gain']]
    x = np.arange(2)
    ax.bar(x - 0.2, gains_r, 0.35, color=[C_HEALTHY, C_UVH], alpha=0.85, label='Rightward (intact)')
    ax.bar(x + 0.2, gains_l, 0.35, color=[C_HEALTHY, C_UVH], alpha=0.55,
           edgecolor='k', linewidth=0.8, label='Leftward (lesioned)')
    ax.axhline(1.0, color='k', lw=0.8, ls='--', label='Ideal gain')
    ax.set_xticks(x); ax.set_xticklabels(['Healthy', 'UVH neuritis'], fontsize=8)
    ax.set_ylabel('VOR gain (SPV/head vel)'); ax.set_ylim(0, 1.4)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    ax.set_title('VOR gain asymmetry — ipsilesional deficit')

    # Hide unused subplot
    axes[1, 2].set_visible(False)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'clin_vest_vhit', show=show)
    return utils.fig_meta(path, rp,
        title='vHIT — Lit Condition, Left UVH',
        description='150 deg/s half-sine impulse left and right. '
                    'Gain measured as baseline-corrected peak SPV / peak head velocity. '
                    'Overt catch-up saccade visible in leftward (ipsilesional) panel.',
        expected='Healthy gain ≈ 0.9–1.0 both directions. '
                 'UVH rightward gain ≈ 0.9 (intact side). '
                 'UVH leftward gain ≈ 0.1–0.3 (lesioned side, catch-up saccade).',
        citation='Halmagyi & Curthoys (1988); MacDougall et al. (2009)',
        fig_type='cascade')


# ─────────────────────────────────────────────────────────────────────────────
# 3. Rotary chair (dark) and OKN — SPV only
# ─────────────────────────────────────────────────────────────────────────────

def _test_rotary_okn(show):
    """Rotary chair in dark + OKN, same temporal parameters.  Only SPV plotted."""
    REST_S  = 2.0
    ROT_V   = 60.0
    ROT_DUR = 20.0
    CST_DUR = 30.0
    TOTAL   = REST_S + ROT_DUR + CST_DUR
    t       = np.arange(0.0, TOTAL, DT)
    T       = len(t)

    # Rotary chair: head rotation in dark
    hv_1d   = np.where(t < REST_S, 0.0,
               np.where(t < REST_S + ROT_DUR, ROT_V, 0.0)).astype(np.float32)
    hv_3d   = _pad3(hv_1d, 'yaw')

    conds = [('Healthy', THETA, C_HEALTHY),
             ('UVH neuritis', THETA_UVH, C_UVH),
             ('VN infarct',   THETA_VNL, C_VNL)]

    rot_spv = {}
    for label, theta, _ in conds:
        st = _sim_dark(theta, t, head_vel_3d=hv_3d)
        _, spv = _spv_from_state(st, theta, t)
        rot_spv[label] = spv

    # OKN: same temporal profile but as visual scene motion (no head rotation)
    v_scene_1d = np.where(t < REST_S, 0.0,
                  np.where(t < REST_S + ROT_DUR, ROT_V, 0.0)).astype(np.float32)
    v_scene_3d = _pad3(v_scene_1d, 'yaw')
    sp_arr     = np.where(t < REST_S, 0.0,
                  np.where(t < REST_S + ROT_DUR, 1.0, 0.0)).astype(np.float32)

    def _sim_okn(params):
        return simulate(params, t,
                        scene=km.build_kinematics(t, rot_vel=v_scene_3d),
                        scene_present_array=sp_arr,
                        target_present_array=np.zeros(T),
                        max_steps=int(T * 1.1) + 500,
                        sim_config=SimConfig(warmup_s=0.0),
                        return_states=True)

    okn_spv = {}
    for label, theta, _ in conds:
        st = _sim_okn(theta)
        _, spv = _spv_from_state(st, theta, t)
        okn_spv[label] = spv

    # Fit VOR and OKAN TCs
    t_rel = t - (REST_S + ROT_DUR)  # relative to rotation stop

    def _fit_decay(spv_arr, sign):
        tc, _, _ = fit_tc(t_rel, sign * spv_arr, t_start=0.0, t_end=CST_DUR * 0.8)
        return tc

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'Rotary Chair (Dark) + OKN  —  {ROT_V:.0f} deg/s step × {ROT_DUR:.0f} s',
                 fontsize=11, fontweight='bold')

    t_stop = REST_S + ROT_DUR

    ax = axes[0, 0]
    ax.axvline(REST_S,   color='gray', lw=0.7, ls=':', alpha=0.4)
    ax.axvline(t_stop,   color='gray', lw=0.7, ls=':', alpha=0.4)
    ax.plot([0, REST_S, REST_S, t_stop, t_stop, TOTAL],
            [0, 0, ROT_V, ROT_V, 0, 0], color='gray', lw=1.0, ls='--', label='Head vel')
    for label, theta, color in conds:
        tc = _fit_decay(rot_spv[label], -1.0)
        tc_str = f' TC≈{tc:.0f}s' if tc else ''
        ax.plot(t, -rot_spv[label], color=color, lw=1.8, label=f'{label}{tc_str}')
    ax_fmt(ax, ylabel='−SPV (deg/s)')
    ax.set_title('Rotary chair in dark — VOR + post-rotatory nystagmus')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.axvline(REST_S,   color='gray', lw=0.7, ls=':', alpha=0.4)
    ax.axvline(t_stop,   color='gray', lw=0.7, ls=':', alpha=0.4)
    ax.plot([0, REST_S, REST_S, t_stop, t_stop, TOTAL],
            [0, 0, ROT_V, ROT_V, 0, 0], color='gray', lw=1.0, ls='--', label='Scene vel')
    for label, theta, color in conds:
        tc = _fit_decay(okn_spv[label], 1.0)
        tc_str = f' TC≈{tc:.0f}s' if tc else ''
        ax.plot(t, okn_spv[label], color=color, lw=1.8, label=f'{label}{tc_str}')
    ax_fmt(ax, ylabel='SPV (deg/s)', xlabel='Time (s)')
    ax.set_title('OKN (full-field) + OKAN — same temporal parameters')
    ax.legend(fontsize=8)

    # Right column: VS state
    def _sim_dark_vs(params, hv_3d):
        st = _sim_dark(params, t, head_vel_3d=hv_3d)
        return vs_net(st)[:, 0]

    ax = axes[0, 1]
    ax.axvline(REST_S,   color='gray', lw=0.7, ls=':', alpha=0.4)
    ax.axvline(t_stop,   color='gray', lw=0.7, ls=':', alpha=0.4)
    for label, theta, color in conds:
        ax.plot(t, _sim_dark_vs(theta, hv_3d), color=color, lw=1.5, label=label)
    ax_fmt(ax, ylabel='VS net yaw (deg/s)')
    ax.set_title('Velocity storage state — rotary chair')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axvline(REST_S,   color='gray', lw=0.7, ls=':', alpha=0.4)
    ax.axvline(t_stop,   color='gray', lw=0.7, ls=':', alpha=0.4)
    for label, theta, color in conds:
        st = _sim_okn(theta)
        ax.plot(t, vs_net(st)[:, 0], color=color, lw=1.5, label=label)
    ax_fmt(ax, ylabel='VS net yaw (deg/s)', xlabel='Time (s)')
    ax.set_title('Velocity storage state — OKN + OKAN')
    ax.legend(fontsize=8)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'clin_vest_rotary_okn', show=show)
    return utils.fig_meta(path, rp,
        title='Rotary Chair (Dark) + OKN — SPV and VS State',
        description=f'Step rotation {ROT_V:.0f} deg/s × {ROT_DUR:.0f} s in dark, '
                    'then post-rotatory nystagmus. Same temporal parameters for OKN/OKAN. '
                    'UVH: reduced VOR and reduced OKAN from lesioned canal/VN.',
        expected='Healthy VOR SPV ≈ 60 deg/s; post-rotatory TC ≈ 15–20 s. '
                 'UVH: reduced SPV on onset, asymmetric post-rotatory. '
                 'OKN/OKAN: unaffected by peripheral vestibular lesion.',
        citation='Raphan et al. (1977); Cohen et al. (1977)',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────

def run(show=False):
    print('\n=== Clinical: Vestibular Lesions ===')
    figs = []
    print('  1/3  Spontaneous nystagmus (dark vs fixation suppression) …')
    figs.append(_test_spontaneous(show))
    print('  2/3  vHIT — lit, left + right impulses …')
    figs.append(_test_vhit(show))
    print('  3/3  Rotary chair (dark) + OKN …')
    figs.append(_test_rotary_okn(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
