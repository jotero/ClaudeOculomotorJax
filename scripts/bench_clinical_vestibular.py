"""Clinical vestibular lesion benchmarks.

Three standardised tests applied to healthy, UVH (neuritis), and VN infarct:
  1. Spontaneous / gaze-evoked nystagmus  — dark vs lit-room fixation
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
from oculomotor.analysis import ax_fmt, vs_net, extract_spv_states, fit_tc

SHOW = '--show' in sys.argv
DT   = 0.001

# Suppress sensory noise and null adaptation for clean deterministic clinical figures.
# tau_vs_adapt set large so the slow null-tracking doesn't contaminate the primary
# VOR/OKAN time constants over the 50 s test window.
THETA = with_brain(
    with_sensory(PARAMS_DEFAULT,
                 sigma_canal=0.0, sigma_pos=0.0,
                 sigma_vel=0.0,   sigma_slip=0.0),
    tau_vs_adapt=9999.0,
)

# Lesioned conditions also lower tau_vs (peripheral VN damage reduces VS pool size
# and therefore the effective storage TC — see Baloh et al. 1984).
THETA_UVH = with_brain(with_uvh(THETA, side='left', canal_gain_frac=0.1), tau_vs=10.0)
THETA_VNL = with_brain(with_vn_lesion(THETA, side='left'), tau_vs=5.0)

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
                    sim_config=SimConfig(warmup_s=3.0),
                    return_states=True,
                    key=jax.random.PRNGKey(key))


def _sim_fixation_lit(params, t_arr, key=0):
    """Lit room with fixation target — scene on, target on, no strobing."""
    T  = len(t_arr)
    t  = np.asarray(t_arr)
    pt_3d = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (T, 1))
    lv    = np.zeros((T, 3), np.float32)
    return simulate(params, t,
                    target=km.build_target(t, lin_pos=pt_3d, lin_vel=lv),
                    scene_present_array=np.ones(T, np.float32),
                    target_present_array=np.ones(T, np.float32),
                    max_steps=int(T * 1.1) + 500,
                    sim_config=SimConfig(warmup_s=3.0),
                    return_states=True,
                    key=jax.random.PRNGKey(key))


def _sim_hit_lit(params, t_arr, head_vel_3d=None, key=0, target_onset_s=None):
    """vHIT: fixation target (1 m), no full-field scene.

    Scene is excluded so OKR does not compensate the VOR deficit — this matches
    the clinical setup where the patient fixates a small dot and the impulse is
    too brief (150 ms) for full-field optokinetic feedback to act.
    Catch-up saccades for lesioned cases fire ~120 ms after impulse onset
    (visual delay cascade) and are visible in raw eye velocity.

    target_onset_s: if given, target appears at this time (warmup uses arr[0]=0 → no
    target during warmup, preventing corrective-saccade NI saturation for conditions
    with spontaneous nystagmus). If None, target is present throughout.
    """
    T  = len(t_arr)
    t  = np.asarray(t_arr)
    hv = head_vel_3d if head_vel_3d is not None else np.zeros((T, 3), np.float32)
    pt_3d = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (T, 1))
    lv    = np.zeros((T, 3), np.float32)
    if target_onset_s is None:
        tgt_pr = np.ones(T, np.float32)
    else:
        tgt_pr = np.where(t >= target_onset_s, 1.0, 0.0).astype(np.float32)
    return simulate(params, t,
                    head=km.build_kinematics(t, rot_vel=hv),
                    target=km.build_target(t, lin_pos=pt_3d, lin_vel=lv),
                    scene_present_array=np.zeros(T, np.float32),
                    target_present_array=tgt_pr,
                    max_steps=int(T * 1.1) + 500,
                    sim_config=SimConfig(warmup_s=2.0),
                    return_states=True,
                    key=jax.random.PRNGKey(key))


def _spv_from_state(st, t_arr):
    ev  = np.gradient(np.array(st.plant[:, 0]), DT)
    spv = extract_spv_states(st, t_arr)[:, 0]
    return ev, spv


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spontaneous / gaze-evoked nystagmus
# ─────────────────────────────────────────────────────────────────────────────

def _test_spontaneous(show):
    """Dark vs lit-room fixation for healthy, UVH, VN infarct."""
    DUR = 10.0
    t   = np.arange(0.0, DUR, DT)

    conds = [('Healthy', THETA, C_HEALTHY),
             ('UVH neuritis', THETA_UVH, C_UVH),
             ('VN infarct',   THETA_VNL, C_VNL)]

    # Dark simulations
    dark_results = {}
    for label, theta, _ in conds:
        st = _sim_dark(theta, t)
        ev, spv = _spv_from_state(st, t)
        dark_results[label] = dict(
            pos=np.array(st.plant[:, 0]),
            ev=ev, spv=spv,
            spv_ss=float(np.mean(spv[t > 5.0])))

    # Lit fixation simulations (scene on + target on)
    fix_results = {}
    for label, theta, _ in conds:
        st = _sim_fixation_lit(theta, t)
        ev, spv = _spv_from_state(st, t)
        fix_results[label] = dict(
            pos=np.array(st.plant[:, 0]),
            ev=ev, spv=spv,
            spv_ss=float(np.mean(spv[t > 5.0])))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Spontaneous Nystagmus — Left UVH (Neuritis vs VN Infarct)\n'
                 'Dark vs Fixation Suppression (Lit Room)',
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

    # Bottom row: lit fixation
    ax = axes[1, 0]
    for label, theta, color in conds:
        ax.plot(t, fix_results[label]['pos'], color=color, lw=0.8, label=label)
    ax_fmt(ax, ylabel='Eye position (deg)', xlabel='Time (s)')
    ax.set_title('Eye position — lit room, fixation target (fixation suppression)')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for label, theta, color in conds:
        r = fix_results[label]
        ax.plot(t, r['spv'], color=color, lw=1.8,
                label=f'{label}  SPV≈{r["spv_ss"]:.0f} deg/s')
    ax_fmt(ax, ylabel='SPV (deg/s)', xlabel='Time (s)', ylim=(-120, 20))
    ax.set_title('SPV — lit room  (nystagmus suppressed by fixation)')
    ax.legend(fontsize=8)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'clin_vest_spontaneous', show=show, params=PARAMS_DEFAULT,
                              conditions='Dark, no head motion — spontaneous nystagmus from unilateral canal asymmetry')
    return utils.fig_meta(path, rp,
        title='Spontaneous Nystagmus — Dark vs Fixation Suppression',
        description='Left UVH (10% canal gain, tau_vs=10 s) vs VN infarct (tau_vs=5 s). '
                    'Dark: spontaneous nystagmus driven by VN firing-rate asymmetry. '
                    'Lit room with fixation target: scene and target both on; '
                    'OKR and saccadic correction suppress nystagmus.',
        expected='Dark SPV: healthy ≈ 0; neuritis ≈ −20–40 deg/s; infarct ≈ −60–100 deg/s. '
                 'Lit fixation: nystagmus reduced or abolished by combined fixation and OKR. '
                 'Failure to suppress (infarct) indicates severely reduced visual–vestibular interaction.',
        citation='Halmagyi & Curthoys (1988); Bense et al. (2004)',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Video head impulse test (vHIT) — lit condition
# ─────────────────────────────────────────────────────────────────────────────

def _test_vhit(show):
    """vHIT — Healthy, Acute UVH (mild spontaneous nystagmus), Chronic UVH (compensated).

    Acute UVH uses b_lesion=85 (w_est≈15 deg/s): mild tonic imbalance visible as
    spontaneous nystagmus in the pre-impulse trace without saturating the NI
    (NI saturation time ≈50/15=3.3 s > 2.5 s warmup+pre-impulse).

    Chronic UVH uses b_lesion=100 (symmetric tonic balance): central compensation
    has restored VN balance; only the peripheral canal deficit remains.  This is
    the usual chronic presentation seen in clinical vHIT months after the lesion.
    """
    HIT_V   = 150.0
    HIT_DUR = 0.15
    PRE_S   = 1.0    # 1 s pre-impulse: shows spontaneous nystagmus baseline for acute
    POST_S  = 1.0
    t_hit   = np.arange(0.0, PRE_S + HIT_DUR + POST_S, DT)
    i0, i1  = int(PRE_S / DT), int((PRE_S + HIT_DUR) / DT)

    THETA_UVH_ACUTE   = with_brain(
        with_uvh(THETA, side='left', canal_gain_frac=0.1, b_lesion=85.0), tau_vs=10.0)
    THETA_UVH_CHRONIC = with_brain(
        with_uvh(THETA, side='left', canal_gain_frac=0.1, b_lesion=100.0), tau_vs=10.0)

    C_ACUTE   = C_UVH       # orange — active / inflammatory
    C_CHRONIC = '#762a83'   # purple — compensated / chronic

    def _impulse(direction):
        hv = np.zeros(len(t_hit))
        n  = i1 - i0
        hv[i0:i1] = direction * HIT_V * np.sin(np.pi * np.arange(n) / n)
        return _pad3(hv, 'yaw')

    conds_hit = [
        ('Healthy',      THETA,             C_HEALTHY, '-'),
        ('Acute UVH',    THETA_UVH_ACUTE,   C_ACUTE,   '-'),
        ('Chronic UVH',  THETA_UVH_CHRONIC, C_CHRONIC, '--'),
    ]

    # Gain: regression of compensatory SPV onto head velocity over the first 60 ms
    # (before catch-up saccades can fire; visual delay ≈ 80 ms).
    GAIN_S = 0.060
    i_gain = i0 + int(GAIN_S / DT)

    # Target appears 120 ms before impulse so target_visible ≈ 1.0 at impulse onset
    # (tau_vis=80 ms cascade; 120 ms = 1.5× mean delay → ~99.9th percentile).
    # This lets the SG fire one pre-impulse corrective saccade and clear x_NI
    # before the measurement window, avoiding centering-mode contamination.
    TARGET_LEAD_S = 0.12

    hit_results = {}
    for label, theta, _, _ in conds_hit:
        for d, dname in [(+1, 'right'), (-1, 'left')]:
            st    = _sim_hit_lit(theta, t_hit, head_vel_3d=_impulse(d), key=d,
                                 target_onset_s=PRE_S - TARGET_LEAD_S)
            ev  = np.gradient(np.array(st.plant[:, 0]), DT)
            spv = extract_spv_states(st, t_hit)[:, 0]
            hv    = _impulse(d)[:, 0]
            hv_w  = hv[i0:i_gain]
            spv_w = spv[i0:i_gain]
            denom = float(np.dot(hv_w, hv_w))
            gain  = float(np.dot(-spv_w, hv_w) / denom) if denom > 1e-6 else 0.0
            hit_results[(label, dname)] = dict(ev=ev, spv=spv, hv=hv, gain=gain)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Video Head Impulse Test (vHIT) — Left UVH: Acute vs Chronic\n'
                 'Raw eye velocity; catch-up saccade visible in ipsilesional (leftward) panel',
                 fontsize=11, fontweight='bold')

    hv_r = _impulse(+1)[:, 0]
    hv_l = _impulse(-1)[:, 0]

    # Top-left: rightward impulse (toward intact side)
    ax = axes[0, 0]
    ax.plot(t_hit, hv_r, color='gray', lw=1.0, ls='--', label='Head vel')
    for label, _, color, ls in conds_hit:
        r = hit_results[(label, 'right')]
        ax.plot(t_hit, -r['ev'], color=color, lw=1.2, ls=ls,
                label=f'{label}  gain={r["gain"]:.2f}')
    ax.axvline(PRE_S,           color='k', lw=0.5, ls=':', alpha=0.4)
    ax.axvline(PRE_S + HIT_DUR, color='k', lw=0.5, ls=':', alpha=0.4)
    ax_fmt(ax, ylabel='−Eye vel (deg/s)')
    ax.set_title('Rightward impulse — toward INTACT side')
    ax.legend(fontsize=8)

    # Bottom-left: leftward impulse (toward lesioned side)
    ax = axes[1, 0]
    ax.plot(t_hit, hv_l, color='gray', lw=1.0, ls='--', label='Head vel')
    for label, _, color, ls in conds_hit:
        r = hit_results[(label, 'left')]
        ax.plot(t_hit, -r['ev'], color=color, lw=1.2, ls=ls,
                label=f'{label}  gain={r["gain"]:.2f}')
    ax.axvline(PRE_S,           color='k', lw=0.5, ls=':', alpha=0.4)
    ax.axvline(PRE_S + HIT_DUR, color='k', lw=0.5, ls=':', alpha=0.4)
    ax_fmt(ax, ylabel='−Eye vel (deg/s)', xlabel='Time (s)')
    ax.set_title('Leftward impulse — toward LESIONED side (catch-up saccade)')
    ax.legend(fontsize=8)

    # Top-right: gain bar chart
    ax = axes[0, 1]
    labels_bar  = [l for l, _, _, _ in conds_hit]
    colors_bar  = [c for _, _, c, _ in conds_hit]
    gains_r     = [hit_results[(l, 'right')]['gain'] for l in labels_bar]
    gains_l     = [hit_results[(l, 'left')]['gain']  for l in labels_bar]
    x = np.arange(len(labels_bar))
    ax.bar(x - 0.2, gains_r, 0.35, color=colors_bar, alpha=0.85, label='Rightward (intact)')
    ax.bar(x + 0.2, gains_l, 0.35, color=colors_bar, alpha=0.50,
           edgecolor='k', linewidth=0.8, label='Leftward (lesioned)')
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, fontsize=8)
    ax.set_ylabel('VOR gain (SPV / head vel)')
    ax.set_ylim(0, 1.4)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_title('VOR gain asymmetry — ipsilesional deficit')

    axes[1, 1].set_visible(False)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'clin_vest_vhit', show=show, params=PARAMS_DEFAULT,
                              conditions='Lit, fast head impulse (vHIT protocol) — control vs canal hypofunction')
    return utils.fig_meta(path, rp,
        title='vHIT — Acute vs Chronic Left UVH (10% left canal gain)',
        description='150 deg/s half-sine impulse, left and right.  Scene absent, fixation target only. '
                    'Acute UVH: b_lesion=85 (mild tonic imbalance, spontaneous nystagmus visible). '
                    'Chronic UVH: b_lesion=100 (central compensation restored VN balance). '
                    'Gain = regression of early SPV (first 60 ms) onto head velocity.',
        expected='Healthy gain ≈ 0.9 both directions. '
                 'Acute/Chronic rightward (intact side) gain ≈ 0.7–0.9. '
                 'Acute/Chronic leftward (lesioned) gain ≈ 0.1–0.25 with catch-up saccade.',
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
        _, spv = _spv_from_state(st, t)
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
        _, spv = _spv_from_state(st, t)
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
    path, rp = utils.save_fig(fig, 'clin_vest_rotary_okn', show=show, params=PARAMS_DEFAULT,
                              conditions='Dark/Lit, sustained yaw rotation — rotary chair + OKN protocol')
    return utils.fig_meta(path, rp,
        title='Rotary Chair (Dark) + OKN — SPV and VS State',
        description=f'Step rotation {ROT_V:.0f} deg/s × {ROT_DUR:.0f} s in dark, '
                    'then post-rotatory nystagmus. Same temporal parameters for OKN/OKAN. '
                    'Lesioned tau_vs (UVH=10 s, VN infarct=5 s) shortens post-rotatory TC. '
                    'OKN/OKAN unaffected by peripheral vestibular lesion (VS driven by visual slip).',
        expected='Healthy post-rotatory TC ≈ 15–20 s. '
                 'UVH: reduced onset SPV (ipsilesional impulse), TC ≈ 10 s. '
                 'VN infarct: absent or severely reduced onset SPV, TC ≈ 5 s. '
                 'OKN/OKAN: healthy and lesioned cases similar (visual drive intact).',
        citation='Raphan et al. (1977); Cohen et al. (1977); Baloh et al. (1984)',
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
