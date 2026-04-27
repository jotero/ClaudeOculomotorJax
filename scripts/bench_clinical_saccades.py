"""Clinical saccadic disorder benchmarks.

Slow saccades (reduced g_burst / PPRF), ocular flutter, square wave jerks,
saccade palsy (no fast phases during VOR), and main-sequence shifts.

Usage:
    python -X utf8 scripts/bench_clinical_saccades.py
    python -X utf8 scripts/bench_clinical_saccades.py --show
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
)
from oculomotor.sim import kinematics as km
from oculomotor.analysis import ax_fmt, extract_burst

SHOW = '--show' in sys.argv
DT   = 0.001

# Base: noise suppressed for deterministic plots
THETA = with_sensory(PARAMS_DEFAULT,
                     sigma_canal=0.0, sigma_pos=0.0,
                     sigma_vel=0.0, sigma_slip=0.0)

SECTION = dict(
    id='clin_saccades',
    title='D. Saccadic Disorders',
    description='Slow saccades from reduced burst-neuron drive (PPRF/IBN lesion), '
                'ocular flutter and square wave jerks from reduced OPN tone, '
                'and complete saccadic palsy (absence of fast phases during VOR).',
)

C_HEALTHY = '#2166ac'
C_SLOW    = '#e08214'
C_FLUTTER = '#d6604d'
C_PALSY   = '#762a83'
C_SWJ     = '#1a9641'


# ── Simulation helpers ────────────────────────────────────────────────────────

_CFG_WARMUP = SimConfig(warmup_s=2.0)
_CFG_NONE   = SimConfig(warmup_s=0.0)


def _pad3(v1d, axis):
    T = len(v1d)
    out = np.zeros((T, 3), np.float32)
    out[:, {'yaw': 0, 'pitch': 1, 'roll': 2}[axis]] = v1d
    return out


def _sim_saccade(params, amp_deg, t_total=0.5, distance_m=1.0, key=0):
    """Simulate a rightward saccade of amp_deg. Returns (T, 6) eye positions."""
    T  = int(t_total / DT) + 1
    t  = np.linspace(0.0, t_total, T, dtype=np.float32)
    pt = np.zeros((T, 3), np.float32)
    pt[0]  = [0.0, 0.0, distance_m]                            # center (warmup)
    pt[1:] = [np.tan(np.radians(amp_deg)) * distance_m, 0.0, distance_m]
    lv = np.zeros((T, 3), np.float32)
    tgt = km.build_target(t, lin_pos=pt, lin_vel=lv)
    eye = simulate(params, t,
                   target=tgt,
                   scene_present_array=np.ones(T, np.float32),
                   target_present_array=np.ones(T, np.float32),
                   max_steps=int(T * 1.1) + 2500,
                   sim_config=_CFG_WARMUP,
                   return_states=False,
                   key=jax.random.PRNGKey(key))
    return t, np.array(eye)


def _sim_fixation(params, t_arr, key=0):
    """Simulate static fixation at center. Returns (T, 6) eye positions."""
    T = len(t_arr)
    return simulate(params, t_arr,
                    scene_present_array=np.ones(T, np.float32),
                    target_present_array=np.ones(T, np.float32),
                    max_steps=int(T * 1.1) + 1000,
                    sim_config=_CFG_NONE,
                    return_states=False,
                    key=jax.random.PRNGKey(key))


def _sim_vor_nystagmus(params, t_arr, head_vel_deg_s, key=0):
    """Simulate rotation in the dark with saccades. Returns (T, 6) eye positions."""
    T  = len(t_arr)
    hv = _pad3(head_vel_deg_s.astype(np.float32), 'yaw')
    return simulate(params, t_arr,
                    head=km.build_kinematics(t_arr, rot_vel=hv),
                    scene_present_array=np.zeros(T, np.float32),
                    target_present_array=np.zeros(T, np.float32),
                    max_steps=int(T * 1.1) + 500,
                    sim_config=_CFG_NONE,
                    return_states=False,
                    key=jax.random.PRNGKey(key))


# ── Figure 1: Slow saccades — main sequence shift ─────────────────────────────

def _slow_saccades(show):
    print('  Running slow saccades / main sequence...')

    amplitudes  = [5, 10, 15, 20, 30, 40]           # deg
    g_levels    = [700.0, 300.0, 120.0]              # burst ceiling (deg/s)
    labels      = ['Healthy (g=700)', 'Moderate slow (g=300)', 'Severe slow (g=120)']
    colors      = [C_HEALTHY, C_SLOW, C_FLUTTER]
    markers     = ['o', 's', '^']

    # Build params
    param_sets = [THETA] + [with_brain(THETA, g_burst=g) for g in g_levels[1:]]

    # Measure peak velocity for each amplitude × g_burst
    pv_all = np.zeros((len(g_levels), len(amplitudes)))
    for gi, params in enumerate(param_sets):
        for ai, amp in enumerate(amplitudes):
            t, eye = _sim_saccade(params, amp, t_total=0.6)
            vel = np.gradient(eye[:, 3], DT)   # R eye yaw velocity
            pv_all[gi, ai] = np.max(np.abs(vel))

    # Example traces: 20 deg saccade
    t_ex, eye_h  = _sim_saccade(THETA,                              20.0)
    t_ex, eye_s1 = _sim_saccade(with_brain(THETA, g_burst=300.0),   20.0)
    t_ex, eye_s2 = _sim_saccade(with_brain(THETA, g_burst=120.0),   20.0)
    traces = [(eye_h, C_HEALTHY, labels[0]),
              (eye_s1, C_SLOW,   labels[1]),
              (eye_s2, C_FLUTTER, labels[2])]

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14, 8))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_ms  = fig.add_subplot(gs[:, 0])           # main sequence — spans both rows
    ax_pos = fig.add_subplot(gs[0, 1])            # position trace (top-right)
    ax_vel = fig.add_subplot(gs[1, 1], sharex=ax_pos)  # velocity trace (bottom-right)

    # ── Main sequence (left, full height) ────────────────────────────────────
    amps_arr = np.array(amplitudes)
    for gi, (col, lbl, mk) in enumerate(zip(colors, labels, markers)):
        ax_ms.plot(amps_arr, pv_all[gi], color=col, lw=1.8, marker=mk, ms=6, label=lbl)
    ax_ms.set_xlabel('Saccade amplitude (deg)', fontsize=9)
    ax_ms.set_ylabel('Peak eye velocity (deg/s)', fontsize=9)
    ax_ms.set_title('Main Sequence: Peak Velocity vs Amplitude', fontsize=9)
    ax_ms.set_xlim(0, 45)
    ax_ms.set_ylim(0, 820)
    ax_ms.legend(fontsize=8)
    ax_ms.grid(True, alpha=0.2)
    ax_ms.tick_params(labelsize=8)

    # ── Position trace (top-right) ───────────────────────────────────────────
    for eye, col, lbl in traces:
        ax_pos.plot(t_ex, eye[:, 3], color=col, lw=1.5, label=lbl)
    ax_pos.set_ylabel('R eye yaw position (deg)', fontsize=9)
    ax_pos.set_title('20° Saccade — Position', fontsize=9)
    ax_pos.set_xlim(t_ex[0], t_ex[0] + 0.5)
    ax_pos.set_ylim(-2, 25)
    ax_pos.legend(fontsize=8)
    ax_pos.grid(True, alpha=0.15)
    ax_pos.tick_params(labelsize=8)
    plt.setp(ax_pos.get_xticklabels(), visible=False)

    # ── Velocity trace (bottom-right) ────────────────────────────────────────
    for eye, col, _ in traces:
        vel_r = np.gradient(eye[:, 3], DT)
        ax_vel.plot(t_ex, vel_r, color=col, lw=1.5)
    ax_vel.set_ylabel('R eye yaw velocity (deg/s)', fontsize=9)
    ax_vel.set_xlabel('Time (s)', fontsize=9)
    ax_vel.set_title('20° Saccade — Velocity', fontsize=9)
    ax_vel.set_ylim(-50, 800)
    ax_vel.grid(True, alpha=0.15)
    ax_vel.tick_params(labelsize=8)

    fig.suptitle('Slow Saccades — Reduced Burst Neuron Drive (PPRF / IBN Lesion)',
                 fontsize=10, fontweight='bold')

    path, rp = utils.save_fig(fig, 'clin_sac_slow', show=show)
    return utils.fig_meta(path, rp,
        title='Slow Saccades (Main Sequence Shift)',
        description='Main sequence curves (left, full height) and 20° saccade position + '
                    'velocity traces (right, stacked) for three levels of burst-neuron '
                    'ceiling (g_burst). Reduced g_burst models PPRF / IBN lesion.',
        expected='Reduced g_burst lowers peak velocity across all amplitudes. '
                 'Healthy: peak ~600 deg/s at 40°. Severe (g=120): ~170 deg/s. '
                 'Duration increases; final position preserved by local feedback.',
        citation='Zee DS et al. (1976) Ann Neurol 1:309-315 — PPRF slow saccades.',
        fig_type='behavior',
    )


# ── Figure 2: Flutter and square wave jerks ───────────────────────────────────

def _flutter_swj(show):
    print('  Running flutter / square wave jerks...')

    DUR_FIX  = 5.0
    t_fix    = np.arange(0.0, DUR_FIX, DT, dtype=np.float32)
    T_fix    = len(t_fix)

    # Ocular flutter: OPN threshold reduced → SG fires without external target error
    THETA_FLT = with_brain(THETA, threshold_sac=0.02, threshold_sac_release=0.02)

    # SWJ: slightly relaxed threshold + retinal position drift noise
    THETA_SWJ = with_sensory(
        with_brain(PARAMS_DEFAULT, threshold_sac=0.15, threshold_sac_release=0.12),
        sigma_canal=0.0, sigma_slip=0.0, sigma_vel=0.0,
        sigma_pos=0.3, tau_pos_drift=0.25,   # OU drift triggers sparse saccades
    )

    eye_h   = _sim_fixation(THETA,     t_fix, key=1)
    eye_flt = _sim_fixation(THETA_FLT, t_fix, key=1)
    eye_swj = np.array(simulate(
        THETA_SWJ, t_fix,
        scene_present_array=np.ones(T_fix, np.float32),
        target_present_array=np.ones(T_fix, np.float32),
        max_steps=int(T_fix * 1.1) + 1000,
        sim_config=_CFG_NONE,
        return_states=False,
        key=jax.random.PRNGKey(7),
    ))

    eye_h   = np.array(eye_h)
    eye_flt = np.array(eye_flt)

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=False)
    fig.suptitle('Saccadic Oscillations — Flutter and Square Wave Jerks',
                 fontsize=10, fontweight='bold')

    def _vel(e, col=0): return np.gradient(e[:, col], DT)

    for row, (eye, col, title_base) in enumerate([
        (eye_h,   C_HEALTHY, 'Healthy — center fixation'),
        (eye_flt, C_FLUTTER, 'Ocular Flutter\n(threshold_sac = 0.02°)'),
        (eye_swj, C_SWJ,     'Square Wave Jerks\n(threshold = 0.15°, pos noise)'),
    ]):
        vel = _vel(eye)
        axes[row, 0].plot(t_fix, eye[:, 3], color=col, lw=0.8)
        axes[row, 1].plot(t_fix, vel,        color=col, lw=0.8)
        axes[row, 0].set_ylabel('R eye yaw (deg)', fontsize=8)
        axes[row, 1].set_ylabel('R eye yaw vel (deg/s)', fontsize=8)
        axes[row, 0].set_title(title_base, fontsize=8)
        axes[row, 1].set_title(title_base + ' — velocity', fontsize=8)
        for ax in axes[row]:
            ax.axhline(0, color='k', lw=0.4, alpha=0.3)
            ax.grid(True, alpha=0.12)
            ax.tick_params(labelsize=7)
            ax.set_xlabel('Time (s)', fontsize=8)

    # Zoom velocity panels
    axes[0, 1].set_ylim(-30, 30)
    axes[1, 1].set_ylim(-600, 600)
    axes[2, 1].set_ylim(-600, 600)
    axes[0, 0].set_ylim(-1, 1)
    axes[1, 0].set_ylim(-5, 5)
    axes[2, 0].set_ylim(-5, 5)

    plt.tight_layout()

    path, rp = utils.save_fig(fig, 'clin_sac_flutter_swj', show=show)
    return utils.fig_meta(path, rp,
        title='Ocular Flutter and Square Wave Jerks',
        description='Static center fixation under healthy, flutter, and SWJ conditions. '
                    'Flutter: OPN threshold set near zero → saccade generator oscillates '
                    'continuously. SWJ: moderately relaxed threshold + retinal position noise '
                    '→ pairs of back-to-back saccades with the inter-saccadic fixation interval. '
                    'Left column = position, right = velocity.',
        expected='Healthy: near-zero drift during fixation (microsaccade-free without noise). '
                 'Flutter: high-frequency sinusoidal saccadic oscillation ~100 Hz, '
                 'amplitude 2–5°. SWJ: paired saccades at ~1 Hz, '
                 'right-left or left-right, separated by fixation interval of ~150–200 ms.',
        citation='Zee DS & Robinson DA (1979) Ann Neurol 5:207-209 — flutter mechanism; '
                 'Hepp K et al. (1989) Exp Brain Res 75:551-564 — OPN and saccadic control.',
        fig_type='cascade',
    )



# ── Entry point ───────────────────────────────────────────────────────────────

FIGURES = None

def run(show=False):
    global FIGURES
    figs = [
        _slow_saccades(show),
        _flutter_swj(show),
    ]
    FIGURES = figs
    return figs


if __name__ == '__main__':
    figs = run(show=SHOW)
    for f in figs:
        print(f'  Saved: {f["path"]}')
