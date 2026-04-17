"""Efference-copy demo — saccadic suppression of OKR contamination.

The efference copy delays u_burst by tau_vis (matching the visual delay cascade)
so that it can be subtracted from raw_slip_delayed before VS — ensuring that
saccade-driven eye motion does not spuriously drive OKR.

Signal flow:
    raw_slip = w_scene - w_head - dx_plant       (instantaneous)
    raw_slip → [visual delay, tau_vis] → raw_slip_delayed
    u_burst  → [EC delay,    tau_vis] → u_burst_delayed
    e_slip_corrected = raw_slip_delayed + u_burst_delayed  → VS

Figures produced
────────────────
    efference_copy.png      — 4-panel: burst delay + corrected slip + VS
    efference_okn_debug.png — 7-panel OKN cascade with / without efference copy

Tests (printed to console)
──────────────────────────
    1. u_burst_delayed stays zero without burst
    2. u_burst_delayed is a delayed copy of u_burst (~tau_vis shift)
    3. VS low after saccade in stationary lit world
    4. VS driven by genuine scene motion
    5. VS not contaminated during OKN nystagmus

Usage
-----
    python -X utf8 scripts/demo_efference.py
"""

import sys
import os

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
SHOW = '--show' in sys.argv
if not SHOW:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, simulate,
    _IDX_C, _IDX_NI, _IDX_SG, _IDX_VIS, _IDX_VS, _IDX_EC,
)
from oculomotor.models.brain_models import saccade_generator as sg_mod
from oculomotor.models.sensory_models.sensory_model import C_slip, C_pos
from oculomotor.models.sensory_models.sensory_model import N_CANALS, canal_nonlinearity
from oculomotor.models.brain_models import velocity_storage as vs_mod

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

_C = {
    'burst':    '#f4a582',
    'delayed':  '#d6604d',
    'eye_vel':  '#2166ac',
    'corrected':'#4dac26',
    'scene':    '#e08214',
    'no_sac':   '#aaaaaa',
}

THETA_SAC     = with_brain(PARAMS_DEFAULT, g_burst=600.0)
THETA_NO_SAC  = with_brain(THETA_SAC,     g_burst=0.0)
THETA_OKN     = with_brain(PARAMS_DEFAULT, g_burst=600.0)
THETA_OKN_OFF = with_brain(PARAMS_DEFAULT, g_burst=0.0)


# ── Utilities ──────────────────────────────────────────────────────────────────

def _extract(states, theta, t_np):
    """Extract efference-copy signals from full state trajectory."""
    tau_p = theta.brain.tau_p
    x_p   = np.array(states.plant[:, :3])   # (T, 3) left eye
    x_ni  = np.array(states.brain[:, _IDX_NI])
    x_vs  = np.array(states.brain[:, _IDX_VS])

    # u_burst_delayed: last 3 states of the EC cascade = delayed burst output
    u_burst_delayed = np.array(states.brain[:, _IDX_EC])[:, -3:]  # (T, 3)

    # u_burst: recompute from SG state + delayed position error
    def _at(state):
        e_pd   = C_pos @ state.sensory[_IDX_VIS]
        _, u_b = sg_mod.step(state.brain[_IDX_SG], e_pd, theta.brain)
        return u_b
    u_burst = np.array(jax.vmap(_at)(states))  # (T, 3)

    # raw_slip_delayed: last 3 states of the slip cascade in sensory
    # C_slip selects states [117:120] of x_vis
    x_vis_np = np.array(states.sensory[:, _IDX_VIS])          # (T, 240)
    raw_slip_delayed = (np.array(C_slip) @ x_vis_np.T).T      # (T, 3)

    # corrected slip = what VS actually receives
    cor_slip = raw_slip_delayed + u_burst_delayed              # (T, 3)

    # w_eye approximation from NI + VS state
    cg = jnp.array(theta.sensory.canal_gains)
    x_c_j   = states.sensory[:, _IDX_C]
    x_vs_j  = states.brain[:, _IDX_VS]
    x_vis_j = states.sensory[:, _IDX_VIS]

    def _w_est_at(xc, xvs, xvis):
        y_c  = canal_nonlinearity(xc, cg)
        e_sl = C_slip @ xvis
        _, w = vs_mod.step(xvs, jnp.concatenate([y_c, e_sl]), theta.brain)
        return w
    w_est = np.array(jax.vmap(_w_est_at)(x_c_j, x_vs_j, x_vis_j))  # (T, 3)
    w_eye = (x_ni - x_p) / tau_p + (-w_est + u_burst)               # (T, 3)

    return dict(x_p=x_p, x_vs=x_vs, u_burst=u_burst,
                u_burst_delayed=u_burst_delayed,
                raw_slip_delayed=raw_slip_delayed,
                cor_slip=cor_slip, w_eye=w_eye)


def _ax_fmt(ax):
    ax.axhline(0, color='k', lw=0.4)
    ax.grid(True, alpha=0.2)


# ── Figure 1: efference_copy.png ──────────────────────────────────────────────

def demo_efference_copy():
    """4-panel efference copy figure:
      P1: u_burst vs u_burst_delayed (shows tau_vis delay)
      P2: raw_slip_delayed vs corrected slip (near zero during saccade in dark)
      P3: VS state — saccade in dark vs scene-driven OKR
      P4: w_eye during saccade
    """
    dt, T_end, t_jump = 0.001, 0.6, 0.1
    t    = jnp.arange(0.0, T_end, dt)
    T    = len(t)
    t_np = np.array(t)

    # 10° saccade — dark (no scene, isolates burst signal)
    pt3 = jnp.stack([
        jnp.where(t >= t_jump, jnp.tan(jnp.radians(10.0)), 0.0),
        jnp.zeros(T), jnp.ones(T),
    ], axis=1)

    max_s = int(T_end / dt) + 200
    states = simulate(THETA_SAC, t, p_target_array=pt3,
                      scene_present_array=jnp.zeros(T),
                      max_steps=max_s, return_states=True)
    s = _extract(states, THETA_SAC, t_np)

    # Scene-driven OKR for comparison (no saccades, scene = 300 deg/s × 50 ms)
    t_dur = 0.6
    t2    = jnp.arange(0.0, t_dur, dt)
    T2    = len(t2)
    t2_np = np.array(t2)
    vs3_scene = jnp.zeros((T2, 3)).at[
        (t2 >= t_jump) & (t2 < t_jump + 0.05), 0
    ].set(300.0)
    states_sc = simulate(THETA_NO_SAC, t2,
                         head_vel_array=jnp.zeros(T2),
                         v_scene_array=vs3_scene,
                         scene_present_array=jnp.ones(T2),
                         max_steps=max_s, return_states=True)
    s_sc = _extract(states_sc, THETA_NO_SAC, t2_np)

    tau_vis = THETA_SAC.sensory.tau_vis

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(
        f'Efference Copy — Burst Delay (tau_vis = {tau_vis*1000:.0f} ms)', fontsize=12)

    for ax in axes:
        ax.axvline(t_jump, color='gray', lw=0.8, ls='--', alpha=0.5)
        _ax_fmt(ax)

    # P1: u_burst vs u_burst_delayed
    axes[0].plot(t_np, s['u_burst'][:, 0],
                 color=_C['burst'], lw=1.5, label='u_burst (immediate)')
    axes[0].plot(t_np, s['u_burst_delayed'][:, 0],
                 color=_C['delayed'], lw=1.5, ls='--',
                 label=f'u_burst_delayed (~{tau_vis*1000:.0f} ms lag)')
    axes[0].set_ylabel('deg/s')
    axes[0].set_title('Burst command and its EC-delayed copy')
    axes[0].legend(fontsize=8)

    # P2: raw_slip_delayed vs corrected slip
    axes[1].plot(t_np, s['raw_slip_delayed'][:, 0],
                 color='#555555', lw=1.2, label='raw_slip_delayed (uncorrected)')
    axes[1].plot(t_np, s['cor_slip'][:, 0],
                 color=_C['corrected'], lw=1.5, label='corrected slip → VS  (should ≈ 0 in dark)')
    axes[1].set_ylabel('deg/s')
    axes[1].set_title('Retinal slip after efference copy correction (dark saccade)')
    axes[1].legend(fontsize=8)

    # P3: VS — scene-driven vs saccade-driven
    axes[2].plot(t2_np, s_sc['x_vs'][:, 0],
                 color=_C['scene'], lw=1.5, label='scene pulse → VS driven')
    axes[2].plot(t_np, s['x_vs'][:, 0],
                 color=_C['eye_vel'], lw=1.5, ls='--',
                 label='saccade in dark → VS suppressed')
    axes[2].set_ylabel('VS state (deg/s)')
    axes[2].set_title('VS response: scene drives OKR; dark saccade does not')
    axes[2].legend(fontsize=8)

    # P4: w_eye
    axes[3].plot(t_np, s['w_eye'][:, 0],
                 color=_C['eye_vel'], lw=1.2, label='w_eye')
    axes[3].set_ylabel('deg/s')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title('Eye velocity during 10° saccade')
    axes[3].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'efference_copy.png')
    fig.savefig(path, dpi=150)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    print(f'  Saved {path}')


# ── Figure 2: efference_okn_debug.png ─────────────────────────────────────────

def demo_okn_debug():
    """7-panel OKN nystagmus cascade — VS with/without saccades."""
    dt    = 0.001
    T_end = 10.0
    t     = jnp.arange(0.0, T_end, dt)
    T     = len(t)
    t_np  = np.array(t)
    max_s = int(T_end / dt) + 1000

    vs3   = jnp.zeros((T, 3)).at[:, 0].set(5.0)   # 5 deg/s scene
    hv0   = jnp.zeros(T)
    pt3   = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (T, 1))
    sp    = jnp.ones(T)

    print('  OKN with saccades...')
    states_sac   = simulate(THETA_OKN,     t,
                             head_vel_array=hv0, v_scene_array=vs3,
                             scene_present_array=sp,
                             max_steps=max_s, return_states=True)
    print('  OKN without saccades...')
    states_nosac = simulate(THETA_OKN_OFF, t,
                             head_vel_array=hv0, v_scene_array=vs3,
                             scene_present_array=sp,
                             max_steps=max_s, return_states=True)

    s    = _extract(states_sac,   THETA_OKN,     t_np)
    s_ns = _extract(states_nosac, THETA_OKN_OFF, t_np)

    x_vs_sac   = s['x_vs'][:, 0]
    x_vs_nosac = s_ns['x_vs'][:, 0]
    x_p_sac    = s['x_p'][:, 0]
    x_ni_sac   = np.array(states_sac.brain[:, _IDX_NI])[:, 0]
    x_sg_sac   = np.array(states_sac.brain[:, _IDX_SG])[:, 3]   # z_ref

    thr       = THETA_OKN.brain.threshold_sac
    w_eye     = np.gradient(x_p_sac, dt)
    SPV_CLIP  = 40.0
    spv       = np.where(np.abs(w_eye) < SPV_CLIP, w_eye, np.nan)

    e_pos_del = (np.array(states_sac.sensory[:, _IDX_VIS]) @ np.array(C_pos).T)[:, 0]
    e_motor   = -x_p_sac   # target at 0°, head stationary

    # corrected slip: raw_slip_delayed + u_burst_delayed (what VS actually receives)
    raw_slip_del = s['raw_slip_delayed'][:, 0]
    cor_slip_val = s['cor_slip'][:, 0]

    fig, axes = plt.subplots(7, 1, figsize=(12, 17), sharex=True)
    fig.suptitle('OKN Nystagmus — Signal Cascade  (scene = 5 deg/s, 10 s)', fontsize=12)

    for ax in axes:
        _ax_fmt(ax)

    axes[0].plot(t_np, x_p_sac, color=_C['eye_vel'], lw=0.8, label='eye pos')
    axes[0].axhline(0, color=_C['scene'], lw=1.2, ls='--', alpha=0.7, label='target (0°)')
    axes[0].set_ylabel('deg')
    axes[0].set_title('Eye position (sawtooth = nystagmus)')
    axes[0].legend(fontsize=8)

    axes[1].plot(t_np, w_eye, color=_C['eye_vel'], lw=0.4, alpha=0.3, label='w_eye (raw)')
    axes[1].plot(t_np, spv,   color=_C['eye_vel'], lw=1.2,
                 label=f'slow phase vel (clip {SPV_CLIP})')
    axes[1].axhline(5, color=_C['scene'], lw=1.2, ls='--', alpha=0.7, label='scene 5 deg/s')
    axes[1].set_ylabel('deg/s')
    axes[1].set_title('Eye velocity — slow phase near scene vel, fast phases spike')
    axes[1].legend(fontsize=8)

    axes[2].plot(t_np, e_motor,   color='#762a83', lw=1.5, label='e_motor = target − eye')
    axes[2].plot(t_np, e_pos_del, color='#d6604d', lw=1.2, ls='--',
                 label='e_pos_delayed (SG input)')
    axes[2].axhline( thr, color='gray', lw=0.8, ls=':', label=f'±threshold ({thr}°)')
    axes[2].axhline(-thr, color='gray', lw=0.8, ls=':')
    axes[2].set_ylabel('deg')
    axes[2].set_title('Position error → saccade trigger')
    axes[2].legend(fontsize=8)

    axes[3].plot(t_np, s['u_burst'][:, 0], color=_C['burst'], lw=1.5, label='u_burst')
    axes[3].plot(t_np, x_sg_sac * 200,     color='purple',    lw=0.8, ls='--',
                 label='z_ref × 200')
    axes[3].set_ylabel('deg/s')
    axes[3].set_title('Burst + refractory state (z_ref × 200)')
    axes[3].legend(fontsize=8)

    axes[4].plot(t_np, raw_slip_del, color='#d6604d', lw=0.8, alpha=0.6,
                 label='raw_slip_delayed (uncorrected)')
    axes[4].plot(t_np, cor_slip_val, color=_C['corrected'], lw=1.2,
                 label='corrected slip → VS')
    axes[4].axhline(5, color=_C['scene'], lw=0.8, ls='--', alpha=0.5, label='scene 5 deg/s')
    axes[4].set_ylabel('deg/s')
    axes[4].set_title('Retinal slip — corrected stays near scene vel during fast phases')
    axes[4].legend(fontsize=8)

    axes[5].plot(t_np, x_vs_nosac, color=_C['no_sac'], lw=1.5, label='no saccades (reference)')
    axes[5].plot(t_np, x_vs_sac,   color=_C['eye_vel'], lw=1.2, ls='--',
                 label='with saccades')
    axes[5].set_ylabel('deg/s')
    axes[5].set_title('VS state — traces overlap if efference copy works')
    axes[5].legend(fontsize=8)

    axes[6].plot(t_np, x_ni_sac, color=_C['corrected'], lw=1.2, label='x_ni')
    axes[6].set_ylabel('deg')
    axes[6].set_title('NI state')
    axes[6].set_xlabel('Time (s)')
    axes[6].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'efference_okn_debug.png')
    fig.savefig(path, dpi=150)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    print(f'  Saved {path}')


# ── Numerical tests ────────────────────────────────────────────────────────────

def _run_tests():
    print('\nRunning numerical tests...')
    dt = 0.001

    # ── Test 1: u_burst_delayed stays zero without burst ──────────────────
    t = jnp.arange(0.0, 0.5, dt)
    states = simulate(THETA_NO_SAC, t, max_steps=int(0.5/dt)+200, return_states=True)
    u_bd = np.array(states.brain[:, _IDX_EC])[:, -3:]   # last 3 of cascade
    assert np.allclose(u_bd, 0.0, atol=1e-6), \
        f'FAIL: u_burst_delayed non-zero without burst; max={np.abs(u_bd).max():.2e}'
    print('  1. u_burst_delayed zero without burst               PASS')

    # ── Test 2: u_burst_delayed is a delayed copy of u_burst (~tau_vis) ───
    t = jnp.arange(0.0, 0.8, dt)
    T = len(t)
    t_np = np.array(t)
    pt3 = jnp.stack([
        jnp.where(t >= 0.1, jnp.tan(jnp.radians(15.0)), 0.0),
        jnp.zeros(T), jnp.ones(T),
    ], axis=1)
    states = simulate(THETA_SAC, t, p_target_array=pt3,
                      scene_present_array=jnp.zeros(T),
                      max_steps=int(0.8/dt)+200, return_states=True)

    def _burst_at(state):
        e_pd   = C_pos @ state.sensory[_IDX_VIS]
        _, u_b = sg_mod.step(state.brain[_IDX_SG], e_pd, THETA_SAC.brain)
        return u_b
    u_burst = np.array(jax.vmap(_burst_at)(states))[:, 0]
    u_bd    = np.array(states.brain[:, _IDX_EC])[:, -3]

    peak_burst = t_np[np.argmax(u_burst)]
    peak_bd    = t_np[np.argmax(u_bd)]
    delay_measured = peak_bd - peak_burst
    tau_vis = THETA_SAC.sensory.tau_vis
    assert abs(delay_measured - tau_vis) < 0.02, \
        f'FAIL: EC delay mismatch; expected ~{tau_vis:.3f} s, got {delay_measured:.3f} s'
    # Both should have a meaningful peak
    assert u_burst.max() > 10.0, \
        f'FAIL: u_burst peak too small; max={u_burst.max():.2f}'
    assert u_bd.max() > 1.0, \
        f'FAIL: u_burst_delayed peak too small; max={u_bd.max():.2f}'
    print(f'  2. u_burst_delayed lags u_burst by ~tau_vis         PASS'
          f'  (measured {delay_measured*1000:.1f} ms, expected {tau_vis*1000:.0f} ms)')

    # ── Test 3: VS low after saccade in stationary lit world ───────────────
    t = jnp.arange(0.0, 0.6, dt)
    T = len(t)
    t_np = np.array(t)
    pt3 = jnp.stack([
        jnp.where(t >= 0.1, jnp.tan(jnp.radians(10.0)), 0.0),
        jnp.zeros(T), jnp.ones(T),
    ], axis=1)
    states = simulate(THETA_SAC, t, p_target_array=pt3,
                      scene_present_array=jnp.ones(T),
                      max_steps=int(0.6/dt)+200, return_states=True)
    x_vs = np.array(states.brain[:, _IDX_VS])
    mask = (t_np > 0.3) & (t_np < 0.55)
    contamination = np.abs(x_vs[mask]).max()
    assert contamination < 5.0, \
        f'FAIL: VS contaminated by saccade; max|x_vs|={contamination:.3f}'
    print(f'  3. VS low after saccade (stationary world)          PASS'
          f'  (max={contamination:.2f})')

    # ── Test 4: VS driven by real scene motion ─────────────────────────────
    t = jnp.arange(0.0, 2.0, dt)
    T = len(t)
    t_np = np.array(t)
    vs3 = jnp.zeros((T, 3)).at[:, 0].set(30.0)
    states = simulate(THETA_NO_SAC, t,
                      head_vel_array=jnp.zeros(T), v_scene_array=vs3,
                      scene_present_array=jnp.ones(T),
                      max_steps=int(2.0/dt)+200, return_states=True)
    x_vs = np.array(states.brain[:, _IDX_VS])
    vs_late = np.abs(x_vs[t_np > 1.0, 0]).mean()
    assert vs_late > 1.0, \
        f'FAIL: VS not driven by scene; mean={vs_late:.2f}'
    print(f'  4. VS driven by scene motion                        PASS'
          f'  (mean={vs_late:.2f})')

    # ── Test 5: VS not contaminated during OKN nystagmus ──────────────────
    t   = jnp.arange(0.0, 10.0, dt)
    T   = len(t)
    t_np = np.array(t)
    vs3 = jnp.zeros((T, 3)).at[:, 0].set(5.0)
    sp  = jnp.ones(T)
    max_s = int(10.0/dt) + 1000

    states_sac   = simulate(THETA_OKN,     t,
                             head_vel_array=jnp.zeros(T), v_scene_array=vs3,
                             scene_present_array=sp, max_steps=max_s, return_states=True)
    states_nosac = simulate(THETA_OKN_OFF, t,
                             head_vel_array=jnp.zeros(T), v_scene_array=vs3,
                             scene_present_array=sp, max_steps=max_s, return_states=True)

    x_vs_sac   = np.array(states_sac.brain[:,   _IDX_VS])[:, 0]
    x_vs_nosac = np.array(states_nosac.brain[:, _IDX_VS])[:, 0]
    mask5   = t_np > 7.0
    vs_diff = np.abs(x_vs_sac[mask5] - x_vs_nosac[mask5]).max()
    assert vs_diff < 5.0, \
        f'FAIL: VS contaminated during OKN; max diff={vs_diff:.2f}'
    print(f'  5. VS uncontaminated during OKN                     PASS'
          f'  (max diff={vs_diff:.2f})')

    print('All tests passed.\n')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=== Efference Copy Demo ===')

    print('\n1. Burst delay + corrected slip + VS suppression')
    demo_efference_copy()

    print('\n2. OKN nystagmus cascade (10 s, may take ~30 s)')
    demo_okn_debug()

    print('\n3. Numerical tests')
    _run_tests()

    print(f'\nDone. Plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
