"""Efference-copy demo — saccadic suppression of OKR contamination.

The efference copy (x_pc) tracks the burst-driven eye velocity through a
plant forward model and subtracts it from the retinal slip before it enters
the visual delay cascade, preventing saccades from spuriously driving OKR.

Figures produced
────────────────
    efference_copy.png      — 4-panel: plant copy dynamics + scene vs saccade OKR
    efference_okn_debug.png — 7-panel OKN cascade with / without efference copy

Tests (printed to console)
──────────────────────────
    1. x_pc stays zero without burst
    2. x_ni_pc holds saccade amplitude; x_pc tracks it
    3. VS low after saccade in stationary lit world
    4. VS driven by genuine scene motion
    5. VS not contaminated during OKN nystagmus

Usage
-----
    python -X utf8 scripts/demo_efference.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
SHOW = '--show' in sys.argv
if not SHOW:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    THETA_DEFAULT, simulate,
    _IDX_C, _IDX_NI, _IDX_SG, _IDX_VIS, _IDX_VS, _IDX_EC, _IDX_PC, _IDX_NI_PC,
)
from oculomotor.models import saccade_generator as sg_mod
from oculomotor.models import visual_delay
from oculomotor.models import canal
from oculomotor.models import velocity_storage as vs_mod

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

_C = {
    'burst':   '#f4a582',
    'eye_vel': '#2166ac',
    'pred':    '#d6604d',
    'pc':      '#4dac26',
    'scene':   '#e08214',
    'no_sac':  '#aaaaaa',
}

THETA_SAC    = {**THETA_DEFAULT, 'g_burst': 600.0}
THETA_NO_SAC = {**THETA_SAC,    'g_burst': 0.0}
THETA_OKN    = {**THETA_DEFAULT, 'g_burst': 600.0}
THETA_OKN_OFF = {**THETA_DEFAULT, 'g_burst': 0.0}


# ── Utilities ──────────────────────────────────────────────────────────────────

def _extract(states, theta, t_np):
    """Extract efference-copy signals from full state trajectory."""
    tau_p   = theta['tau_p']
    x_p     = np.array(states.plant)
    x_ni    = np.array(states.brain[:, _IDX_NI])
    x_pc    = np.array(states.brain[:, _IDX_EC][:, _IDX_PC])
    x_ni_pc = np.array(states.brain[:, _IDX_EC][:, _IDX_NI_PC])
    x_vs    = np.array(states.brain[:, _IDX_VS])

    def _at(state):
        e_pd    = visual_delay.C_pos @ state.sensory[_IDX_VIS]
        _, u_b  = sg_mod.step(state.brain[_IDX_SG], e_pd, theta)
        return u_b
    u_burst = np.array(jax.vmap(_at)(states))  # (T, 3)

    w_burst_pred = (x_ni_pc - x_pc) / tau_p + u_burst  # (T, 3)

    # w_est from canal + VS state (needed for exact w_eye)
    cg = jnp.array(theta.get('canal_gains', jnp.ones(canal.N_CANALS)))
    x_c_j   = states.sensory[:, _IDX_C]
    x_vs_j  = states.brain[:, _IDX_VS]
    x_vis_j = states.sensory[:, _IDX_VIS]
    def _w_est_at(xc, xvs, xvis):
        y_c   = canal.canal_nonlinearity(xc, cg)
        e_sl  = visual_delay.C_slip @ xvis
        _, w  = vs_mod.step(xvs, jnp.concatenate([y_c, e_sl]), theta)
        return w
    w_est = np.array(jax.vmap(_w_est_at)(x_c_j, x_vs_j, x_vis_j))  # (T, 3)
    w_eye = (x_ni - x_p) / tau_p + (-w_est + u_burst)               # (T, 3)

    dt = float(t_np[1] - t_np[0])
    return dict(x_p=x_p, x_pc=x_pc, x_ni_pc=x_ni_pc, x_vs=x_vs,
                u_burst=u_burst, w_burst_pred=w_burst_pred, w_eye=w_eye)


def _ax_fmt(ax):
    ax.axhline(0, color='k', lw=0.4)
    ax.grid(True, alpha=0.2)


# ── Figure 1: efference_copy.png ──────────────────────────────────────────────

def demo_efference_copy():
    """4-panel efference copy figure:
      P1: u_burst, w_burst_pred, w_eye during a 10° saccade
      P2: residual slip after cancellation
      P3: x_ni_pc and x_pc (2-state plant copy states)
      P4: VS state: scene-driven vs saccade-driven (should differ)
    """
    dt, T_end, t_jump = 0.001, 0.6, 0.1
    t    = jnp.arange(0.0, T_end, dt)
    T    = len(t)
    t_np = np.array(t)

    # 10° saccade — single step
    pt3 = jnp.stack([
        jnp.where(t >= t_jump, jnp.tan(jnp.radians(10.0)), 0.0),
        jnp.zeros(T), jnp.ones(T),
    ], axis=1)

    max_s = int(T_end / dt) + 200
    states = simulate(THETA_SAC, t, p_target_array=pt3,
                      scene_present_array=jnp.zeros(T),  # dark — isolate burst signal
                      max_steps=max_s, return_states=True)
    s = _extract(states, THETA_SAC, t_np)

    # Scene pulse for comparison: 300 deg/s × 50 ms ≈ same slip as a 10° saccade
    t_dur = 0.6
    t2    = jnp.arange(0.0, t_dur, dt)
    T2    = len(t2)
    vs3_scene = jnp.zeros((T2, 3)).at[
        (t2 >= t_jump) & (t2 < t_jump + 0.05), 0
    ].set(300.0)
    hv0 = jnp.zeros(T2)
    states_sc = simulate(THETA_NO_SAC, t2,
                         head_vel_array=hv0,
                         v_scene_array=vs3_scene,
                         scene_present_array=jnp.ones(T2),
                         max_steps=max_s, return_states=True)
    s_sc = _extract(states_sc, THETA_NO_SAC, np.array(t2))

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle('Efference Copy — Plant Forward Model During 10° Saccade', fontsize=12)

    for ax in axes:
        ax.axvline(t_jump, color='gray', lw=0.8, ls='--', alpha=0.5)
        _ax_fmt(ax)

    axes[0].plot(t_np, s['u_burst'][:, 0],      color=_C['burst'],   lw=1.5, label='u_burst')
    axes[0].plot(t_np, s['w_burst_pred'][:, 0], color=_C['pred'],    lw=1.5, label='w_burst_pred (efference copy)')
    axes[0].plot(t_np, s['w_eye'][:, 0],        color=_C['eye_vel'], lw=1.0, ls='--', alpha=0.7, label='w_eye (actual)')
    axes[0].set_ylabel('deg/s')
    axes[0].set_title('Burst command, efference copy prediction, actual eye velocity')
    axes[0].legend(fontsize=8)

    residual = s['w_eye'][:, 0] - s['w_burst_pred'][:, 0]
    axes[1].plot(t_np, residual, color='#555555', lw=1.2, label='w_eye − w_burst_pred')
    axes[1].set_ylabel('deg/s')
    axes[1].set_title('Residual entering visual delay  (should be ~0 during saccade)')
    axes[1].legend(fontsize=8)

    axes[2].plot(t_np, s['x_ni_pc'][:, 0], color='#1b7837', lw=1.5, label='x_ni_pc (NI copy)')
    axes[2].plot(t_np, s['x_pc'][:, 0],    color=_C['pc'],  lw=1.5, ls='--', label='x_pc (plant copy)')
    axes[2].set_ylabel('deg')
    axes[2].set_title('2-state plant copy: x_ni_pc holds position, x_pc follows with tau_p')
    axes[2].legend(fontsize=8)

    axes[3].plot(t_np, s_sc['x_vs'][:, 0],    color=_C['scene'],   lw=1.5, label='scene motion → VS driven')
    axes[3].plot(t_np, s['x_vs'][:, 0],       color=_C['eye_vel'], lw=1.5, ls='--', label='saccade in dark → VS suppressed')
    axes[3].set_ylabel('VS state (deg/s)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title('VS response: scene drives OKR; saccade does not (efference copy)')
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
    states_sac   = simulate(THETA_OKN,    t,
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

    thr     = THETA_OKN.get('threshold_sac', 0.5)
    w_eye   = np.gradient(x_p_sac, dt)
    raw_slip = 5.0 - w_eye
    cor_slip = 5.0 - w_eye + s['w_burst_pred'][:, 0]
    SPV_CLIP = 40.0
    spv      = np.where(np.abs(w_eye) < SPV_CLIP, w_eye, np.nan)

    e_pos_del = (np.array(states_sac.sensory[:, _IDX_VIS]) @ np.array(visual_delay.C_pos).T)[:, 0]
    e_motor   = -x_p_sac   # target at 0°, head stationary

    fig, axes = plt.subplots(7, 1, figsize=(12, 17), sharex=True)
    fig.suptitle('OKN Nystagmus — Signal Cascade  (scene = 5 deg/s, 10 s)', fontsize=12)

    for ax in axes: _ax_fmt(ax)

    axes[0].plot(t_np, x_p_sac, color=_C['eye_vel'], lw=0.8, label='eye pos')
    axes[0].axhline(0, color=_C['scene'], lw=1.2, ls='--', alpha=0.7, label='target (0°)')
    axes[0].set_ylabel('deg'); axes[0].set_title('Eye position (sawtooth = nystagmus)')
    axes[0].legend(fontsize=8)

    axes[1].plot(t_np, w_eye, color=_C['eye_vel'], lw=0.4, alpha=0.3, label='w_eye (raw)')
    axes[1].plot(t_np, spv,   color=_C['eye_vel'], lw=1.2, label=f'slow phase vel (clip {SPV_CLIP})')
    axes[1].axhline(5, color=_C['scene'], lw=1.2, ls='--', alpha=0.7, label='scene 5 deg/s')
    axes[1].set_ylabel('deg/s'); axes[1].set_title('Eye velocity — slow phase near scene vel, fast phases spike')
    axes[1].legend(fontsize=8)

    axes[2].plot(t_np, e_motor,   color='#762a83', lw=1.5, label='e_motor = target − eye')
    axes[2].plot(t_np, e_pos_del, color='#d6604d', lw=1.2, ls='--', label='e_pos_delayed (SG input)')
    axes[2].axhline( thr, color='gray', lw=0.8, ls=':', label=f'±threshold ({thr}°)')
    axes[2].axhline(-thr, color='gray', lw=0.8, ls=':')
    axes[2].set_ylabel('deg'); axes[2].set_title('Position error → saccade trigger')
    axes[2].legend(fontsize=8)

    axes[3].plot(t_np, s['u_burst'][:, 0], color=_C['burst'], lw=1.5, label='u_burst')
    axes[3].plot(t_np, x_sg_sac * 200,     color='purple',    lw=0.8, ls='--', label='z_ref × 200')
    axes[3].set_ylabel('deg/s'); axes[3].set_title('Burst + refractory state (z_ref × 200)')
    axes[3].legend(fontsize=8)

    axes[4].plot(t_np, raw_slip, color='#d6604d', lw=0.8, alpha=0.6, label='raw slip')
    axes[4].plot(t_np, cor_slip, color=_C['pred'], lw=1.2, label='corrected slip (+ efference copy)')
    axes[4].axhline(5, color=_C['scene'], lw=0.8, ls='--', alpha=0.5, label='scene 5 deg/s')
    axes[4].set_ylabel('deg/s'); axes[4].set_title('Retinal slip — corrected stays near scene vel')
    axes[4].legend(fontsize=8)

    axes[5].plot(t_np, x_vs_nosac, color=_C['no_sac'], lw=1.5, label='no saccades (reference)')
    axes[5].plot(t_np, x_vs_sac,   color=_C['eye_vel'], lw=1.2, ls='--', label='with saccades')
    axes[5].set_ylabel('deg/s'); axes[5].set_title('VS state — traces overlap if efference copy works')
    axes[5].legend(fontsize=8)

    axes[6].plot(t_np, x_ni_sac, color=_C['pc'], lw=1.2, label='x_ni')
    axes[6].set_ylabel('deg'); axes[6].set_title('NI state')
    axes[6].set_xlabel('Time (s)'); axes[6].legend(fontsize=8)

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

    # ── Test 1: x_pc stays zero without burst ─────────────────────────────
    t = jnp.arange(0.0, 0.5, dt); T = len(t)
    states = simulate(THETA_NO_SAC, t, max_steps=int(0.5/dt)+200, return_states=True)
    x_pc = np.array(states.brain[:, _IDX_EC][:, _IDX_PC])
    assert np.allclose(x_pc, 0.0, atol=1e-6), \
        f'FAIL: x_pc non-zero without burst; max={np.abs(x_pc).max():.2e}'
    print('  1. x_pc zero without burst                   PASS')

    # ── Test 2: x_ni_pc holds position; x_pc tracks it ────────────────────
    t = jnp.arange(0.0, 0.8, dt); T = len(t); t_np = np.array(t)
    pt3 = jnp.stack([
        jnp.where(t >= 0.1, jnp.tan(jnp.radians(10.0)), 0.0),
        jnp.zeros(T), jnp.ones(T),
    ], axis=1)
    states = simulate(THETA_SAC, t, p_target_array=pt3,
                      scene_present_array=jnp.zeros(T),
                      max_steps=int(0.8/dt)+200, return_states=True)
    x_ni_pc = np.array(states.brain[:, _IDX_EC][:, _IDX_NI_PC])[:, 0]
    x_pc    = np.array(states.brain[:, _IDX_EC][:, _IDX_PC])[:, 0]
    mask_after   = t_np > 0.2
    mask_settled = t_np > 0.6
    assert np.abs(x_ni_pc[mask_after]).max() > 1.0, \
        f'FAIL: x_ni_pc near zero after saccade; max={np.abs(x_ni_pc[mask_after]).max():.3f}'
    diff = np.abs(x_pc[mask_settled] - x_ni_pc[mask_settled]).max()
    assert diff < 2.0, \
        f'FAIL: x_pc does not track x_ni_pc; max|diff|={diff:.3f}'
    print('  2. x_ni_pc holds position; x_pc tracks it    PASS')

    # ── Test 3: VS low after saccade in stationary lit world ───────────────
    t = jnp.arange(0.0, 0.6, dt); T = len(t); t_np = np.array(t)
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
    print(f'  3. VS low after saccade (stationary world)   PASS  (max={contamination:.2f})')

    # ── Test 4: VS driven by real scene motion ─────────────────────────────
    t = jnp.arange(0.0, 2.0, dt); T = len(t); t_np = np.array(t)
    vs3 = jnp.zeros((T, 3)).at[:, 0].set(30.0)
    states = simulate(THETA_NO_SAC, t,
                      head_vel_array=jnp.zeros(T), v_scene_array=vs3,
                      scene_present_array=jnp.ones(T),
                      max_steps=int(2.0/dt)+200, return_states=True)
    x_vs = np.array(states.brain[:, _IDX_VS])
    vs_late = np.abs(x_vs[t_np > 1.0, 0]).mean()
    assert vs_late > 1.0, \
        f'FAIL: VS not driven by scene; mean={vs_late:.2f}'
    print(f'  4. VS driven by scene motion                 PASS  (mean={vs_late:.2f})')

    # ── Test 5: VS not contaminated during OKN nystagmus ──────────────────
    t  = jnp.arange(0.0, 10.0, dt); T = len(t); t_np = np.array(t)
    vs3 = jnp.zeros((T, 3)).at[:, 0].set(5.0)
    sp  = jnp.ones(T)
    max_s = int(10.0/dt) + 1000

    states_sac   = simulate(THETA_OKN,     t,
                             head_vel_array=jnp.zeros(T), v_scene_array=vs3,
                             scene_present_array=sp, max_steps=max_s, return_states=True)
    states_nosac = simulate(THETA_OKN_OFF, t,
                             head_vel_array=jnp.zeros(T), v_scene_array=vs3,
                             scene_present_array=sp, max_steps=max_s, return_states=True)

    x_vs_sac   = np.array(states_sac.brain[:, _IDX_VS])[:, 0]
    x_vs_nosac = np.array(states_nosac.brain[:, _IDX_VS])[:, 0]
    mask5    = t_np > 7.0
    vs_diff  = np.abs(x_vs_sac[mask5] - x_vs_nosac[mask5]).max()
    assert vs_diff < 5.0, \
        f'FAIL: VS contaminated during OKN; max diff={vs_diff:.2f}'
    print(f'  5. VS uncontaminated during OKN              PASS  (max diff={vs_diff:.2f})')

    print('All tests passed.\n')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=== Efference Copy Demo ===')

    print('\n1. Plant copy dynamics + scene vs saccade OKR')
    demo_efference_copy()

    print('\n2. OKN nystagmus cascade (10 s, may take ~30 s)')
    demo_okn_debug()

    print('\n3. Numerical tests')
    _run_tests()

    print(f'\nDone. Plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
