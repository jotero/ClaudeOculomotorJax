"""Efference-copy demo — saccadic suppression of OKR contamination.

The efference copy delays u_burst by tau_vis (matching the visual delay cascade)
so that it can be added to raw_slip_delayed before VS — cancelling saccade-driven
eye motion from the retinal slip signal.

Signal flow:
    raw_slip = w_scene - w_head - dx_plant       (instantaneous)
    raw_slip → [visual delay, tau_vis] → raw_slip_delayed
    u_burst  → [EC delay,    tau_vis] → u_burst_delayed
    e_slip_corrected = raw_slip_delayed + u_burst_delayed  → VS

Figures produced
────────────────
    efference_plant_copy.png   — burst delay cascade: u_burst vs u_burst_delayed
    efference_okr_compare.png  — scene slip vs saccadic slip: OKR response
    efference_cancellation.png — corrected vs raw_slip_delayed during saccade
    efference_okn_debug.png    — OKN nystagmus: VS with/without saccades

Tests (printed to console)
──────────────────────────
    1. u_burst_delayed stays zero without burst
    2. u_burst_delayed peaks ~tau_vis after u_burst
    3. OKR store uncontaminated after saccade in dark
    4. OKR still driven by genuine scene motion
    5. VS not contaminated by repeated fast-phase saccades during OKN

Usage
-----
    python -X utf8 scripts/demo_efference_copy.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from oculomotor.sim.simulator import (
    simulate, PARAMS_DEFAULT, with_brain,
    _IDX_C, _IDX_NI, _IDX_SG, _IDX_VIS, _IDX_VS, _IDX_EC,
)
from oculomotor.models import saccade_generator as sg
from oculomotor.models.sensory_model import C_slip, C_pos
from oculomotor.models.sensory_model import N_CANALS, canal_nonlinearity
from oculomotor.models import velocity_storage as vs_mod
from oculomotor.models import retina

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

_C = {
    'burst':    '#f4a582',
    'delayed':  '#d6604d',
    'eye_vel':  '#2166ac',
    'corrected':'#4dac26',
    'okr':      '#762a83',
    'scene':    '#e08214',
    'no_sac':   '#aaaaaa',
}

THETA_SAC    = with_brain(PARAMS_DEFAULT, g_burst=600.0)
THETA_NO_SAC = with_brain(THETA_SAC,     g_burst=0.0)
THETA_OKN     = with_brain(PARAMS_DEFAULT, g_burst=600.0)
THETA_OKN_OFF = with_brain(PARAMS_DEFAULT, g_burst=0.0)


# ── Shared runner ──────────────────────────────────────────────────────────────

def _run_full(theta, t, hv3, vs3, pt3, scene_present=True, max_steps=200_000):
    """Run ODE and return full SimState trajectory."""
    T = len(t)
    if scene_present is True:
        sg1 = jnp.ones(T, dtype=jnp.float32)
    elif scene_present is False:
        sg1 = jnp.zeros(T, dtype=jnp.float32)
    else:
        sg1 = jnp.asarray(scene_present, dtype=jnp.float32)

    return simulate(theta, t,
                    head_vel_array=hv3,
                    v_scene_array=vs3,
                    p_target_array=pt3,
                    scene_present_array=sg1,
                    max_steps=max_steps,
                    return_states=True)


def _extract_signals(theta, t, hv3, vs3, pt3, scene_present=True, max_steps=200_000):
    """Run and extract all relevant signals as a dict of arrays."""
    ys   = _run_full(theta, t, hv3, vs3, pt3,
                     scene_present=scene_present, max_steps=max_steps)
    x_p  = np.array(ys.plant)
    x_vs = np.array(ys.brain[:, _IDX_VS])
    x_ni = np.array(ys.brain[:, _IDX_NI])

    # u_burst_delayed: last 3 states of the EC cascade
    u_burst_delayed = np.array(ys.brain[:, _IDX_EC])[:, -3:]  # (T, 3)

    # u_burst: recompute from SG + delayed position error
    def _burst_at(state):
        e_pd   = C_pos @ state.sensory[_IDX_VIS]
        _, u_b = sg.step(state.brain[_IDX_SG], e_pd, theta)
        return u_b
    u_burst = np.array(jax.vmap(_burst_at)(ys))  # (T, 3)

    # raw_slip_delayed from sensory cascade (C_slip selects last stage)
    x_vis_np = np.array(ys.sensory[:, _IDX_VIS])
    raw_slip_delayed = (np.array(C_slip) @ x_vis_np.T).T  # (T, 3)

    # corrected slip = what VS actually receives
    cor_slip = raw_slip_delayed + u_burst_delayed  # (T, 3)

    # w_eye approximation
    tau_p = theta.phys.tau_p
    cg = jnp.array(theta.phys.canal_gains)
    def _w_est_at(x_c_t, x_vs_t, x_vis_t):
        y_canals = canal_nonlinearity(x_c_t, cg)
        e_sl     = C_slip @ x_vis_t
        _, w     = vs_mod.step(x_vs_t, jnp.concatenate([y_canals, e_sl]), theta)
        return w
    w_est = np.array(jax.vmap(_w_est_at)(
        ys.sensory[:, _IDX_C], ys.brain[:, _IDX_VS], ys.sensory[:, _IDX_VIS],
    ))
    w_eye = (x_ni - x_p) / tau_p + (-w_est + u_burst)

    return {
        'x_p':              x_p,
        'x_vs':             x_vs,
        'u_burst':          u_burst,
        'u_burst_delayed':  u_burst_delayed,
        'raw_slip_delayed': raw_slip_delayed,
        'cor_slip':         cor_slip,
        'w_eye':            w_eye,
    }


# ── Demo 1: burst delay cascade ────────────────────────────────────────────────

def demo_plant_copy_dynamics():
    """Show u_burst vs u_burst_delayed during a 10° saccade.

    Verifies the EC cascade delays the burst signal by tau_vis, matching
    the visual delay so cancellation happens at the right phase.
    """
    dt, T_end, t_jump = 0.001, 0.6, 0.1
    t    = jnp.arange(0.0, T_end, dt)
    T    = len(t)
    t_np = np.array(t)
    tau_vis = THETA_SAC.phys.tau_vis

    hv3 = jnp.zeros((T, 3))
    vs3 = jnp.zeros((T, 3))
    pt3 = jnp.zeros((T, 3)).at[:, 2].set(1.0).at[t >= t_jump, 0].set(
        jnp.tan(jnp.radians(10.0)))

    sig = _extract_signals(THETA_SAC, t, hv3, vs3, pt3, scene_present=False)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(
        f'Efference Copy — Burst Delay Cascade  (tau_vis = {tau_vis*1000:.0f} ms)', fontsize=12)

    for ax in axes:
        ax.axvline(t_jump, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.axhline(0, color='k', lw=0.4)

    axes[0].plot(t_np, sig['u_burst'][:, 0],
                 color=_C['burst'], lw=1.5, label='u_burst (immediate)')
    axes[0].plot(t_np, sig['u_burst_delayed'][:, 0],
                 color=_C['delayed'], lw=1.5, ls='--',
                 label=f'u_burst_delayed (~{tau_vis*1000:.0f} ms lag)')
    axes[0].set_ylabel('deg/s')
    axes[0].set_title('Burst command and EC-delayed copy')
    axes[0].legend(fontsize=8)

    axes[1].plot(t_np, sig['raw_slip_delayed'][:, 0],
                 color='#555555', lw=1.2, label='raw_slip_delayed (uncorrected)')
    axes[1].plot(t_np, sig['cor_slip'][:, 0],
                 color=_C['corrected'], lw=1.5,
                 label='corrected slip → VS  (should ≈ 0 in dark)')
    axes[1].set_ylabel('deg/s')
    axes[1].set_title('Slip after efference copy correction')
    axes[1].legend(fontsize=8)

    axes[2].plot(t_np, sig['w_eye'][:, 0], color=_C['eye_vel'], lw=1.2, label='w_eye')
    axes[2].set_ylabel('deg/s')
    axes[2].set_title('Eye velocity during 10° saccade')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'efference_plant_copy.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Saved {path}')


# ── Demo 2: scene slip vs saccadic slip — OKR comparison ──────────────────────

def demo_okr_scene_vs_saccade():
    """Compare OKR response to real scene motion vs matched saccadic eye motion.

    Both produce similar raw retinal motion, but only the scene should drive OKR.
    The EC delay cascade cancels the saccadic contribution in VS.
    """
    dt, T_end = 0.001, 1.0
    t     = jnp.arange(0.0, T_end, dt)
    T     = len(t)
    t_np  = np.array(t)
    t_stim = 0.1

    # ── Case A: real scene motion (no saccade) ─────────────────────────────
    w_scene_pulse = 300.0
    vs3_scene = jnp.zeros((T, 3)).at[
        (t >= t_stim) & (t < t_stim + 0.05), 0].set(w_scene_pulse)
    hv3 = jnp.zeros((T, 3))
    pt3 = jnp.zeros((T, 3)).at[:, 2].set(1.0)

    sig_scene = _extract_signals(THETA_NO_SAC, t, hv3, vs3_scene, pt3,
                                 scene_present=True)

    # ── Case B: saccade in dark (no scene) ─────────────────────────────────
    vs3_dark = jnp.zeros((T, 3))
    pt3_sac  = jnp.zeros((T, 3)).at[:, 2].set(1.0).at[t >= t_stim, 0].set(
        jnp.tan(jnp.radians(10.0)))

    sig_sac = _extract_signals(THETA_SAC, t, hv3, vs3_dark, pt3_sac,
                               scene_present=False)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    fig.suptitle('Efference Copy — Scene Slip vs Saccadic Slip: OKR Response', fontsize=12)

    for ax in axes:
        ax.axvline(t_stim, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.axhline(0, color='k', lw=0.4)

    axes[0].plot(t_np, sig_scene['w_eye'][:, 0], color=_C['scene'],   lw=1.5,
                 label='scene: w_eye')
    axes[0].plot(t_np, sig_sac['w_eye'][:, 0],   color=_C['eye_vel'], lw=1.5,
                 label='saccade: w_eye')
    axes[0].set_ylabel('deg/s')
    axes[0].set_title('Eye velocity (both cases produce retinal motion)')
    axes[0].legend(fontsize=8)

    axes[1].plot(t_np, sig_sac['u_burst_delayed'][:, 0], color=_C['delayed'], lw=1.5,
                 label='saccade: u_burst_delayed (EC)')
    axes[1].plot(t_np, sig_scene['u_burst_delayed'][:, 0], color=_C['scene'], lw=1.5,
                 ls='--', label='scene: u_burst_delayed = 0 (no burst)')
    axes[1].set_ylabel('deg/s')
    axes[1].set_title('EC signal (zero for scene, cancels saccade in VS)')
    axes[1].legend(fontsize=8)

    axes[2].plot(t_np, sig_scene['x_vs'][:, 0], color=_C['scene'],   lw=1.5,
                 label='scene → VS driven')
    axes[2].plot(t_np, sig_sac['x_vs'][:, 0],   color=_C['eye_vel'], lw=1.5,
                 label='saccade → VS suppressed')
    axes[2].set_ylabel('deg/s')
    axes[2].set_title('VS storage state (charged by retinal slip)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'efference_okr_compare.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Saved {path}')


# ── Demo 3: corrected vs raw slip during saccade ───────────────────────────────

def demo_slip_cancellation():
    """Show raw_slip_delayed vs corrected slip side by side during a saccade."""
    dt, T_end, t_jump = 0.001, 0.5, 0.1
    t    = jnp.arange(0.0, T_end, dt)
    T    = len(t)
    t_np = np.array(t)

    hv3 = jnp.zeros((T, 3))
    vs3 = jnp.zeros((T, 3))
    pt3 = jnp.zeros((T, 3)).at[:, 2].set(1.0).at[t >= t_jump, 0].set(
        jnp.tan(jnp.radians(10.0)))

    sig = _extract_signals(THETA_SAC, t, hv3, vs3, pt3, scene_present=False)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    fig.suptitle('Retinal Slip — Raw vs Efference-Copy Corrected', fontsize=12)

    for ax in axes:
        ax.axvline(t_jump, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.axhline(0, color='k', lw=0.4)

    axes[0].plot(t_np, sig['raw_slip_delayed'][:, 0], color=_C['eye_vel'], lw=1.5)
    axes[0].set_ylabel('deg/s')
    axes[0].set_title('raw_slip_delayed: burst-driven eye motion contaminates slip')

    axes[1].plot(t_np, sig['cor_slip'][:, 0], color=_C['corrected'], lw=1.5)
    axes[1].set_ylabel('deg/s')
    axes[1].set_title('Corrected slip (+ u_burst_delayed): saccade contribution cancelled')
    axes[1].set_xlabel('Time (s)')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'efference_cancellation.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Saved {path}')


# ── Demo 4: OKN nystagmus — comprehensive signal cascade ──────────────────────

def demo_okn_nystagmus():
    """Comprehensive signal-cascade debug for OKN nystagmus.

    7 panels:
        1. Eye position + target (sawtooth)
        2. Eye velocity + scene velocity reference
        3. Position error + delayed version + thresholds
        4. u_burst + reset integrator x_sg
        5. Retinal slip — raw_slip_delayed vs corrected
        6. VS state with / without saccades
        7. NI state x_ni
    """
    dt    = 0.001
    T_end = 10.0
    t     = jnp.arange(0.0, T_end, dt)
    T     = len(t)
    t_np  = np.array(t)
    hv3   = jnp.zeros((T, 3))
    vs3   = jnp.zeros((T, 3)).at[:, 0].set(5.0)
    pt3   = jnp.zeros((T, 3)).at[:, 2].set(1.0)

    print('  Running OKN with saccades...')
    ys_sac   = _run_full(THETA_OKN,     t, hv3, vs3, pt3,
                         scene_present=True, max_steps=1_200_000)
    print('  Running OKN without saccades...')
    ys_nosac = _run_full(THETA_OKN_OFF, t, hv3, vs3, pt3,
                         scene_present=True, max_steps=1_200_000)

    tau_p = THETA_OKN.phys.tau_p
    thr   = THETA_OKN.brain.threshold_sac

    # Signals via vmap (no old EC fields)
    def _signals_at(state):
        x_p            = state.plant
        x_ni           = state.brain[_IDX_NI]
        x_sg           = state.brain[_IDX_SG]
        x_vis          = state.sensory[_IDX_VIS]
        u_burst_delayed = state.brain[_IDX_EC][-3:]   # last 3 of cascade
        e_pos_delayed  = C_pos  @ x_vis
        e_slip_delayed = C_slip @ x_vis
        e_motor        = retina.target_to_angle(jnp.array([0.0, 0.0, 1.0])) - x_p
        _, u_burst     = sg.step(x_sg, e_pos_delayed, THETA_OKN)
        cor_slip_val   = e_slip_delayed + u_burst_delayed
        return (x_p[0], x_ni[0], x_sg[0],
                e_motor[0], e_pos_delayed[0],
                u_burst[0], e_slip_delayed[0], cor_slip_val[0])

    out = jax.vmap(_signals_at)(ys_sac)
    (x_p_sac, x_ni_sac, x_sg_sac,
     e_motor, e_pos_del,
     u_burst_all, raw_slip_del, cor_slip_all) = [np.array(o) for o in out]

    x_vs_sac   = np.array(ys_sac.brain[:,   _IDX_VS])[:, 0]
    x_vs_nosac = np.array(ys_nosac.brain[:, _IDX_VS])[:, 0]

    w_eye_sac = np.gradient(x_p_sac, dt)
    SPV_CLIP  = 40.0
    spv       = np.where(np.abs(w_eye_sac) < SPV_CLIP, w_eye_sac, np.nan)

    fig, axes = plt.subplots(7, 1, figsize=(12, 17), sharex=True)
    fig.suptitle('OKN Nystagmus — Signal Cascade Debug  (scene = 5 deg/s)', fontsize=13)

    for ax in axes:
        ax.axhline(0, color='k', lw=0.4)
        ax.grid(True, alpha=0.2)

    axes[0].plot(t_np, x_p_sac, color=_C['eye_vel'], lw=0.8, label='eye pos (x_p)')
    axes[0].axhline(0, color=_C['scene'], lw=1.5, ls='--', alpha=0.8, label='target (0 deg)')
    axes[0].set_ylabel('deg')
    axes[0].set_title('Eye position — sawtooth if fast phases fire')
    axes[0].legend(fontsize=8)

    axes[1].plot(t_np, w_eye_sac, color=_C['eye_vel'], lw=0.5, alpha=0.3, label='w_eye (raw)')
    axes[1].plot(t_np, spv,       color=_C['eye_vel'], lw=1.2,
                 label=f'SPV (clip {SPV_CLIP})')
    axes[1].axhline(5, color=_C['scene'], lw=1.2, ls='--', alpha=0.8, label='scene 5 deg/s')
    axes[1].set_ylabel('deg/s')
    axes[1].set_title('Eye velocity (slow phase near scene; fast phases spike negative)')
    axes[1].legend(fontsize=8)

    axes[2].plot(t_np, e_motor,   color='#762a83', lw=1.5, label='e_motor = target - eye')
    axes[2].plot(t_np, e_pos_del, color='#d6604d', lw=1.2, ls='--',
                 label='e_pos_delayed (SG input)')
    axes[2].axhline( thr, color='gray', lw=0.8, ls=':', alpha=0.8,
                    label=f'threshold ({thr} deg)')
    axes[2].axhline(-thr, color='gray', lw=0.8, ls=':', alpha=0.8)
    axes[2].set_ylabel('deg')
    axes[2].set_title('Position error & delayed error entering saccade generator')
    axes[2].legend(fontsize=8)

    axes[3].plot(t_np, u_burst_all, color=_C['burst'], lw=1.5, label='u_burst')
    axes[3].plot(t_np, x_sg_sac,    color=_C['scene'], lw=1.2, ls='--',
                 label='x_sg (reset integrator)')
    axes[3].set_ylabel('deg/s | deg')
    axes[3].set_title('Burst command (spikes = fast phases) + Robinson copy integrator')
    axes[3].legend(fontsize=8)

    axes[4].plot(t_np, raw_slip_del, color='#d6604d', lw=0.8, alpha=0.6,
                 label='raw_slip_delayed')
    axes[4].plot(t_np, cor_slip_all, color=_C['corrected'], lw=1.2,
                 label='corrected slip → VS')
    axes[4].axhline(5, color=_C['scene'], lw=0.8, ls='--', alpha=0.5,
                    label='scene 5 deg/s')
    axes[4].set_ylabel('deg/s')
    axes[4].set_title('Retinal slip — corrected should stay near 5 deg/s')
    axes[4].legend(fontsize=8)

    axes[5].plot(t_np, x_vs_nosac, color=_C['no_sac'], lw=1.5,
                 label='x_vs  no saccades')
    axes[5].plot(t_np, x_vs_sac,   color=_C['eye_vel'], lw=1.2, ls='--',
                 label='x_vs  with saccades')
    axes[5].set_ylabel('deg/s')
    axes[5].set_title('VS state — traces overlap if efference copy works')
    axes[5].legend(fontsize=8)

    axes[6].plot(t_np, x_ni_sac, color=_C['corrected'], lw=1.2,
                 label='x_ni (NI position command)')
    axes[6].set_ylabel('deg')
    axes[6].set_title('NI state')
    axes[6].set_xlabel('Time (s)')
    axes[6].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'efference_okn_debug.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Saved {path}')


# ── Numerical tests ────────────────────────────────────────────────────────────

def _run_tests():
    print('\nRunning numerical tests...')
    dt = 0.001

    # ── Test 1: u_burst_delayed stays zero without burst ──────────────────
    t = jnp.arange(0.0, 0.5, dt); T = len(t)
    hv3 = jnp.zeros((T, 3)); vs3 = jnp.zeros((T, 3))
    pt3 = jnp.zeros((T, 3)).at[:, 2].set(1.0)
    ys  = _run_full(THETA_NO_SAC, t, hv3, vs3, pt3)
    u_bd = np.array(ys.brain[:, _IDX_EC])[:, -3:]
    assert np.allclose(u_bd, 0.0, atol=1e-6), \
        f'FAIL: u_burst_delayed non-zero without burst; max={np.abs(u_bd).max():.2e}'
    print('  1. u_burst_delayed zero without burst               PASS')

    # ── Test 2: u_burst_delayed peaks ~tau_vis after u_burst ──────────────
    t = jnp.arange(0.0, 0.8, dt); T = len(t); t_np = np.array(t)
    pt3 = jnp.zeros((T, 3)).at[:, 2].set(1.0).at[t >= 0.1, 0].set(
        jnp.tan(jnp.radians(15.0)))
    hv3 = jnp.zeros((T, 3)); vs3 = jnp.zeros((T, 3))
    ys  = _run_full(THETA_SAC, t, hv3, vs3, pt3, scene_present=False)

    def _burst_at(state):
        e_pd   = C_pos @ state.sensory[_IDX_VIS]
        _, u_b = sg.step(state.brain[_IDX_SG], e_pd, THETA_SAC)
        return u_b
    u_burst = np.array(jax.vmap(_burst_at)(ys))[:, 0]
    u_bd    = np.array(ys.brain[:, _IDX_EC])[:, -3]

    peak_burst = t_np[np.argmax(u_burst)]
    peak_bd    = t_np[np.argmax(u_bd)]
    delay_measured = peak_bd - peak_burst
    tau_vis = THETA_SAC.phys.tau_vis
    assert abs(delay_measured - tau_vis) < 0.02, \
        f'FAIL: EC delay mismatch; expected ~{tau_vis:.3f} s, got {delay_measured:.3f} s'
    assert u_burst.max() > 10.0 and u_bd.max() > 1.0, \
        f'FAIL: u_burst or u_bd peak too small'
    print(f'  2. u_burst_delayed lags u_burst by ~tau_vis         PASS'
          f'  (measured {delay_measured*1000:.1f} ms, expected {tau_vis*1000:.0f} ms)')

    # ── Test 3: VS not contaminated after saccade in stationary lit world ─
    t = jnp.arange(0.0, 0.6, dt); T = len(t); t_np = np.array(t)
    pt3 = jnp.zeros((T, 3)).at[:, 2].set(1.0).at[t >= 0.1, 0].set(
        jnp.tan(jnp.radians(10.0)))
    hv3 = jnp.zeros((T, 3)); vs3 = jnp.zeros((T, 3))
    ys  = _run_full(THETA_SAC, t, hv3, vs3, pt3, scene_present=True)
    x_vs = np.array(ys.brain[:, _IDX_VS])
    mask = (t_np > 0.3) & (t_np < 0.55)
    contamination = np.abs(x_vs[mask]).max()
    assert contamination < 5.0, \
        f'FAIL: VS contaminated by saccade; max|x_vs|={contamination:.3f} deg/s'
    print(f'  3. VS low after saccade (stationary world)          PASS'
          f'  (max={contamination:.2f})')

    # ── Test 4: VS still driven by real scene ─────────────────────────────
    t = jnp.arange(0.0, 2.0, dt); T = len(t); t_np = np.array(t)
    hv3 = jnp.zeros((T, 3))
    vs3 = jnp.zeros((T, 3)).at[:, 0].set(30.0)
    pt3 = jnp.zeros((T, 3)).at[:, 2].set(1.0)
    ys  = _run_full(THETA_NO_SAC, t, hv3, vs3, pt3)
    x_vs = np.array(ys.brain[:, _IDX_VS])
    vs_late = np.abs(x_vs[t_np > 1.0, 0]).mean()
    assert vs_late > 1.0, \
        f'FAIL: VS not driven by scene; mean={vs_late:.2f} deg/s'
    print(f'  4. VS driven by genuine scene motion                PASS'
          f'  (mean={vs_late:.2f})')

    # ── Test 5: VS not contaminated by repeated OKN fast phases ──────────
    t_okn = jnp.arange(0.0, 10.0, dt)
    Tn    = len(t_okn); t_okn_np = np.array(t_okn)
    hv_okn = jnp.zeros((Tn, 3))
    vs_okn = jnp.zeros((Tn, 3)).at[:, 0].set(5.0)
    pt_okn = jnp.zeros((Tn, 3)).at[:, 2].set(1.0)

    ys_sac   = _run_full(THETA_OKN,     t_okn, hv_okn, vs_okn, pt_okn,
                         scene_present=True, max_steps=1_200_000)
    ys_nosac = _run_full(THETA_OKN_OFF, t_okn, hv_okn, vs_okn, pt_okn,
                         scene_present=True, max_steps=1_200_000)

    xvs_sac   = np.array(ys_sac.brain[:,   _IDX_VS])
    xvs_nosac = np.array(ys_nosac.brain[:, _IDX_VS])
    mask5    = t_okn_np > 7.0
    vs_diff  = np.abs(xvs_sac[mask5, 0] - xvs_nosac[mask5, 0]).max()
    assert vs_diff < 5.0, \
        f'FAIL: VS contaminated by OKN fast phases; max diff={vs_diff:.2f} deg/s'
    print(f'  5. VS uncontaminated during OKN nystagmus           PASS'
          f'  (max diff x_vs={vs_diff:.2f} deg/s)')

    print('All tests passed.\n')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=== Efference Copy Demo ===')

    print('\n1. Burst delay cascade dynamics')
    demo_plant_copy_dynamics()

    print('\n2. Scene slip vs saccadic slip — OKR comparison')
    demo_okr_scene_vs_saccade()

    print('\n3. Corrected vs raw slip during saccade')
    demo_slip_cancellation()

    print('\n4. OKN nystagmus cascade (10 s, may take ~30 s)...')
    demo_okn_nystagmus()

    print('\n5. Numerical tests')
    _run_tests()

    print(f'\nDone. Plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
