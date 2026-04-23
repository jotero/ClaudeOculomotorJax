"""Efference copy benchmarks — EC cancellation of self-generated slip.

Tests that saccadic and pursuit motor commands do not contaminate VS/OKR.

Usage:
    python -X utf8 scripts/bench_efference.py
    python -X utf8 scripts/bench_efference.py --show
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as utils

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, with_sensory, simulate,
    _IDX_EC, _IDX_VIS_L, _IDX_PURSUIT,
)
from oculomotor.models.sensory_models.retina import C_slip
from oculomotor.models.brain_models import efference_copy as ec_mod
from oculomotor.analysis import ax_fmt, extract_burst, extract_spv, vs_net

SHOW  = '--show' in sys.argv
DT    = 0.001
THETA = with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_pos=0.0, sigma_vel=0.0)


def _run(theta, t_np, p_target=None, v_target=None, v_scene=None,
         scene_present=None, target_present=None, key=0):
    T  = len(t_np)
    pt = p_target if p_target is not None else jnp.tile(jnp.array([0., 0., 1.]), (T, 1))
    vt = v_target if v_target is not None else jnp.zeros((T, 3))
    vs = v_scene  if v_scene  is not None else jnp.zeros((T, 3))
    sp = scene_present  if scene_present  is not None else jnp.ones(T)
    tp = target_present if target_present is not None else jnp.ones(T)
    return simulate(theta, jnp.array(t_np),
                    p_target_array=pt, v_target_array=vt, v_scene_array=vs,
                    scene_present_array=sp, target_present_array=tp,
                    max_steps=int(T * 1.05) + 500,
                    return_states=True, key=jax.random.PRNGKey(key))


def _ec_signals(st):
    """Return slip_raw (L-eye yaw), motor_ec (yaw), slip_corrected (yaw)."""
    x_vis_L  = np.array(st.sensory[:, _IDX_VIS_L])
    slip_raw = (np.array(C_slip) @ x_vis_L.T)[0, :]

    x_ec_all = jnp.array(st.brain[:, _IDX_EC])
    motor_ec = np.array(jax.vmap(ec_mod.read_delayed)(x_ec_all))[:, 0]

    return slip_raw, motor_ec, slip_raw + motor_ec


# ── Figure 1: saccade in lit scene — EC cancellation ─────────────────────────

def _saccade_ec(show):
    """20° saccade in lit stationary scene — verify EC prevents OKR/pursuit drive."""
    T_end, t_jump = 1.0, 0.15
    t_np = np.arange(0.0, T_end, DT)
    T    = len(t_np)

    pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
    pt3[t_np >= t_jump, 0] = np.tan(np.radians(20.0))

    st = _run(THETA, t_np, p_target=jnp.array(pt3),
              scene_present=jnp.ones(T), target_present=jnp.ones(T), key=0)

    eye   = (np.array(st.plant[:, 0]) + np.array(st.plant[:, 3])) / 2.0
    ev    = np.gradient(eye, DT)
    burst = extract_burst(st, THETA)[:, 0]
    vs_sig = vs_net(st)[:, 0]
    u_pur  = np.array(st.brain[:, _IDX_PURSUIT])[:, 0]

    slip_raw, motor_ec, slip_corr = _ec_signals(st)

    fig, axes = plt.subplots(6, 1, figsize=(12, 2.2 * 6), sharex=True)
    fig.suptitle('Efference Copy — Saccade in Lit Scene\n'
                 '20° saccade, stationary background. EC should cancel retinal slip → VS flat.',
                 fontsize=11)

    row_labels = ['Eye velocity (deg/s)', 'Burst command (deg/s)',
                  'Slip raw (deg/s)', 'Motor EC (deg/s)',
                  'Slip corrected (raw + EC, deg/s)', 'VS net + Pursuit state (deg/s)']
    for r, lbl in enumerate(row_labels):
        axes[r].set_ylabel(lbl, fontsize=8)

    vl = dict(color='gray', lw=0.8, ls='--', alpha=0.5)
    for ax in axes:
        ax.axvline(t_jump, **vl); ax.axhline(0, color='k', lw=0.4); ax.grid(True, alpha=0.2)

    axes[0].plot(t_np, ev,    color=utils.C['eye'],    lw=1.2, label='eye vel')
    axes[1].plot(t_np, burst, color=utils.C['burst'],  lw=1.2, label='u_burst')
    axes[2].plot(t_np, slip_raw,  color='darkorange',  lw=1.0, label='slip_raw')
    axes[3].plot(t_np, motor_ec,  color=utils.C['vs'], lw=1.0, label='motor_ec')
    axes[4].plot(t_np, slip_corr, color=utils.C['ni'], lw=1.5, label='slip_corr ≈ 0?')
    axes[5].plot(t_np, vs_sig, color=utils.C['vs'],     lw=1.8, label='VS net')
    axes[5].plot(t_np, u_pur,  color=utils.C['pursuit'], lw=1.2, ls='--', label='pursuit state')
    axes[5].set_xlabel('Time (s)', fontsize=8)

    for r, ax in enumerate(axes):
        ax_fmt(ax)
        ax.legend(fontsize=7, loc='upper right')

    # enforce minimum velocity range on slip panels
    for r in [2, 3, 4]:
        lo, hi = axes[r].get_ylim()
        span   = max(abs(hi - lo), 15.0)
        mid    = (lo + hi) / 2
        axes[r].set_ylim(mid - span / 2, mid + span / 2)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'efference_saccade', show=show)
    return utils.fig_meta(path, rp,
        title='EC — Saccade in Lit Scene',
        description='20° saccade in stationary lit scene. Row-by-row: eye velocity, burst command, '
                    'raw retinal slip (spikes during saccade), motor EC (delayed opposing signal), '
                    'corrected slip (raw + EC), VS net + pursuit state (both should stay near 0).',
        expected='slip_raw spikes during burst. motor_ec provides equal/opposite delayed correction. '
                 'slip_corr ≈ 0. VS net and pursuit state remain flat (< 1 deg/s) post-saccade.',
        citation='Henn et al. (1974); Robinson (1977)',
        fig_type='cascade')


# ── Figure 2: large pursuit to orbital limit ─────────────────────────────────

def _pursuit_orbital(show):
    """40 deg/s ramp: eye tracks until orbital limit (~50°), VS stays smooth."""
    T_end  = 3.0
    t_jump = 0.3
    vel    = 40.0
    t_np   = np.arange(0.0, T_end, DT)
    T      = len(t_np)

    tgt_deg = np.where(t_np >= t_jump, vel * (t_np - t_jump), 0.0)
    pt3     = np.zeros((T, 3)); pt3[:, 2] = 1.0
    pt3[:, 0] = np.tan(np.radians(tgt_deg))
    vt3     = np.zeros((T, 3))
    vt3[:, 0] = np.where(t_np >= t_jump, vel, 0.0).astype(np.float32)

    st = _run(THETA, t_np, p_target=jnp.array(pt3), v_target=jnp.array(vt3),
              scene_present=jnp.ones(T), target_present=jnp.ones(T), key=1)

    eye    = (np.array(st.plant[:, 0]) + np.array(st.plant[:, 3])) / 2.0
    ev     = np.gradient(eye, DT)
    vs_sig = vs_net(st)[:, 0]
    u_pur  = np.array(st.brain[:, _IDX_PURSUIT])[:, 0]
    burst  = extract_burst(st, THETA)[:, 0]
    _, _, slip_corr = _ec_signals(st)

    orbital_lim = float(THETA.brain.orbital_limit)

    fig, axes = plt.subplots(4, 1, figsize=(12, 2.5 * 4), sharex=True)
    fig.suptitle(f'Efference Copy — Large Pursuit to Orbital Limit\n'
                 f'{vel:.0f} deg/s ramp; orbital limit = ±{orbital_lim:.0f}°',
                 fontsize=11)

    row_labels = ['Position (deg)', 'Eye velocity (deg/s)',
                  'VS net state (deg/s)', 'Pursuit state (deg/s)']
    for r, lbl in enumerate(row_labels):
        axes[r].set_ylabel(lbl, fontsize=8)

    vl = dict(color='gray', lw=0.8, ls='--', alpha=0.5)
    for ax in axes:
        ax.axvline(t_jump, **vl); ax.axhline(0, color='k', lw=0.4); ax.grid(True, alpha=0.2)

    axes[0].plot(t_np, tgt_deg, color=utils.C['target'], lw=1.5, ls=':', label='target')
    axes[0].plot(t_np, eye,     color=utils.C['eye'],    lw=1.5, label='eye (version)')
    axes[0].axhline(orbital_lim, color='tomato', lw=0.8, ls=':', alpha=0.6,
                    label=f'+{orbital_lim:.0f}° limit')
    axes[0].legend(fontsize=8)

    axes[1].plot(t_np, ev,    color=utils.C['eye'],    lw=1.2, label='eye vel')
    axes[1].plot(t_np, burst, color=utils.C['burst'],  lw=0.8, alpha=0.6, label='burst')
    axes[1].axhline(vel, color=utils.C['target'], lw=0.8, ls=':', alpha=0.7,
                    label=f'{vel:.0f} deg/s ref')
    axes[1].set_ylim(-10, vel * 1.3)
    axes[1].legend(fontsize=8)

    axes[2].plot(t_np, vs_sig, color=utils.C['vs'],     lw=1.8, label='VS net')
    axes[3].plot(t_np, u_pur,  color=utils.C['pursuit'], lw=1.5, label='pursuit state')
    axes[3].set_xlabel('Time (s)', fontsize=8)

    for ax in axes:
        ax_fmt(ax)
        ax.legend(fontsize=7, loc='upper left')

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'efference_pursuit_limit', show=show)
    return utils.fig_meta(path, rp,
        title='EC — Large Pursuit to Orbital Limit',
        description=f'{vel:.0f} deg/s ramp pursuit. Eye tracks target until orbital limit '
                    f'(±{orbital_lim:.0f}°), then stops. VS net and pursuit state shown.',
        expected=f'Eye tracks at {vel:.0f} deg/s until ~{orbital_lim:.0f}°, then stops cleanly. '
                 'VS net stays near 0 (pursuit EC prevents OKR contamination). '
                 'No runaway pursuit state after hitting the limit.',
        citation='Collewijn & Tamminga (1984)',
        fig_type='behavior')


# ── Figure 3: OKN to orbital limit — fast-phase EC smoothing ─────────────────

def _okn_orbital(show):
    """30 deg/s scene OKN: fast phases reset at orbital limit; EC keeps VS smooth."""
    T_end = 20.0
    t_np  = np.arange(0.0, T_end, DT)
    T     = len(t_np)
    t_j   = jnp.array(t_np)

    sv = jnp.zeros((T, 3)).at[:, 0].set(30.0)
    sp = jnp.ones(T)

    st = _run(THETA, t_np,
              p_target=jnp.tile(jnp.array([0., 0., 1.]), (T, 1)),
              v_scene=sv, scene_present=sp,
              target_present=jnp.zeros(T), key=2)

    eye   = (np.array(st.plant[:, 0]) + np.array(st.plant[:, 3])) / 2.0
    ev    = np.gradient(eye, DT)
    burst = extract_burst(st, THETA)[:, 0]
    spv   = extract_spv(t_np, ev, burst)
    vs_sig = vs_net(st)[:, 0]
    slip_raw, motor_ec, slip_corr = _ec_signals(st)

    orbital_lim = float(THETA.brain.orbital_limit)

    fig, axes = plt.subplots(4, 1, figsize=(14, 2.5 * 4), sharex=True)
    fig.suptitle('Efference Copy — OKN (30 deg/s) to Orbital Limit\n'
                 'Sawtooth nystagmus with fast phases resetting at ±50°. '
                 'EC should keep VS charging smoothly despite fast phases.',
                 fontsize=11)

    row_labels = ['Eye position (deg)', 'Eye velocity (deg/s)',
                  'VS net state (deg/s)', 'Slip corrected (EC-cancelled, deg/s)']
    for r, lbl in enumerate(row_labels):
        axes[r].set_ylabel(lbl, fontsize=8)

    axes[0].plot(t_np, eye, color=utils.C['eye'], lw=0.7, label='eye position')
    axes[0].axhline(-orbital_lim, color='tomato', lw=0.8, ls=':', alpha=0.6)
    axes[0].axhline( orbital_lim, color='tomato', lw=0.8, ls=':', alpha=0.6,
                    label=f'±{orbital_lim:.0f}° limit')
    axes[0].legend(fontsize=8)

    axes[1].plot(t_np, ev,  color='steelblue', lw=0.5, alpha=0.4, label='eye vel (raw)')
    axes[1].plot(t_np, spv, color=utils.C['spv'], lw=1.8, label='SPV (fast phases removed)')
    axes[1].axhline(30.0, color=utils.C['scene'], lw=1.0, ls=':', alpha=0.7,
                    label='scene = 30 deg/s')
    axes[1].set_ylim(-80, 80)
    axes[1].legend(fontsize=8)

    axes[2].plot(t_np, vs_sig, color=utils.C['vs'], lw=1.8, label='VS net state')
    axes[2].axhline(30.0, color=utils.C['scene'], lw=0.8, ls=':', alpha=0.6,
                    label='scene vel (ref)')
    axes[2].legend(fontsize=8)

    axes[3].plot(t_np, slip_corr, color=utils.C['ni'], lw=0.8, label='slip corrected')
    axes[3].axhline(30.0, color=utils.C['scene'], lw=0.8, ls=':', alpha=0.6,
                    label='scene vel (ref)')
    axes[3].legend(fontsize=8)
    axes[3].set_xlabel('Time (s)', fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.2); ax_fmt(ax)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'efference_okn_orbital', show=show)
    return utils.fig_meta(path, rp,
        title='EC — OKN to Orbital Limit',
        description='30 deg/s full-field scene. OKN sawtooth nystagmus; fast phases reset eye at '
                    'orbital limit. EC-corrected slip tracks scene velocity during slow phases.',
        expected='VS charges smoothly to ~30 deg/s despite fast phases. '
                 'slip_corr ≈ 30 deg/s during slow phases, drops to ~0 during fast phases (EC cancels). '
                 'VS should not show spikes locked to fast phases.',
        citation='Raphan et al. (1979)',
        fig_type='behavior')


# ── Section entry point ────────────────────────────────────────────────────────

SECTION = dict(
    id='efference', title='7. Efference Copy',
    description='Tests that self-generated motor commands (saccades, pursuit) do not contaminate '
                'velocity storage or pursuit integrator via retinal slip. '
                'EC-corrected slip ≈ 0 during saccades; VS stays flat in stationary scene.',
)


def run(show=False):
    print('\n=== Efference Copy ===')
    figs = []
    print('  1/3  saccade EC cancellation …')
    figs.append(_saccade_ec(show))
    print('  2/3  large pursuit to orbital limit …')
    figs.append(_pursuit_orbital(show))
    print('  3/3  OKN to orbital limit …')
    figs.append(_okn_orbital(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
