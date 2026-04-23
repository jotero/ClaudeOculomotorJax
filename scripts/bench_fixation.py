"""Fixation benchmarks — noise source comparison.

Usage:
    python -X utf8 scripts/bench_fixation.py
    python -X utf8 scripts/bench_fixation.py --show
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

from oculomotor.sim.simulator import PARAMS_DEFAULT, with_sensory, simulate
from oculomotor.analysis import ax_fmt, extract_burst

SHOW  = '--show' in sys.argv
DT    = 0.001
TEND  = 3.0


def _gen_noise(params, T, key):
    """Regenerate noise arrays (must match simulate() exactly)."""
    k_canal, _, k_pos, k_vel = jax.random.split(key, 4)
    noise_canal = np.array(jax.random.normal(k_canal, (T, 6)) * params.sensory.sigma_canal)
    noise_vel   = np.array(jax.random.normal(k_vel,   (T, 3)) * params.sensory.sigma_vel)
    alpha = float(jnp.exp(-DT / params.sensory.tau_pos_drift))
    drive = float(jnp.sqrt(1.0 - alpha ** 2) * params.sensory.sigma_pos)
    white = np.array(jax.random.normal(k_pos, (T, 3)))
    pos   = np.zeros((T, 3))
    for i in range(1, T):
        pos[i] = alpha * pos[i - 1] + drive * white[i]
    return dict(canal=noise_canal, pos=pos, vel=noise_vel)


def _run_fixation(sigma_canal, sigma_pos, sigma_vel, seed, T):
    params = with_sensory(PARAMS_DEFAULT,
                          sigma_canal=sigma_canal,
                          sigma_pos=sigma_pos,
                          sigma_vel=sigma_vel)
    t      = jnp.arange(0.0, TEND, DT)
    T_act  = len(t)
    pt3    = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (T_act, 1))
    states = simulate(params, t,
                      p_target_array=pt3,
                      scene_present_array=jnp.ones(T_act),
                      target_present_array=jnp.ones(T_act),
                      max_steps=int(TEND / DT) + 2000,
                      return_states=True,
                      key=jax.random.PRNGKey(seed))
    t_np   = np.array(t)
    eye    = np.array(states.plant[:, :3])
    ev     = np.gradient(eye, DT, axis=0)
    burst  = extract_burst(states, params)
    noise  = _gen_noise(params, T_act, jax.random.PRNGKey(seed))
    return t_np, eye, ev, burst, noise, params


# ── Figure 1: noise source comparison ────────────────────────────────────────

def _noise_comparison(show):
    conditions = [
        (0.0, 0.0, 0.0,  'Noiseless',                'none'),
        (3.0, 0.0, 0.0,  'Canal  σ=3 deg/s',          'canal'),
        (0.0, 0.3, 0.0,  'Retinal pos  σ=0.3 deg',    'pos'),
        (0.0, 0.0, 5.0,  'Retinal vel  σ=5 deg/s',    'vel'),
        (3.0, 0.3, 5.0,  'All combined',               'all'),
    ]
    T    = int(TEND / DT)
    seed = 7

    n_rows, n_cols = 4, len(conditions)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 2.6 * n_rows), sharex=True)
    fig.suptitle('Fixational Eye Movements — Noise Source Comparison\n'
                 '(head stationary, lit scene, target at 0°)', fontsize=11)

    row_labels = ['Eye position (deg)', 'Eye velocity (deg/s)',
                  'Saccade burst (deg/s)', 'Noise signal']
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8)

    pos_lims = [None] * n_cols
    vel_lims = [None] * n_cols

    results = []
    for ci, (sc, sp, sv, title, nkey) in enumerate(conditions):
        t_np, eye, ev, burst, noise, params = _run_fixation(sc, sp, sv, seed, T)
        results.append((t_np, eye, ev, burst, noise, nkey, title))

    # Compute shared y-axis limits across all conditions for rows 0+1
    all_pos = np.concatenate([r[1][:, 0] for r in results])
    all_vel = np.concatenate([r[2][:, 0] for r in results])
    pos_lo, pos_hi = np.percentile(all_pos, 1), np.percentile(all_pos, 99)
    vel_lo, vel_hi = np.percentile(all_vel, 1), np.percentile(all_vel, 99)
    pos_span = max(pos_hi - pos_lo, 0.5)
    vel_span = max(vel_hi - vel_lo, 10.0)
    pos_mid  = (pos_lo + pos_hi) / 2
    vel_mid  = (vel_lo + vel_hi) / 2
    p_lim = (pos_mid - pos_span * 0.6, pos_mid + pos_span * 0.6)
    v_lim = (vel_mid - vel_span * 0.6, vel_mid + vel_span * 0.6)

    for ci, (t_np, eye, ev, burst, noise, nkey, title) in enumerate(results):
        axes[0, ci].set_title(title, fontsize=9, fontweight='bold')

        axes[0, ci].axhline(0, color=utils.C['target'], lw=1.0, ls=':', alpha=0.7)
        axes[0, ci].plot(t_np, eye[:, 0], color=utils.C['eye'], lw=0.8)
        ax_fmt(axes[0, ci]); axes[0, ci].set_ylim(p_lim)

        axes[1, ci].plot(t_np, ev[:, 0], color=utils.C['pursuit'], lw=0.7)
        ax_fmt(axes[1, ci]); axes[1, ci].set_ylim(v_lim)

        axes[2, ci].plot(t_np, burst[:, 0], color=utils.C['burst'], lw=0.8)
        ax_fmt(axes[2, ci])
        b_span = max(np.max(np.abs(burst[:, 0])) * 1.2, 10.0)
        axes[2, ci].set_ylim(-b_span, b_span)

        ax3 = axes[3, ci]
        if nkey == 'none':
            ax3.plot(t_np, np.zeros(len(t_np)), color='gray', lw=0.7)
        elif nkey == 'all':
            ax3.plot(t_np, noise['canal'][:, 0], color='#555555', lw=0.5, label='canal')
            ax3.plot(t_np, noise['pos'][:,   0], color=utils.C['eye'],   lw=0.8, label='pos')
            ax3.plot(t_np, noise['vel'][:,   0], color=utils.C['scene'], lw=0.5, label='vel')
            ax3.legend(fontsize=6, loc='upper right')
        else:
            ax3.plot(t_np, noise[nkey][:, 0], color=utils.C['vs'], lw=0.7)
        ax_fmt(ax3)
        ax3.set_xlabel('Time (s)', fontsize=8)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'fixation_noise_comparison', show=show)
    return utils.fig_meta(path, rp,
        title='Fixational Eye Movements — Noise Source Comparison',
        description='5-column comparison: noiseless, canal noise only, retinal position OU drift, '
                    'retinal velocity noise, and all three combined. '
                    'Rows: eye position, velocity, saccade burst, noise signal.',
        expected='Noiseless: eye stays at 0°. Canal noise: slow drift + rare microsaccades. '
                 'Pos noise: OU drift → corrective microsaccades when error exceeds threshold. '
                 'Vel noise: smooth pursuit-like drift.',
        citation='Rolfs (2009) Neurosci Biobehav Rev 33:1597–1627',
        fig_type='behavior')


SECTION = dict(
    id='fixation', title='6. Fixation',
    description='Fixational eye movements driven by different noise sources. '
                'Tests canal noise filtering, retinal position OU drift (microsaccades), '
                'and retinal velocity noise (pursuit-like drift).',
)


def run(show=False):
    print('\n=== Fixation ===')
    figs = []
    print('  1/1  noise source comparison …')
    figs.append(_noise_comparison(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
