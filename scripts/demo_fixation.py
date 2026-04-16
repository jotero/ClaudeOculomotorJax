"""Fixational eye movements driven by canal noise.

Compares noiseless fixation against two noise levels.  Canal noise perturbs
the VS → NI chain, causing slow drift that accumulates until the SG fires a
corrective microsaccade.

Figures produced
────────────────
    fixation.png   — 3-row × 3-column: noiseless / low / high canal noise

Usage
-----
    python -X utf8 scripts/demo_fixation.py
    python -X utf8 scripts/demo_fixation.py --show
"""

import sys
import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
SHOW = '--show' in sys.argv
if not SHOW:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, with_sensory, simulate,
    _IDX_NI, _IDX_SG, _IDX_VIS,
)
from oculomotor.models.brain_models import saccade_generator as sg_mod
from oculomotor.models.sensory_models.sensory_model import C_pos, C_gate

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

_C = {
    'target': '#d6604d',
    'eye':    '#2166ac',
    'burst':  '#f4a582',
    'vel':    '#1a9641',
    'zero':   '#aaaaaa',
}


# ── Shared setup ───────────────────────────────────────────────────────────────

DT   = 0.001
TEND = 2.0       # 2 s fixation


def _make_t():
    return jnp.arange(0.0, TEND, DT)


def _extract(states, params, t_np):
    """Extract eye position, velocity, and burst from a state trajectory."""
    dt   = float(t_np[1] - t_np[0])
    x_p  = np.array(states.plant)          # (T, 3)

    eye_vel = np.gradient(x_p, dt, axis=0) # (T, 3)

    # Saccade burst — re-compute from SG state + delayed retinal signals
    def _burst_at(state):
        x_vis_ = state.sensory[_IDX_VIS]
        e_pd   = C_pos  @ x_vis_
        gate   = (C_gate @ x_vis_)[0]
        x_ni_  = state.brain[_IDX_NI]
        _, u   = sg_mod.step(state.brain[_IDX_SG], e_pd, gate, x_ni_, params.brain)
        return u

    u_burst = np.array(jax.vmap(_burst_at)(states))  # (T, 3)

    return dict(
        eye_pos = x_p,
        eye_vel = eye_vel,
        u_burst = u_burst,
    )


def _ax_fmt(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.axhline(0, color=_C['zero'], lw=0.5, ls='--')
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=7)


def _equalize_row(axes_row, min_span=None):
    """Set the same y-limits across all axes in a row, with an optional minimum span."""
    lo = min(ax.get_ylim()[0] for ax in axes_row)
    hi = max(ax.get_ylim()[1] for ax in axes_row)
    if min_span is not None and (hi - lo) < min_span:
        mid = (lo + hi) / 2
        lo, hi = mid - min_span / 2, mid + min_span / 2
    for ax in axes_row:
        ax.set_ylim(lo, hi)


def _gen_noise(params, T, key):
    """Regenerate the same noise arrays used by simulate() for plotting."""
    k_canal, _, k_pos, k_vel = jax.random.split(key, 4)

    noise_canal = np.array(jax.random.normal(k_canal, (T, 6)) * params.sensory.sigma_canal)
    noise_vel   = np.array(jax.random.normal(k_vel,   (T, 3)) * params.sensory.sigma_vel)

    # OU process for pos — must match simulator exactly
    alpha_ou = float(jnp.exp(-DT / params.sensory.tau_pos_drift))
    ou_drive = float(jnp.sqrt(1.0 - alpha_ou ** 2) * params.sensory.sigma_pos)
    white    = np.array(jax.random.normal(k_pos, (T, 3)))
    pos      = np.zeros((T, 3))
    for i in range(1, T):
        pos[i] = alpha_ou * pos[i - 1] + ou_drive * white[i]

    return dict(canal=noise_canal, pos=pos, vel=noise_vel)


# ── Demo ───────────────────────────────────────────────────────────────────────

def demo_fixation():
    """4-column comparison: noiseless / canal / retinal pos / retinal vel noise."""
    # (sigma_canal, sigma_pos, sigma_vel, title, noise_key, noise_label)
    conditions = [
        (0.0, 0.0, 0.0, 'Noiseless',               None,  ''),
        (3.0, 0.0, 0.0, 'Canal  σ = 3 deg/s',      'canal', 'Canal noise (deg/s)'),
        (0.0, 0.3, 0.0, 'Retinal pos  σ = 0.3 deg', 'pos',  'Pos drift (deg)'),
        (0.0, 0.0, 5.0, 'Retinal vel  σ = 5 deg/s', 'vel',  'Vel noise (deg/s)'),
        (3.0, 0.3, 5.0, 'All combined',              'all',  'All noise signals'),
    ]
    seeds = [7, 7, 7, 7, 7]

    t    = _make_t()
    T    = len(t)
    t_np = np.array(t)

    pt3       = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (T, 1))
    max_steps = int(TEND / DT) + 2000

    n_rows, n_cols = 4, len(conditions)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 2.5 * n_rows),
                             sharex=True)
    fig.suptitle('Fixational eye movements — noise source comparison (head stationary, lit scene, target at 0°)',
                 fontsize=11, fontweight='bold')

    # ── Run all conditions, plot rows 0–2 ─────────────────────────────────────
    for ci, (sigma_c, sigma_p, sigma_v, title, noise_key, noise_label) in enumerate(conditions):
        params = with_sensory(PARAMS_DEFAULT, sigma_canal=sigma_c, sigma_pos=sigma_p, sigma_vel=sigma_v)
        key    = jax.random.PRNGKey(seeds[ci])

        states = simulate(
            params, t,
            p_target_array       = pt3,
            scene_present_array  = jnp.ones(T),
            target_present_array = jnp.ones(T),
            max_steps            = max_steps,
            return_states        = True,
            key                  = key,
        )
        s     = _extract(states, params, t_np)
        noise = _gen_noise(params, T, key)

        axes[0, ci].set_title(title, fontsize=9, fontweight='bold')

        # Row 0: eye position
        axes[0, ci].axhline(0, color=_C['target'], lw=1.0, ls=':', label='target (0°)')
        axes[0, ci].plot(t_np, s['eye_pos'][:, 0], color=_C['eye'], lw=0.8, label='eye yaw')
        _ax_fmt(axes[0, ci], 'Eye position (deg)')
        if ci == 0:
            axes[0, ci].legend(fontsize=7, loc='upper right')

        # Row 1: eye velocity
        axes[1, ci].plot(t_np, s['eye_vel'][:, 0], color=_C['vel'], lw=0.7)
        _ax_fmt(axes[1, ci], 'Eye velocity (deg/s)')

        # Row 2: saccade burst
        axes[2, ci].plot(t_np, s['u_burst'][:, 0], color=_C['burst'], lw=0.8)
        _ax_fmt(axes[2, ci], 'Burst  u_burst (deg/s)')

        # Row 3: noise signal(s) — yaw component of each active source
        ax3 = axes[3, ci]
        if noise_key is None:
            ax3.plot(t_np, np.zeros(T), color='#9970ab', lw=0.7)
        elif noise_key == 'all':
            ax3.plot(t_np, noise['canal'][:, 0], color='#555555', lw=0.6, label='canal (deg/s)')
            ax3.plot(t_np, noise['pos'][:, 0],   color='#2166ac', lw=0.8, label='pos (deg)')
            ax3.plot(t_np, noise['vel'][:, 0],   color='#1a9850', lw=0.6, label='vel (deg/s)')
            ax3.legend(fontsize=6, loc='upper right')
        elif noise_key == 'canal':
            ax3.plot(t_np, noise['canal'][:, 0], color='#555555', lw=0.7)
        else:
            ax3.plot(t_np, noise[noise_key][:, 0], color='#9970ab', lw=0.7)
        _ax_fmt(ax3, noise_label or 'Noise signal')
        ax3.set_xlabel('Time (s)', fontsize=8)

    # ── Equalize y-axes row by row (rows 0–2 share limits; row 3 independent) ──
    _equalize_row(axes[0], min_span=0.5)
    _equalize_row(axes[1], min_span=5.0)
    _equalize_row(axes[2], min_span=5.0)
    # Row 3: equalize only within same-unit groups (canal+vel share deg/s; pos is deg)
    # Simplest: let each column autoscale independently but enforce a minimum span
    for ax in axes[3]:
        lo, hi = ax.get_ylim()
        if hi - lo < 1.0:
            mid = (lo + hi) / 2
            ax.set_ylim(mid - 0.5, mid + 0.5)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fixation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    print(f'  Saved {path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=== Fixation Demo ===')
    demo_fixation()
    print(f'\nDone. Plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
