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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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


def _ax_fmt(ax, ylabel, ylim=None):
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.axhline(0, color=_C['zero'], lw=0.5, ls='--')
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=7)
    if ylim is not None:
        ax.set_ylim(ylim)


# ── Demo ───────────────────────────────────────────────────────────────────────

def demo_fixation():
    """3-column comparison: noiseless / canal noise / retinal position noise."""
    # (sigma_canal, sigma_pos)
    conditions = [
        (0.0,  0.0,  'Noiseless'),
        (3.0,  0.0,  'Canal noise  σ = 3 deg/s'),
        (0.0,  0.3,  'Retinal pos noise  σ = 0.3 deg'),
    ]
    seeds = [42, 42, 42]

    t    = _make_t()
    T    = len(t)
    t_np = np.array(t)

    # Stationary target straight ahead — never moves
    pt3 = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (T, 1))   # [0, 0, 1] m

    n_rows, n_cols = 3, len(conditions)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 2.8 * n_rows),
                             sharex=True)
    fig.suptitle('Fixational eye movements — canal vs retinal position noise (head stationary, lit scene, target at 0°)',
                 fontsize=11, fontweight='bold')

    max_steps = int(TEND / DT) + 2000

    for ci, ((sigma_c, sigma_p, title), seed) in enumerate(zip(conditions, seeds)):
        params = with_sensory(PARAMS_DEFAULT, sigma_canal=sigma_c, sigma_pos=sigma_p)
        key    = jax.random.PRNGKey(seed)

        states = simulate(
            params, t,
            p_target_array       = pt3,
            scene_present_array  = jnp.ones(T),
            target_present_array = jnp.ones(T),
            max_steps            = max_steps,
            return_states        = True,
            key                  = key,
        )
        s = _extract(states, params, t_np)

        axes[0, ci].set_title(title, fontsize=9, fontweight='bold')

        # ── Row 0: eye position + target ──────────────────────────────────────
        axes[0, ci].axhline(0, color=_C['target'], lw=1.0, ls=':', label='target (0°)')
        axes[0, ci].plot(t_np, s['eye_pos'][:, 0], color=_C['eye'], lw=0.8, label='eye yaw')
        _ax_fmt(axes[0, ci], 'Eye position (deg)')
        if ci == 0:
            axes[0, ci].legend(fontsize=7, loc='upper right')
        # Enforce minimum visible range so noiseless panel doesn't autoscale to noise
        lo, hi = axes[0, ci].get_ylim()
        if hi - lo < 0.5:
            mid = (lo + hi) / 2
            axes[0, ci].set_ylim(mid - 0.25, mid + 0.25)

        # ── Row 1: eye velocity ───────────────────────────────────────────────
        axes[1, ci].plot(t_np, s['eye_vel'][:, 0], color=_C['vel'], lw=0.7)
        _ax_fmt(axes[1, ci], 'Eye velocity (deg/s)')
        lo, hi = axes[1, ci].get_ylim()
        if hi - lo < 5.0:
            mid = (lo + hi) / 2
            axes[1, ci].set_ylim(mid - 2.5, mid + 2.5)

        # ── Row 2: saccade burst ──────────────────────────────────────────────
        axes[2, ci].plot(t_np, s['u_burst'][:, 0], color=_C['burst'], lw=0.8)
        _ax_fmt(axes[2, ci], 'Burst  u_burst (deg/s)')
        axes[2, ci].set_xlabel('Time (s)', fontsize=8)
        lo, hi = axes[2, ci].get_ylim()
        if hi - lo < 5.0:
            mid = (lo + hi) / 2
            axes[2, ci].set_ylim(mid - 2.5, mid + 2.5)

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
