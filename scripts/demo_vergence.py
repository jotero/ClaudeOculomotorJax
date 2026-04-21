"""Vergence demos.

Figures produced
────────────────
    vergence_midline.png  — near/far steps on the midline: convergence / divergence cascade
    vergence_offaxis.png  — off-axis sequence requiring saccades + vergence simultaneously

Usage
-----
    python -X utf8 scripts/demo_vergence.py
    python -X utf8 scripts/demo_vergence.py --show
"""

import os, sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
SHOW = '--show' in sys.argv
if not SHOW:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, simulate,
    _IDX_VERG,
)
from oculomotor.analysis import extract_burst

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

_C = {
    'L':      '#2166ac',   # left eye  — blue
    'R':      '#d6604d',   # right eye — red
    'target': '#fdae61',   # target    — orange
    'verg':   '#4dac26',   # vergence  — green
    'vers':   '#762a83',   # version   — purple
    'depth':  '#8c510a',   # depth     — brown
}

PARAMS = PARAMS_DEFAULT


# ── Target-array helpers ────────────────────────────────────────────────────────

def _pt3_sequence(t_np, segs):
    """Build (T, 3) target array [x_m, y_m, z_m] from list of (t_start, x, y, z)."""
    T  = len(t_np)
    pt = np.zeros((T, 3))
    pt[:, 0] = segs[0][1]
    pt[:, 1] = segs[0][2]
    pt[:, 2] = segs[0][3]
    for t_start, x, y, z in segs:
        mask = t_np >= t_start
        pt[mask, 0] = x
        pt[mask, 1] = y
        pt[mask, 2] = z
    return jnp.array(pt, dtype=jnp.float32)


def _seg_arrays(t_np, segs):
    """Return (depth_arr, yaw_demand_deg) from segment list."""
    depth = np.full(len(t_np), segs[0][3])
    yaw   = np.zeros(len(t_np))
    for t_start, x, y, z in segs:
        m = t_np >= t_start
        depth[m] = z
        yaw[m]   = np.degrees(np.arctan2(x, z))
    return depth, yaw


def _verg_demand(depth_arr, ipd=0.063):
    """Theoretical vergence demand (deg) = 2·arctan(ipd / 2z)."""
    return np.degrees(2.0 * np.arctan2(ipd / 2.0, depth_arr))


def _vline_segs(axes, segs):
    for t_start, *_ in segs[1:]:
        for ax in np.asarray(axes).flat:
            ax.axvline(t_start, color='gray', lw=0.6, ls='--', alpha=0.5)


# ── Figure 1: midline near / far alternation ────────────────────────────────────

def demo_midline():
    """Pure vergence: near/far steps on the midline, no version change."""
    dt    = 0.001
    T_end = 12.0
    t     = jnp.arange(0.0, T_end, dt)
    T     = len(t)
    t_np  = np.array(t)

    FAR, NEAR = 3.0, 0.4

    segs = [
        (0.0,  0.0, 0.0, FAR),   # baseline — far
        (0.5,  0.0, 0.0, NEAR),  # converge
        (4.0,  0.0, 0.0, FAR),   # diverge
        (7.5,  0.0, 0.0, NEAR),  # converge again
        (11.0, 0.0, 0.0, FAR),   # diverge
    ]

    pt3                = _pt3_sequence(t_np, segs)
    depth_arr, _       = _seg_arrays(t_np, segs)
    verg_demand        = _verg_demand(depth_arr)

    states = simulate(PARAMS, t,
                      p_target_array=pt3,
                      scene_present_array=jnp.ones(T),
                      return_states=True,
                      key=jax.random.PRNGKey(0))

    eye_L   = np.array(states.plant[:, :3])
    eye_R   = np.array(states.plant[:, 3:6])
    x_verg  = np.array(states.brain[:, _IDX_VERG])

    verg_angle = eye_L[:, 0] - eye_R[:, 0]   # positive = converging
    version    = (eye_L[:, 0] + eye_R[:, 0]) / 2.0

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # ── Target depth
    ax = axes[0]
    ax.plot(t_np, depth_arr, color=_C['depth'], lw=1.5)
    ax.set_ylabel('Depth (m)')
    ax.set_title('Target depth — near / far steps on midline')
    ax.set_ylim(0, 3.8)
    ax.grid(True, alpha=0.2)

    # ── Vergence demand vs response
    ax = axes[1]
    ax.plot(t_np, verg_demand,  color=_C['target'], lw=1.4, ls='--', label='demand')
    ax.plot(t_np, verg_angle,   color=_C['verg'],   lw=1.5,          label='L − R (actual)')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('Vergence (deg)')
    ax.set_title('Vergence demand vs. response')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Left / right eye yaw (should be mirror-symmetric, close to 0)
    ax = axes[2]
    ax.plot(t_np, eye_L[:, 0], color=_C['L'], lw=1.4, label='L eye')
    ax.plot(t_np, eye_R[:, 0], color=_C['R'], lw=1.4, label='R eye')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('Eye yaw (deg)')
    ax.set_title('Left / right eye yaw — midline: symmetric divergence / convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Vergence integrator state
    ax = axes[3]
    ax.plot(t_np, x_verg[:, 0], color=_C['verg'], lw=1.5, label='x_verg yaw')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('x_verg (deg)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Vergence integrator state')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    _vline_segs(axes, segs)
    fig.suptitle('Midline Vergence — Near / Far Alternation', fontsize=12, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'vergence_midline.png')
    fig.savefig(path, dpi=150)
    plt.show() if SHOW else plt.close(fig)
    print(f'  Saved {path}')


# ── Figure 2: off-axis sequence — saccades + vergence ──────────────────────────

def demo_offaxis():
    """Off-axis targets: each step requires a conjugate saccade (version) AND vergence."""
    dt    = 0.001
    T_end = 12.0
    t     = jnp.arange(0.0, T_end, dt)
    T     = len(t)
    t_np  = np.array(t)

    FAR, NEAR = 3.0, 0.4

    # Lateral offsets at 0.4 m depth so angular demand is ~15° and ~20°
    x_r15 =  NEAR * np.tan(np.radians(15.0))
    x_l20 = -NEAR * np.tan(np.radians(20.0))
    x_r10 =  FAR  * np.tan(np.radians(10.0))

    segs = [
        (0.0,  0.0,   0.0, FAR),    # centre, far        — baseline
        (0.5,  x_r15, 0.0, NEAR),   # right 15°, near    — saccade right + converge
        (4.0,  x_l20, 0.0, NEAR),   # left  20°, near    — saccade left, stay converged
        (7.0,  x_r10, 0.0, FAR),    # right 10°, far     — saccade right + diverge
        (10.5, 0.0,   0.0, FAR),    # centre, far        — return
    ]

    pt3                   = _pt3_sequence(t_np, segs)
    depth_arr, yaw_demand = _seg_arrays(t_np, segs)
    verg_demand           = _verg_demand(depth_arr)

    states = simulate(PARAMS, t,
                      p_target_array=pt3,
                      scene_present_array=jnp.ones(T),
                      return_states=True,
                      key=jax.random.PRNGKey(0))

    eye_L   = np.array(states.plant[:, :3])
    eye_R   = np.array(states.plant[:, 3:6])
    u_burst = np.array(extract_burst(states, PARAMS))

    verg_angle = eye_L[:, 0] - eye_R[:, 0]
    version    = (eye_L[:, 0] + eye_R[:, 0]) / 2.0

    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)

    # ── Target yaw demand
    ax = axes[0]
    ax.plot(t_np, yaw_demand, color=_C['target'], lw=1.5, label='target yaw')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('Yaw (deg)')
    ax.set_title('Target lateral position')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Target depth
    ax = axes[1]
    ax.plot(t_np, depth_arr,  color=_C['depth'],  lw=1.5, label='depth (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Target depth')
    ax.set_ylim(0, 3.8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Version (conjugate component — saccades)
    ax = axes[2]
    ax.plot(t_np, yaw_demand, color=_C['target'], lw=1.2, ls='--', label='target yaw', alpha=0.7)
    ax.plot(t_np, version,    color=_C['vers'],   lw=1.5,           label='version (L+R)/2')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('Yaw (deg)')
    ax.set_title('Version — conjugate saccades track target azimuth')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Vergence angle
    ax = axes[3]
    ax.plot(t_np, verg_demand, color=_C['target'], lw=1.2, ls='--', label='demand', alpha=0.7)
    ax.plot(t_np, verg_angle,  color=_C['verg'],   lw=1.5,           label='L − R (actual)')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('Vergence (deg)')
    ax.set_title('Vergence angle — diverges / converges with depth')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Left and right eye yaw separately
    ax = axes[4]
    ax.plot(t_np, eye_L[:, 0], color=_C['L'],      lw=1.4, label='L eye yaw')
    ax.plot(t_np, eye_R[:, 0], color=_C['R'],      lw=1.4, label='R eye yaw')
    ax.plot(t_np, yaw_demand,  color=_C['target'], lw=1.0, ls='--', alpha=0.5, label='target yaw')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('Eye yaw (deg)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Per-eye yaw — saccade + vergence dissociation visible')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    _vline_segs(axes, segs)
    fig.suptitle('Off-Axis Vergence with Saccades', fontsize=12, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'vergence_offaxis.png')
    fig.savefig(path, dpi=150)
    plt.show() if SHOW else plt.close(fig)
    print(f'  Saved {path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=== Vergence Demo ===')
    print('\n1. Midline near/far alternation (JIT compile on first call)...')
    demo_midline()
    print('\n2. Off-axis with saccades...')
    demo_offaxis()
    print(f'\nDone. Plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
