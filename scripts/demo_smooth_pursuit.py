"""Smooth pursuit / ramp target demo.

Target starts moving at t=0.2 s at a constant velocity.  Four velocities
(1, 2, 5, 10 deg/s) are tested.  Saccades are enabled (same THETA as the
saccade demo) so we can see whether the model produces catch-up saccades,
smooth pursuit, or a mixture.

Figure
------
    smooth_pursuit.png  — 4 rows × 4 velocity columns (same layout as
                          saccade_single.png)

Usage
-----
    python scripts/demo_smooth_pursuit.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import diffrax

from oculomotor.models.ocular_motor_simulator import (
    THETA_DEFAULT, ODE_ocular_motor, _N_TOTAL,
    _IDX_NI, _IDX_P, _IDX_SG, _IDX_VIS,
)
from oculomotor.models import retina
from oculomotor.models import saccade_generator as sg
from oculomotor.models import visual_delay

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

_C = {
    'target': '#d6604d',
    'eye':    '#2166ac',
    'ni':     '#4dac26',
    'error':  '#762a83',
    'burst':  '#f4a582',
    'vel':    '#1a9641',
    'reset':  '#e08214',
}

THETA_SAC = THETA_DEFAULT   # head is still and no scene motion → VOR/OKR inactive naturally

VELOCITIES = [1.0, 2.0, 5.0, 10.0]   # deg/s
T_END   = 3.0
T_JUMP  = 0.2
DT      = 0.001


def _extract_all(theta, t_array, pt3, max_steps=10000):
    T      = len(t_array)
    hv3    = jnp.zeros((T, 3))
    vs3    = jnp.zeros((T, 3))
    hp3    = jnp.zeros((T, 3))
    sg1    = jnp.ones(T, dtype=jnp.float32)

    target_interp     = diffrax.LinearInterpolation(ts=t_array, ys=pt3)
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(ODE_ocular_motor),
        diffrax.Heun(),
        t0=t_array[0], t1=t_array[-1], dt0=DT,
        y0=jnp.zeros(_N_TOTAL),
        args=(theta,
              diffrax.LinearInterpolation(ts=t_array, ys=hv3),
              diffrax.LinearInterpolation(ts=t_array, ys=hp3),
              diffrax.LinearInterpolation(ts=t_array, ys=vs3),
              target_interp,
              diffrax.LinearInterpolation(ts=t_array, ys=sg1)),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=t_array),
        max_steps=max_steps,
    )
    ys = solution.ys

    def _signals_at(x, t):
        x_p           = x[_IDX_P]
        x_ni          = x[_IDX_NI]
        x_vis         = x[_IDX_VIS]
        x_reset_int   = x[_IDX_SG]
        p_t           = target_interp.evaluate(t)
        e_motor       = retina.target_to_angle(p_t) - x_p
        e_pos_delayed = visual_delay.C_pos @ x_vis
        _, u_burst    = sg.step(x_reset_int, e_pos_delayed, theta)
        return {'q_eye': x_p, 'x_ni': x_ni, 'e_motor': e_motor,
                'e_pos_delayed': e_pos_delayed,
                'x_reset_int': x_reset_int, 'u_burst': u_burst}

    raw = jax.vmap(lambda x, t: _signals_at(x, t))(ys, t_array)
    return {k: np.array(v) for k, v in raw.items()}


def demo_smooth_pursuit():
    t     = jnp.arange(0.0, T_END, DT)
    T     = len(t)
    t_np  = np.array(t)

    n_rows, n_cols = 4, len(VELOCITIES)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 2.8 * n_rows),
                             sharex=True)
    fig.suptitle('Ramp Target — Catch-up Saccades (no smooth pursuit pathway)',
                 fontsize=12)

    row_labels = ['Position (deg)',
                  'Error / Copy (deg)',
                  'Burst + Eye Vel (deg/s)',
                  'Copy vs NI (deg)']
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8)

    for ci, vel in enumerate(VELOCITIES):
        # Target: ramp starting at T_JUMP
        target_deg = np.where(t_np >= T_JUMP, vel * (t_np - T_JUMP), 0.0)
        pt3 = jnp.zeros((T, 3))
        pt3 = pt3.at[:, 2].set(1.0)
        pt3 = pt3.at[:, 0].set(jnp.array(np.tan(np.radians(target_deg))))

        sig     = _extract_all(THETA_SAC, t, pt3)
        eye_vel = np.gradient(sig['q_eye'][:, 0], t_np)
        tgt_vel = np.gradient(target_deg, t_np)   # should be ~vel after T_JUMP

        def _vl(ax):
            ax.axvline(T_JUMP, color='gray', lw=0.7, ls='--', alpha=0.5)

        axes[0, ci].set_title(f'{vel:.0f} deg/s', fontsize=10)

        # Row 0: position
        axes[0, ci].plot(t_np, target_deg,          color=_C['target'], lw=1.5, label='target')
        axes[0, ci].plot(t_np, sig['q_eye'][:, 0],  color=_C['eye'],    lw=1.5, label='eye')
        axes[0, ci].plot(t_np, sig['x_ni'][:, 0],   color=_C['ni'],     lw=1.0, ls='--', label='NI cmd')
        _vl(axes[0, ci]); axes[0, ci].axhline(0, color='k', lw=0.4)
        if ci == 0: axes[0, ci].legend(fontsize=7)

        # Row 1: motor error + delayed error + copy integrator
        axes[1, ci].plot(t_np, sig['e_motor'][:, 0],       color=_C['error'],  lw=1.5, label='e_motor')
        axes[1, ci].plot(t_np, sig['e_pos_delayed'][:, 0], color=_C['error'],  lw=1.0, ls='--', label='e_delayed')
        axes[1, ci].plot(t_np, sig['x_reset_int'][:, 0],   color=_C['reset'],  lw=1.5, label='x_reset_int')
        _vl(axes[1, ci]); axes[1, ci].axhline(0, color='k', lw=0.4)
        if ci == 0: axes[1, ci].legend(fontsize=7)

        # Row 2: burst command + eye velocity + target velocity
        axes[2, ci].plot(t_np, sig['u_burst'][:, 0], color=_C['burst'],  lw=1.5, label='u_burst')
        axes[2, ci].plot(t_np, eye_vel,               color=_C['vel'],    lw=1.2, ls='--', label='eye vel')
        axes[2, ci].axhline(vel, color=_C['target'], lw=0.8, ls=':', label=f'tgt vel {vel}')
        _vl(axes[2, ci]); axes[2, ci].axhline(0, color='k', lw=0.4)
        if ci == 0: axes[2, ci].legend(fontsize=7)

        # Row 3: copy integrator vs NI
        axes[3, ci].plot(t_np, sig['x_reset_int'][:, 0], color=_C['reset'], lw=1.5, label='x_reset_int')
        axes[3, ci].plot(t_np, sig['x_ni'][:, 0],         color=_C['ni'],    lw=1.0, ls='--', label='x_ni')
        _vl(axes[3, ci]); axes[3, ci].axhline(0, color='k', lw=0.4)
        if ci == 0: axes[3, ci].legend(fontsize=7)

        axes[n_rows - 1, ci].set_xlabel('Time (s)')
        for r in range(n_rows):
            axes[r, ci].set_xlim(0, T_END)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'smooth_pursuit.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Saved {path}')


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    demo_smooth_pursuit()
