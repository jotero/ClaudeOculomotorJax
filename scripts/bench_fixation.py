"""Fixation benchmarks — noise source comparison + drift quiver across visual field.

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
from oculomotor.sim import kinematics as km
from oculomotor.analysis import ax_fmt, extract_burst, extract_spv_states

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
    states = simulate(params, t,
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
    sp_def = PARAMS_DEFAULT.sensory
    sc_d, sp_d, sv_d = float(sp_def.sigma_canal), float(sp_def.sigma_pos), float(sp_def.sigma_vel)
    tau_canal = float(sp_def.tau_canal_drift)
    tau_pos   = float(sp_def.tau_pos_drift)
    tau_vel   = float(sp_def.tau_vel_drift)

    # Each row: (sigma_canal, sigma_pos, sigma_vel, key). Default values pulled from
    # SensoryParams; per-row only the relevant σ is enabled, so each panel shows
    # the contribution of one noise source in isolation.
    sweeps = [
        (0.0,  0.0,  0.0,  'none'),
        (sc_d, 0.0,  0.0,  'canal'),
        (0.0,  sp_d, 0.0,  'pos'),
        (0.0,  0.0,  sv_d, 'vel'),
        (sc_d, sp_d, sv_d, 'all'),
    ]

    def _title(sc, sp, sv):
        if sc == 0 and sp == 0 and sv == 0:
            return 'Noiseless'
        parts = []
        if sc:  parts.append(f'canal σ={sc:g} deg/s, τ={tau_canal:g} s')
        if sp:  parts.append(f'pos σ={sp:g} deg, τ={tau_pos:g} s')
        if sv:  parts.append(f'vel σ={sv:g} deg/s, τ={tau_vel:g} s')
        return '  +  '.join(parts) if len(parts) > 1 else parts[0]

    conditions = [(sc, sp, sv, _title(sc, sp, sv), key) for (sc, sp, sv, key) in sweeps]
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
    path, rp = utils.save_fig(fig, 'fixation_noise_comparison', show=show, params=PARAMS_DEFAULT,
                              conditions='Lit, fixation on midline target — each panel sweeps a different sensory noise σ (canal/pos/vel)')
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


def _drift_quiver(show):
    """Mean slow-phase drift velocity at multiple fixation positions across the
    visual field. 17 positions: origin + 8 directions × 2 eccentricities (5°, 10°).
    Each fixation runs for 10 s with all default params (realistic noise on).
    The quiver arrows show mean (vx, vy) of the SPV at the fixation location.
    """
    DEPTH    = 1.0     # m, screen distance
    DURATION = 10.0    # s
    DROP_S   = 1.0     # discard the first 1 s (initial transients) when averaging
    NDIR     = 8       # cardinal + 45° = 8 directions

    # Build the position grid (degrees on the visual field)
    angs = np.linspace(0, 360, NDIR, endpoint=False)
    targets_deg = [(0.0, 0.0)]                   # origin: only one entry
    for ecc in [5.0, 10.0]:
        for a in angs:
            x = ecc * np.cos(np.radians(a))
            y = ecc * np.sin(np.radians(a))
            targets_deg.append((x, y))

    t  = jnp.arange(0.0, DURATION, DT)
    T  = len(t)
    drop_n = int(DROP_S / DT)

    drifts = []  # list of (px_deg, py_deg, vx_mean, vy_mean)
    for k, (px_deg, py_deg) in enumerate(targets_deg):
        # Convert visual-field angle (deg) to a Cartesian world point on a screen
        # at DEPTH m. Small-angle approximation OK at ≤10°: x = D·tan(yaw), y = D·tan(pitch).
        wx = DEPTH * np.tan(np.radians(px_deg))
        wy = DEPTH * np.tan(np.radians(py_deg))
        lin_pos = np.tile(np.array([wx, wy, DEPTH]), (T, 1))
        target  = km.build_target(t, lin_pos=lin_pos)
        states  = simulate(PARAMS_DEFAULT, t,
                           target=target,
                           scene_present_array=jnp.ones(T),
                           target_present_array=jnp.ones(T),
                           max_steps=int(DURATION / DT) + 2000,
                           return_states=True,
                           key=jax.random.PRNGKey(100 + k))
        # SPV (deg/s) per axis using OPN gate; first axis = yaw (H), second = pitch (V)
        spv = extract_spv_states(states, np.array(t), margin_s=0.05, eye='left')
        spv_h = spv[drop_n:, 0]
        spv_v = spv[drop_n:, 1]
        drifts.append((px_deg, py_deg, float(np.nanmean(spv_h)), float(np.nanmean(spv_v))))

    drifts = np.array(drifts)
    px = drifts[:, 0]
    py = drifts[:, 1]
    vx = drifts[:, 2]
    vy = drifts[:, 3]
    speed = np.hypot(vx, vy)

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Reference circles at 5° and 10°
    for r in [5.0, 10.0]:
        circle = plt.Circle((0, 0), r, fill=False, ls=':', lw=0.7, color='#bbbbbb')
        ax.add_patch(circle)
        ax.text(r * np.cos(np.radians(45)) + 0.3, r * np.sin(np.radians(45)) + 0.3,
                f'{r:.0f}°', color='#888888', fontsize=8)

    # Fixation points
    ax.plot(px, py, 'o', color='black', ms=4, zorder=4)

    # Quiver — autoscale so the largest arrow is ~3 deg in plot space.
    max_speed = max(float(np.nanmax(speed)), 1e-6)
    arrow_max_plot_deg = 3.0
    QUIVER_SCALE = max_speed / arrow_max_plot_deg   # data deg/s per plot deg
    q = ax.quiver(px, py, vx, vy, speed,
                  cmap='viridis', angles='xy', scale_units='xy',
                  scale=QUIVER_SCALE,
                  width=0.005, headwidth=4, headlength=5, zorder=5)
    cbar = fig.colorbar(q, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label('|SPV| (deg/s)', fontsize=9)

    # Reference arrow — round the max speed to a nice value, draw at bottom-right.
    if max_speed >= 0.5:    SCALE_VAL = 0.5
    elif max_speed >= 0.2:  SCALE_VAL = 0.2
    elif max_speed >= 0.1:  SCALE_VAL = 0.1
    else:                   SCALE_VAL = 0.05
    sx, sy = 11.0, -11.5
    ax.quiver([sx], [sy], [SCALE_VAL], [0.0], color='red',
              angles='xy', scale_units='xy', scale=QUIVER_SCALE,
              width=0.005, headwidth=4, headlength=5, zorder=5)
    ax.text(sx + (SCALE_VAL / QUIVER_SCALE) / 2, sy - 0.7,
            f'{SCALE_VAL:g} deg/s', color='red', ha='center', fontsize=9)

    ax.set_xlim(-13, 14)
    ax.set_ylim(-13, 13)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color='#999999', lw=0.5)
    ax.axvline(0, color='#999999', lw=0.5)
    ax.set_xlabel('Horizontal eccentricity (deg)')
    ax.set_ylabel('Vertical eccentricity (deg)')
    ax.set_title(f'Fixation drift quiver — {DURATION:.0f} s per position, screen at {DEPTH:.1f} m\n'
                 f'(arrow = mean slow-phase velocity; first {DROP_S:.0f} s discarded)',
                 fontsize=10)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'fixation_drift_quiver', show=show, params=PARAMS_DEFAULT,
                              conditions=f'Lit, midline+eccentric foveal targets at {DEPTH:.1f} m '
                                         f'(0°, 5°, 10° eccentricity × 8 directions); {DURATION:.0f} s/fixation')
    return utils.fig_meta(
        path, rp,
        title='Fixation Drift Quiver',
        description=f'Mean slow-phase drift at 17 fixation positions over {DURATION:.0f} s each. '
                    'Default params (canal/pos/vel noise on). Origin + 8 directions × 5°/10° eccentricity. '
                    'Arrows show drift direction + magnitude (color = |SPV|, red ref arrow = 1 deg/s).',
        expected='Drift magnitudes typically <1 deg/s at all positions; with default '
                 'noise levels and an OU position drift, traces drift slowly between '
                 'occasional microsaccades.  Direction roughly random / centripetal '
                 'at eccentric positions (slight pull toward primary).',
        citation='Cherici et al. (2012) J Vis 12(6):31; Martinez-Conde & Macknik 2017 Neuron.',
    )


def run(show=False):
    print('\n=== Fixation ===')
    figs = []
    print('  1/2  noise source comparison …')
    figs.append(_noise_comparison(show))
    print('  2/2  drift quiver across visual field …')
    figs.append(_drift_quiver(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
