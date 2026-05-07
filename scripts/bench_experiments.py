"""Experimental / exploratory benchmarks.

Currently contains:
    1. Monocular occlusion — binocular fixation maintenance under three viewing conditions.
    2. Fixation drift quiver — slow-phase drift across visual field, noise on/off comparison.

Usage:
    python -X utf8 scripts/bench_experiments.py
    python -X utf8 scripts/bench_experiments.py --show
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
    PARAMS_DEFAULT, SimConfig, simulate, with_brain, with_sensory,
    _IDX_ACC, _IDX_NI_L, _IDX_NI_R, _IDX_VERG, _IDX_VIS,
)
from oculomotor.models.sensory_models.sensory_model import (
    C_vel as _C_vel, C_target_disp as _C_disp, C_defocus as _C_defocus,
)
from oculomotor.sim import kinematics as km
from oculomotor.sim.kinematics import build_target
from oculomotor.analysis import extract_spv_states

SHOW  = '--show' in sys.argv
DT    = 0.001
_THETA_BASE = with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_pos=0.0, sigma_vel=0.0,
                             proximal=2.0)   # 2 D HMD-equivalent proximal cue
_THETA_BASE = with_brain(_THETA_BASE, sigma_acc=0.0)   # noiseless — deterministic


# ── Monocular occlusion ────────────────────────────────────────────────────────

_T_END  = 15.0
_T_FIX  = 2.0       # binocular fixation period before occlusion onset (s)

_CFG = SimConfig(warmup_s=30.0)   # 5 × tau_verg → vergence fully settled

_COND_LABELS = {
    'dark':       'Dark (both lose target)',
    'strobed':    'Strobed (position only)',
    'continuous': 'Continuous (monocular)',
}


def _make_flags(t_np, cond, occ_eye):
    T    = len(t_np)
    ones = np.ones(T,  dtype=np.float32)
    off  = np.where(t_np >= _T_FIX, 0.0, 1.0).astype(np.float32)
    no_strobe = np.zeros(T, dtype=np.float32)

    if cond == 'dark':
        return off, off, no_strobe

    if cond == 'pulsed':
        # 80 ms target ON, 900 ms target OFF (period 980 ms) after T_FIX.
        # Before T_FIX target is continuously on. Implemented via target_present
        # pulsing — no strobe-flag mechanism (target_strobed stays 0).
        T_ON     = 0.050
        T_PERIOD = 0.980
        rel_t  = t_np - _T_FIX
        phase  = np.mod(np.maximum(rel_t, 0.0), T_PERIOD)
        viewing = np.where(rel_t < 0, 1.0, (phase < T_ON).astype(np.float32)).astype(np.float32)
    else:   # 'continuous'
        viewing = ones

    if occ_eye == 'left':
        tL, tR = off, viewing
    else:
        tL, tR = viewing, off

    return tL, tR, no_strobe


def _run_cond(t_np, cond, occ_eye, *, theta_base, dist_m, lens_d, dark_tonic_verg):
    t  = jnp.array(t_np)
    T  = len(t_np)
    pt = jnp.tile(jnp.array([0.0, 0.0, dist_m]), (T, 1))
    tL, tR, ts = _make_flags(t_np, cond, occ_eye)
    lens_arr = jnp.full((T,), lens_d, dtype=jnp.float32)
    # Dark condition: per-experiment override of tonic vergence to test how far
    # the eyes drift when nothing constrains them.
    theta = with_brain(theta_base, tonic_verg=dark_tonic_verg) if cond == 'dark' else theta_base
    return simulate(
        theta, t,
        target                 = build_target(t, lin_pos=pt),
        scene_present_array    = jnp.zeros(T),
        target_present_L_array = jnp.array(tL),
        target_present_R_array = jnp.array(tR),
        target_strobed_array   = jnp.array(ts),
        lens_L_array           = lens_arr,
        lens_R_array           = lens_arr,
        return_states          = True,
        sim_config             = _CFG,
    )


def _occlusion(show, *, save_name, plot_title, dist_m, lens_d, theta_base, dark_tonic_verg,
               description, expected):
    t_np = np.arange(0.0, _T_END, DT, dtype=np.float32)

    # Five columns by viewing-eye + condition.
    #   _run_cond(cond, occ_eye):
    #     occ_eye='left'  → L eye occluded → R eye is the viewing eye
    #     occ_eye='right' → R eye occluded → L eye is the viewing eye
    #     cond='continuous' → solid target  ;  cond='strobed' → flashing target
    #     cond='dark'       → both eyes occluded
    columns = [
        ('R eye viewing — continuous',            'continuous', 'left'),
        ('R eye viewing — 80 ms on / 900 ms off', 'pulsed',     'left'),
        ('L eye viewing — continuous',            'continuous', 'right'),
        ('L eye viewing — 80 ms on / 900 ms off', 'pulsed',     'right'),
        ('Dark (both occluded)',                  'dark',       'left'),
    ]
    sims = [_run_cond(t_np, c, o, theta_base=theta_base, dist_m=dist_m,
                      lens_d=lens_d, dark_tonic_verg=dark_tonic_verg)
            for _, c, o in columns]
    # Recover the per-column visibility flags for the top row of the figure.
    flag_arrays = [_make_flags(t_np, c, o) for _, c, o in columns]

    # Pretty distance label: cm for near targets, m for far ones.
    dist_label = f'{dist_m*100:.0f} cm' if dist_m < 1.0 else f'{dist_m:.0f} m'
    lens_clinical = -lens_d   # model sign opposite to clinical
    lens_label = f'{lens_clinical:+.1f} D'

    N_COLS = len(columns)
    N_ROWS = 12
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(4 * N_COLS, 28), sharex=True)
    fig.suptitle(
        f'{plot_title} — fixation at {dist_label}, {lens_label} lens, '
        f'tonic vergence {theta_base.brain.tonic_verg:+.0f}°  '
        f'(dark: {dark_tonic_verg:+.0f}°)\n'
        f'Vertical line = occlusion onset (t = {_T_FIX:.0f} s)',
        fontsize=10,
    )

    row_labels = [
        'Target visibility L / R',
        'Eye yaw L+R (deg)',
        'Vergence (eye_L − eye_R) (deg)',
        'Slow-phase velocity L+R (deg/s)',
        'Version slow-phase velocity (deg/s)',
        'Vergence slow-phase velocity (deg/s)',
        'Accommodation (D)',
        'ACA drive (deg)',
        'CAC drive (D)',
        'Cyclopean target motion[H] (deg/s)',
        'Cyclopean target disparity[H] (deg)',
        'Cyclopean defocus (D)',
    ]
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8.5)

    C_L = utils.C['eye']
    C_R = utils.C['target']

    for ci, ((title, _cond, _occ), st) in enumerate(zip(columns, sims)):
        axes[0, ci].set_title(title, fontsize=9)
        axes[N_ROWS - 1, ci].set_xlabel('Time (s)', fontsize=8)

        eye_L    = np.array(st.plant[:, 0])
        eye_R    = np.array(st.plant[:, 3])
        spv_L_yaw    = extract_spv_states(st, np.array(t_np), eye='left')[:, 0]
        spv_R_yaw    = extract_spv_states(st, np.array(t_np), eye='right')[:, 0]
        vers_spv_yaw = extract_spv_states(st, np.array(t_np), eye='version')[:, 0]
        verg_spv_yaw = spv_L_yaw - spv_R_yaw   # vergence slow-phase velocity
        # Accommodation: lens optical power (D) + phasic/tonic neural states
        acc_lens  = np.array(st.acc_plant[:, 0])                     # lens (D)
        acc_brain = np.array(st.brain[:, _IDX_ACC])                  # (T, 2) — fast, slow
        acc_fast  = acc_brain[:, 0]
        acc_slow  = acc_brain[:, 1]
        # ACA drive in deg: AC_A (pd/D) × 0.5729 (deg/pd) × x_acc_fast (D)
        aca_drive = float(theta_base.brain.AC_A) * 0.5729 * acc_fast
        # CAC drive in D: CA_C (D/pd) × (x_verg_fast[H] / 0.5729 deg/pd)
        # x_verg layout = [x_fast(3) | x_slow(3) | x_copy(3)]; [H] component is index 0.
        x_verg_v_h = np.array(st.brain[:, _IDX_VERG])[:, 0]
        cac_drive  = float(theta_base.brain.CA_C) * (x_verg_v_h / 0.5729)
        # Cyclopean cascade outputs (what the brain receives after the visual delay)
        x_vis_np  = np.array(st.sensory[:, _IDX_VIS])
        cyc_motion = (x_vis_np @ np.array(_C_vel).T)[:, 0]      # H component (deg/s)
        cyc_disp   = (x_vis_np @ np.array(_C_disp).T)[:, 0]     # H component (deg)
        cyc_defocus = (x_vis_np @ np.array(_C_defocus).T)[:, 0] # scalar (D)

        # Row 0: target visibility per eye as colored patches (L on top, R on bottom)
        tL_flag, tR_flag, _ts = flag_arrays[ci]
        ax_vis = axes[0, ci]
        ax_vis.fill_between(t_np, 0.55, 1.0, where=tL_flag > 0.5,
                            color=C_L, alpha=0.7, step='post', label='L eye target on')
        ax_vis.fill_between(t_np, 0.0, 0.45, where=tR_flag > 0.5,
                            color=C_R, alpha=0.7, step='post', label='R eye target on')
        ax_vis.axhline(0.5, color='gray', lw=0.5, alpha=0.5)
        ax_vis.text(t_np[0] + 0.05, 0.78, 'L', fontsize=7, va='center')
        ax_vis.text(t_np[0] + 0.05, 0.22, 'R', fontsize=7, va='center')
        ax_vis.set_ylim(-0.1, 1.1)
        ax_vis.set_yticks([])

        # Row 1: per-eye position
        axes[1, ci].plot(t_np, eye_L, color=C_L, lw=1.3, label='L eye')
        axes[1, ci].plot(t_np, eye_R, color=C_R, lw=1.3, label='R eye')

        # Row 2: vergence angle (eye_L − eye_R)
        verg_angle = eye_L - eye_R
        axes[2, ci].plot(t_np, verg_angle, color=utils.C.get('vs', '#8B4513'),
                         lw=1.2, label='vergence (L−R)')
        axes[2, ci].axhline(0, color='gray', lw=0.5, alpha=0.5)
        # tonic_verg is per-column: dark uses dark_tonic_verg, others use theta_base's
        col_tonic = dark_tonic_verg if _cond == 'dark' else float(theta_base.brain.tonic_verg)
        axes[2, ci].axhline(col_tonic, color='red', lw=0.6, ls=':',
                             alpha=0.5, label=f'tonic={col_tonic:.1f}°')

        # Row 3: per-eye slow-phase velocity
        axes[3, ci].plot(t_np, spv_L_yaw, color=C_L, lw=1.0, label='L eye SPV')
        axes[3, ci].plot(t_np, spv_R_yaw, color=C_R, lw=1.0, label='R eye SPV')

        # Row 4: version slow-phase velocity
        axes[4, ci].plot(t_np, vers_spv_yaw, color=utils.C.get('ni', '#4dac26'),
                         lw=1.0, label='Version SPV')

        # Row 5: vergence slow-phase velocity (saccades excluded)
        axes[5, ci].plot(t_np, verg_spv_yaw, color=utils.C.get('vs', '#8B4513'),
                         lw=1.0, label='Vergence SPV')

        # Row 6: accommodation — lens (D) + phasic/tonic neural states
        axes[6, ci].plot(t_np, acc_lens, color='#8B4513', lw=1.3, label='lens (D)')
        axes[6, ci].plot(t_np, acc_fast, color='#2166ac', lw=1.0, ls='--',
                         label='x_acc_fast')
        axes[6, ci].plot(t_np, acc_slow, color='#762a83', lw=1.0, ls=':',
                         label='x_acc_slow')

        # Row 7: ACA drive (deg) — AC_A · 0.5729 · x_acc_fast
        axes[7, ci].plot(t_np, aca_drive, color='#cc6622', lw=1.2,
                          label=f'ACA drive (AC_A={theta_base.brain.AC_A:.1f})')
        axes[7, ci].axhline(0, color='gray', lw=0.6, ls=':', alpha=0.5)

        # Row 8: CAC drive (D) — CA_C × x_verg_fast[H] / 0.5729 deg/pd
        axes[8, ci].plot(t_np, cac_drive, color='#1f78b4', lw=1.2,
                          label=f'CAC drive (CA_C={theta_base.brain.CA_C:.2f})')
        axes[8, ci].axhline(0, color='gray', lw=0.6, ls=':', alpha=0.5)

        # Row 9: cyclopean target motion (post-cascade, what the brain sees)
        axes[9, ci].plot(t_np, cyc_motion, color='#1f77b4', lw=1.0,
                          label='cyc target motion[H]')
        axes[9, ci].axhline(0, color='gray', lw=0.5, alpha=0.5)

        # Row 10: cyclopean disparity (post-cascade, what vergence sees)
        axes[10, ci].plot(t_np, cyc_disp, color='#762a83', lw=1.0,
                           label='cyc disparity[H]')
        axes[10, ci].axhline(0, color='gray', lw=0.5, alpha=0.5)

        # Row 11: cyclopean defocus (post-cascade, what accommodation sees)
        axes[11, ci].plot(t_np, cyc_defocus, color='#cc6622', lw=1.0,
                           label='cyc defocus')
        axes[11, ci].axhline(0, color='gray', lw=0.5, alpha=0.5)

        for row in range(N_ROWS):
            ax = axes[row, ci]
            ax.axvline(_T_FIX, color='gray', lw=0.8, ls='--', alpha=0.5)
            ax.grid(True, alpha=0.15)
            if ci == 0:
                ax.legend(fontsize=6)

    # Sync y-limits across columns per row (skip row 0 — visibility patches use a fixed range)
    for row in range(1, N_ROWS):
        lo_row =  np.inf
        hi_row = -np.inf
        for ci in range(N_COLS):
            ylo, yhi = axes[row, ci].get_ylim()
            lo_row = min(lo_row, ylo)
            hi_row = max(hi_row, yhi)
        span = max(hi_row - lo_row, 3.0)
        mid  = 0.5 * (lo_row + hi_row)
        for ci in range(N_COLS):
            axes[row, ci].set_ylim(mid - span / 2, mid + span / 2)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, save_name, show=show,
                              figs_dir=utils.EXPT_FIGS_DIR, base_dir=utils.EXPT_DIR)
    meta = utils.fig_meta(
        path, rp,
        title=plot_title,
        description=description,
        expected=expected,
        citation='Typical clinical dissociated nystagmus / monocular occlusion paradigm',
    )
    # Return the per-column sims alongside the meta so the summary plot can
    # reuse them without re-running the simulations.
    return meta, list(zip(columns, sims)), t_np


# Two flavours: convergent vs divergent initial fixation state.
# "From convergence" = eyes start converged on the near target, drift toward
#   divergent tonic during occlusion.
# "From divergence" = eyes start near-parallel on the far target, drift toward
#   convergent tonic during occlusion.

def _occlusion_from_convergence(show):
    theta = with_brain(_THETA_BASE, tonic_verg=10.0, AC_A=2.0)
    return _occlusion(
        show,
        save_name='occlusion_from_convergence',
        plot_title='Monocular occlusion — from convergence',
        dist_m=0.25,        # 25 cm near target — eyes converged ~14.6°
        lens_d=-3.0,        # +3 D plus lens (model sign convention is opposite)
        theta_base=theta,
        dark_tonic_verg=10.0,
        description='Near binocular fixation (25 cm) with +3 D plus lens; resting tonic '
                    'vergence +15° (convergent), dark drift to +25°.',
        expected='Eyes converged on near target during binocular phase; tonic is also '
                 'convergent so drift during occlusion is small; dark → drift to ~+25°.',
    )


def _occlusion_from_divergence(show):
    theta = with_brain(_THETA_BASE, tonic_verg=10.0, AC_A=2.0)
    return _occlusion(
        show,
        save_name='occlusion_from_divergence',
        plot_title='Monocular occlusion — from divergence',
        dist_m=10.0,        # 10 m far target — eyes near-parallel
        lens_d=1.0,         # -1 D minus lens (model sign convention is opposite)
        theta_base=theta,
        dark_tonic_verg=10.0,
        description='Far binocular fixation (10 m) with -1 D minus lens; tonic vergence '
                    'fixed at +20° in all conditions (no light/dark shift).',
        expected='Eyes near-parallel during binocular phase; after occlusion all three '
                 'conditions drift toward the same tonic +20°. Difference between '
                 'continuous/flashing/dark is purely from how strongly visual input '
                 'pulls the slow integrator off its setpoint.',
    )


# ── Fixation drift quiver ──────────────────────────────────────────────────────


def _drift_quiver(show):
    """Mean slow-phase drift velocity at multiple fixation positions across the
    visual field. Two side-by-side panels: noise on (default) vs noise off.
    17 positions: origin + 8 directions × 2 eccentricities (5°, 10°).
    """
    DEPTH    = 1.0     # m, screen distance
    DURATION = 5.0     # s
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

    # Two conditions: noise on (default) vs noise off (all sigmas → 0)
    params_noise_on  = PARAMS_DEFAULT
    params_noise_off = with_brain(
        with_sensory(PARAMS_DEFAULT,
                     sigma_canal=0.0, sigma_slip=0.0,
                     sigma_pos=0.0, sigma_vel=0.0),
        sigma_acc=0.0,
    )

    def _run_condition(params, strobed=False):
        drifts = []
        strobe_arr = jnp.ones(T) if strobed else jnp.zeros(T)
        for k, (px_deg, py_deg) in enumerate(targets_deg):
            wx = DEPTH * np.tan(np.radians(px_deg))
            wy = DEPTH * np.tan(np.radians(py_deg))
            lin_pos = np.tile(np.array([wx, wy, DEPTH]), (T, 1))
            target  = km.build_target(t, lin_pos=lin_pos)
            states  = simulate(params, t,
                               target=target,
                               scene_present_array=jnp.ones(T),
                               target_present_array=jnp.ones(T),
                               target_strobed_array=strobe_arr,
                               max_steps=int(DURATION / DT) + 2000,
                               return_states=True,
                               key=jax.random.PRNGKey(100 + k))
            spv = extract_spv_states(states, np.array(t), margin_s=0.05, eye='left')
            spv_h = spv[drop_n:, 0]
            spv_v = spv[drop_n:, 1]
            drifts.append((px_deg, py_deg,
                           float(np.nanmean(spv_h)), float(np.nanmean(spv_v))))
        return np.array(drifts)

    drifts_on  = _run_condition(params_noise_on, strobed=True)
    drifts_off = _run_condition(params_noise_off, strobed=False)

    # Common color scale across both panels for fair comparison
    speed_on  = np.hypot(drifts_on[:, 2],  drifts_on[:, 3])
    speed_off = np.hypot(drifts_off[:, 2], drifts_off[:, 3])
    max_speed = max(float(np.nanmax(speed_on)),
                    float(np.nanmax(speed_off)), 1e-6)
    arrow_max_plot_deg = 3.0
    QUIVER_SCALE = max_speed / arrow_max_plot_deg

    if max_speed >= 0.5:    SCALE_VAL = 0.5
    elif max_speed >= 0.2:  SCALE_VAL = 0.2
    elif max_speed >= 0.1:  SCALE_VAL = 0.1
    else:                   SCALE_VAL = 0.05

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle(f'Fixation drift quiver — noise on vs noise off  '
                 f'({DURATION:.0f} s per fixation, screen at {DEPTH:.1f} m, '
                 f'first {DROP_S:.0f} s dropped)',
                 fontsize=11, fontweight='bold')

    for ax, drifts, label in [(axes[0], drifts_on,  'Noise on, target strobed'),
                               (axes[1], drifts_off, 'Noise off, target steady')]:
        px = drifts[:, 0]; py = drifts[:, 1]
        vx = drifts[:, 2]; vy = drifts[:, 3]
        speed = np.hypot(vx, vy)

        for r in [5.0, 10.0]:
            circle = plt.Circle((0, 0), r, fill=False, ls=':', lw=0.7, color='#bbbbbb')
            ax.add_patch(circle)
            ax.text(r * np.cos(np.radians(45)) + 0.3, r * np.sin(np.radians(45)) + 0.3,
                    f'{r:.0f}°', color='#888888', fontsize=8)

        ax.plot(px, py, 'o', color='black', ms=4, zorder=4)
        q = ax.quiver(px, py, vx, vy, speed,
                      cmap='viridis', angles='xy', scale_units='xy',
                      scale=QUIVER_SCALE, clim=(0, max_speed),
                      width=0.005, headwidth=4, headlength=5, zorder=5)

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
        ax.set_title(f'{label} — max |SPV| = {float(speed.max()):.3f} deg/s', fontsize=10)

    cbar = fig.colorbar(q, ax=axes, fraction=0.04, pad=0.04)
    cbar.set_label('|SPV| (deg/s)', fontsize=9)

    path, rp = utils.save_fig(fig, 'fixation_drift_quiver', show=show, params=PARAMS_DEFAULT)
    return utils.fig_meta(
        path, rp,
        title='Fixation Drift Quiver — noise on vs off',
        description=f'Mean slow-phase drift at 17 fixation positions over {DURATION:.0f} s each, '
                    f'compared with default sensory noise vs all noise sigmas zeroed. '
                    'Origin + 8 directions × 5°/10° eccentricity.',
        expected='Noise-on: drift magnitudes typically <1 deg/s at all positions, slightly '
                 'centripetal at eccentric positions. '
                 'Noise-off: residual drift is from deterministic dynamics (NI leak, plant) '
                 'and should be near zero at primary, with small centripetal pull at eccentricity.',
        citation='Cherici et al. (2012) J Vis 12(6):31; Martinez-Conde & Macknik 2017 Neuron.',
    )


# ── Section entry point ────────────────────────────────────────────────────────

SECTION = dict(
    id='experiments', title='Experimental',
    description='Exploratory paradigms: monocular occlusion, binocular fixation maintenance, '
                'fixation drift quiver across visual field.',
)


def _occlusion_summary(show, conv_columns, div_columns, t_np):
    """Average vergence + vergence velocity across L/R viewing eyes per condition.

    Six lines per panel:
        from convergence — continuous   (avg of L+R viewing)
        from convergence — flashing     (avg of L+R viewing)
        from convergence — dark         (single column)
        from divergence — continuous    (avg of L+R viewing)
        from divergence — flashing      (avg of L+R viewing)
        from divergence — dark          (single column)
    """

    def collect(columns_data):
        # columns_data is list of ((title, cond, occ), sim) tuples in the order
        # ['cont/L-occ', 'pulsed/L-occ', 'cont/R-occ', 'pulsed/R-occ', 'dark'].
        by_cond = {'continuous': [], 'pulsed': [], 'dark': []}
        for (title, cond, occ), st in columns_data:
            eye_L = np.array(st.plant[:, 0])
            eye_R = np.array(st.plant[:, 3])
            verg  = eye_L - eye_R
            spv_L = extract_spv_states(st, np.array(t_np), eye='left')[:, 0]
            spv_R = extract_spv_states(st, np.array(t_np), eye='right')[:, 0]
            verg_spv = spv_L - spv_R
            by_cond[cond].append((verg, verg_spv))
        # Average across viewing-eye repetitions for cont/pulsed; dark is single.
        out = {}
        for cond, runs in by_cond.items():
            verg_stack     = np.stack([r[0] for r in runs], axis=0)
            verg_spv_stack = np.stack([r[1] for r in runs], axis=0)
            out[cond] = (verg_stack.mean(axis=0), verg_spv_stack.mean(axis=0))
        return out

    conv = collect(conv_columns)
    div  = collect(div_columns)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle('Occlusion summary — vergence and vergence velocity averaged across L/R viewing',
                 fontsize=11)

    palette = {
        ('conv', 'continuous'): ('#1b7837', '-',  'from conv — continuous'),
        ('conv', 'pulsed'):     ('#5aae61', '--', 'from conv — flashing'),
        ('conv', 'dark'):       ('#a6dba0', ':',  'from conv — dark'),
        ('div',  'continuous'): ('#762a83', '-',  'from div — continuous'),
        ('div',  'pulsed'):     ('#9970ab', '--', 'from div — flashing'),
        ('div',  'dark'):       ('#c2a5cf', ':',  'from div — dark'),
    }

    for tag, data in (('conv', conv), ('div', div)):
        for cond in ('continuous', 'pulsed', 'dark'):
            color, ls, lbl = palette[(tag, cond)]
            verg, vverg = data[cond]
            axes[0].plot(t_np, verg,  color=color, ls=ls, lw=1.2, label=lbl)
            axes[1].plot(t_np, vverg, color=color, ls=ls, lw=1.0, label=lbl)

    for ax, ylab in zip(axes, ['Vergence (deg)', 'Vergence slow-phase velocity (deg/s)']):
        ax.axvline(_T_FIX, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.axhline(0, color='gray', lw=0.5, alpha=0.4)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, ncol=2, loc='best')
    axes[1].set_xlabel('Time (s)', fontsize=9)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'occlusion_summary', show=show,
                              figs_dir=utils.EXPT_FIGS_DIR, base_dir=utils.EXPT_DIR)
    return utils.fig_meta(
        path, rp,
        title='Occlusion summary — vergence + vergence velocity',
        description='Averaged vergence across L/R viewing eyes per occlusion condition.',
        expected='From-convergence runs sit near tonic with small drift; '
                 'from-divergence runs show larger drift toward convergent tonic '
                 'when fusion fails (dark) or weakens (flashing).',
        citation='Composite of from-convergence and from-divergence occlusion paradigms',
    )


def run(show=False):
    print('\n=== Experiments ===')
    figs = []
    print('  1/4  monocular occlusion — from convergence …')
    meta_conv, conv_cols, t_np = _occlusion_from_convergence(show)
    figs.append(meta_conv)
    print('  2/4  monocular occlusion — from divergence …')
    meta_div, div_cols, _ = _occlusion_from_divergence(show)
    figs.append(meta_div)
    print('  3/4  occlusion summary — averaged vergence …')
    figs.append(_occlusion_summary(show, conv_cols, div_cols, t_np))
    print('  4/4  fixation drift quiver across visual field …')
    figs.append(_drift_quiver(show))
    return figs


# ── HTML generation ────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f5f5; color: #222; display: flex; }
nav  { width: 200px; min-height: 100vh; background: #1a1a2e; color: #eee;
       padding: 20px 0; position: sticky; top: 0; flex-shrink: 0; }
nav h2  { font-size: 13px; padding: 0 16px 12px; color: #aaa;
          text-transform: uppercase; letter-spacing: 0.05em; }
nav a   { display: block; padding: 8px 16px; color: #ccc; text-decoration: none;
          font-size: 13px; border-left: 3px solid transparent; }
nav a:hover { background: #2a2a4e; color: #fff; border-left-color: #9b59b6; }
main { flex: 1; padding: 32px; max-width: 1400px; }
h1   { font-size: 22px; margin-bottom: 4px; }
.meta { font-size: 12px; color: #888; margin-bottom: 32px; }
.section    { margin-bottom: 48px; }
.section h2 { font-size: 18px; margin-bottom: 6px; border-bottom: 2px solid #9b59b6;
              padding-bottom: 6px; }
.section > p { font-size: 13px; color: #555; margin-bottom: 16px; }
.fig-grid   { display: grid; grid-template-columns: repeat(auto-fill, minmax(580px, 1fr));
              gap: 20px; }
.fig-card   { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
              padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.fig-card a img { width: 100%; border-radius: 4px; display: block;
                  border: 1px solid #eee; cursor: zoom-in; }
.fig-card h3 { font-size: 14px; margin: 12px 0 6px; }
.fig-card .desc { font-size: 12px; color: #555; margin-bottom: 8px; }
.expected   { background: #f5eeff; border-left: 3px solid #9b59b6;
              padding: 8px 10px; font-size: 12px; border-radius: 0 4px 4px 0;
              margin-bottom: 8px; }
.expected strong { display: block; font-size: 11px; color: #888;
                   text-transform: uppercase; margin-bottom: 2px; }
.citation   { font-size: 11px; color: #888; font-style: italic; }
.badge      { display: inline-block; padding: 2px 8px; border-radius: 12px;
              font-size: 10px; font-weight: 600; letter-spacing: 0.04em;
              text-transform: uppercase; margin-top: 8px; }
.badge.behavior { background: #d4edda; color: #155724; }
.badge.cascade  { background: #cce5ff; color: #004085; }
"""

_LIGHTBOX = """
<div id="lb" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;
     background:rgba(0,0,0,.85);z-index:1000;cursor:zoom-out;align-items:center;
     justify-content:center;">
  <img id="lb-img" style="max-width:95vw;max-height:95vh;border-radius:4px;">
</div>
<script>
(function(){
  var lb=document.getElementById('lb'),li=document.getElementById('lb-img');
  document.querySelectorAll('.fig-card a').forEach(function(a){
    a.addEventListener('click',function(e){e.preventDefault();li.src=a.href;lb.style.display='flex';});
  });
  lb.addEventListener('click',function(){lb.style.display='none';});
})();
</script>
"""


def _fig_card(fig):
    rel, title = fig.get('rel',''), fig.get('title','')
    desc, exp  = fig.get('description',''), fig.get('expected','')
    cit        = fig.get('citation','')
    ftype      = fig.get('type','behavior')
    path       = fig.get('path','')
    if path and not os.path.isfile(path):
        img = '<div style="padding:30px;text-align:center;color:#aaa;font-size:13px;">Figure not yet generated</div>'
    else:
        img = f'<a href="{rel}" target="_blank"><img src="{rel}" alt="{title}"></a>'
    badge = f'<span class="badge {ftype}">{ftype}</span>'
    return f"""
    <div class="fig-card">
      {img}
      <h3>{title}</h3>
      <p class="desc">{desc}</p>
      <div class="expected"><strong>Expected behavior</strong>{exp}</div>
      <p class="citation">&#128214; {cit}</p>
      {badge}
    </div>"""


def generate_html(figs):
    import datetime, oculomotor
    ts  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    ver = oculomotor.__version__
    nav_sections = '\n'.join(
        f'    <a href="#{f["title"].lower().replace(" ","_")}">{f["title"]}</a>' for f in figs
    )
    cards = '\n'.join(_fig_card(f) for f in figs)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OculomotorJax — Experiments</title>
  <style>{_CSS}</style>
</head>
<body>
  <nav>
    <h2 style="margin-bottom:4px;">Pages</h2>
    <a href="../">LLM Simulator</a>
    <a href="../benchmarks/">Benchmarks</a>
    <a href="../clinical_benchmarks/">Clinical Benchmarks</a>
    <a href="../parameters.html">Parameters</a>
    <div style="border-top:1px solid #2a2a4e;margin:10px 0 8px;"></div>
    <h2>Experiments</h2>
{nav_sections}
  </nav>
  <main>
    <h1>OculomotorJax — Experiments</h1>
    <p class="meta">
      Generated: <strong>{ts}</strong> &nbsp;|&nbsp;
      Version: <strong>{ver}</strong>
    </p>
    <section class="section">
      <h2>{SECTION['title']}</h2>
      <p>{SECTION['description']}</p>
      <div class="fig-grid">
        {cards}
      </div>
    </section>
  </main>
  {_LIGHTBOX}
</body>
</html>"""
    os.makedirs(utils.EXPT_DIR, exist_ok=True)
    with open(utils.EXPT_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\nHTML report written: {utils.EXPT_HTML_PATH}')


if __name__ == '__main__':
    figs = run(show=SHOW)
    generate_html(figs)
