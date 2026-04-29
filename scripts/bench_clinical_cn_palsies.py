"""Clinical cranial nerve palsy, motor nucleus, and INO benchmarks.

9-position gaze, INO saccade time-series, graded CN VI / INO.

Usage:
    python -X utf8 scripts/bench_clinical_cn_palsies.py
    python -X utf8 scripts/bench_clinical_cn_palsies.py --show
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_clinical_utils as utils

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, with_sensory, simulate, SimConfig,
)
from oculomotor.sim import kinematics as km
from oculomotor.models.plant_models.muscle_geometry import (
    ABN_R,
    LR_R, MR_R, SR_R, IR_R, SO_R, IO_R,
    G_NUCLEUS_DEFAULT, G_NERVE_DEFAULT,
)
from oculomotor.analysis import ax_fmt

SHOW = '--show' in sys.argv
DT   = 0.001

# Suppress noise for clean deterministic figures
THETA = with_sensory(PARAMS_DEFAULT,
                     sigma_canal=0.0, sigma_pos=0.0,
                     sigma_vel=0.0, sigma_slip=0.0)

SECTION = dict(
    id='clin_cn_palsies',
    title='C. Cranial Nerve Palsies, Motor Nuclear Lesions & INO',
    description='Simulated 9 positions of gaze and saccade trajectories under '
                'CN VI (abducens nerve / nucleus), CN III (oculomotor), CN IV (trochlear) '
                'palsies and internuclear ophthalmoplegia (INO). '
                'Graded recovery series for CN VI nerve palsy and partial INO.',
)

# ── Lesion parameters ─────────────────────────────────────────────────────────

# CN VI nerve (R): LR_R paretic only (nucleus and fellow-eye MR intact)
THETA_CN6_NERVE = with_brain(THETA,
    g_nerve=G_NERVE_DEFAULT.at[LR_R].set(0.0))

# CN VI nucleus (R): ABN_R → only ipsilateral LR_R affected (no MLF in model)
THETA_CN6_NUC = with_brain(THETA,
    g_nucleus=G_NUCLEUS_DEFAULT.at[ABN_R].set(0.0))

# CN III nerve (R): MR/SR/IR/IO all paretic; LR (CN VI) and SO (CN IV) intact
_CN3_R = jnp.array([MR_R, SR_R, IR_R, IO_R])
THETA_CN3 = with_brain(THETA,
    g_nerve=G_NERVE_DEFAULT.at[_CN3_R].set(0.0))

# CN IV nerve (R): SO_R paretic → hypertropia in adduction, loss of intorsion
THETA_CN4 = with_brain(THETA,
    g_nerve=G_NERVE_DEFAULT.at[SO_R].set(0.0))

# Left INO: version_yaw drive to L MR (CN3_MR_L) cut
#   Rightward saccade: R eye abducts normally, L eye adduction slow/absent
THETA_INO_L = with_brain(THETA, g_mlf_ver_L=0.0)

# Right INO: version_yaw drive to R MR (CN3_MR_R) cut
#   Leftward saccade: L eye abducts normally, R eye adduction slow/absent
THETA_INO_R = with_brain(THETA, g_mlf_ver_R=0.0)

# Bilateral INO (BIMLF): both MR subnuclei lose version drive
THETA_BIMLF = with_brain(THETA, g_mlf_ver_L=0.0, g_mlf_ver_R=0.0)

CONDITIONS_9POS = [
    ('Healthy',                           THETA),
    ('CN VI nerve (R)',                   THETA_CN6_NERVE),
    ('CN VI nucleus (R)',                 THETA_CN6_NUC),
    ('CN III nerve (R)',                  THETA_CN3),
    ('CN IV nerve (R)',                   THETA_CN4),
    ('Left INO',                          THETA_INO_L),
    ('Right INO',                         THETA_INO_R),
    ('Bilateral INO',                     THETA_BIMLF),
]

# ── Gaze targets ──────────────────────────────────────────────────────────────

H_DEG = 20.0   # horizontal amplitude
V_DEG = 15.0   # vertical amplitude

# (yaw_deg, pitch_deg) for the 9 standard gaze positions
TARGETS_DEG = {
    'Center'    : (   0,    0),
    'Right'     : ( H_DEG,  0),
    'Left'      : (-H_DEG,  0),
    'Up'        : (   0,  V_DEG),
    'Down'      : (   0, -V_DEG),
    'Up-Right'  : ( H_DEG,  V_DEG),
    'Up-Left'   : (-H_DEG,  V_DEG),
    'Down-Right': ( H_DEG, -V_DEG),
    'Down-Left' : (-H_DEG, -V_DEG),
}

# ── Simulation helpers ────────────────────────────────────────────────────────

T_SAC   = 0.5   # saccade simulation duration (s)
N_SAC   = int(T_SAC / DT) + 1
T_ARR   = np.linspace(0.0, T_SAC, N_SAC, dtype=np.float32)
_CFG    = SimConfig(warmup_s=2.0)   # settle eyes at center before saccade


def _xyz(yaw_deg, pitch_deg, distance_m=1.0):
    """Convert (yaw, pitch) in deg to Cartesian target position (tan-projection)."""
    return np.array([
        np.tan(np.radians(yaw_deg))   * distance_m,
        np.tan(np.radians(pitch_deg)) * distance_m,
        distance_m,
    ], dtype=np.float32)


def _target_jump(yaw_deg, pitch_deg, distance_m=1.0):
    """Target at center during warmup (t[0]), jumps to (yaw, pitch) at t[1]."""
    T = N_SAC
    pt = np.zeros((T, 3), np.float32)
    pt[0]  = _xyz(0.0, 0.0, distance_m)    # warmup anchors at center
    pt[1:] = _xyz(yaw_deg, pitch_deg, distance_m)
    lv = np.zeros((T, 3), np.float32)      # suppress velocity spike from step
    return km.build_target(T_ARR, lin_pos=pt, lin_vel=lv)


def _sim_saccade(params, yaw_deg, pitch_deg, distance_m=1.0, key=0):
    """Simulate a saccade to (yaw, pitch) deg. Returns (T, 6) eye positions."""
    tgt = _target_jump(yaw_deg, pitch_deg, distance_m)
    T   = N_SAC
    eye = simulate(
        params, T_ARR,
        target=tgt,
        scene_present_array=np.ones(T, np.float32),
        target_present_array=np.ones(T, np.float32),
        max_steps=int(T * 1.1) + 2500,
        sim_config=_CFG,
        return_states=False,
        key=jax.random.PRNGKey(key),
    )
    return np.array(eye)   # (T, 6): left(3) | right(3)


# ── Figure 1: 9 positions of gaze ─────────────────────────────────────────────

def _nine_position(show):
    print('  Running 9-position gaze (8 conditions × 9 targets = 72 simulations)...')

    # Run all simulations
    all_results = []
    for cond_name, params in CONDITIONS_9POS:
        results = {}
        for tgt_name, (ya, pa) in TARGETS_DEG.items():
            eye = _sim_saccade(params, ya, pa)
            results[tgt_name] = (
                eye[-1, :2].copy(),   # L eye (yaw, pitch) deg
                eye[-1, 3:5].copy(),  # R eye (yaw, pitch) deg
            )
        all_results.append(results)
        print(f'    {cond_name}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    # 4×4 grid: each condition occupies one column of 2 stacked panels
    # (top = L eye, bottom = R eye), laid out as 2 condition-rows of 4.
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    lim_h = H_DEG + 8
    lim_v = V_DEG + 7
    names  = list(TARGETS_DEG.keys())

    offsets = {'Center': (-2, 1.5), 'Right': (0.5, 1), 'Left': (-5, 1),
               'Up': (-2, 1.5), 'Down': (-2, -2.5),
               'Up-Right': (0.5, 1), 'Up-Left': (-6, 1),
               'Down-Right': (0.5, -2.5), 'Down-Left': (-6, -2.5)}

    for i, ((cond_name, _), results) in enumerate(zip(CONDITIONS_9POS, all_results)):
        L = np.array([results[n][0] for n in names])  # (9, 2)
        R = np.array([results[n][1] for n in names])  # (9, 2)

        row_l = (i // 4) * 2      # 0 for conditions 0-3, 2 for conditions 4-7
        row_r = row_l + 1
        col   = i % 4
        ax_l  = axes[row_l, col]
        ax_r  = axes[row_r, col]

        for ax, pts, color, marker, eye_label in [
            (ax_l, L, 'royalblue', 'o', 'L eye'),
            (ax_r, R, 'crimson',   's', 'R eye'),
        ]:
            ax.scatter(pts[:, 0], pts[:, 1], c=color, s=55, zorder=3, marker=marker)
            ax.axhline(0, color='k', lw=0.4, alpha=0.25)
            ax.axvline(0, color='k', lw=0.4, alpha=0.25)
            ax.set_xlim(-lim_h, lim_h)
            ax.set_ylim(-lim_v, lim_v)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.12)
            ax.set_ylabel(f'{eye_label}\nPitch (deg)', fontsize=7)
            if ax is ax_r:
                ax.set_xlabel('Yaw (deg)', fontsize=7)

        ax_l.set_title(cond_name, fontsize=8, fontweight='bold')

        # Label target positions in the first condition's L-eye panel only
        if i == 0:
            for j, n in enumerate(names):
                dx, dy = offsets.get(n, (1, 1))
                ax_l.annotate(n, xy=L[j], xytext=(L[j, 0]+dx, L[j, 1]+dy),
                              fontsize=5.5, color='navy', alpha=0.8)

    fig.suptitle('9 Positions of Gaze  –  top row: L eye (blue)  ·  bottom row: R eye (red)',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()

    path, rp = utils.save_fig(fig, 'clin_cn_9positions', show=show)
    return utils.fig_meta(path, rp,
        title='9 Positions of Gaze',
        description='Standard clinical 9-position gaze test. Each panel shows '
                    'left (blue circles) and right (red squares) final eye positions '
                    'after a saccade to each of the 9 standard fixation targets. '
                    'Vertical or horizontal gaps between paired symbols reveal the '
                    'duction deficit for each lesion.',
        expected='CN VI (R): right esotropia increasing on rightward gaze; '
                 'CN III (R): right eye down-and-out, can\'t adduct/elevate; '
                 'CN IV (R): right hypertropia maximal in adduction; '
                 'Left INO: left adduction lag on rightward gaze; '
                 'Bilateral INO: bilateral adduction failure, exotropia on lateral gaze.',
        citation='Leigh RJ & Zee DS (2015) The Neurology of Eye Movements, 5th ed.',
        fig_type='behavior',
    )


# ── Figure 2: INO saccade time-series ─────────────────────────────────────────

def _ino_timeseries(show):
    print('  Running INO saccade time-series...')

    cases = [
        ('Healthy',        THETA,        '#444444'),
        ('Left INO',       THETA_INO_L,  '#7b2d8b'),
        ('Right INO',      THETA_INO_R,  '#e08214'),
        ('Bilateral INO',  THETA_BIMLF,  '#d6604d'),
    ]

    # Rightward saccade: L eye should adduct → left INO impairs L adduction
    # Leftward saccade: R eye should adduct → right INO impairs R adduction
    traj_right = {name: _sim_saccade(p, H_DEG, 0.0, key=42)
                  for name, p, _ in cases}
    traj_left  = {name: _sim_saccade(p, -H_DEG, 0.0, key=42)
                  for name, p, _ in cases}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    t = T_ARR

    for name, _, col in cases:
        eye_r = traj_right[name]
        eye_l = traj_left[name]

        vel_L_r = np.gradient(eye_r[:, 0], DT)
        vel_R_r = np.gradient(eye_r[:, 3], DT)
        vel_L_l = np.gradient(eye_l[:, 0], DT)
        vel_R_l = np.gradient(eye_l[:, 3], DT)

        ls_L = '--'
        ls_R = '-'

        # Rightward saccade — position
        axes[0, 0].plot(t, eye_r[:, 0], color=col, lw=1.3, ls=ls_L, label=f'{name} L')
        axes[0, 0].plot(t, eye_r[:, 3], color=col, lw=1.3, ls=ls_R, label=f'{name} R')
        # Rightward saccade — velocity
        axes[1, 0].plot(t, vel_L_r, color=col, lw=1.2, ls=ls_L)
        axes[1, 0].plot(t, vel_R_r, color=col, lw=1.2, ls=ls_R)

        # Leftward saccade — position
        axes[0, 1].plot(t, eye_l[:, 0], color=col, lw=1.3, ls=ls_L)
        axes[0, 1].plot(t, eye_l[:, 3], color=col, lw=1.3, ls=ls_R)
        # Leftward saccade — velocity
        axes[1, 1].plot(t, vel_L_l, color=col, lw=1.2, ls=ls_L)
        axes[1, 1].plot(t, vel_R_l, color=col, lw=1.2, ls=ls_R)

    for ax in axes.flat:
        ax.axhline(0, color='k', lw=0.4, alpha=0.3)
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Time (s)', fontsize=8)

    axes[0, 0].set_ylabel('Eye yaw (deg)', fontsize=8)
    axes[1, 0].set_ylabel('Eye yaw vel (deg/s)', fontsize=8)
    axes[0, 1].set_ylabel('Eye yaw (deg)', fontsize=8)
    axes[1, 1].set_ylabel('Eye yaw vel (deg/s)', fontsize=8)

    axes[0, 0].set_title(f'Rightward {H_DEG}° saccade — position', fontsize=9)
    axes[1, 0].set_title(f'Rightward {H_DEG}° saccade — velocity', fontsize=9)
    axes[0, 1].set_title(f'Leftward {H_DEG}° saccade — position', fontsize=9)
    axes[1, 1].set_title(f'Leftward {H_DEG}° saccade — velocity', fontsize=9)

    for ax in axes[:, 0]:
        ax.annotate('dashed = L eye  /  solid = R eye',
                    xy=(0.02, 0.04), xycoords='axes fraction',
                    fontsize=6.5, color='gray')

    axes[0, 0].legend(fontsize=7, ncol=2, loc='upper left')

    fig.suptitle('INO Saccade Trajectories  (dashed = L eye, solid = R eye)',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()

    path, rp = utils.save_fig(fig, 'clin_cn_ino_timeseries', show=show)
    return utils.fig_meta(path, rp,
        title='INO Saccade Trajectories',
        description='Horizontal saccade position and velocity for healthy, left INO, '
                    'right INO, and bilateral INO. Left INO impairs left-eye adduction '
                    'on rightward saccades; right INO impairs right-eye adduction on '
                    'leftward saccades. Dashed = left eye; solid = right eye.',
        expected='Left INO: left adducting eye (dashed) velocity reduced/delayed on rightward '
                 'saccade; abducting right eye normal. Right INO: mirror pattern on leftward saccade. '
                 'Bilateral INO: both adducting eyes impaired; both abducting eyes normal.',
        citation='Bhidayasiri R et al. (2000) Brain 123:1241-1267 — INO pathophysiology; '
                 'Zee DS et al. (1992) Ann Neurol 32:756-764 — BIMLF syndrome.',
        fig_type='cascade',
    )


# ── Figure 3: Graded palsy recovery series ────────────────────────────────────

def _graded_palsy(show):
    """Continuous ipsilateral → contralateral saccade sequence at 5 gain levels.

    Ipsilateral = rightward (reveals both CN VI R and left INO defects).
    Contralateral = leftward.
    Both eyes shown on same axes: L eye dashed, R eye solid.
    """
    print('  Running graded CN VI / INO recovery series...')

    gains  = [1.0, 0.75, 0.5, 0.25, 0.0]
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(gains)))

    # Continuous trace: centre → ipsilateral (+H_DEG) → contralateral (−H_DEG)
    T_TOTAL  = 2.0
    T_IPSI   = 0.3    # target jumps ipsilaterally at this time
    T_CONTRA = 1.1    # target jumps contralaterally at this time

    t   = np.arange(0.0, T_TOTAL, DT)
    T   = len(t)

    tgt_yaw = np.where(t < T_IPSI, 0.0,
              np.where(t < T_CONTRA, float(H_DEG), -float(H_DEG))).astype(np.float32)
    pt = np.zeros((T, 3), np.float32)
    pt[:, 0] = np.tan(np.radians(tgt_yaw))
    pt[:, 2] = 1.0
    tgt_km = km.build_target(t, lin_pos=pt, lin_vel=np.zeros((T, 3), np.float32))
    cfg    = SimConfig(warmup_s=0.5)

    def _sim_seq(params):
        return np.array(simulate(
            params, t,
            target=tgt_km,
            scene_present_array=np.ones(T, np.float32),
            target_present_array=np.ones(T, np.float32),
            max_steps=int(T_TOTAL / DT) + 2500,
            sim_config=cfg,
            return_states=False,
        ))

    cn6_eyes = [_sim_seq(with_brain(THETA, g_nerve=G_NERVE_DEFAULT.at[LR_R].set(float(g))))
                for g in gains]
    ino_eyes = [_sim_seq(with_brain(THETA, g_mlf_ver_L=float(g)))
                for g in gains]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Graded Palsy Recovery — Ipsilateral (+{H_DEG}°) then Contralateral (−{H_DEG}°) Saccade\n'
        'Scene on · Target on  |  dashed = left eye · solid = right eye',
        fontsize=10, fontweight='bold')

    for g, col, cn6, ino in zip(gains, colors, cn6_eyes, ino_eyes):
        lbl = f'g={g:.2f}'
        axes[0].plot(t, cn6[:, 0], color=col, lw=1.0, ls='--')        # L eye (fellow)
        axes[0].plot(t, cn6[:, 3], color=col, lw=1.6, ls='-',  label=lbl)  # R eye (paretic)
        axes[1].plot(t, ino[:, 0], color=col, lw=1.6, ls='--', label=lbl)  # L eye (adducting, affected)
        axes[1].plot(t, ino[:, 3], color=col, lw=1.0, ls='-')          # R eye (abducting)

    for ax in axes:
        ax.plot(t, tgt_yaw, color='k', lw=0.6, ls=':', alpha=0.35, label='Target')
        ax.axvline(T_IPSI,   color='gray', lw=0.7, ls='--', alpha=0.4)
        ax.axvline(T_CONTRA, color='gray', lw=0.7, ls='--', alpha=0.4)
        ax.axhline(0, color='k', lw=0.3, alpha=0.25)
        ax.grid(True, alpha=0.15)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Eye yaw (deg)', fontsize=9)
        ax.tick_params(labelsize=7)

    axes[0].set_title(f'CN VI nerve palsy (R)\n'
                      'solid = R paretic eye · dashed = L fellow', fontsize=9)
    axes[0].legend(title='nerve gain', fontsize=7, loc='lower left', ncol=2)

    axes[1].set_title(f'Left INO (g_mlf_ver_L)\n'
                      'dashed = L adducting (affected) · solid = R abducting', fontsize=9)
    axes[1].legend(title='MLF gain', fontsize=7, loc='lower right', ncol=2)

    plt.tight_layout()
    path, rp = utils.save_fig(fig, 'clin_cn_graded', show=show)
    return utils.fig_meta(path, rp,
        title='Graded CN VI and INO Recovery',
        description=f'Continuous ipsilateral (+{H_DEG}°) then contralateral (−{H_DEG}°) saccade. '
                    'CN VI R nerve palsy (left): R (paretic) eye undershoots and slows. '
                    'Left INO (right): L (adducting) eye lags and undershoots on rightward saccade. '
                    'Both eyes on same axes; lit room, target present.',
        expected='CN VI: R eye undershoots rightward saccade, proportional to gain; L eye normal. '
                 'Left INO: L eye adduction lag/undershoot on rightward saccade; R eye normal. '
                 'Both eyes normal on leftward saccade (CN VI fellow normal; INO abducting R normal).',
        citation='Keller & Robinson (1972) J Neurophysiol 35:466; '
                 'Frohman et al. (2003) J Neuro-ophthalmol 23:106.',
        fig_type='behavior',
    )


# ── Entry point ───────────────────────────────────────────────────────────────

FIGURES = None  # populated by run()

def run(show=False):
    global FIGURES
    figs = [
        _nine_position(show),
        _ino_timeseries(show),
        _graded_palsy(show),
    ]
    FIGURES = figs
    return figs


if __name__ == '__main__':
    figs = run(show=SHOW)
    for f in figs:
        print(f'  Saved: {f["path"]}')
