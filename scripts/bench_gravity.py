"""Gravity estimator benchmarks — OCR, OVAR (Fig 5), tilt suppression (Fig 6).

Replicates key figures from Laurens & Angelaki (2011) Exp Brain Res 210:407-422.

Parameters (Laurens & Angelaki 2011):
    K_grav = 0.6  (go — somatogravic otolith correction gain)
    K_gd   = 0.5  (gravity dumping — damps VS ⊥ gravity; enables tilt suppression / OVAR)

Usage:
    python -X utf8 scripts/bench_gravity.py
    python -X utf8 scripts/bench_gravity.py --show
"""

import sys, os, argparse
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
    PARAMS_DEFAULT, with_brain, simulate, SimConfig,
    _IDX_GRAV, _IDX_C, _IDX_SG, _IDX_VERG, _IDX_PURSUIT,
)
from oculomotor.sim import kinematics as km
from oculomotor.analysis import ax_fmt, vs_net, ni_net, fit_tc, extract_spv_states
from oculomotor.models.sensory_models.sensory_model import PINV_SENS as CANAL_PINV
from oculomotor.models.sensory_models.canal import N_CANALS, FLOOR, _SOFTNESS

SHOW = '--show' in sys.argv
DT   = 0.001
G0   = 9.81

K_GD     = 0.05 * (180 / 3.14159265)   # 0.05 rad/s² → deg/s² ≈ 2.86
TAU_GRAV = 5.0                          # gravity estimate TC (s); somatogravic BW = 1/(2π×5) ≈ 0.032 Hz
G_OCR  = 10.0 / 9.81   # OCR gain (deg/(m/s²)): ~10° at 90° tilt (Howard & Templeton 1966)


SECTION = dict(
    id='gravity', title='3. Gravity Estimator',
    description='Canal-otolith interaction: OCR, OVAR, VOR tilt suppression, '
                'OCR vs tilt angle, somatogravic OCR frequency dependence. '
                f'Parameters: tau_grav={TAU_GRAV}, K_gd={K_GD}, g_ocr={G_OCR} (Laurens & Angelaki 2011).',
)


def _pad3(v1d, axis):
    T = len(v1d)
    out = np.zeros((T, 3))
    out[:, {'yaw': 0, 'pitch': 1, 'roll': 2}[axis]] = v1d
    return out


def _stim_twin(ax, t, pos, vel, pos_label, vel_label, pos_color, vel_color):
    """Plot position on left y-axis and velocity on right y-axis (twin)."""
    ax.plot(t, pos, color=pos_color, lw=1.5, label=pos_label)
    ax.set_ylabel(pos_label, fontsize=8, color=pos_color)
    ax.tick_params(axis='y', labelcolor=pos_color)
    ax2 = ax.twinx()
    ax2.plot(t, vel, color=vel_color, lw=1.0, ls='--', alpha=0.8, label=vel_label)
    ax2.set_ylabel(vel_label, fontsize=8, color=vel_color)
    ax2.tick_params(axis='y', labelcolor=vel_color)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.15)
    return ax2


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ocular Counterroll (OCR) — multi-angle traces + SS main sequence
# ─────────────────────────────────────────────────────────────────────────────

def _ocr(show):
    """OCR for a sweep of tilt angles: time traces + SS torsion vs tilt scatter."""
    TILTS_DEG = [5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    TILT_VEL  = 60.0
    HOLD_T    = 10.0

    params = with_brain(PARAMS_DEFAULT, g_ocr=G_OCR)
    cmap   = plt.get_cmap('plasma')
    colors = [cmap(i / (len(TILTS_DEG) - 1)) for i in range(len(TILTS_DEG))]

    torsion_ss  = []
    traces_t    = {}
    traces_eye  = {}
    traces_gest = {}

    for i, tilt_deg in enumerate(TILTS_DEG):
        tilt_dur = tilt_deg / TILT_VEL
        total    = tilt_dur + HOLD_T
        t        = np.arange(0.0, total, DT)
        hv_roll  = np.where(t < tilt_dur, TILT_VEL, 0.0)
        head_km  = km.build_kinematics(t, rot_vel=_pad3(hv_roll, 'roll'))

        st = simulate(params, t,
                      head=head_km,
                      target_present_array=np.zeros(len(t)),
                      sim_config=SimConfig(warmup_s=0.0),
                      return_states=True)

        eye_roll = (np.array(st.plant[:, 2]) + np.array(st.plant[:, 5])) / 2.0
        g_est    = np.array(st.brain[:, _IDX_GRAV])
        t_hold   = t - tilt_dur
        traces_t[tilt_deg]    = t_hold
        traces_eye[tilt_deg]  = eye_roll
        traces_gest[tilt_deg] = g_est[:, 0]
        torsion_ss.append(float(eye_roll[-1]))

    torsion_expected = [-G_OCR * G0 * np.sin(np.radians(d)) for d in TILTS_DEG]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    fig.suptitle(f'Ocular Counterroll (OCR)  (g_ocr={G_OCR:.2f}, tau_grav={TAU_GRAV})',
                 fontsize=12, fontweight='bold')

    # Row 1: eye torsion traces aligned to hold onset
    ax1 = axes[0]
    for i, tilt_deg in enumerate(TILTS_DEG):
        ax1.plot(traces_t[tilt_deg], traces_eye[tilt_deg],
                 color=colors[i], lw=1.5, label=f'{tilt_deg:.0f}°')
        ax1.axhline(torsion_expected[i], color=colors[i], lw=0.6, ls=':', alpha=0.5)
    ax1.axvline(0.0, color='gray', lw=0.8, ls='-',
                label='tilt end / hold onset')
    ax1.axhline(0.0, color='k', lw=0.4)
    ax_fmt(ax1, ylabel='Eye torsion (deg)')
    ax1.set_title('Eye counter-roll (dotted = expected G_OCR×G0×sin(θ))', fontsize=9)
    ax1.legend(fontsize=7, ncol=4, title='Tilt angle', title_fontsize=7)
    ax1.set_xlim(-1.0, HOLD_T)

    # Row 2: g_est[0] traces (world x = right/interaural)
    ax2 = axes[1]
    for i, tilt_deg in enumerate(TILTS_DEG):
        expected_y = G0 * np.sin(np.radians(tilt_deg))   # +G0·sin(θ) in world frame
        ax2.plot(traces_t[tilt_deg], traces_gest[tilt_deg],
                 color=colors[i], lw=1.2, label=f'{tilt_deg:.0f}°')
        ax2.axhline(expected_y, color=colors[i], lw=0.6, ls=':', alpha=0.5)
    ax2.axvline(0.0, color='gray', lw=0.8, ls='-')
    ax2.axhline(0.0, color='k', lw=0.4)
    ax2.text(-0.85, -0.3, '0 = upright', fontsize=7, color='k', style='italic')
    ax2.set_ylabel('g_est[0] interaural/right (m/s²)', fontsize=8)
    ax2.set_title('Gravity estimate (dotted = +G0·sin(θ)); stays flat during hold', fontsize=9)
    ax2.legend(fontsize=7, ncol=3)
    ax2.set_xlim(-1.0, HOLD_T)
    ax2.set_ylim(-1, 12)
    ax2.grid(True, alpha=0.15)
    ax2.set_xlabel('Time rel. hold onset (s)', fontsize=8)

    # Row 3: SS scatter — torsion vs tilt angle
    ax3 = axes[2]
    theta_fine = np.linspace(0, 90, 200)
    ax3.plot(theta_fine, [-G_OCR * G0 * np.sin(np.radians(d)) for d in theta_fine],
             color='tomato', lw=1.5, ls='--',
             label=f'Sin prediction: −G_OCR×G0×sin(θ)  (min {-G_OCR*G0:.1f}° at 90°)')
    ax3.scatter(TILTS_DEG, torsion_ss,
                color=[colors[i] for i in range(len(TILTS_DEG))], s=70, zorder=5)
    for i, (td, ts) in enumerate(zip(TILTS_DEG, torsion_ss)):
        ax3.annotate(f'{ts:.1f}°', (td, ts), fontsize=7,
                     xytext=(3, 4), textcoords='offset points')
    ax3.axhline(0.0, color='k', lw=0.4)
    ax3.set_xlabel('Head tilt (deg)', fontsize=9)
    ax3.set_ylabel('Steady-state torsion (deg)', fontsize=9)
    ax3.set_title('OCR main sequence: steady-state torsion vs tilt angle', fontsize=9)
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.2)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'gravity_ocr', show=show)
    ocr_30 = G_OCR * G0 * np.sin(np.radians(30))
    return utils.fig_meta(path, rp,
        title='Ocular Counterroll (OCR)',
        description=f'Counter-roll traces for tilts {TILTS_DEG}° + SS scatter. g_ocr={G_OCR:.2f}.',
        expected=f'Torsion ≈ −G_OCR×G0×sin(θ). 30° → ≈−{ocr_30:.1f}°. '
                 f'g_est holds flat after tilt (no drift).',
        citation='Boff, Kaufman & Thomas (1986); Laurens & Angelaki (2011)',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────
# 2. OVAR — Off-Vertical Axis Rotation  (Laurens Fig 5)
# ─────────────────────────────────────────────────────────────────────────────

def _ovar(show):
    from oculomotor.models.brain_models.brain_model import _IDX_HEAD
    from oculomotor.models.plant_models.readout import rotation_matrix
    from oculomotor.models.sensory_models.retina import ypr_to_xyz
    SPIN_VEL    = 60.0
    TILTS_DEG   = [10.0, 30.0, 60.0, 90.0]
    # Stimulus protocol (separated tilt and rotation events):
    #   [0, T_TILT_START):           upright at rest
    #   [T_TILT_START, T_TILT_END):  tilt ramp at TILT_VEL deg/s up to tilt_deg
    #   [T_TILT_END, T_ROT_START):   rest at tilted position — let g_est settle
    #   [T_ROT_START, T_ROT_RAMP):   rotation ramp 0 → SPIN_VEL
    #   [T_ROT_RAMP, TOTAL):         constant SPIN_VEL
    TILT_VEL      = 30.0   # deg/s tilt rate
    T_TILT_START  = 2.0
    T_TILT_END_MAX = T_TILT_START + 90.0 / TILT_VEL   # 5 s for biggest tilt
    T_ROT_START   = 35.0                              # delay rotation onset to let a_est / v_lin fully settle after tilt
    ROT_RAMP      = 1.0                                # 1 s smooth ramp
    T_ROT_RAMP    = T_ROT_START + ROT_RAMP
    TOTAL         = T_ROT_RAMP + 50.0                  # 50 s of clean rotation

    params = with_brain(PARAMS_DEFAULT, K_gd=K_GD)
    cfg    = SimConfig(warmup_s=0.0)   # start from rest (v[0]=0 required)
    t      = np.arange(0.0, TOTAL, DT)
    T      = len(t)
    period = 360.0 / SPIN_VEL

    colors = ['#1b7837', '#762a83', '#e08214', '#c0392b']
    # Track right-axis handles so we can format them after the per-condition loop.
    a_est_axes = []

    # Layout: SPV | g_est[x,y,z] (3) | v_lin[x,y,z] (3) | VS | Stimulus = 9 rows
    fig, axes = plt.subplots(9, 1, figsize=(14, 22), sharex=True)
    fig.suptitle(
        f'OVAR — Off-Vertical Axis Rotation  (Laurens & Angelaki 2011, Fig 5)\n'
        f'{SPIN_VEL:.0f} °/s rotation, K_gd={K_GD}, tau_grav={TAU_GRAV}',
        fontsize=12, fontweight='bold')

    for ci, tilt_deg in enumerate(TILTS_DEG):
        # Build per-condition velocity profile with separated tilt and rotation phases.
        T_TILT_END = T_TILT_START + tilt_deg / TILT_VEL
        # Roll velocity: ramp up to TILT_VEL during [T_TILT_START, T_TILT_END), 0 elsewhere.
        # Integrates to tilt_deg by T_TILT_END.
        roll_vel = np.where((t >= T_TILT_START) & (t < T_TILT_END), TILT_VEL, 0.0)
        # Yaw velocity: 0 until T_ROT_START, then linear ramp to SPIN_VEL over ROT_RAMP s, hold.
        yaw_vel = np.zeros(T)
        ramp_mask = (t >= T_ROT_START) & (t < T_ROT_RAMP)
        hold_mask = t >= T_ROT_RAMP
        yaw_vel[ramp_mask] = SPIN_VEL * (t[ramp_mask] - T_ROT_START) / ROT_RAMP
        yaw_vel[hold_mask] = SPIN_VEL
        head_vel = np.stack([yaw_vel, np.zeros(T), roll_vel], axis=1)
        head_km  = km.build_kinematics(t, rot_vel=head_vel,
                                       rot_pos_0=[0.0, 0.0, 0.0])  # start upright

        st       = simulate(params, t,
                            head=head_km,
                            scene_present_array=np.zeros(T),
                            target_present_array=np.zeros(T),
                            sim_config=cfg, return_states=True)

        eye_pos = (np.array(st.plant[:, 0]) + np.array(st.plant[:, 3])) / 2.0
        eye_vel = np.gradient(eye_pos, DT)
        spv     = extract_spv_states(st, t)[:, 0]
        g_est   = np.array(st.brain[:, _IDX_GRAV])[:, :3]
        v_lin   = np.array(st.brain[:, _IDX_HEAD])

        # Compute gia per timestep from head pose (no head translation here, so a_head=0).
        # gia_head = R(q_head)^T @ G_WORLD.  Then a_est = gia − g_est is what HE integrates.
        # Vectorized via jax.vmap.
        q_head_arr = jnp.asarray(head_km.rot_pos)
        def _gia_one(q):
            R = rotation_matrix(ypr_to_xyz(q))
            return R.T @ jnp.array([0.0, G0, 0.0])
        gia   = np.asarray(jax.vmap(_gia_one)(q_head_arr))
        a_est = gia - g_est   # what HE integrates into v_lin

        col = colors[ci]
        lbl = f'{tilt_deg:.0f}° tilt'
        axes[0].plot(t, spv,             color=col, lw=1.2, alpha=0.85, label=lbl)
        axes[1].plot(t, g_est[:, 0],     color=col, lw=1.2, label=lbl)
        axes[2].plot(t, g_est[:, 1],     color=col, lw=1.2, label=lbl)
        axes[3].plot(t, g_est[:, 2],     color=col, lw=1.2, label=lbl)
        axes[4].plot(t, v_lin[:, 0]*100, color=col, lw=1.2, label=lbl)    # cm/s
        axes[5].plot(t, v_lin[:, 1]*100, color=col, lw=1.2, label=lbl)
        axes[6].plot(t, v_lin[:, 2]*100, color=col, lw=1.2, label=lbl)
        # Overlay a_est (HE integrand) on a right axis for each v_lin panel.
        for axis_idx, row in [(0, 4), (1, 5), (2, 6)]:
            if ci == 0:
                a_est_axes.append(axes[row].twinx())
            ax_r = a_est_axes[axis_idx]
            ax_r.plot(t, a_est[:, axis_idx], color=col, lw=0.8, ls='--', alpha=0.6,
                      label=f'a_est {lbl}' if axis_idx == 0 else None)
        axes[7].plot(t, vs_net(st)[:,0], color=col, lw=1.2, label=lbl)
        # Stimulus: yaw velocity is identical for all conditions — plot once
        if ci == 0:
            axes[8].plot(t, head_km.rot_vel[:, 0], 'k-', lw=1.5, label=f'{SPIN_VEL:.0f}°/s body yaw')

    # Twin axis: right = initial roll tilt per condition (horizontal colored lines)
    ax_stim_b = axes[8].twinx()
    for ci, tilt_deg in enumerate(TILTS_DEG):
        ax_stim_b.axhline(tilt_deg, color=colors[ci], lw=1.5, ls='--',
                     label=f'Initial roll tilt = {tilt_deg:.0f}°')
    ax_stim_b.set_ylabel('Initial roll tilt (°)', fontsize=9, color='gray')
    ax_stim_b.tick_params(axis='y', labelcolor='gray')
    ax_stim_b.set_ylim(-5, 100)

    # Period markers start after rotation reaches steady state (T_ROT_RAMP).
    pm = np.arange(T_ROT_RAMP + period, TOTAL, period)
    # Also mark the tilt and rotation onset events.
    event_marks = [(T_TILT_START, '#1f77b4', 'tilt start'),
                   (T_ROT_START,  '#d62728', 'rot start')]
    for ax in axes[:8]:
        for p in pm: ax.axvline(p, color='gray', lw=0.4, ls=':', alpha=0.5)
        for ev_t, ev_c, _ in event_marks: ax.axvline(ev_t, color=ev_c, lw=0.6, ls='--', alpha=0.4)
        ax.axhline(0, color='k', lw=0.4); ax.grid(True, alpha=0.15)

    axes[8].grid(True, alpha=0.15); axes[8].axhline(0, color='k', lw=0.4)
    for p in pm: axes[8].axvline(p, color='gray', lw=0.4, ls=':', alpha=0.5)
    for ev_t, ev_c, _ in event_marks: axes[8].axvline(ev_t, color=ev_c, lw=0.6, ls='--', alpha=0.4)

    axes[0].set_ylabel('SPV (deg/s)', fontsize=9); axes[0].legend(fontsize=8)
    axes[0].set_title('SPV sinusoidally modulated at rotation period', fontsize=9)

    amp_ref = G0 * np.sin(np.radians(TILTS_DEG[-1]))
    axes[1].set_ylabel('g_est[x]\ninteraural (m/s²)', fontsize=9); axes[1].legend(fontsize=8)
    axes[1].set_title(f'g_est interaural — oscillates ±G₀ sin(α); 90° → ±{amp_ref:.1f} m/s²', fontsize=9)
    axes[1].set_ylim(-12, 12)
    axes[2].set_ylabel('g_est[y]\nvertical (m/s²)', fontsize=9); axes[2].legend(fontsize=8)
    axes[2].set_title(f'g_est vertical — should stay near G₀ ≈ {G0:.2f} m/s² (deviation = somatogravic illusion)', fontsize=9)
    axes[3].set_ylabel('g_est[z]\nforward (m/s²)', fontsize=9); axes[3].legend(fontsize=8)
    axes[3].set_title('g_est forward — should oscillate too (orthogonal to interaural at 90°)', fontsize=9)

    axes[4].set_ylabel('v_lin[x]\nrightward (cm/s)', fontsize=9); axes[4].legend(fontsize=8, loc='upper left')
    axes[4].set_title('v_lin (left axis, solid) and a_est = gia − g_est (right axis, dashed) — HE integrates a_est into v_lin', fontsize=9)
    axes[5].set_ylabel('v_lin[y]\nupward (cm/s)', fontsize=9); axes[5].legend(fontsize=8, loc='upper left')
    axes[6].set_ylabel('v_lin[z]\nforward (cm/s)', fontsize=9); axes[6].legend(fontsize=8, loc='upper left')
    for axis_idx, row in [(0, 4), (1, 5), (2, 6)]:
        ax_r = a_est_axes[axis_idx]
        ax_r.set_ylabel(f'a_est[{["x","y","z"][axis_idx]}]\n(m/s²)', fontsize=8, color='gray')
        ax_r.tick_params(axis='y', labelcolor='gray', labelsize=7)
        ax_r.axhline(0, color='gray', lw=0.3, alpha=0.5)

    axes[7].set_ylabel('VS net yaw (deg/s)', fontsize=9); axes[7].legend(fontsize=8)
    axes[7].set_title('Velocity Storage modulated by gravity dumping (K_gd)', fontsize=9)

    axes[8].set_ylabel('Head yaw vel (°/s)', fontsize=9, color='k')
    axes[8].set_xlabel('Time (s)', fontsize=9)
    axes[8].set_title('Stimulus: constant body-yaw rotation (solid) + initial roll tilt per condition (dashed, right)',
                      fontsize=9)
    lines_l, labels_l = axes[8].get_legend_handles_labels()
    lines_r, labels_r = ax_stim_b.get_legend_handles_labels()
    axes[8].legend(lines_l + lines_r, labels_l + labels_r, fontsize=7, loc='center right', ncol=2)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'gravity_ovar', show=show)
    return utils.fig_meta(path, rp,
        title='OVAR — Off-Vertical Axis Rotation',
        description=f'{SPIN_VEL:.0f}°/s rotation, tilt angles {TILTS_DEG}°. '
                    'Replicates Laurens & Angelaki (2011) Fig 5.',
        expected='SPV sinusoidally modulated at rotation period. '
                 'Modulation amplitude ∝ sin(tilt). g_est oscillates ±G₀ sin(α). '
                 'A small residual v_lin DC bias along the rotation axis ("screw direction") '
                 'reflects the perceived sustained linear translation reported by subjects '
                 'during prolonged OVAR (Denise et al. 1988; Wood 2002).',
        citation='Laurens & Angelaki (2011) Exp Brain Res 210:407; '
                 'Denise, Darlot, Droulez, Cohen, Berthoz (1988) Exp Brain Res 67:629 — '
                 'perceived linear translation during OVAR; '
                 'Wood (2002) J Vest Res 12:223 — tilt-translation discrimination',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────
# 3. VOR Tilt Suppression  (Laurens Fig 6)
# ─────────────────────────────────────────────────────────────────────────────

def _tilt_suppression(show):
    ROT_VEL   = 60.0
    ROT_T     = 20.0     # rotation duration (s): identical for all conditions
    COAST_T   = 60.0     # post-tilt coast (s)
    TILTS_DEG = [0.0, 30.0, 60.0, 90.0]
    TILT_VEL  = 60.0     # roll tilt speed applied AFTER rotation stops

    # Protocol: rotate upright for ROT_T s (all conditions identical),
    # then tilt to θ°, then coast for COAST_T s.
    # t_rel = 0 at rotation stop.
    max_tilt_dur = max(TILTS_DEG) / TILT_VEL   # = 1.5 s (90° at 60°/s)
    t_total = ROT_T + max_tilt_dur + COAST_T
    t_arr   = np.arange(0.0, t_total, DT)
    T       = len(t_arr)
    t_rel   = t_arr - ROT_T   # aligned to rotation stop

    params = with_brain(PARAMS_DEFAULT, K_gd=K_GD)
    cfg    = SimConfig(warmup_s=0.0)
    colors = ['steelblue', '#2196a8', '#e08214', '#c62e2e']

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        f'VOR Tilt Suppression  (Laurens & Angelaki 2011, Fig 6)\n'
        f'Upright {ROT_VEL:.0f}°/s yaw for {ROT_T:.0f} s; tilt applied after stop;  '
        f'K_gd={K_GD}, tau_grav={TAU_GRAV}',
        fontsize=12, fontweight='bold')

    taus = {}

    hv_yaw_base = np.where(t_arr < ROT_T, ROT_VEL, 0.0)   # same for all conditions

    for ci, tilt_deg in enumerate(TILTS_DEG):
        tilt_dur = tilt_deg / TILT_VEL if tilt_deg > 0 else 0.0
        # Tilt starts right when rotation stops (t=ROT_T), ends at t=ROT_T+tilt_dur
        hv_roll = np.where(
            (t_arr >= ROT_T) & (t_arr < ROT_T + tilt_dur), TILT_VEL, 0.0)
        hv_3d   = np.stack([hv_yaw_base, np.zeros(T), hv_roll], axis=1)

        head_km = km.build_kinematics(t_arr, rot_vel=hv_3d)

        st = simulate(params, t_arr,
                      head=head_km,
                      scene_present_array=np.zeros(T),
                      target_present_array=np.zeros(T),
                      sim_config=cfg, return_states=True)

        eye_pos = (np.array(st.plant[:, 0]) + np.array(st.plant[:, 3])) / 2.0
        eye_vel = np.gradient(eye_pos, DT)
        spv     = extract_spv_states(st, t_arr)[:, 0]
        g_est_y = np.array(st.brain[:, _IDX_GRAV])[:, 0]

        # Fit TC starting after tilt completes
        fit_start = tilt_dur
        fit_end   = tilt_dur + COAST_T
        tau, t_fit, y_fit = fit_tc(t_rel, spv, fit_start, fit_end)
        taus[tilt_deg] = tau

        col = colors[ci]
        upright_sfx = ' (upright)' if tilt_deg == 0.0 else ''
        lbl = f'{tilt_deg:.0f}° roll{upright_sfx}' + (f'  τ={tau:.1f} s' if tau else '')

        axes[0].plot(t_rel, spv,     color=col, lw=1.2, label=lbl)
        axes[1].plot(t_rel, g_est_y, color=col, lw=1.2, label=f'{tilt_deg:.0f}°{upright_sfx}')
        if t_fit is not None:
            axes[0].plot(t_fit, y_fit, color=col, lw=2.5, ls=':', alpha=0.9)

        # Stimulus panel: 3D head velocity — yaw (shared) and roll (per condition)
        if ci == 0:
            axes[2].plot(t_rel, hv_yaw_base, color='#333333', lw=1.5, ls='--',
                         label='Yaw vel (all cond.)')
        axes[2].plot(t_rel, hv_roll, color=col, lw=1.5, ls='-',
                     label=f'Roll vel {tilt_deg:.0f}°')

    axes[2].set_ylabel('Head velocity (°/s)', fontsize=8)
    axes[2].set_title('Stimulus: head velocity 3D — yaw (dashed, shared); roll (solid, per condition)',
                      fontsize=9)
    axes[2].legend(fontsize=7, ncol=2, loc='upper left')

    xlim = (-ROT_T - 2.0, max_tilt_dur + COAST_T + 2.0)
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.axvline(0.0, color='gray', lw=1.0, ls='-',  label='rotation stop')
        ax.axhline(0.0, color='k',   lw=0.4)
        ax.grid(True, alpha=0.15)

    axes[0].set_ylabel('SPV (°/s)', fontsize=9); axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_title('Post-rotatory SPV: all conditions identical during rotation; '
                      'TC shortened by tilt after stop', fontsize=9)

    axes[1].set_ylabel('g_est[0] interaural/right (m/s²)', fontsize=9)
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].set_ylim(-12, 12)
    axes[1].set_title('Gravity estimate (interaural): 0 during upright rotation; '
                      'steps to +G0·sin(θ) after tilt', fontsize=9)

    axes[2].set_xlabel('Time relative to rotation stop (s)', fontsize=9)

    # Inset: TC vs tilt — bottom-right to avoid overlap with data and legend
    ax_ins = axes[0].inset_axes([0.72, 0.05, 0.25, 0.38])
    valid = [(α, τ) for α, τ in taus.items() if τ is not None]
    if valid:
        alphas, tau_vals = zip(*valid)
        ax_ins.plot(alphas, tau_vals, 'o-', color='k', lw=1.5, ms=5)
        ax_ins.set_xlabel('Tilt (°)', fontsize=7); ax_ins.set_ylabel('TC (s)', fontsize=7)
        ax_ins.tick_params(labelsize=6); ax_ins.set_title('TC vs tilt', fontsize=7)
        ax_ins.grid(True, alpha=0.2)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'gravity_tilt_suppression', show=show)
    return utils.fig_meta(path, rp,
        title='VOR Tilt Suppression',
        description=f'Upright {ROT_VEL:.0f}°/s yaw; tilt {TILTS_DEG}° applied after stop. '
                    'Replicates Laurens & Angelaki (2011) Fig 6.',
        expected='Per-rotatory identical. TC decreases with post-stop tilt. '
                 '0°: TC ≈ τ_vs. 90°: TC ≈ τ_canal.',
        citation='Laurens & Angelaki (2011) Exp Brain Res',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────
# 4. Somatogravic OCR — frequency dependence of lateral translation
# ─────────────────────────────────────────────────────────────────────────────

def _somatogravic_frequency(show):
    """Sinusoidal left-right translation; torsion amplitude drops above ~0.1 Hz.

    Physics: GIA in head frame = gravity + lateral_acceleration.
    The gravity estimator low-pass filters GIA with TC = 1/K_grav ≈ 1.7 s.
    At low f: g_est tracks lateral GIA → perceived tilt → OCR.
    At high f: GIA fluctuates too fast for g_est → no perceived tilt → no OCR.

    Theoretical corner frequency: fc = K_grav / (2π) ≈ 0.095 Hz.
    """
    FREQS_HZ = [0.03, 0.07, 0.15, 0.3, 0.7, 1.5]
    A_ACCEL  = 2.0       # m/s² peak lateral acceleration (≈ 0.2g)
    N_CYCLES = 6
    SETTLE_S = 5.0 * TAU_GRAV

    # Note: saccades left ON. Disabling them (g_burst=0) caused NaN at long durations
    # (≥120 s) due to a numerical interaction between un-corrected eye drift and the
    # saccade generator's accumulator state.  Quick phases now appear as small
    # transients on top of the OCR slow-phase response.
    params = with_brain(PARAMS_DEFAULT, g_ocr=G_OCR)

    cmap   = plt.get_cmap('coolwarm')
    colors = [cmap(i / (len(FREQS_HZ) - 1)) for i in range(len(FREQS_HZ))]

    amp_model  = []
    SHOW_FREQS = [0.03, 0.15, 1.5]
    trace_data = {}

    for i, freq in enumerate(FREQS_HZ):
        omega   = 2.0 * np.pi * freq
        total   = SETTLE_S + N_CYCLES / freq
        t       = np.arange(0.0, total, DT)
        T       = len(t)

        # Lateral (interaural = world x = rightward) translation.
        # World frame: x=right, y=up, z=fwd  → lateral acceleration is component 0.
        a_lat   = A_ACCEL * np.sin(omega * t)
        lin_acc = np.stack([a_lat, np.zeros(T), np.zeros(T)], axis=1)

        st = simulate(params, t,
                      head=km.build_kinematics(t, lin_acc=lin_acc),
                      scene_present_array=np.zeros(T),
                      target_present_array=np.zeros(T),
                      return_states=True)

        eye_roll = (np.array(st.plant[:, 2]) + np.array(st.plant[:, 5])) / 2.0
        g_est    = np.array(st.brain[:, _IDX_GRAV])

        # Align display start to a complete cycle boundary so a_lat starts at 0.
        n_full_cycles   = int(total * freq)
        display_cycle   = max(0, n_full_cycles - 3)
        i_ss            = int(display_cycle / freq / DT)
        peak = float(np.max(np.abs(eye_roll[i_ss:])))
        amp_model.append(peak)

        if freq in SHOW_FREQS:
            trace_data[freq] = {
                't': t, 'a_lat': a_lat, 'eye_roll': eye_roll, 'g_est_0': g_est[:, 0],
                'i_ss': i_ss,
            }

    fc          = 1.0 / (2.0 * np.pi * TAU_GRAV)
    f_theory    = np.logspace(np.log10(0.01), np.log10(5.0), 200)
    gain_theory = 1.0 / np.sqrt(1.0 + (f_theory / fc) ** 2)
    torsion_dc  = G_OCR * A_ACCEL
    amp_theory  = gain_theory * torsion_dc

    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        f'Somatogravic OCR — Frequency Dependence of Lateral Translation\n'
        f'Constant {A_ACCEL:.0f} m/s² peak acceleration;  g_ocr={G_OCR},  '
        f'tau_grav={TAU_GRAV}  (corner freq fc ≈ {fc:.3f} Hz)',
        fontsize=12, fontweight='bold')

    gs        = fig.add_gridspec(3, len(SHOW_FREQS), hspace=0.45, wspace=0.3,
                                 height_ratios=[1, 1, 1.3])
    axes_acc  = [fig.add_subplot(gs[0, j]) for j in range(len(SHOW_FREQS))]
    axes_gest = [fig.add_subplot(gs[1, j]) for j in range(len(SHOW_FREQS))]
    ax_bode   = fig.add_subplot(gs[2, :])

    show_colors = {f: cmap(FREQS_HZ.index(f) / (len(FREQS_HZ) - 1))
                   for f in SHOW_FREQS if f in FREQS_HZ}

    for j, freq in enumerate(SHOW_FREQS):
        if freq not in trace_data:
            continue
        d    = trace_data[freq]
        t    = d['t']
        i_ss = d['i_ss']
        col  = show_colors[freq]
        t_show = t[i_ss:] - t[i_ss]

        ax_acc = axes_acc[j]
        ax_acc.plot(t_show, d['a_lat'][i_ss:], color=col, lw=1.2)
        ax_acc.axhline(0, color='k', lw=0.4)
        ax_acc.set_title(f'{freq:.2f} Hz', fontsize=9, fontweight='bold')
        if j == 0: ax_acc.set_ylabel('Lateral accel (m/s²)', fontsize=8)
        ax_acc.grid(True, alpha=0.15)

        # Middle row: g_est[0] (right/interaural, m/s²) and eye torsion (right, deg) on twin axes.
        ax_ge = axes_gest[j]
        g_data = d['g_est_0'][i_ss:]
        eye_data = d['eye_roll'][i_ss:]
        g_amp   = max(float(np.max(np.abs(g_data))),   1e-3)
        eye_amp = max(float(np.max(np.abs(eye_data))), 1e-3)

        ax_ge.plot(t_show, g_data, color=col, lw=1.8)
        ax_ge.axhline(0, color='k', lw=0.4)
        ax_ge.set_ylabel('g_est[0] (m/s²)', fontsize=7, color=col)
        ax_ge.tick_params(axis='y', labelcolor=col, labelsize=7)
        ax_ge.grid(True, alpha=0.15)
        ax_ge.set_ylim(-1.3 * g_amp, 1.3 * g_amp)
        # Annotate peak amplitude on each panel for clarity
        ax_ge.text(0.02, 0.96, f'g_est peak ≈ {g_amp:.3f} m/s²',
                   transform=ax_ge.transAxes, fontsize=7, va='top', color=col,
                   bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        if j == 1:
            ax_ge.set_title('g_est[0] interaural/right (left axis, m/s²)  |  '
                            'eye torsion (right axis, deg)', fontsize=8)

        ax_eye2 = ax_ge.twinx()
        ax_eye2.plot(t_show, eye_data, color='darkorange', lw=1.4, ls='-.')
        ax_eye2.set_ylabel('Eye torsion (deg)', fontsize=7, color='darkorange')
        ax_eye2.tick_params(axis='y', labelcolor='darkorange', labelsize=7)
        ax_eye2.set_ylim(-1.3 * eye_amp, 1.3 * eye_amp)
        ax_eye2.text(0.98, 0.96, f'torsion peak ≈ {eye_amp:.3f}°',
                     transform=ax_eye2.transAxes, fontsize=7, va='top', ha='right',
                     color='darkorange',
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    ax_bode.loglog(f_theory, amp_theory, color='gray', lw=2.0, ls='--',
                   label=f'LP theory: fc={fc:.3f} Hz  (TC=tau_grav={TAU_GRAV:.1f} s)')
    ax_bode.axvline(fc, color='gray', lw=0.8, ls=':', alpha=0.7)
    ax_bode.text(fc * 1.15, amp_theory[0] * 0.6, f'fc={fc:.3f} Hz', fontsize=8, color='gray')

    for i, (freq, amp) in enumerate(zip(FREQS_HZ, amp_model)):
        col = cmap(i / (len(FREQS_HZ) - 1))
        ax_bode.scatter([freq], [amp], color=col, s=80, zorder=5)
        ax_bode.annotate(f'{freq:.2f} Hz\n{amp:.3f}°', (freq, amp),
                         fontsize=7, xytext=(5, 5), textcoords='offset points', color=col)

    ax_bode.set_xlabel('Frequency (Hz)', fontsize=9)
    ax_bode.set_ylabel('Peak torsion amplitude (deg)', fontsize=9)
    ax_bode.set_title(
        f'Somatogravic OCR amplitude vs frequency  '
        f'(A={A_ACCEL:.0f} m/s² lateral accel;  low-f → tilt percept → OCR; high-f → no tilt percept)',
        fontsize=9)
    ax_bode.legend(fontsize=8); ax_bode.grid(True, which='both', alpha=0.2)
    ax_bode.set_xlim(0.01, 5.0)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path, rp = utils.save_fig(fig, 'gravity_somatogravic_freq', show=show)
    return utils.fig_meta(path, rp,
        title='Somatogravic OCR — Frequency Dependence',
        description=f'Sinusoidal lateral translation at {FREQS_HZ} Hz, '
                    f'{A_ACCEL} m/s² peak. OCR via gravity estimator LP filter.',
        expected=f'Corner frequency fc = K_grav/(2π) ≈ {fc:.3f} Hz. '
                 f'OCR amplitude ∝ 1/√(1+(f/fc)²). '
                 f'Low f → peak ≈ {torsion_dc:.2f}°; high f → near 0°.',
        citation='Mayne (1974); Laurens & Angelaki (2011)',
        fig_type='behavior')


# ─────────────────────────────────────────────────────────────────────────────

def _canal_vel_3d(states):
    """Extract bilateral canal velocity estimate (3,) for each timestep.  (T, 3) deg/s."""
    import scipy.special as sp
    x_c  = np.array(states.sensory[:, _IDX_C])          # (T, 12)
    x2   = x_c[:, N_CANALS:]                             # inertia states (T, 6)
    k, f = float(_SOFTNESS), float(FLOOR)
    # Same softplus nonlinearity as analysis.extract_canal
    y_c  = -f + sp.log1p(np.exp(k * (x2 + f))) / k + sp.log1p(np.exp(k * (x2 - f))) / k
    pinv = np.array(CANAL_PINV)                          # (3, 12)
    return (pinv @ y_c.T).T                              # (T, 3)


# ─────────────────────────────────────────────────────────────────────────────
# OCR signal cascade — debug torsion drift
# ─────────────────────────────────────────────────────────────────────────────

def _ocr_cascade(show):
    """OCR signal cascade — 30° left-ear-down tilt, 30 s hold.

    4-column comparison: scene+target on / target only / scene only / dark.
    Shows how visual conditions affect VS torsion and downstream cascade.

    Rows (top→bottom):
        1.  Head roll velocity (stimulus — same all conditions)
        2.  OCR = -g_ocr × g_est[0] (deg)
        3.  VS net torsion (deg/s)
        4.  −VS → NI input (deg/s)
        5.  Total NI input (numerical: d/dt NI + (1/τ_i)·NI, deg/s)
        6.  NI net torsion (deg)
        7.  Motor cmd torsion = NI + OCR (deg)
        8.  L eye torsion (deg)
        9.  L eye yaw (deg)
        10. L eye pitch (deg)
        11. SG z_sac (saccade accumulator)
    """
    TILT_DEG = 30.0
    TILT_VEL = 20.0
    HOLD_T   = 30.0
    tilt_dur = TILT_DEG / TILT_VEL

    params  = with_brain(PARAMS_DEFAULT, g_ocr=G_OCR)
    t_np    = np.arange(0.0, tilt_dur + HOLD_T, DT)
    T       = len(t_np)
    hv_roll = np.where(t_np < tilt_dur, TILT_VEL, 0.0)
    head_km = km.build_kinematics(t_np, rot_vel=_pad3(hv_roll, 'roll'))
    t_rel   = t_np - tilt_dur
    tau_i   = float(params.brain.tau_i)

    COLUMNS = [
        ('Roll (tilt)',   2, km.build_kinematics(t_np, rot_vel=_pad3(hv_roll, 'roll'))),
        ('Yaw (horiz)',   0, km.build_kinematics(t_np, rot_vel=_pad3(hv_roll, 'yaw'))),
        ('Pitch (vert)',  1, km.build_kinematics(t_np, rot_vel=_pad3(hv_roll, 'pitch'))),
    ]
    COL_COLORS = ['#2166ac', '#e08214', '#c51b7d']

    def _extract(st, axis):
        grav       = np.array(st.brain[:, _IDX_GRAV])
        ocr_sig    = -G_OCR * grav[:, 0]              # always torsion (g_est x-component)
        vs_a       = vs_net(st)[:, axis]
        ni_a       = ni_net(st)[:, axis]
        spv_a      = extract_spv_states(st, t_np)[:, axis]
        x2         = np.array(st.sensory[:, _IDX_C])[:, 6:12]
        canal_ft_a = float(params.brain.g_vor) * (np.array(CANAL_PINV) @ x2.T)[axis, :]
        plant_arr = np.array(st.plant)
        return dict(
            ocr=ocr_sig, vs_neg=-vs_a, ni=ni_a,
            eye_L=plant_arr[:, axis],
            eye_R=plant_arr[:, 3 + axis],
            spv_tor=spv_a, canal_ft=canal_ft_a,
        )

    results = []
    for _lbl, axis, head in COLUMNS:
        st = simulate(params, t_np,
                      head=head,
                      scene_present_array=np.ones(T),
                      target_present_array=np.ones(T),
                      sim_config=SimConfig(warmup_s=0.0),
                      return_states=True)
        results.append(_extract(st, axis))

    ocr_ss = float(results[0]['ocr'][-1])

    PANELS = [
        ('_hv',    'Head roll vel (deg/s)',             '#555555'),
        ('ocr',    'OCR = -g_ocr×g_est[0] (deg)',       '#d6604d'),
        ('vs_neg', '-VS → NI input (deg/s)',           '#92c5de'),
        ('ni',     'NI net torsion (deg)',             '#4dac26'),
        ('eye_LR', 'L (blue) vs R (orange) eye torsion (deg)', '#1a9641'),
        ('spv_tor','Eye SPV torsion (deg/s)',          '#e08214'),
    ]
    N_ROWS = len(PANELS)
    N_COLS = len(COLUMNS)

    fig, axes = plt.subplots(N_ROWS, N_COLS,
                             figsize=(4.5 * N_COLS, 2.1 * N_ROWS),
                             sharex=True, squeeze=False)
    fig.suptitle(
        f'OCR cascade — {TILT_DEG:.0f}° tilt, {HOLD_T:.0f} s hold — roll vs. yaw vs. pitch\n'
        f'g_ocr={G_OCR:.2f}  |  roll SS OCR ≈ {ocr_ss:.2f}°',
        fontsize=11, fontweight='bold')

    for col_idx, ((col_label, axis, _head), col_color, res) in enumerate(
            zip(COLUMNS, COL_COLORS, results)):
        axes[0, col_idx].set_title(col_label, fontsize=10, fontweight='bold', color=col_color)

        for row_idx, (key, ylabel, row_color) in enumerate(PANELS):
            ax = axes[row_idx, col_idx]
            if key == 'eye_LR':
                # Plot L (blue) and R (orange) eye torsion to expose any vergence
                ax.plot(t_rel, res['eye_L'], color='#1f77b4', lw=0.9, label='L' if col_idx == 0 else None)
                ax.plot(t_rel, res['eye_R'], color='#ff7f0e', lw=0.9, label='R' if col_idx == 0 else None)
                if col_idx == 0:
                    ax.legend(fontsize=5, loc='upper right')
                sig = res['eye_L']   # for the trailing annotation
            else:
                sig = hv_roll if key == '_hv' else res[key]
                ax.plot(t_rel, sig, color=col_color, lw=0.9)
            if key == 'spv_tor':
                ax.plot(t_rel, res['vs_neg'], color='#2166ac', lw=0.7, ls='--', alpha=0.8,
                        label='-VS' if col_idx == 0 else None)
                ax.plot(t_rel, -res['canal_ft'], color='#d6604d', lw=0.7, ls=':', alpha=0.8,
                        label='-canal FT' if col_idx == 0 else None)
                if col_idx == 0:
                    ax.legend(fontsize=5, loc='upper right')
            ax.axvline(0.0, color='gray', lw=0.6, ls='--')
            ax.axhline(0.0, color='k', lw=0.3, alpha=0.4)
            ax.grid(True, alpha=0.12)
            ax.tick_params(axis='y', labelsize=5)
            ax.annotate(f'{float(sig[-1]):.2f}', xy=(t_rel[-1], sig[-1]),
                        fontsize=5.5, ha='right', va='bottom', color=col_color)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=6.5, color=row_color)

        # Mark OCR target on eye_L (row 4) for roll column only
        if col_idx == 0:
            axes[4, 0].axhline(ocr_ss, color='tomato', lw=0.7, ls=':', alpha=0.8,
                               label=f'{ocr_ss:.2f}°')
            axes[4, 0].legend(fontsize=5.5, loc='upper right')

        axes[N_ROWS - 1, col_idx].set_xlabel('Time rel. hold onset (s)', fontsize=8)

    # Clamp SPV y-axis so slow-phase structure is visible.
    SPV_LIM = 15.0
    for row_idx, (key, *_) in enumerate(PANELS):
        if key == 'spv_tor':
            for col_idx in range(N_COLS):
                axes[row_idx, col_idx].set_ylim(-SPV_LIM, SPV_LIM)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'gravity_ocr_cascade', show=show)
    return utils.fig_meta(
        path, rp,
        title='OCR cascade (4-condition comparison)',
        description=(f'{TILT_DEG}° tilt — VS, NI torsion cascade + SPV for '
                     f'scene+target / target only / scene only / dark.'),
        expected=(f'Torsion stabilises at ≈{ocr_ss:.2f}°. '
                  f'SPV row: -VS (dashed) and -canal FT (dotted) should bracket the measured SPV. '
                  f'Yaw/pitch near 0 (pure roll tilt).'),
        citation='Debug diagnostic',
        fig_type='cascade')


_BENCH_MAP = {
    'ocr':            _ocr,
    'ocr_cascade':    _ocr_cascade,
    'ovar':           _ovar,
    'tilt':           _tilt_suppression,
    'somatogravic':   _somatogravic_frequency,
}


def run(show=False, only=None):
    names = list(_BENCH_MAP.keys()) if only is None else [only]
    print(f'\n=== Gravity Estimator ({", ".join(names)}) ===')
    figs = []
    for i, name in enumerate(names, 1):
        print(f'  {i}/{len(names)}  {name} …')
        figs.append(_BENCH_MAP[name](show))
    return figs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--bench', choices=list(_BENCH_MAP.keys()), default=None,
                    help='Run only this benchmark (default: all)')
    args = ap.parse_args()
    run(show=args.show, only=args.bench)
    if args.show:
        # save_fig uses non-blocking show; final blocking show keeps windows open.
        import matplotlib.pyplot as _plt
        _plt.show()
