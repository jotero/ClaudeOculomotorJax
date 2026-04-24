"""VOR demo suite — cascade diagnostics with burst visibility.

Figures
───────
    vor_cascade.png  — 9-panel VOR signal cascade in dark (+ burst panel)
    okr_cascade.png  — 7-panel OKN signal cascade (+ burst panel)
    vvor.png         — 6-panel VVOR dark vs lit

Usage
-----
    python -X utf8 scripts/demo_vor.py
"""

import sys
import os

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
SHOW = '--show' in sys.argv
if not SHOW:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, with_sensory, simulate,
    _IDX_C, _IDX_VS, _IDX_NI, _IDX_VIS, _IDX_SG,
)
from oculomotor.sim import kinematics as km
from oculomotor.models.sensory_models.sensory_model import N_CANALS, FLOOR, _SOFTNESS, PINV_SENS
from oculomotor.models.sensory_models.sensory_model import C_slip, C_pos
from oculomotor.analysis import ax_fmt, extract_burst, vs_net, ni_net

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
THETA = with_sensory(PARAMS_DEFAULT, sigma_canal=0.5, sigma_pos=0.2, sigma_vel=0.2)

_C = {
    'head':   '#555555',
    'eye':    '#2166ac',
    'no_vs':  '#d6604d',
    'scene':  '#1b7837',
    'burst':  '#b2182b',
    'vs':     '#35978f',
}


# ── Utilities ──────────────────────────────────────────────────────────────────

def _build_vor_head(v_deg_s=30.0, rest_dur=5.0, rotate_dur=15.0, coast_dur=60.0,
                    sample_rate=100.0):
    """Build head KinematicTrajectory: rest → constant yaw → coast."""
    total = rest_dur + rotate_dur + coast_dur
    t     = km.make_time(total, dt=1.0 / sample_rate)
    vel1d = np.where((t >= rest_dur) & (t < rest_dur + rotate_dur),
                     v_deg_s, 0.0).astype(np.float32)
    return t, km.build_kinematics(t, rot_vel=km._pad3(vel1d, 'yaw'))


def _extract_signals(theta, t_array, head, states):
    """Reconstruct intermediate signals from saved state trajectory."""
    from scipy.special import softplus as sp_softplus
    hv   = np.array(head.rot_vel[:, 0])   # yaw only
    x_c  = np.array(states.sensory[:, _IDX_C])
    x_vs = np.array(states.brain[:, _IDX_VS])
    x_ni = np.array(states.brain[:, _IDX_NI])
    x_p  = np.array(states.plant[:, :3])   # (T, 3) left eye

    nc  = N_CANALS
    x1  = x_c[:, :nc]
    x2  = x_c[:, nc:]
    k   = _SOFTNESS
    f   = float(FLOOR)
    y_c = -f + sp_softplus(k * (x2 + f)) / k + sp_softplus(k * (x2 - f)) / k

    pinv      = np.array(PINV_SENS)
    u_canal   = (pinv @ y_c.T).T
    x_vs_net  = x_vs[:, :3] - x_vs[:, 3:6]   # x_L − x_R  (net VS signal)
    w_est     = x_vs_net + u_canal
    u_p       = x_ni[:, :3] - x_ni[:, 3:6] - theta.brain.tau_p * w_est

    eye_pos = x_p[:, 0]
    dt      = float(t_array[1] - t_array[0])
    eye_vel = np.gradient(eye_pos, dt)

    return dict(
        head_vel=hv, x1_c0=x1[:, 0], x1_c1=x1[:, 1],
        x2_c0=x2[:, 0], x2_c1=x2[:, 1],
        y_c0=y_c[:, 0], y_c1=y_c[:, 1],
        u_canal=u_canal[:, 0], x_vs=x_vs_net[:, 0],
        w_est=w_est[:, 0], x_ni=u_p[:, 0],
        u_p=u_p[:, 0], eye_pos=eye_pos, eye_vel=eye_vel,
    )


# ── Plot 1: VOR in the dark — 9-panel signal cascade ──────────────────────────

def demo_vor_cascade():
    """9-panel cascade: head → canal → VS → NI → plant → burst."""
    rest_dur   = 5.0
    rotate_dur = 15.0
    coast_dur  = 60.0
    t_np, head = _build_vor_head(v_deg_s=30.0, rest_dur=rest_dur,
                                  rotate_dur=rotate_dur, coast_dur=coast_dur)
    t_np  = np.array(t_np)
    max_s = int((t_np[-1] - t_np[0]) / 0.001) + 500
    T     = len(t_np)

    states    = simulate(THETA, t_np, head=head,
                         target_present_array=jnp.zeros(T),
                         max_steps=max_s, return_states=True,
                         key=jax.random.PRNGKey(0))
    sigs      = _extract_signals(THETA, t_np, head, states)
    burst     = extract_burst(states, THETA)

    theta_no_vs = with_brain(THETA, tau_vs=0.1, K_vs=0.001)
    states_nv   = simulate(theta_no_vs, t_np, head=head,
                           target_present_array=jnp.zeros(T),
                           max_steps=max_s, return_states=True,
                           key=jax.random.PRNGKey(0))
    sigs_nv     = _extract_signals(theta_no_vs, t_np, head, states_nv)

    tau_eff = THETA.brain.tau_vs

    fig, axes = plt.subplots(10, 1, figsize=(24, 27), sharex=True)
    fig.suptitle(f'VOR in the dark — step rotation  ({rotate_dur:.0f} s @ 30 deg/s)\n'
                 f'Signal cascade: head → canal → VS → NI → plant → burst',
                 fontsize=11)

    vline_kw = dict(color='k', lw=0.8, ls='--', alpha=0.4)
    for ax in axes:
        ax.axvline(rest_dur,              **vline_kw)
        ax.axvline(rest_dur + rotate_dur, **vline_kw)
        ax_fmt(ax)

    axes[0].plot(t_np, sigs['head_vel'], color=_C['head'], lw=1.5)
    axes[0].set_ylabel('Head vel\n(deg/s)')
    axes[0].set_title('Input')

    axes[1].plot(t_np, sigs['x1_c0'], color='steelblue', lw=1.2, label='RHC (c0)')
    axes[1].plot(t_np, sigs['x1_c1'], color='tomato',    lw=1.2, label='LHC (c1)')
    axes[1].set_ylabel('x1 (deg/s)')
    axes[1].set_title(f'Canal — adaptation LP  (tau_c = {THETA.sensory.tau_c} s)')
    axes[1].legend(fontsize=8, loc='upper right')

    axes[2].plot(t_np, sigs['x2_c0'], color='steelblue', lw=1.2, label='RHC (c0)')
    axes[2].plot(t_np, sigs['x2_c1'], color='tomato',    lw=1.2, label='LHC (c1)')
    axes[2].set_ylabel('x2 (deg/s)')
    axes[2].set_title(f'Canal — inertia LP / bandpass  (tau_s = {THETA.sensory.tau_s} s)')
    axes[2].legend(fontsize=8, loc='upper right')

    axes[3].plot(t_np, sigs['y_c0'] + FLOOR, color='steelblue', lw=1.5, label='RHC (c0)')
    axes[3].plot(t_np, sigs['y_c1'] + FLOOR, color='tomato',    lw=1.5, label='LHC (c1)')
    axes[3].axhline(FLOOR, color='gray', lw=1.0, ls=':', label=f'Rest = {FLOOR:.0f}')
    axes[3].set_ylabel('Afferent\n(spk/s)')
    axes[3].set_title('Canal afferent output  (nonlinear push-pull)')
    axes[3].legend(fontsize=8)

    axes[4].plot(t_np, sigs['u_canal'],  color='purple', lw=1.5, label='u_canal  (PINV @ y)')
    axes[4].plot(t_np, sigs['head_vel'], color='gray',   lw=1.0, ls=':', alpha=0.6, label='Head vel')
    axes[4].set_ylabel('deg/s')
    axes[4].set_title('Head vel estimate  (PINV_SENS @ y_canals)')
    axes[4].legend(fontsize=8)

    axes[5].plot(t_np, sigs['x_vs'],    color='steelblue', lw=1.5,
                 label=f'with VS  (tau_vs = {tau_eff:.0f} s)')
    axes[5].plot(t_np, sigs_nv['x_vs'], color=_C['no_vs'], lw=1.5, ls='--', label='no VS')
    axes[5].set_ylabel('VS state\n(deg/s)')
    axes[5].set_title('Velocity storage')
    axes[5].legend(fontsize=8)

    axes[6].plot(t_np, sigs['w_est'], color='seagreen', lw=1.5, label='w_est (deg/s)')
    axes[6].plot(t_np, sigs['x_ni'],  color='seagreen', lw=1.5, ls='--', label='x_ni (deg)')
    axes[6].set_ylabel('deg/s | deg')
    axes[6].set_title(f'VS output w_est and NI state  (tau_i = {THETA.brain.tau_i} s)')
    axes[6].legend(fontsize=8)

    axes[7].plot(t_np, sigs['eye_vel'],    color='steelblue', lw=1.5, label='with VS')
    axes[7].plot(t_np, sigs_nv['eye_vel'], color=_C['no_vs'], lw=1.5, ls='--', label='no VS')
    axes[7].plot(t_np, -sigs['head_vel'],  color='gray',      lw=1.0, ls=':', alpha=0.6,
                 label='-head vel  (ideal VOR)')
    axes[7].set_ylabel('Eye vel\n(deg/s)')
    axes[7].set_title('Plant output  (eye velocity)')
    axes[7].legend(fontsize=8)

    orbital_limit = THETA.plant.orbital_limit
    axes[8].plot(t_np, sigs['eye_pos'], color='steelblue', lw=1.5, label='eye pos')
    axes[8].axhline( orbital_limit, color='tomato', lw=1.0, ls='--', alpha=0.8,
                     label=f'±orbital limit ({orbital_limit:.0f}°)')
    axes[8].axhline(-orbital_limit, color='tomato', lw=1.0, ls='--', alpha=0.8)
    axes[8].set_ylabel('Eye pos\n(deg)')
    axes[8].set_title('Eye position')
    axes[8].legend(fontsize=8)

    axes[9].plot(t_np, burst[:, 0], color=_C['burst'], lw=1.2)
    axes[9].set_ylabel('Burst\n(deg/s)')
    axes[9].set_xlabel('Time (s)')
    axes[9].set_title('Saccade burst u_burst')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'vor_cascade.png')
    fig.savefig(path, dpi=150)
    if SHOW: plt.show()
    else:    plt.close(fig)
    print(f'  Saved {path}')


# ── Plot 2: OKR cascade ────────────────────────────────────────────────────────

def demo_okr_cascade():
    """8-panel OKN cascade: scene → slip → VS → eye velocity → burst."""
    sr        = 200.0
    on_dur    = 30.0
    off_dur   = 40.0
    total_dur = on_dur + off_dur
    scene_vel = 30.0

    theta_okn    = with_brain(THETA, g_burst=600.0)
    theta_no_vis = with_brain(THETA, K_vis=0.0, g_vis=0.0)

    t_arr = km.make_time(total_dur, dt=1.0 / sr)
    scene = km.scene_rotation_step(scene_vel, on_dur=on_dur, total_dur=total_dur,
                                    dt=1.0 / sr, axis='yaw')
    # scene_present: 1 while scene moves, 0 after scene off (pure OKAN at tau_vs)
    scene_present = jnp.where(jnp.array(t_arr) < on_dur, 1.0, 0.0)
    max_s = int(total_dur / 0.001) + 500
    T_okn = len(t_arr)

    states = simulate(theta_okn, t_arr, scene=scene,
                      scene_present_array=scene_present,
                      target_present_array=jnp.zeros(T_okn),
                      max_steps=max_s, return_states=True,
                      key=jax.random.PRNGKey(1))
    t_np   = np.array(t_arr)
    dt     = 1.0 / sr
    burst  = extract_burst(states, theta_okn)

    x_vis    = np.array(states.sensory[:, _IDX_VIS])
    x_vs_net = vs_net(states)
    x_ni     = ni_net(states)
    x_p      = np.array(states.plant[:, :3])

    w_scene_np   = np.where(t_np < on_dur, scene_vel, 0.0)
    eye_vel      = np.gradient(x_p[:, 0], dt)
    e_delayed    = (x_vis @ np.array(C_slip).T)[:, 0]
    u_vis_direct = THETA.brain.g_vis * e_delayed
    SPV_CLIP     = 80.0
    spv          = np.where(np.abs(eye_vel) < SPV_CLIP, eye_vel, np.nan)

    eye_no_vis = simulate(theta_no_vis, t_arr, scene=scene,
                          scene_present_array=scene_present,
                          target_present_array=jnp.zeros(T_okn),
                          max_steps=max_s, key=jax.random.PRNGKey(1))[:, 0]
    ev_no_vis  = np.gradient(np.array(eye_no_vis), dt)

    fig, axes = plt.subplots(8, 1, figsize=(12, 22), sharex=True)
    fig.suptitle(f'OKN nystagmus — scene on {on_dur:.0f} s then off  (OKAN)\n'
                 f'K_vis={THETA.brain.K_vis},  g_vis={THETA.brain.g_vis},  '
                 f'tau_vs={THETA.brain.tau_vs} s,  tau_vis={THETA.sensory.tau_vis} s',
                 fontsize=10)

    vline_kw = dict(color='k', lw=0.8, ls='--', alpha=0.4)
    for ax in axes:
        ax.axvline(on_dur, **vline_kw)
        ax_fmt(ax)

    axes[0].plot(t_np, w_scene_np, color=_C['scene'], lw=1.5)
    axes[0].set_ylabel('Scene vel\n(deg/s)')
    axes[0].set_title('Input: visual scene velocity')

    axes[1].plot(t_np, e_delayed, color='darkorange', lw=1.5,
                 label=f'e_delayed  (tau_vis={THETA.sensory.tau_vis} s)')
    axes[1].set_ylabel('Delayed slip\n(deg/s)')
    axes[1].set_title('Visual delay cascade output')
    axes[1].legend(fontsize=8)

    axes[2].plot(t_np, x_vs_net[:, 0], color='steelblue', lw=1.5,
                 label=f'x_vs  (tau_vs = {THETA.brain.tau_vs} s)')
    axes[2].set_ylabel('VS state\n(deg/s)')
    axes[2].set_title('Velocity storage')
    axes[2].legend(fontsize=8)

    axes[3].plot(t_np, u_vis_direct,    color=_C['scene'],  lw=1.5,
                 label=f'g_vis * e_delayed  (g_vis={THETA.brain.g_vis})')
    axes[3].plot(t_np, -x_vs_net[:, 0], color='steelblue', lw=1.5, ls='--',
                 label='-x_vs  (VS contribution)')
    axes[3].set_ylabel('Visual drive\n(deg/s)')
    axes[3].set_title('Visual contributions to eye-velocity command')
    axes[3].legend(fontsize=8)

    axes[4].plot(t_np, w_scene_np, color='gray',      lw=1.0, ls=':', alpha=0.5, label='Scene vel')
    axes[4].plot(t_np, ev_no_vis,  color=_C['no_vs'], lw=1.5, ls='--', label='No visual drive')
    axes[4].plot(t_np, eye_vel,    color='steelblue', lw=0.5, alpha=0.4, label='Eye vel (raw)')
    axes[4].plot(t_np, spv,        color='steelblue', lw=1.8, label='Eye vel (slow phase)')
    axes[4].set_ylim(-45, 45)
    axes[4].set_ylabel('Eye vel\n(deg/s)')
    axes[4].set_title('Eye velocity — OKN nystagmus + OKAN  [clipped]')
    axes[4].legend(fontsize=8)

    orbital_limit_okn = THETA.plant.orbital_limit
    axes[5].plot(t_np, x_p[:, 0], color=_C['eye'], lw=0.8, label='x_p (eye pos)')
    axes[5].axhline( orbital_limit_okn, color='tomato', lw=0.8, ls='--', alpha=0.7,
                     label=f'±orbital limit ({orbital_limit_okn:.0f}°)')
    axes[5].axhline(-orbital_limit_okn, color='tomato', lw=0.8, ls='--', alpha=0.7)
    axes[5].set_ylabel('Eye pos\n(deg)')
    axes[5].set_title('Eye position')
    axes[5].legend(fontsize=8)

    axes[6].plot(t_np, x_ni[:, 0], color='darkorchid', lw=1.0, label='x_ni')
    axes[6].plot(t_np, x_p[:, 0],  color=_C['eye'],    lw=0.8, ls='--', alpha=0.6, label='x_p')
    axes[6].set_ylabel('NI / pos\n(deg)')
    axes[6].set_title('Neural integrator state vs eye position')
    axes[6].legend(fontsize=8)

    axes[7].plot(t_np, burst[:, 0], color=_C['burst'], lw=1.0)
    axes[7].set_ylabel('Burst\n(deg/s)')
    axes[7].set_xlabel('Time (s)')
    axes[7].set_title('Saccade burst u_burst  (fast-phase resets)')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'okr_cascade.png')
    fig.savefig(path, dpi=150)
    if SHOW: plt.show()
    else:    plt.close(fig)
    print(f'  Saved {path}')


# ── Plot 3: VVOR ───────────────────────────────────────────────────────────────

def demo_vvor():
    """6-panel VVOR: dark vs lit across a rotation large enough to reach orbital limit."""
    sr         = 200.0
    v_head     = 30.0
    rotate_dur = 30.0
    coast_dur  = 20.0
    dt_vv      = 1.0 / sr

    head  = km.head_rotation_step(v_head, rotate_dur=rotate_dur,
                                   coast_dur=coast_dur, dt=dt_vv)
    t_vv  = head.t
    hv_vv = head.rot_vel[:, 0]
    T     = len(t_vv)
    max_sv = int((rotate_dur + coast_dur) / 0.001) + 500

    theta_dark = with_brain(THETA, K_vis=0.0, g_vis=0.0)
    theta_lit  = THETA

    states_dark = simulate(theta_dark, t_vv, head=head,
                           scene_present_array=jnp.zeros(T),
                           target_present_array=jnp.zeros(T),
                           max_steps=max_sv, return_states=True,
                           key=jax.random.PRNGKey(2))
    states_lit  = simulate(theta_lit, t_vv, head=head,
                           scene_present_array=jnp.ones(T),
                           target_present_array=jnp.ones(T),
                           max_steps=max_sv, return_states=True,
                           key=jax.random.PRNGKey(2))

    t_vv     = np.array(t_vv)
    head_pos = np.cumsum(hv_vv) * dt_vv

    eye_dark = np.array(states_dark.plant)[:, 0]
    eye_lit  = np.array(states_lit.plant)[:, 0]
    ev_dark  = np.gradient(eye_dark, dt_vv)
    ev_lit   = np.gradient(eye_lit,  dt_vv)

    x_vis_dark = np.array(states_dark.sensory[:, _IDX_VIS])
    x_vis_lit  = np.array(states_lit.sensory[:,  _IDX_VIS])
    e_delayed_dark = (x_vis_dark @ np.array(C_pos).T)[:, 0]
    e_delayed_lit  = (x_vis_lit  @ np.array(C_pos).T)[:, 0]

    burst_dark = extract_burst(states_dark, theta_dark)
    burst_lit  = extract_burst(states_lit,  theta_lit)

    sg_dark   = np.array(states_dark.brain[:, _IDX_SG])
    sg_lit    = np.array(states_lit.brain[:,  _IDX_SG])
    z_sac_dark = sg_dark[:, 7]
    z_sac_lit  = sg_lit[:,  7]

    orbital_limit = THETA.plant.orbital_limit

    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(
        f'VVOR — {rotate_dur:.0f} s @ {v_head:.0f} deg/s, stationary world target\n'
        f'Dark: VOR → orbital nystagmus   |   Lit: OKR keeps gaze on target',
        fontsize=10)

    vline_kw = dict(color='k', lw=0.8, ls='--', alpha=0.4)
    for ax in axes:
        ax.axvline(rotate_dur, **vline_kw)
        ax_fmt(ax)

    axes[0].plot(t_vv, -hv_vv,   color='gray',      lw=1.0, ls=':', alpha=0.6,
                 label='-head vel  (ideal VOR)')
    axes[0].plot(t_vv, ev_dark,  color=_C['no_vs'], lw=1.0, alpha=0.7, label='Eye vel (dark)')
    axes[0].plot(t_vv, ev_lit,   color=_C['eye'],   lw=1.0, alpha=0.7, label='Eye vel (lit)')
    axes[0].set_ylabel('Eye vel (deg/s)')
    axes[0].set_title('Eye velocity')
    axes[0].legend(fontsize=8)

    axes[1].axhspan( orbital_limit,       orbital_limit + 20, color='tomato', alpha=0.08)
    axes[1].axhspan(-orbital_limit - 20, -orbital_limit,      color='tomato', alpha=0.08)
    axes[1].axhline( orbital_limit, color='tomato', lw=1.0, ls='--', alpha=0.7,
                     label=f'±orbital limit ({orbital_limit:.0f}°)')
    axes[1].axhline(-orbital_limit, color='tomato', lw=1.0, ls='--', alpha=0.7)
    axes[1].plot(t_vv, eye_dark, color=_C['no_vs'], lw=1.5, ls='--', label='Eye pos (dark)')
    axes[1].plot(t_vv, eye_lit,  color=_C['eye'],   lw=1.5,          label='Eye pos (lit)')
    axes[1].set_ylabel('Eye pos (deg)')
    axes[1].set_title('Eye position')
    axes[1].legend(fontsize=8)

    axes[2].plot(t_vv, head_pos + eye_dark, color=_C['no_vs'], lw=1.5, ls='--', label='Gaze (dark)')
    axes[2].plot(t_vv, head_pos + eye_lit,  color=_C['eye'],   lw=1.5,          label='Gaze (lit)')
    axes[2].axhline(0, color='k', lw=0.8)
    axes[2].set_ylabel('Gaze error (deg)')
    axes[2].set_title('Gaze error  (head pos + eye pos;  0 = perfect stabilisation)')
    axes[2].legend(fontsize=8)

    axes[3].plot(t_vv, e_delayed_dark, color=_C['no_vs'], lw=1.0, ls='--', alpha=0.7,
                 label='e_pos_delayed (dark)')
    axes[3].plot(t_vv, e_delayed_lit,  color=_C['eye'],   lw=1.0, alpha=0.7,
                 label='e_pos_delayed (lit)')
    axes[3].set_ylabel('Retinal error (deg)')
    axes[3].set_title('Delayed retinal position error')
    axes[3].legend(fontsize=8)

    axes[4].plot(t_vv, burst_dark[:, 0], color=_C['no_vs'], lw=1.0, ls='--', alpha=0.8,
                 label='Burst (dark)')
    axes[4].plot(t_vv, burst_lit[:, 0],  color=_C['eye'],   lw=1.0, alpha=0.8,
                 label='Burst (lit)')
    axes[4].set_ylabel('Burst (deg/s)')
    axes[4].set_title('Saccade burst u_burst')
    axes[4].legend(fontsize=8)

    axes[5].plot(t_vv, z_sac_dark, color=_C['no_vs'], lw=1.0, ls='--', alpha=0.8,
                 label='z_sac (dark)')
    axes[5].plot(t_vv, z_sac_lit,  color=_C['eye'],   lw=1.0, alpha=0.8,
                 label='z_sac (lit)')
    axes[5].set_ylim(-0.05, 1.15)
    axes[5].set_ylabel('z_sac')
    axes[5].set_xlabel('Time (s)')
    axes[5].set_title('Saccade latch z_sac')
    axes[5].legend(fontsize=8)
    axes[5].grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'vvor.png')
    fig.savefig(path, dpi=150)
    if SHOW: plt.show()
    else:    plt.close(fig)
    print(f'  Saved {path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=== VOR Demo Suite ===')
    print(f'  tau_c={THETA.sensory.tau_c} s  tau_i={THETA.brain.tau_i} s  '
          f'tau_p={THETA.brain.tau_p} s  tau_vs={THETA.brain.tau_vs} s')

    print('\n1. VOR cascade (dark)')
    demo_vor_cascade()

    print('\n2. OKR cascade (OKN + OKAN)')
    demo_okr_cascade()

    print('\n3. VVOR (dark vs lit)')
    demo_vvor()

    print(f'\nDone. Plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
