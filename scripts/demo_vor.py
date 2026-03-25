"""VOR demo suite — diagnostic signal-cascade plots.

Two main diagnostics:
1. vor_dark.png  — full 8-panel signal cascade for step rotation in the dark.
     Head vel → canal (adaptation, bandpass, afferent) → u_canal → VS → NI → eye.
     Overlays with/without VS on the VS-state and eye-velocity panels.

2. okr.png       — 6-panel OKN cascade: scene on then off (OKAN).
     Scene vel → retinal slip → visual delay → OKR store → u_okr → eye vel.

3. vvor.png      — VVOR comparison: step rotation, VOR only vs VOR + OKR.

Usage
-----
    python scripts/demo_vor.py

Outputs saved to outputs/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
import diffrax

from oculomotor.models.ocular_motor_simulator import THETA_DEFAULT, simulate
from oculomotor.models import canal as canal_ssm
from oculomotor.models import visual_delay
from oculomotor.models.ocular_motor_simulator import (
    vor_vector_field, _N_TOTAL,
    _IDX_C, _IDX_VS, _IDX_NI, _IDX_P, _IDX_VIS, _DT_SOLVE,
)
from oculomotor.sim.stimulus import rotation_step, Stimulus

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
THETA = THETA_DEFAULT


# ---------------------------------------------------------------------------
# Stimulus helpers
# ---------------------------------------------------------------------------

def _sustained_rotation(v_deg_s=60.0, rotate_dur=15.0, coast_dur=60.0,
                         sample_rate=100.0):
    total = rotate_dur + coast_dur
    t  = jnp.arange(0.0, total, 1.0 / sample_rate)
    hv = jnp.where(t < rotate_dur, v_deg_s, 0.0)
    return t, hv


def _eye_velocity(eye_pos, dt):
    return np.gradient(np.array(eye_pos), dt)


# ---------------------------------------------------------------------------
# Full state simulation + signal extraction
# ---------------------------------------------------------------------------

def _simulate_all_states(theta, t_or_stimulus, head_vel_array=None,
                          canal_gains=None, scene_present=None,
                          max_steps=5000, dt_solve=None):
    """Return full state matrix (T, N_TOTAL).

    scene_present: None  → infer: 1 if vs3 is non-zero, 0 (dark) if all-zero.
                   True  → force scene present (use for stationary lit world).
                   False → force dark.
                   (T,) array → time-varying gain.
    """
    dt = _DT_SOLVE if dt_solve is None else dt_solve

    if hasattr(t_or_stimulus, 'omega'):          # Stimulus object
        stim    = t_or_stimulus
        t_array = stim.t
        hv3     = stim.omega     # already (T, 3)
        vs3     = stim.v_scene   # already (T, 3)
    else:
        t_array = t_or_stimulus
        hv1d    = head_vel_array
        hv3     = jnp.stack([hv1d, jnp.zeros_like(hv1d), jnp.zeros_like(hv1d)], axis=1)
        vs3     = jnp.zeros_like(hv3)

    T = len(t_array)

    # Head position: trapezoidal integral of head velocity
    dt_arr = jnp.diff(t_array)
    hp3    = jnp.concatenate([
        jnp.zeros((1, 3)),
        jnp.cumsum(0.5 * (hv3[:-1] + hv3[1:]) * dt_arr[:, None], axis=0),
    ])

    # Default target: straight ahead (no saccade drive in VOR/OKR demos)
    pt3 = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (T, 1))

    # Scene gain: time-varying scalar in [0, 1]
    if scene_present is None:
        # Infer: scene present whenever vs3 is non-zero at that timestep
        sg1 = jnp.where(jnp.any(vs3 != 0.0, axis=1), 1.0, 0.0)
    elif scene_present is True:
        sg1 = jnp.ones(T, dtype=jnp.float32)
    elif scene_present is False:
        sg1 = jnp.zeros(T, dtype=jnp.float32)
    else:
        sg1 = jnp.asarray(scene_present, dtype=jnp.float32)   # explicit array

    hv_interp         = diffrax.LinearInterpolation(ts=t_array, ys=hv3)
    hp_interp         = diffrax.LinearInterpolation(ts=t_array, ys=hp3)
    vs_interp         = diffrax.LinearInterpolation(ts=t_array, ys=vs3)
    target_interp     = diffrax.LinearInterpolation(ts=t_array, ys=pt3)
    scene_gain_interp = diffrax.LinearInterpolation(ts=t_array, ys=sg1)
    x0                = jnp.zeros(_N_TOTAL)
    gains             = (jnp.ones(canal_ssm.N_CANALS) if canal_gains is None
                         else jnp.array(list(canal_gains), dtype=jnp.float32))

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vor_vector_field),
        diffrax.Heun(),
        t0=t_array[0], t1=t_array[-1], dt0=dt, y0=x0,
        args=(theta, hv_interp, hp_interp, vs_interp, target_interp,
              scene_gain_interp, gains),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=t_array),
        max_steps=max_steps,
    )
    return sol.ys   # (T, N_TOTAL)


def _extract_signals(theta, t_array, head_vel_1d, states):
    """Reconstruct intermediate signals from a saved state trajectory.

    Returns a dict of horizontal-axis (yaw) signals.
    Canal afferents y_c0/y_c1 are the y_nl component only (no FLOOR offset);
    add canal_ssm.FLOOR to get absolute firing rates for plotting.
    u_canal is the head velocity estimate after PINV mixing (correct regardless
    of FLOOR offset because push-pull pairs cancel it).
    """
    from scipy.special import softplus as sp_softplus

    hv   = np.array(head_vel_1d)
    x_c  = np.array(states[:, _IDX_C])     # (T, 12)
    x_vs = np.array(states[:, _IDX_VS])    # (T, 3)
    x_ni = np.array(states[:, _IDX_NI])    # (T, 3)
    x_p  = np.array(states[:, _IDX_P])     # (T, 3)

    nc = canal_ssm.N_CANALS
    x1 = x_c[:, :nc]   # adaptation LP states
    x2 = x_c[:, nc:]   # inertia LP (= bandpass) states

    k   = canal_ssm._SOFTNESS
    f   = float(canal_ssm.FLOOR)
    y_c = -f + sp_softplus(k * (x2 + f)) / k + sp_softplus(k * (x2 - f)) / k   # (T,6) y_nl

    pinv    = np.array(canal_ssm.PINV_SENS)   # (3, 6)
    u_canal = (pinv @ y_c.T).T                # (T, 3) head vel estimate from canals
    # VS output w_est = x_vs + PINV@y_canals - g_vis*e_slip_delayed
    # In dark (VOR demo): e_slip_delayed ≈ 0, so w_est ≈ x_vs + PINV@y_canals
    w_est   = x_vs + u_canal                  # (T, 3) velocity estimate sent to NI
    # NI motor command: u_p = x_ni + tau_p * u_vel, u_vel = -g_vor * w_est
    u_p     = x_ni - theta['g_vor'] * theta['tau_p'] * w_est   # (T, 3) pulse-step

    eye_pos = x_p[:, 0]
    dt      = float(t_array[1] - t_array[0])
    eye_vel = np.gradient(eye_pos, dt)

    return dict(
        head_vel = hv,
        x1_c0    = x1[:, 0],   x1_c1    = x1[:, 1],
        x2_c0    = x2[:, 0],   x2_c1    = x2[:, 1],
        y_c0     = y_c[:, 0],  y_c1     = y_c[:, 1],   # y_nl (no FLOOR offset)
        u_canal  = u_canal[:, 0],
        x_vs     = x_vs[:, 0],
        w_est    = w_est[:, 0],
        x_ni     = x_ni[:, 0],
        u_p      = u_p[:, 0],
        eye_pos  = eye_pos,
        eye_vel  = eye_vel,
    )


# ---------------------------------------------------------------------------
# Plot 1: VOR in the dark — full signal cascade
# ---------------------------------------------------------------------------

def demo_vor_dark():
    """8-panel signal cascade: head → canal → VS → NI → plant → eye."""
    rotate_dur = 15.0
    t, hv = _sustained_rotation(v_deg_s=60.0, rotate_dur=rotate_dur, coast_dur=60.0)
    t_np  = np.array(t)
    max_s = int((float(t[-1]) - float(t[0])) / _DT_SOLVE) + 500

    states    = _simulate_all_states(THETA, t, hv, max_steps=max_s)
    sigs      = _extract_signals(THETA, t, hv, states)

    theta_no_vs = {**THETA, 'tau_vs': 0.1, 'K_vs': 0.001}
    states_nv   = _simulate_all_states(theta_no_vs, t, hv, max_steps=max_s)
    sigs_nv     = _extract_signals(theta_no_vs, t, hv, states_nv)

    tau_eff = 1.0 / (1.0 / THETA['tau_vs'] + THETA['K_vs'])
    FLOOR   = float(canal_ssm.FLOOR)

    fig, axes = plt.subplots(8, 1, figsize=(12, 22), sharex=True)
    fig.suptitle(f'VOR in the dark — step rotation  ({rotate_dur:.0f} s @ 60 deg/s)\n'
                 f'Signal cascade: head → canal → VS → NI → plant',
                 fontsize=11)

    vline_kw = dict(color='k', lw=0.8, ls='--', alpha=0.4)
    for ax in axes:
        ax.axvline(rotate_dur, **vline_kw)
        ax.axhline(0, color='gray', lw=0.5)
        ax.grid(True, alpha=0.25)

    # 0 — Head velocity
    axes[0].plot(t_np, sigs['head_vel'], color='k', lw=1.5)
    axes[0].set_ylabel('Head vel\n(deg/s)')
    axes[0].set_title('Input')

    # 1 — Canal adaptation LP (x1) — slow drift-tracking filter
    axes[1].plot(t_np, sigs['x1_c0'], color='steelblue', lw=1.2, label='RHC (c0)')
    axes[1].plot(t_np, sigs['x1_c1'], color='tomato',    lw=1.2, label='LHC (c1)')
    axes[1].set_ylabel('Adaptation LP\nx1  (deg/s)')
    axes[1].set_title(f'Canal — adaptation state  (τ_c = {THETA["tau_c"]} s)')
    axes[1].legend(fontsize=8, loc='upper right')

    # 2 — Canal inertia LP = bandpass (x2)
    axes[2].plot(t_np, sigs['x2_c0'], color='steelblue', lw=1.2, label='RHC (c0)')
    axes[2].plot(t_np, sigs['x2_c1'], color='tomato',    lw=1.2, label='LHC (c1)')
    axes[2].set_ylabel('Inertia LP\nx2  (deg/s)')
    axes[2].set_title(f'Canal — bandpass state  (τ_s = {THETA["tau_s"]} s)')
    axes[2].legend(fontsize=8, loc='upper right')

    # 3 — Canal afferent absolute firing rate (y_nl + FLOOR)
    axes[3].plot(t_np, sigs['y_c0'] + FLOOR, color='steelblue', lw=1.5, label='RHC (c0)')
    axes[3].plot(t_np, sigs['y_c1'] + FLOOR, color='tomato',    lw=1.5, label='LHC (c1)')
    axes[3].axhline(FLOOR, color='gray', lw=1.0, ls=':', label=f'Resting = {FLOOR:.0f}')
    axes[3].set_ylabel('Afferent\nfiring rate  (spk/s)')
    axes[3].set_title('Canal afferent output  (nonlinear, push-pull)')
    axes[3].legend(fontsize=8)

    # 4 — u_canal: head vel estimate after PINV mixing
    axes[4].plot(t_np, sigs['u_canal'],   color='purple', lw=1.5, label='u_canal  (PINV @ y)')
    axes[4].plot(t_np, sigs['head_vel'],  color='gray',   lw=1.0, ls=':', alpha=0.6,
                 label='Head vel  (true)')
    axes[4].set_ylabel('Head vel\nestimate  (deg/s)')
    axes[4].set_title('Canal → head velocity estimate  (PINV_SENS @ y_canals)')
    axes[4].legend(fontsize=8)

    # 5 — VS state: with vs without VS
    axes[5].plot(t_np, sigs['x_vs'],    color='steelblue', lw=1.5,
                 label=f'x_vs  (τ_eff = {tau_eff:.0f} s)')
    axes[5].plot(t_np, sigs_nv['x_vs'], color='tomato',    lw=1.5, ls='--',
                 label='x_vs  (no VS)')
    axes[5].set_ylabel('VS state\n(deg/s)')
    axes[5].set_title(f'Velocity storage  (τ_vs = {THETA["tau_vs"]} s,  K_vs = {THETA["K_vs"]} /s)')
    axes[5].legend(fontsize=8)

    # 6 — NI: VS velocity estimate w_est feeding NI + NI position state x_ni
    #   w_est is the VS output (positive = head vel direction)
    #   actual vel cmd to NI = -g_vor * w_est (sign-flipped, not shown separately)
    axes[6].plot(t_np, sigs['w_est'], color='seagreen', lw=1.5,          label='w_est  (VS velocity estimate, deg/s)')
    axes[6].plot(t_np, sigs['x_ni'],  color='seagreen', lw=1.5, ls='--', label='x_ni   (NI position state, deg)')
    axes[6].set_ylabel('deg/s  /  deg')
    axes[6].set_title(f'VS output w_est and NI state  (τ_i = {THETA["tau_i"]} s,  g_vor = {THETA["g_vor"]})')
    axes[6].legend(fontsize=8)

    # 7 — Eye velocity: with VS vs without VS vs ideal
    axes[7].plot(t_np, sigs['eye_vel'],    color='steelblue', lw=1.5,
                 label='Eye vel  (with VS)')
    axes[7].plot(t_np, sigs_nv['eye_vel'], color='tomato',    lw=1.5, ls='--',
                 label='Eye vel  (no VS)')
    axes[7].plot(t_np, -sigs['head_vel'],  color='gray',      lw=1.0, ls=':',  alpha=0.6,
                 label='−Head vel  (ideal VOR)')
    axes[7].set_ylabel('Eye vel\n(deg/s)')
    axes[7].set_xlabel('Time (s)')
    axes[7].set_title('Plant output  (eye velocity)')
    axes[7].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'vor_dark.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


# ---------------------------------------------------------------------------
# Plot 2: OKR — signal cascade + VVOR comparison
# ---------------------------------------------------------------------------

def demo_okr():
    """OKN signal cascade (scene on → OKAN) and VVOR comparison."""
    sr        = 200.0
    on_dur    = 30.0
    off_dur   = 40.0
    total_dur = on_dur + off_dur
    scene_vel = 30.0

    t_arr    = jnp.arange(0.0, total_dur, 1.0 / sr)
    v_sc     = jnp.where(t_arr < on_dur, scene_vel, 0.0)
    v_scene  = jnp.zeros((len(t_arr), 3)).at[:, 0].set(v_sc)
    stim_okn = Stimulus(t_arr, omega=jnp.zeros(len(t_arr)), v_scene=v_scene)

    max_s        = int(total_dur / _DT_SOLVE) + 500
    theta_no_vis = {**THETA, 'K_vis': 0.0, 'g_vis': 0.0}

    states    = _simulate_all_states(THETA,        stim_okn, max_steps=max_s)
    t_np      = np.array(stim_okn.t)
    dt        = float(stim_okn.dt)

    x_vis = np.array(states[:, _IDX_VIS])
    x_vs  = np.array(states[:, _IDX_VS])
    x_p   = np.array(states[:, _IDX_P])

    w_scene_np  = np.where(t_np < on_dur, scene_vel, 0.0)
    eye_vel     = np.gradient(x_p[:, 0], dt)
    e_slip      = w_scene_np - 0.0 - eye_vel    # w_scene - w_head - w_eye (head=0)
    # Delayed slip: last stage of slip cascade via C_slip selector
    e_delayed_3 = x_vis @ np.array(visual_delay.C_slip).T           # (T, 3)
    e_delayed   = e_delayed_3[:, 0]                                   # horizontal axis
    # VS visual direct drive (instantaneous contribution from delayed slip)
    u_vis_direct = THETA['g_vis'] * e_delayed

    eye_no_vis = np.array(simulate(theta_no_vis, stim_okn, max_steps=max_s))[:, 0]
    ev_no_vis  = np.gradient(eye_no_vis, dt)

    # ── OKN cascade ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(6, 1, figsize=(12, 17), sharex=True)
    fig.suptitle(f'OKN signal cascade — scene on {on_dur:.0f} s then off  (OKAN phase)\n'
                 f'K_vis = {THETA["K_vis"]},  g_vis = {THETA["g_vis"]},  '
                 f'τ_eff ≈ 20 s,  τ_vis = {THETA["tau_vis"]} s',
                 fontsize=11)

    vline_kw = dict(color='k', lw=0.8, ls='--', alpha=0.4)
    for ax in axes:
        ax.axvline(on_dur, **vline_kw)
        ax.axhline(0, color='gray', lw=0.5)
        ax.grid(True, alpha=0.25)

    # 0 — Scene velocity (input)
    axes[0].plot(t_np, w_scene_np, color='steelblue', lw=1.5)
    axes[0].set_ylabel('Scene vel\n(deg/s)')
    axes[0].set_title('Input: visual scene velocity')

    # 1 — Retinal slip
    axes[1].plot(t_np, e_slip, color='purple', lw=1.5,
                 label='e_slip = w_scene − w_head − w_eye')
    axes[1].set_ylabel('Retinal slip\n(deg/s)')
    axes[1].set_title('Retinal slip  (sensory error)')
    axes[1].legend(fontsize=8)

    # 2 — Delayed slip (visual delay cascade output)
    axes[2].plot(t_np, e_delayed, color='darkorange', lw=1.5,
                 label=f'e_delayed  (τ_vis = {THETA["tau_vis"]} s,  40-stage gamma)')
    axes[2].set_ylabel('Delayed slip\n(deg/s)')
    axes[2].set_title('Visual delay cascade output')
    axes[2].legend(fontsize=8)

    # 3 — VS storage state (charges during OKN, drives OKAN)
    axes[3].plot(t_np, x_vs[:, 0], color='steelblue', lw=1.5,
                 label=f'x_vs  (τ_eff ≈ 20 s,  K_vis={THETA["K_vis"]})')
    axes[3].set_ylabel('VS state\n(deg/s)')
    axes[3].set_title('Velocity storage state  (negative = scene drive; sustains OKAN)')
    axes[3].legend(fontsize=8)

    # 4 — Visual direct drive vs VS state contribution to u_vel
    u_vel_from_vs  = -x_vs[:, 0]                  # -g_vor * x_vs (g_vor=1)
    axes[4].plot(t_np, u_vis_direct,   color='green',     lw=1.5,
                 label=f'g_vis · e_delayed  (direct,  g_vis={THETA["g_vis"]})')
    axes[4].plot(t_np, u_vel_from_vs,  color='steelblue', lw=1.5, ls='--',
                 label='−x_vs  (VS state contribution to u_vel)')
    axes[4].set_ylabel('Visual drive\nto NI  (deg/s)')
    axes[4].set_title('Visual contributions to eye-velocity command')
    axes[4].legend(fontsize=8)

    # 5 — Eye velocity: with vs without visual drive
    axes[5].plot(t_np, w_scene_np, color='gray',      lw=1.0, ls=':',  alpha=0.5, label='Scene vel')
    axes[5].plot(t_np, ev_no_vis,  color='tomato',    lw=1.5, ls='--', label='Eye vel  (no visual drive)')
    axes[5].plot(t_np, eye_vel,    color='steelblue', lw=1.5,          label='Eye vel  (with visual drive)')
    axes[5].set_ylabel('Eye vel\n(deg/s)')
    axes[5].set_xlabel('Time (s)')
    axes[5].set_title('Eye velocity  (OKN slow phase + OKAN)')
    axes[5].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'okr.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')

    # ── VVOR comparison ───────────────────────────────────────────────────────
    # VVOR paradigm: head rotates at 10 deg/s for 3 s (30 deg total — within
    # comfortable range without plant limits).  World is visually stationary
    # (v_scene=0 but scene IS present → scene_present=1).
    # Dark case: scene_present=0 → pure VOR (OKR cannot fire).
    # Lit case:  residual VOR slip drives OKR → partially compensates.
    vvor_dur   = 3.0 + 20.0   # 3 s rotation + 20 s coast to observe VS/OKR decay
    stim_vor   = rotation_step(v_deg_s=10.0, rotate_dur=3.0,
                                coast_dur=20.0, sample_rate=sr)
    T_vv       = len(stim_vor.t)
    sg_dark    = jnp.zeros(T_vv, dtype=jnp.float32)   # dark
    sg_lit     = jnp.ones(T_vv,  dtype=jnp.float32)   # stationary lit world
    max_sv     = int(vvor_dur / _DT_SOLVE) + 500

    eye_dark  = np.array(simulate(theta_no_vis, stim_vor,
                                   scene_present_array=sg_dark, max_steps=max_sv))[:, 0]
    eye_vvor  = np.array(simulate(THETA,        stim_vor,
                                   scene_present_array=sg_lit,  max_steps=max_sv))[:, 0]
    t_vv      = np.array(stim_vor.t)
    hv_vv     = np.array(stim_vor.omega[:, 0])
    ev_dark   = np.gradient(eye_dark, float(stim_vor.dt))
    ev_vvor   = np.gradient(eye_vvor, float(stim_vor.dt))

    fig2, axes2 = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig2.suptitle('VVOR — step rotation, stationary world\n'
                  'Dark (VOR only) vs stationary world (VOR + OKR on residual slip)',
                  fontsize=11)

    for ax in axes2:
        ax.axvline(15.0, color='k', lw=0.8, ls='--', alpha=0.4)
        ax.axhline(0, color='gray', lw=0.5)
        ax.grid(True, alpha=0.25)

    axes2[0].plot(t_vv, hv_vv,   color='gray',      lw=1.0, alpha=0.7, label='Head vel')
    axes2[0].plot(t_vv, ev_dark, color='tomato',    lw=1.5, ls='--',   label='VOR only (dark, K_vis=g_vis=0)')
    axes2[0].plot(t_vv, ev_vvor, color='steelblue', lw=1.5,            label=f'VOR + OKR (K_vis={THETA["K_vis"]}, g_vis={THETA["g_vis"]})')
    axes2[0].set_ylabel('Eye vel (deg/s)')
    axes2[0].set_title('Eye velocity  (−head vel = ideal VOR)')
    axes2[0].legend(fontsize=9)

    axes2[1].plot(t_vv, eye_dark, color='tomato',    lw=1.5, ls='--', label='Dark')
    axes2[1].plot(t_vv, eye_vvor, color='steelblue', lw=1.5,          label='Stationary world')
    axes2[1].set_ylabel('Eye pos (deg)')
    axes2[1].set_title('Eye position')
    axes2[1].legend(fontsize=9)

    # Gaze error = head pos + eye pos (should be 0 for perfect gaze stabilisation)
    head_pos_vv = np.cumsum(hv_vv) * float(stim_vor.dt)
    axes2[2].plot(t_vv, head_pos_vv + eye_dark, color='tomato',    lw=1.5, ls='--', label='Gaze error  (dark)')
    axes2[2].plot(t_vv, head_pos_vv + eye_vvor, color='steelblue', lw=1.5,          label='Gaze error  (stationary world)')
    axes2[2].set_ylabel('Gaze error (deg)')
    axes2[2].set_xlabel('Time (s)')
    axes2[2].set_title('Gaze error  (head pos + eye pos;  0 = perfect stabilisation)')
    axes2[2].legend(fontsize=9)

    fig2.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, 'vvor.png')
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f'  Saved {path2}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=== VOR Demo Suite ===')
    print(f'  τ_c={THETA["tau_c"]} s  g_vor={THETA["g_vor"]}  τ_i={THETA["tau_i"]} s  '
          f'τ_p={THETA["tau_p"]} s  τ_vs={THETA["tau_vs"]} s  K_vs={THETA["K_vs"]}')
    print(f'  K_vis={THETA["K_vis"]}  g_vis={THETA["g_vis"]}  τ_vis={THETA["tau_vis"]} s  τ_eff≈20s')

    print('\n1. VOR in the dark — signal cascade')
    demo_vor_dark()

    print('\n2. OKR — OKN cascade + VVOR')
    demo_okr()

    print(f'\nDone. Plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
