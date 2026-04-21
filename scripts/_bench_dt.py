"""Benchmark dt=0.001 vs dt=0.002 on the saccade summary demo.

Runs each dt twice: first run warms up JIT, second run is the timed result.
Saves saccade_summary_dt001.png and saccade_summary_dt002.png.
"""
import os, sys, time
import jax, jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, with_sensory, simulate
from oculomotor.analysis import extract_burst, ni_net
from oculomotor.models.sensory_models.sensory_model import C_pos
from oculomotor.sim.simulator import _IDX_SG, _IDX_VIS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

_NOISE    = with_sensory(PARAMS_DEFAULT, sigma_canal=0.5, sigma_pos=0.2, sigma_vel=0.2)
THETA_SAC = with_brain(_NOISE, g_burst=700.0)

amplitudes_deg = np.array([0.5, 1, 2, 3, 5, 8, 10, 15, 20], dtype=np.float32)
oblique_jumps  = [(0.3, 10., 0.), (0.9, 10., 8.), (1.6, 0., 8.),
                  (2.2, 0., 0.), (2.8, -10., 5.)]


def _make_pt3(t_np, jumps, T):
    pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
    yaw = np.zeros(T); pit = np.zeros(T)
    for tj, y, p in jumps:
        yaw[t_np >= tj] = y; pit[t_np >= tj] = p
    pt3[:, 0] = np.tan(np.radians(yaw))
    pt3[:, 1] = np.tan(np.radians(pit))
    return jnp.array(pt3), yaw, pit


def run_summary(dt, suffix, timed=False):
    T_end   = 3.5
    t       = jnp.arange(0.0, T_end, dt)
    T       = len(t)
    t_np    = np.array(t)
    t_jump  = 0.1
    t_ms    = jnp.arange(0.0, 0.8, dt)
    T_ms    = len(t_ms)
    t_np_ms = np.array(t_ms)
    max_s   = int(T_end / dt) + 500

    t0 = time.perf_counter()

    # Main sequence
    amps_out, peak_vels = [], []
    for i, amp in enumerate(amplitudes_deg):
        pt3_ms = jnp.stack([
            jnp.where(t_ms >= t_jump, jnp.tan(jnp.radians(float(amp))), 0.0),
            jnp.zeros(T_ms), jnp.ones(T_ms),
        ], axis=1)
        eye = simulate(THETA_SAC, t_ms, p_target_array=pt3_ms,
                       scene_present_array=jnp.ones(T_ms),
                       max_steps=int(0.8/dt)+200, key=jax.random.PRNGKey(i))
        eye_yaw = np.array(eye[:, 0])
        vel     = np.gradient(eye_yaw, dt)
        amps_out.append(float(eye_yaw[-1] - eye_yaw[0]))
        peak_vels.append(float(np.max(np.abs(vel))))

    # Oblique sequence
    pt3_obl, tgt_yaw, tgt_pitch = _make_pt3(t_np, oblique_jumps, T)
    states = simulate(THETA_SAC, t, p_target_array=pt3_obl,
                      scene_present_array=jnp.ones(T),
                      max_steps=max_s, return_states=True, key=jax.random.PRNGKey(0))
    eye_pos = np.array(states.plant[:, :3])
    u_burst = extract_burst(states, THETA_SAC)

    # Eye traces (re-simulate each amplitude)
    eye_traces = []
    for i, amp in enumerate(amplitudes_deg):
        pt3_ms = jnp.stack([
            jnp.where(t_ms >= t_jump, jnp.tan(jnp.radians(float(amp))), 0.0),
            jnp.zeros(T_ms), jnp.ones(T_ms),
        ], axis=1)
        eye_i = np.array(simulate(THETA_SAC, t_ms, p_target_array=pt3_ms,
                                  scene_present_array=jnp.ones(T_ms),
                                  max_steps=int(0.8/dt)+200, key=jax.random.PRNGKey(i))[:, 0])
        eye_traces.append(eye_i)

    elapsed = time.perf_counter() - t0

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'Saccade Summary  (dt = {dt:.3f} s)', fontsize=12)
    gs      = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)
    ax_ms   = fig.add_subplot(gs[0, 0])
    ax_traj = fig.add_subplot(gs[1, 0])
    ax_eye  = fig.add_subplot(gs[2, 0])
    ax_ph   = fig.add_subplot(gs[0, 1:])
    ax_pv   = fig.add_subplot(gs[1, 1:], sharex=ax_ph)
    ax_bur  = fig.add_subplot(gs[2, 1:], sharex=ax_ph)

    A_ref = np.linspace(0, 22, 200)
    ax_ms.plot(A_ref, 700*(1-np.exp(-A_ref/7)), lw=1.2, ls='--', color='gray')
    ax_ms.scatter(amps_out, peak_vels, s=60, color='steelblue', zorder=5)
    ax_ms.set_xlabel('Amplitude (deg)'); ax_ms.set_ylabel('Peak vel (deg/s)')
    ax_ms.set_title('Main Sequence'); ax_ms.set_xlim(0, 22); ax_ms.set_ylim(0)
    ax_ms.grid(True, alpha=0.2)

    ax_traj.plot(eye_pos[:, 0], eye_pos[:, 1], color='steelblue', lw=1.2)
    for _, y, p in oblique_jumps:
        ax_traj.plot(y, p, 'x', color='tomato', ms=8, markeredgewidth=2)
    ax_traj.set_xlabel('Yaw (deg)'); ax_traj.set_ylabel('Pitch (deg)')
    ax_traj.set_title('2-D Trajectory'); ax_traj.set_aspect('equal'); ax_traj.grid(True, alpha=0.2)

    cmap = plt.get_cmap('plasma')
    for i, (amp, tr) in enumerate(zip(amplitudes_deg, eye_traces)):
        ax_eye.plot(t_np_ms - t_jump, tr, color=cmap(i/(len(amplitudes_deg)-1)), lw=1.2,
                    label=f'{amp:.0f}°' if amp in [1,5,10,20] else None)
    ax_eye.set_xlabel('Time from step (s)'); ax_eye.set_ylabel('Eye pos (deg)')
    ax_eye.set_title('Eye traces'); ax_eye.set_xlim(-0.05, 0.5)
    ax_eye.legend(fontsize=7); ax_eye.grid(True, alpha=0.2)

    for tj, _, _ in oblique_jumps:
        for ax in [ax_ph, ax_pv, ax_bur]:
            ax.axvline(tj, color='gray', lw=0.6, ls='--', alpha=0.5)

    ax_ph.plot(t_np, tgt_yaw,       color='tomato', lw=1.5, label='target H')
    ax_ph.plot(t_np, eye_pos[:, 0], color='steelblue', lw=1.5, label='eye H')
    ax_ph.set_ylabel('Yaw (deg)'); ax_ph.set_title('Horizontal'); ax_ph.legend(fontsize=8); ax_ph.grid(True, alpha=0.2)

    ax_pv.plot(t_np, tgt_pitch,     color='tomato', lw=1.5, ls='--', label='target V')
    ax_pv.plot(t_np, eye_pos[:, 1], color='steelblue', lw=1.5, ls='--', label='eye V')
    ax_pv.set_ylabel('Pitch (deg)'); ax_pv.set_title('Vertical'); ax_pv.legend(fontsize=8); ax_pv.grid(True, alpha=0.2)

    ax_bur.plot(t_np, np.array(u_burst[:, 0]), color='sandybrown', lw=1.2, label='burst H')
    ax_bur.plot(t_np, np.array(u_burst[:, 1]), color='steelblue',  lw=1.2, ls='--', label='burst V')
    ax_bur.set_ylabel('Burst (deg/s)'); ax_bur.set_xlabel('Time (s)')
    ax_bur.set_title('Burst'); ax_bur.legend(fontsize=8); ax_bur.grid(True, alpha=0.2)

    path = os.path.join(OUTPUT_DIR, f'saccade_summary_{suffix}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')
    return elapsed


print('=== dt benchmark ===')
print('\ndt = 0.001 — warmup (JIT compile)...')
run_summary(0.001, 'dt001_warmup')
print('dt = 0.001 — timed run...')
t001 = run_summary(0.001, 'dt001')
print(f'  → {t001:.1f} s')

print('\ndt = 0.002 — warmup (JIT compile)...')
run_summary(0.002, 'dt002_warmup')
print('dt = 0.002 — timed run...')
t002 = run_summary(0.002, 'dt002')
print(f'  → {t002:.1f} s')

print(f'\nSpeedup: {t001/t002:.2f}×  ({t001:.1f}s → {t002:.1f}s)')
