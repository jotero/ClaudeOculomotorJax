"""VOR demo suite — replicates key literature findings + internal signal trace.

Demonstrations
--------------
1. Velocity storage — step rotation + post-rotatory nystagmus
     Canal adapts with τ_c; VS extends eye velocity signal with τ_eff ≈ 20 s.
     After head stops, canal briefly signals opposite direction (post-rotatory).
     Refs: Raphan et al. (1979); Cohen et al. (1977).

2. Head impulse test (HIT) — push-pull canal pair with half-wave rectification
     Fast (150 ms) high-amplitude (200 deg/s) head rotation.
     Ewald's 2nd law: slight asymmetry in healthy subject.
     Unilateral loss: dramatically reduced VOR gain ipsilateral to lesion.
     Refs: Halmagyi & Curthoys (1988); Ewald (1892).

3. Low-frequency VOR attenuation
     Canal high-pass cuts off below ~0.03 Hz. Eye position fails to track
     head position for very slow sinusoidal rotation.

4. Ewald's 2nd law — VOR gain vs. head velocity amplitude
     Half-wave rectification causes saturation / asymmetry at large amplitudes.

5. Signal trace — internal states for step and HIT
     6-panel plot: head_vel → canal → VS → NI → plant → eye_vel.
     Useful for verifying signal flow and diagnosing model behaviour.

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

from oculomotor.sim.synthetic import THETA_TRUE
from oculomotor.models.vor import simulate
from oculomotor.models import canal as canal_ssm
from oculomotor.models.vor import (
    vor_vector_field, _N_TOTAL, _IDX_C, _IDX_VS, _IDX_NI, _IDX_P, _DT_SOLVE
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

THETA = THETA_TRUE  # ground-truth parameters for all demos


# ---------------------------------------------------------------------------
# Stimulus helpers
# ---------------------------------------------------------------------------

def sustained_rotation(v_deg_s=60.0, rotate_dur=20.0, coast_dur=60.0,
                        sample_rate=100.0):
    """Constant-velocity rotation then sudden stop."""
    total = rotate_dur + coast_dur
    t = jnp.arange(0.0, total, 1.0 / sample_rate)
    hv = jnp.where(t < rotate_dur, v_deg_s, 0.0)
    return t, hv


def hit_stimulus(direction=1.0, v_peak=200.0, duration=0.15,
                 total_time=8.0, sample_rate=200.0):
    """Haversine head-velocity pulse (smooth onset/offset HIT)."""
    t = jnp.arange(0.0, total_time, 1.0 / sample_rate)
    within = (t >= 0.0) & (t <= duration)
    pulse = jnp.where(within, jnp.sin(jnp.pi * t / duration) ** 2, 0.0)
    return t, direction * v_peak * pulse


def sinusoidal(freq_hz=0.02, amplitude=30.0, duration=100.0, sample_rate=100.0):
    """Sinusoidal head velocity."""
    t = jnp.arange(0.0, duration, 1.0 / sample_rate)
    hv = amplitude * jnp.sin(2.0 * jnp.pi * freq_hz * t)
    return t, hv


def eye_velocity(eye_pos_array, dt):
    """Numerical derivative of eye position → eye velocity."""
    return np.gradient(np.array(eye_pos_array), dt)


# ---------------------------------------------------------------------------
# Demo 1: Velocity storage
# ---------------------------------------------------------------------------

def demo_velocity_storage():
    """Step rotation: per-rotatory nystagmus and post-rotatory aftereffect."""
    t, hv = sustained_rotation(v_deg_s=60.0, rotate_dur=15.0, coast_dur=60.0)
    dt        = float(t[1] - t[0])
    max_steps = int((float(t[-1]) - float(t[0])) / _DT_SOLVE) + 500
    eye_pos_vs = np.array(simulate(THETA, t, hv, max_steps=max_steps))
    ev_vs = eye_velocity(eye_pos_vs, dt)

    theta_no_vs = dict(THETA)
    theta_no_vs['tau_vs'] = 0.1
    theta_no_vs['K_vs']   = 0.001
    eye_pos_novs = np.array(simulate(theta_no_vs, t, hv, max_steps=max_steps))
    ev_novs = eye_velocity(eye_pos_novs, dt)

    t_np = np.array(t)
    rotate_dur = 15.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(t_np, np.array(hv), color='gray', lw=1.5)
    axes[0].axvline(rotate_dur, color='k', ls='--', lw=0.8, alpha=0.5)
    axes[0].set_ylabel('Head vel (deg/s)')
    axes[0].set_title('Velocity storage — step rotation')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_np, ev_vs,   color='steelblue', lw=1.5, label='With VS (τ_eff≈20s)')
    axes[1].plot(t_np, ev_novs, color='tomato',    lw=1.5, ls='--', label='Without VS')
    axes[1].axvline(rotate_dur, color='k', ls='--', lw=0.8, alpha=0.5)
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_ylabel('Eye vel (deg/s)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    post_mask = t_np >= rotate_dur
    axes[2].plot(t_np[post_mask] - rotate_dur,
                 ev_vs[post_mask],   color='steelblue', lw=1.5, label='With VS')
    axes[2].plot(t_np[post_mask] - rotate_dur,
                 ev_novs[post_mask], color='tomato',    lw=1.5, ls='--', label='Without VS')
    axes[2].axhline(0, color='k', lw=0.5)
    axes[2].set_xlabel('Time after rotation stop (s)')
    axes[2].set_ylabel('Eye vel (deg/s)\n[post-rotatory]')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    tau_eff = 1.0 / (1.0 / THETA['tau_vs'] + THETA['K_vs'])
    axes[2].set_title(f'Post-rotatory nystagmus  (τ_eff = {tau_eff:.1f} s)')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'velocity_storage.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


# ---------------------------------------------------------------------------
# Demo 2: Head impulse test
# ---------------------------------------------------------------------------

def demo_hit():
    """HIT: normal canal pair vs. unilateral loss."""
    FLOOR = 80.0

    conditions = [
        ('Normal (rightward)',     +1.0, (1.0, 1.0)),
        ('Normal (leftward)',      -1.0, (1.0, 1.0)),
        ('Right loss (rightward)', +1.0, (0.0, 1.0)),
        ('Right loss (leftward)',  -1.0, (0.0, 1.0)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flat

    for ax, (label, direction, gains) in zip(axes, conditions):
        t, hv = hit_stimulus(direction=direction, v_peak=200.0,
                             duration=0.15, total_time=4.0, sample_rate=200.0)
        dt = float(t[1] - t[0])
        max_steps = int(4.0 / dt) + 100
        eye_pos = np.array(simulate(THETA, t, hv,
                                    canal_floor=FLOOR, canal_gains=gains,
                                    max_steps=max_steps, dt_solve=dt))
        ev      = eye_velocity(eye_pos, dt)
        hv_np   = np.array(hv)
        t_np    = np.array(t)

        hit_mask   = t_np <= 0.2
        peak_head  = np.max(np.abs(hv_np[hit_mask]))
        peak_eye   = np.max(np.abs(ev[hit_mask]))
        gain       = peak_eye / peak_head if peak_head > 0 else 0.0

        ax.plot(t_np, hv_np * (-1), color='gray',     lw=1.0, alpha=0.7, label='−Head vel')
        ax.plot(t_np, ev,           color='steelblue', lw=1.5, label='Eye vel')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlim(0, 1.5)
        ax.set_title(f'{label}\nVOR gain = {gain:.2f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (deg/s)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Head Impulse Test  (floor = 80 deg/s)')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'hit.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


# ---------------------------------------------------------------------------
# Demo 3: Low-frequency VOR attenuation
# ---------------------------------------------------------------------------

def demo_low_freq():
    """Canal high-pass: VOR fails for very slow head movements."""
    freqs = [0.005, 0.02, 0.1, 0.5]
    fig, axes = plt.subplots(len(freqs), 1, figsize=(10, 10), sharex=False)

    for ax, freq in zip(axes, freqs):
        dur = max(3.0 / freq, 20.0)
        sr  = 100.0
        t, hv = sinusoidal(freq_hz=freq, amplitude=30.0,
                           duration=dur, sample_rate=sr)
        max_steps = int(dur / _DT_SOLVE) + 500
        eye_pos   = np.array(simulate(THETA, t, hv, max_steps=max_steps))
        head_pos  = np.cumsum(np.array(hv)) / sr

        t_np = np.array(t)
        ax.plot(t_np, head_pos,  color='gray',      lw=1.0, alpha=0.8,
                label='Head pos (expected −eye)')
        ax.plot(t_np, -eye_pos,  color='steelblue', lw=1.5,
                label='−Eye pos (actual)')
        ax.set_ylabel('Position (deg)')
        ax.set_title(f'{freq} Hz  (canal HP corner ≈ 0.03 Hz)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Low-frequency VOR attenuation — canal high-pass')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'low_freq_vor.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


# ---------------------------------------------------------------------------
# Demo 4: Ewald's 2nd law across amplitudes
# ---------------------------------------------------------------------------

def demo_ewald():
    """VOR gain vs. head velocity amplitude: asymmetry from half-wave rectification."""
    amplitudes = np.array([20, 50, 100, 150, 200, 250, 300], dtype=float)
    FLOOR = 80.0

    gains_right = []
    gains_left  = []

    for amp in amplitudes:
        for direction, gains_list in [(+1.0, gains_right), (-1.0, gains_left)]:
            t, hv = hit_stimulus(direction=direction, v_peak=amp,
                                 duration=0.15, total_time=2.0, sample_rate=200.0)
            dt        = float(t[1] - t[0])
            max_steps = int(2.0 / dt) + 100
            eye_pos   = np.array(simulate(THETA, t, hv,
                                          canal_floor=FLOOR, canal_gains=(1.0, 1.0),
                                          max_steps=max_steps, dt_solve=dt))
            ev       = eye_velocity(eye_pos, dt)
            hit_mask = np.array(t) <= 0.2
            peak_eye = np.max(np.abs(ev[hit_mask]))
            gains_list.append(peak_eye / amp)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(amplitudes, gains_right, 'o-', color='steelblue', label='Rightward (canal A)')
    ax.plot(amplitudes, gains_left,  's-', color='tomato',    label='Leftward  (canal B)')
    ax.axhline(1.0, color='k', ls='--', lw=0.8, label='Ideal gain = 1')
    ax.set_xlabel('Head impulse amplitude (deg/s)')
    ax.set_ylabel('VOR gain (peak eye vel / head vel)')
    ax.set_title("Ewald's 2nd law — VOR gain saturation with canal floor")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'ewald.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


# ---------------------------------------------------------------------------
# Demo 5: Signal trace — internal states
# ---------------------------------------------------------------------------

def _simulate_all_states(theta, t_array, head_vel_array,
                          canal_floor=1e6, canal_gains=(1.0, 1.0),
                          max_steps=2000, dt_solve=None):
    """Run the VOR ODE and return the full state matrix (T, N_TOTAL)."""
    dt          = _DT_SOLVE if dt_solve is None else dt_solve
    hv_interp   = diffrax.LinearInterpolation(ts=t_array, ys=head_vel_array)
    x0          = jnp.zeros(_N_TOTAL)
    gains_array = jnp.array(list(canal_gains), dtype=jnp.float32)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(vor_vector_field),
        diffrax.Heun(),
        t0=t_array[0],
        t1=t_array[-1],
        dt0=dt,
        y0=x0,
        args=(theta, hv_interp, jnp.float32(canal_floor), gains_array),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=t_array),
        max_steps=max_steps,
    )
    return solution.ys  # (T, N_TOTAL)


def _extract_signals(theta, t_array, head_vel_array, states, canal_floor=1e6):
    """Reconstruct intermediate signals from a saved state trajectory.

    Canal state layout: [x1_c0, x1_c1, x2_c0, x2_c1]
      x1 = adaptation LP (slow, tau_c)  — HP intermediate = head_vel - x1
      x2 = inertia LP   (fast, tau_s)  — bandpass output (fed to VS)
    """
    hv   = np.array(head_vel_array)
    x_c  = np.array(states[:, _IDX_C])   # (T, 4): [x1_c0, x1_c1, x2_c0, x2_c1]
    x_vs = np.array(states[:, _IDX_VS])  # (T, 1)
    x_ni = np.array(states[:, _IDX_NI])  # (T, 1)
    x_p  = np.array(states[:, _IDX_P])   # (T, 1)

    nc = canal_ssm.N_CANALS
    x1 = x_c[:, :nc]    # (T, 2) adaptation LP states
    x2 = x_c[:, nc:]    # (T, 2) inertia LP states = bandpass output

    # Canal nonlinearity applied to x2 (same smooth softplus as canal_ssm)
    from scipy.special import softplus as sp_softplus
    k = canal_ssm._SOFTNESS
    f = float(canal_floor)
    y_c = -f + sp_softplus(k * (x2 + f)) / k + sp_softplus(k * (x2 - f)) / k  # (T, 2)

    pinv = np.array(canal_ssm.PINV_SENS_1D)
    u_vs = (pinv @ y_c.T).T                            # (T, 1)
    u_ni = x_vs + u_vs                                 # (T, 1) VS state + direct canal
    u_p  = x_ni - theta['g_vor'] * theta['tau_p'] * u_ni  # (T, 1) pulse-step

    eye_pos = x_p[:, 0]
    dt      = float(t_array[1] - t_array[0])
    eye_vel = np.gradient(eye_pos, dt)

    return dict(head_vel=hv,
                x1_c0=x1[:, 0], x1_c1=x1[:, 1],
                x2_c0=x2[:, 0], x2_c1=x2[:, 1],
                y_c0=y_c[:, 0], y_c1=y_c[:, 1],
                u_vs=u_vs[:, 0], x_vs=x_vs[:, 0],
                u_ni=u_ni[:, 0], x_ni=x_ni[:, 0],
                u_p=u_p[:, 0],
                eye_pos=eye_pos, eye_vel=eye_vel)


def _plot_signal_trace(t_np, sigs, title, path):
    """6-panel waterfall: head → canal (bandpass) → VS → NI → plant → eye_vel."""
    fig, axes = plt.subplots(6, 1, figsize=(11, 14), sharex=True)

    axes[0].plot(t_np, sigs['head_vel'], color='k', lw=1.5)
    axes[0].set_ylabel('Head vel\n(deg/s)')
    axes[0].set_title(title)

    axes[1].plot(t_np, sigs['x1_c0'], 'steelblue', lw=1.0, ls=':',  label='x1_c0 (adapt LP)')
    axes[1].plot(t_np, sigs['x1_c1'], 'tomato',    lw=1.0, ls=':',  label='x1_c1 (adapt LP)')
    axes[1].plot(t_np, sigs['x2_c0'], 'steelblue', lw=1.2, ls='--', label='x2_c0 (inertia LP)')
    axes[1].plot(t_np, sigs['x2_c1'], 'tomato',    lw=1.2, ls='--', label='x2_c1 (inertia LP)')
    axes[1].plot(t_np, sigs['y_c0'],  'steelblue', lw=1.5,          label='y_c0 (BP out)')
    axes[1].plot(t_np, sigs['y_c1'],  'tomato',    lw=1.5,          label='y_c1 (BP out)')
    axes[1].set_ylabel('Canal\n(deg/s)')
    axes[1].legend(fontsize=7, ncol=2)

    axes[2].plot(t_np, sigs['u_vs'], 'purple', lw=1.5,          label='u_vs (canal → VS)')
    axes[2].plot(t_np, sigs['x_vs'], 'purple', lw=1.5, ls='--', label='x_vs (VS state)')
    axes[2].set_ylabel('VS\n(deg/s)')
    axes[2].legend(fontsize=7)

    axes[3].plot(t_np, sigs['u_ni'], 'green', lw=1.5,          label='u_ni = x_vs + u_vs')
    axes[3].plot(t_np, sigs['x_ni'], 'green', lw=1.5, ls='--', label='x_ni (eye pos cmd)')
    axes[3].set_ylabel('NI\n(deg/s or deg)')
    axes[3].legend(fontsize=7)

    axes[4].plot(t_np, sigs['u_p'],    'darkorange', lw=1.5,          label='u_p (pulse-step)')
    axes[4].plot(t_np, sigs['eye_pos'], 'darkorange', lw=1.5, ls='--', label='eye_pos')
    axes[4].set_ylabel('Plant\n(deg)')
    axes[4].legend(fontsize=7)

    axes[5].plot(t_np, sigs['eye_vel'],    'steelblue', lw=1.5, label='eye_vel')
    axes[5].plot(t_np, -sigs['head_vel'],  'k',         lw=1.0, ls='--', alpha=0.5, label='−head_vel (ideal)')
    axes[5].set_ylabel('Eye vel\n(deg/s)')
    axes[5].set_xlabel('Time (s)')
    axes[5].legend(fontsize=7)

    for ax in axes:
        ax.axhline(0, color='gray', lw=0.5)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def demo_signal_trace():
    """Internal signal waterfall for step rotation and HIT."""
    # Step rotation
    t, hv = sustained_rotation(v_deg_s=60.0, rotate_dur=15.0, coast_dur=60.0)
    max_steps_step = int((float(t[-1]) - float(t[0])) / _DT_SOLVE) + 500
    states = _simulate_all_states(THETA, t, hv, max_steps=max_steps_step)
    sigs   = _extract_signals(THETA, t, hv, states)
    _plot_signal_trace(np.array(t), sigs,
                       title='Signal trace — step rotation (60 deg/s, 15 s)',
                       path=os.path.join(OUTPUT_DIR, 'trace_step.png'))

    # HIT — use fine dt matching sample rate so plant is accurately integrated
    t_hit, hv_hit = hit_stimulus(v_peak=200.0, duration=0.15,
                                  total_time=4.0, sample_rate=200.0)
    dt_hit        = float(t_hit[1] - t_hit[0])   # 0.005 s
    max_steps_hit = int(4.0 / dt_hit) + 200
    states_hit = _simulate_all_states(THETA, t_hit, hv_hit,
                                       canal_floor=80.0,
                                       max_steps=max_steps_hit, dt_solve=dt_hit)
    sigs_hit   = _extract_signals(THETA, t_hit, hv_hit, states_hit, canal_floor=80.0)
    t_np_hit   = np.array(t_hit)
    mask       = t_np_hit <= 1.0
    _plot_signal_trace(t_np_hit[mask], {k: v[mask] for k, v in sigs_hit.items()},
                       title='Signal trace — HIT (200 deg/s, canal_floor=80)',
                       path=os.path.join(OUTPUT_DIR, 'trace_hit.png'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=== VOR Demo Suite ===')
    print(f'Parameters: tau_c={THETA["tau_c"]}s  g_vor={THETA["g_vor"]}  '
          f'tau_i={THETA["tau_i"]}s  tau_p={THETA["tau_p"]}s  '
          f'tau_vs={THETA["tau_vs"]}s  K_vs={THETA["K_vs"]}')

    print('\n1. Velocity storage + post-rotatory nystagmus')
    demo_velocity_storage()

    print('\n2. Head impulse test (normal & unilateral loss)')
    demo_hit()

    print('\n3. Low-frequency VOR attenuation')
    demo_low_freq()

    print("\n4. Ewald's 2nd law")
    demo_ewald()

    print('\n5. Signal trace (step + HIT)')
    demo_signal_trace()

    print(f'\nDone. Plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
