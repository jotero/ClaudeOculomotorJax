"""Otolith demo — gravity estimator, OCR, and somatogravic feedback.

Three paradigms:
  1. OCR (ocular counter-rolling): head roll → g_hat tracking + eye counter-roll
  2. Gravity estimator convergence: effect of K_grav on convergence speed
  3. Somatogravic: constant forward linear acceleration → pitch eye response

Axis convention (canal.py):
    x = yaw / vertical axis  (specific force = +x ≈ +9.81 when upright)
    y = pitch / interaural
    z = CW roll / naso-occipital

Run:
    python -X utf8 scripts/demo_otolith.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import jax.numpy as jnp

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from oculomotor.sim.simulator import (
    simulate, default_params, with_brain,
    _IDX_GRAV, _IDX_VS,
)

G0    = 9.81
OUT   = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUT, exist_ok=True)


# ── Shared time array ──────────────────────────────────────────────────────────

def make_t(duration, dt=0.001):
    return jnp.arange(0, duration, dt)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  OCR:  head rolls CW to 30°, holds; g_hat tracks, eye counter-rolls
# ──────────────────────────────────────────────────────────────────────────────

def run_ocr(g_ocr=0.3, K_grav=0.5, roll_rate=30.0, hold_deg=30.0, total=6.0):
    """Ramp head CW to hold_deg at roll_rate deg/s, then hold."""
    dt     = 0.001
    t      = make_t(total, dt)
    T      = len(t)
    t_np   = np.array(t)

    ramp_dur = hold_deg / roll_rate          # seconds to reach target angle
    w_roll   = np.zeros((T, 3))
    w_roll[t_np < ramp_dur, 2] = roll_rate   # CW roll = z+

    params = with_brain(default_params(), g_ocr=g_ocr, K_grav=K_grav)
    states = simulate(params, t, head_vel_array=w_roll, return_states=True)

    eye_roll  = np.array(states.plant[:, 2])       # z-component
    g_hat     = np.array(states.brain[:, _IDX_GRAV])

    # head roll (integrate w_roll)
    dt_s     = t_np[1] - t_np[0]
    head_roll = np.cumsum(w_roll[:, 2]) * dt_s

    return t_np, head_roll, eye_roll, g_hat


# ──────────────────────────────────────────────────────────────────────────────
# 2.  K_grav convergence: compare three gain values
# ──────────────────────────────────────────────────────────────────────────────

def run_kgrav_sweep(roll_rate=30.0, hold_deg=30.0, total=8.0):
    gains = [0.1, 0.5, 2.0]
    results = []
    for K in gains:
        t_np, _, _, g_hat = run_ocr(g_ocr=0.0, K_grav=K,
                                    roll_rate=roll_rate,
                                    hold_deg=hold_deg, total=total)
        results.append((K, t_np, g_hat[:, 1]))   # y-component
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Somatogravic:  constant forward (z-axis) linear acceleration
# ──────────────────────────────────────────────────────────────────────────────

def run_somatogravic(a_fwd=5.0, accel_on=2.0, accel_off=8.0, total=14.0, g_ocr=0.3):
    """Forward linear acceleration in z-axis (nasal direction)."""
    dt    = 0.001
    t     = make_t(total, dt)
    T     = len(t)
    t_np  = np.array(t)

    a_head = np.zeros((T, 3))
    on  = t_np >= accel_on
    off = t_np >= accel_off
    a_head[on & ~off, 2] = a_fwd     # z+ = forward/nasal

    params = with_brain(default_params(), g_ocr=g_ocr, K_grav=0.5)
    states = simulate(params, t, head_accel_array=a_head, return_states=True,
                      max_steps=20000)

    eye_pitch = np.array(states.plant[:, 1])       # y-component
    g_hat     = np.array(states.brain[:, _IDX_GRAV])

    return t_np, a_head[:, 2], eye_pitch, g_hat


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_all():
    print("1. OCR — head roll 30°")
    t, head_roll, eye_no_ocr, g_no   = run_ocr(g_ocr=0.0)
    _,          _, eye_ocr,    g_ocr_ = run_ocr(g_ocr=0.3)

    print("2. K_grav convergence sweep")
    kgrav_results = run_kgrav_sweep()

    print("3. Somatogravic — forward acceleration 5 m/s²")
    ts, a_fwd_arr, eye_pitch, g_somato = run_somatogravic()

    # ── True specific force after 30° CW roll (for reference line) ────────────
    phi_max = np.radians(30.0)
    g_hat_y_true = -G0 * np.sin(phi_max)   # expected: −G0·sin(30°) ≈ −4.9 m/s²

    fig = plt.figure(figsize=(14, 11))
    fig.suptitle('Otolith model — gravity estimator, OCR, somatogravic',
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: OCR head roll & g_hat ─────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, head_roll, 'k', lw=1.5, label='head roll (CW = +)')
    ax0.set_ylabel('deg')
    ax0.set_title('Head roll stimulus')
    ax0.axhline(30, color='gray', ls=':', lw=0.8)
    ax0.legend(fontsize=8)
    ax0.set_xlim(t[0], t[-1])

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, g_no[:, 0], color='steelblue',  lw=1.5, label='g_hat[x]')
    ax1.plot(t, g_no[:, 1], color='firebrick',  lw=1.5, label='g_hat[y]')
    ax1.axhline(g_hat_y_true, color='firebrick', ls='--', lw=0.8,
                label=f'−G0·sin(30°) = {g_hat_y_true:.2f}')
    ax1.set_ylabel('m/s²')
    ax1.set_title(f'Gravity estimator (K_grav={0.5})')
    ax1.legend(fontsize=8)
    ax1.set_xlim(t[0], t[-1])

    # ── Row 1: OCR eye response & K_grav sweep ───────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, eye_no_ocr, color='gray',      lw=1.5, label='g_ocr = 0 (off)')
    ax2.plot(t, eye_ocr,    color='darkorange', lw=1.5, label='g_ocr = 0.3')
    expected_ocr = -0.3 * 30.0
    ax2.axhline(expected_ocr, color='darkorange', ls='--', lw=0.8,
                label=f'expected {expected_ocr:.1f}°')
    ax2.set_ylabel('deg')
    ax2.set_title('OCR: eye roll (CCW = −)')
    ax2.legend(fontsize=8)
    ax2.set_xlim(t[0], t[-1])
    yspan = max(abs(np.min(eye_ocr)), abs(np.max(eye_ocr)), 1.0) * 1.4
    ax2.set_ylim(-yspan, yspan)

    ax3 = fig.add_subplot(gs[1, 1])
    colors = ['steelblue', 'darkorange', 'forestgreen']
    for (K, tk, gy), col in zip(kgrav_results, colors):
        ax3.plot(tk, gy, color=col, lw=1.5, label=f'K_grav={K}')
    ax3.axhline(g_hat_y_true, color='k', ls='--', lw=0.8,
                label=f'steady-state {g_hat_y_true:.2f}')
    ax3.set_ylabel('g_hat[y]  (m/s²)')
    ax3.set_title('K_grav convergence after 30° roll')
    ax3.legend(fontsize=8)
    ax3.set_xlim(tk[0], tk[-1])

    # ── Row 2: Somatogravic ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(ts, a_fwd_arr, 'k', lw=1.5)
    ax4.set_ylabel('m/s²')
    ax4.set_title('Linear accel (z+, forward/nasal)')
    ax4.set_ylim(-1, 7)
    ax4.set_xlim(ts[0], ts[-1])
    ax4.set_xlabel('time (s)')

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(ts, eye_pitch, color='steelblue', lw=1.5, label='eye pitch (−y = down)')
    ax5.plot(ts, g_somato[:, 2], color='firebrick', lw=1.2, ls='--', label='g_hat[z] / G0 × 57.3°')
    ax5.set_ylabel('deg  /  m/s²')
    ax5.set_title('Somatogravic: eye pitch + g_hat[z]')
    ax5.legend(fontsize=8)
    ax5.set_xlim(ts[0], ts[-1])
    ax5.set_xlabel('time (s)')
    yspan = max(abs(np.nanmin(eye_pitch)), abs(np.nanmax(eye_pitch)),
                abs(np.nanmax(g_somato[:, 2])), 1.0) * 1.4
    ax5.set_ylim(-yspan, yspan)

    for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
        ax.axhline(0, color='k', lw=0.4)
        ax.grid(True, alpha=0.3)

    path = os.path.join(OUT, 'otolith_demo.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f'  Saved {os.path.abspath(path)}')
    plt.close(fig)


if __name__ == '__main__':
    print('=== Otolith Demo ===\n')
    plot_all()
    print('\nDone.')
