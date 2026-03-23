"""Main script: generate synthetic data, fit model, produce diagnostic plots.

Usage:
    python scripts/run_recovery.py

Outputs (saved to outputs/):
    loss_curve.png
    parameter_trajectories.png
    bode_plot.png
    time_domain.png
    residuals.png
"""

import sys
import os

# Ensure project root is on path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp

from oculomotor.sim.synthetic import generate_dataset, THETA_TRUE
from oculomotor.sim.stimuli import FREQUENCIES_HZ, AMPLITUDE_DEG_S
from oculomotor.fitting.optimize import fit
from oculomotor.models.vor import simulate

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

TAU_C_FIXED = THETA_TRUE['tau_c']  # 5.0 s — assumed known, not optimized
TAU_S_FIXED = THETA_TRUE['tau_s']  # 0.005 s — assumed known, not optimized

THETA_INIT = {
    'g_vor':  0.7,    # true: 1.0
    'tau_i':  10.0,   # true: 25.0
    'tau_p':  0.30,   # true: 0.15  (lower-bounded at 0.05 in reparameterization)
    'tau_vs': 20.0,   # true: 50.0  (τ_eff_init ≈ 8 s vs true ≈ 20 s)
    'K_vs':   0.08,   # true: 0.03
}

N_STEPS = 2000  # used only if METHOD='adam'
METHOD = 'lbfgs'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_bode(theta, frequencies=FREQUENCIES_HZ, amplitude=AMPLITUDE_DEG_S,
                 duration=20.0, sample_rate=500.0):
    """Compute gain and phase at each frequency for a given theta."""
    gains = []
    phases = []
    for freq in frequencies:
        t = jnp.arange(0.0, duration, 1.0 / sample_rate)
        head_vel = amplitude * jnp.sin(2.0 * jnp.pi * freq * t)
        eye_pos = simulate(theta, t, head_vel)[:, 0]   # horizontal component

        # Steady-state: use last half of the trial
        half = len(t) // 2
        t_ss = np.array(t[half:])
        eye_ss = np.array(eye_pos[half:])
        head_vel_ss = np.array(head_vel[half:])

        # Expected: VOR drives eye_pos ≈ -g_eff * (head_vel / (2*pi*f))
        # Estimate gain as amplitude ratio, phase via cross-correlation
        omega = 2.0 * np.pi * freq
        # Reference: integrated head_vel gives head_pos; VOR ideal = -head_pos
        head_pos_ss = -np.cumsum(head_vel_ss) / sample_rate  # approximate
        # Fit gain and phase with least-squares sinusoid fitting
        A = np.column_stack([
            np.sin(omega * t_ss),
            np.cos(omega * t_ss),
        ])
        # Fit eye_pos
        c_eye, _, _, _ = np.linalg.lstsq(A, eye_ss, rcond=None)
        amp_eye = np.sqrt(c_eye[0]**2 + c_eye[1]**2)
        phase_eye = np.arctan2(c_eye[1], c_eye[0])

        # Fit head_pos (ideal reference = -integral of head_vel)
        head_pos_ref = amplitude / omega * np.cos(omega * t_ss)  # integral of sin
        c_hp, _, _, _ = np.linalg.lstsq(A, head_pos_ref, rcond=None)
        amp_hp = np.sqrt(c_hp[0]**2 + c_hp[1]**2)
        phase_hp = np.arctan2(c_hp[1], c_hp[0])

        gain = amp_eye / amp_hp
        phase_diff = np.degrees(phase_eye - phase_hp)
        # Wrap to [-180, 180]
        phase_diff = (phase_diff + 180) % 360 - 180

        gains.append(gain)
        phases.append(phase_diff)

    return np.array(gains), np.array(phases)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_curve(history):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(history['loss'])
    ax.set_xlabel('Optimisation step')
    ax.set_ylabel('MSE loss (deg²)')
    ax.set_title('Loss curve')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'loss_curve.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_parameter_trajectories(history):
    keys = ['g_vor', 'tau_i', 'tau_p', 'tau_vs', 'K_vs']
    true_vals = {k: THETA_TRUE[k] for k in keys}
    labels = {'g_vor': r'$g_{vor}$', 'tau_i': r'$\tau_i$ (s)',
              'tau_p': r'$\tau_p$ (s)', 'tau_vs': r'$\tau_{vs}$ (s)',
              'K_vs': r'$K_{vs}$ (1/s)'}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, key in zip(axes.flat, keys):
        ax.plot(history[key], color='steelblue', lw=1.5, label='fitted')
        ax.axhline(true_vals[key], color='tomato', ls='--', lw=1.5, label='true')
        ax.set_xlabel('Step')
        ax.set_ylabel(labels[key])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Parameter trajectories')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'parameter_trajectories.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_bode(theta_true, theta_fit):
    freqs = FREQUENCIES_HZ
    gains_true, phases_true = compute_bode(theta_true, freqs)
    gains_fit, phases_fit = compute_bode(theta_fit, freqs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax1.semilogx(freqs, gains_true, 'o--', color='tomato', label='Ground truth')
    ax1.semilogx(freqs, gains_fit, 's-', color='steelblue', label='Fitted')
    ax1.set_ylabel('VOR Gain (eye/head)')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)
    ax1.set_title('Bode plot')

    ax2.semilogx(freqs, phases_true, 'o--', color='tomato', label='Ground truth')
    ax2.semilogx(freqs, phases_fit, 's-', color='steelblue', label='Fitted')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (deg)')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'bode_plot.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_time_domain(stimuli, observations, theta_fit):
    n = len(stimuli)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes = axes.flat

    condition_labels = [f'{f} Hz' for f in FREQUENCIES_HZ] + ['Step']

    for i, ((t, head_vel), eye_obs) in enumerate(zip(stimuli, observations)):
        eye_pred = np.array(simulate(theta_fit, t, head_vel))[:, 0]
        t_np = np.array(t)
        # Show only first 5 s for clarity (sinusoids); full trial for step
        mask = t_np <= 5.0 if i < len(FREQUENCIES_HZ) else np.ones(len(t_np), dtype=bool)
        ax = axes[i]
        ax.plot(t_np[mask], np.array(eye_obs)[mask], alpha=0.5, lw=0.8,
                color='gray', label='Observed')
        ax.plot(t_np[mask], eye_pred[mask], lw=1.5, color='steelblue', label='Predicted')
        ax.set_title(condition_labels[i])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Eye pos (deg)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, rows * cols):
        axes[j].set_visible(False)

    fig.suptitle('Time-domain overlays (fitted model)')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'time_domain.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_residuals(stimuli, observations, theta_fit):
    n = len(stimuli)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes = axes.flat

    condition_labels = [f'{f} Hz' for f in FREQUENCIES_HZ] + ['Step']

    for i, ((t, head_vel), eye_obs) in enumerate(zip(stimuli, observations)):
        eye_pred = np.array(simulate(theta_fit, t, head_vel))[:, 0]
        residual = np.array(eye_obs) - eye_pred
        t_np = np.array(t)
        mask = t_np <= 5.0 if i < len(FREQUENCIES_HZ) else np.ones(len(t_np), dtype=bool)
        ax = axes[i]
        ax.plot(t_np[mask], residual[mask], lw=0.8, color='tomato')
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_title(condition_labels[i])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Residual (deg)')
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, rows * cols):
        axes[j].set_visible(False)

    fig.suptitle('Residuals (observed − predicted)')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'residuals.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Generating synthetic data ===")
    stimuli, observations = generate_dataset(theta=THETA_TRUE, seed=0)
    print(f"  {len(stimuli)} conditions, {len(stimuli[0][0])} samples each")

    print("\n=== Fitting model ===")
    theta_fit, history = fit(
        stimuli, observations,
        theta_init=THETA_INIT,
        tau_c=TAU_C_FIXED,
        tau_s=TAU_S_FIXED,
        n_steps=N_STEPS,
        method=METHOD,
        print_every=25,
    )

    print(f"\n  tau_c = {TAU_C_FIXED:.4f} s  [FIXED — not optimized]")
    print(f"  tau_s = {TAU_S_FIXED:.4f} s  [FIXED — not optimized]")
    print("\n=== Parameter recovery summary ===")
    for key in ('g_vor', 'tau_i', 'tau_p', 'tau_vs', 'K_vs'):
        true_val = THETA_TRUE[key]
        fit_val = float(theta_fit[key])
        err = abs(fit_val - true_val) / abs(true_val) * 100
        status = "OK" if err <= 10.0 else "FAIL"
        print(f"  {key:8s}: true={true_val:.4f}  fit={fit_val:.4f}  err={err:.1f}%  [{status}]")

    print("\n=== Saving diagnostic plots ===")
    plot_loss_curve(history)
    plot_parameter_trajectories(history)
    plot_bode(THETA_TRUE, theta_fit)
    plot_time_domain(stimuli, observations, theta_fit)
    plot_residuals(stimuli, observations, theta_fit)

    print("\nDone. All plots saved to outputs/")


if __name__ == '__main__':
    main()
