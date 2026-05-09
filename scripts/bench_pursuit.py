"""Smooth pursuit benchmarks — velocity range, sinusoidal, signal cascade.

Usage:
    python -X utf8 scripts/bench_pursuit.py
    python -X utf8 scripts/bench_pursuit.py --show
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
    PARAMS_DEFAULT, with_brain, with_sensory, simulate,
    _IDX_SG, _IDX_VIS, _IDX_PURSUIT,
)
from oculomotor.sim import kinematics as km
from oculomotor.models.brain_models.perception_cyclopean import C_pos  # noqa: F401
from oculomotor.analysis import ax_fmt, extract_burst, extract_sg, ni_net

SHOW  = '--show' in sys.argv
DT    = 0.001
THETA = PARAMS_DEFAULT
THETA_NOISELESS = with_brain(with_sensory(THETA, sigma_canal=0.0, sigma_pos=0.0, sigma_vel=0.0), sigma_acc=0.0)


def _ramp(t_np, vel, t_jump=0.2):
    T = len(t_np)
    tgt = np.where(t_np >= t_jump, vel * (t_np - t_jump), 0.0)
    pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
    pt3[:, 0] = np.tan(np.radians(tgt))
    vt3 = np.zeros((T, 3))
    vt3[:, 0] = np.where(t_np >= t_jump, float(vel), 0.0).astype(np.float32)
    return tgt, jnp.array(pt3), jnp.array(vt3)


def _run(theta, t_np, pt3, vt3=None, target_present=True, key=0):
    t  = jnp.array(t_np)
    T  = len(t)
    tp = jnp.ones(T) if target_present else jnp.zeros(T)
    return simulate(theta, t,
                    target=km.build_target(t_np, lin_pos=np.array(pt3)),
                    scene_present_array=jnp.ones(T), target_present_array=tp,
                    max_steps=int(len(t_np) * 1.05) + 500,
                    return_states=True, key=jax.random.PRNGKey(key))


# ── Figure 1: velocity range comparison ──────────────────────────────────────

def _velocity_range(show):
    velocities = [5.0, 10.0, 20.0, 40.0]
    T_end, T_jump = 3.0, 0.2
    t_np = np.arange(0.0, T_end, DT)

    theta_pur = THETA
    theta_nop = with_brain(THETA, K_pursuit=0.0, K_phasic_pursuit=0.0)

    n_rows, n_cols = 3, len(velocities)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 2.5 * n_rows), sharex=True)
    fig.suptitle('Smooth Pursuit — Velocity Range  (blue: pursuit on, gray: saccades only)', fontsize=11)
    row_labels = ['Position (deg)', 'Eye velocity (deg/s)', 'Pursuit drive u_pursuit (deg/s)']
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8)

    for ci, vel in enumerate(velocities):
        tgt, pt3, vt3 = _ramp(t_np, vel, T_jump)
        st_pur = _run(theta_pur, t_np, pt3, vt3, key=ci)
        st_nop = _run(theta_nop, t_np, pt3,      key=ci + 10)

        eye_pur = (np.array(st_pur.plant[:, 0]) + np.array(st_pur.plant[:, 3])) / 2.0
        eye_nop = (np.array(st_nop.plant[:, 0]) + np.array(st_nop.plant[:, 3])) / 2.0
        ev_pur  = np.gradient(eye_pur, DT)
        ev_nop  = np.gradient(eye_nop, DT)
        u_pur   = np.array(st_pur.brain[:, _IDX_PURSUIT])[:, 0]

        axes[0, ci].set_title(f'{vel:.0f} deg/s', fontsize=10)
        for ax in axes[:, ci]:
            ax.axvline(T_jump, color='gray', lw=0.5, ls='--', alpha=0.4)

        axes[0, ci].plot(t_np, tgt,     color=utils.C['target'],    lw=1.5, label='target')
        axes[0, ci].plot(t_np, eye_nop, color=utils.C['dark'],      lw=1.0, ls='--', label='no pursuit')
        axes[0, ci].plot(t_np, eye_pur, color=utils.C['eye'],       lw=1.5, label='pursuit')
        ax_fmt(axes[0, ci])
        if ci == 0: axes[0, ci].legend(fontsize=7)

        axes[1, ci].axhline(vel, color=utils.C['target'], lw=0.8, ls=':', alpha=0.7,
                            label=f'{vel} deg/s')
        axes[1, ci].plot(t_np, ev_nop, color=utils.C['dark'], lw=0.8, ls='--', label='no pursuit')
        axes[1, ci].plot(t_np, ev_pur, color=utils.C['eye'],  lw=1.2, label='pursuit')
        axes[1, ci].set_ylim(-max(vel*0.15, 3), vel * 1.35)
        ax_fmt(axes[1, ci])
        if ci == 0: axes[1, ci].legend(fontsize=7)

        K_ph  = float(THETA.brain.K_phasic_pursuit)
        K_int = float(THETA.brain.K_pursuit)
        tau_p = float(THETA.brain.tau_pursuit)
        dx_pur     = np.gradient(u_pur, DT)
        e_pred_est = (dx_pur + u_pur / tau_p) / K_int
        phasic     = K_ph * e_pred_est
        u_total    = u_pur + phasic

        axes[2, ci].axhline(vel, color=utils.C['target'], lw=0.8, ls=':', alpha=0.7)
        axes[2, ci].plot(t_np, u_total, color=utils.C['pursuit'], lw=1.8, label='u_pursuit (total)')
        axes[2, ci].plot(t_np, u_pur,   color='#1a6ebd',          lw=1.1, ls='--', label='integrator x_p')
        axes[2, ci].plot(t_np, phasic,  color='#d94801',          lw=1.1, ls='--', label='phasic K·e_pred')
        axes[2, ci].set_ylim(-vel * 0.1, vel * 1.35)
        ax_fmt(axes[2, ci])
        axes[2, ci].set_xlabel('Time (s)', fontsize=8)
        if ci == 0: axes[2, ci].legend(fontsize=7)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'pursuit_velocity_range', show=show, params=THETA,
                              conditions='Lit, foveal target ramping at 5–80 °/s (constant velocity pursuit)')
    return utils.fig_meta(path, rp,
        title='Smooth Pursuit — Velocity Range',
        description='Step-ramp target at 5, 10, 20, 40 deg/s. '
                    'Blue: smooth pursuit enabled. Gray: saccades only (pursuit off). '
                    'Rows: position, velocity, pursuit drive (total + integrator + phasic).',
        expected='At 5–10 deg/s: pursuit tracks closely, steady-state gain > 0.8. '
                 'At higher velocities: catch-up saccades + partial pursuit.',
        citation='Lisberger & Westbrook (1985) J Neurosci; Rashbass (1961)',
        fig_type='behavior')


# ── Figure 2: sinusoidal pursuit ─────────────────────────────────────────────

def _sinusoidal(show):
    """Sinusoidal target: horizontal + vertical, 3 frequencies."""
    freqs = [0.2, 0.5, 1.0]   # Hz
    AMP   = 15.0               # deg/s peak velocity
    T_END = 12.0
    t_np  = np.arange(0.0, T_END, DT)
    T     = len(t_np)

    # 2D: H sinusoid at freq f, V sinusoid at 2*freq (Lissajous-like)
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    fig.suptitle(f'Sinusoidal Pursuit — H: A·sin(2πft), V: A·sin(4πft),  A = {AMP:.0f} deg/s peak',
                 fontsize=11)
    row_labels = ['H position (deg)', 'V position (deg)', 'H+V trajectory (deg)']
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8)

    for ci, freq in enumerate(freqs):
        axes[0, ci].set_title(f'{freq} Hz', fontsize=11)
        # Position = integral of velocity
        tgt_h = -(AMP / (2 * np.pi * freq))     * np.cos(2 * np.pi * freq       * t_np)
        tgt_v = -(AMP / (2 * np.pi * 2 * freq)) * np.cos(2 * np.pi * 2 * freq * t_np)
        vel_h = AMP * np.sin(2 * np.pi * freq       * t_np)
        vel_v = AMP * np.sin(2 * np.pi * 2 * freq * t_np)

        pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
        pt3[:, 0] = np.tan(np.radians(tgt_h))
        pt3[:, 1] = np.tan(np.radians(tgt_v))
        vt3 = np.zeros((T, 3))
        vt3[:, 0] = vel_h.astype(np.float32)
        vt3[:, 1] = vel_v.astype(np.float32)

        st   = _run(THETA, t_np, jnp.array(pt3), jnp.array(vt3), key=ci + 20)
        # Version eye position = mean of left and right
        eye_L = np.array(st.plant[:, :2])
        eye_R = np.array(st.plant[:, 3:5])
        eye   = (eye_L + eye_R) / 2.0

        mask = t_np > 2.0  # skip warm-up
        axes[0, ci].plot(t_np[mask], tgt_h[mask], color=utils.C['target'], lw=1.2, label='target H')
        axes[0, ci].plot(t_np[mask], eye[mask, 0], color=utils.C['eye'],   lw=1.5, label='eye H (version)')
        ax_fmt(axes[0, ci]); axes[0, ci].legend(fontsize=7)
        absmax = max(np.abs(tgt_h[mask]).max(), np.abs(eye[mask, 0]).max()) * 1.1
        axes[0, ci].set_ylim(-absmax, absmax)

        axes[1, ci].plot(t_np[mask], tgt_v[mask], color=utils.C['target'], lw=1.2, ls='--', label='target V')
        axes[1, ci].plot(t_np[mask], eye[mask, 1], color=utils.C['eye'],   lw=1.5, ls='--', label='eye V (version)')
        ax_fmt(axes[1, ci]); axes[1, ci].legend(fontsize=7)
        absmax = max(np.abs(tgt_v[mask]).max(), np.abs(eye[mask, 1]).max()) * 1.1
        axes[1, ci].set_ylim(-absmax, absmax)

        axes[2, ci].plot(tgt_h[mask], tgt_v[mask], color=utils.C['target'], lw=1.0, label='target')
        axes[2, ci].plot(eye[mask, 0], eye[mask, 1], color=utils.C['eye'],  lw=1.5, label='eye')
        axes[2, ci].set_xlabel('H (deg)', fontsize=8)
        axes[2, ci].set_aspect('equal'); axes[2, ci].legend(fontsize=7)
        axes[2, ci].grid(True, alpha=0.25)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'pursuit_sinusoidal', show=show, params=THETA,
                              conditions='Lit, foveal target sinusoidal in horizontal position')
    return utils.fig_meta(path, rp,
        title='Sinusoidal Pursuit (H + V)',
        description='Horizontal sinusoidal target at 0.2, 0.5, 1.0 Hz (peak 15 deg/s). '
                    'Vertical at 2× horizontal frequency for Lissajous trajectory. '
                    'Rows: H position, V position, 2D trajectory.',
        expected='Low freq (0.2 Hz): near-unity gain, small phase lag. '
                 'High freq (1.0 Hz): gain < 1, larger lag, catch-up saccades.',
        citation='Lisberger et al. (1981) J Neurophysiol',
        fig_type='behavior')


# ── Figure 3: pursuit signal cascade ─────────────────────────────────────────

def _cascade(show):
    """Signal cascade for 20 deg/s pursuit: target on 0.2–2.0 s, then target stops."""
    vel    = 20.0
    t_jump = 0.2
    t_stop = 2.0   # target stops; eye keeps moving (pursuit integrator decays slowly)
    T_end  = 5.0
    t_np   = np.arange(0.0, T_end, DT)
    T      = len(t_np)

    # Position: ramp until t_stop, then hold at final position
    tgt_pos = float(vel * (t_stop - t_jump))
    tgt = np.where(t_np < t_jump, 0.0,
          np.where(t_np < t_stop, vel * (t_np - t_jump), tgt_pos))
    pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
    pt3[:, 0] = np.tan(np.radians(tgt))
    # Velocity: on during ramp, zero after target stops
    vt3 = np.zeros((T, 3))
    vt3[:, 0] = np.where((t_np >= t_jump) & (t_np < t_stop), float(vel), 0.0).astype(np.float32)

    st  = _run(THETA_NOISELESS, t_np, jnp.array(pt3), jnp.array(vt3), key=30)
    sg  = extract_sg(st, THETA)
    eye = (np.array(st.plant[:, 0]) + np.array(st.plant[:, 3])) / 2.0
    ev  = np.gradient(eye, DT)
    x_pur = np.array(st.brain[:, _IDX_PURSUIT])[:, 0]

    n_rows = 7
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.5 * n_rows))
    fig.suptitle(f'Smooth Pursuit Signal Cascade — {vel:.0f} deg/s ramp, target stops at {t_stop:.0f} s\n'
                 f'(pursuit integrator persists after target stop, decays with τ ≈ {THETA.brain.tau_pursuit:.0f} s)',
                 fontsize=11)

    vl_start = dict(color='gray',   lw=0.7, ls='--', alpha=0.5)
    vl_stop  = dict(color='tomato', lw=0.9, ls='--', alpha=0.7)
    for ax in axes:
        ax.axvline(t_jump, **vl_start)
        ax.axvline(t_stop, **vl_stop)
        ax_fmt(ax)

    axes[0].plot(t_np, tgt,          color=utils.C['target'],  lw=1.5, label='target pos (stops 2 s)')
    axes[0].plot(t_np, eye,          color=utils.C['eye'],     lw=1.5, label='eye pos')
    axes[0].plot(t_np, ni_net(st)[:,0], color=utils.C['ni'],   lw=0.9, ls='--', label='NI')
    axes[0].set_ylabel('Position (deg)'); axes[0].set_title('Eye + Target Position')
    axes[0].legend(fontsize=8)

    axes[1].axhline(vel, color=utils.C['target'], lw=0.8, ls=':', alpha=0.7, label=f'{vel:.0f} deg/s')
    axes[1].plot(t_np, ev, color=utils.C['eye'], lw=1.2, label='eye vel (persists after stop)')
    axes[1].set_ylabel('Velocity (deg/s)'); axes[1].set_title('Eye Velocity')
    axes[1].legend(fontsize=8)

    # Reconstruct phasic drive from integrator state:
    #   dx_p/dt = -x_p/tau + K_int * e_pred  →  e_pred = (dx_p/dt + x_p/tau) / K_int
    #   phasic  = K_phasic * e_pred  (fast onset, decays to 0 at steady state)
    #   u_total = x_p + phasic        (full pursuit command)
    K_ph  = float(THETA.brain.K_phasic_pursuit)
    K_int = float(THETA.brain.K_pursuit)
    tau_p = float(THETA.brain.tau_pursuit)
    dx_pur     = np.gradient(x_pur, DT)
    e_pred_est = (dx_pur + x_pur / tau_p) / K_int
    phasic     = K_ph * e_pred_est
    u_total    = x_pur + phasic

    axes[2].axhline(vel, color=utils.C['target'], lw=0.8, ls=':', alpha=0.7, label=f'target {vel:.0f} deg/s')
    axes[2].plot(t_np, u_total, color=utils.C['pursuit'],  lw=1.8, label='u_pursuit (total)')
    axes[2].plot(t_np, x_pur,   color='#1a6ebd',           lw=1.3, ls='--', label='integrator x_p (slow)')
    axes[2].plot(t_np, phasic,  color='#d94801',           lw=1.3, ls='--', label='phasic K·e_pred (fast)')
    axes[2].set_ylabel('Pursuit drive (deg/s)'); axes[2].set_title('Pursuit Drive: Integrator + Phasic Feedthrough')
    axes[2].legend(fontsize=8)

    axes[3].plot(t_np, sg['e_pd'][:,0],   color='darkorange', lw=1.0, ls='--', label='e_delayed')
    axes[3].plot(t_np, sg['e_held'][:,0], color=utils.C['vs'], lw=1.8, label='e_held (frozen)')
    axes[3].set_ylabel('Error (deg)'); axes[3].set_title('Visual Cascade Output + Sample-Hold')
    axes[3].legend(fontsize=8)

    axes[4].plot(t_np, sg['z_acc'], color='#e08214', lw=1.5, label='z_acc')
    axes[4].plot(t_np, sg['z_opn'] / 100, color='#1b7837', lw=1.5, label='OPN (norm, 1=tonic)')
    axes[4].axhline(THETA.brain.threshold_acc, color='#e08214', lw=0.8, ls=':')
    axes[4].set_ylim(-0.05, 1.15)
    axes[4].set_ylabel('Accumulator'); axes[4].set_title('Catch-up Saccade Trigger')
    axes[4].legend(fontsize=8)

    axes[5].plot(t_np, sg['u_burst'][:,0], color=utils.C['burst'], lw=1.5, label='burst (catch-up)')
    axes[5].set_ylabel('Burst (deg/s)'); axes[5].set_title('Saccade Burst (Catch-up Saccades)')
    axes[5].legend(fontsize=8)

    axes[6].plot(t_np, sg['z_acc'], color=utils.C['refractory'], lw=1.5, label='z_acc (accumulator)')
    axes[6].axhline(0.5, color='k', lw=0.6, ls='--', alpha=0.4)
    axes[6].set_ylabel('z_acc'); axes[6].set_title('Accumulator (refractory proxy)')
    axes[6].set_xlabel('Time (s)', fontsize=9)
    axes[6].legend(fontsize=8)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'pursuit_cascade', show=show, params=THETA_NOISELESS,
                              conditions='Lit, foveal ramp + catch-up saccades (noiseless cascade trace)')
    return utils.fig_meta(path, rp,
        title='Smooth Pursuit Signal Cascade (Internal)',
        description='Full signal chain for 20 deg/s pursuit: target ramps 0.2–2.0 s then holds. '
                    'Pursuit integrator persists after target stop (τ ≈ 40 s). '
                    'Rows: position, velocity, pursuit drive (total/integrator/phasic), '
                    'visual error, saccade accumulator, burst, refractory state.',
        expected='Phasic drive rises fast at onset, decays to 0 at steady state. '
                 'After target stops at 2 s: phasic reverses (retinal slip now backward), '
                 'integrator decays slowly — eye coasts then corrects with catch-up saccades. '
                 'Integrator TC visible as slow drift in x_p post-stop.',
        citation='Lisberger & Westbrook (1985)',
        fig_type='cascade')


# ── Section entry point ────────────────────────────────────────────────────────

SECTION = dict(
    id='pursuit', title='4. Smooth Pursuit',
    description='Smooth pursuit and catch-up saccades. Tests velocity gain at multiple speeds, '
                'sinusoidal tracking, and the pursuit integrator + saccade interaction cascade.',
)


def run(show=False):
    print('\n=== Smooth Pursuit ===')
    figs = []
    print('  1/3  velocity range …')
    figs.append(_velocity_range(show))
    print('  2/3  sinusoidal pursuit …')
    figs.append(_sinusoidal(show))
    print('  3/3  signal cascade …')
    figs.append(_cascade(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
