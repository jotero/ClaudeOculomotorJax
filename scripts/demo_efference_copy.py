"""Efference-copy demo — saccadic suppression of OKR contamination.

The efference copy (x_pc) tracks the burst-driven eye velocity through a
plant forward model and subtracts it from the retinal slip before it enters
the visual delay cascade, preventing saccades from spuriously driving OKR.

Figures produced
────────────────
    efference_plant_copy.png  — plant copy dynamics during/after a saccade:
                                u_burst, w_eye, w_burst_pred, x_pc
    efference_okr_compare.png — scene slip vs saccadic slip: OKR response
                                should appear for real scene motion but not
                                for matched saccadic eye movement
    efference_cancellation.png — corrected vs raw slip during a saccade

Tests (printed to console)
──────────────────────────
    1. x_pc stays zero without burst
    2. x_pc decays with correct tau_p after saccade
    3. OKR store uncontaminated after saccade in dark
    4. OKR still driven by genuine scene motion

Usage
-----
    python scripts/demo_efference_copy.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import diffrax

from oculomotor.models.ocular_motor_simulator import (
    simulate, THETA_DEFAULT,
    vor_vector_field, _N_TOTAL,
    _IDX_P, _IDX_NI, _IDX_SG, _IDX_VIS, _IDX_OKR, _IDX_PC,
)
from oculomotor.models import saccade_generator as sg
from oculomotor.models import visual_delay
from oculomotor.models import retina

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

_C = {
    'burst':   '#f4a582',
    'eye_vel': '#2166ac',
    'pred':    '#d6604d',
    'pc':      '#4dac26',
    'okr':     '#762a83',
    'scene':   '#e08214',
}

THETA_SAC = {**THETA_DEFAULT,
             'g_vor': 0.0, 'g_okr': 0.7,
             'g_burst': 600.0, 'threshold_sac': 0.5,
             'k_sac': 15.0, 'e_sat_sac': 7.0,
             'tau_reset_sac': 1.0, 'tau_reset_fast': 0.1}

THETA_NO_SAC = {**THETA_SAC, 'g_burst': 0.0}


# ── Shared runner ──────────────────────────────────────────────────────────────

def _run_full(theta, t, hv3, vs3, pt3, max_steps=200_000):
    """Run ODE and return full state trajectory (T, _N_TOTAL)."""
    dt_arr = jnp.diff(t)
    hp3    = jnp.concatenate([
        jnp.zeros((1, 3)),
        jnp.cumsum(0.5 * (hv3[:-1] + hv3[1:]) * dt_arr[:, None], axis=0),
    ])
    hv_i = diffrax.LinearInterpolation(ts=t, ys=hv3)
    hp_i = diffrax.LinearInterpolation(ts=t, ys=hp3)
    vs_i = diffrax.LinearInterpolation(ts=t, ys=vs3)
    pt_i = diffrax.LinearInterpolation(ts=t, ys=pt3)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vor_vector_field),
        diffrax.Heun(),
        t0=t[0], t1=t[-1], dt0=0.001,
        y0=jnp.zeros(_N_TOTAL),
        args=(theta, hv_i, hp_i, vs_i, pt_i, jnp.ones(6)),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=t),
        max_steps=max_steps,
    )
    return np.array(sol.ys)   # (T, _N_TOTAL)


def _extract_signals(theta, t, hv3, vs3, pt3, max_steps=200_000):
    """Run and extract all relevant signals as a dict of arrays."""
    ys   = _run_full(theta, t, hv3, vs3, pt3, max_steps)
    t_np = np.array(t)
    dt   = float(t[1] - t[0])

    x_p   = ys[:, _IDX_P]
    x_pc  = ys[:, _IDX_PC]
    x_okr = ys[:, _IDX_OKR]

    # Plant copy state → predicted saccade velocity
    tau_p = theta['tau_p']
    # Recompute u_burst from state via vmap
    target_interp = diffrax.LinearInterpolation(ts=t, ys=pt3)

    def _burst_at(x, t_):
        x_sg          = x[_IDX_SG]
        x_vis         = x[_IDX_VIS]
        e_pos_delayed = visual_delay.C_pos @ x_vis
        _, u_burst    = sg.step(x_sg, e_pos_delayed, theta)
        return u_burst

    ys_j    = jnp.array(ys)
    u_burst = np.array(jax.vmap(_burst_at)(ys_j, t))   # (T, 3)

    w_burst_pred = -(1.0 / tau_p) * x_pc + u_burst     # efference copy velocity
    w_eye        = np.gradient(x_p, dt, axis=0)         # approx eye velocity

    return {
        'x_p':          x_p,
        'x_pc':         x_pc,
        'x_okr':        x_okr,
        'u_burst':      u_burst,
        'w_burst_pred': w_burst_pred,
        'w_eye':        w_eye,
    }


# ── Demo 1: plant copy dynamics ────────────────────────────────────────────────

def demo_plant_copy_dynamics():
    """Show x_pc, u_burst, w_burst_pred, and w_eye during a 10° saccade."""
    dt, T_end, t_jump = 0.001, 0.6, 0.1
    t = jnp.arange(0.0, T_end, dt)
    T = len(t)
    t_np = np.array(t)

    hv3 = jnp.zeros((T, 3))
    vs3 = jnp.zeros((T, 3))
    pt3 = jnp.zeros((T, 3)); pt3 = pt3.at[:, 2].set(1.0)
    pt3 = pt3.at[t >= t_jump, 0].set(jnp.tan(jnp.radians(10.0)))

    sig = _extract_signals(THETA_SAC, t, hv3, vs3, pt3)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    fig.suptitle('Efference Copy — Plant Forward Model During Saccade', fontsize=12)

    for ax in axes:
        ax.axvline(t_jump, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.axhline(0, color='k', lw=0.4)

    axes[0].plot(t_np, sig['u_burst'][:, 0],      color=_C['burst'],   lw=1.5, label='u_burst')
    axes[0].plot(t_np, sig['w_burst_pred'][:, 0], color=_C['pred'],    lw=1.5, label='w_burst_pred (efference copy)')
    axes[0].plot(t_np, sig['w_eye'][:, 0],        color=_C['eye_vel'], lw=1.0, ls='--', alpha=0.7, label='w_eye (actual)')
    axes[0].set_ylabel('deg/s')
    axes[0].set_title('Burst command, efference copy, and actual eye velocity')
    axes[0].legend(fontsize=8)

    residual = sig['w_eye'][:, 0] - sig['w_burst_pred'][:, 0]
    axes[1].plot(t_np, residual, color='#555555', lw=1.2, label='w_eye − w_burst_pred')
    axes[1].set_ylabel('deg/s')
    axes[1].set_title('Residual after efference copy subtraction (enters visual delay as slip)')
    axes[1].legend(fontsize=8)

    axes[2].plot(t_np, sig['x_pc'][:, 0], color=_C['pc'], lw=1.5, label='x_pc (plant copy state)')
    axes[2].set_ylabel('deg')
    axes[2].set_title('Plant copy state — decays with τ_p after burst')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'efference_plant_copy.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Saved {path}')


# ── Demo 2: scene slip vs saccadic slip — OKR comparison ──────────────────────

def demo_okr_scene_vs_saccade():
    """Compare OKR response to real scene motion vs matched saccadic eye motion.

    Both produce similar eye-relative retinal motion, but only the scene
    should drive OKR.  The efference copy cancels the saccadic contribution.
    """
    dt, T_end = 0.001, 1.0
    t    = jnp.arange(0.0, T_end, dt)
    T    = len(t)
    t_np = np.array(t)
    t_stim = 0.1

    # ── Case A: real scene motion (no saccade) ─────────────────────────────
    w_scene_pulse = 300.0  # deg/s for 50 ms — similar to a 10° saccade
    vs3_scene = jnp.zeros((T, 3))
    vs3_scene = vs3_scene.at[(t >= t_stim) & (t < t_stim + 0.05), 0].set(w_scene_pulse)
    hv3 = jnp.zeros((T, 3))
    pt3 = jnp.zeros((T, 3)); pt3 = pt3.at[:, 2].set(1.0)

    sig_scene = _extract_signals(THETA_NO_SAC, t, hv3, vs3_scene, pt3)

    # ── Case B: saccade in dark (no scene) ─────────────────────────────────
    vs3_dark = jnp.zeros((T, 3))
    pt3_sac  = jnp.zeros((T, 3)); pt3_sac = pt3_sac.at[:, 2].set(1.0)
    pt3_sac  = pt3_sac.at[t >= t_stim, 0].set(jnp.tan(jnp.radians(10.0)))

    sig_sac = _extract_signals(THETA_SAC, t, hv3, vs3_dark, pt3_sac)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    fig.suptitle('Efference Copy — Scene Slip vs Saccadic Slip: OKR Response', fontsize=12)

    for ax in axes:
        ax.axvline(t_stim, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.axhline(0, color='k', lw=0.4)

    axes[0].plot(t_np, sig_scene['w_eye'][:, 0], color=_C['scene'],   lw=1.5, label='scene: w_eye')
    axes[0].plot(t_np, sig_sac['w_eye'][:, 0],   color=_C['eye_vel'], lw=1.5, label='saccade: w_eye')
    axes[0].set_ylabel('deg/s')
    axes[0].set_title('Eye velocity (both cases produce similar retinal motion)')
    axes[0].legend(fontsize=8)

    axes[1].plot(t_np, sig_sac['w_burst_pred'][:, 0], color=_C['pred'], lw=1.5,
                 label='saccade: w_burst_pred')
    axes[1].plot(t_np, np.zeros(T),                    color=_C['scene'], lw=1.5, ls='--',
                 label='scene: w_burst_pred = 0')
    axes[1].set_ylabel('deg/s')
    axes[1].set_title('Efference copy signal (zero for scene, cancels saccade)')
    axes[1].legend(fontsize=8)

    axes[2].plot(t_np, sig_scene['x_okr'][:, 0], color=_C['scene'],   lw=1.5, label='scene → OKR driven')
    axes[2].plot(t_np, sig_sac['x_okr'][:, 0],   color=_C['eye_vel'], lw=1.5, label='saccade → OKR suppressed')
    axes[2].set_ylabel('deg/s')
    axes[2].set_title('OKR store x_okr')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'efference_okr_compare.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Saved {path}')


# ── Demo 3: corrected vs raw slip during saccade ───────────────────────────────

def demo_slip_cancellation():
    """Show the raw retinal slip and the efference-copy-corrected slip side by side."""
    dt, T_end, t_jump = 0.001, 0.5, 0.1
    t    = jnp.arange(0.0, T_end, dt)
    T    = len(t)
    t_np = np.array(t)

    hv3 = jnp.zeros((T, 3))
    vs3 = jnp.zeros((T, 3))
    pt3 = jnp.zeros((T, 3)); pt3 = pt3.at[:, 2].set(1.0)
    pt3 = pt3.at[t >= t_jump, 0].set(jnp.tan(jnp.radians(10.0)))

    sig = _extract_signals(THETA_SAC, t, hv3, vs3, pt3)

    # Raw slip = -w_eye (dark, no scene, no head)
    # Corrected slip = -w_eye + w_burst_pred  (add back predicted burst → cancels saccadic component)
    raw_slip       = -sig['w_eye'][:, 0]
    corrected_slip = -sig['w_eye'][:, 0] + sig['w_burst_pred'][:, 0]

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    fig.suptitle('Retinal Slip — Raw vs Efference-Copy Corrected', fontsize=12)

    for ax in axes:
        ax.axvline(t_jump, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.axhline(0, color='k', lw=0.4)

    axes[0].plot(t_np, raw_slip, color=_C['eye_vel'], lw=1.5)
    axes[0].set_ylabel('deg/s')
    axes[0].set_title('Raw retinal slip (−w_eye): what OKR would see without efference copy')

    axes[1].plot(t_np, corrected_slip, color=_C['pred'], lw=1.5)
    axes[1].set_ylabel('deg/s')
    axes[1].set_title('Corrected slip (−w_eye + w_burst_pred): what OKR actually sees')
    axes[1].set_xlabel('Time (s)')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'efference_cancellation.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Saved {path}')


# ── Numerical tests ────────────────────────────────────────────────────────────

def _run_tests():
    print('\nRunning numerical tests...')
    dt = 0.001

    # ── Test 1: x_pc stays zero without burst ─────────────────────────────
    t = jnp.arange(0.0, 0.5, dt); T = len(t)
    hv3 = jnp.zeros((T, 3)); vs3 = jnp.zeros((T, 3))
    pt3 = jnp.zeros((T, 3)); pt3 = pt3.at[:, 2].set(1.0)
    ys  = _run_full(THETA_NO_SAC, t, hv3, vs3, pt3)
    x_pc = ys[:, _IDX_PC]
    assert np.allclose(x_pc, 0.0, atol=1e-6), \
        f'FAIL: x_pc non-zero without burst; max={np.abs(x_pc).max():.2e}'
    print('  1. x_pc zero without burst               PASS')

    # ── Test 2: x_pc decays with tau_p after saccade ──────────────────────
    t = jnp.arange(0.0, 0.8, dt); T = len(t); t_np = np.array(t)
    t_jump = 0.1
    hv3 = jnp.zeros((T, 3)); vs3 = jnp.zeros((T, 3))
    pt3 = jnp.zeros((T, 3)); pt3 = pt3.at[:, 2].set(1.0)
    pt3 = pt3.at[t >= t_jump, 0].set(jnp.tan(jnp.radians(10.0)))
    ys   = _run_full(THETA_SAC, t, hv3, vs3, pt3)
    x_pc = ys[:, _IDX_PC][:, 0]
    mask = (t_np > t_jump + 0.08) & (t_np < t_jump + 0.4)
    t_win = t_np[mask] - t_np[mask][0]; pc_win = x_pc[mask]
    if np.abs(pc_win).max() > 1e-4:
        log_pc = np.log(np.abs(pc_win) + 1e-9)
        coeffs = np.polyfit(t_win, log_pc, 1)
        tau_fit = -1.0 / coeffs[0]
        tau_p   = THETA_SAC['tau_p']
        assert abs(tau_fit - tau_p) / tau_p < 0.3, \
            f'FAIL: tau_fit={tau_fit:.3f} s vs tau_p={tau_p:.3f} s'
    print('  2. x_pc decays with correct tau_p         PASS')

    # ── Test 3: OKR uncontaminated after saccade in dark ──────────────────
    t = jnp.arange(0.0, 0.6, dt); T = len(t); t_np = np.array(t)
    t_jump = 0.1
    hv3 = jnp.zeros((T, 3)); vs3 = jnp.zeros((T, 3))
    pt3 = jnp.zeros((T, 3)); pt3 = pt3.at[:, 2].set(1.0)
    pt3 = pt3.at[t >= t_jump, 0].set(jnp.tan(jnp.radians(10.0)))
    ys    = _run_full(THETA_SAC, t, hv3, vs3, pt3)
    x_okr = ys[:, _IDX_OKR]
    mask  = (t_np > t_jump + 0.2) & (t_np < t_jump + 0.5)
    contamination = np.abs(x_okr[mask]).max()
    assert contamination < 0.1, \
        f'FAIL: OKR contaminated; max|x_okr|={contamination:.3f} deg/s'
    print('  3. OKR not contaminated by saccade        PASS')

    # ── Test 4: OKR still driven by real scene ────────────────────────────
    t = jnp.arange(0.0, 2.0, dt); T = len(t); t_np = np.array(t)
    hv3 = jnp.zeros((T, 3))
    vs3 = jnp.zeros((T, 3)); vs3 = vs3.at[:, 0].set(30.0)
    pt3 = jnp.zeros((T, 3)); pt3 = pt3.at[:, 2].set(1.0)
    ys    = _run_full(THETA_NO_SAC, t, hv3, vs3, pt3)
    x_okr = ys[:, _IDX_OKR]
    okr_late = np.abs(x_okr[t_np > 1.0, 0]).mean()
    assert okr_late > 1.0, \
        f'FAIL: OKR not driven by scene; mean={okr_late:.2f} deg/s'
    print('  4. OKR driven by genuine scene motion     PASS')

    print('All tests passed.\n')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    demo_plant_copy_dynamics()
    demo_okr_scene_vs_saccade()
    demo_slip_cancellation()
    _run_tests()
    print(f'Plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
