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
    efference_okn_debug.png   — OKN nystagmus: VS trajectory with/without
                                saccades, raw vs corrected slip, eye position

Tests (printed to console)
──────────────────────────
    1. x_pc stays zero without burst
    2. x_pc decays with correct tau_p after saccade
    3. OKR store uncontaminated after saccade in dark
    4. OKR still driven by genuine scene motion
    5. VS not contaminated by repeated fast-phase saccades during OKN

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
    ODE_ocular_motor, _N_TOTAL,
    _IDX_C, _IDX_P, _IDX_NI, _IDX_SG, _IDX_VIS, _IDX_VS, _IDX_PC, _IDX_NI_PC,
)
from oculomotor.models import saccade_generator as sg
from oculomotor.models import visual_delay
from oculomotor.models import canal
from oculomotor.models import velocity_storage as vs_mod
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

THETA_SAC = {**THETA_DEFAULT, 'g_burst': 600.0}   # head is still → VOR/OKR inactive naturally

THETA_NO_SAC = {**THETA_SAC, 'g_burst': 0.0}

# OKN context: full default params + saccades enabled (same as demo_vor OKN section)
THETA_OKN     = {**THETA_DEFAULT, 'g_burst': 600.0}
THETA_OKN_OFF = {**THETA_DEFAULT, 'g_burst': 0.0}   # no saccades — reference


# ── Shared runner ──────────────────────────────────────────────────────────────

def _run_full(theta, t, hv3, vs3, pt3, scene_present=True, max_steps=200_000):
    """Run ODE and return full state trajectory (T, _N_TOTAL).

    scene_present: True  → scene always visible (sg=1, default).
                   False → dark (sg=0, OKR/visual pathways suppressed).
                   (T,) array → time-varying gain.
    """
    T      = len(t)
    dt_arr = jnp.diff(t)
    hp3    = jnp.concatenate([
        jnp.zeros((1, 3)),
        jnp.cumsum(0.5 * (hv3[:-1] + hv3[1:]) * dt_arr[:, None], axis=0),
    ])
    if scene_present is True:
        sg1 = jnp.ones(T, dtype=jnp.float32)
    elif scene_present is False:
        sg1 = jnp.zeros(T, dtype=jnp.float32)
    else:
        sg1 = jnp.asarray(scene_present, dtype=jnp.float32)

    hv_i  = diffrax.LinearInterpolation(ts=t, ys=hv3)
    hp_i  = diffrax.LinearInterpolation(ts=t, ys=hp3)
    vs_i  = diffrax.LinearInterpolation(ts=t, ys=vs3)
    pt_i  = diffrax.LinearInterpolation(ts=t, ys=pt3)
    sg_i  = diffrax.LinearInterpolation(ts=t, ys=sg1)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(ODE_ocular_motor),
        diffrax.Heun(),
        t0=t[0], t1=t[-1], dt0=0.001,
        y0=jnp.zeros(_N_TOTAL),
        args=(theta, hv_i, hp_i, vs_i, pt_i, sg_i),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=t),
        max_steps=max_steps,
    )
    return np.array(sol.ys)   # (T, _N_TOTAL)


def _extract_signals(theta, t, hv3, vs3, pt3, scene_present=True, max_steps=200_000):
    """Run and extract all relevant signals as a dict of arrays."""
    ys   = _run_full(theta, t, hv3, vs3, pt3, scene_present=scene_present, max_steps=max_steps)
    t_np = np.array(t)
    dt   = float(t[1] - t[0])

    x_p     = ys[:, _IDX_P]
    x_pc    = ys[:, _IDX_PC]
    x_ni_pc = ys[:, _IDX_NI_PC]
    x_vs    = ys[:, _IDX_VS]

    # Plant copy state → predicted saccade velocity (2-state: NI copy + plant copy)
    tau_p = theta['tau_p']
    # Recompute u_burst from state via vmap
    target_interp = diffrax.LinearInterpolation(ts=t, ys=pt3)

    # Extract x_ni so we can compute w_eye analytically (avoids gradient artifacts)
    x_ni = ys[:, _IDX_NI]

    def _burst_at(x, t_):
        x_sg          = x[_IDX_SG]
        x_vis         = x[_IDX_VIS]
        e_pos_delayed = visual_delay.C_pos @ x_vis
        _, u_burst    = sg.step(x_sg, e_pos_delayed, theta)
        return u_burst

    ys_j    = jnp.array(ys)
    u_burst = np.array(jax.vmap(_burst_at)(ys_j, t))   # (T, 3)

    w_burst_pred = (x_ni_pc - x_pc) / tau_p + u_burst  # 2-state efference copy velocity

    # Exact w_eye from plant ODE: w_eye = (x_ni - x_p)/tau_p + u_vel
    # where u_vel = -w_est + u_burst and w_est = VS output.
    # Recompute w_est analytically from stored state (no numerical gradient needed).
    x_c_j   = jnp.array(ys[:, _IDX_C])
    x_vs_j  = jnp.array(ys[:, _IDX_VS])
    x_vis_j = jnp.array(ys[:, _IDX_VIS])
    cg      = jnp.array(theta.get('canal_gains', jnp.ones(canal.N_CANALS)))
    def _w_est_at(x_c_t, x_vs_t, x_vis_t):
        y_canals       = canal.canal_nonlinearity(x_c_t, cg)
        e_slip_delayed = visual_delay.C_slip @ x_vis_t
        u_vs           = jnp.concatenate([y_canals, e_slip_delayed])
        _, w_est_t     = vs_mod.step(x_vs_t, u_vs, theta)
        return w_est_t
    w_est = np.array(jax.vmap(_w_est_at)(x_c_j, x_vs_j, x_vis_j))   # (T, 3)
    w_eye = (x_ni - x_p) / tau_p + (-w_est + u_burst)                # exact eye velocity

    return {
        'x_p':          x_p,
        'x_pc':         x_pc,
        'x_vs':         x_vs,
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

    x_ni_pc_sig = np.array(sig['x_pc'])  # (T,3) placeholder — extract from ys
    # Re-extract x_ni_pc directly
    t_j = jnp.arange(0.0, 0.6, 0.001)
    T_j = len(t_j)
    hv3_j = jnp.zeros((T_j, 3))
    vs3_j = jnp.zeros((T_j, 3))
    pt3_j = jnp.zeros((T_j, 3)); pt3_j = pt3_j.at[:, 2].set(1.0)
    pt3_j = pt3_j.at[t_j >= t_jump, 0].set(jnp.tan(jnp.radians(10.0)))
    ys_j2    = _run_full(THETA_SAC, t_j, hv3_j, vs3_j, pt3_j)
    x_ni_pc2 = ys_j2[:, _IDX_NI_PC][:, 0]
    x_pc2    = ys_j2[:, _IDX_PC][:, 0]
    axes[2].plot(t_np, x_pc2,    color=_C['pc'],    lw=1.5, label='x_pc  (plant copy)')
    axes[2].plot(t_np, x_ni_pc2, color='#1b7837', lw=1.2, ls='--', label='x_ni_pc  (NI copy)')
    axes[2].set_ylabel('deg')
    axes[2].set_title('2-state plant copy — x_ni_pc holds position, x_pc follows with τ_p')
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

    axes[2].plot(t_np, sig_scene['x_vs'][:, 0], color=_C['scene'],   lw=1.5, label='scene → VS driven')
    axes[2].plot(t_np, sig_sac['x_vs'][:, 0],   color=_C['eye_vel'], lw=1.5, label='saccade → VS suppressed')
    axes[2].set_ylabel('deg/s')
    axes[2].set_title('VS storage state x_vs  (charged by retinal slip)')
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

    # ── Test 2: x_ni_pc holds saccade amplitude; x_pc tracks x_ni_pc ─────
    # With the 2-state plant copy, after a 10° saccade x_ni_pc should
    # accumulate ~10° (NI copy integrates burst → holds position like real NI).
    # x_pc follows x_ni_pc with tau_p lag → both hold the new eye position.
    t = jnp.arange(0.0, 0.8, dt); T = len(t); t_np = np.array(t)
    t_jump = 0.1
    hv3 = jnp.zeros((T, 3)); vs3 = jnp.zeros((T, 3))
    pt3 = jnp.zeros((T, 3)); pt3 = pt3.at[:, 2].set(1.0)
    pt3 = pt3.at[t >= t_jump, 0].set(jnp.tan(jnp.radians(10.0)))
    ys          = _run_full(THETA_SAC, t, hv3, vs3, pt3)
    x_ni_pc_tr  = ys[:, _IDX_NI_PC][:, 0]
    x_pc_tr     = ys[:, _IDX_PC][:, 0]
    # NI copy should hold a meaningful position after saccade
    mask_after   = t_np > t_jump + 0.1
    assert np.abs(x_ni_pc_tr[mask_after]).max() > 1.0, \
        f'FAIL: x_ni_pc near zero after saccade; max={np.abs(x_ni_pc_tr[mask_after]).max():.3f} deg'
    # Plant copy should track NI copy (small difference after settling)
    mask_settled = t_np > t_jump + 0.5
    diff = np.abs(x_pc_tr[mask_settled] - x_ni_pc_tr[mask_settled]).max()
    assert diff < 2.0, \
        f'FAIL: x_pc does not track x_ni_pc; max|diff|={diff:.3f} deg'
    print('  2. x_ni_pc holds position; x_pc tracks x_ni_pc  PASS')

    # ── Test 3: VS not strongly contaminated by saccade in stationary lit world ──
    # Scene IS visible (scene_present=1) but stationary (vs3=0).
    # Without efference copy, saccadic w_eye (~600 deg/s) would look like retinal
    # slip and strongly drive VS.  With efference copy, the burst component cancels.
    # Remaining contamination comes from the unavoidable NI-plant settling transient
    # (x_ni−x_p)/tau_p, which efference copy cannot predict.
    # With K_vis=0.3 (10× old K_vs*g_okr≈0.02), threshold scales accordingly to 5 deg/s.
    t = jnp.arange(0.0, 0.6, dt); T = len(t); t_np = np.array(t)
    t_jump = 0.1
    hv3 = jnp.zeros((T, 3)); vs3 = jnp.zeros((T, 3))
    pt3 = jnp.zeros((T, 3)); pt3 = pt3.at[:, 2].set(1.0)
    pt3 = pt3.at[t >= t_jump, 0].set(jnp.tan(jnp.radians(10.0)))
    ys    = _run_full(THETA_SAC, t, hv3, vs3, pt3, scene_present=True)
    x_vs  = ys[:, _IDX_VS]
    mask  = (t_np > t_jump + 0.2) & (t_np < t_jump + 0.5)
    contamination = np.abs(x_vs[mask]).max()
    assert contamination < 5.0, \
        f'FAIL: VS contaminated by saccade; max|x_vs|={contamination:.3f} deg/s'
    print('  3. VS low after saccade (stationary world) PASS')

    # ── Test 4: VS still driven by real scene ────────────────────────────
    t = jnp.arange(0.0, 2.0, dt); T = len(t); t_np = np.array(t)
    hv3 = jnp.zeros((T, 3))
    vs3 = jnp.zeros((T, 3)); vs3 = vs3.at[:, 0].set(30.0)
    pt3 = jnp.zeros((T, 3)); pt3 = pt3.at[:, 2].set(1.0)
    ys    = _run_full(THETA_NO_SAC, t, hv3, vs3, pt3)
    x_vs  = ys[:, _IDX_VS]
    vs_late = np.abs(x_vs[t_np > 1.0, 0]).mean()
    assert vs_late > 1.0, \
        f'FAIL: VS not driven by scene; mean={vs_late:.2f} deg/s'
    print('  4. VS driven by genuine scene motion      PASS')

    # ── Test 5: VS not contaminated by repeated OKN fast phases ──────────────
    # Scene moves at 5 deg/s for 10 s.  Without efference copy, each fast-phase
    # saccade (~600 deg/s, backward) would look like a huge negative slip and
    # heavily discharge x_vs.  With efference copy the burst component cancels.
    # Check: |x_vs_with_sac − x_vs_no_sac| < 5 deg/s at end of trial.
    dt_okn = 0.001
    T_okn  = 10.0
    t_okn  = jnp.arange(0.0, T_okn, dt_okn); Tn = len(t_okn); t_okn_np = np.array(t_okn)
    hv_okn = jnp.zeros((Tn, 3))
    vs_okn = jnp.zeros((Tn, 3)); vs_okn = vs_okn.at[:, 0].set(5.0)    # 5 deg/s scene
    pt_okn = jnp.zeros((Tn, 3)); pt_okn = pt_okn.at[:, 2].set(1.0)    # target straight ahead

    ys_sac   = _run_full(THETA_OKN,     t_okn, hv_okn, vs_okn, pt_okn,
                         scene_present=True, max_steps=1_200_000)
    ys_nosac = _run_full(THETA_OKN_OFF, t_okn, hv_okn, vs_okn, pt_okn,
                         scene_present=True, max_steps=1_200_000)

    xvs_sac   = ys_sac[:, _IDX_VS]
    xvs_nosac = ys_nosac[:, _IDX_VS]
    # Compare in the last 3 s (after transient)
    mask5    = t_okn_np > 7.0
    vs_diff  = np.abs(xvs_sac[mask5, 0] - xvs_nosac[mask5, 0]).max()
    assert vs_diff < 5.0, \
        f'FAIL: VS contaminated by OKN fast phases; max diff={vs_diff:.2f} deg/s'
    print(f'  5. VS uncontaminated during OKN nystagmus  PASS  (max diff x_vs={vs_diff:.2f} deg/s)')

    print('All tests passed.\n')


# ── Demo 4: OKN nystagmus — comprehensive signal cascade ──────────────────────

def demo_okn_nystagmus():
    """Comprehensive signal-cascade debug for OKN nystagmus.

    Scene moves at 5 deg/s; target stays at straight ahead so OKR drives the
    eye away and the saccade generator should fire fast-phase resets → OKN.

    7 panels (similar to saccade_single layout):
        1. Eye position + target (sawtooth waveform)
        2. Eye velocity + scene velocity reference
        3. Position error (e_motor = target - eye) + delayed version + thresholds
        4. u_burst (saccade drive) + reset integrator x_sg
        5. Retinal slip — raw vs efference-copy corrected
        6. VS state x_vs with / without saccades (efference-copy check)
        7. NI state x_ni
    """
    dt    = 0.001
    T_end = 10.0
    t     = jnp.arange(0.0, T_end, dt); T = len(t); t_np = np.array(t)
    hv3   = jnp.zeros((T, 3))
    vs3   = jnp.zeros((T, 3)); vs3 = vs3.at[:, 0].set(5.0)
    pt3   = jnp.zeros((T, 3)); pt3 = pt3.at[:, 2].set(1.0)

    print('  Running OKN with saccades...')
    ys_sac   = _run_full(THETA_OKN,     t, hv3, vs3, pt3, scene_present=True, max_steps=1_200_000)
    print('  Running OKN without saccades...')
    ys_nosac = _run_full(THETA_OKN_OFF, t, hv3, vs3, pt3, scene_present=True, max_steps=1_200_000)

    # ── Extract signals via vmap ───────────────────────────────────────────────
    ys_j  = jnp.array(ys_sac)
    tau_p = THETA_OKN['tau_p']
    thr   = THETA_OKN.get('threshold_sac', 0.5)

    def _signals_at(x):
        x_p           = x[_IDX_P]
        x_ni          = x[_IDX_NI]
        x_sg          = x[_IDX_SG]
        x_vis         = x[_IDX_VIS]
        x_pc          = x[_IDX_PC]
        x_ni_pc_      = x[_IDX_NI_PC]
        e_pos_delayed = visual_delay.C_pos  @ x_vis
        e_slip_delayed= visual_delay.C_slip @ x_vis
        e_motor       = retina.target_to_angle(jnp.array([0.0, 0.0, 1.0])) - x_p
        _, u_burst    = sg.step(x_sg, e_pos_delayed, THETA_OKN)
        w_burst_pred  = (x_ni_pc_ - x_pc) / tau_p + u_burst
        return (x_p[0], x_ni[0], x_sg[0],
                e_motor[0], e_pos_delayed[0],
                u_burst[0], w_burst_pred[0], e_slip_delayed[0])

    out = jax.vmap(_signals_at)(ys_j)
    (x_p_sac, x_ni_sac, x_sg_sac,
     e_motor, e_pos_del,
     u_burst_all, w_pred_all, e_slip_del) = [np.array(o) for o in out]

    x_vs_sac   = ys_sac[:, _IDX_VS][:, 0]
    x_vs_nosac = ys_nosac[:, _IDX_VS][:, 0]

    w_eye_sac = np.gradient(x_p_sac, dt)
    raw_slip  = 5.0 - w_eye_sac
    cor_slip  = 5.0 - w_eye_sac + w_pred_all
    SPV_CLIP  = 40.0
    spv       = np.where(np.abs(w_eye_sac) < SPV_CLIP, w_eye_sac, np.nan)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(7, 1, figsize=(12, 17), sharex=True)
    fig.suptitle('OKN Nystagmus — Signal Cascade Debug  (scene = 5 deg/s)', fontsize=13)

    for ax in axes:
        ax.axhline(0, color='k', lw=0.4)

    # P1: eye position + target
    axes[0].plot(t_np, x_p_sac, color=_C['eye_vel'], lw=0.8, label='eye pos (x_p)')
    axes[0].axhline(0, color=_C['scene'], lw=1.5, ls='--', alpha=0.8, label='target (0 deg)')
    axes[0].set_ylabel('deg')
    axes[0].set_title('Eye position — should show sawtooth if fast phases fire')
    axes[0].legend(fontsize=8)

    # P2: eye velocity + scene
    axes[1].plot(t_np, w_eye_sac, color=_C['eye_vel'], lw=0.5, alpha=0.3, label='w_eye (raw)')
    axes[1].plot(t_np, spv,       color=_C['eye_vel'], lw=1.2, label=f'SPV (clip {SPV_CLIP})')
    axes[1].axhline(5, color=_C['scene'], lw=1.2, ls='--', alpha=0.8, label='scene 5 deg/s')
    axes[1].set_ylabel('deg/s')
    axes[1].set_title('Eye velocity (slow phase near scene; fast phases spike negative)')
    axes[1].legend(fontsize=8)

    # P3: position error cascade — key diagnostic for saccade trigger
    axes[2].plot(t_np, e_motor,  color='#762a83', lw=1.5, label='e_motor = target - eye')
    axes[2].plot(t_np, e_pos_del, color='#d6604d', lw=1.2, ls='--', label='e_pos_delayed (SG input)')
    axes[2].axhline( thr, color='gray', lw=0.8, ls=':', alpha=0.8, label=f'threshold ({thr} deg)')
    axes[2].axhline(-thr, color='gray', lw=0.8, ls=':', alpha=0.8)
    axes[2].set_ylabel('deg')
    axes[2].set_title('Position error & delayed error entering saccade generator')
    axes[2].legend(fontsize=8)

    # P4: burst command + reset integrator
    axes[3].plot(t_np, u_burst_all, color=_C['burst'],  lw=1.5, label='u_burst (SG output)')
    axes[3].plot(t_np, x_sg_sac,    color=_C['scene'],  lw=1.2, ls='--', label='x_sg (reset integrator)')
    axes[3].set_ylabel('deg/s | deg')
    axes[3].set_title('Burst command (spikes = fast phases) + Robinson copy integrator')
    axes[3].legend(fontsize=8)

    # P5: retinal slip raw vs corrected
    axes[4].plot(t_np, raw_slip, color='#d6604d', lw=0.8, alpha=0.6, label='raw slip (scene - eye)')
    axes[4].plot(t_np, cor_slip, color=_C['pred'],  lw=1.2, label='corrected slip (+ efference copy)')
    axes[4].axhline(5, color=_C['scene'], lw=0.8, ls='--', alpha=0.5, label='scene 5 deg/s')
    axes[4].set_ylabel('deg/s')
    axes[4].set_title('Retinal slip — corrected should stay near 5 deg/s')
    axes[4].legend(fontsize=8)

    # P6: VS state with / without saccades
    axes[5].plot(t_np, x_vs_nosac, color=_C['scene'],   lw=1.5, label='x_vs  no saccades')
    axes[5].plot(t_np, x_vs_sac,   color=_C['eye_vel'], lw=1.2, ls='--', label='x_vs  with saccades')
    axes[5].set_ylabel('deg/s')
    axes[5].set_title('VS state — traces overlap if efference copy is working')
    axes[5].legend(fontsize=8)

    # P7: NI state
    axes[6].plot(t_np, x_ni_sac, color=_C['pc'], lw=1.2, label='x_ni (NI position command)')
    axes[6].set_ylabel('deg')
    axes[6].set_title('NI state (grows as OKR accumulates; resets with fast phases)')
    axes[6].set_xlabel('Time (s)')
    axes[6].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'efference_okn_debug.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Saved {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    demo_plant_copy_dynamics()
    demo_okr_scene_vs_saccade()
    demo_slip_cancellation()
    print('\nRunning OKN efference-copy debug demo (10 s simulation, may take ~30 s)...')
    demo_okn_nystagmus()
    _run_tests()
    print(f'Plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
