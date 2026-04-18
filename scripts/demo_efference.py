"""Efference-copy demo — two focused checks.

Figure 1 — efference_saccade.png
    10° saccade in a stationary lit room.
    Verifies that the saccade burst does NOT contaminate:
      (a) Velocity storage / OKN  — corrected slip ≈ 0; VS identical with/without saccade
      (b) Smooth pursuit integrator — x_pursuit stays near 0

Figure 2 — efference_pursuit.png
    Smooth pursuit of a 10 deg/s ramp in a stationary lit room.
    Verifies that the pursuit eye motion does NOT contaminate:
      (a) Velocity storage / OKN  — corrected slip ≈ 0; VS stays near 0

Signal flow reminder
────────────────────
    raw_slip        = w_scene − w_head − w_eye   (instantaneous)
    raw_slip_delayed → [visual delay, tau_vis] → raw_slip_delayed
    motor_ec        = delay(u_burst + u_pursuit)  [EC cascade, same tau_vis]
    e_slip_corrected = scene_visible · (raw_slip_delayed + motor_ec)  → VS

Usage
-----
    python -X utf8 scripts/demo_efference.py [--show]
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
    PARAMS_DEFAULT, with_brain, simulate,
    _IDX_C, _IDX_NI, _IDX_SG, _IDX_VIS, _IDX_VS, _IDX_EC, _IDX_PURSUIT,
)
from oculomotor.models.brain_models import saccade_generator as sg_mod
from oculomotor.models.sensory_models.sensory_model import C_slip, C_pos, C_gate
from oculomotor.models.sensory_models.sensory_model import N_CANALS, canal_nonlinearity
from oculomotor.models.brain_models import velocity_storage as vs_mod

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

_C = {
    'burst':      '#f4a582',
    'delayed':    '#d6604d',
    'eye_vel':    '#2166ac',
    'corrected':  '#4dac26',
    'scene':      '#e08214',
    'no_sac':     '#aaaaaa',
    'target':     '#762a83',
    'motor_ec':   '#1b7837',
}

THETA_SAC    = with_brain(PARAMS_DEFAULT, g_burst=600.0)
THETA_NO_SAC = with_brain(PARAMS_DEFAULT, g_burst=0.0)


# ── Utilities ──────────────────────────────────────────────────────────────────

def _vs_net(states):
    """Extract VS net signal (x_L − x_R) from state trajectory.  (T, 3)"""
    raw = np.array(states.brain[:, _IDX_VS])
    return raw[:, :3] - raw[:, 3:6]


def _ni_net(states):
    """Extract NI net signal (T, 3)."""
    raw = np.array(states.brain[:, _IDX_NI])
    return raw[:, :3] - raw[:, 3:6]


def _extract(states, theta):
    """Extract efference-copy signals from full state trajectory.

    Returns dict with keys:
        x_p, x_vs, x_ni, x_pursuit,
        u_burst, u_burst_delayed, motor_ec,
        raw_slip_delayed, cor_slip, w_eye
    """
    tau_p     = theta.brain.tau_p
    x_p       = np.array(states.plant[:, :3])                   # (T, 3) left eye
    x_vs      = _vs_net(states)                                  # (T, 3) VS net
    x_ni      = _ni_net(states)                                  # (T, 3) NI net
    x_pursuit = np.array(states.brain[:, _IDX_PURSUIT])         # (T, 3) pursuit mem

    # motor_ec = delay(u_burst + u_pursuit): last 3 states of EC cascade
    motor_ec        = np.array(states.brain[:, _IDX_EC])[:, -3:]  # (T, 3)
    u_burst_delayed = motor_ec                                    # alias for saccade figure

    # u_burst: recompute from SG state at each time step
    def _burst_at(state):
        e_pd      = C_pos @ state.sensory[_IDX_VIS]
        gate_vf   = (C_gate @ state.sensory[_IDX_VIS])[0]
        x_ni_raw  = state.brain[_IDX_NI]
        x_ni_net  = x_ni_raw[:3] - x_ni_raw[3:6]
        _, u_b    = sg_mod.step(state.brain[_IDX_SG], e_pd, gate_vf, x_ni_net, theta.brain)
        return u_b
    u_burst = np.array(jax.vmap(_burst_at)(states))              # (T, 3)

    # raw_slip_delayed: C_slip reads last stage of visual delay cascade
    x_vis_np         = np.array(states.sensory[:, _IDX_VIS])    # (T, 240)
    raw_slip_delayed = (np.array(C_slip) @ x_vis_np.T).T        # (T, 3)

    # corrected slip = raw + motor_ec  (what VS actually receives)
    cor_slip = raw_slip_delayed + motor_ec                       # (T, 3)

    # approximate w_eye from NI state
    cg      = jnp.array(theta.sensory.canal_gains)
    x_c_j   = states.sensory[:, _IDX_C]
    x_vs_j  = states.brain[:, _IDX_VS]
    x_vis_j = states.sensory[:, _IDX_VIS]

    def _w_est_at(xc, xvs, xvis):
        y_c   = canal_nonlinearity(xc, cg)
        e_sl  = C_slip @ xvis
        g_hat = jnp.array([9.81, 0.0, 0.0])
        _, w  = vs_mod.step(xvs, jnp.concatenate([y_c, e_sl, g_hat]), theta.brain)
        return w
    w_est = np.array(jax.vmap(_w_est_at)(x_c_j, x_vs_j, x_vis_j))   # (T, 3)
    w_eye = (x_ni - x_p) / tau_p + (-w_est + u_burst)                 # (T, 3)

    return dict(x_p=x_p, x_vs=x_vs, x_ni=x_ni, x_pursuit=x_pursuit,
                u_burst=u_burst, u_burst_delayed=u_burst_delayed,
                motor_ec=motor_ec,
                raw_slip_delayed=raw_slip_delayed,
                cor_slip=cor_slip, w_eye=w_eye)


def _ax_fmt(ax, ylim_min=None):
    ax.axhline(0, color='k', lw=0.4)
    ax.grid(True, alpha=0.2)
    if ylim_min is not None:
        lo, hi = ax.get_ylim()
        span = hi - lo
        if span < ylim_min:
            mid = (hi + lo) / 2
            ax.set_ylim(mid - ylim_min / 2, mid + ylim_min / 2)


# ── Figure 1: efference_saccade.png ───────────────────────────────────────────

def demo_efference_saccade():
    """5-panel figure: 10° saccade in a stationary lit room.

    Verifies EC suppresses saccade contamination of both OKN and pursuit.
    """
    dt, T_end, t_jump = 0.001, 0.6, 0.1
    t    = jnp.arange(0.0, T_end, dt)
    T    = len(t)
    t_np = np.array(t)

    pt3 = jnp.stack([
        jnp.where(t >= t_jump, jnp.tan(jnp.radians(10.0)), 0.0),
        jnp.zeros(T), jnp.ones(T),
    ], axis=1)
    max_s = int(T_end / dt) + 200

    # Lit stationary scene, target jumps to 10°
    kw = dict(p_target_array=pt3, scene_present_array=jnp.ones(T),
              target_present_array=jnp.ones(T), max_steps=max_s, return_states=True)

    print('  Saccade with EC...')
    states_sac   = simulate(THETA_SAC,    t, **kw)
    print('  Saccade without EC (g_burst=0 control)...')
    states_nosac = simulate(THETA_NO_SAC, t, **kw)

    s    = _extract(states_sac,   THETA_SAC)
    s_ns = _extract(states_nosac, THETA_NO_SAC)

    tau_vis = THETA_SAC.sensory.tau_vis

    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Saccade Efference Copy — 10° saccade in stationary lit room', fontsize=12)

    for ax in axes:
        ax.axvline(t_jump, color='gray', lw=0.8, ls='--', alpha=0.5)
        _ax_fmt(ax)

    # P1: eye position vs target
    axes[0].plot(t_np, s['x_p'][:, 0],
                 color=_C['eye_vel'], lw=1.5, label='eye pos (yaw)')
    axes[0].plot(t_np, np.degrees(np.arctan(np.array(pt3[:, 0]))),
                 color=_C['target'], lw=1.2, ls='--', label='target (10°)')
    axes[0].set_ylabel('deg')
    axes[0].set_title('Eye position — saccade executes correctly')
    axes[0].legend(fontsize=8)

    # P2: burst command and its delayed EC copy
    axes[1].plot(t_np, s['u_burst'][:, 0],
                 color=_C['burst'], lw=1.5, label='u_burst (immediate)')
    axes[1].plot(t_np, s['u_burst_delayed'][:, 0],
                 color=_C['delayed'], lw=1.5, ls='--',
                 label=f'u_burst_delayed (EC, ~{tau_vis*1000:.0f} ms lag)')
    axes[1].set_ylabel('deg/s')
    axes[1].set_title('Burst command and its efference copy (matched delays)')
    axes[1].legend(fontsize=8)

    # P3: retinal slip — raw vs corrected (OKN suppression)
    axes[2].plot(t_np, s['raw_slip_delayed'][:, 0],
                 color='#555555', lw=1.2, label='raw slip (uncorrected, would drive OKN)')
    axes[2].plot(t_np, s['cor_slip'][:, 0],
                 color=_C['corrected'], lw=1.5,
                 label='corrected slip → VS  (should ≈ 0 during saccade)')
    axes[2].set_ylabel('deg/s')
    axes[2].set_title('Retinal slip after EC correction — saccade does not drive OKN')
    axes[2].legend(fontsize=8)

    # P4: VS state — saccade vs no-saccade reference (should overlap)
    axes[3].plot(t_np, s_ns['x_vs'][:, 0],
                 color=_C['no_sac'], lw=2.0, label='no saccade (reference)')
    axes[3].plot(t_np, s['x_vs'][:, 0],
                 color=_C['eye_vel'], lw=1.5, ls='--',
                 label='with saccade (should overlap → OKN not driven)')
    axes[3].set_ylabel('VS net (deg/s)')
    axes[3].set_title('Velocity storage — identical with/without saccade confirms OKN suppression')
    axes[3].legend(fontsize=8)

    # P5: pursuit integrator — should stay near 0 (saccade does not drive pursuit)
    axes[4].plot(t_np, s['x_pursuit'][:, 0],
                 color=_C['scene'], lw=1.5, label='x_pursuit (should ≈ 0)')
    axes[4].set_ylabel('deg/s')
    axes[4].set_xlabel('Time (s)')
    axes[4].set_title('Pursuit integrator — saccade does not drive pursuit')
    axes[4].legend(fontsize=8)
    _ax_fmt(axes[4], ylim_min=2.0)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'efference_saccade.png')
    fig.savefig(path, dpi=150)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    print(f'  Saved {path}')


# ── Figure 2: efference_pursuit.png ───────────────────────────────────────────

def demo_efference_pursuit():
    """4-panel figure: smooth pursuit of 10 deg/s ramp, stationary lit room.

    Verifies EC suppresses pursuit eye motion from contaminating OKN (VS).
    """
    dt     = 0.001
    T_end  = 4.0
    t_ramp = 0.2    # ramp onset
    V_TGT  = 10.0   # deg/s target velocity
    t      = jnp.arange(0.0, T_end, dt)
    T      = len(t)
    t_np   = np.array(t)
    max_s  = int(T_end / dt) + 500

    # Ramp target: position accumulates from t_ramp; velocity step at t_ramp
    v_tgt = jnp.where(t >= t_ramp, V_TGT, 0.0)  # (T,) yaw component
    v_tgt3 = jnp.stack([v_tgt, jnp.zeros(T), jnp.zeros(T)], axis=1)  # (T, 3)

    # Integrate target velocity to get target position
    p_tgt_yaw = jnp.cumsum(v_tgt) * dt               # (T,)
    p_tgt3 = jnp.stack([
        jnp.tan(jnp.radians(p_tgt_yaw)),              # tan() for retinal geometry
        jnp.zeros(T), jnp.ones(T),
    ], axis=1)

    kw = dict(p_target_array=p_tgt3, v_target_array=v_tgt3,
              scene_present_array=jnp.ones(T),
              target_present_array=jnp.ones(T),
              max_steps=max_s, return_states=True)

    # Comparison: same scene motion (not pursuit) to show genuine OKN drive
    v_scene3 = jnp.stack([jnp.where(t >= t_ramp, V_TGT, 0.0),
                           jnp.zeros(T), jnp.zeros(T)], axis=1)

    print('  Smooth pursuit with EC...')
    states_pur = simulate(THETA_NO_SAC, t, **kw)     # no saccades: pure pursuit

    print('  Scene motion reference (genuine OKN)...')
    states_okn = simulate(THETA_NO_SAC, t,
                          head_vel_array=jnp.zeros(T),
                          v_scene_array=v_scene3,
                          scene_present_array=jnp.ones(T),
                          target_present_array=jnp.zeros(T),
                          max_steps=max_s, return_states=True)

    s     = _extract(states_pur, THETA_NO_SAC)
    s_okn = _extract(states_okn, THETA_NO_SAC)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(
        f'Pursuit Efference Copy — {V_TGT} deg/s ramp in stationary lit room', fontsize=12)

    for ax in axes:
        ax.axvline(t_ramp, color='gray', lw=0.8, ls='--', alpha=0.5)
        _ax_fmt(ax)

    # P1: tracking quality — eye vs target (position and velocity)
    w_eye = np.gradient(s['x_p'][:, 0], dt)
    ax1b = axes[0].twinx()
    axes[0].plot(t_np, s['x_p'][:, 0],
                 color=_C['eye_vel'], lw=1.5, label='eye pos', alpha=0.8)
    axes[0].plot(t_np, np.degrees(np.arctan(np.array(p_tgt3[:, 0]))),
                 color=_C['target'], lw=1.2, ls='--', label='target pos', alpha=0.8)
    ax1b.plot(t_np, w_eye,
              color=_C['eye_vel'], lw=1.0, ls=':', label='eye vel (right axis)', alpha=0.6)
    ax1b.axhline(V_TGT, color=_C['target'], lw=0.8, ls=':', alpha=0.5, label=f'target vel ({V_TGT} deg/s)')
    axes[0].set_ylabel('deg (position)')
    ax1b.set_ylabel('deg/s (velocity)')
    axes[0].set_title('Smooth pursuit tracking — eye follows target')
    lines0, labs0 = axes[0].get_legend_handles_labels()
    lines1, labs1 = ax1b.get_legend_handles_labels()
    axes[0].legend(lines0 + lines1, labs0 + labs1, fontsize=7, loc='upper left')

    # P2: slip signals — raw vs corrected vs motor_ec
    axes[1].plot(t_np, s['raw_slip_delayed'][:, 0],
                 color='#555555', lw=1.2, alpha=0.8,
                 label='raw slip (≈ −v_eye, would drive OKN)')
    axes[1].plot(t_np, s['motor_ec'][:, 0],
                 color=_C['motor_ec'], lw=1.2, ls='--',
                 label='motor_ec = delay(u_pursuit) (≈ +v_eye)')
    axes[1].plot(t_np, s['cor_slip'][:, 0],
                 color=_C['corrected'], lw=1.5,
                 label='corrected slip → VS  (should ≈ 0 at steady state)')
    axes[1].set_ylabel('deg/s')
    axes[1].set_title('EC cancels pursuit eye motion from retinal slip')
    axes[1].legend(fontsize=8)

    # P3: VS state — pursuit (should stay low) vs genuine scene motion (should drive)
    axes[2].plot(t_np, s_okn['x_vs'][:, 0],
                 color=_C['scene'], lw=1.5, label=f'genuine scene {V_TGT} deg/s → OKN (reference)')
    axes[2].plot(t_np, s['x_vs'][:, 0],
                 color=_C['eye_vel'], lw=1.5, ls='--',
                 label='pursuit in lit room → VS near 0 (EC working)')
    axes[2].set_ylabel('VS net (deg/s)')
    axes[2].set_title('Velocity storage — pursuit does not contaminate OKN')
    axes[2].legend(fontsize=8)

    # P4: pursuit integrator — should charge up steadily (correct output)
    axes[3].plot(t_np, s['x_pursuit'][:, 0],
                 color=_C['scene'], lw=1.5, label='x_pursuit (pursuit integrator)')
    axes[3].axhline(V_TGT, color=_C['target'], lw=0.8, ls='--', alpha=0.6,
                    label=f'target vel ({V_TGT} deg/s)')
    axes[3].set_ylabel('deg/s')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title('Pursuit integrator — charges to target velocity (not OKN)')
    axes[3].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'efference_pursuit.png')
    fig.savefig(path, dpi=150)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    print(f'  Saved {path}')


# ── Numerical tests ────────────────────────────────────────────────────────────

def _run_tests():
    print('\nRunning numerical tests...')
    dt = 0.001

    # ── Test 1: motor_ec stays zero without motor command ─────────────────
    t      = jnp.arange(0.0, 0.5, dt)
    states = simulate(THETA_NO_SAC, t, max_steps=int(0.5/dt)+200, return_states=True)
    ec_out = np.array(states.brain[:, _IDX_EC])[:, -3:]
    assert np.allclose(ec_out, 0.0, atol=1e-5), \
        f'FAIL: motor_ec non-zero without motor command; max={np.abs(ec_out).max():.2e}'
    print('  1. motor_ec zero without motor command                 PASS')

    # ── Test 2: motor_ec lags u_burst by ~tau_vis ─────────────────────────
    t    = jnp.arange(0.0, 0.8, dt)
    T    = len(t)
    t_np = np.array(t)
    pt3  = jnp.stack([
        jnp.where(t >= 0.1, jnp.tan(jnp.radians(15.0)), 0.0),
        jnp.zeros(T), jnp.ones(T),
    ], axis=1)
    states = simulate(THETA_SAC, t, p_target_array=pt3,
                      scene_present_array=jnp.zeros(T),
                      max_steps=int(0.8/dt)+200, return_states=True)

    def _burst_at(state):
        e_pd      = C_pos @ state.sensory[_IDX_VIS]
        gate_vf   = (C_gate @ state.sensory[_IDX_VIS])[0]
        x_ni_raw  = state.brain[_IDX_NI]
        x_ni_net  = x_ni_raw[:3] - x_ni_raw[3:6]
        _, u_b    = sg_mod.step(state.brain[_IDX_SG], e_pd, gate_vf, x_ni_net, THETA_SAC.brain)
        return u_b
    u_burst = np.array(jax.vmap(_burst_at)(states))[:, 0]
    ec_out  = np.array(states.brain[:, _IDX_EC])[:, -3]

    peak_burst = t_np[np.argmax(u_burst)]
    peak_ec    = t_np[np.argmax(ec_out)]
    delay_meas = peak_ec - peak_burst
    tau_vis    = THETA_SAC.sensory.tau_vis
    assert abs(delay_meas - tau_vis) < 0.02, \
        f'FAIL: EC delay mismatch; expected ~{tau_vis:.3f} s, got {delay_meas:.3f} s'
    assert u_burst.max() > 10.0, f'FAIL: u_burst too small; max={u_burst.max():.2f}'
    assert ec_out.max()  > 1.0,  f'FAIL: motor_ec too small; max={ec_out.max():.2f}'
    print(f'  2. motor_ec lags u_burst by ~tau_vis                  PASS'
          f'  (measured {delay_meas*1000:.1f} ms, expected {tau_vis*1000:.0f} ms)')

    # ── Test 3: saccade — VS contamination stays low in lit stationary room ─
    t    = jnp.arange(0.0, 0.6, dt)
    T    = len(t)
    t_np = np.array(t)
    pt3  = jnp.stack([
        jnp.where(t >= 0.1, jnp.tan(jnp.radians(10.0)), 0.0),
        jnp.zeros(T), jnp.ones(T),
    ], axis=1)
    states = simulate(THETA_SAC, t, p_target_array=pt3,
                      scene_present_array=jnp.ones(T),
                      max_steps=int(0.6/dt)+200, return_states=True)
    x_vs_net = _vs_net(states)
    mask     = (t_np > 0.3) & (t_np < 0.55)
    contam   = np.abs(x_vs_net[mask]).max()
    assert contam < 5.0, \
        f'FAIL: VS contaminated by saccade; max|x_vs|={contam:.3f}'
    print(f'  3. Saccade: VS contamination low in lit room           PASS'
          f'  (max={contam:.2f})')

    # ── Test 4: saccade — pursuit integrator stays near 0 ─────────────────
    x_pur = np.array(states.brain[:, _IDX_PURSUIT])
    pur_peak = np.abs(x_pur[mask]).max()
    assert pur_peak < 5.0, \
        f'FAIL: pursuit driven by saccade; max|x_pursuit|={pur_peak:.3f}'
    print(f'  4. Saccade: pursuit integrator stays near 0            PASS'
          f'  (max={pur_peak:.2f})')

    # ── Test 5: VS driven by real scene motion ─────────────────────────────
    t    = jnp.arange(0.0, 2.0, dt)
    T    = len(t)
    vs3  = jnp.zeros((T, 3)).at[:, 0].set(30.0)
    states = simulate(THETA_NO_SAC, t,
                      head_vel_array=jnp.zeros(T), v_scene_array=vs3,
                      scene_present_array=jnp.ones(T),
                      max_steps=int(2.0/dt)+200, return_states=True)
    x_vs_net = _vs_net(states)
    t_np     = np.array(t)
    vs_late  = np.abs(x_vs_net[t_np > 1.0, 0]).mean()
    assert vs_late > 1.0, f'FAIL: VS not driven by scene; mean={vs_late:.2f}'
    print(f'  5. VS driven by genuine scene motion                   PASS'
          f'  (mean={vs_late:.2f})')

    # ── Test 6: pursuit — VS not contaminated at steady state ─────────────
    dt2   = 0.001
    T_end = 4.0
    t2    = jnp.arange(0.0, T_end, dt2)
    T2    = len(t2)
    t2_np = np.array(t2)
    V_TGT = 10.0
    v_tgt = jnp.where(t2 >= 0.2, V_TGT, 0.0)
    v_tgt3 = jnp.stack([v_tgt, jnp.zeros(T2), jnp.zeros(T2)], axis=1)
    p_tgt3 = jnp.stack([jnp.tan(jnp.radians(jnp.cumsum(v_tgt) * dt2)),
                         jnp.zeros(T2), jnp.ones(T2)], axis=1)
    states2 = simulate(THETA_NO_SAC, t2,
                       p_target_array=p_tgt3, v_target_array=v_tgt3,
                       scene_present_array=jnp.ones(T2),
                       target_present_array=jnp.ones(T2),
                       max_steps=int(T_end/dt2)+500, return_states=True)
    x_vs_net2 = _vs_net(states2)
    mask6     = t2_np > 2.0
    vs_pur    = np.abs(x_vs_net2[mask6, 0]).mean()
    assert vs_pur < 3.0, \
        f'FAIL: VS contaminated by pursuit; mean|x_vs|={vs_pur:.3f}'
    print(f'  6. Pursuit: VS stays low (EC working)                  PASS'
          f'  (mean={vs_pur:.2f})')

    print('All tests passed.\n')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=== Efference Copy Demo ===')

    print('\n1. Saccade EC — does saccade contaminate OKN or pursuit?')
    demo_efference_saccade()

    print('\n2. Pursuit EC — does pursuit contaminate OKN?')
    demo_efference_pursuit()

    print('\n3. Numerical tests')
    _run_tests()

    print(f'\nDone. Plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
