"""Compare brain_model.step vs unified_brain.step on pursuit and saccade tasks.

Swap between models via simulator.set_brain_step() and run the same trajectory.

Usage:
    python -X utf8 scripts/bench_compare_unified.py
    python -X utf8 scripts/bench_compare_unified.py --show
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim import simulator
from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, with_sensory, simulate,
)
from oculomotor.sim import kinematics as km
from oculomotor.models.brain_models import brain_model, unified_brain


SHOW = '--show' in sys.argv
DT   = 0.001

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)


def _disable_non_subset(theta):
    """Zero out brain params for subsystems not in the unified subset.

    Disables: gravity-driven OCR, T-VOR, Listing's torsion, NI-OCR coupling,
    saccade burst (use g_burst=0 only for pursuit-only test).
    """
    return with_brain(theta,
                      g_ocr=0.0,
                      g_tvor=0.0, g_tvor_verg=0.0, K_phasic_tvor=0.0,
                      listing_gain=0.0)


def _run_with(brain_step_fn, theta, t_np, target_traj, key=0):
    """Run simulate() with the given brain step function."""
    simulator.set_brain_step(brain_step_fn)
    jax.clear_caches()        # invalidate any prior JIT trace bound to old _BRAIN_STEP
    t = jnp.array(t_np)
    T = len(t)
    out = simulate(theta, t,
                   target=target_traj,
                   scene_present_array=jnp.ones(T),
                   target_present_array=jnp.ones(T),
                   max_steps=int(len(t_np) * 1.05) + 500,
                   return_states=True,
                   key=jax.random.PRNGKey(key))
    return out


def _max_diff(name, a, b):
    a = np.asarray(a); b = np.asarray(b)
    d = float(np.max(np.abs(a - b)))
    s = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-9)
    print(f'  {name:25s}  max abs diff: {d:.4e}   rel: {d/s:.2%}')
    return d


# ── Pursuit benchmark ───────────────────────────────────────────────────────

def bench_pursuit(theta_base):
    """Pursuit ramp at 5 deg/s — small enough that no saccade is triggered."""
    print('\n' + '='*70)
    print('PURSUIT  (5 deg/s ramp, 1.5 s, T-VOR & OCR disabled)')
    print('='*70)

    theta = _disable_non_subset(theta_base)
    # Keep saccade machinery enabled (g_burst≠0) — brain_model NaNs with g_burst=0.
    # 5 deg/s × 1.5 s = 7.5 deg displacement; with pursuit gain ≈ 1 the position
    # error stays below threshold_sac so no saccade is triggered in either model.
    theta = with_sensory(theta, sigma_canal=0.0, sigma_pos=0.0,
                         sigma_vel=0.0, sigma_slip=0.0)
    theta = with_brain(theta, sigma_acc=0.0)

    T_end, T_jump = 1.5, 0.2
    t_np = np.arange(0.0, T_end, DT)
    T_n  = len(t_np)
    vel = 5.0
    pos = np.where(t_np >= T_jump, vel * (t_np - T_jump), 0.0)
    pt3 = np.zeros((T_n, 3));  pt3[:, 2] = 1.0
    pt3[:, 0] = np.tan(np.radians(pos))
    target_traj = km.build_target(t_np, lin_pos=np.array(pt3))

    st_b = _run_with(brain_model.step,    theta, t_np, target_traj)
    st_u = _run_with(unified_brain.step, theta, t_np, target_traj)
    simulator.set_brain_step(brain_model.step)   # restore

    # Compare key quantities
    eye_b = (np.array(st_b.plant.left[:, 0]) + np.array(st_b.plant.right[:, 0])) / 2.0
    eye_u = (np.array(st_u.plant.left[:, 0]) + np.array(st_u.plant.right[:, 0])) / 2.0
    pu_b  = np.concatenate([np.array(st_b.brain.pu.R), np.array(st_b.brain.pu.L)], axis=1)
    pu_u  = np.concatenate([np.array(st_u.brain.pu.R), np.array(st_u.brain.pu.L)], axis=1)
    ni_b  = np.concatenate([np.array(st_b.brain.ni.L), np.array(st_b.brain.ni.R), np.array(st_b.brain.ni.null)], axis=1)
    ni_u  = np.concatenate([np.array(st_u.brain.ni.L), np.array(st_u.brain.ni.R), np.array(st_u.brain.ni.null)], axis=1)

    _max_diff('eye position (cyclop)', eye_b, eye_u)
    _max_diff('pursuit state',         pu_b, pu_u)
    _max_diff('NI state',              ni_b, ni_u)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t_np, eye_b, '-',  label='brain_model',   color='C0', lw=1.5)
    axes[0].plot(t_np, eye_u, '--', label='unified_brain', color='C1', lw=1.0)
    axes[0].plot(t_np, pos,   ':',  label='target',        color='gray', lw=1.0)
    axes[0].set_ylabel('Eye / target pos (deg)'); axes[0].legend(fontsize=8)
    axes[0].set_title('Pursuit — brain_model vs unified_brain')
    axes[1].plot(t_np, eye_b - eye_u, color='red', lw=1.0)
    axes[1].set_ylabel('eye(b) − eye(u)  (deg)')
    axes[1].set_xlabel('Time (s)')
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'compare_unified_pursuit.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  → saved {out_path}')


# ── Saccade benchmark ───────────────────────────────────────────────────────

def bench_saccade(theta_base):
    """Step target at 10 deg, saccade fires."""
    print('\n' + '='*70)
    print('SACCADE  (10 deg target step, 1 s, T-VOR & OCR disabled)')
    print('='*70)

    theta = _disable_non_subset(theta_base)
    theta = with_sensory(theta, sigma_canal=0.0, sigma_pos=0.0,
                         sigma_vel=0.0, sigma_slip=0.0)
    theta = with_brain(theta, sigma_acc=0.0)

    T_end, T_jump = 1.0, 0.2
    t_np = np.arange(0.0, T_end, DT)
    T_n  = len(t_np)
    pos = np.where(t_np >= T_jump, 10.0, 0.0)
    pt3 = np.zeros((T_n, 3));  pt3[:, 2] = 1.0
    pt3[:, 0] = np.tan(np.radians(pos))
    target_traj = km.build_target(t_np, lin_pos=np.array(pt3))

    st_b = _run_with(brain_model.step,    theta, t_np, target_traj)
    st_u = _run_with(unified_brain.step, theta, t_np, target_traj)
    simulator.set_brain_step(brain_model.step)   # restore

    eye_b = (np.array(st_b.plant.left[:, 0]) + np.array(st_b.plant.right[:, 0])) / 2.0
    eye_u = (np.array(st_u.plant.left[:, 0]) + np.array(st_u.plant.right[:, 0])) / 2.0
    def _sg_flat(st):
        sg = st.brain.sg
        return np.concatenate([
            np.array(sg.e_held), np.array(sg.z_opn)[:, None],
            np.array(sg.z_acc)[:, None], np.array(sg.z_trig)[:, None],
            np.array(sg.ebn_R), np.array(sg.ebn_L),
            np.array(sg.ibn_R), np.array(sg.ibn_L),
        ], axis=1)
    sg_b  = _sg_flat(st_b)
    sg_u  = _sg_flat(st_u)

    _max_diff('eye position (cyclop)', eye_b, eye_u)
    _max_diff('SG state',              sg_b, sg_u)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t_np, eye_b, '-',  label='brain_model',   color='C0', lw=1.5)
    axes[0].plot(t_np, eye_u, '--', label='unified_brain', color='C1', lw=1.0)
    axes[0].plot(t_np, pos,   ':',  label='target',        color='gray', lw=1.0)
    axes[0].set_ylabel('Eye / target pos (deg)'); axes[0].legend(fontsize=8)
    axes[0].set_title('Saccade — brain_model vs unified_brain')
    axes[1].plot(t_np, eye_b - eye_u, color='red', lw=1.0)
    axes[1].set_ylabel('eye(b) − eye(u)  (deg)')
    axes[1].set_xlabel('Time (s)')
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'compare_unified_saccade.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  → saved {out_path}')


# ── VOR benchmark ───────────────────────────────────────────────────────────

def bench_vor(theta_base):
    """Small-amplitude sinusoidal head rotation — exercises VS + canal + NI.

    Saccade generator is sensitive to per-step floating-point error: if
    z_acc crosses threshold at a slightly different time in the two models,
    the saccade fires at different moments and trajectories diverge by the
    saccade amplitude (chaos via threshold-crossing). To isolate the
    self-motion path, use small-amplitude head motion that stays well below
    the saccade trigger threshold.
    """
    print('\n' + '='*70)
    print('VOR  (small-amp sinusoidal head rotation, 1 Hz, 2 s — self-motion path)')
    print('='*70)

    theta = with_brain(theta_base,
                       g_tvor=0.0, g_tvor_verg=0.0, K_phasic_tvor=0.0,
                       listing_gain=0.0)
    theta = with_sensory(theta, sigma_canal=0.0, sigma_pos=0.0,
                         sigma_vel=0.0, sigma_slip=0.0)
    theta = with_brain(theta, sigma_acc=0.0)

    T_end = 2.0
    head_traj = km.head_rotation_sinusoid(amplitude_deg_s=10.0, frequency_hz=1.0,
                                            duration=T_end, dt=DT, axis='yaw')
    t_np = np.array(head_traj.t)
    T_n = len(t_np)
    pt3 = np.zeros((T_n, 3));  pt3[:, 2] = 1.0
    target_traj = km.build_target(t_np, lin_pos=np.array(pt3))
    head_yaw = np.array(head_traj.rot_pos)[:, 0]

    def _run_with_head(brain_step_fn):
        simulator.set_brain_step(brain_step_fn)
        jax.clear_caches()
        return simulate(theta, jnp.array(t_np),
                        head=head_traj,
                        target=target_traj,
                        scene_present_array=jnp.zeros(T_n),
                        target_present_array=jnp.ones(T_n),
                        max_steps=int(T_n * 1.05) + 500,
                        return_states=True, key=jax.random.PRNGKey(0))

    st_b = _run_with_head(brain_model.step)
    st_u = _run_with_head(unified_brain.step)
    simulator.set_brain_step(brain_model.step)

    eye_b = (np.array(st_b.plant.left[:, 0]) + np.array(st_b.plant.right[:, 0])) / 2.0
    eye_u = (np.array(st_u.plant.left[:, 0]) + np.array(st_u.plant.right[:, 0])) / 2.0
    def _vs_flat(st):
        return np.concatenate([np.array(st.brain.sm.vs_L), np.array(st.brain.sm.vs_R), np.array(st.brain.sm.vs_null)], axis=1)
    def _grav_flat(st):
        return np.concatenate([np.array(st.brain.sm.g_est), np.array(st.brain.sm.a_lin), np.array(st.brain.sm.rf)], axis=1)
    vs_b   = _vs_flat(st_b);   vs_u   = _vs_flat(st_u)
    grav_b = _grav_flat(st_b); grav_u = _grav_flat(st_u)

    _max_diff('eye position (cyclop)', eye_b, eye_u)
    _max_diff('VS state',              vs_b, vs_u)
    _max_diff('GRAV state',            grav_b, grav_u)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t_np, eye_b, '-',  label='brain_model',   color='C0', lw=1.5)
    axes[0].plot(t_np, eye_u, '--', label='unified_brain', color='C1', lw=1.0)
    axes[0].plot(t_np, head_yaw, ':', label='head', color='gray', lw=1.0)
    axes[0].set_ylabel('Eye / head pos (deg)'); axes[0].legend(fontsize=8)
    axes[0].set_title('VOR — brain_model vs unified_brain')
    axes[1].plot(t_np, eye_b - eye_u, color='red', lw=1.0)
    axes[1].set_ylabel('eye(b) − eye(u)  (deg)')
    axes[1].set_xlabel('Time (s)')
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'compare_unified_vor.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  → saved {out_path}')


# ── Tilt / gravity benchmark ────────────────────────────────────────────────

def bench_tilt(theta_base):
    """Slow roll-tilt — exercises OCR, gravity transport, ω×g bilinear.

    Uses a small, slow tilt that doesn't trigger saccades, isolating the
    self-motion / gravity-estimator path. (Saccade-trigger chaos amplifies
    tiny per-step diffs by entire saccade amplitudes — see VOR bench note.)
    """
    print('\n' + '='*70)
    print('TILT  (slow 10 deg/s roll for 0.5 s, 1 s total — gravity/OCR/ω×g)')
    print('='*70)

    theta = with_brain(theta_base,
                       g_tvor=0.0, g_tvor_verg=0.0, K_phasic_tvor=0.0,
                       listing_gain=0.0)
    theta = with_sensory(theta, sigma_canal=0.0, sigma_pos=0.0,
                         sigma_vel=0.0, sigma_slip=0.0)
    theta = with_brain(theta, sigma_acc=0.0)

    # Slow roll: 10 deg/s for 0.5 s → 5° tilt; coast 0.5 s.
    head_traj = km.head_rotation_step(velocity_deg_s=10.0, rotate_dur=0.5,
                                        coast_dur=0.5, dt=DT, axis='roll')
    t_np = np.array(head_traj.t)
    T_n = len(t_np)
    pt3 = np.zeros((T_n, 3));  pt3[:, 2] = 1.0
    target_traj = km.build_target(t_np, lin_pos=np.array(pt3))

    def _run_with_head(brain_step_fn):
        simulator.set_brain_step(brain_step_fn)
        jax.clear_caches()
        return simulate(theta, jnp.array(t_np),
                        head=head_traj,
                        target=target_traj,
                        scene_present_array=jnp.zeros(T_n),
                        target_present_array=jnp.ones(T_n),
                        max_steps=int(T_n * 1.05) + 500,
                        return_states=True, key=jax.random.PRNGKey(0))

    st_b = _run_with_head(brain_model.step)
    st_u = _run_with_head(unified_brain.step)
    simulator.set_brain_step(brain_model.step)

    eye_torsion_b = (np.array(st_b.plant.left[:, 2]) + np.array(st_b.plant.right[:, 2])) / 2.0
    eye_torsion_u = (np.array(st_u.plant.left[:, 2]) + np.array(st_u.plant.right[:, 2])) / 2.0
    def _vs_flat2(st):
        return np.concatenate([np.array(st.brain.sm.vs_L), np.array(st.brain.sm.vs_R), np.array(st.brain.sm.vs_null)], axis=1)
    def _grav_flat2(st):
        return np.concatenate([np.array(st.brain.sm.g_est), np.array(st.brain.sm.a_lin), np.array(st.brain.sm.rf)], axis=1)
    grav_b = _grav_flat2(st_b); grav_u = _grav_flat2(st_u)
    vs_b   = _vs_flat2(st_b);   vs_u   = _vs_flat2(st_u)

    _max_diff('eye torsion (cyclop)', eye_torsion_b, eye_torsion_u)
    _max_diff('VS state',             vs_b, vs_u)
    _max_diff('GRAV state',           grav_b, grav_u)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t_np, eye_torsion_b, '-',  label='brain_model',   color='C0', lw=1.5)
    axes[0].plot(t_np, eye_torsion_u, '--', label='unified_brain', color='C1', lw=1.0)
    axes[0].set_ylabel('Eye torsion (deg)'); axes[0].legend(fontsize=8)
    axes[0].set_title('Roll tilt — brain_model vs unified_brain (OCR + ω×g transport)')
    axes[1].plot(t_np, eye_torsion_b - eye_torsion_u, color='red', lw=1.0)
    axes[1].set_ylabel('eye_T(b) − eye_T(u)  (deg)')
    axes[1].set_xlabel('Time (s)')
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'compare_unified_tilt.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  → saved {out_path}')


# ── T-VOR benchmark ─────────────────────────────────────────────────────────

def bench_tvor(theta_base):
    """Head sideways translation with target — exercises T-VOR cross-product path."""
    print('\n' + '='*70)
    print('T-VOR  (head heave 0.1 m/s, 1 s, near target — depth-scaled cross-product)')
    print('='*70)

    # Keep T-VOR + self-motion on; disable Listing's
    theta = with_brain(theta_base, listing_gain=0.0)
    theta = with_sensory(theta, sigma_canal=0.0, sigma_pos=0.0,
                         sigma_vel=0.0, sigma_slip=0.0)
    theta = with_brain(theta, sigma_acc=0.0)

    T_end = 1.0
    t_np = np.arange(0.0, T_end, DT)
    T_n = len(t_np)

    # Head moves sideways: lin_vel along x = 0.1 m/s
    lin_vel = np.zeros((T_n, 3));  lin_vel[:, 0] = 0.1
    lin_pos = np.zeros((T_n, 3));  lin_pos[:, 0] = 0.1 * t_np
    head_traj = km.build_kinematics(t_np, lin_pos=lin_pos, lin_vel=lin_vel)

    pt3 = np.zeros((T_n, 3));  pt3[:, 2] = 0.4   # near target at 40 cm
    target_traj = km.build_target(t_np, lin_pos=np.array(pt3))

    def _run(fn):
        simulator.set_brain_step(fn);  jax.clear_caches()
        return simulate(theta, jnp.array(t_np), head=head_traj, target=target_traj,
                        scene_present_array=jnp.zeros(T_n),
                        target_present_array=jnp.ones(T_n),
                        max_steps=int(T_n * 1.05) + 500,
                        return_states=True, key=jax.random.PRNGKey(0))

    st_b = _run(brain_model.step)
    st_u = _run(unified_brain.step)
    simulator.set_brain_step(brain_model.step)

    eye_b = (np.array(st_b.plant.left[:, 0]) + np.array(st_b.plant.right[:, 0])) / 2.0
    eye_u = (np.array(st_u.plant.left[:, 0]) + np.array(st_u.plant.right[:, 0])) / 2.0
    head_b = np.array(st_b.brain.sm.v_lin)
    head_u = np.array(st_u.brain.sm.v_lin)
    def _verg_flat(st):
        return np.concatenate([np.array(st.brain.va.verg_fast), np.array(st.brain.va.verg_tonic), np.array(st.brain.va.verg_copy)], axis=1)
    verg_b = _verg_flat(st_b); verg_u = _verg_flat(st_u)

    _max_diff('eye position',     eye_b, eye_u)
    _max_diff('HEAD (v_lin)',     head_b, head_u)
    _max_diff('vergence state',   verg_b, verg_u)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(t_np, eye_b, '-',  label='brain_model',   color='C0', lw=1.5)
    ax.plot(t_np, eye_u, '--', label='unified_brain', color='C1', lw=1.0)
    ax.set_ylabel('Eye position (deg)'); ax.set_xlabel('Time (s)')
    ax.set_title('T-VOR (head heave) — brain_model vs unified_brain')
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'compare_unified_tvor.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  → saved {out_path}')


def main():
    theta = PARAMS_DEFAULT
    bench_pursuit(theta)
    bench_saccade(theta)
    bench_vor(theta)
    bench_tilt(theta)
    bench_tvor(theta)
    if SHOW:
        plt.show()


if __name__ == '__main__':
    main()
