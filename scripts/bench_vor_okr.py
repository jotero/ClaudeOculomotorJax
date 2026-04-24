"""VOR / OKR benchmarks — Raphan Fig.9 replication, nystagmus zoom, TC comparison, cascade.

Usage:
    python -X utf8 scripts/bench_vor_okr.py
    python -X utf8 scripts/bench_vor_okr.py --show
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
    _IDX_C, _IDX_VS, _IDX_NI, _IDX_VIS,
)
from oculomotor.models.sensory_models.sensory_model import (
    N_CANALS, FLOOR, _SOFTNESS, PINV_SENS, C_slip,
)
from oculomotor.sim import kinematics as km
from oculomotor.analysis import ax_fmt, extract_burst, extract_canal, vs_net, vs_null, ni_net, fit_tc, extract_spv

SHOW  = '--show' in sys.argv
DT    = 0.001
THETA = with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_pos=0.0, sigma_vel=0.0)


def _simulate(theta, t_arr, head_vel=None, scene_vel=None, scene_present=None,
              target_present=None, key=0):
    T   = len(t_arr)
    t   = np.array(t_arr)
    hv  = np.array(head_vel)  if head_vel  is not None else np.zeros((T, 3), np.float32)
    sv  = np.array(scene_vel) if scene_vel is not None else np.zeros((T, 3), np.float32)
    sp  = scene_present  if scene_present  is not None else jnp.zeros(T)
    tp  = target_present if target_present is not None else jnp.zeros(T)
    ms  = int(len(t_arr) * 1.05) + 500
    return simulate(theta, t_arr,
                    head=km.build_kinematics(t, rot_vel=hv),
                    scene=km.build_kinematics(t, rot_vel=sv),
                    scene_present_array=sp, target_present_array=tp,
                    max_steps=ms, return_states=True,
                    key=jax.random.PRNGKey(key))


# ── Figure 1: Raphan 1979 Fig.9 replication ───────────────────────────────────

def _raphan(show):
    """3×2 Raphan 1979 Fig.9 replication: SPV (left col) + CUP/INT/SPV (right col)."""
    V_STIM   = 30.0
    ON_DUR   = 30.0
    OFF_DUR  = 50.0
    TOTAL    = ON_DUR + OFF_DUR
    BASELINE = 3.0
    B = int(BASELINE / DT)

    theta = THETA

    # ── A/B: VOR in dark ──────────────────────────────────────────────────────
    _head = km.head_rotation_step(V_STIM, rotate_dur=ON_DUR, coast_dur=OFF_DUR, dt=DT)
    t_rot, hv_3d = _head.t, _head.rot_vel
    t_vor   = np.concatenate([np.arange(1 - B, 1) * DT - DT, np.array(t_rot)])
    hv_full = np.concatenate([np.zeros((B, 3)), np.array(hv_3d)])
    T_vor   = len(t_vor)
    st_vor  = _simulate(theta, jnp.array(t_vor), head_vel=jnp.array(hv_full),
                        scene_present=jnp.zeros(T_vor),
                        target_present=jnp.zeros(T_vor), key=0)
    ev_vor    = np.gradient(np.array(st_vor.plant[:, 0]), DT)
    burst_vor = extract_burst(st_vor, theta)[:, 0]
    spv_vor_d = -extract_spv(t_vor, ev_vor, burst_vor)   # negate: compensatory = positive
    cup_vor   = extract_canal(st_vor)
    int_vor   = vs_net(st_vor)[:, 0]                     # x_L−x_R > 0 during rightward VOR
    ni_vor    = ni_net(st_vor)[:, 0]
    eye_vor   = np.array(st_vor.plant[:, 0])
    hv_1d     = hv_full[:, 0]

    tau_vor, t_fit_vor, y_fit_vor = fit_tc(
        t_vor, spv_vor_d, t_start=ON_DUR + 1.0, t_end=ON_DUR + OFF_DUR - 5.0,
        label='VOR post-rot TC')

    # ── C/D: OKN + OKAN ───────────────────────────────────────────────────────
    t_stim  = jnp.arange(0.0, TOTAL, DT)
    t_okn   = np.concatenate([np.arange(1 - B, 1) * DT - DT, np.array(t_stim)])
    T_okn   = len(t_okn)
    t_okn_j = jnp.array(t_okn)
    sv = jnp.zeros((T_okn, 3)).at[:, 0].set(
             jnp.where((t_okn_j >= 0.0) & (t_okn_j < ON_DUR), V_STIM, 0.0))
    sp = jnp.where((t_okn_j >= 0.0) & (t_okn_j < ON_DUR), 1.0, 0.0)
    st_okn  = _simulate(theta, t_okn_j, scene_vel=sv,
                        scene_present=sp, target_present=jnp.zeros(T_okn), key=1)
    ev_okn    = np.gradient(np.array(st_okn.plant[:, 0]), DT)
    burst_okn = extract_burst(st_okn, theta)[:, 0]
    spv_okn_d = extract_spv(t_okn, ev_okn, burst_okn)    # positive: eye tracks scene
    int_okn   = -vs_net(st_okn)[:, 0]                    # x_L−x_R < 0 → negate for display
    ni_okn    = ni_net(st_okn)[:, 0]
    eye_okn   = np.array(st_okn.plant[:, 0])

    tau_okan, t_fit_okan, y_fit_okan = fit_tc(
        t_okn, spv_okn_d, t_start=ON_DUR + 1.0, t_end=TOTAL - 5.0,
        label='OKAN TC')

    # ── E/F: VVOR (rotation in lit stationary scene, stop in dark) ────────────
    # target_present=0: pursuit suppressed — it would fight the VOR by integrating
    # the VOR-induced target slip (the EC cancels u_pursuit but not u_vor, so
    # pursuit sees persistent error and saturates against the VOR).
    # Fast phases are centering saccades (SG uses x_ni centering when target_in_vf=0),
    # matching the Raphan (1979) paradigm.  scene_present=1 enables OKR/OKAN.
    t_vor_j    = jnp.array(t_vor)
    scene_vvor = jnp.where((t_vor_j >= 0.0) & (t_vor_j < ON_DUR), 1.0, 0.0)
    st_vvor   = _simulate(theta, t_vor_j, head_vel=jnp.array(hv_full),
                          scene_present=scene_vvor,
                          target_present=jnp.zeros(T_vor),
                          key=2)
    ev_vvor    = np.gradient(np.array(st_vvor.plant[:, 0]), DT)
    burst_vvor = extract_burst(st_vvor, theta)[:, 0]
    spv_vvor_d = -extract_spv(t_vor, ev_vvor, burst_vvor)
    cup_vvor   = extract_canal(st_vvor)
    int_vvor   = vs_net(st_vvor)[:, 0]
    ni_vvor    = ni_net(st_vvor)[:, 0]
    eye_vvor   = np.array(st_vvor.plant[:, 0])

    mask_ss   = (t_vor > 10.0) & (t_vor < 25.0)
    vvor_gain = (np.mean(np.abs(spv_vvor_d[mask_ss])) /
                 (np.mean(np.abs(hv_1d[mask_ss])) + 1e-6))
    print(f'  VVOR yaw gain (10–25 s): {vvor_gain:.3f}  (target > 0.85)')

    cup_okn  = extract_canal(st_okn)
    cup_vvor = extract_canal(st_vvor)

    # ── Plotting: 3×2 layout matching Raphan Fig.9 ───────────────────────────
    # Left col (A, C, E): SPV only.  Right col (B, D, F): SPV + Cupula + Integrator.
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    fig.suptitle(
        'Raphan, Matsuo & Cohen (1979) Fig. 9 — Replication\n'
        'Left: slow-phase velocity  |  Right: S.P.VEL + Cupula + Integrator (VS)',
        fontsize=10, fontweight='bold')
    xlim = (-BASELINE, TOTAL)
    vl   = dict(color='k', lw=0.8, ls='--', alpha=0.5)

    def _lbl(ax, letter):
        ax.text(0.02, 0.92, letter, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')

    # A — VOR dark: SPV only
    ax = axes[0, 0]
    ax.plot(t_vor, -hv_1d,    color=utils.C['head'], lw=1.0, ls=':', alpha=0.7, label='−head vel')
    ax.plot(t_vor, spv_vor_d, color=utils.C['spv'],  lw=1.8, label='S.P.VEL')
    if tau_vor is not None:
        ax.plot(t_fit_vor, y_fit_vor, color='tomato', lw=1.5, ls='--', label=f'fit τ={tau_vor:.1f}s')
    ax.axvline(0.0, **vl); ax.axvline(ON_DUR, **vl)
    ax.set_ylabel('deg/s'); ax.set_title(f'Step rotation {V_STIM:.0f} deg/s — darkness')
    ax.legend(fontsize=7); ax_fmt(ax); ax.set_xlim(*xlim); _lbl(ax, 'A')

    # B — VOR dark: SPV + Cupula + Integrator
    ax = axes[0, 1]
    ax.plot(t_vor, spv_vor_d, color=utils.C['spv'],  lw=1.8, label='S.P.VEL')
    ax.plot(t_vor, cup_vor,   color=utils.C['canal'], lw=1.2, ls='--', label='Cupula')
    ax.plot(t_vor, int_vor,   color=utils.C['vs'],    lw=1.2, ls='-.',  label='Integrator (VS)')
    ax.axvline(0.0, **vl); ax.axvline(ON_DUR, **vl)
    ax.set_ylabel('deg/s'); ax.set_title('VOR dark: S.P.VEL + Cupula + Integrator')
    ax.legend(fontsize=7); ax_fmt(ax); ax.set_xlim(*xlim); _lbl(ax, 'B')

    # C — OKN + OKAN: SPV only
    ax = axes[1, 0]
    scene_ref = np.where((t_okn >= 0.0) & (t_okn < ON_DUR), V_STIM, 0.0)
    ax.plot(t_okn, scene_ref,  color=utils.C['scene'], lw=1.0, ls=':', alpha=0.7, label='scene vel')
    ax.plot(t_okn, spv_okn_d,  color=utils.C['spv'],   lw=1.8, label='S.P.VEL')
    if tau_okan is not None:
        ax.plot(t_fit_okan, y_fit_okan, color='tomato', lw=1.5, ls='--', label=f'OKAN τ={tau_okan:.1f}s')
    ax.axvline(0.0, **vl); ax.axvline(ON_DUR, **vl)
    ax.set_ylabel('deg/s'); ax.set_title(f'Surround velocity {V_STIM:.0f} deg/s — OKN then OKAN')
    ax.legend(fontsize=7); ax_fmt(ax); ax.set_xlim(-BASELINE, TOTAL); _lbl(ax, 'C')

    # D — OKN + OKAN: SPV + Cupula (≈0) + Integrator
    ax = axes[1, 1]
    ax.plot(t_okn, spv_okn_d, color=utils.C['spv'],   lw=1.8, label='S.P.VEL')
    ax.plot(t_okn, cup_okn,   color=utils.C['canal'],  lw=1.2, ls='--', label='Cupula')
    ax.plot(t_okn, int_okn,   color=utils.C['vs'],     lw=1.2, ls='-.',  label='Integrator (VS)')
    ax.axvline(0.0, **vl); ax.axvline(ON_DUR, **vl)
    ax.set_ylabel('deg/s'); ax.set_title('OKN: S.P.VEL + Cupula + Integrator')
    ax.legend(fontsize=7); ax_fmt(ax); ax.set_xlim(-BASELINE, TOTAL); _lbl(ax, 'D')

    # E — VVOR: SPV only
    ax = axes[2, 0]
    ax.plot(t_vor, -hv_1d,    color=utils.C['head'], lw=1.0, ls=':', alpha=0.7, label='−head vel')
    ax.plot(t_vor, spv_vvor_d, color=utils.C['spv'],  lw=1.8, label=f'S.P.VEL (gain={vvor_gain:.2f})')
    ax.axvline(0.0, **vl); ax.axvline(ON_DUR, **vl)
    ax.set_ylabel('deg/s'); ax.set_xlabel('Time (s)')
    ax.set_title(f'Rotation {V_STIM:.0f} deg/s in light → stop in dark')
    ax.legend(fontsize=7); ax_fmt(ax); ax.set_xlim(*xlim); _lbl(ax, 'E')

    # F — VVOR: SPV + Cupula + Integrator
    ax = axes[2, 1]
    ax.plot(t_vor, spv_vvor_d, color=utils.C['spv'],   lw=1.8, label='S.P.VEL')
    ax.plot(t_vor, cup_vvor,   color=utils.C['canal'],  lw=1.2, ls='--', label='Cupula')
    ax.plot(t_vor, int_vvor,   color=utils.C['vs'],     lw=1.2, ls='-.',  label='Integrator (VS)')
    ax.axvline(0.0, **vl); ax.axvline(ON_DUR, **vl)
    ax.set_ylabel('deg/s'); ax.set_xlabel('Time (s)')
    ax.set_title('VVOR: S.P.VEL + Cupula + Integrator')
    ax.legend(fontsize=7); ax_fmt(ax); ax.set_xlim(*xlim); _lbl(ax, 'F')

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'vor_raphan_fig9', show=show)
    return utils.fig_meta(path, rp,
        title='Raphan 1979 Fig. 9 Replication',
        description='Panels A–F matching Raphan et al. (1979) Fig.9. Left col: SPV only. '
                    'Right col: CUP (canal estimate), INT (velocity storage), SPV overlaid. '
                    'A/B: VOR in dark. C/D: OKN+OKAN. E/F: VVOR (light→dark).',
        expected='A: post-rot SPV TC 10–30 s. C: OKN gain~1, OKAN TC~20 s. '
                 'E: VVOR gain>0.85 during rotation; post-rot TC similar to A. '
                 'B/D/F: INT follows SPV; CUP decays at canal TC (~5 s).',
        citation='Raphan, Matsuo & Cohen (1979) Exp Brain Res 35:229–248',
        fig_type='behavior')


# ── Figure 2: OKN nystagmus zoom ──────────────────────────────────────────────

def _okn_zoom(show):
    """Zoomed OKN trace showing sawtooth nystagmus in first 15 s."""
    ON_DUR = 15.0
    t_arr  = jnp.arange(0.0, ON_DUR, DT)
    T      = len(t_arr)
    t_np   = np.array(t_arr)

    sv = jnp.zeros((T, 3)).at[:, 0].set(30.0)
    sp = jnp.ones(T)

    st     = _simulate(THETA, t_arr, scene_vel=sv,
                       scene_present=sp, target_present=jnp.zeros(T), key=3)
    eye    = np.array(st.plant[:, 0])
    ev     = np.gradient(eye, DT)
    burst  = extract_burst(st, THETA)[:, 0]
    spv    = extract_spv(t_np, ev, burst)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle('OKN Nystagmus — Sawtooth Waveform Zoom (30 deg/s scene)', fontsize=11)

    axes[0].plot(t_np, eye, color=utils.C['eye'], lw=0.8, label='eye position')
    axes[0].set_ylabel('Eye position (deg)')
    axes[0].set_title('Eye Position — sawtooth fast phases visible')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.25)

    axes[1].plot(t_np, ev,  color='steelblue', lw=0.5, alpha=0.4, label='eye vel (raw)')
    axes[1].plot(t_np, spv, color=utils.C['spv'], lw=2.0, label='SPV (fast phases removed)')
    axes[1].axhline(30.0, color=utils.C['scene'], lw=1.0, ls=':', alpha=0.7,
                    label='scene vel = 30 deg/s')
    axes[1].set_ylim(-80, 80)
    axes[1].set_ylabel('Eye velocity (deg/s)'); axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Eye Velocity — slow phases (OKN) + fast phase resets')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'okn_nystagmus_zoom', show=show)
    return utils.fig_meta(path, rp,
        title='OKN Nystagmus Zoom',
        description='First 15 s of OKN at 30 deg/s. Top: eye position showing sawtooth waveform. '
                    'Bottom: raw eye velocity (fast phases visible) and SPV (fast phases removed).',
        expected='Clear sawtooth; SPV ≈ 30 deg/s during slow phases; fast phases reset periodically.',
        citation='Raphan et al. (1979)',
        fig_type='behavior')


# ── Figure 3: VOR time-constant comparison ────────────────────────────────────

def _vor_tc(show):
    """SPV decay with VS vs without VS — TC extension comparison."""
    V_STIM   = 30.0
    ROT      = 20.0
    COAST    = 60.0
    BASELINE = 3.0

    B = int(BASELINE / DT)
    _head = km.head_rotation_step(V_STIM, rotate_dur=ROT, coast_dur=COAST, dt=DT)
    t_rot, hv_3d = _head.t, _head.rot_vel
    t_np    = np.concatenate([np.arange(1 - B, 1) * DT - DT, np.array(t_rot)])
    hv_full = np.concatenate([np.zeros((B, 3)), np.array(hv_3d)])
    T       = len(t_np)
    hv_1d   = hv_full[:, 0]

    theta_vs    = THETA
    theta_no_vs = with_brain(THETA, tau_vs=0.5, K_vs=0.001)   # near-zero VS

    results = {}
    for label, theta in [('with VS', theta_vs), ('no VS', theta_no_vs)]:
        st   = _simulate(theta, jnp.array(t_np), head_vel=jnp.array(hv_full),
                         scene_present=jnp.zeros(T), target_present=jnp.zeros(T),
                         key=4)
        eye  = np.array(st.plant[:, 0])
        ev   = np.gradient(eye, DT)
        bst  = extract_burst(st, theta)[:, 0]
        spv  = extract_spv(t_np, ev, bst)
        tau, t_fit, y_fit = fit_tc(t_np, spv, t_start=ROT + 1.0,
                                    t_end=ROT + COAST - 5.0,
                                    label=f'{label} post-rot TC')
        results[label] = (spv, tau, t_fit, y_fit)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('VOR Time-Constant Extension by Velocity Storage\n'
                 f'Step rotation {V_STIM:.0f} deg/s for {ROT:.0f} s, then coast',
                 fontsize=11)
    colors = {'with VS': utils.C['eye'], 'no VS': utils.C['no_vis']}
    vl = dict(color='k', lw=0.8, ls='--', alpha=0.5)

    axes[0].plot(t_np, -hv_1d, color=utils.C['head'], lw=1.0, ls=':', alpha=0.6,
                 label='−head vel')
    for label, (spv, tau, t_fit, y_fit) in results.items():
        axes[0].plot(t_np, spv, color=colors[label], lw=1.8, label=f'{label} SPV')
        if tau is not None:
            axes[0].plot(t_fit, y_fit, color=colors[label], lw=1.5, ls='--',
                         label=f'{label} τ = {tau:.1f} s')
    axes[0].axvline(0.0, **vl)
    axes[0].axvline(ROT, **vl)
    axes[0].set_ylabel('Slow-phase velocity (deg/s)')
    axes[0].set_title('SPV: with VS vs without VS')
    axes[0].legend(fontsize=9); ax_fmt(axes[0]); axes[0].set_xlim(-BASELINE, ROT + COAST)

    st_vs   = _simulate(theta_vs, jnp.array(t_np), head_vel=jnp.array(hv_full),
                         scene_present=jnp.zeros(T), target_present=jnp.zeros(T), key=4)
    vs_with = vs_net(st_vs)[:, 0]
    axes[1].plot(t_np, vs_with, color=utils.C['vs'], lw=1.8,
                 label=f'VS state (τ_vs = {theta_vs.brain.tau_vs:.0f} s)')
    axes[1].axvline(0.0, **vl)
    axes[1].axvline(ROT, **vl)
    axes[1].set_ylabel('VS net state (deg/s)'); axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Velocity Storage State — charges during rotation, drives post-rotatory SPV')
    axes[1].legend(fontsize=9); ax_fmt(axes[1]); axes[1].set_xlim(-BASELINE, ROT + COAST)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'vor_tc_comparison', show=show)
    return utils.fig_meta(path, rp,
        title='VOR Time-Constant Extension',
        description='SPV decay after step rotation: with VS (tau_vs=20 s) vs without VS (tau_vs≈0). '
                    'Bottom: velocity storage state trajectory during and after rotation.',
        expected='Without VS: TC ≈ 5 s (canal); with VS: TC ≈ 15–25 s (velocity storage extends it ~4×).',
        citation='Cohen, Matsuo & Raphan (1977) J Neurophysiol',
        fig_type='behavior')


# ── Figure 4: VOR/OKR signal cascade ──────────────────────────────────────────

def _cascade(show):
    """Internal signal cascade for VOR (left) and OKR (right)."""
    from scipy.special import softplus as _spf

    # ── VOR cascade ───────────────────────────────────────────────────────────
    _head = km.head_rotation_step(30.0, rotate_dur=15.0, coast_dur=15.0, dt=DT)
    t_rot, hv_3d = _head.t, _head.rot_vel
    t_vor = np.array(t_rot); T_vor = len(t_vor)
    st_v  = _simulate(THETA, jnp.array(t_rot), head_vel=jnp.array(hv_3d),
                      scene_present=jnp.zeros(T_vor),
                      target_present=jnp.zeros(T_vor), key=5)

    hv_1d  = np.array(hv_3d)[:, 0]
    x_c    = np.array(st_v.sensory[:, _IDX_C])
    x2     = x_c[:, N_CANALS:]
    k, f   = float(_SOFTNESS), float(FLOOR)
    y_c    = -f + _spf(k * (x2 + f)) / k + _spf(k * (x2 - f)) / k
    u_can  = (np.array(PINV_SENS) @ y_c.T).T[:, 0]
    x_vs_v = vs_net(st_v)[:, 0]
    x_ni_v = ni_net(st_v)[:, 0]
    eye_v  = np.array(st_v.plant[:, 0])
    ev_v   = np.gradient(eye_v, DT)
    bst_v  = extract_burst(st_v, THETA)[:, 0]

    # ── OKR cascade ───────────────────────────────────────────────────────────
    t_okn  = jnp.arange(0.0, 20.0, DT); t_okn_np = np.array(t_okn); T_okn = len(t_okn)
    sv     = jnp.zeros((T_okn, 3)).at[:, 0].set(30.0)
    sp     = jnp.ones(T_okn)
    st_o   = _simulate(THETA, t_okn, scene_vel=sv,
                       scene_present=sp, target_present=jnp.zeros(T_okn), key=6)

    x_vis  = np.array(st_o.sensory[:, _IDX_VIS])
    slip   = (np.array(C_slip) @ x_vis.T)[0, :]
    x_vs_o = vs_net(st_o)[:, 0]
    x_ni_o = ni_net(st_o)[:, 0]
    eye_o  = np.array(st_o.plant[:, 0])
    ev_o   = np.gradient(eye_o, DT)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 2, figsize=(14, 14),
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.35})
    fig.suptitle('VOR / OKR Signal Cascade\n'
                 'Left: VOR in dark (head rotation)   ·   Right: OKN (scene motion)',
                 fontsize=11)

    row_labels = ['Head / scene input (deg/s)',
                  'Canal afferents / visual delay output (deg/s)',
                  'Velocity storage x_VS (deg/s)',
                  'Neural integrator x_NI (deg)',
                  'Eye velocity (deg/s)']
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8)

    axes[0, 0].set_title('VOR (dark)'); axes[0, 1].set_title('OKR (lit scene, head still)')
    vl = dict(color='k', lw=0.8, ls='--', alpha=0.4)

    # VOR
    axes[0, 0].plot(t_vor,  hv_1d,      color=utils.C['head'],  lw=1.5, label='head vel')
    axes[0, 0].plot(t_vor, -hv_1d,      color=utils.C['dark'],  lw=1.0, ls=':', alpha=0.6,
                    label='−head vel (ideal eye)')
    axes[1, 0].plot(t_vor, u_can,        color=utils.C['canal'], lw=1.5, label='canal → u_canal')
    axes[2, 0].plot(t_vor, x_vs_v,       color=utils.C['vs'],    lw=1.5, label='x_VS net')
    axes[3, 0].plot(t_vor, x_ni_v,       color=utils.C['ni'],    lw=1.5, label='x_NI net')
    axes[4, 0].plot(t_vor, ev_v,          color=utils.C['eye'],   lw=1.0, alpha=0.5, label='eye vel (raw)')
    bst_spv = extract_spv(t_vor, ev_v, bst_v)
    axes[4, 0].plot(t_vor, bst_spv,       color=utils.C['spv'],   lw=1.8, label='SPV')
    axes[4, 0].set_xlabel('Time (s)', fontsize=8)

    for ax in axes[:, 0]:
        ax.axvline(15.0, **vl); ax_fmt(ax); ax.legend(fontsize=7)

    # OKR
    axes[0, 1].plot(t_okn_np, np.full(T_okn, 30.0), color=utils.C['scene'], lw=1.5,
                    label='scene vel = 30 deg/s')
    axes[1, 1].plot(t_okn_np, slip,   color='darkorange', lw=1.5, label='slip_delayed')
    axes[2, 1].plot(t_okn_np, x_vs_o, color=utils.C['vs'],  lw=1.5, label='x_VS net')
    axes[3, 1].plot(t_okn_np, x_ni_o, color=utils.C['ni'],  lw=1.5, label='x_NI net')
    axes[4, 1].plot(t_okn_np, ev_o,   color=utils.C['eye'], lw=1.0, alpha=0.5, label='eye vel (raw)')
    bst_o   = extract_burst(st_o, THETA)[:, 0]
    spv_okn = extract_spv(t_okn_np, ev_o, bst_o)
    axes[4, 1].plot(t_okn_np, spv_okn, color=utils.C['spv'], lw=1.8, label='SPV')
    axes[4, 1].set_xlabel('Time (s)', fontsize=8)

    for ax in axes[:, 1]:
        ax_fmt(ax); ax.legend(fontsize=7)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'vor_okr_cascade', show=show)
    return utils.fig_meta(path, rp,
        title='VOR / OKR Signal Cascade (Internal)',
        description='Left column (VOR in dark): head velocity → canal afferents → VS state → NI state → eye velocity. '
                    'Right column (OKN): scene velocity → delayed retinal slip → VS → NI → eye velocity.',
        expected='VOR: canal signal decays (TC ~5 s), VS extends it. '
                 'OKR: slip_delayed drives VS and NI until eye velocity matches scene.',
        citation='Raphan et al. (1979); Robinson (1975)',
        fig_type='cascade')


# ── Section entry point ────────────────────────────────────────────────────────

SECTION = dict(
    id='vor_okr', title='2. VOR / OKR',
    description='Vestibulo-ocular reflex and optokinetic response. '
                'Replicates Raphan et al. (1979) Fig.9: VOR in dark, OKN+OKAN, and VVOR. '
                'Tests post-rotatory and OKAN time constants, VVOR gain, and nystagmus waveform.',
)


def run(show=False):
    print('\n=== VOR / OKR ===')
    figs = []
    print('  1/4  Raphan Fig.9 replication …')
    figs.append(_raphan(show))
    print('  2/4  OKN nystagmus zoom …')
    figs.append(_okn_zoom(show))
    print('  3/4  VOR TC comparison …')
    figs.append(_vor_tc(show))
    print('  4/4  signal cascade …')
    figs.append(_cascade(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
