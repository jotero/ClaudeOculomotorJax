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
    N_CANALS, FLOOR, _SOFTNESS, PINV_SENS,
)
from oculomotor.models.brain_models.perception_cyclopean import C_slip
from oculomotor.models.brain_models.brain_model         import _IDX_CYC_BRAIN
from oculomotor.sim import kinematics as km
from oculomotor.analysis import ax_fmt, extract_canal, vs_net, vs_null, ni_net, fit_tc, extract_spv_states

SHOW  = '--show' in sys.argv
DT    = 0.001

# Behavioral params: realistic noise on (defaults from SensoryParams + BrainParams).
THETA = PARAMS_DEFAULT

# Cascade DEBUG params: all sensory + accumulator noise off so the cascade traces
# show only the pure signal flow (no microsaccades, no fixational drift).
THETA_NOISELESS = with_brain(
    with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_slip=0.0, sigma_pos=0.0, sigma_vel=0.0),
    sigma_acc=0.0)


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
    spv_vor_d = -extract_spv_states(st_vor, t_vor)[:, 0]   # negate: compensatory = positive
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
    spv_okn_d = extract_spv_states(st_okn, t_okn)[:, 0]    # positive: eye tracks scene
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
    spv_vvor_d = -extract_spv_states(st_vvor, t_vor)[:, 0]
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
    path, rp = utils.save_fig(fig, 'vor_raphan_fig9', show=show, params=THETA,
                              conditions='Dark/lit conditions per panel — head velocity step + scene step (Raphan 1979 Fig.9 protocol)')
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
    ev  = np.gradient(eye, DT)
    spv = extract_spv_states(st, t_np)[:, 0]

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
    path, rp = utils.save_fig(fig, 'okn_nystagmus_zoom', show=show, params=THETA,
                              conditions='Lit, full-field scene velocity step (OKN sawtooth + post-OKAN)')
    return utils.fig_meta(path, rp,
        title='OKN Nystagmus Zoom',
        description='First 15 s of OKN at 30 deg/s. Top: eye position showing sawtooth waveform. '
                    'Bottom: raw eye velocity (fast phases visible) and SPV (fast phases removed).',
        expected='Clear sawtooth; SPV ≈ 30 deg/s during slow phases; fast phases reset periodically.',
        citation='Raphan et al. (1979)',
        fig_type='behavior')


# ── Figure 3: VOR/OKR signal cascade ──────────────────────────────────────────

def _cascade(show):
    """Internal signal cascade for VOR (left) and OKR (right)."""
    from scipy.special import softplus as _spf

    # ── VOR cascade ───────────────────────────────────────────────────────────
    _head = km.head_rotation_step(30.0, rotate_dur=15.0, coast_dur=15.0, dt=DT)
    t_rot, hv_3d = _head.t, _head.rot_vel
    t_vor = np.array(t_rot); T_vor = len(t_vor)
    st_v  = _simulate(THETA_NOISELESS, jnp.array(t_rot), head_vel=jnp.array(hv_3d),
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

    # ── OKR cascade ───────────────────────────────────────────────────────────
    t_okn  = jnp.arange(0.0, 20.0, DT); t_okn_np = np.array(t_okn); T_okn = len(t_okn)
    sv     = jnp.zeros((T_okn, 3)).at[:, 0].set(30.0)
    sp     = jnp.ones(T_okn)
    st_o   = _simulate(THETA_NOISELESS, t_okn, scene_vel=sv,
                       scene_present=sp, target_present=jnp.zeros(T_okn), key=6)

    x_cyc  = np.array(st_o.brain[:, _IDX_CYC_BRAIN])
    slip   = (np.array(C_slip) @ x_cyc.T)[0, :]
    x_vs_o = vs_net(st_o)[:, 0]
    x_ni_o = ni_net(st_o)[:, 0]
    eye_o  = np.array(st_o.plant[:, 0])
    ev_o   = np.gradient(eye_o, DT)

    ZOOM_T0, ZOOM_T1 = 5.0, 10.0   # 5-second window for nystagmus zoom rows

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(7, 2, figsize=(14, 14),
                             gridspec_kw={'hspace': 0.38, 'wspace': 0.32,
                                          'height_ratios': [1, 1, 1, 1, 1, 0.8, 0.8]})
    fig.suptitle('VOR / OKR Signal Cascade\n'
                 'Left: VOR in dark (head rotation)   ·   Right: OKN (scene motion)',
                 fontsize=11)

    row_labels = ['Head / scene input (deg/s)',
                  'Canal afferents / visual delay output (deg/s)',
                  'Velocity storage x_VS (deg/s)',
                  'Neural integrator x_NI (deg)',
                  'Eye velocity (deg/s)',
                  f'Eye pos zoom [{ZOOM_T0:.0f}–{ZOOM_T1:.0f} s] (deg)',
                  f'Eye vel zoom [{ZOOM_T0:.0f}–{ZOOM_T1:.0f} s] (deg/s)']
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
    bst_spv = extract_spv_states(st_v, t_vor)[:, 0]
    axes[4, 0].plot(t_vor, ev_v,    color=utils.C['eye'], lw=1.0, alpha=0.5, label='eye vel (raw)')
    axes[4, 0].plot(t_vor, bst_spv, color=utils.C['spv'], lw=1.8, label='SPV')
    axes[4, 0].set_xlabel('Time (s)', fontsize=8)

    for ax in axes[:5, 0]:
        ax.axvline(15.0, **vl); ax_fmt(ax); ax.legend(fontsize=7)

    # OKR
    axes[0, 1].plot(t_okn_np, np.full(T_okn, 30.0), color=utils.C['scene'], lw=1.5,
                    label='scene vel = 30 deg/s')
    axes[1, 1].plot(t_okn_np, slip,   color='darkorange', lw=1.5, label='slip_delayed')
    axes[2, 1].plot(t_okn_np, x_vs_o, color=utils.C['vs'],  lw=1.5, label='x_VS net')
    axes[3, 1].plot(t_okn_np, x_ni_o, color=utils.C['ni'],  lw=1.5, label='x_NI net')
    axes[4, 1].plot(t_okn_np, ev_o,   color=utils.C['eye'], lw=1.0, alpha=0.5, label='eye vel (raw)')
    spv_okn = extract_spv_states(st_o, t_okn_np)[:, 0]
    axes[4, 1].plot(t_okn_np, spv_okn, color=utils.C['spv'], lw=1.8, label='SPV')
    axes[4, 1].set_xlabel('Time (s)', fontsize=8)

    for ax in axes[:5, 1]:
        ax_fmt(ax); ax.legend(fontsize=7)

    # ── Nystagmus zoom rows (rows 5–6): 5-second window, ±100 deg/s ──────────
    # VOR zoom: t=[5, 10] s — during rotation, fast phases visible
    zm_v = (t_vor >= ZOOM_T0) & (t_vor <= ZOOM_T1)
    eye_v_pos = np.array(st_v.plant[:, 0])
    axes[5, 0].plot(t_vor[zm_v], eye_v_pos[zm_v], color=utils.C['eye'], lw=1.0)
    axes[5, 0].set_xlim(ZOOM_T0, ZOOM_T1); ax_fmt(axes[5, 0])
    axes[5, 0].set_xlabel('Time (s)', fontsize=8)

    axes[6, 0].plot(t_vor[zm_v], ev_v[zm_v],     color=utils.C['eye'], lw=0.8, alpha=0.5, label='eye vel')
    axes[6, 0].plot(t_vor[zm_v], bst_spv[zm_v],  color=utils.C['spv'], lw=1.8, label='SPV')
    axes[6, 0].set_xlim(ZOOM_T0, ZOOM_T1); axes[6, 0].set_ylim(-100, 100)
    ax_fmt(axes[6, 0]); axes[6, 0].legend(fontsize=7)
    axes[6, 0].set_xlabel('Time (s)', fontsize=8)

    # OKR zoom: t=[5, 10] s — during steady-state OKN nystagmus
    zm_o = (t_okn_np >= ZOOM_T0) & (t_okn_np <= ZOOM_T1)
    eye_o_pos = np.array(st_o.plant[:, 0])
    axes[5, 1].plot(t_okn_np[zm_o], eye_o_pos[zm_o], color=utils.C['eye'], lw=1.0)
    axes[5, 1].set_xlim(ZOOM_T0, ZOOM_T1); ax_fmt(axes[5, 1])
    axes[5, 1].set_xlabel('Time (s)', fontsize=8)

    axes[6, 1].plot(t_okn_np[zm_o], ev_o[zm_o],     color=utils.C['eye'], lw=0.8, alpha=0.5, label='eye vel')
    axes[6, 1].plot(t_okn_np[zm_o], spv_okn[zm_o],  color=utils.C['spv'], lw=1.8, label='SPV')
    axes[6, 1].set_xlim(ZOOM_T0, ZOOM_T1); axes[6, 1].set_ylim(-100, 100)
    ax_fmt(axes[6, 1]); axes[6, 1].legend(fontsize=7)
    axes[6, 1].set_xlabel('Time (s)', fontsize=8)
    fig.tight_layout(pad=0.4)
    path, rp = utils.save_fig(fig, 'vor_okr_cascade', show=show, params=THETA_NOISELESS,
                              conditions='VOR, VVOR, OKR cascade — head + scene combinations across 3 columns (noiseless DEBUG)')
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
    print('  1/2  Raphan Fig.9 replication …')
    figs.append(_raphan(show))
    print('  2/2  signal cascade …')
    figs.append(_cascade(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
