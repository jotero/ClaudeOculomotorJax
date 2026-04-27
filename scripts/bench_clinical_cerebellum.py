"""Clinical cerebellar lesion benchmarks.

Three standardised protocols applied to healthy vs lesioned conditions:
  1. Flocculus/Paraflocculus (FL/PFL)  — gaze-evoked nystagmus, rebound, impaired pursuit
  2. Bruns nystagmus                   — FL/PFL + UVH (cerebellopontine angle lesion)
  3. Nodulus/Uvula                     — prolonged OKAN via VS null adaptation

Usage:
    python -X utf8 scripts/bench_clinical_cerebellum.py
    python -X utf8 scripts/bench_clinical_cerebellum.py --show
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_clinical_utils as utils

import numpy as np
import jax.numpy as jnp
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, with_sensory, simulate, SimConfig,
    with_uvh,
)
from oculomotor.sim import kinematics as km
from oculomotor.analysis import (
    ax_fmt, extract_burst, vs_net, ni_net, ni_null, vs_null, extract_spv,
)

SHOW = '--show' in sys.argv
DT   = 0.001

# ── Base parameters — no noise, no null adaptation ────────────────────────────
# tau_vs_adapt frozen so VS null doesn't contaminate primary VOR/OKN TCs.
# tau_ni_adapt left at default (20 s) — NI null adaptation is the mechanism
# being studied in sections 1 and 2.
THETA = with_brain(
    with_sensory(PARAMS_DEFAULT,
                 sigma_canal=0.0, sigma_pos=0.0,
                 sigma_vel=0.0,   sigma_slip=0.0),
    tau_vs_adapt=9999.0,
)

# ── Lesion parameters ──────────────────────────────────────────────────────────
# FL/PFL: leaky NI + reduced pursuit gain; null adaptation intact.
THETA_FL = with_brain(THETA,
    tau_i=2.0,             # leaky NI → gaze-evoked nystagmus + rebound
    K_pursuit=0.5,         # reduced pursuit drive
    K_phasic_pursuit=1.0,  # reduced phasic pursuit onset
)

# Nodulus/Uvula: faster VS null adaptation → prolonged OKAN.
THETA_NOD = with_brain(THETA, tau_vs_adapt=60.0)

SECTION = dict(
    id='clin_cerebellum', title='B. Cerebellar Lesions',
    description='FL/PFL lesion (GEN, rebound, pursuit), Bruns nystagmus, Nodulus/Uvula (OKAN).',
)

C_HEALTHY = '#2166ac'
C_FL      = '#d6604d'
C_BRUNS   = '#35978f'
C_NOD     = '#1a9641'


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _eye_pos(states):
    return np.array(states.plant[:, 0])

def _eye_vel(states, dt=DT):
    return np.gradient(_eye_pos(states), dt)

def _spv(t_np, states, theta, **kw):
    ev    = _eye_vel(states)
    burst = np.array(extract_burst(states, theta)[:, 0])
    return extract_spv(t_np, ev, burst, **kw)

def _ni_net_yaw(states):
    return np.array(ni_net(states)[:, 0])

def _ni_null_yaw(states):
    return np.array(ni_null(states)[:, 0])

def _vs_net_yaw(states):
    return np.array(vs_net(states)[:, 0])

def _vs_null_yaw(states):
    return np.array(vs_null(states)[:, 0])


# ── 1. Flocculus / Paraflocculus ───────────────────────────────────────────────

def _test_fl_pfl(show):
    """FL/PFL lesion: GEN + rebound (Protocol A) + impaired smooth pursuit (Protocol B)."""
    print('  1/3  FL/PFL — gaze-evoked nystagmus, rebound, impaired pursuit …')

    # ── Protocol A: centre → eccentric hold → return ──────────────────────────
    PRE_DUR  = 1.0
    HOLD_DUR = 20.0
    POST_DUR = 15.0
    TOTAL_A  = PRE_DUR + HOLD_DUR + POST_DUR

    t_a     = np.arange(0.0, TOTAL_A, DT)
    T_A     = len(t_a)
    t_hold_end = PRE_DUR + HOLD_DUR

    # Step to +20° at t=PRE_DUR; return to 0° at t=t_hold_end.
    yaw_a = np.where(t_a < PRE_DUR, 0.0, np.where(t_a < t_hold_end, 20.0, 0.0))
    tgt_a = km.build_target(t_a, yaw_deg=yaw_a,
                             lin_vel=np.zeros((T_A, 3), np.float32))

    max_a = int(TOTAL_A / DT) + 1000
    sp_a  = jnp.ones(T_A)

    st_a_h = simulate(THETA,    t_a, target=tgt_a, scene_present_array=sp_a,
                      max_steps=max_a, return_states=True)
    st_a_c = simulate(THETA_FL, t_a, target=tgt_a, scene_present_array=sp_a,
                      max_steps=max_a, return_states=True)

    spv_a_h = _spv(t_a, st_a_h, THETA)
    spv_a_c = _spv(t_a, st_a_c, THETA_FL)

    # ── Protocol B: 15 deg/s pursuit ramp ────────────────────────────────────
    RAMP_VEL = 15.0
    RAMP_DUR = 4.0
    TOTAL_B  = 7.0

    t_b  = np.arange(0.0, TOTAL_B, DT)
    T_B  = len(t_b)

    # Constant-velocity ramp: zero vel after ramp ends (target holds position)
    tgt_b = km.build_target(t_b,
                             vel_yaw_deg_s=np.where(t_b < RAMP_DUR, RAMP_VEL, 0.0))

    max_b = int(TOTAL_B / DT) + 500
    sp_b  = jnp.ones(T_B)

    st_b_h = simulate(THETA,    t_b, target=tgt_b, scene_present_array=sp_b,
                      max_steps=max_b, return_states=True)
    st_b_c = simulate(THETA_FL, t_b, target=tgt_b, scene_present_array=sp_b,
                      max_steps=max_b, return_states=True)

    # ── Figure: 3 rows × 2 cols ────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        'Flocculus / Paraflocculus Lesion — Gaze-Evoked Nystagmus, Rebound, Impaired Pursuit\n'
        f'NI leak: τ_i {THETA.brain.tau_i:.0f} s → {THETA_FL.brain.tau_i:.0f} s   '
        f'Pursuit gain: K_p {THETA.brain.K_pursuit:.1f} → {THETA_FL.brain.K_pursuit:.1f}',
        fontsize=11, fontweight='bold')

    # Left column: GEN + rebound (Protocol A)
    ax = axes[0, 0]
    ax.plot(t_a, _eye_pos(st_a_h), color=C_HEALTHY, lw=1.5, label='Healthy')
    ax.plot(t_a, _eye_pos(st_a_c), color=C_FL,      lw=1.5, label='FL/PFL lesion')
    ax.axhline(20, color='gray', lw=0.6, ls=':', alpha=0.4)
    ax.axvline(PRE_DUR,    color='gray', lw=0.6, ls=':', alpha=0.4)
    ax.axvline(t_hold_end, color='k',   lw=0.7, ls='--', alpha=0.4, label='Gaze return')
    ax_fmt(ax, ylabel='Eye position (deg)')
    ax.legend(fontsize=8)
    ax.set_title('Eye position — GEN drift during eccentric hold; rebound after return')

    ax = axes[1, 0]
    ax.plot(t_a, spv_a_h, color=C_HEALTHY, lw=2.0, label='Healthy SPV')
    ax.plot(t_a, spv_a_c, color=C_FL,      lw=2.0, label='FL/PFL lesion SPV')
    ax.axvline(PRE_DUR,    color='gray', lw=0.6, ls=':', alpha=0.4)
    ax.axvline(t_hold_end, color='k',   lw=0.7, ls='--', alpha=0.4)
    ax_fmt(ax, ylabel='SPV (deg/s)', ylim=(-25, 25))
    ax.legend(fontsize=8)
    ax.set_title('SPV — centripetal GEN during hold; centrifugal rebound after return')

    ax = axes[2, 0]
    ax.plot(t_a, _ni_net_yaw(st_a_c),  color=C_FL,     lw=1.5, label='NI net (lesion)')
    ax.plot(t_a, _ni_null_yaw(st_a_c), color='#e08214', lw=1.5, label='NI null (lesion)')
    ax.plot(t_a, _ni_net_yaw(st_a_h),  color=C_HEALTHY, lw=0.8, ls='--', alpha=0.4, label='NI net (healthy)')
    ax.axvline(PRE_DUR,    color='gray', lw=0.6, ls=':', alpha=0.4)
    ax.axvline(t_hold_end, color='k',   lw=0.7, ls='--', alpha=0.4)
    ax_fmt(ax, ylabel='NI state (deg)', xlabel='Time (s)')
    ax.legend(fontsize=8)
    ax.set_title('Neural Integrator — null drifts during hold; persists after return → rebound')

    # Right column: pursuit (Protocol B)
    tgt_angle = np.clip(RAMP_VEL * t_b, 0.0, RAMP_VEL * RAMP_DUR)
    tgt_vel_b = np.where(t_b < RAMP_DUR, RAMP_VEL, 0.0)

    ax = axes[0, 1]
    ax.plot(t_b, tgt_angle,           color='gray', lw=1.2, ls='--', label='Target')
    ax.plot(t_b, _eye_pos(st_b_h),    color=C_HEALTHY, lw=1.5, label='Healthy')
    ax.plot(t_b, _eye_pos(st_b_c),    color=C_FL,      lw=1.5, label='FL/PFL lesion')
    ax.axvline(RAMP_DUR, color='k', lw=0.7, ls='--', alpha=0.4)
    ax_fmt(ax, ylabel='Eye / target position (deg)')
    ax.legend(fontsize=8)
    ax.set_title(f'Pursuit — {RAMP_VEL:.0f} deg/s ramp; catch-up saccades in lesion')

    ax = axes[1, 1]
    ax.plot(t_b, tgt_vel_b,           color='gray', lw=1.2, ls='--', label='Target velocity')
    ax.plot(t_b, _eye_vel(st_b_h),    color=C_HEALTHY, lw=1.5, label='Healthy')
    ax.plot(t_b, _eye_vel(st_b_c),    color=C_FL,      lw=1.5, label='FL/PFL lesion')
    ax.axvline(RAMP_DUR, color='k', lw=0.7, ls='--', alpha=0.4)
    ax_fmt(ax, ylabel='Eye velocity (deg/s)', xlabel='Time (s)')
    ax.legend(fontsize=8)
    ax.set_title('Pursuit velocity — reduced gain (K_pursuit); catch-up saccades visible')

    # Parameter summary panel
    ax = axes[2, 1]
    ax.axis('off')
    ax.text(0.05, 0.95,
        'FL/PFL lesion parameters\n\n'
        'Neural Integrator leak:\n'
        f'  τ_i         : {THETA.brain.tau_i:.0f} s → {THETA_FL.brain.tau_i:.0f} s\n\n'
        'NI null adaptation:\n'
        f'  τ_ni_adapt  : {THETA.brain.tau_ni_adapt:.0f} s (unchanged)\n\n'
        'Smooth pursuit:\n'
        f'  K_pursuit   : {THETA.brain.K_pursuit:.1f} → {THETA_FL.brain.K_pursuit:.1f}\n'
        f'  K_phasic    : {THETA.brain.K_phasic_pursuit:.1f} → {THETA_FL.brain.K_phasic_pursuit:.1f}\n\n'
        'Refs:\n'
        '  Cannon & Robinson (1985) Biol Cybern\n'
        '  Zee et al. (1980) Brain\n'
        '  Stone & Lisberger (1990) J Neurophysiol',
        transform=ax.transAxes, fontsize=9, va='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', alpha=0.8))

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'clin_cereb_fl_pfl', show=show)
    return utils.fig_meta(path, rp,
        title='FL/PFL Lesion — GEN, Rebound Nystagmus, Impaired Pursuit',
        description='Leaky NI (τ_i = 2 s) causes centripetal slow drift during eccentric gaze. '
                    'NI null adaptation persists after gaze return → rebound nystagmus. '
                    'Reduced pursuit gain (K_pursuit = 0.5) with catch-up saccades.',
        expected='GEN: slow drift centripetal at eccentric position; SPV grows with eccentricity. '
                 'Rebound: nystagmus reversal after return to primary; direction opposite to prior gaze. '
                 'Pursuit: catch-up saccades; steady-state velocity error at ramp velocity.',
        citation='Cannon & Robinson (1985); Zee et al. (1980); Stone & Lisberger (1990)',
    )


# ── 2. Bruns Nystagmus ─────────────────────────────────────────────────────────

def _test_bruns(show):
    """Bruns nystagmus: FL/PFL lesion combined with left UVH (CPA tumour model)."""
    print('  2/3  Bruns nystagmus — FL/PFL + left UVH …')

    # Bruns: leaky NI + null adaptation + left UVH (canal asymmetry).
    # b_lesion=85 keeps spontaneous nystagmus moderate (w_est ≈ 15 deg/s)
    # to avoid NI saturation over the 48 s protocol.
    THETA_BRUNS = with_brain(
        with_uvh(THETA, side='left', canal_gain_frac=0.1, b_lesion=85.0),
        tau_i=3.0,
        tau_ni_adapt=15.0,
        tau_vs=10.0,
    )

    SEG   = 12.0
    TOTAL = 4 * SEG
    t_br  = np.arange(0.0, TOTAL, DT)
    T_BR  = len(t_br)

    # Four gaze positions: centre → +20° contra → centre → −20° ipsi (lesioned side)
    def _step_target(t_arr, segs):
        yaw = np.zeros_like(t_arr)
        for t_start, deg in segs:
            yaw[t_arr >= t_start] = deg
        return yaw

    yaw_br = _step_target(t_br, [(0, 0), (SEG, 20), (2*SEG, 0), (3*SEG, -20)])
    tgt_br = km.build_target(t_br, yaw_deg=yaw_br,
                              lin_vel=np.zeros((T_BR, 3), np.float32))

    max_br = int(TOTAL / DT) + 2000
    sp_br  = jnp.ones(T_BR)

    st_br = simulate(THETA_BRUNS, t_br, target=tgt_br,
                     scene_present_array=sp_br,
                     max_steps=max_br, return_states=True)

    spv_br = _spv(t_br, st_br, THETA_BRUNS)

    seg_labels = ['Centre', '+20° contra\n(intact side)', 'Centre', '−20° ipsi\n(lesioned side)']
    seg_vlines = [SEG * i for i in range(1, 4)]

    # ── Figure: 3 rows × 1 col ────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        'Bruns Nystagmus — Leaky NI + Rebound + Left UVH\n'
        'Cerebellopontine angle lesion: large slow ipsilesional, small fast contraversive',
        fontsize=11, fontweight='bold')

    ax = axes[0]
    ax.plot(t_br, yaw_br,            color='gray', lw=0.8, ls='--', label='Target')
    ax.plot(t_br, _eye_pos(st_br),   color=C_BRUNS, lw=1.2, label='Eye pos')
    for v in seg_vlines:
        ax.axvline(v, color='k', lw=0.5, ls='--', alpha=0.3)
    for i, lbl in enumerate(seg_labels):
        ax.text(i * SEG + SEG / 2, -32, lbl, ha='center', fontsize=7.5, color='#444444')
    ax_fmt(ax, ylabel='Eye position (deg)')
    ax.legend(fontsize=8)
    ax.set_title('Eye position — large drift ipsilesional; smaller oscillation contraversive')

    ax = axes[1]
    ax.plot(t_br, spv_br, color=C_BRUNS, lw=2.0)
    for v in seg_vlines:
        ax.axvline(v, color='k', lw=0.5, ls='--', alpha=0.3)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax_fmt(ax, ylabel='SPV (deg/s)', ylim=(-60, 60))
    ax.set_title('SPV — spontaneous at centre; large slow ipsilesional; rebound after each shift')

    ax = axes[2]
    ax.plot(t_br, _ni_net_yaw(st_br),  color=C_BRUNS,   lw=1.5, label='NI net')
    ax.plot(t_br, _ni_null_yaw(st_br), color='#e08214',  lw=1.5, ls='--', label='NI null')
    for v in seg_vlines:
        ax.axvline(v, color='k', lw=0.5, ls='--', alpha=0.3)
    ax_fmt(ax, ylabel='NI state (deg)', xlabel='Time (s)')
    ax.legend(fontsize=8)
    ax.set_title('Neural Integrator — null follows eye position; persists after shift → rebound')

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'clin_cereb_bruns', show=show)
    return utils.fig_meta(path, rp,
        title='Bruns Nystagmus — FL/PFL + Left UVH',
        description='Combination of leaky NI (τ_i = 3 s) and left UVH (canal gain × 0.1, '
                    'b_lesion = 85). Ipsilesional gaze: large-amplitude, slow GEN amplified by '
                    'peripheral VS imbalance. Contraversive gaze: small, faster rebound nystagmus.',
        expected='Primary position: mild spontaneous rightward nystagmus from UVH. '
                 'Ipsilesional (leftward) gaze: large slow centripetal drift + fast phases. '
                 'Contraversive (rightward) gaze: smaller oscillation with rebound component.',
        citation='Bruns (1902); Leigh & Zee (2015) The Neurology of Eye Movements',
    )


# ── 3. Nodulus / Uvula ─────────────────────────────────────────────────────────

def _test_nodulus(show):
    """Nodulus/Uvula lesion: prolonged OKAN via velocity storage null adaptation."""
    print('  3/3  Nodulus/Uvula — prolonged OKAN (VS null adaptation) …')

    ON_DUR  = 30.0
    OFF_DUR = 60.0
    TOTAL   = ON_DUR + OFF_DUR

    t_nd   = np.arange(0.0, TOTAL, DT)
    T_ND   = len(t_nd)
    t_np   = t_nd

    # Full-field scene motion (OKN stimulus); no fixation target during OKAN
    v_sc   = np.where(t_nd < ON_DUR, 30.0, 0.0)
    v_sc_3d = np.zeros((T_ND, 3), np.float32)
    v_sc_3d[:, 0] = v_sc
    sc_pr  = jnp.array(np.where(t_nd < ON_DUR, 1.0, 0.0), dtype=jnp.float32)
    tgt_pr = jnp.zeros(T_ND, dtype=jnp.float32)
    scene  = km.build_kinematics(t_nd, rot_vel=v_sc_3d)

    max_nd = int(TOTAL / DT) + 1000
    st_h = simulate(THETA,     t_nd,
                    scene=scene, scene_present_array=sc_pr,
                    target_present_array=tgt_pr,
                    max_steps=max_nd, return_states=True)
    st_a = simulate(THETA_NOD, t_nd,
                    scene=scene, scene_present_array=sc_pr,
                    target_present_array=tgt_pr,
                    max_steps=max_nd, return_states=True)

    spv_h = _spv(t_np, st_h, THETA)
    spv_a = _spv(t_np, st_a, THETA_NOD)

    # ── Figure: 3 rows × 1 col ────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(
        'Nodulus / Uvula Lesion — Prolonged OKAN via Velocity Storage Null Adaptation\n'
        f'τ_vs_adapt: {THETA.brain.tau_vs_adapt:.0f} s (frozen) → {THETA_NOD.brain.tau_vs_adapt:.0f} s',
        fontsize=11, fontweight='bold')

    ax = axes[0]
    ax.plot(t_np, spv_h, color=C_HEALTHY, lw=2.0,
            label=f'Healthy  τ_vs_adapt = {THETA.brain.tau_vs_adapt:.0f} s')
    ax.plot(t_np, spv_a, color=C_NOD,     lw=2.0,
            label=f'Nodulus lesion  τ_vs_adapt = {THETA_NOD.brain.tau_vs_adapt:.0f} s')
    ax.axvline(ON_DUR, color='k', lw=0.7, ls='--', alpha=0.5, label='Lights off')
    ax.axhline(30, color='gray', lw=0.7, ls=':', alpha=0.4)
    ax_fmt(ax, ylabel='SPV (deg/s)')
    ax.legend(fontsize=8)
    ax.set_title('OKN + OKAN — nodulus lesion prolongs OKAN after lights off')

    ax = axes[1]
    ax.plot(t_np, _vs_net_yaw(st_h), color=C_HEALTHY, lw=1.5,
            label='VS net (healthy)')
    ax.plot(t_np, _vs_net_yaw(st_a), color=C_NOD,     lw=1.5,
            label='VS net (nodulus lesion)')
    ax.axvline(ON_DUR, color='k', lw=0.7, ls='--', alpha=0.5)
    ax_fmt(ax, ylabel='VS net yaw (deg/s)')
    ax.legend(fontsize=8)
    ax.set_title('Velocity Storage — same charging; decays toward null (not 0) → extended OKAN')

    ax = axes[2]
    ax.plot(t_np, _vs_null_yaw(st_h), color=C_HEALTHY, lw=1.0, ls='--', alpha=0.5,
            label=f'VS null (healthy, τ_vs_adapt = {THETA.brain.tau_vs_adapt:.0f} s)')
    ax.plot(t_np, _vs_null_yaw(st_a), color=C_NOD,     lw=2.0,
            label=f'VS null (nodulus lesion, τ_vs_adapt = {THETA_NOD.brain.tau_vs_adapt:.0f} s)')
    ax.axvline(ON_DUR, color='k', lw=0.7, ls='--', alpha=0.5)
    ax_fmt(ax, ylabel='VS null yaw (deg/s)', xlabel='Time (s)')
    ax.legend(fontsize=8)
    ax.set_title('VS null — builds during OKN; residual shifts decay target → longer OKAN')

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'clin_cereb_nodulus', show=show)
    return utils.fig_meta(path, rp,
        title='Nodulus/Uvula Lesion — Prolonged OKAN',
        description='Faster VS null adaptation (τ_vs_adapt = 60 s) causes the null to build '
                    'during sustained OKN. After lights off, VS decays toward null (not 0), '
                    'producing an extended OKAN beyond what τ_vs alone would predict.',
        expected='OKN gain ≈ same in both conditions. OKAN: healthy decays with τ_vs ≈ 20 s; '
                 'nodulus lesion OKAN sustained longer due to non-zero VS null floor. '
                 'VS null plot shows build-up during OKN then persistence after lights off.',
        citation='Cohen et al. (1992) J Neurophysiol',
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f'\n=== Clinical: Cerebellar Lesions ===')
    figs = [
        _test_fl_pfl(SHOW),
        _test_bruns(SHOW),
        _test_nodulus(SHOW),
    ]
    print()


if __name__ == '__main__':
    main()
