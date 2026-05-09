"""Clinical benchmarks for vergence and accommodation.

Simulates four standard clinical tests across five patient profiles:

  Tests
  -----
  1. Near Point of Convergence (NPC) pull-in + Cover test (phorias)
  2. Fusional vergence ranges (BI/BO prism ramp at distance and near)
  3. Amplitude of accommodation (push-up) + accommodative facility (±2 D flipper)
  4. AC/A measurement (gradient + heterophoria methods)

  Patient profiles
  ----------------
  - Healthy           — PARAMS_DEFAULT
  - Convergence insufficiency (CI)        — weak fusional convergence
  - Convergence excess (CE) / accomm ET   — high AC/A + esophoric bias
  - Accommodative insufficiency (AI)      — weak blur drive
  - Presbyopia                            — slow stiff lens plant

Usage:
    python -X utf8 scripts/bench_clinical.py
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

from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain
from oculomotor.sim import kinematics as km
from oculomotor.analysis import ax_fmt

SHOW = '--show' in sys.argv
DT   = 0.001
IPD  = 0.064
KEY  = jax.random.PRNGKey(0)


SECTION = dict(
    id='clinical', title='7. Clinical benchmarks',
    description='Standard clinical tests (NPC, cover test, fusional vergence ranges, '
                'amplitude of accommodation, AC/A measurement) across five patient profiles: '
                'healthy, convergence insufficiency, convergence excess (accommodative ET), '
                'accommodative insufficiency, and presbyopia.',
)


# ── Patient profiles ──────────────────────────────────────────────────────────
# Each profile = (label, color, Params). All derived from PARAMS_DEFAULT.

def _make_patients():
    return {
        'Healthy': {
            'color': '#222222',
            'params': PARAMS_DEFAULT,
        },
        'Convergence insufficiency': {
            # Weak fusional vergence: low phasic + fast-integrator gain, weak SVBN burst.
            # Clinical: exo at near, remote NPC, reduced PFV at near.
            'color': '#1f77b4',
            'params': with_brain(PARAMS_DEFAULT,
                                 K_phasic_verg=1.5, K_verg=0.5,
                                 g_svbn_conv=25.0),
        },
        'Convergence excess (accomm. ET)': {
            # High AC/A; accommodative effort drives excessive convergence at near.
            # Clinical: eso at near, normal at distance.
            'color': '#d62728',
            'params': with_brain(PARAMS_DEFAULT, AC_A=12.0),
        },
        'Accommodative insufficiency': {
            # Weak blur drive: reduced K_acc_fast/slow → inadequate accommodative response.
            # Clinical: lag of accommodation at near, asthenopia, blur with near work.
            'color': '#2ca02c',
            'params': with_brain(PARAMS_DEFAULT,
                                 K_acc_fast=1.0, K_acc_slow=0.05),
        },
        'Presbyopia': {
            # Stiff lens: long plant TC, reduced amplitude of accommodation.
            # Clinical: near blur from age ~40; lens cannot rapidly change shape.
            'color': '#9467bd',
            'params': with_brain(PARAMS_DEFAULT,
                                 tau_acc_plant=1.0, K_acc_fast=1.5),
        },
    }


def _verg_angle_deg(depth_m):
    return 2.0 * np.degrees(np.arctan(IPD / 2.0 / depth_m))


def _depth_for_verg(verg_deg):
    return IPD / 2.0 / np.tan(np.radians(verg_deg) / 2.0)


# ── Test 1: NPC pull-in + Cover test ──────────────────────────────────────────

def _npc_and_cover(show):
    """Two panels: NPC pull-in (target moves toward subject) and Cover test (eye
    occluded → drifts to phoria position).

    NPC: target z(t) ramps from 1.0 m to 0.05 m linearly over 10 s. Break is
         where vergence stops growing (eye position decouples from target).
    Cover: at t=2 s cover R eye (target_present_R = 0). L eye keeps fixating;
           R eye drifts to its dissociated phoria position. Repeat at distance
           (3 m) and near (40 cm).
    """
    patients = _make_patients()

    # ── NPC pull-in ──────────────────────────────────────────────────────────
    T_NPC = 10.0
    t_npc = np.arange(0.0, T_NPC, DT)
    Tn = len(t_npc)
    z_far  = 1.0
    z_near = 0.05
    depth_npc = z_far + (z_near - z_far) * (t_npc / T_NPC)        # linear pull-in
    pt_npc = np.stack([np.zeros(Tn), np.zeros(Tn), depth_npc], axis=1)
    target_npc = km.build_target(t_npc, lin_pos=pt_npc)
    geo_npc = np.array([_verg_angle_deg(d) for d in depth_npc])

    # ── Cover test: distance (3 m) and near (0.4 m) ──────────────────────────
    T_COV = 5.0
    t_cov = np.arange(0.0, T_COV, DT)
    Tc = len(t_cov)
    cover_t = 2.0   # cover R eye at t=2 s
    target_present_R = np.where(t_cov >= cover_t, 0.0, 1.0).astype(np.float32)
    target_present_L = np.ones(Tc, dtype=np.float32)

    fig, axes = plt.subplots(3, len(patients), figsize=(20, 11), sharex='row')
    fig.suptitle('Clinical: Near Point of Convergence + Cover Test (phorias)',
                 fontsize=12, fontweight='bold')

    for col, (name, info) in enumerate(patients.items()):
        params = info['params']
        col_color = info['color']

        # NPC pull-in
        st_npc = simulate(params, t_npc, target=target_npc,
                          scene_present_array=np.ones(Tn),
                          return_states=True, key=KEY)
        eL = np.array(st_npc.plant.left[:, 0]); eR = np.array(st_npc.plant.right[:, 0])
        verg_npc = eL - eR

        ax = axes[0, col]
        ax.plot(depth_npc * 100, verg_npc, color=col_color, lw=1.5, label='Measured')
        ax.plot(depth_npc * 100, geo_npc,  color='gray', lw=0.8, ls='--', label='Geometric')
        # Find break point: where vergence error from geometric exceeds 5° or vergence stops
        err = geo_npc - verg_npc
        break_idx = np.argmax(err > 5.0) if (err > 5.0).any() else len(err) - 1
        break_cm = depth_npc[break_idx] * 100
        ax.axvline(break_cm, color='red', lw=0.8, ls=':',
                   label=f'NPC break ≈ {break_cm:.1f} cm')
        ax.set_xlim(z_far * 100, z_near * 100)   # near at right
        ax.set_xscale('log')
        ax.set_xticks([100, 50, 20, 10, 5])
        ax.set_xticklabels(['100', '50', '20', '10', '5'])
        ax_fmt(ax, ylabel='Vergence (deg)' if col == 0 else '',
               xlabel='Target distance (cm, log)')
        ax.set_title(name, fontsize=10, color=col_color, fontweight='bold')
        ax.legend(fontsize=7.5)

        # Cover test at distance (3 m)
        p_far = np.tile([0.0, 0.0, 3.0], (Tc, 1)).astype(np.float32)
        st_cov_f = simulate(params, t_cov, target=km.build_target(t_cov, lin_pos=p_far),
                            scene_present_array=np.ones(Tc),
                            target_present_L_array=target_present_L,
                            target_present_R_array=target_present_R,
                            return_states=True, key=KEY)
        eL_f = np.array(st_cov_f.plant.left[:, 0]); eR_f = np.array(st_cov_f.plant.right[:, 0])

        ax = axes[1, col]
        ax.plot(t_cov, eL_f, color='#1f77b4', lw=1.3, label='L eye (fixating)')
        ax.plot(t_cov, eR_f, color='#d62728', lw=1.3, ls='--', label='R eye (covered)')
        ax.axvline(cover_t, color='gray', lw=0.7, ls=':')
        ax.axhline(0, color='gray', lw=0.5, ls='-', alpha=0.3)
        # Phoria: sign convention — converged eyes (eL > eR) give negative R−L, which is ESO.
        # Positive R−L = eyes diverged = EXO.
        phoria_f = float(eR_f[int(0.9 * Tc):].mean()) - float(eL_f[int(0.9 * Tc):].mean())
        sign_f = 'exo' if phoria_f > 0 else 'eso'
        ax.text(0.02, 0.95, f'Distance phoria: {abs(phoria_f):.1f}° {sign_f}',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax_fmt(ax, ylabel='Eye yaw (deg)' if col == 0 else '', xlabel='Time (s)')
        if col == 0: ax.legend(fontsize=7.5)

        # Cover test at near (0.4 m)
        p_near = np.tile([0.0, 0.0, 0.4], (Tc, 1)).astype(np.float32)
        st_cov_n = simulate(params, t_cov, target=km.build_target(t_cov, lin_pos=p_near),
                            scene_present_array=np.ones(Tc),
                            target_present_L_array=target_present_L,
                            target_present_R_array=target_present_R,
                            return_states=True, key=KEY)
        eL_n = np.array(st_cov_n.plant.left[:, 0]); eR_n = np.array(st_cov_n.plant.right[:, 0])

        ax = axes[2, col]
        ax.plot(t_cov, eL_n, color='#1f77b4', lw=1.3, label='L eye (fixating)')
        ax.plot(t_cov, eR_n, color='#d62728', lw=1.3, ls='--', label='R eye (covered)')
        ax.axvline(cover_t, color='gray', lw=0.7, ls=':')
        ax.axhline(0, color='gray', lw=0.5, ls='-', alpha=0.3)
        # Phoria at near: same sign convention as distance
        phoria_n = float(eR_n[int(0.9 * Tc):].mean()) - float(eL_n[int(0.9 * Tc):].mean())
        sign_n = 'exo' if phoria_n > 0 else 'eso'
        ax.text(0.02, 0.95, f'Near phoria: {abs(phoria_n):.1f}° {sign_n}',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax_fmt(ax, ylabel='Eye yaw (deg)' if col == 0 else '', xlabel='Time (s)')

    axes[0, 0].set_ylabel('NPC pull-in\nVergence (deg)', fontsize=9)
    axes[1, 0].set_ylabel('Cover @ 3 m\nEye yaw (deg)',  fontsize=9)
    axes[2, 0].set_ylabel('Cover @ 0.4 m\nEye yaw (deg)', fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path, rp = utils.save_fig(fig, 'clinical_npc_cover', show=show, figs_dir=utils.CLIN_FIGS_DIR, base_dir=utils.CLIN_DIR,
                              params=PARAMS_DEFAULT,
                              conditions='Lit, near-point convergence pull-in + cover test (3 m and 0.4 m) — control vs simulated patient profiles')
    return utils.fig_meta(path, rp,
        title='NPC pull-in and Cover Test (phorias at distance + near)',
        description='Top: vergence vs. target distance during NPC pull-in (1 m → 5 cm over 10 s); '
                    'red line marks break point (vergence err > 5° from geometric). '
                    'Middle/bottom: cover test at 3 m and 0.4 m — R eye covered at t=2 s; '
                    'covered eye drifts to dissociated phoria position. Phoria sign labelled.',
        expected='Healthy: NPC < 6 cm; orthophoria or small exo at distance, slight exo at near. '
                 'CI: remote NPC (≥10 cm), large exo at near. '
                 'CE/accomm. ET: large eso at near, small/normal at distance. '
                 'AI/Presbyopia: NPC may be remote due to weak vergence drive secondary to weak accomm.',
        citation='Scheiman & Wick (2008) Clinical Management of Binocular Vision; '
                 'Cooper & Jamal (2012) Optometry; '
                 'CITT Study Group (2008) Arch Ophthalmol',
    )


# ── Test 2: Fusional vergence ranges ──────────────────────────────────────────

def _fusional_ranges(show):
    """Slowly ramp prism BI then BO until break.

    BO (base-out) prism on R eye creates a temporal image displacement → demands
    extra convergence (positive fusional vergence, PFV). The fusional system
    compensates until the disparity exceeds Panum's area / NPC limits.

    BI (base-in) demands divergence (NFV).

    Clinical norms (Sheard's): PFV BO break at near ≈ 18–22°; PFV at distance ≈ 12–15°.
    NFV BI break at near ≈ 13°; NFV at distance ≈ 7°.
    """
    patients = _make_patients()

    T_RAMP = 10.0   # 0 → 30° over 10 s
    t = np.arange(0.0, T_RAMP, DT)
    T = len(t)
    DEPTHS = [(3.0, 'Distance (3 m)'), (0.4, 'Near (0.4 m)')]

    fig, axes = plt.subplots(len(DEPTHS) * 2, len(patients),
                             figsize=(20, 13), sharex='row')
    fig.suptitle('Clinical: Fusional Vergence Ranges (BI + BO prism ramps)',
                 fontsize=12, fontweight='bold')

    for col, (name, info) in enumerate(patients.items()):
        params = info['params']
        col_color = info['color']

        for row_pair, (depth_m, dep_lbl) in enumerate(DEPTHS):
            geo = _verg_angle_deg(depth_m)
            target = km.build_target(
                t, lin_pos=np.tile([0.0, 0.0, depth_m], (T, 1)).astype(np.float32))

            # BO ramp: positive yaw shift on R eye = nasal → forces convergence
            # Convention: prism positive yaw = apparent rightward shift of field;
            # for R eye, BO = nasal = leftward = NEGATIVE yaw. So ramp negative.
            prism_R_BO = np.zeros((T, 3), dtype=np.float32)
            prism_R_BO[:, 0] = -np.linspace(0.0, 30.0, T)
            st_BO = simulate(params, t, target=target,
                             scene_present_array=np.ones(T),
                             prism_R_array=prism_R_BO,
                             return_states=True, key=KEY)
            eL_BO = np.array(st_BO.plant.left[:, 0]); eR_BO = np.array(st_BO.plant.right[:, 0])
            verg_BO  = eL_BO - eR_BO

            prism_R_BI = np.zeros((T, 3), dtype=np.float32)
            prism_R_BI[:, 0] = np.linspace(0.0, 30.0, T)   # BI = positive yaw on R eye
            st_BI = simulate(params, t, target=target,
                             scene_present_array=np.ones(T),
                             prism_R_array=prism_R_BI,
                             return_states=True, key=KEY)
            eL_BI = np.array(st_BI.plant.left[:, 0]); eR_BI = np.array(st_BI.plant.right[:, 0])
            verg_BI  = eL_BI - eR_BI

            # Identify break: where vergence stops following the ramp
            prism_BO = np.linspace(0.0, 30.0, T)
            demand_BO = geo + prism_BO
            err_BO = demand_BO - verg_BO
            # Break: vergence error > 5° (lost fusion)
            br_BO_idx = np.argmax(err_BO > 5.0) if (err_BO > 5.0).any() else len(err_BO) - 1
            break_BO_prism = prism_BO[br_BO_idx]

            demand_BI = geo - prism_BO
            err_BI    = verg_BI - demand_BI
            br_BI_idx = np.argmax(err_BI > 5.0) if (err_BI > 5.0).any() else len(err_BI) - 1
            break_BI_prism = prism_BO[br_BI_idx]

            # BO panel
            ax = axes[row_pair * 2, col]
            ax.plot(prism_BO, verg_BO,    color=col_color, lw=1.5, label='Measured vergence')
            ax.plot(prism_BO, demand_BO,  color='gray',    lw=0.8, ls='--', label='Demand (geo + BO)')
            ax.axvline(break_BO_prism, color='red', lw=0.8, ls=':',
                       label=f'BO break ≈ {break_BO_prism:.1f}°')
            if row_pair == 0 and col == 0:
                ax.set_title(name, fontsize=10, color=col_color, fontweight='bold')
            else:
                ax.set_title(name if row_pair == 0 else '', fontsize=10,
                             color=col_color, fontweight='bold')
            ax_fmt(ax, ylabel=f'{dep_lbl}\nVerg (deg) — BO' if col == 0 else '',
                   xlabel='Prism (deg)')
            ax.legend(fontsize=7)

            # BI panel
            ax = axes[row_pair * 2 + 1, col]
            ax.plot(prism_BO, verg_BI,    color=col_color, lw=1.5, label='Measured vergence')
            ax.plot(prism_BO, demand_BI,  color='gray',    lw=0.8, ls='--', label='Demand (geo − BI)')
            ax.axvline(break_BI_prism, color='red', lw=0.8, ls=':',
                       label=f'BI break ≈ {break_BI_prism:.1f}°')
            ax_fmt(ax, ylabel=f'Verg (deg) — BI' if col == 0 else '',
                   xlabel='Prism (deg)')
            ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path, rp = utils.save_fig(fig, 'clinical_fusional_ranges', show=show, figs_dir=utils.CLIN_FIGS_DIR, base_dir=utils.CLIN_DIR,
                              params=PARAMS_DEFAULT,
                              conditions='Lit, prism-induced disparity ramps — fusional vergence ranges (BO/BI) per profile')
    return utils.fig_meta(path, rp,
        title='Fusional vergence ranges (BI + BO ramps at distance and near)',
        description='Slow prism ramp (0 → 30° over 10 s) on R eye. '
                    'Top two rows: distance (3 m) BO then BI; bottom two: near (0.4 m). '
                    'Red line marks vergence break (error > 5°).',
        expected='Healthy: PFV (BO) break ≈ 18–25° at near, 12–15° at distance; '
                 'NFV (BI) break ≈ 13° near, 7° distance. '
                 'CI: reduced PFV at near (<10°). '
                 'CE: reduced NFV at near (eyes already over-converged).',
        citation='Sheard (1930); Saladin (1988) Optom Vis Sci; '
                 'Scheiman & Wick (2008) Clinical Management of Binocular Vision',
    )


# ── Test 3: Amplitude of accommodation + accommodative facility ──────────────

def _accom_amplitude_facility(show):
    """Push-up amplitude (slow target approach to first sustained blur) and
    ±2 D flipper facility (cycles per minute).
    """
    patients = _make_patients()

    # Push-up amplitude: target approaches from 1 m to as close as possible over 15 s
    T_PUSH = 15.0
    t_p = np.arange(0.0, T_PUSH, DT)
    Tp = len(t_p)
    z_far = 1.0; z_close = 0.04   # 4 cm closest
    depth_p = z_far + (z_close - z_far) * (t_p / T_PUSH)
    pt_p = np.stack([np.zeros(Tp), np.zeros(Tp), depth_p], axis=1)
    target_p = km.build_target(t_p, lin_pos=pt_p)
    demand_p = 1.0 / depth_p

    # Flipper facility: alternate +2 D / −2 D every 1 s for 30 s; target at 40 cm
    T_FLIP = 30.0
    t_f = np.arange(0.0, T_FLIP, DT)
    Tf = len(t_f)
    flip_period = 1.0    # alternate every 1 s (1 cycle = 2 s)
    # +2D from 0 to flip_period, −2D from flip_period to 2*flip_period, repeat
    phase = (t_f / flip_period).astype(int) % 2
    lens_flip = np.where(phase == 0, 2.0, -2.0).astype(np.float32)
    pt_f = np.tile([0.0, 0.0, 0.4], (Tf, 1)).astype(np.float32)
    target_f = km.build_target(t_f, lin_pos=pt_f)
    demand_f_base = 1.0 / 0.4   # 2.5 D base demand
    demand_f = demand_f_base + lens_flip   # plus lens reduces demand, minus increases it
    # Wait — plus lens reduces accommodation needed (target appears closer requires less accom).
    # In our model lens_L_array adds to defocus (1/z + lens + RE − x_plant). So lens > 0
    # increases defocus → increases accommodation demand. That matches "minus lens" in clinical
    # terminology (minus lens = simulates near = needs more accomm). Flip sign:
    demand_f = demand_f_base - lens_flip   # +2D lens (clinical plus) → lens_flip < 0 in defocus drive
    # Simpler: track what x_plant should be when accommodation tracks the lens-modified demand.
    # For visualization, just plot: demand = 1/z + lens (model's defocus offset).
    demand_f_visualize = demand_f_base + lens_flip   # 4.5 / 0.5 alternating

    fig, axes = plt.subplots(2, len(patients), figsize=(20, 8), sharex='row')
    fig.suptitle('Clinical: Amplitude of Accommodation + ±2 D Flipper Facility',
                 fontsize=12, fontweight='bold')

    for col, (name, info) in enumerate(patients.items()):
        params = info['params']
        col_color = info['color']

        # Push-up amplitude
        st_p = simulate(params, t_p, target=target_p,
                        scene_present_array=np.ones(Tp),
                        return_states=True, key=KEY)
        acc_p = np.array(st_p.acc_plant[:, 0])
        # Find peak accommodation reached (= amplitude)
        amp_d = float(np.max(acc_p))
        # Hofstetter age-50 expected: 18.5 - 0.3*50 = 3.5 D minimum
        ax = axes[0, col]
        ax.plot(depth_p * 100, demand_p, 'k--', lw=0.9, alpha=0.6, label='Demand')
        ax.plot(depth_p * 100, acc_p,    color=col_color, lw=1.5, label='Accommodation')
        ax.axhline(amp_d, color='red', lw=0.7, ls=':',
                   label=f'Peak amp ≈ {amp_d:.1f} D')
        ax.set_xlim(z_far * 100, z_close * 100)
        ax.set_xscale('log')
        ax.set_xticks([100, 50, 20, 10, 5])
        ax.set_xticklabels(['100', '50', '20', '10', '5'])
        ax.set_title(name, fontsize=10, color=col_color, fontweight='bold')
        ax_fmt(ax, ylabel='Accommodation (D)' if col == 0 else '',
               xlabel='Target distance (cm, log)')
        ax.legend(fontsize=7.5)

        # Flipper facility
        st_f = simulate(params, t_f, target=target_f,
                        scene_present_array=np.ones(Tf),
                        lens_L_array=lens_flip, lens_R_array=lens_flip,
                        return_states=True, key=KEY)
        acc_f = np.array(st_f.acc_plant[:, 0])
        # Count cycles where accommodation reaches within 0.5 D of demand
        # (one cycle = +2D phase + −2D phase)
        # Detect by zero-crossings of (acc - demand_avg) — count completed flips
        n_phase_changes = int(np.sum(np.abs(np.diff(phase))))   # = T_FLIP / flip_period
        acc_at_phase_end = []
        for i in range(n_phase_changes):
            t_end = (i + 1) * flip_period
            idx = int(t_end / DT) - 1
            if idx < Tf:
                acc_at_phase_end.append(acc_f[idx])
        acc_at_phase_end = np.array(acc_at_phase_end)
        # Successful tracking: |acc - demand_for_that_phase| < 0.75 D
        target_per_phase = np.array([demand_f_visualize[int((i + 0.5) * flip_period / DT)]
                                      for i in range(n_phase_changes)])
        success = np.abs(acc_at_phase_end - target_per_phase) < 0.75
        cpm = (np.sum(success) / 2.0) / (T_FLIP / 60.0)   # cycles (one cycle = 2 phases) per minute

        ax = axes[1, col]
        ax.plot(t_f, demand_f_visualize, 'k--', lw=0.8, alpha=0.6, label='Demand')
        ax.plot(t_f, acc_f, color=col_color, lw=1.3, label='Accommodation')
        ax.text(0.02, 0.95, f'Facility ≈ {cpm:.1f} cpm',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax_fmt(ax, ylabel='Accommodation (D)' if col == 0 else '',
               xlabel='Time (s)')
        ax.legend(fontsize=7.5)

    axes[0, 0].set_ylabel('Push-up\nAccommodation (D)', fontsize=9)
    axes[1, 0].set_ylabel('±2 D flipper\nAccommodation (D)', fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path, rp = utils.save_fig(fig, 'clinical_accom_amp_facility', show=show, figs_dir=utils.CLIN_FIGS_DIR, base_dir=utils.CLIN_DIR,
                              params=PARAMS_DEFAULT,
                              conditions='Lit — push-up amplitude + ±2D flipper facility test (per simulated profile)')
    return utils.fig_meta(path, rp,
        title='Amplitude of accommodation + accommodative facility',
        description='Top: push-up to 4 cm; peak accommodation = amplitude. '
                    'Bottom: ±2 D flipper at 40 cm, 1 s alternation; cycles per minute counted '
                    'when accommodation reaches within 0.75 D of demand within each phase.',
        expected='Healthy adult <40 yo: amplitude ≥ 5 D; facility ≥ 11 cpm. '
                 'Presbyopia: amplitude ≤ 2 D, very low facility. '
                 'AI: amplitude reduced, facility reduced. '
                 'CI/CE: typically normal accommodation amplitude.',
        citation='Hofstetter (1965) Optom Mon; Zellers et al. (1984) Am J Optom; '
                 'Wick et al. (1992); Donders, Hofstetter formulas',
    )


# ── Test 4: AC/A measurement (gradient + heterophoria) ────────────────────────

def _ac_a_measurement(show):
    """Gradient AC/A: at fixed 0.4 m target, apply +1 D and −1 D lenses bilaterally
    and measure vergence change. AC/A = Δvergence / 2 D.

    Heterophoria AC/A: subtract phoria at distance (3 m) from phoria at near (0.4 m),
    divide by accommodative demand difference. AC/A_heterophoria = (P_near − P_far) / Δdemand.
    Both expressed in pd/D (1 pd ≈ 0.5729° vergence).
    """
    patients = _make_patients()

    T_SETTLE = 6.0
    t = np.arange(0.0, T_SETTLE, DT)
    T = len(t)

    # Gradient method: target at 40 cm; sweep lens 0 → +1 D at t=1, then 0 → −1 D
    # Simpler protocol: three separate runs (no lens, +1 D, −1 D) at near, measure SS vergence
    p_near = np.tile([0.0, 0.0, 0.4], (T, 1)).astype(np.float32)
    p_far  = np.tile([0.0, 0.0, 3.0], (T, 1)).astype(np.float32)
    target_near = km.build_target(t, lin_pos=p_near)
    target_far  = km.build_target(t, lin_pos=p_far)

    # Cover test (heterophoria) traces — reuse cover-test protocol
    cover_t = 2.0
    target_present_R = np.where(t >= cover_t, 0.0, 1.0).astype(np.float32)
    target_present_L = np.ones(T, dtype=np.float32)

    fig, axes = plt.subplots(2, len(patients), figsize=(20, 8))
    fig.suptitle('Clinical: AC/A Measurement (gradient + heterophoria)',
                 fontsize=12, fontweight='bold')

    for col, (name, info) in enumerate(patients.items()):
        params = info['params']
        col_color = info['color']

        # Gradient method: 3 lens conditions at 0.4 m
        ss_verg = []
        for lens_d in [-1.0, 0.0, +1.0]:
            lens_arr = np.full(T, lens_d, dtype=np.float32)
            st = simulate(params, t, target=target_near,
                          scene_present_array=np.ones(T),
                          lens_L_array=lens_arr, lens_R_array=lens_arr,
                          return_states=True, key=KEY)
            v = float((st.plant[int(0.85 * T):, 0] - st.plant[int(0.85 * T):, 3]).mean())
            ss_verg.append(v)

        # AC/A_gradient = Δvergence (deg) / 2 D × (1 pd / 0.5729 deg) = pd/D
        delta_v_deg = ss_verg[2] - ss_verg[0]    # +1 D vs −1 D
        aca_gradient = (delta_v_deg / 2.0) / 0.5729   # pd/D

        ax = axes[0, col]
        ax.plot([-1, 0, +1], ss_verg, 'o-', color=col_color, lw=1.8, ms=10)
        # Slope line
        slope_deg_per_D = delta_v_deg / 2.0
        ax.plot([-1, +1], [ss_verg[1] - slope_deg_per_D, ss_verg[1] + slope_deg_per_D],
                'k--', lw=0.8, alpha=0.6)
        ax.set_xticks([-1, 0, 1])
        ax.set_title(name, fontsize=10, color=col_color, fontweight='bold')
        ax.text(0.02, 0.95, f'AC/A_grad ≈ {aca_gradient:.1f} pd/D',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_fmt(ax, ylabel='SS vergence at 40 cm (deg)' if col == 0 else '',
               xlabel='Lens (D)')

        # Heterophoria method
        st_far = simulate(params, t, target=target_far,
                          scene_present_array=np.ones(T),
                          target_present_L_array=target_present_L,
                          target_present_R_array=target_present_R,
                          return_states=True, key=KEY)
        st_near = simulate(params, t, target=target_near,
                           scene_present_array=np.ones(T),
                           target_present_L_array=target_present_L,
                           target_present_R_array=target_present_R,
                           return_states=True, key=KEY)
        eL_f, eR_f = np.array(st_far.plant.left[:, 0]),  np.array(st_far.plant.right[:, 0])
        eL_n, eR_n = np.array(st_near.plant.left[:, 0]), np.array(st_near.plant.right[:, 0])
        # Phoria = R − L when R is dissociated (positive = R drifted right relative to L = exo)
        idx_ss = slice(int(0.85 * T), T)
        phoria_f_deg = float((eR_f[idx_ss] - eL_f[idx_ss]).mean())
        phoria_n_deg = float((eR_n[idx_ss] - eL_n[idx_ss]).mean())
        # AC/A_heterophoria = (phoria_near − phoria_far) / (acc_demand_near − acc_demand_far) (pd/D)
        delta_demand = (1.0 / 0.4) - (1.0 / 3.0)   # ≈ 2.17 D
        # Convert deg → pd: pd = deg / 0.5729
        delta_phoria_pd = (phoria_n_deg - phoria_f_deg) / 0.5729
        aca_het = delta_phoria_pd / delta_demand    # pd/D

        ax = axes[1, col]
        ax.bar(['Distance\n(3 m)', 'Near\n(0.4 m)'],
               [phoria_f_deg, phoria_n_deg],
               color=[col_color, col_color], alpha=0.7,
               edgecolor='k', linewidth=0.8)
        ax.axhline(0, color='gray', lw=0.7)
        for x, p in zip([0, 1], [phoria_f_deg, phoria_n_deg]):
            sign = 'exo' if p > 0 else 'eso'   # +R−L = R diverged → exo; −R−L = R converged → eso
            ax.text(x, p + (0.3 if p >= 0 else -0.5), f'{abs(p):.1f}° {sign}',
                    ha='center', fontsize=8)
        ax.text(0.02, 0.95, f'AC/A_het ≈ {aca_het:.1f} pd/D',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_fmt(ax, ylabel='Phoria (deg)' if col == 0 else '')

    axes[0, 0].set_ylabel('Gradient AC/A\nSS vergence (deg)', fontsize=9)
    axes[1, 0].set_ylabel('Heterophoria AC/A\nPhoria (deg)', fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path, rp = utils.save_fig(fig, 'clinical_aca_measurement', show=show, figs_dir=utils.CLIN_FIGS_DIR, base_dir=utils.CLIN_DIR,
                              params=PARAMS_DEFAULT,
                              conditions='Lit, gradient AC/A measurement protocol — vergence response to ±lens add')
    return utils.fig_meta(path, rp,
        title='AC/A measurement (gradient + heterophoria)',
        description='Top: gradient AC/A — vergence change for ±1 D lens at 0.4 m, slope = AC/A. '
                    'Bottom: heterophoria AC/A — distance vs near phoria difference / Δdemand.',
        expected='Healthy: AC/A 4–6 pd/D both methods. '
                 'CE/accomm. ET: AC/A high (≥8 pd/D), classic sign. '
                 'CI: AC/A low (<3 pd/D). '
                 'Heterophoria method always ≥ gradient by 1–2 pd/D (proximal vergence component).',
        citation='Morgan (1944) Am J Optom; Eskridge (1971); '
                 'Scheiman & Wick (2008) Ch 1',
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run(show=False):
    print('\n=== Clinical ===')
    figs = []
    print('  1/4  NPC + cover test …')
    figs.append(_npc_and_cover(show))
    print('  2/4  fusional vergence ranges …')
    figs.append(_fusional_ranges(show))
    print('  3/4  amplitude + facility …')
    figs.append(_accom_amplitude_facility(show))
    print('  4/4  AC/A measurement …')
    figs.append(_ac_a_measurement(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
