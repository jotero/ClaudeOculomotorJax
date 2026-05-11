"""Runner — converts a SimulationScenario into a simulation + figure.

Entry point::

    from oculomotor.llm_pipeline.runner import run_scenario
    fig = run_scenario(scenario)      # returns matplotlib Figure
    fig.savefig('output.png', dpi=150, bbox_inches='tight')

The runner does three things:
    1. Build stimulus arrays from ``scenario.head_motion``, ``scenario.target``,
       and ``scenario.visual`` using ``oculomotor.sim.stimuli``.
    2. Build model parameters from ``scenario.patient`` overrides.
    3. Run ``simulate()`` and plot the requested panels.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from oculomotor.sim import stimuli as stim
from oculomotor.sim import kinematics as km
from oculomotor.sim.kinematics import TargetTrajectory
from oculomotor.llm_pipeline.scenario import SimulationScenario, SimulationComparison, Patient
from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, with_sensory, simulate,
)
from oculomotor.models.brain_models import saccade_generator as sg_mod
from oculomotor.models.brain_models.perception_cyclopean import C_slip, C_pos, C_vel, C_target_visible
from oculomotor.models.brain_models import perception_cyclopean as _pc_mod
from oculomotor.analysis import read_brain_acts


# ── Stimulus builder ──────────────────────────────────────────────────────────

def _resolve_target_angular_shorthand(segments):
    """Convert rot_* angular shorthand on target segments to Cartesian lin_*.

    For each target segment: if rot_yaw_0 / rot_pitch_0 is set but the
    corresponding lin_x_0 / lin_y_0 is not, compute the Cartesian equivalent
    at the segment's viewing distance (lin_z_0 or carry-forward z, default 1 m).
    Same for velocity: rot_yaw_vel → lin_x_vel.
    """
    import math
    resolved = []
    carry_z = 1.0   # default viewing distance

    for seg in segments:
        z = seg.lin_z_0 if seg.lin_z_0 is not None else carry_z
        carry_z = z

        updates = {}

        # Position shorthand: rot_yaw_0 / rot_pitch_0 → lin_x_0 / lin_y_0
        if seg.rot_yaw_0 is not None and seg.lin_x_0 is None:
            updates['lin_x_0'] = math.tan(math.radians(seg.rot_yaw_0)) * z
        if seg.rot_pitch_0 is not None and seg.lin_y_0 is None:
            updates['lin_y_0'] = math.tan(math.radians(seg.rot_pitch_0)) * z

        # Velocity shorthand: rot_*_vel (deg/s) → lin_*_vel (m/s)
        # Use small-angle approximation: Δ(tan θ)/Δt ≈ (dθ/dt)·(π/180) for small θ
        if seg.rot_yaw_vel is not None and seg.lin_x_vel is None:
            updates['lin_x_vel'] = seg.rot_yaw_vel * (math.pi / 180.0) * z
        if seg.rot_pitch_vel is not None and seg.lin_y_vel is None:
            updates['lin_y_vel'] = seg.rot_pitch_vel * (math.pi / 180.0) * z

        resolved.append(seg.model_copy(update=updates) if updates else seg)

    return resolved


def _build_stimulus(scenario: SimulationScenario) -> dict:
    """Convert BodySegment lists to simulator inputs.

    Returns a dict containing:
    - '_head_km', '_scene_km', '_target_tt': trajectory objects for simulate()
    - flat arrays for _draw_panel() / _build_sim_data()
    - visual flag arrays used by both simulate() and _draw_panel()
    """
    dt  = 0.001
    T   = int(round(scenario.duration_s / dt))
    t   = np.arange(T, dtype=np.float32) * dt

    # ── Build 6-DOF trajectories ──────────────────────────────────────────────
    head_km  = km.build_kinematics_from_segments(scenario.head,  T, dt, default_lin_pos=(0.0, 0.0, 0.0))
    scene_km = km.build_kinematics_from_segments(scenario.scene, T, dt, default_lin_pos=(0.0, 0.0, 5.0))

    target_segs_resolved = _resolve_target_angular_shorthand(scenario.target)
    target_km = km.build_kinematics_from_segments(target_segs_resolved, T, dt, default_lin_pos=(0.0, 0.0, 1.0))
    target_tt = TargetTrajectory(t=target_km.t, lin_pos=target_km.lin_pos, lin_vel=target_km.lin_vel)

    # ── Flat arrays for plotting ───────────────────────────────────────────────

    head_vel = head_km.rot_vel                              # (T, 3) deg/s
    v_scene  = scene_km.rot_vel                            # (T, 3) deg/s

    rel_pos  = target_km.lin_pos - head_km.lin_pos         # (T, 3) m
    rel_vel  = target_km.lin_vel - head_km.lin_vel         # (T, 3) m/s
    p_target = rel_pos.astype(np.float32)

    # Angular velocity of target in head frame (deg/s) — for plotting only
    depth   = np.maximum(rel_pos[:, 2], 0.05)
    denom_x = depth ** 2 + rel_pos[:, 0] ** 2
    denom_y = depth ** 2 + rel_pos[:, 1] ** 2
    v_target = np.zeros((T, 3), dtype=np.float32)
    v_target[:, 0] = ((rel_vel[:, 0] * depth - rel_pos[:, 0] * rel_vel[:, 2]) / denom_x * (180.0 / np.pi))
    v_target[:, 1] = ((rel_vel[:, 1] * depth - rel_pos[:, 1] * rel_vel[:, 2]) / denom_y * (180.0 / np.pi))

    # Visual flags — per-eye scene_present, target_present, and strobe flag
    spL, spR, tpL, tpR, ts = stim.build_visual_flags(scenario.visual, T, dt)

    return dict(
        t_array                 = t,
        # Trajectory objects for simulate() — stripped before _draw_panel
        _head_km                = head_km,
        _scene_km               = scene_km,
        _target_tt              = target_tt,
        # Flat arrays for _draw_panel() / _build_sim_data()
        head_vel_array          = jnp.array(head_vel),
        p_target_array          = jnp.array(p_target),
        v_target_array          = jnp.array(v_target),
        v_scene_array           = jnp.array(v_scene),
        # Visual flags — used by both simulate() and _draw_panel()
        scene_present_L_array   = jnp.array(spL),
        scene_present_R_array   = jnp.array(spR),
        target_present_L_array  = jnp.array(tpL),
        target_present_R_array  = jnp.array(tpR),
        target_strobed_array    = jnp.array(ts),
    )


# ── Parameter builder ─────────────────────────────────────────────────────────

def _build_params(patient: Patient):
    """Convert Patient overrides to a Params NamedTuple.

    Delegates to patient_builder.apply_patient — that module owns the
    introspection that maps each YAML-defined Patient field onto the right
    BrainParams / SensoryParams / PlantParams slot (including aliases like
    b_vs_L / b_vs_R that write into a slice of the underlying b_vs array).
    """
    from oculomotor.llm_pipeline.patient_builder import apply_patient
    return apply_patient(patient, PARAMS_DEFAULT)


# ── Signal extraction helpers ─────────────────────────────────────────────────

def _extract_signals(states, params, t_np: np.ndarray) -> dict:
    """Extract named signals from a SimState trajectory."""
    dt = t_np[1] - t_np[0]

    eye_pos_L = np.array(states.plant.left)        # (T, 3) left  eye rotation (deg)
    eye_pos_R = np.array(states.plant.right)       # (T, 3) right eye rotation (deg)
    version   = 0.5 * (eye_pos_L + eye_pos_R)       # (T, 3) conjugate (version)
    vergence  = eye_pos_L - eye_pos_R               # (T, 3) vergence angle (deg, + = converged)

    # ── BrainState field reads ───────────────────────────────────────────────
    sm_st = states.brain.sm
    ni_st = states.brain.ni
    sg_st = states.brain.sg
    pu_st = states.brain.pu
    va_st = states.brain.va
    pc_st = states.brain.pc

    # Vergence integrator (yaw component, signed)
    x_verg = np.array(va_st.verg_fast)            # (T, 3) full vergence integrator state

    # Pursuit NET signed signal (T, 3) — push-pull difference
    x_pursuit = np.array(pu_st.R - pu_st.L)        # (T, 3) NET

    # Cyclopean LP block (T, 43) — only used here for legacy compatibility with
    # the C_pos readout matrix; safe to flatten via pc.to_array.
    x_vis = np.array(jax.vmap(_pc_mod.to_array)(pc_st))   # (T, 43)

    # Eye velocity (version derivative — same as L eye vel when version ≈ L)
    w_eye = np.gradient(version, dt, axis=0)

    # VS and NI nets: x_L − x_R
    w_est = np.array(sm_st.vs_L - sm_st.vs_R)   # VS net  (T, 3)
    x_ni  = np.array(ni_st.L - ni_st.R)         # NI net  (T, 3)

    # Retinal signals — cyclopean (single cascade, pre-fused)
    e_pos_delayed = x_vis @ np.array(C_pos).T   # (T, 3)

    # Saccade burst (re-compute from SG state + cyclopean delayed retinal signals)
    def _burst_at(state):
        x_vis_   = _pc_mod.to_array(state.brain.pc)
        e_pd     = C_pos @ x_vis_
        gate     = jnp.clip((C_target_visible @ x_vis_)[0], 0.0, 1.0)
        x_ni_net = state.brain.ni.L - state.brain.ni.R   # (3,)
        sg_acts  = sg_mod.read_activations(state.brain.sg)
        sg_w     = sg_mod.read_weights(state.brain.sg)
        _, u     = sg_mod.step(sg_acts, sg_w, e_pd, gate, x_ni_net,
                                jnp.zeros(3), jnp.zeros(3), params.brain)
        return u
    u_burst = np.array(jax.vmap(_burst_at)(states))  # (T, 3)

    # SG sub-states — direct field access on (T, ...) leaves
    e_held = np.array(sg_st.e_held)
    z_opn  = np.array(sg_st.z_opn)
    z_acc  = np.array(sg_st.z_acc)
    z_trig = np.array(sg_st.z_trig)

    # ── Cerebellar activations (vmapped over time) ───────────────────────────
    cb = read_brain_acts(states, params).cb
    cb_vpf_drive    = np.array(cb.vpf_drive)                  # (T, 3) gated target EC → pursuit
    cb_fl_okr_drive = np.array(cb.fl_okr_drive)               # (T, 3) gated scene  EC → VS
    cb_fl_drive     = np.array(cb.fl_drive)                   # (T, 3) NI leak cancellation
    cb_fl_vs_drive  = np.array(cb.fl_vs_drive)                # (T, 3) VS leak cancellation
    cb_nu_drive     = np.array(cb.nu_drive)                   # (T, 3) gravity-axis dumping
    cb_sat_target   = np.array(cb.saccadic_suppression_target)  # (T,) pursuit gate
    cb_sat_scene    = np.array(cb.saccadic_suppression_scene)   # (T,) scene gate
    cb_pred_err     = np.array(cb.pred_err)                   # (T, 3) target_slip + ec (diagnostic)
    cb_ec_target    = np.array(cb.ec_target)                  # (T, 3) delayed target EC
    cb_ec_scene     = np.array(cb.ec_scene)                   # (T, 3) delayed scene  EC

    return dict(
        eye_pos        = version,          # conjugate version — used by most panels
        eye_pos_L      = eye_pos_L,        # per-eye: left
        eye_pos_R      = eye_pos_R,        # per-eye: right
        vergence       = vergence,         # L − R (deg); positive = converged
        x_verg         = x_verg,           # vergence integrator state
        eye_vel        = w_eye,
        w_est          = w_est,
        x_ni           = x_ni,
        x_pursuit      = x_pursuit,
        e_pos_delayed  = e_pos_delayed,
        u_burst        = u_burst,
        z_opn          = z_opn,
        z_acc          = z_acc,
        z_trig         = z_trig,
        e_held         = e_held,
        # Cerebellar signals
        cb_vpf_drive    = cb_vpf_drive,
        cb_fl_okr_drive = cb_fl_okr_drive,
        cb_fl_drive     = cb_fl_drive,
        cb_fl_vs_drive  = cb_fl_vs_drive,
        cb_nu_drive     = cb_nu_drive,
        cb_sat_target   = cb_sat_target,
        cb_sat_scene    = cb_sat_scene,
        cb_pred_err     = cb_pred_err,
        cb_ec_target    = cb_ec_target,
        cb_ec_scene     = cb_ec_scene,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

_C = {
    'eye':    '#2166ac',
    'head':   '#555555',
    'target': '#d6604d',
    'vs':     '#762a83',
    'ni':     '#4dac26',
    'burst':  '#f4a582',
    'pursuit':'#1a9850',
    'error':  '#9970ab',
    'ref':    '#d73027',
    'zero':   '#aaaaaa',
}

def _add_visual_shading(ax, t, sp, tp):
    """Gray tint for dark periods; no shading when scene is always on (avoids clutter)."""
    if not (sp < 0.5).any():
        return
    in_seg = False
    for i in range(len(t)):
        if sp[i] < 0.5 and not in_seg:
            t0, in_seg = t[i], True
        elif sp[i] >= 0.5 and in_seg:
            ax.axvspan(t0, t[i], color='#333333', alpha=0.10, lw=0, zorder=0)
            in_seg = False
    if in_seg:
        ax.axvspan(t0, t[-1], color='#333333', alpha=0.10, lw=0, zorder=0)


_PANEL_LABELS = {
    'eye_position':      'Eye position (deg)',
    'eye_velocity':      'Eye velocity (deg/s)',
    'head_velocity':     'Head velocity (deg/s)',
    'gaze_error':        'Gaze error (deg)',
    'retinal_error':     'Retinal position error (deg)',
    'canal_afferents':   'Velocity storage (deg/s)',
    'velocity_storage':  'Velocity storage (deg/s)',
    'neural_integrator': 'Neural integrator (deg)',
    'saccade_burst':     'Saccade burst (deg/s)',
    'pursuit_drive':     'Pursuit integrator (deg/s)',
    'refractory':        'Accumulator / trigger IBN',
    'vergence':          'Vergence angle (deg)',
    # cerebellar diagnostic panels
    'cerebellum_pursuit': 'Cerebellum — pursuit (deg/s)',
    'cerebellum_vor':     'Cerebellum — VOR/OKR (deg/s)',
    # stimulus panels
    'target_position':   'Target position (deg)',
    'target_velocity':   'Target velocity (deg/s)',
    'scene_velocity':    'Scene velocity (deg/s)',
    'visual_flags':      'Visual context',
}


def _draw_panel(ax, panel_name: str, t: np.ndarray, sig: dict,
                stim_kw: dict, scenario: SimulationScenario):
    """Draw one signal panel onto ax."""
    ax.set_ylabel(_PANEL_LABELS.get(panel_name, panel_name), fontsize=8)
    ax.tick_params(labelsize=7)

    ep  = sig['eye_pos']
    ev  = sig['eye_vel']
    ep_d = sig['e_pos_delayed']

    # Stimulus arrays (always available)
    hv = np.array(stim_kw['head_vel_array'])           # (T, 3) deg/s
    pt = np.array(stim_kw['p_target_array'])           # (T, 3) Cartesian
    vt = np.array(stim_kw['v_target_array'])           # (T, 3) deg/s
    vs = np.array(stim_kw['v_scene_array'])            # (T, 3) deg/s
    sp  = np.maximum(np.array(stim_kw['scene_present_L_array']),
                     np.array(stim_kw['scene_present_R_array']))   # (T,)
    tpL = np.array(stim_kw['target_present_L_array'])              # (T,)
    tpR = np.array(stim_kw['target_present_R_array'])              # (T,)

    target_yaw_deg = np.degrees(np.arctan(pt[:, 0]))

    tp_combined = np.maximum(tpL, tpR)

    # Visual-flags panels don't need a zero line; all others do
    if panel_name != 'visual_flags':
        ax.axhline(0, color=_C['zero'], lw=0.5, ls='--')
        _add_visual_shading(ax, t, sp, tp_combined)

    if panel_name == 'eye_position':
        ep_L = sig['eye_pos_L']
        ep_R = sig['eye_pos_R']
        bino_spread = np.max(np.abs(ep_L[:, 0] - ep_R[:, 0]))

        # Show gaze in world frame only when head actually moves (displacement > 2 deg).
        dt_val = t[1] - t[0] if len(t) > 1 else 0.001
        head_angle = np.cumsum(hv[:, 0]) * dt_val   # integrated head yaw (deg)
        head_moves = np.max(np.abs(head_angle)) > 2.0

        if bino_spread > 0.5:
            lbl_L = 'L eye (head)' if head_moves else 'L eye'
            lbl_R = 'R eye (head)' if head_moves else 'R eye'
            ax.plot(t, ep_L[:, 0], color='#2166ac', lw=1.2, label=lbl_L)
            ax.plot(t, ep_R[:, 0], color='#d6604d', lw=1.2, label=lbl_R)
        else:
            lbl_eye = 'Eye (head frame)' if head_moves else 'Eye position'
            ax.plot(t, ep[:, 0], color=_C['eye'], lw=1.2, label=lbl_eye)

        if head_moves:
            gaze = ep[:, 0] + head_angle
            ax.plot(t, gaze, color=_C['head'], lw=1.2, ls='--', label='Gaze (world)')

        # Target: solid when visible, dashed+faded when absent
        ax.plot(t, np.where(tp_combined > 0.5, target_yaw_deg, np.nan),
                color=_C['target'], lw=1.2, ls='-', label='Target (visible)')
        if (tp_combined < 0.5).any():
            ax.plot(t, np.where(tp_combined < 0.5, target_yaw_deg, np.nan),
                    color=_C['target'], lw=0.8, ls='--', alpha=0.4, label='Target (absent)')
        ax.legend(fontsize=6, loc='upper right')
        ax.set_ylabel('Eye / target position (deg)', fontsize=8)

    elif panel_name == 'eye_velocity':
        ax.plot(t, ev[:, 0],  color=_C['eye'],  lw=1.2, label='Eye vel')
        ax.plot(t, hv[:, 0],  color=_C['head'], lw=1.0, ls=':', label='Head vel')
        # Scene velocity as reference when OKR is relevant
        if np.any(np.abs(vs[:, 0]) > 0.5):
            ax.plot(t, vs[:, 0], color='#8c510a', lw=0.9, ls='--', alpha=0.7, label='Scene vel')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'head_velocity':
        ax.plot(t, hv[:, 0], color=_C['head'], lw=1.2, label='Head velocity (yaw)')
        if hv.shape[1] > 1 and np.any(hv[:, 1] != 0):
            ax.plot(t, hv[:, 1], color=_C['burst'], lw=1.0, ls='--', label='pitch')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'gaze_error':
        gaze = ep[:, 0] + np.cumsum(hv[:, 0]) * (t[1] - t[0])
        ax.plot(t, gaze, color=_C['error'], lw=1.2, label='Gaze error')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'retinal_error':
        ax.plot(t, ep_d[:, 0], color=_C['error'], lw=1.2, label='Retinal position error (yaw)')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'velocity_storage':
        ax.plot(t, sig['w_est'][:, 0], color=_C['vs'], lw=1.2, label='Velocity storage (yaw)')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'neural_integrator':
        ax.plot(t, sig['x_ni'][:, 0], color=_C['ni'], lw=1.2, label='Neural integrator (yaw)')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'saccade_burst':
        ax.plot(t, sig['u_burst'][:, 0], color=_C['burst'], lw=1.2, label='Saccade burst (yaw)')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'pursuit_drive':
        ax.plot(t, sig['x_pursuit'][:, 0], color=_C['pursuit'], lw=1.2, label='Pursuit integrator')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'refractory':
        ax.plot(t, sig['z_acc'],  color=_C['ref'],   lw=1.2, label='z_acc (accumulator)')
        ax.plot(t, sig['z_trig'], color='#d62728',   lw=0.9, ls='--', label='z_trig (trigger IBN)')
        ax.axhline(1.0, color='k', lw=0.6, ls='--', alpha=0.4, label='threshold')
        ax.axhline(0.0, color='k', lw=0.4, alpha=0.2)
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'vergence':
        ax.plot(t, sig['vergence'][:, 0],  color='#1b7837', lw=1.5, label='Vergence angle')
        ax.plot(t, sig['x_verg'][:, 0],    color='#762a83', lw=1.0, ls=':', label='Vergence integrator')
        tonic_val = scenario.patient.tonic_verg
        if abs(tonic_val) > 0.1:
            ax.axhline(tonic_val, color='#ff8c00', lw=1.0, ls='--',
                       label=f'Tonic verg ({tonic_val:+.1f}°)')
        ax.legend(fontsize=6, loc='upper right')
        ax.set_ylabel('Vergence angle (deg)', fontsize=8)

    # ── Cerebellar diagnostic panels ──────────────────────────────────────────

    elif panel_name == 'cerebellum_pursuit':
        # VPF pursuit EC correction + pursuit saccadic-suppression gate (right axis)
        ax.plot(t, sig['cb_vpf_drive'][:, 0], color='#1f78b4', lw=1.3, label='VPF EC drive (vpf_drive)')
        ax.plot(t, sig['cb_pred_err'][:, 0],  color='#7b3294', lw=0.9, ls=':', label='pred_err (slip+EC)')
        ax.legend(fontsize=6, loc='upper left')
        ax2 = ax.twinx()
        ax2.plot(t, sig['cb_sat_target'], color='#999999', lw=1.1, ls='--', alpha=0.9)
        ax2.set_ylim(-0.05, 1.10)
        ax2.set_ylabel('sacc. supp. gate', fontsize=7, color='#777777')
        ax2.tick_params(labelsize=6, colors='#777777')
        ax.set_ylabel('Cerebellum — pursuit (deg/s)', fontsize=8)

    elif panel_name == 'cerebellum_vor':
        ax.plot(t, sig['cb_fl_drive'][:, 0],     color='#33a02c', lw=1.3, label='FL NI leak-cancel (fl_drive)')
        ax.plot(t, sig['cb_fl_okr_drive'][:, 0], color='#1f78b4', lw=1.1, ls='-.', label='FL OKR EC (fl_okr_drive)')
        ax.plot(t, sig['cb_nu_drive'][:, 0],     color='#e31a1c', lw=1.0, ls=':',  label='NU gravity dump (nu_drive)')
        if np.any(np.abs(sig['cb_fl_vs_drive'][:, 0]) > 1e-3):
            ax.plot(t, sig['cb_fl_vs_drive'][:, 0], color='#6a3d9a', lw=0.9, ls='--', label='FL VS leak-cancel')
        ax.legend(fontsize=6, loc='upper left')
        ax2 = ax.twinx()
        ax2.plot(t, sig['cb_sat_scene'], color='#999999', lw=1.1, ls='--', alpha=0.9)
        ax2.set_ylim(-0.05, 1.10)
        ax2.set_ylabel('sacc. supp. gate', fontsize=7, color='#777777')
        ax2.tick_params(labelsize=6, colors='#777777')
        ax.set_ylabel('Cerebellum — VOR/OKR (deg/s)', fontsize=8)

    # ── Stimulus panels ───────────────────────────────────────────────────────

    elif panel_name == 'target_position':
        ax.plot(t, target_yaw_deg, color=_C['target'], lw=1.2, label='Target yaw')
        tp_pitch = np.degrees(np.arctan(pt[:, 1]))
        if np.any(np.abs(tp_pitch) > 0.5):
            ax.plot(t, tp_pitch, color=_C['burst'], lw=1.0, ls='--', label='Target pitch')
        ax.legend(fontsize=6, loc='upper right')
        ax.set_ylabel('Target position (deg)', fontsize=8)

    elif panel_name == 'target_velocity':
        ax.plot(t, vt[:, 0], color=_C['target'], lw=1.2, label='Target vel yaw')
        if np.any(np.abs(vt[:, 1]) > 0.5):
            ax.plot(t, vt[:, 1], color=_C['burst'], lw=1.0, ls='--', label='Target vel pitch')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'scene_velocity':
        ax.plot(t, vs[:, 0], color='#8c510a', lw=1.2, label='Scene vel yaw')
        if np.any(np.abs(vs[:, 1]) > 0.5):
            ax.plot(t, vs[:, 1], color='#bf812d', lw=1.0, ls='--', label='Scene vel pitch')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'visual_flags':
        # Gantt-style timeline: one labelled lane per visual context channel.
        # Filled bar = "on" (lit / target present), light-hatched bar = "off"
        # (dark / target absent), with the state written in the bar and the
        # transition times tick-marked on the time axis.
        bino = np.any(tpL != tpR)
        ax.set_xlim(t[0], t[-1])
        ax.set_facecolor('white')

        # Build lane list: (label, flag_array, on_label, off_label, color_on)
        lanes = [('Scene', sp, 'LIT', 'DARK', '#4dac26')]
        if not bino:
            lanes.append(('Target', tp_combined, 'TARGET ON', 'no target', '#e08214'))
        else:
            lanes.append(('L eye target', tpL, 'ON', 'covered', '#2166ac'))
            lanes.append(('R eye target', tpR, 'ON', 'covered', '#d6604d'))

        n_lanes = len(lanes)
        ax.set_ylim(0, n_lanes)
        ax.set_yticks([n_lanes - 0.5 - i for i in range(n_lanes)])
        ax.set_yticklabels([lbl for lbl, *_ in lanes], fontsize=7)
        ax.tick_params(axis='y', length=0)
        ax.set_ylabel('Visual context', fontsize=8)

        def _segments(t, flag):
            """Yield (t0, t1, state) runs of a binary flag array."""
            i0 = 0
            cur = flag[0] > 0.5
            for i in range(1, len(flag)):
                v = flag[i] > 0.5
                if v != cur:
                    yield (t[i0], t[i], cur)
                    i0, cur = i, v
            yield (t[i0], t[-1], cur)

        all_transitions = set()
        for lane_i, (lbl, flag, on_lbl, off_lbl, c_on) in enumerate(lanes):
            y_bot = n_lanes - 1 - lane_i + 0.15
            y_top = n_lanes - 1 - lane_i + 0.85
            h     = y_top - y_bot
            for (t0, t1, state) in _segments(t, flag):
                if t1 - t0 <= 0:
                    continue
                if state:
                    ax.add_patch(plt.Rectangle((t0, y_bot), t1 - t0, h,
                                                facecolor=c_on, alpha=0.55,
                                                edgecolor=c_on, lw=0.8))
                else:
                    ax.add_patch(plt.Rectangle((t0, y_bot), t1 - t0, h,
                                                facecolor='#dddddd', alpha=0.45,
                                                edgecolor='#bbbbbb', lw=0.5, hatch='///'))
                # Label the run if it's wide enough to fit text
                if (t1 - t0) > 0.04 * (t[-1] - t[0]):
                    ax.text((t0 + t1) / 2, (y_bot + y_top) / 2,
                            on_lbl if state else off_lbl,
                            ha='center', va='center', fontsize=6,
                            color='#1a1a1a' if state else '#666666')
                if t0 > t[0] + 1e-9:
                    all_transitions.add(round(float(t0), 4))

        # Mark transition times on the x-axis
        for tt in sorted(all_transitions):
            ax.axvline(tt, color='#444444', lw=0.7, ls=':', alpha=0.6)

        # Scene velocity overlay on right y-axis (when non-zero)
        if np.any(np.abs(vs[:, 0]) > 0.5):
            ax2 = ax.twinx()
            ax2.step(t, vs[:, 0], where='post', color='#8c510a', lw=1.3, alpha=0.85,
                     label='scene vel')
            ax2.axhline(0, color='#8c510a', lw=0.4, alpha=0.5)
            vs_max = max(np.max(np.abs(vs[:, 0])), 1.0)
            ax2.set_ylim(-vs_max * 1.4, vs_max * 1.4)
            ax2.set_ylabel('Scene vel (°/s)', fontsize=7, color='#8c510a')
            ax2.tick_params(labelsize=6, colors='#8c510a')

    # Enforce a minimum visible range on velocity / derivative panels
    if panel_name in ('eye_velocity', 'head_velocity', 'saccade_burst', 'pursuit_drive',
                      'target_velocity', 'scene_velocity',
                      'cerebellum_pursuit', 'cerebellum_vor'):
        lo, hi = ax.get_ylim()
        span = hi - lo
        if span < 5.0:
            mid = (lo + hi) / 2
            ax.set_ylim(mid - 5.0, mid + 5.0)


def _build_figure(
    t: np.ndarray,
    sig: dict,
    stim_kw: dict,
    scenario: SimulationScenario,
) -> plt.Figure:
    """Assemble the multi-panel figure."""
    panels = scenario.plot.panels
    n = len(panels)
    fig = plt.figure(figsize=(10, 2.2 * n))
    gs  = gridspec.GridSpec(n, 1, hspace=0.45)

    title = scenario.plot.title or scenario.description
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.0)

    axes = [fig.add_subplot(gs[i]) for i in range(n)]
    for ax, panel in zip(axes, panels):
        _draw_panel(ax, panel, t, sig, stim_kw, scenario)
        if ax is not axes[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)', fontsize=8)

    return fig


# ── Simulation data export ────────────────────────────────────────────────────

def _build_sim_data(t_array: np.ndarray, sig: dict, stim_kw: dict) -> dict:
    """Build a flat dict of simulation arrays suitable for CSV download.

    Arrays are plain numpy, shape (T, 3) for 3-D channels or (T,) for scalars.
    All angular quantities in deg or deg/s.
    """
    return dict(
        t               = np.array(t_array),
        eye_pos         = sig['eye_pos'],                              # (T, 3) deg — conjugate version
        eye_pos_L       = sig['eye_pos_L'],                           # (T, 3) deg — left eye
        eye_pos_R       = sig['eye_pos_R'],                           # (T, 3) deg — right eye
        eye_vel         = sig['eye_vel'],                              # (T, 3) deg/s
        head_vel        = np.array(stim_kw['head_vel_array']),        # (T, 3) deg/s
        scene_vel       = np.array(stim_kw['v_scene_array']),         # (T, 3) deg/s
        target_vel      = np.array(stim_kw['v_target_array']),        # (T, 3) deg/s
        scene_present_L  = np.array(stim_kw['scene_present_L_array']),  # (T,)
        scene_present_R  = np.array(stim_kw['scene_present_R_array']),  # (T,)
        target_present_L = np.array(stim_kw['target_present_L_array']), # (T,)
        target_present_R = np.array(stim_kw['target_present_R_array']), # (T,)
    )


# ── Public API ────────────────────────────────────────────────────────────────

def run_scenario(scenario: SimulationScenario, output_path: str | None = None,
                 return_data: bool = False) -> plt.Figure | tuple[plt.Figure, dict]:
    """Run a SimulationScenario end-to-end and return a matplotlib Figure.

    Args:
        scenario:    Fully populated SimulationScenario object.
        output_path: If given, save figure to this path (PNG/SVG/PDF).
        return_data: If True, return (fig, sim_data_dict) instead of just fig.
                     sim_data_dict keys: t, eye_pos, eye_vel, head_vel,
                     scene_vel, target_vel — all numpy arrays.

    Returns:
        fig                      when return_data=False (default)
        (fig, sim_data_dict)     when return_data=True
    """
    # Build stimulus and params
    stim_kw = _build_stimulus(scenario)
    t_array = stim_kw.pop('t_array')

    # Extract trajectory objects for simulate(); keep flat arrays for plotting
    head_km   = stim_kw.pop('_head_km')
    scene_km  = stim_kw.pop('_scene_km')
    target_tt = stim_kw.pop('_target_tt')
    # Strip any remaining _ keys
    for k in list(stim_kw):
        if k.startswith('_'):
            stim_kw.pop(k)

    params    = _build_params(scenario.patient)
    max_steps = int(len(t_array) * 1.5) + 2000

    # Run simulation
    states = simulate(
        params, t_array,
        head=head_km, scene=scene_km, target=target_tt,
        scene_present_L_array=stim_kw['scene_present_L_array'],
        scene_present_R_array=stim_kw['scene_present_R_array'],
        target_present_L_array=stim_kw['target_present_L_array'],
        target_present_R_array=stim_kw['target_present_R_array'],
        target_strobed_array=stim_kw['target_strobed_array'],
        return_states=True,
        max_steps=max_steps,
    )

    # Extract signals
    sig = _extract_signals(states, params, t_array)

    # Build figure
    fig = _build_figure(t_array, sig, stim_kw, scenario)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {output_path}")

    if return_data:
        return fig, _build_sim_data(t_array, sig, stim_kw)
    return fig


# ── Comparison ────────────────────────────────────────────────────────────────

# Distinct colors for up to 4 compared conditions
_COMPARE_COLORS = ['#2166ac', '#d6604d', '#1a9850', '#762a83']
_COMPARE_STYLES = ['-', '--', '-.', ':']


def _build_comparison_figure(
    results: list[tuple[np.ndarray, dict, dict]],  # [(t, sig, stim_kw), ...]
    comparison: SimulationComparison,
) -> plt.Figure:
    """Overlay N simulation results on the same set of panels."""
    panels = comparison.panels
    n = len(panels)
    labels = [s.description for s in comparison.scenarios]

    fig = plt.figure(figsize=(10, 2.2 * n))
    gs  = gridspec.GridSpec(n, 1, hspace=0.45)
    fig.suptitle(comparison.title, fontsize=11, fontweight='bold', y=1.0)

    axes = [fig.add_subplot(gs[i]) for i in range(n)]

    for ax, panel in zip(axes, panels):
        ax.set_ylabel(_PANEL_LABELS.get(panel, panel), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color=_C['zero'], lw=0.5, ls='--')

        for idx, (t, sig, stim_kw) in enumerate(results):
            color = _COMPARE_COLORS[idx % len(_COMPARE_COLORS)]
            ls    = _COMPARE_STYLES[idx % len(_COMPARE_STYLES)]
            label = labels[idx]

            ep  = sig['eye_pos']
            ev  = sig['eye_vel']
            hv  = np.array(stim_kw['head_vel_array'])
            pt  = np.array(stim_kw['p_target_array'])
            target_yaw = np.degrees(np.arctan(pt[:, 0]))

            if panel == 'eye_position':
                dt_val = t[1] - t[0] if len(t) > 1 else 0.001
                head_angle = np.cumsum(hv[:, 0]) * dt_val
                head_moves = np.max(np.abs(head_angle)) > 2.0
                ax.plot(t, ep[:, 0], color=color, ls=ls, lw=1.5,
                        label=f'{label} (head)' if head_moves else label)
                if head_moves:
                    ax.plot(t, ep[:, 0] + head_angle, color=color, ls=ls, lw=0.9,
                            alpha=0.55, label=f'{label} (world)')
                if idx == 0:
                    ax.plot(t, target_yaw, color=_C['target'], lw=1.0, ls=':', label='Target')
                ax.set_ylabel('Eye / target position (deg)', fontsize=8)

            elif panel == 'eye_velocity':
                ax.plot(t, ev[:, 0], color=color, ls=ls, lw=1.5, label=label)
                if idx == 0:
                    ax.plot(t, hv[:, 0], color=_C['head'], lw=1.0, ls=':', label='Head vel')

            elif panel == 'head_velocity':
                if idx == 0:  # head motion is shared — draw once
                    ax.plot(t, hv[:, 0], color=_C['head'], lw=1.5, label='Head vel')

            elif panel == 'velocity_storage':
                ax.plot(t, sig['w_est'][:, 0], color=color, ls=ls, lw=1.5, label=label)

            elif panel == 'neural_integrator':
                ax.plot(t, sig['x_ni'][:, 0], color=color, ls=ls, lw=1.5, label=label)

            elif panel == 'saccade_burst':
                ax.plot(t, sig['u_burst'][:, 0], color=color, ls=ls, lw=1.5, label=label)

            elif panel == 'pursuit_drive':
                ax.plot(t, sig['x_pursuit'][:, 0], color=color, ls=ls, lw=1.5, label=label)

            elif panel == 'refractory':
                ax.plot(t, sig['z_acc'], color=color, ls=ls, lw=1.5, label=label)

            elif panel == 'vergence':
                ax.plot(t, sig['vergence'][:, 0], color=color, ls=ls, lw=1.5, label=label)

            elif panel == 'gaze_error':
                dt = t[1] - t[0]
                gaze = ep[:, 0] + np.cumsum(hv[:, 0]) * dt
                ax.plot(t, gaze, color=color, ls=ls, lw=1.5, label=label)

            elif panel == 'retinal_error':
                ax.plot(t, sig['e_pos_delayed'][:, 0], color=color, ls=ls, lw=1.5, label=label)

        ax.legend(fontsize=6, loc='upper right')

        # Min y-range for velocity panels
        if panel in ('eye_velocity', 'head_velocity', 'saccade_burst', 'pursuit_drive'):
            lo, hi = ax.get_ylim()
            if hi - lo < 5.0:
                mid = (lo + hi) / 2
                ax.set_ylim(mid - 5.0, mid + 5.0)

        if ax is not axes[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)', fontsize=8)

    return fig


def run_comparison(
    comparison: SimulationComparison,
    output_path: str | None = None,
    return_data: bool = False,
) -> 'plt.Figure | tuple[plt.Figure, list[dict]]':
    """Run all scenarios in a SimulationComparison and overlay them on one figure.

    Args:
        comparison:  Fully populated SimulationComparison object.
        output_path: If given, save figure to this path.
        return_data: If True, return (fig, sim_data_list) where sim_data_list is
                     one dict per scenario (same format as run_scenario return_data).

    Returns:
        fig                       when return_data=False (default)
        (fig, sim_data_list)      when return_data=True
    """
    results       = []
    sim_data_list = []
    for scenario in comparison.scenarios:
        stim_kw = _build_stimulus(scenario)
        t_array = stim_kw.pop('t_array')

        head_km   = stim_kw.pop('_head_km')
        scene_km  = stim_kw.pop('_scene_km')
        target_tt = stim_kw.pop('_target_tt')
        for k in list(stim_kw):
            if k.startswith('_'):
                stim_kw.pop(k)

        params    = _build_params(scenario.patient)
        max_steps = int(len(t_array) * 1.5) + 2000
        states = simulate(
            params, t_array,
            head=head_km, scene=scene_km, target=target_tt,
            scene_present_L_array=stim_kw['scene_present_L_array'],
            scene_present_R_array=stim_kw['scene_present_R_array'],
            target_present_L_array=stim_kw['target_present_L_array'],
            target_present_R_array=stim_kw['target_present_R_array'],
            target_strobed_array=stim_kw['target_strobed_array'],
            return_states=True,
            max_steps=max_steps,
        )
        sig = _extract_signals(states, params, t_array)
        results.append((t_array, sig, stim_kw))
        if return_data:
            sim_data_list.append(_build_sim_data(t_array, sig, stim_kw))
        print(f"  ✓ {scenario.description}")

    fig = _build_comparison_figure(results, comparison)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {output_path}")

    if return_data:
        return fig, sim_data_list
    return fig
