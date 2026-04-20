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
from oculomotor.llm_pipeline.scenario import SimulationScenario, SimulationComparison, Patient
from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, with_sensory, simulate,
    _IDX_VS, _IDX_VS_L, _IDX_VS_R, _IDX_NI, _IDX_SG, _IDX_EC, _IDX_VIS, _IDX_VIS_L, _IDX_VIS_R, _IDX_PURSUIT,
    _IDX_VERG,
)
from oculomotor.models.brain_models import saccade_generator as sg_mod
from oculomotor.models.sensory_models.sensory_model import C_slip, C_pos, C_vel, C_target_visible


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


def _project_target_to_retina(
    target_arrays: dict,
    head_arrays: dict,
    T: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3-D world-frame target position to retinal stereographic coordinates.

    Both arrays are in metres (world frame).  The eye is assumed to be at the
    head origin for now; this function is the single place to add inter-ocular
    baseline offsets when binocular support is added.

    Returns
    -------
    p_target : (T, 3) float32
        Stereographic target position [tan(yaw), tan(pitch), 1].
    v_target : (T, 3) float32
        Target angular velocity as seen from the head (deg/s).
        Accounts for head translation: d/dt(arctan(rel_x / rel_z)).
    """
    target_pos = target_arrays['lin_pos']          # (T, 3) m
    target_vel = target_arrays['lin_vel']          # (T, 3) m/s
    head_pos   = head_arrays['lin_pos']            # (T, 3) m
    head_vel   = head_arrays['lin_vel']            # (T, 3) m/s

    # Vector from head/eye origin to target in world frame
    rel_pos = target_pos - head_pos                # (T, 3) m
    rel_vel = target_vel - head_vel                # (T, 3) m/s

    # Depth (z), clipped to a physiological minimum (5 cm)
    depth = np.maximum(rel_pos[:, 2], 0.05)       # (T,)

    # Stereographic projection  p = [x/z, y/z, 1]
    p_target = np.zeros((T, 3), dtype=np.float32)
    p_target[:, 0] = (rel_pos[:, 0] / depth).astype(np.float32)
    p_target[:, 1] = (rel_pos[:, 1] / depth).astype(np.float32)
    p_target[:, 2] = 1.0

    # Angular velocity of target as seen from head (deg/s)
    # d/dt(arctan(x/z)) = (vx·z − x·vz) / (z² + x²)  [rad/s] × (180/π)
    denom_x = depth ** 2 + rel_pos[:, 0] ** 2
    denom_y = depth ** 2 + rel_pos[:, 1] ** 2
    v_target = np.zeros((T, 3), dtype=np.float32)
    v_target[:, 0] = (
        (rel_vel[:, 0] * depth - rel_pos[:, 0] * rel_vel[:, 2]) / denom_x * (180.0 / np.pi)
    ).astype(np.float32)
    v_target[:, 1] = (
        (rel_vel[:, 1] * depth - rel_pos[:, 1] * rel_vel[:, 2]) / denom_y * (180.0 / np.pi)
    ).astype(np.float32)

    return p_target, v_target


def _build_stimulus(scenario: SimulationScenario) -> dict:
    """Convert BodySegment lists to simulator keyword arrays.

    Returns a dict for ``**stim_kwargs`` in ``simulate()``, plus 't_array'.
    All three body channels are integrated to consistent 6-DOF trajectories;
    the target is projected to retinal coordinates via full 3-D geometry.
    """
    dt  = 0.001
    T   = int(round(scenario.duration_s / dt))
    t   = np.arange(T, dtype=np.float32) * dt

    # ── Build 6-DOF arrays for each body ──────────────────────────────────────
    head_arr   = stim.build_body_arrays(scenario.head,  T, dt, default_lin_pos=(0.0, 0.0, 0.0))
    scene_arr  = stim.build_body_arrays(scenario.scene, T, dt, default_lin_pos=(0.0, 0.0, 5.0))

    # Resolve angular shorthand on target before building
    target_segs_resolved = _resolve_target_angular_shorthand(scenario.target)
    target_arr = stim.build_body_arrays(target_segs_resolved, T, dt, default_lin_pos=(0.0, 0.0, 1.0))

    # ── Simulator inputs ───────────────────────────────────────────────────────

    # Head angular velocity → semicircular canals
    head_vel = head_arr['rot_vel']                 # (T, 3) deg/s

    # Target 3-D projection → retinal position + angular velocity
    p_target, v_target = _project_target_to_retina(target_arr, head_arr, T)

    # Scene angular velocity → OKR / velocity storage
    v_scene = scene_arr['rot_vel']                 # (T, 3) deg/s

    # Visual flags — per-eye scene_present and target_present
    spL, spR, tpL, tpR = stim.build_visual_flags(scenario.visual, T, dt)

    # Head linear acceleration in world frame → simulator adds gravity rotation to get
    # specific force in head frame (already done inside ODE_ocular_motor)
    head_lin_acc = head_arr['lin_acc']   # (T, 3) m/s²

    return dict(
        t_array                 = t,
        head_vel_array          = jnp.array(head_vel),
        head_accel_array        = jnp.array(head_lin_acc),
        p_target_array          = jnp.array(p_target),
        v_target_array          = jnp.array(v_target),
        v_scene_array           = jnp.array(v_scene),
        scene_present_L_array   = jnp.array(spL),
        scene_present_R_array   = jnp.array(spR),
        target_present_L_array  = jnp.array(tpL),
        target_present_R_array  = jnp.array(tpR),
        # 6-DOF arrays stored for plotting (stripped before passing to ODE)
        _head_lin_pos           = head_arr['lin_pos'],
        _head_lin_vel           = head_arr['lin_vel'],
        _target_world_pos       = target_arr['lin_pos'],
    )


# ── Parameter builder ─────────────────────────────────────────────────────────

def _build_params(patient: Patient):
    """Convert Patient overrides to a Params NamedTuple."""
    params = PARAMS_DEFAULT

    # Sensory overrides
    params = with_sensory(params, canal_gains=jnp.array(patient.canal_gains, dtype=float))

    # b_vs_L/R → 6-vector (model LEFT pop = anatomical RIGHT VN, model RIGHT pop = anatomical LEFT VN)
    b_vs = jnp.array([
        patient.b_vs_R, patient.b_vs_R, patient.b_vs_R,   # model LEFT pop = anatomical RIGHT VN
        patient.b_vs_L, patient.b_vs_L, patient.b_vs_L,   # model RIGHT pop = anatomical LEFT  VN
    ], dtype=float)

    # Brain overrides
    params = with_brain(params,
        b_vs            = b_vs,
        tau_vs          = patient.tau_vs,
        K_vs            = patient.K_vs,
        K_vis           = patient.K_vis,
        g_vis           = patient.g_vis,
        tau_i           = patient.tau_i,
        g_burst         = patient.g_burst,
        K_pursuit       = patient.K_pursuit,
        K_phasic_pursuit= patient.K_phasic_pursuit,
        tau_pursuit     = patient.tau_pursuit,
        K_grav          = patient.K_grav,
        tau_vs_adapt    = patient.tau_vs_adapt,
        tau_ni_adapt    = patient.tau_ni_adapt,
        K_verg          = patient.K_verg,
        K_phasic_verg   = patient.K_phasic_verg,
        tau_verg        = patient.tau_verg,
        phoria          = jnp.array(patient.phoria, dtype=float),
    )
    return params


# ── Signal extraction helpers ─────────────────────────────────────────────────

def _extract_signals(states, params, t_np: np.ndarray) -> dict:
    """Extract named signals from a SimState trajectory."""
    dt = t_np[1] - t_np[0]

    eye_pos_L = np.array(states.plant[:, :3])      # (T, 3) left  eye rotation (deg)
    eye_pos_R = np.array(states.plant[:, 3:])       # (T, 3) right eye rotation (deg)
    version   = 0.5 * (eye_pos_L + eye_pos_R)       # (T, 3) conjugate (version)
    vergence  = eye_pos_L - eye_pos_R               # (T, 3) vergence angle (deg, + = converged)

    x_vs_raw  = np.array(states.brain[:, _IDX_VS])           # (T, 9) x_L + x_R + x_null
    x_ni_raw  = np.array(states.brain[:, _IDX_NI])           # (T, 9) x_L + x_R + x_null
    x_sg      = np.array(states.brain[:, _IDX_SG])
    x_pursuit = np.array(states.brain[:, _IDX_PURSUIT])
    x_verg    = np.array(states.brain[:, _IDX_VERG])   # (T, 3) vergence integrator state
    x_vis_L   = np.array(states.sensory[:, _IDX_VIS_L])
    x_vis_R   = np.array(states.sensory[:, _IDX_VIS_R])

    # Eye velocity (version derivative — same as L eye vel when version ≈ L)
    w_eye = np.gradient(version, dt, axis=0)

    # VS and NI nets: x_L − x_R (x_null at [6:9] excluded)
    w_est = x_vs_raw[:, :3] - x_vs_raw[:, 3:6]   # VS net  (T, 3)
    x_ni  = x_ni_raw[:, :3] - x_ni_raw[:, 3:6]   # NI net  (T, 3)

    # Retinal signals — gate-weighted average consistent with sensory_model fix
    gate_L = x_vis_L @ np.array(C_target_visible).T          # (T, 1)
    gate_R = x_vis_R @ np.array(C_target_visible).T          # (T, 1)
    gate_sum = gate_L + gate_R + 1e-6               # (T, 1)
    pos_L  = x_vis_L @ np.array(C_pos).T            # (T, 3)
    pos_R  = x_vis_R @ np.array(C_pos).T            # (T, 3)
    e_pos_delayed = (gate_L * pos_L + gate_R * pos_R) / gate_sum  # (T, 3)

    # Saccade burst (re-compute from SG state + weighted delayed retinal signals)
    def _burst_at(state):
        x_vis_L_ = state.sensory[_IDX_VIS_L]
        x_vis_R_ = state.sensory[_IDX_VIS_R]
        gL = (C_target_visible @ x_vis_L_)[0]
        gR = (C_target_visible @ x_vis_R_)[0]
        norm = jnp.maximum(gL + gR, 1e-6)
        e_pd = (gL * (C_pos @ x_vis_L_) + gR * (C_pos @ x_vis_R_)) / norm
        gate = jnp.clip(gL + gR, 0.0, 1.0)
        x_ni_     = state.brain[_IDX_NI]
        x_ni_net  = x_ni_[:3] - x_ni_[3:6]   # bilateral → net (x_L − x_R), (3,)
        _, u  = sg_mod.step(state.brain[_IDX_SG], e_pd, gate, x_ni_net, params.brain)
        return u
    u_burst = np.array(jax.vmap(_burst_at)(states))  # (T, 3)

    # SG sub-states
    x_copy = x_sg[:, :3]
    z_ref  = x_sg[:, 3]
    e_held = x_sg[:, 4:7]
    z_sac  = x_sg[:, 7]
    z_acc  = x_sg[:, 8]

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
        z_ref          = z_ref,
        z_sac          = z_sac,
        z_acc          = z_acc,
        e_held         = e_held,
        x_copy         = x_copy,
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
    'refractory':        'OPN / refractory state',
    'vergence':          'Vergence angle (deg)',
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
    sp = np.array(stim_kw['scene_present_array'])      # (T,)
    tpL = np.array(stim_kw['target_present_L_array'])  # (T,)
    tpR = np.array(stim_kw['target_present_R_array'])  # (T,)

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
        if bino_spread > 0.5:
            ax.plot(t, ep_L[:, 0], color='#2166ac', lw=1.2, label='L eye')
            ax.plot(t, ep_R[:, 0], color='#d6604d', lw=1.2, label='R eye')
        else:
            ax.plot(t, ep[:, 0], color=_C['eye'], lw=1.2, label='Eye yaw (version)')
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
        ax.plot(t, sig['z_ref'], color=_C['ref'], lw=1.2, label='OPN refractory state')
        ax.axhline(scenario.patient.g_burst * 0.0 + 0.5, color='k', lw=0.6, ls='--', alpha=0.4)
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'vergence':
        ax.plot(t, sig['vergence'][:, 0],  color='#1b7837', lw=1.5, label='Vergence angle')
        ax.plot(t, sig['x_verg'][:, 0],    color='#762a83', lw=1.0, ls=':', label='Vergence integrator')
        phoria_val = scenario.patient.phoria[0]
        if abs(phoria_val) > 0.1:
            ax.axhline(phoria_val, color='#ff8c00', lw=1.0, ls='--',
                       label=f'Phoria ({phoria_val:+.1f}°)')
        ax.legend(fontsize=6, loc='upper right')
        ax.set_ylabel('Vergence angle (deg)', fontsize=8)

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
        bino = np.any(tpL != tpR)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel('Visual context', fontsize=8)

        # Layout: bottom row = scene, top row = target (split L/R if cover test)
        S0, S1 = 0.04, 0.36      # scene row y-limits (data coords, ylim=[0,1])
        T0, T1 = 0.44, 0.96      # target row

        def _flag_spans(ax, t, flag, y0, y1, color_on, color_off, alpha_on=0.70, alpha_off=0.25):
            """Draw axvspan blocks for a binary flag array."""
            in_on = in_off = False
            t0_on = t0_off = t[0]
            for i in range(len(t)):
                val = flag[i] > 0.5
                if val and not in_on:
                    if in_off:
                        ax.axvspan(t0_off, t[i], ymin=y0, ymax=y1,
                                   color=color_off, alpha=alpha_off, lw=0)
                        in_off = False
                    t0_on, in_on = t[i], True
                elif not val and not in_off:
                    if in_on:
                        ax.axvspan(t0_on, t[i], ymin=y0, ymax=y1,
                                   color=color_on, alpha=alpha_on, lw=0)
                        in_on = False
                    t0_off, in_off = t[i], True
            if in_on:
                ax.axvspan(t0_on, t[-1], ymin=y0, ymax=y1,
                           color=color_on, alpha=alpha_on, lw=0)
            if in_off:
                ax.axvspan(t0_off, t[-1], ymin=y0, ymax=y1,
                           color=color_off, alpha=alpha_off, lw=0)

        # Scene row — green when lit, red when dark
        _flag_spans(ax, t, sp, S0, S1, '#4dac26', '#d73027', alpha_on=0.70, alpha_off=0.55)
        ax.text(0.01, (S0 + S1) / 2, 'Scene', va='center', ha='left',
                fontsize=7, color='#222222')

        # Target row
        if not bino:
            _flag_spans(ax, t, tp_combined, T0, T1, '#d6604d', '#aaaaaa',
                        alpha_on=0.75, alpha_off=0.30)
            ax.text(0.01, (T0 + T1) / 2, 'Target', va='center', ha='left',
                    fontsize=7, color='#222222')
        else:  # cover-test: split into L/R sub-rows
            mid = (T0 + T1) / 2 - 0.01
            for (pres, y0, y1, lbl) in [(tpL, mid + 0.02, T1, 'L eye'),
                                         (tpR, T0,        mid, 'R eye')]:
                _flag_spans(ax, t, pres, y0, y1, '#d6604d', '#aaaaaa',
                            alpha_on=0.75, alpha_off=0.30)
                ax.text(0.01, (y0 + y1) / 2, lbl, va='center', ha='left',
                        fontsize=7, color='#222222')

        # Scene velocity overlay on right y-axis (when non-zero)
        if np.any(np.abs(vs[:, 0]) > 0.5):
            ax2 = ax.twinx()
            ax2.step(t, vs[:, 0], where='post', color='#8c510a', lw=1.3, alpha=0.85)
            ax2.axhline(0, color='#8c510a', lw=0.4, alpha=0.5)
            vs_max = max(np.max(np.abs(vs[:, 0])), 1.0)
            ax2.set_ylim(-vs_max * 1.4, vs_max * 1.4)
            ax2.set_ylabel('Scene vel (°/s)', fontsize=7, color='#8c510a')
            ax2.tick_params(labelsize=6, colors='#8c510a')

    # Enforce a minimum visible range on velocity / derivative panels
    if panel_name in ('eye_velocity', 'head_velocity', 'saccade_burst', 'pursuit_drive',
                      'target_velocity', 'scene_velocity'):
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
    has_narrative = bool(scenario.narrative)
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02 if has_narrative else 1.0)

    if has_narrative:
        import textwrap
        wrapped = textwrap.fill(scenario.narrative, width=115)
        fig.text(0.5, 1.0, wrapped, ha='center', va='top',
                 fontsize=7.5, color='#444444', style='italic',
                 transform=fig.transFigure)

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
        t          = np.array(t_array),
        eye_pos    = sig['eye_pos'],                              # (T, 3) deg — conjugate version
        eye_pos_L  = sig['eye_pos_L'],                           # (T, 3) deg — left eye
        eye_pos_R  = sig['eye_pos_R'],                           # (T, 3) deg — right eye
        eye_vel    = sig['eye_vel'],                              # (T, 3) deg/s
        head_vel   = np.array(stim_kw['head_vel_array']),        # (T, 3) deg/s
        scene_vel  = np.array(stim_kw['v_scene_array']),         # (T, 3) deg/s
        target_vel = np.array(stim_kw['v_target_array']),        # (T, 3) deg/s
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
    # Strip private keys (6-DOF arrays stored for plotting but not passed to ODE)
    plot_extra = {k: stim_kw.pop(k) for k in list(stim_kw) if k.startswith('_')}
    params  = _build_params(scenario.patient)

    # max_steps must cover all ODE steps: duration_s / dt_solve = T
    # Add 50% headroom — diffrax may take more steps near discontinuities
    max_steps = int(len(t_array) * 1.5) + 2000

    # Run simulation
    states = simulate(
        params, t_array,
        return_states=True,
        max_steps=max_steps,
        **stim_kw,
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
    has_narrative = bool(comparison.narrative)
    fig.suptitle(comparison.title, fontsize=11, fontweight='bold', y=1.02 if has_narrative else 1.0)

    if has_narrative:
        import textwrap
        wrapped = textwrap.fill(comparison.narrative, width=115)
        fig.text(0.5, 1.0, wrapped, ha='center', va='top',
                 fontsize=7.5, color='#444444', style='italic',
                 transform=fig.transFigure)

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
                ax.plot(t, ep[:, 0], color=color, ls=ls, lw=1.5, label=label)
                if idx == 0:  # target is the same across conditions — draw once
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
                ax.plot(t, sig['z_ref'], color=color, ls=ls, lw=1.5, label=label)

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
) -> plt.Figure:
    """Run all scenarios in a SimulationComparison and overlay them on one figure.

    Args:
        comparison:  Fully populated SimulationComparison object.
        output_path: If given, save figure to this path.

    Returns:
        matplotlib.figure.Figure
    """
    results = []
    for scenario in comparison.scenarios:
        stim_kw = _build_stimulus(scenario)
        t_array = stim_kw.pop('t_array')
        # Strip private 6-DOF arrays before passing to ODE solver
        for k in list(stim_kw):
            if k.startswith('_'):
                stim_kw.pop(k)
        params  = _build_params(scenario.patient)
        max_steps = int(len(t_array) * 1.5) + 2000
        states = simulate(params, t_array, return_states=True,
                          max_steps=max_steps, **stim_kw)
        sig = _extract_signals(states, params, t_array)
        results.append((t_array, sig, stim_kw))
        print(f"  ✓ {scenario.description}")

    fig = _build_comparison_figure(results, comparison)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {output_path}")

    return fig
