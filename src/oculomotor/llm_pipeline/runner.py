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
from oculomotor.models.sensory_models.sensory_model import C_slip, C_pos, C_vel, C_gate


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

    # Visual flags — per-eye target_present (supports cover test)
    sp, tpL, tpR = stim.build_visual_flags(scenario.visual, T, dt)

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
        scene_present_array     = jnp.array(sp),
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

    # Brain overrides
    params = with_brain(params,
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

    x_vs      = np.array(states.brain[:, _IDX_VS])          # (T, 6) L+R populations
    x_ni      = np.array(states.brain[:, _IDX_NI])
    x_sg      = np.array(states.brain[:, _IDX_SG])
    x_pursuit = np.array(states.brain[:, _IDX_PURSUIT])
    x_verg    = np.array(states.brain[:, _IDX_VERG])   # (T, 3) vergence integrator state
    x_vis_L   = np.array(states.sensory[:, _IDX_VIS_L])
    x_vis_R   = np.array(states.sensory[:, _IDX_VIS_R])

    # Eye velocity (version derivative — same as L eye vel when version ≈ L)
    w_eye = np.gradient(version, dt, axis=0)

    # VS state ≈ head velocity estimate: net of bilateral populations
    w_est = x_vs[:, :3] - x_vs[:, 3:]   # x_L − x_R  →  (T, 3)

    # Retinal signals — gate-weighted average consistent with sensory_model fix
    gate_L = x_vis_L @ np.array(C_gate).T           # (T, 1)
    gate_R = x_vis_R @ np.array(C_gate).T           # (T, 1)
    gate_sum = gate_L + gate_R + 1e-6               # (T, 1)
    pos_L  = x_vis_L @ np.array(C_pos).T            # (T, 3)
    pos_R  = x_vis_R @ np.array(C_pos).T            # (T, 3)
    e_pos_delayed = (gate_L * pos_L + gate_R * pos_R) / gate_sum  # (T, 3)

    # Saccade burst (re-compute from SG state + weighted delayed retinal signals)
    def _burst_at(state):
        x_vis_L_ = state.sensory[_IDX_VIS_L]
        x_vis_R_ = state.sensory[_IDX_VIS_R]
        gL = (C_gate @ x_vis_L_)[0]
        gR = (C_gate @ x_vis_R_)[0]
        norm = jnp.maximum(gL + gR, 1e-6)
        e_pd = (gL * (C_pos @ x_vis_L_) + gR * (C_pos @ x_vis_R_)) / norm
        gate = jnp.clip(gL + gR, 0.0, 1.0)
        x_ni_ = state.brain[_IDX_NI]
        _, u  = sg_mod.step(state.brain[_IDX_SG], e_pd, gate, x_ni_, params.brain)
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

_PANEL_LABELS = {
    'eye_position':      'Eye position (deg)',
    'eye_velocity':      'Eye velocity (deg/s)',
    'head_velocity':     'Head velocity (deg/s)',
    'gaze_error':        'Gaze error (deg)',
    'retinal_error':     'Retinal error (deg)',
    'canal_afferents':   'VS state (deg/s equiv.)',
    'velocity_storage':  'Velocity storage x_vs',
    'neural_integrator': 'NI state x_ni',
    'saccade_burst':     'Burst u_burst (deg/s)',
    'pursuit_drive':     'Pursuit drive (deg/s)',
    'refractory':        'Refractory z_ref',
    'vergence':          'Vergence angle (deg)',
    # stimulus panels
    'target_position':   'Target position (deg)',
    'target_velocity':   'Target velocity (deg/s)',
    'scene_velocity':    'Scene velocity (deg/s)',
    'visual_flags':      'Visual flags',
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

    # Visual-flags panels don't need a zero line; all others do
    if panel_name != 'visual_flags':
        ax.axhline(0, color=_C['zero'], lw=0.5, ls='--')

    if panel_name == 'eye_position':
        # Show L and R eyes separately if they differ meaningfully (binocular scenario)
        ep_L = sig['eye_pos_L']
        ep_R = sig['eye_pos_R']
        bino_spread = np.max(np.abs(ep_L[:, 0] - ep_R[:, 0]))
        if bino_spread > 0.5:   # binocular — show L/R individually
            ax.plot(t, ep_L[:, 0], color='#2166ac', lw=1.2, label='L eye')
            ax.plot(t, ep_R[:, 0], color='#d6604d', lw=1.2, label='R eye')
        else:
            ax.plot(t, ep[:, 0], color=_C['eye'], lw=1.2, label='Eye yaw (version)')
        ax.plot(t, target_yaw_deg, color=_C['target'], lw=1.0, ls='--', label='Target')
        ax.legend(fontsize=6, loc='upper right')
        ax.set_ylabel('Eye / target position (deg)', fontsize=8)

    elif panel_name == 'eye_velocity':
        ax.plot(t, ev[:, 0],  color=_C['eye'],  lw=1.2, label='Eye vel')
        ax.plot(t, hv[:, 0],  color=_C['head'], lw=1.0, ls=':', label='Head vel')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'head_velocity':
        ax.plot(t, hv[:, 0], color=_C['head'], lw=1.2, label='Head vel yaw')
        if hv.shape[1] > 1 and np.any(hv[:, 1] != 0):
            ax.plot(t, hv[:, 1], color=_C['burst'], lw=1.0, ls='--', label='pitch')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'gaze_error':
        gaze = ep[:, 0] + np.cumsum(hv[:, 0]) * (t[1] - t[0])
        ax.plot(t, gaze, color=_C['error'], lw=1.2, label='Gaze error')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'retinal_error':
        ax.plot(t, ep_d[:, 0], color=_C['error'], lw=1.2, label='e_pos_delayed yaw')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'velocity_storage':
        ax.plot(t, sig['w_est'][:, 0], color=_C['vs'], lw=1.2, label='x_vs yaw')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'neural_integrator':
        ax.plot(t, sig['x_ni'][:, 0], color=_C['ni'], lw=1.2, label='x_ni yaw')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'saccade_burst':
        ax.plot(t, sig['u_burst'][:, 0], color=_C['burst'], lw=1.2, label='u_burst yaw')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'pursuit_drive':
        ax.plot(t, sig['x_pursuit'][:, 0], color=_C['pursuit'], lw=1.2, label='x_pursuit')
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'refractory':
        ax.plot(t, sig['z_ref'], color=_C['ref'], lw=1.2, label='z_ref (OPN)')
        ax.axhline(scenario.patient.g_burst * 0.0 + 0.5, color='k', lw=0.6, ls='--', alpha=0.4)
        ax.legend(fontsize=6, loc='upper right')

    elif panel_name == 'vergence':
        ax.plot(t, sig['vergence'][:, 0],  color='#1b7837', lw=1.5, label='Vergence (L−R)')
        ax.plot(t, sig['x_verg'][:, 0],    color='#762a83', lw=1.0, ls=':', label='x_verg state')
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
        # scene_present as a filled band; target flags as step lines
        ax.fill_between(t, 0, sp,  color='#aaaaaa', alpha=0.35, step='post', label='scene present')
        # Only show per-eye lines if they ever differ; otherwise one combined line
        if np.any(tpL != tpR):
            ax.step(t, tpL * 0.9 + 0.05, color='#2166ac', lw=1.5, where='post', label='target L')
            ax.step(t, tpR * 0.8 + 0.05, color='#d6604d', lw=1.5, where='post', label='target R')
        else:
            ax.step(t, tpL * 0.9 + 0.05, color=_C['target'], lw=1.5, where='post', label='target present')
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['off', 'on'], fontsize=7)
        ax.legend(fontsize=6, loc='upper right')
        ax.set_ylabel('Visual flags', fontsize=8)

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
