"""Runner — converts a SimulationScenario into a simulation + figure.

Entry point::

    from oculomotor.runner import run_scenario
    fig = run_scenario(scenario)      # returns matplotlib Figure
    fig.savefig('output.png', dpi=150, bbox_inches='tight')

The runner does three things:
    1. Build stimulus arrays from ``scenario.head_motion``, ``scenario.target``,
       and ``scenario.visual`` using ``oculomotor.stimuli``.
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

from oculomotor import stimuli as stim
from oculomotor.scenario import SimulationScenario, SimulationComparison, Patient
from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, with_sensory, simulate,
    _IDX_VS, _IDX_NI, _IDX_SG, _IDX_EC, _IDX_VIS, _IDX_PURSUIT,
)
from oculomotor.models.brain_models import saccade_generator as sg_mod
from oculomotor.models.sensory_models.sensory_model import C_slip, C_pos, C_vel


# ── Stimulus builder ──────────────────────────────────────────────────────────

def _build_stimulus(
    scenario: SimulationScenario,
) -> dict:
    """Convert segment lists to simulator keyword arrays.

    Returns a dict suitable for ``**stim_kwargs`` in ``simulate()``, plus
    a 't_array' key with the time axis.
    """
    dt  = 0.001
    dur = scenario.duration_s
    T   = int(round(dur / dt))
    t   = np.arange(T, dtype=np.float32) * dt

    hv          = stim.build_head_array(scenario.head,   T, dt)
    pt, vt      = stim.build_target_arrays(scenario.target, T, dt)
    vs, sp, tp  = stim.build_visual_arrays(scenario.visual, T, dt)

    return dict(
        t_array              = t,
        head_vel_array       = jnp.array(hv),
        p_target_array       = jnp.array(pt),
        v_target_array       = jnp.array(vt),
        v_scene_array        = jnp.array(vs),
        scene_present_array  = jnp.array(sp),
        target_present_array = jnp.array(tp),
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
    )
    return params


# ── Signal extraction helpers ─────────────────────────────────────────────────

def _extract_signals(states, params, t_np: np.ndarray) -> dict:
    """Extract named signals from a SimState trajectory."""
    dt = t_np[1] - t_np[0]

    x_p       = np.array(states.plant)           # (T, 3)
    x_vs      = np.array(states.brain[:, _IDX_VS])
    x_ni      = np.array(states.brain[:, _IDX_NI])
    x_sg      = np.array(states.brain[:, _IDX_SG])
    x_pursuit = np.array(states.brain[:, _IDX_PURSUIT])
    x_vis     = np.array(states.sensory[:, _IDX_VIS])

    # Eye velocity (numerical derivative, smoothed)
    w_eye = np.gradient(x_p, dt, axis=0)

    # Canal afferents — approximate from first-order derivative of eye position
    # (use x_vs as proxy for head velocity estimate)
    w_est = x_vs  # VS state ≈ w_est

    # Retinal signals from visual delay cascade
    e_pos_delayed = x_vis @ np.array(C_pos).T    # (T, 3)

    # Saccade burst (re-compute from SG state + e_pos_delayed)
    def _burst_at(state):
        e_pd = C_pos @ state.sensory[_IDX_VIS]
        _, u = sg_mod.step(state.brain[_IDX_SG], e_pd, params.brain)
        return u
    u_burst = np.array(jax.vmap(_burst_at)(states))  # (T, 3)

    # SG sub-states
    x_copy = x_sg[:, :3]
    z_ref  = x_sg[:, 3]
    e_held = x_sg[:, 4:7]
    z_sac  = x_sg[:, 7]
    z_acc  = x_sg[:, 8]

    return dict(
        eye_pos        = x_p,
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
}


def _draw_panel(ax, panel_name: str, t: np.ndarray, sig: dict,
                stim_kw: dict, scenario: SimulationScenario):
    """Draw one signal panel onto ax."""
    ax.set_ylabel(_PANEL_LABELS.get(panel_name, panel_name), fontsize=8)
    ax.tick_params(labelsize=7)
    ax.axhline(0, color=_C['zero'], lw=0.5, ls='--')

    ep  = sig['eye_pos']
    ev  = sig['eye_vel']
    ep_d = sig['e_pos_delayed']

    # Head velocity from stim
    hv = np.array(stim_kw['head_vel_array'])
    # Target position from stim
    pt = np.array(stim_kw['p_target_array'])
    target_yaw_deg = np.degrees(np.arctan(pt[:, 0]))

    if panel_name == 'eye_position':
        ax.plot(t, ep[:, 0],  color=_C['eye'],    lw=1.2, label='Eye yaw')
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

    # Enforce a minimum visible range on velocity / derivative panels
    if panel_name in ('eye_velocity', 'head_velocity', 'saccade_burst', 'pursuit_drive'):
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


# ── Public API ────────────────────────────────────────────────────────────────

def run_scenario(scenario: SimulationScenario, output_path: str | None = None) -> plt.Figure:
    """Run a SimulationScenario end-to-end and return a matplotlib Figure.

    Args:
        scenario:    Fully populated SimulationScenario object.
        output_path: If given, save figure to this path (PNG/SVG/PDF).

    Returns:
        matplotlib.figure.Figure
    """
    # Build stimulus and params
    stim_kw = _build_stimulus(scenario)
    t_array = stim_kw.pop('t_array')
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

    # Reconstruct stim_kw with t for plotting (without t_array key)
    plot_stim = {k: v for k, v in stim_kw.items()}

    # Build figure
    fig = _build_figure(t_array, sig, plot_stim, scenario)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {output_path}")

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
