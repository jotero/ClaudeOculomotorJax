"""Focused look at DARK eye velocity for T-VOR cascade — zoomed in for slow phase visibility.

Runs the same trapezoid head-velocity stimulus as the cascade panel, but only the DARK
condition (no scene, no target → pure vestibular T-VOR), and plots eye velocity at a
tight ±20 deg/s zoom so the slow-phase compensation is clearly visible between any
residual saccade-like spikes.
"""

import numpy as np
import jax
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate
from oculomotor.sim import kinematics as km

DT = 0.001
KEY = jax.random.PRNGKey(0)

PEAK = 0.20
RAMP = 0.2
HOLD = 1.6
T_PULSE = 0.5
T_TOTAL = 5.0
DEPTH = 0.4

t = np.arange(0.0, T_TOTAL, DT)
T = len(t)

t_rel = t - T_PULSE
env = np.zeros(T)
env[(t_rel >= 0) & (t_rel < RAMP)] = t_rel[(t_rel >= 0) & (t_rel < RAMP)] / RAMP
env[(t_rel >= RAMP) & (t_rel < RAMP + HOLD)] = 1.0
env[(t_rel >= RAMP + HOLD) & (t_rel < 2*RAMP + HOLD)] = 1.0 - (t_rel[(t_rel >= RAMP + HOLD) & (t_rel < 2*RAMP + HOLD)] - RAMP - HOLD) / RAMP

AXES = [(0, +1, 'Sway',  'rightward'),
        (1, +1, 'Heave', 'upward'),
        (2, -1, 'Surge', 'backward')]

fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
fig.suptitle('T-VOR cascade — DARK only (no scene, no target, pure vestibular)\n'
             'Eye velocity zoomed to show slow phase',
             fontsize=12, fontweight='bold')

for col, (axis, sign, name, dir_descr) in enumerate(AXES):
    head_vel = np.zeros((T, 3), dtype=np.float32)
    head_vel[:, axis] = (sign * PEAK * env).astype(np.float32)
    head = km.build_kinematics(t, lin_vel=head_vel)
    pt = np.tile([0.0, 0.0, DEPTH], (T, 1)).astype(np.float32)
    target = km.build_target(t, lin_pos=pt)

    st = simulate(
        PARAMS_DEFAULT, t, head=head, target=target,
        scene_present_array=np.zeros(T),
        target_present_array=np.zeros(T),
        return_states=True, key=KEY)

    eye_L = np.array(st.plant.left)
    eye_R = np.array(st.plant.right)
    eye_version = (eye_L + eye_R) / 2.0
    eye_verg    = eye_L[:, 0] - eye_R[:, 0]

    eye_vel_yaw   = np.gradient(eye_version[:, 0], DT)
    eye_vel_pitch = np.gradient(eye_version[:, 1], DT)
    eye_vel_verg  = np.gradient(eye_verg, DT)

    # Row 0: stimulus
    ax = axes[0, col]
    ax.plot(t, head_vel[:, axis] * 100, color='gray', lw=1.4)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_title(f'{name} ({dir_descr})', fontsize=11, fontweight='bold')
    if col == 0: ax.set_ylabel('Head vel\n(cm/s)', fontsize=10)
    ax.grid(True, alpha=0.2)

    # Row 1: eye position
    ax = axes[1, col]
    ax.plot(t, eye_version[:, 0], color='#1f77b4', lw=1.5, label='Yaw')
    ax.plot(t, eye_version[:, 1], color='#2ca02c', lw=1.5, label='Pitch')
    ax.plot(t, eye_verg,          color='#d62728', lw=1.5, label='Vergence (L−R)')
    ax.axhline(0, color='k', lw=0.4)
    if col == 0: ax.set_ylabel('Eye position\n(deg)', fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2)

    # Row 2: eye velocity, zoomed tight
    ax = axes[2, col]
    ax.plot(t, eye_vel_yaw,   color='#1f77b4', lw=1.2, label='Yaw')
    ax.plot(t, eye_vel_pitch, color='#2ca02c', lw=1.2, label='Pitch')
    ax.plot(t, eye_vel_verg,  color='#d62728', lw=1.2, label='Vergence')
    ax.axhline(0, color='k', lw=0.4)
    if col == 0: ax.set_ylabel('Eye velocity\n(deg/s, ±20 zoom)', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylim(-20, 20)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = 'docs/benchmarks/figures/tvor_cascade_dark_zoom.png'
fig.savefig(out, dpi=130)
print(f'Saved: {out}')
