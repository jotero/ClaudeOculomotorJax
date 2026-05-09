"""Detailed SG diagnostic: check NI x_net vs x_copy vs eye position."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import jax, jax.numpy as jnp
from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_sensory, simulate, SimConfig,
)
from oculomotor.sim import kinematics as km
from oculomotor.analysis import extract_sg

DT = 0.001
THETA = with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_pos=0.0,
                     sigma_vel=0.0, sigma_slip=0.0)

def run(amp):
    T = int(0.6 / DT) + 1
    t = np.linspace(0.0, 0.6, T, dtype=np.float32)
    d = 1.0
    pt = np.zeros((T, 3), np.float32)
    pt[0] = [0.0, 0.0, d]
    pt[1:] = [np.tan(np.radians(amp)) * d, 0.0, d]
    lv = np.zeros((T, 3), np.float32)
    tgt = km.build_target(t, lin_pos=pt, lin_vel=lv)
    states = simulate(THETA, t, target=tgt,
                      scene_present_array=np.ones(T, np.float32),
                      target_present_array=np.ones(T, np.float32),
                      max_steps=int(T*1.1)+2500,
                      sim_config=SimConfig(warmup_s=2.0),
                      return_states=True,
                      key=jax.random.PRNGKey(0))

    # Plant states: left eye [0:3], right eye [3:6]
    eye_L_yaw = np.array(states.plant.left[:, 0])   # left eye yaw
    eye_R_yaw = np.array(states.plant.right[:, 0])   # right eye yaw

    # NI net state (x_L - x_R), yaw component
    x_ni_L = np.array(states.brain.ni.L)[:, 0]   # left NI pop, yaw
    x_ni_R = np.array(states.brain.ni.R)[:, 0]   # right NI pop, yaw
    x_ni_net_yaw = x_ni_L - x_ni_R

    # SG states
    sg = extract_sg(states, THETA)
    x_copy_yaw = sg['x_copy'][:, 0]
    e_held_yaw = sg['e_held'][:, 0]
    z_opn = sg['z_opn']
    u_burst_yaw = sg['u_burst'][:, 0]

    # Find saccade window: OPN paused
    sac_mask = z_opn < 50
    if sac_mask.any():
        sac_start = np.argmax(sac_mask)
        sac_end = sac_start + np.argmin(sac_mask[sac_start:]) if sac_mask[sac_start:].all() == False else len(t)-1
        sac_end = min(sac_end, len(t)-1)
        s, e = sac_start, sac_end
        print(f"\namp={amp}° saccade:")
        print(f"  OPN paused: t={t[s]:.3f}s to t={t[e]:.3f}s ({(e-s)*DT*1000:.0f}ms)")
        print(f"  At onset  (t={t[s]:.3f}): x_ni_net={x_ni_net_yaw[s]:.3f}  x_copy={x_copy_yaw[s]:.3f}  e_held={e_held_yaw[s]:.3f}")
        print(f"  At offset (t={t[e]:.3f}): x_ni_net={x_ni_net_yaw[e]:.3f}  x_copy={x_copy_yaw[e]:.3f}  e_held={e_held_yaw[e]:.3f}")
        print(f"  At t=0.6s:  x_ni_net={x_ni_net_yaw[-1]:.3f}  eye_R={eye_R_yaw[-1]:.3f}  eye_L={eye_L_yaw[-1]:.3f}")
        print(f"  NI Δ: {x_ni_net_yaw[e] - x_ni_net_yaw[s]:.3f}°  copy Δ: {x_copy_yaw[e] - x_copy_yaw[s]:.3f}°  e_held at offset: {e_held_yaw[e]:.3f}°")
        print(f"  Peak u_burst: {np.max(np.abs(u_burst_yaw)):.1f} deg/s")
    else:
        print(f"\namp={amp}°: no saccade detected!")

if __name__ == '__main__':
    for amp in [5, 20]:
        run(amp)
