"""Trace individual noisy trials to see if early termination, corrective saccades, or pursuit contamination."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import jax
from oculomotor.sim.simulator import PARAMS_DEFAULT, with_sensory, simulate, SimConfig, _IDX_SG, _IDX_NI_L, _IDX_NI_R, _IDX_PURSUIT
from oculomotor.sim import kinematics as km

DT = 0.001

def run_trial(seed, sigma_pos=0.3, sigma_vel=5.0, amp=5):
    params = with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_pos=sigma_pos,
                          sigma_vel=sigma_vel, sigma_slip=0.0)
    T = int(0.6 / DT) + 1
    t = np.linspace(0.0, 0.6, T, dtype=np.float32)
    d = 1.0
    pt = np.zeros((T, 3), np.float32)
    pt[0] = [0.0, 0.0, d]
    pt[1:] = [np.tan(np.radians(amp)) * d, 0.0, d]
    tgt = km.build_target(t, lin_pos=pt, lin_vel=np.zeros((T, 3), np.float32))

    states = simulate(params, t, target=tgt,
                      scene_present_array=np.ones(T, np.float32),
                      target_present_array=np.ones(T, np.float32),
                      max_steps=int(T*1.1)+2500,
                      sim_config=SimConfig(warmup_s=2.0),
                      return_states=True,
                      key=jax.random.PRNGKey(seed))

    x_sg = np.array(states.brain[:, _IDX_SG])
    ni_L = np.array(states.brain[:, _IDX_NI_L])
    ni_R = np.array(states.brain[:, _IDX_NI_R])
    x_pursuit = np.array(states.brain[:, _IDX_PURSUIT])
    z_opn = x_sg[:, 6]
    x_copy_yaw = x_sg[:, 0]
    e_held_yaw = x_sg[:, 3]
    z_acc = x_sg[:, 7]
    x_ni_net = (ni_L - ni_R)[:, 0]

    # All saccades: find each OPN pause
    sac_mask = z_opn < 50
    saccades = []
    i = 0
    while i < len(sac_mask):
        if sac_mask[i]:
            s = i
            while i < len(sac_mask) and sac_mask[i]:
                i += 1
            e = i - 1
            saccades.append((s, e))
        else:
            i += 1

    print(f"\nseed={seed}:")
    for si, (s, e) in enumerate(saccades):
        dur = (e - s) * DT * 1000
        delta_ni = x_ni_net[e] - x_ni_net[max(0, s-1)]
        print(f"  sac#{si+1}: t={t[s]*1000:.0f}→{t[e]*1000:.0f}ms dur={dur:.0f}ms  "
              f"e_held={e_held_yaw[s]:.2f}  x_copy@end={x_copy_yaw[e]:.2f}  "
              f"x_ni_net@end={x_ni_net[e]:.2f}  Δni={delta_ni:.2f}")
    # pursuit state at burst offset
    if saccades:
        e = saccades[0][1]
        print(f"  pursuit@offset: {x_pursuit[e, 0]:.3f} deg/s")
        print(f"  x_ni_net@t=0.6s: {x_ni_net[-1]:.2f}")

print("=== Individual noisy trials (pos+vel noise, amp=5°) ===")
for seed in range(8):
    run_trial(seed)

print("\n=== Individual noisy trials (pos noise only, amp=5°) ===")
for seed in [0, 1, 2, 3]:
    run_trial(seed, sigma_pos=0.3, sigma_vel=0.0)

print("\n=== Individual noisy trials (vel noise only, amp=5°) ===")
for seed in [0, 1, 2, 3]:
    run_trial(seed, sigma_pos=0.0, sigma_vel=5.0)
