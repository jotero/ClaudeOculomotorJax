"""Isolate which noise source degrades saccade accuracy."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import jax
from oculomotor.sim.simulator import PARAMS_DEFAULT, with_sensory, simulate, SimConfig, _IDX_SG, _IDX_NI_L, _IDX_NI_R
from oculomotor.sim import kinematics as km

DT = 0.001
N_TRIALS = 12

def run_condition(label, sigma_canal=0.0, sigma_pos=0.0, sigma_vel=0.0, amp=5):
    params = with_sensory(PARAMS_DEFAULT, sigma_canal=sigma_canal, sigma_pos=sigma_pos,
                          sigma_vel=sigma_vel, sigma_slip=0.0)
    T = int(0.6 / DT) + 1
    t = np.linspace(0.0, 0.6, T, dtype=np.float32)
    d = 1.0
    pt = np.zeros((T, 3), np.float32)
    pt[0] = [0.0, 0.0, d]
    pt[1:] = [np.tan(np.radians(amp)) * d, 0.0, d]
    tgt = km.build_target(t, lin_pos=pt, lin_vel=np.zeros((T, 3), np.float32))

    results = []
    for seed in range(N_TRIALS):
        states = simulate(params, t, target=tgt,
                          scene_present_array=np.ones(T, np.float32),
                          target_present_array=np.ones(T, np.float32),
                          max_steps=int(T*1.1)+2500,
                          sim_config=SimConfig(warmup_s=2.0),
                          return_states=True,
                          key=jax.random.PRNGKey(seed))

        x_sg = np.array(states.brain[:, _IDX_SG])
        # New layout: e_held(0:3) | z_opn(3) | z_acc(4) | ...
        z_opn = x_sg[:, 3]
        sac_mask = z_opn < 50
        if not sac_mask.any():
            continue
        s = int(np.argmax(sac_mask))
        # Find burst offset
        after = sac_mask[s:]
        if after.all():
            e = len(t) - 1
        else:
            e = s + int(np.argmin(after))

        ni_L = np.array(states.brain[:, _IDX_NI_L])
        ni_R = np.array(states.brain[:, _IDX_NI_R])
        x_ni_net_yaw = (ni_L - ni_R)[e, 0]
        e_held_yaw = x_sg[s, 0]  # e_held yaw at onset
        results.append((x_ni_net_yaw, e_held_yaw))

    if results:
        vals = [r[0] for r in results]
        ehs  = [r[1] for r in results]
        print(f"  {label:35s}  x_ni@offset={np.mean(vals):5.2f}±{np.std(vals):.2f}  "
              f"e_held@onset={np.mean(ehs):5.2f}±{np.std(ehs):.2f}  n={len(vals)}")
    else:
        print(f"  {label}: no saccades detected")

print(f"=== Noise isolation (amp=5°, {N_TRIALS} trials) ===")
run_condition("no noise",                  0.0, 0.0, 0.0)
run_condition("canal only (σ=2)",          sigma_canal=2.0)
run_condition("pos only   (σ=0.3)",        sigma_pos=0.3)
run_condition("vel only   (σ=5)",          sigma_vel=5.0)
run_condition("full noise",                sigma_canal=2.0, sigma_pos=0.3, sigma_vel=5.0)

print()
print(f"=== Noise isolation (amp=20°, {N_TRIALS} trials) ===")
run_condition("no noise",                  0.0, 0.0, 0.0, amp=20)
run_condition("canal only (σ=2)",          sigma_canal=2.0, amp=20)
run_condition("pos only   (σ=0.3)",        sigma_pos=0.3, amp=20)
run_condition("vel only   (σ=5)",          sigma_vel=5.0, amp=20)
run_condition("full noise",                sigma_canal=2.0, sigma_pos=0.3, sigma_vel=5.0, amp=20)
