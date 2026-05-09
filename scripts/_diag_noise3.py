"""Trace internal SG states during flutter (seed=3, full noise) to find trigger."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import jax
from oculomotor.sim.simulator import PARAMS_DEFAULT, with_sensory, simulate, SimConfig, _IDX_SG, _IDX_NI_L, _IDX_NI_R, _IDX_PURSUIT
from oculomotor.sim import kinematics as km

DT = 0.001

def trace(seed, sigma_canal=0.0, sigma_pos=0.3, sigma_vel=5.0, amp=5):
    params = with_sensory(PARAMS_DEFAULT, sigma_canal=sigma_canal, sigma_pos=sigma_pos,
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
    _xp = np.array(states.brain[:, _IDX_PURSUIT])           # (T, 6) bilateral
    x_pursuit = _xp[:, :3] - _xp[:, 3:6]                    # (T, 3) NET

    z_opn    = x_sg[:, 6]
    z_acc    = x_sg[:, 7]
    x_copy   = x_sg[:, 0]
    e_held   = x_sg[:, 3]
    x_ebn_R  = x_sg[:, 8]   # yaw component
    x_ebn_L  = x_sg[:, 11]
    x_ibn_R  = x_sg[:, 14]
    x_ibn_L  = x_sg[:, 17]
    e_res    = e_held - x_copy
    x_ni_net = (ni_L - ni_R)[:, 0]

    # Find start of burst
    sac_mask = z_opn < 50
    if not sac_mask.any():
        print(f"seed={seed}: no saccade"); return
    s = int(np.argmax(sac_mask))

    print(f"\nseed={seed} (σ_pos={sigma_pos}, σ_vel={sigma_vel}):")
    print(f"{'t_ms':>6} {'z_opn':>7} {'z_acc':>7} {'e_held':>7} {'x_copy':>7} "
          f"{'e_res':>7} {'ebn_R':>7} {'ebn_L':>7} {'ibn_R':>7} {'ibn_L':>7} {'x_ni':>7} {'x_pur':>7}")
    # Print every 5ms for first 150ms of burst
    end_idx = min(s + 150, len(t)-1)
    for i in range(s, end_idx, 5):
        print(f"{t[i]*1000:6.0f} {z_opn[i]:7.1f} {z_acc[i]:7.3f} {e_held[i]:7.3f} {x_copy[i]:7.3f} "
              f"{e_res[i]:7.3f} {x_ebn_R[i]:7.3f} {x_ebn_L[i]:7.3f} {x_ibn_R[i]:7.3f} {x_ibn_L[i]:7.3f} "
              f"{x_ni_net[i]:7.3f} {x_pursuit[i,0]:7.3f}")

# Trace a flutter seed
trace(seed=3)
trace(seed=6)

# Compare with a clean seed
print("\n=== CLEAN SEED FOR COMPARISON ===")
trace(seed=0)
