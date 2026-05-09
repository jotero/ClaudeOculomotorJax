"""Trace z_opn, e_held over saccades to diagnose OPN oscillation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import jax
from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_sensory, simulate, SimConfig,
)
from oculomotor.sim import kinematics as km

DT = 0.001
THETA = with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_pos=0.0,
                     sigma_vel=0.0, sigma_slip=0.0)

for amp in [5, 20]:
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

    sg_st  = states.brain.sg
    e_held = np.array(sg_st.e_held)[:, 0]   # yaw
    z_opn  = np.array(sg_st.z_opn)
    z_acc  = np.array(sg_st.z_acc)
    x_ebn_R = np.array(sg_st.ebn_R)[:, 0]
    x_ibn_R = np.array(sg_st.ibn_R)[:, 0]

    # Find saccade window
    sac_mask = z_opn < 50
    if not sac_mask.any():
        print(f"amp={amp}: no saccade"); continue
    s = int(np.argmax(sac_mask))
    # print every 10ms for 400ms after onset
    print(f"\n=== amp={amp}° ===")
    print(f"{'t_ms':>6} {'z_opn':>10} {'e_held':>8} {'ebn_R':>8} {'ibn_R':>8} {'z_acc':>7}")
    end_idx = min(s + 400, len(t)-1)
    for i in range(s, end_idx, 10):
        print(f"{t[i]*1000:6.1f} {z_opn[i]:10.1f} {e_held[i]:8.3f} {x_ebn_R[i]:8.3f} {x_ibn_R[i]:8.3f} {z_acc[i]:7.3f}")
