"""Trace VOR quick-phase generator state to diagnose why QP not firing."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import jax
from oculomotor.sim.simulator import PARAMS_DEFAULT, with_sensory, simulate, SimConfig
from oculomotor.sim import kinematics as km

DT = 0.001
THETA = with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_pos=0.0, sigma_vel=0.0, sigma_slip=0.0)

T = int(3.0 / DT) + 1
t = np.linspace(0.0, 3.0, T, dtype=np.float32)
hv = np.zeros((T, 3), np.float32)
hv[:, 0] = 50.0  # 50 deg/s yaw rightward

states = simulate(THETA, t,
                  head=km.build_kinematics(t, rot_vel=hv),
                  scene_present_array=np.zeros(T, np.float32),
                  target_present_array=np.zeros(T, np.float32),
                  max_steps=int(T*1.1)+500,
                  sim_config=SimConfig(warmup_s=0.0),
                  return_states=True,
                  key=jax.random.PRNGKey(0))

sg_st = states.brain.sg
ni_L  = np.array(states.brain.ni.L)
ni_R  = np.array(states.brain.ni.R)
vs_L  = np.array(states.brain.sm.vs_L)
vs_R  = np.array(states.brain.sm.vs_R)

z_opn  = np.array(sg_st.z_opn)
z_acc  = np.array(sg_st.z_acc)
e_held = np.array(sg_st.e_held)[:, 0]   # yaw
x_copy = e_held                          # legacy alias
x_ni_net = (ni_L - ni_R)[:, 0]
w_est_yaw = (vs_L - vs_R)[:, 0]  # VS net = head velocity estimate (yaw)

eye_yaw = np.array(states.plant.right[:, 0])
eye_vel = np.gradient(eye_yaw, DT)

print("Tracing VOR quick-phase generator:")
print(f"{'t_ms':>6} {'z_opn':>10} {'z_acc':>7} {'x_ni':>7} {'w_est':>7} {'e_held':>7} {'eye':>7} {'vel':>7}")
for ti in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.9]:
    i = int(ti / DT)
    print(f"{ti*1000:6.0f} {z_opn[i]:10.2f} {z_acc[i]:7.3f} {x_ni_net[i]:7.2f} {w_est_yaw[i]:7.2f} {e_held[i]:7.2f} {eye_yaw[i]:7.2f} {eye_vel[i]:7.1f}")

# Also show any quick phases (z_opn dips)
print("\nQuick phases detected (z_opn < 50):")
sac_mask = z_opn < 50
if sac_mask.any():
    i = 0
    while i < len(sac_mask):
        if sac_mask[i]:
            s = i
            while i < len(sac_mask) and sac_mask[i]: i += 1
            e = i - 1
            print(f"  t={t[s]*1000:.0f}→{t[e]*1000:.0f}ms  x_ni@onset={x_ni_net[s]:.1f}  e_held={e_held[s]:.1f}  Δni={x_ni_net[e]-x_ni_net[s]:.1f}")
        else:
            i += 1
else:
    print("  None!")
