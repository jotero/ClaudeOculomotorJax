"""Diagnostic: trace z_acc, e_held, u_burst for the oblique sequence."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax.numpy as jnp
import jax

from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, with_sensory, simulate
from oculomotor.sim import kinematics as km
from oculomotor.analysis import extract_sg

DT   = 0.001
THETA = with_brain(PARAMS_DEFAULT, g_burst=700.0)

T_end = 3.5
t_np  = np.arange(0.0, T_end, DT)
T     = len(t_np)

jumps = [(0.3, 12.0, 0.0), (0.9, 12.0, 8.0),
         (1.6,  0.0, 8.0), (2.2,  0.0, 0.0), (2.8, -10.0, 5.0)]
pt3   = np.zeros((T, 3)); pt3[:, 2] = 1.0
tgt_h = np.zeros(T); tgt_v = np.zeros(T)
for t_j, y, p in jumps:
    tgt_h[t_np >= t_j] = y; tgt_v[t_np >= t_j] = p
pt3[:, 0] = np.tan(np.radians(tgt_h))
pt3[:, 1] = np.tan(np.radians(tgt_v))

print("Running oblique simulation...")
st = simulate(THETA, jnp.array(t_np),
              target=km.build_target(t_np, lin_pos=np.array(pt3)),
              scene_present_array=jnp.ones(T),
              max_steps=int(T_end / DT) + 500,
              return_states=True, key=jax.random.PRNGKey(0))

sg  = extract_sg(st, THETA)
eye_L = np.array(st.plant.left)
eye_R = np.array(st.plant.right)
eye   = (eye_L + eye_R) / 2.0

z_acc  = np.array(sg['z_acc'])
z_opn  = np.array(sg['z_opn'])
e_held = np.array(sg['e_held'])      # (T, 3)
uburst = np.array(sg['u_burst'])     # (T, 3)

# Print key signals around saccade windows
print(f"\n{'t':>6} {'eye_H':>7} {'eye_V':>7} {'e_held_H':>9} {'e_held_V':>9} "
      f"{'z_acc':>7} {'z_opn':>7} {'uburst_H':>9} {'uburst_V':>9}")
print("-" * 80)

# Window: 0.4 to 2.5s, every 20ms
for i in range(int(0.4/DT), int(2.5/DT), 20):
    t = t_np[i]
    print(f"{t:6.3f} {eye[i,0]:7.2f} {eye[i,1]:7.2f} {e_held[i,0]:9.3f} {e_held[i,1]:9.3f} "
          f"{z_acc[i]:7.3f} {z_opn[i]:7.1f} {uburst[i,0]:9.1f} {uburst[i,1]:9.1f}")

# Find saccade onset times (|u_burst| > 20 deg/s)
sac_h = np.abs(uburst[:, 0]) > 20
sac_v = np.abs(uburst[:, 1]) > 20
on_h  = np.where(np.diff(sac_h.astype(int)) > 0)[0]
on_v  = np.where(np.diff(sac_v.astype(int)) > 0)[0]

print(f"\nHorizontal burst onsets (t): {t_np[on_h]}")
print(f"Vertical   burst onsets (t): {t_np[on_v]}")

print(f"\nTarget jump times: {[t for t,_,_ in jumps]}")
print(f"Eye H at each 0.1s step (0.3-1.8s):")
for t in np.arange(0.3, 1.81, 0.1):
    i = int(t/DT)
    print(f"  t={t:.1f}: eye=({eye[i,0]:.2f}H, {eye[i,1]:.2f}V)  z_acc={z_acc[i]:.3f}")
