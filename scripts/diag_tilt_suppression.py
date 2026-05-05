"""Diagnostic: peak v_lin / vergence / eye pitch during tilt-suppression test.

Compares 4 parameter combos so we can see how K_grav / K_lin / tau_a_lin trade off:
  A: pre-Laurens   (K_grav=0.2, K_lin=0.1, tau_a=0.5)
  B: K_grav up     (K_grav=0.5, K_lin=0.1, tau_a=0.5)
  C: + tau_a long  (K_grav=0.5, K_lin=0.1, tau_a=1.5)   ← current default after Laurens pass
  D: + K_lin down  (K_grav=0.5, K_lin=0.05, tau_a=1.5)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from oculomotor.sim.simulator import (PARAMS_DEFAULT, simulate, with_brain,
                                       SimConfig, _IDX_GRAV)
from oculomotor.models.brain_models.brain_model import _IDX_HEAD
from oculomotor.sim import kinematics as km
from oculomotor.analysis import vs_net

DT = 0.001
# Tilt suppression: 60°/s yaw constant rotation, then nose-down 60° pitch tilt
SPIN_VEL = 60.0
PITCH_DEG = 60.0
PITCH_VEL = 30.0
T_PRE   = 5.0     # spin in upright
T_PITCH = PITCH_DEG / PITCH_VEL
T_POST  = 50.0    # spin in tilted position
TOTAL   = T_PRE + T_PITCH + T_POST
t = np.arange(0.0, TOTAL, DT)
T = len(t)

yaw_vel = np.full(T, SPIN_VEL, dtype=np.float32)
pitch_vel = np.zeros(T, dtype=np.float32)
pitch_mask = (t >= T_PRE) & (t < T_PRE + T_PITCH)
pitch_vel[pitch_mask] = PITCH_VEL
head_vel = np.stack([yaw_vel, pitch_vel, np.zeros(T)], axis=1)
head_km = km.build_kinematics(t, rot_vel=head_vel)

combos = [
    ('Laurens 2011 (K=0.1, gate=5)  ', 0.6, 0.1,  1.5, 2.0),
    ('K=1.0 (over-tuned)            ', 0.5, 1.0,  1.5, 2.0),
    ('K=0.05 τ=0.5 (pre-Laurens)    ', 0.2, 0.05, 0.5, 2.0),
]

print(f"Tilt-suppression: yaw {SPIN_VEL}°/s constant, pitch {PITCH_DEG}° at "
      f"{PITCH_VEL}°/s starting t={T_PRE}s. Total {TOTAL}s. Dark.")
print()
print(f"{'combo':<18}{'|v_lin|max':>12}{'|verg|max':>12}{'pitchmax':>12}{'g_est_x':>10}{'g_est_y':>10}")
print("-" * 74)

for name, kg, kl, ta, th in combos:
    p = with_brain(PARAMS_DEFAULT, K_grav=kg, K_lin=kl, tau_a_lin=ta, tau_head=th)
    st = simulate(p, t, head=head_km,
                  scene_present_array=np.zeros(T),
                  target_present_array=np.zeros(T),
                  sim_config=SimConfig(warmup_s=0.0),
                  return_states=True)
    v_lin = np.array(st.brain[:, _IDX_HEAD])
    verg  = np.array(st.plant[:, 0] - st.plant[:, 3])
    pitch = (np.array(st.plant[:, 1]) + np.array(st.plant[:, 4])) / 2.0
    g_est = np.array(st.brain[:, _IDX_GRAV])

    # window: post-pitch onset
    post = t > T_PRE
    post5s = (t > T_PRE) & (t < T_PRE + 5.0)

    vmag = np.linalg.norm(v_lin[post], axis=1)
    print(f"{name:<18}"
          f"{vmag.max():>12.3f}"   # m/s
          f"{np.abs(verg[post]).max():>12.3f}"   # deg
          f"{np.abs(pitch[post5s]).max():>12.3f}"   # deg (5s post-tilt)
          f"{g_est[-1, 0]:>10.3f}"
          f"{g_est[-1, 1]:>10.3f}")
