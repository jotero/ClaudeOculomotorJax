"""Quick SG diagnostic: saccade metrics + VOR NaN check."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import jax, jax.numpy as jnp
from oculomotor.sim.simulator import PARAMS_DEFAULT, with_sensory, with_brain, simulate, SimConfig
from oculomotor.sim import kinematics as km
from oculomotor.analysis import extract_sg

DT = 0.001
THETA = with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_pos=0.0,
                     sigma_vel=0.0, sigma_slip=0.0)

def saccade_test():
    print("=== Saccade test ===")
    for amp in [5, 10, 20, 30, 40]:
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
        eye_yaw = np.array(states.plant[:, 3])  # right eye yaw
        vel = np.gradient(eye_yaw, DT)
        pv = np.max(np.abs(vel))
        fp = eye_yaw[-1]
        # check NaN
        nan_t = np.where(np.isnan(eye_yaw))[0]
        nan_str = f"NaN@{nan_t[0]*DT:.3f}s" if len(nan_t) else "OK"
        print(f"  amp={amp:2d}  peak_vel={pv:5.0f} deg/s  final={fp:6.2f} deg  {nan_str}")
        # OPN diagnostics
        sg = extract_sg(states, THETA)
        z_opn = sg['z_opn']
        print(f"         OPN min={z_opn.min():.1f} max={z_opn.max():.1f}")

def vor_test():
    print("\n=== VOR NaN test ===")
    T = int(3.0 / DT) + 1
    t = np.linspace(0.0, 3.0, T, dtype=np.float32)
    hv = np.zeros((T, 3), np.float32)
    hv[:, 0] = 50.0  # 50 deg/s yaw
    states = simulate(THETA, t,
                      head=km.build_kinematics(t, rot_vel=hv),
                      scene_present_array=np.zeros(T, np.float32),
                      target_present_array=np.zeros(T, np.float32),
                      max_steps=int(T*1.1)+500,
                      sim_config=SimConfig(warmup_s=0.0),
                      return_states=True,
                      key=jax.random.PRNGKey(0))
    eye_yaw = np.array(states.plant[:, 3])
    vel = np.gradient(eye_yaw, DT)
    for ti in [0.1, 0.5, 1.0, 2.0, 2.9]:
        idx = int(ti / DT)
        print(f"  t={ti:.1f}s: eye_vel={vel[idx]:.1f} deg/s  pos={eye_yaw[idx]:.1f} deg")
    nan_t = np.where(np.isnan(eye_yaw))[0]
    if len(nan_t):
        print(f"  *** NaN first at t={nan_t[0]*DT:.3f}s ***")
    else:
        print("  No NaN — stable!")

if __name__ == '__main__':
    saccade_test()
    vor_test()
