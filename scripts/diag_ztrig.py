"""Diagnostic: trace z_acc, z_trig, z_opn for a single saccade with the z_trig design."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, with_sensory, simulate
from oculomotor.sim import kinematics as km
from oculomotor.analysis import extract_sg

DT   = 0.001
THETA = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_pos=0.0, sigma_vel=0.0), sigma_acc=0.0, g_burst=700.0)

T_end = 0.8
t_jump = 0.1

for amp in [0.5, 2.0, 5.0, 15.0]:
    t_np = np.arange(0.0, T_end, DT)
    T = len(t_np)
    pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
    pt3[t_np >= t_jump, 0] = np.tan(np.radians(amp))

    st = simulate(THETA, jnp.array(t_np),
                  target=km.build_target(t_np, lin_pos=np.array(pt3)),
                  scene_present_array=jnp.ones(T),
                  max_steps=int(T_end/DT)+500, return_states=True,
                  key=jax.random.PRNGKey(0))

    sg_dict = extract_sg(st, THETA)
    eye = (np.array(st.plant.left[:, 0]) + np.array(st.plant.right[:, 0])) / 2.0

    z_acc  = sg_dict['z_acc']
    z_opn  = sg_dict['z_opn']
    z_trig = sg_dict['z_trig']
    uburst = sg_dict['u_burst'][:, 0]
    e_held = sg_dict['e_held'][:, 0]

    max_burst = np.max(np.abs(uburst))
    fired = max_burst > 20.0
    final_eye = eye[-1]

    print(f"\n=== {amp}deg target: burst={max_burst:.0f} deg/s, eye_final={final_eye:.2f}deg, FIRED={'YES' if fired else 'NO'} ===")
    print(f"  max z_trig={np.max(z_trig):.4f}  min z_opn={np.min(z_opn):.2f}  max z_acc={np.max(z_acc):.4f}")
    if fired:
        # find saccade window
        i_on = np.argmax(np.abs(uburst) > 20.0)
        i_off = i_on + np.argmax(np.abs(uburst[i_on:]) < 20.0) if np.any(np.abs(uburst[i_on:]) < 20.0) else len(uburst)-1
        print(f"  saccade: t_on={t_np[i_on]:.3f}s  t_off={t_np[i_off]:.3f}s  dur={1000*(t_np[i_off]-t_np[i_on]):.0f}ms")

print("\nDone.")
