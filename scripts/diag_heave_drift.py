"""Diagnose the heave LIT post-pulse g_est[z] drift.

Prints internal signals over time during heave to expose:
  - canal afferents (should be 0 for pure translation)
  - VS net w_est (should be 0 for pure translation; non-zero = OKR contamination)
  - scene_slip into VS (should be 0 with perfect EC)
  - g_est components (should stay [0, 9.81, 0])
  - omega_tvor (T-VOR's commanded eye velocity)
  - actual eye velocity (from plant gradient)
  - EC mismatch (omega_tvor − actual eye velocity)
"""

import numpy as np
import jax
import jax.numpy as jnp

from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate
from oculomotor.sim import kinematics as km
from oculomotor.models.sensory_models import sensory_model
from oculomotor.models.sensory_models.canal import PINV_SENS as CANAL_PINV

DT = 0.001
PEAK = 0.20
RAMP = 0.2
HOLD = 1.6
T_PULSE = 0.5
T_TOTAL = 6.0

t = np.arange(0.0, T_TOTAL, DT)
T = len(t)

t_rel = t - T_PULSE
env = np.zeros(T)
env[(t_rel >= 0) & (t_rel < RAMP)] = t_rel[(t_rel >= 0) & (t_rel < RAMP)] / RAMP
env[(t_rel >= RAMP) & (t_rel < RAMP + HOLD)] = 1.0
env[(t_rel >= RAMP + HOLD) & (t_rel < 2*RAMP + HOLD)] = 1.0 - (t_rel[(t_rel >= RAMP + HOLD) & (t_rel < 2*RAMP + HOLD)] - RAMP - HOLD) / RAMP

# Heave +y, LIT scene only
head_vel = np.zeros((T, 3), dtype=np.float32)
head_vel[:, 1] = PEAK * env
head = km.build_kinematics(t, lin_vel=head_vel)
target = km.build_target(t, lin_pos=np.tile([0.0, 0.0, 0.4], (T, 1)).astype(np.float32))

st = simulate(PARAMS_DEFAULT, t, head=head, target=target,
              scene_present_array=np.ones(T),
              target_present_array=np.zeros(T),
              return_states=True, key=jax.random.PRNGKey(0))

# Read brain states
g_est_arr = np.asarray(st.brain.sm.g_est)
v_lin_arr = np.asarray(st.brain.sm.v_lin)
vs_L = np.asarray(st.brain.sm.vs_L)
vs_R = np.asarray(st.brain.sm.vs_R)
w_est_net = vs_L - vs_R                                    # (T, 3) yaw/pitch/roll

# Read sensory outputs (canal + delayed scene_slip) per timestep
theta = PARAMS_DEFAULT.sensory
q_head_arr = jnp.asarray(head.rot_pos, dtype=jnp.float32)
a_head_arr = jnp.asarray(head.lin_acc, dtype=jnp.float32)

def read_one(x_sens, q_h, a_h):
    out = sensory_model.read_outputs(x_sens, theta, q_h, a_h)
    return out.canal, out.scene_slip

canal_arr, slip_arr = jax.vmap(read_one)(st.sensory, q_head_arr, a_head_arr)
canal_arr = np.asarray(canal_arr)
slip_arr  = np.asarray(slip_arr)

# Canal-derived rotation estimate (what GE *would* see if using canal-only)
w_canal = (CANAL_PINV @ canal_arr.T).T   # (T, 3)

# Plant eye position and velocity
eye_L = np.asarray(st.plant.left)   # (T, 3) yaw/pitch/roll
eye_R = np.asarray(st.plant.right)
eye_version = (eye_L + eye_R) / 2.0
eye_vel = np.gradient(eye_version, DT, axis=0)   # (T, 3)

# Sample at key times
print(f'{"t":>5s} | '
      f'{"canal_p":>8s} {"w_est_p":>8s} {"slip_p":>8s} | '
      f'{"g[y]-G":>8s} {"g[z]":>8s} | '
      f'{"v_lin[z]":>9s} | '
      f'{"eye_v_p":>8s}')
print('  ' + '-' * 92)
for tt in [0.0, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    i = int(tt / DT)
    print(f'{tt:5.2f} | '
          f'{w_canal[i, 1]:8.3f} {w_est_net[i, 1]:8.3f} {slip_arr[i, 1]:8.3f} | '
          f'{g_est_arr[i, 1] - 9.81:8.4f} {g_est_arr[i, 2]:8.4f} | '
          f'{v_lin_arr[i, 2]:9.5f} | '
          f'{eye_vel[i, 1]:8.3f}')

# Time-integrated comparison: how much did w_est_net contain non-canal components?
print('\n--- Integrated w_est_net pitch (deg = ∫ω dt over 5s) ---')
mask_motion = (t > 0.5) & (t < 5.0)
print(f'∫w_canal[pitch] dt   = {np.trapezoid(w_canal[mask_motion, 1], t[mask_motion]):.3f} deg')
print(f'∫w_est_net[pitch] dt = {np.trapezoid(w_est_net[mask_motion, 1], t[mask_motion]):.3f} deg')
print(f'Difference (OKR contribution to gravity transport input) = '
      f'{np.trapezoid((w_est_net - w_canal)[mask_motion, 1], t[mask_motion]):.3f} deg')
