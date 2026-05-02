"""Diagnose why v_lin[y] grows during heave plateau."""

import numpy as np
import jax.numpy as jnp
import jax

from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, _IDX_GRAV
from oculomotor.sim import kinematics as km
from oculomotor.models.brain_models.brain_model import _IDX_HEAD, _IDX_VS, _IDX_VS_L, _IDX_VS_R
from oculomotor.models.sensory_models import sensory_model

DT = 0.001
T_TOTAL = 5.0
PEAK = 0.20
RAMP = 0.2
HOLD = 1.6
T_PULSE = 0.5
DEPTH = 0.4

t = np.arange(0.0, T_TOTAL, DT)
T = len(t)

# Trapezoid velocity envelope
t_rel = t - T_PULSE
env = np.zeros(T)
ramp_up = (t_rel >= 0) & (t_rel < RAMP)
plateau = (t_rel >= RAMP) & (t_rel < RAMP + HOLD)
ramp_dn = (t_rel >= RAMP + HOLD) & (t_rel < 2*RAMP + HOLD)
env[ramp_up] = t_rel[ramp_up] / RAMP
env[plateau] = 1.0
env[ramp_dn] = 1.0 - (t_rel[ramp_dn] - RAMP - HOLD) / RAMP

# Heave: vertical translation (axis=1, sign=+1)
head_vel = np.zeros((T, 3), dtype=np.float32)
head_vel[:, 1] = PEAK * env

head = km.build_kinematics(t, lin_vel=head_vel)

pt = np.tile([0.0, 0.0, DEPTH], (T, 1)).astype(np.float32)
target = km.build_target(t, lin_pos=pt)

st = simulate(PARAMS_DEFAULT, t, head=head, target=target,
              scene_present_array=np.ones(T),
              return_states=True, key=jax.random.PRNGKey(0))

# Extract delayed scene_linear_vel by reading sensory state outputs
# read_outputs needs theta, q_head, a_head per sample
theta = PARAMS_DEFAULT.sensory
q_head_arr = jnp.asarray(head.rot_pos, dtype=jnp.float32)
a_head_arr = jnp.asarray(head.lin_acc, dtype=jnp.float32)

def read_one(x_sens, q_h, a_h):
    out = sensory_model.read_outputs(x_sens, theta, q_h, a_h)
    return out.scene_linear_vel

scene_lin_vel_arr = jax.vmap(read_one)(st.sensory, q_head_arr, a_head_arr)
scene_lin_vel_arr = np.asarray(scene_lin_vel_arr)

# Brain state extraction
g_est = np.asarray(st.brain[:, _IDX_GRAV])[:, :3]
v_lin = np.asarray(st.brain[:, _IDX_HEAD])
vs_L = np.asarray(st.brain[:, _IDX_VS_L])
vs_R = np.asarray(st.brain[:, _IDX_VS_R])
vs_net = vs_L - vs_R

# Sample at key times
sample_t = [0.0, 0.6, 1.0, 1.5, 2.0, 2.4, 3.0, 4.0]
print(f'{"t":>5s}  {"v_head[y]":>10s}  {"v_lin[y]":>10s}  {"v_visual[y]":>12s}  {"scene_lv[y]":>12s}  {"a_est[y]":>10s}  {"g_est[y]-9.81":>14s}  {"vs_net":>10s}')
for tt in sample_t:
    i = int(tt / DT)
    vh = head_vel[i, 1]
    vl = v_lin[i, 1]
    slv = scene_lin_vel_arr[i, 1]
    vv = -slv
    ge = g_est[i, 1]
    aest = (PARAMS_DEFAULT.brain.tonic_acc * 0)  # placeholder, don't have it directly
    # a_est = gia - g_est. gia[y] = G0 (no a in y during plateau — only during ramps)
    # But during ramps, a_head[y] != 0. Compute from head.lin_acc:
    a_head_y = float(head.lin_acc[i, 1])
    gia_y = 9.81 + a_head_y  # head not rotated, so head-frame y component = world y
    aest_y = gia_y - ge
    vs_y = vs_net[i, 1]   # pitch component? actually [yaw, pitch, roll]
    print(f'{tt:5.2f}  {vh:10.4f}  {vl:10.4f}  {vv:12.4f}  {slv:12.4f}  {aest_y:10.4f}  {ge - 9.81:14.4f}  {vs_y:10.3f}')
