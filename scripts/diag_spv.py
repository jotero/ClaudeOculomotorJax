"""Quick SPV diagnostic — print eye position trace + raw velocity for sway LIT."""

import numpy as np
import jax

from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate
from oculomotor.sim import kinematics as km
from oculomotor.analysis import extract_z_opn

DT = 0.001
PEAK = 0.20
RAMP = 0.2
HOLD = 1.6
T_PULSE = 0.5
T_TOTAL = 5.0
DEPTH = 0.4

t = np.arange(0.0, T_TOTAL, DT)
T = len(t)

t_rel = t - T_PULSE
env = np.zeros(T)
env[(t_rel >= 0) & (t_rel < RAMP)] = t_rel[(t_rel >= 0) & (t_rel < RAMP)] / RAMP
env[(t_rel >= RAMP) & (t_rel < RAMP + HOLD)] = 1.0
env[(t_rel >= RAMP + HOLD) & (t_rel < 2*RAMP + HOLD)] = 1.0 - (t_rel[(t_rel >= RAMP + HOLD) & (t_rel < 2*RAMP + HOLD)] - RAMP - HOLD) / RAMP

# Sway, LIT scene only no target
head_vel = np.zeros((T, 3), dtype=np.float32)
head_vel[:, 0] = PEAK * env
head = km.build_kinematics(t, lin_vel=head_vel)
pt = np.tile([0.0, 0.0, DEPTH], (T, 1)).astype(np.float32)
target = km.build_target(t, lin_pos=pt)

st = simulate(PARAMS_DEFAULT, t, head=head, target=target,
              scene_present_array=np.ones(T),
              target_present_array=np.zeros(T),
              return_states=True, key=jax.random.PRNGKey(0))

eye_L = np.array(st.plant[:, :3])
z_opn = extract_z_opn(st)
vel_yaw_raw = np.gradient(eye_L[:, 0], DT)

# Sample at key times
print(f'{"t":>5s}  {"eye_yaw":>10s}  {"raw_vel":>10s}  {"z_opn":>8s}')
for tt in np.arange(0.0, 5.0, 0.25):
    i = int(tt / DT)
    print(f'{tt:5.2f}  {eye_L[i, 0]:10.4f}  {vel_yaw_raw[i]:10.4f}  {z_opn[i]:8.2f}')

# Check if any saccades fire (z_opn drops below 50)
fast_count = (z_opn < 50).sum()
print(f'\nz_opn < 50 (saccade firing): {fast_count} samples ({fast_count*DT*1000:.0f} ms total)')
print(f'z_opn min: {z_opn.min():.2f}, max: {z_opn.max():.2f}')
