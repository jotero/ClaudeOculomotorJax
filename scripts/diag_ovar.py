"""Diagnostic: OVAR — peak v_lin and SPV modulation amplitude after the gating fix.

Tilted yaw rotation produces head-frame oscillating gravity → modulated VOR.
Ideal: v_lin stays small (no actual translation), SPV modulation amplitude scales
with sin(tilt). Canal gating should suppress K_lin during the constant rotation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax.numpy as jnp
from oculomotor.sim.simulator import (PARAMS_DEFAULT, simulate, with_brain, with_sensory,
                                       SimConfig, _IDX_GRAV)
from oculomotor.sim import kinematics as km
from oculomotor.models.brain_models.brain_model import _IDX_HEAD
from oculomotor.analysis import vs_net, extract_spv_states

DT = 0.001
SPIN_VEL = 60.0
TILTS = [10.0, 30.0, 60.0, 90.0]
T_TILT_START = 2.0
TILT_VEL = 30.0
T_ROT_START = 35.0
ROT_RAMP = 1.0
TOTAL = T_ROT_START + ROT_RAMP + 50.0

t = np.arange(0.0, TOTAL, DT)
T = len(t)

combos = [
    ('current (K_gd=2.86, K_grav=0.6) ',  0.1, 1.5, 1e6),
]

print(f"OVAR: yaw {SPIN_VEL}°/s spin in tilted body. Spin starts t={T_ROT_START}s.")
print()
print(f"Expected: SPV modulation amplitude ∝ sin(tilt). 90° should be max, 10° tiny.")
print()
print(f"{'tilt':<6}{'sin(tilt)':<11}{'K_gd':>10}{'SPV pk-pk':>11}{'SPV-mean':>11}"
      f"{'v_lin pk':>10}")
print("-" * 80)

for tilt_deg in TILTS:
    T_TILT_END = T_TILT_START + tilt_deg / TILT_VEL
    roll_vel = np.where((t >= T_TILT_START) & (t < T_TILT_END), TILT_VEL, 0.0)
    yaw_vel  = np.zeros(T)
    ramp_mask = (t >= T_ROT_START) & (t < T_ROT_START + ROT_RAMP)
    hold_mask = t >= T_ROT_START + ROT_RAMP
    yaw_vel[ramp_mask] = SPIN_VEL * (t[ramp_mask] - T_ROT_START) / ROT_RAMP
    yaw_vel[hold_mask] = SPIN_VEL
    head_vel = np.stack([yaw_vel, np.zeros(T), roll_vel], axis=1)
    head_km  = km.build_kinematics(t, rot_vel=head_vel, rot_pos_0=[0.0, 0.0, 0.0])

    # Sweep K_gd to test its effect on OVAR modulation across tilts
    for kgd in [2.86, 1.0, 0.5, 0.0]:
        params = with_brain(
            with_sensory(PARAMS_DEFAULT,
                         sigma_canal=0.0, sigma_slip=0.0, sigma_pos=0.0, sigma_vel=0.0),
            K_gd=kgd,
            sigma_acc=0.0,
        )
        name = f'K_gd={kgd:.2f}'
        st = simulate(params, t, head=head_km,
                      scene_present_array=np.zeros(T),
                      target_present_array=np.zeros(T),
                      sim_config=SimConfig(warmup_s=0.0),
                      return_states=True)
        v_lin = np.array(st.brain[:, _IDX_HEAD])
        # Use saccade-masked SPV (excludes quick-phase velocity)
        spv_full = extract_spv_states(st, t)[:, 0]   # yaw axis
        # Look at last 30s of constant rotation
        steady = t > (T_ROT_START + ROT_RAMP + 5.0)
        v_norm = np.linalg.norm(v_lin[steady], axis=1)
        spv = spv_full[steady]
        print(f"{tilt_deg:<6.0f}{np.sin(np.radians(tilt_deg)):<11.3f}"
              f"{kgd:>10.2f}{spv.max()-spv.min():>11.2f}{spv.mean():>11.2f}"
              f"{v_norm.max():>10.3f}")
    print()
