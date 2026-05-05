"""Diagnostic: T-VOR DARK gain across param combos.

Translational VOR gain (eye velocity / required compensatory velocity).
For sway pulse with target at distance D=tonic_verg→1m, peak head vel V=0.2 m/s:
  required eye angular velocity ≈ V/D rad/s = (V/D)*(180/π) deg/s
For V=0.2, D=1: required ≈ 11.46 deg/s.
DARK T-VOR gain = peak eye SPV / required SPV.
Paige & Tomko (1991): healthy DARK T-VOR ≈ 0.5–0.7.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax.numpy as jnp
from oculomotor.sim.simulator import (PARAMS_DEFAULT, simulate, with_brain, with_sensory,
                                       SimConfig)
from oculomotor.sim import kinematics as km
from oculomotor.models.brain_models.brain_model import _IDX_HEAD

DT = 0.001
PEAK = 0.20    # 20 cm/s
RAMP = 0.2
HOLD = 1.6
T_PULSE = 0.5
T_TOTAL = 8.0
DEPTH = 1.0    # 1 m (matches tonic_verg)

t = np.arange(0.0, T_TOTAL, DT)
T = len(t)

# Trapezoid envelope
t_rel = t - T_PULSE
env = np.zeros(T)
env[(t_rel >= 0) & (t_rel < RAMP)] = t_rel[(t_rel >= 0) & (t_rel < RAMP)] / RAMP
env[(t_rel >= RAMP) & (t_rel < RAMP + HOLD)] = 1.0
mask_d = (t_rel >= RAMP + HOLD) & (t_rel < 2*RAMP + HOLD)
env[mask_d] = 1.0 - (t_rel[mask_d] - RAMP - HOLD) / RAMP

# Sway: rightward head motion → leftward target → eye should rotate left (yaw negative)
head_vel = np.zeros((T, 3), dtype=np.float32)
head_vel[:, 0] = (PEAK * env).astype(np.float32)
head = km.build_kinematics(t, lin_vel=head_vel)

# Required eye velocity to keep target on retina:
# At plateau (V_head = +0.2 m/s sway), with target fixed at depth D:
# eye angular vel ≈ -V_head / D rad/s = -V_head/D * 180/pi deg/s
required_spv = -PEAK / DEPTH * (180.0 / np.pi)   # = -11.46 deg/s

print(f"Sway pulse: peak {PEAK*100} cm/s, target at {DEPTH} m")
print(f"Required eye SPV ≈ {required_spv:.2f} deg/s for ideal T-VOR (gain=1.0)")
print(f"Paige & Tomko (1991) DARK T-VOR gain target: 0.5–0.7")
print()

# Format: (label, K_grav, K_lin, tau_a_lin, tau_head)
combos = [
    ('Laurens 2011 (K=0.1, gate=5)  ', 0.6, 0.1,  1.5, 2.0),
    ('K=1.0 (over-tuned)            ', 0.5, 1.0,  1.5, 2.0),
    ('K=0.05 τ=0.5 (pre-Laurens)    ', 0.2, 0.05, 0.5, 2.0),
]

print(f"{'combo':<26}{'peak v_lin':>12}{'peak eye yaw':>15}"
      f"{'peak eye yaw vel':>18}{'gain':>10}")
print("-" * 81)

for name, kg, kl, ta, th in combos:
    # DARK: scene off, target off
    params = with_brain(
        with_sensory(PARAMS_DEFAULT,
                     sigma_canal=0.0, sigma_slip=0.0, sigma_pos=0.0, sigma_vel=0.0),
        K_grav=kg, K_lin=kl, tau_a_lin=ta, tau_head=th,
        sigma_acc=0.0,
    )
    pt = np.tile([0.0, 0.0, DEPTH], (T, 1)).astype(np.float32)
    target = km.build_target(t, lin_pos=pt)
    st = simulate(params, t, head=head, target=target,
                  scene_present_array=np.zeros(T),
                  target_present_array=np.zeros(T),
                  return_states=True,
                  sim_config=SimConfig(warmup_s=0.0))
    eye_yaw_L = np.array(st.plant[:, 0])
    eye_yaw_R = np.array(st.plant[:, 3])
    eye_yaw = (eye_yaw_L + eye_yaw_R) / 2.0
    eye_yaw_vel = np.gradient(eye_yaw, DT)
    v_lin = np.array(st.brain[:, _IDX_HEAD])

    plateau_mask = (t_rel >= RAMP + 0.3) & (t_rel < RAMP + HOLD)
    peak_v_lin = float(np.linalg.norm(v_lin[plateau_mask], axis=1).max())
    peak_eye_yaw = float(eye_yaw[plateau_mask].min())   # signed; will be negative
    peak_eye_yaw_vel = float(eye_yaw_vel[plateau_mask].min())   # negative
    gain = peak_eye_yaw_vel / required_spv

    print(f"{name:<26}{peak_v_lin:>12.4f}{peak_eye_yaw:>15.4f}"
          f"{peak_eye_yaw_vel:>18.3f}{gain:>10.3f}")
