"""Diagnostic: VVOR stability vs K_vis and otolith.

Tests:
1. VVOR with K_vis=0.1 (current) — shows oscillations?
2. VVOR with K_grav=0 (no otolith correction) — does gravity estimate cause oscillations?
3. VVOR with K_vis=0.05 — proposed fix
4. OKN SS gain with K_vis=0.1 vs 0.05
"""
import numpy as np
import sys
sys.path.insert(0, "src")

from oculomotor.sim.simulator import simulate
from oculomotor.sim.stimuli   import rotation_step, scene_stationary, scene_dark
from oculomotor.models.brain_models.brain_model import BrainParams
from oculomotor.models.plant_models.plant_model_first_order import PlantParams
from oculomotor.models.sensory_models.sensory_model import SensoryParams
from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, with_sensory


def run_vvor(K_vis_val, K_grav_val=0.5, g_burst=0.0, label=""):
    """Run VVOR: 60 deg/s step rotation in lit stationary room for 30 s."""
    T = 30.0
    t, h = rotation_step(velocity_deg_s=60.0, rotate_dur=25.0, coast_dur=5.0)
    _, v_scene, sp = scene_stationary(T)

    p = with_brain(PARAMS_DEFAULT, K_vis=K_vis_val, K_grav=K_grav_val, g_burst=g_burst)
    states = simulate(p, t, head_vel_array=h,
                      v_scene_array=v_scene, scene_present_array=sp,
                      return_states=True)

    # VS net (yaw) and eye velocity
    from oculomotor.analysis import vs_net
    vs = vs_net(states)[:, 0]  # yaw

    # Eye velocity = derivative of eye position (left eye, yaw)
    eye_pos = states.plant[:, 0]
    dt = t[1] - t[0]
    eye_vel = np.gradient(eye_pos, dt)

    # Find oscillation: check peak-to-peak of VS in steady rotation window (5-25s)
    mask = (t > 5) & (t < 25)
    vs_range = vs[mask].max() - vs[mask].min()
    ev_range = eye_vel[mask].max() - eye_vel[mask].min()

    print(f"{label:40s}  VS pk-pk={vs_range:6.1f} deg/s  EyeVel pk-pk={ev_range:6.1f} deg/s")
    return t, vs, eye_vel


def run_okn(K_vis_val, label=""):
    """OKN: 30 deg/s scene motion 20 s on, 10 s off."""
    T = 30.0
    from oculomotor.sim.stimuli import scene_motion
    t, v_sc, sp = scene_motion(velocity_deg_s=30.0, on_dur=20.0, total_dur=T)

    p = with_brain(PARAMS_DEFAULT, K_vis=K_vis_val, g_burst=0.0)
    states = simulate(p, t, head_vel_array=np.zeros((len(t), 3), dtype=np.float32),
                      v_scene_array=v_sc, scene_present_array=sp,
                      return_states=True)

    eye_pos = states.plant[:, 0]
    dt = t[1] - t[0]
    eye_vel = np.gradient(eye_pos, dt)

    # SS OKN: mean eye velocity in 15-19 s window (before scene off)
    mask = (t > 15) & (t < 19)
    ss_vel = np.mean(eye_vel[mask])
    gain = ss_vel / 30.0

    print(f"{label:40s}  OKN SS gain = {gain:.3f}  (target ≈ 0.82)")
    return gain


print("=" * 70)
print("VVOR stability (head 60 deg/s, lit room; no saccades)")
print("  Healthy VVOR: VS should be ~60 deg/s flat, eye vel ~−60 deg/s flat")
print("  Oscillations appear as large pk-pk values")
print("=" * 70)

run_vvor(0.10, label="K_vis=0.10  K_grav=0.50  (current)")
run_vvor(0.10, K_grav_val=0.0, label="K_vis=0.10  K_grav=0.00  (no otolith corr)")
run_vvor(0.05, label="K_vis=0.05  K_grav=0.50  (half K_vis)")
run_vvor(0.02, label="K_vis=0.02  K_grav=0.50  (low K_vis)")
run_vvor(0.10, K_grav_val=0.5, g_burst=700.0, label="K_vis=0.10  saccades on    (with fast phases)")

print()
print("=" * 70)
print("OKN steady-state gain (30 deg/s scene motion, no saccades)")
print("=" * 70)
run_okn(0.10, label="K_vis=0.10")
run_okn(0.05, label="K_vis=0.05")
run_okn(0.02, label="K_vis=0.02")
