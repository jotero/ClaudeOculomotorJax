"""Vergence diagnostic v2 — traces cascade input vs output vs geometric expectation.

Checks if the disparity cascade correctly reflects actual eye positions.
Key questions:
  1. Does the cascade INPUT (stage 0) show geometric disparity from plant positions?
  2. Does the cascade OUTPUT (stage 39) match the input with ~80ms delay?
  3. Is there a positive-feedback sign error somewhere?

Usage:
    python -X utf8 scripts/diag_vergence2.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import jax
import jax.numpy as jnp

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, simulate,
    with_brain, SimConfig
)
from oculomotor.sim import kinematics as km
from oculomotor.models.sensory_models.retina import _N_PER_SIG, N_STAGES
from oculomotor.models.sensory_models.retina import world_to_retina as _wtr

IPD    = 0.064
DEMAND_M = 0.40
VERG_DEMAND_DEG = 2.0 * np.degrees(np.arctan(IPD / 2.0 / DEMAND_M))
print(f"Vergence demand: {VERG_DEMAND_DEG:.3f} deg")

# target_disparity in cyclopean is a 1-pole LP — input and output are the same
# 3-element buffer.  No need for stage-0 readout matrix; just read the field.

t = np.arange(0.0, 4.0, 0.001)
T = len(t)
pt = np.tile([0.0, 0.0, DEMAND_M], (T, 1))

p = with_brain(PARAMS_DEFAULT, CA_C=0.0, AC_A=0.0)
cfg = SimConfig(warmup_s=1.0)   # short warmup so we can see the dynamics

print("Running simulation (CA_C=0, AC_A=0, warmup=1s)...")
st = simulate(p, t, target=km.build_target(t, lin_pos=pt),
              scene_present_array=np.ones(T),
              return_states=True, key=jax.random.PRNGKey(0),
              sim_config=cfg)

eye_L    = np.array(st.plant.left[:, 0])   # left eye yaw (deg)
eye_R    = np.array(st.plant.right[:, 0])   # right eye yaw (deg)
vergence = eye_L - eye_R              # positive = converged

# 1-pole LP target_disparity buffer — input and output identical
disp_del = np.array(st.brain.pc.target_disparity)   # (T, 3)
disp_in  = disp_del                                  # same for 1-pole

# Geometric disparity: computed directly from actual plant positions
# For target at [0, 0, DEMAND_M], no head rotation, IPD = 0.064m
def geom_disp(yaw_L_deg, yaw_R_deg):
    """Compute expected retinal disparity from eye yaw positions."""
    ipd_half = IPD / 2.0
    tgt = jnp.array([0.0, 0.0, DEMAND_M])
    q_head = jnp.zeros(3)
    w_head = jnp.zeros(3)
    x_head = jnp.zeros(3)
    w_eye  = jnp.zeros(3)
    w_scn  = jnp.zeros(3)
    v_scn  = jnp.zeros(3)
    dp_dt  = jnp.zeros(3)

    q_L = jnp.array([yaw_L_deg, 0.0, 0.0])
    q_R = jnp.array([yaw_R_deg, 0.0, 0.0])
    eye_off_L = jnp.array([-ipd_half, 0.0, 0.0])
    eye_off_R = jnp.array([ ipd_half, 0.0, 0.0])

    pos_L, _, _, _, _, _ = _wtr(tgt, eye_off_L, q_head, w_head, x_head,
                                  q_L, w_eye, w_scn, v_scn, dp_dt,
                                  1.0, 1.0, 90.0, 1.0)
    pos_R, _, _, _, _, _ = _wtr(tgt, eye_off_R, q_head, w_head, x_head,
                                  q_R, w_eye, w_scn, v_scn, dp_dt,
                                  1.0, 1.0, 90.0, 1.0)
    return float(pos_L[0] - pos_R[0])

geom = np.vectorize(geom_disp)(eye_L, eye_R)

# Print detailed trace
print()
print(f"{'t':>6} | {'verg':>7} | {'disp_in[0]':>11} | {'disp_del[0]':>12} | {'geom_disp':>10} | {'x_verg[0]':>10}")
print("-" * 75)
show_idx = [0, 50, 100, 200, 499, 999, 1999, 3999]
for i in show_idx:
    if i >= T: continue
    xv = float(st.brain.va.verg_fast[i, 0])
    print(f"{t[i]:6.3f} | {vergence[i]:7.3f} | {disp_in[i,0]:11.4f} | {disp_del[i,0]:12.4f} | {geom[i]:10.4f} | {xv:10.4f}")

print()
print("--- NPC gate check at key vergence values ---")
for i in [0, 100, 499, 999]:
    if i >= T: continue
    xv    = float(st.brain.va.verg_fast[i, 0])
    verg  = vergence[i]
    gd    = geom[i]
    raw_d = disp_in[i, 0]      # approximate cascade input
    total = gd + xv
    print(f"  t={t[i]:.3f}: verg={verg:.2f}° x_verg={xv:.2f}° geom_disp={gd:.3f}° "
          f"total_demand={total:.3f}°  npc=50° -> gate={1.0/(1.0+np.exp(-100*(50-total))):.4f}")

print()
print(f"SS (t>3s): vergence={vergence[3000:].mean():.3f}° ±{vergence[3000:].std():.4f}°")
print(f"Expected SS vergence ≈ 8.38°  (SS error from demand: ~0.77°)")
print(f"disp_del SS: {disp_del[3000:,0].mean():.4f}°  (expected ≈ +0.77°)")
print(f"geom_disp SS: {geom[3000:].mean():.4f}°  (expected ≈ +0.77° at SS)")
