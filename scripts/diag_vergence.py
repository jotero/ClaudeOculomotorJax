"""Vergence feedback diagnostic — traces disparity, x_verg, u_verg, plant positions.

Minimal test: binocular target at 0.4 m (both eyes), no head movement, no scene.
CA_C=0 and AC_A=0 to isolate pure vergence loop.

Expected:
  - Vergence (eye_L - eye_R) should reach ~9.14 deg (geometric for 40 cm, IPD=64 mm)
  - x_verg (horizontal vergence state) should reach ~8.9 deg at SS
  - acc_plant should reach ~2.5 D (accommodation demand = 1/0.4 = 2.5 D)

Usage:
    python -X utf8 scripts/diag_vergence.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import jax

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, simulate, _IDX_VERG, _IDX_ACC,
    with_brain, SimConfig
)
from oculomotor.sim import kinematics as km

DEMAND_M = 0.40
IPD = 0.064
VERG_DEMAND_DEG = 2.0 * np.degrees(np.arctan(IPD / 2.0 / DEMAND_M))
ACC_DEMAND_D    = 1.0 / DEMAND_M

print(f"Target: {DEMAND_M} m")
print(f"Vergence demand: {VERG_DEMAND_DEG:.3f} deg")
print(f"Accommodation demand: {ACC_DEMAND_D:.3f} D")
print()

t = np.arange(0.0, 8.0, 0.001)
T = len(t)
pt = np.tile([0.0, 0.0, DEMAND_M], (T, 1))

p = with_brain(PARAMS_DEFAULT, CA_C=0.0, AC_A=0.0)

cfg = SimConfig(warmup_s=3.0)

print("Running simulation (CA_C=0, AC_A=0)...")
st = simulate(p, t, target=km.build_target(t, lin_pos=pt),
              scene_present_array=np.ones(T),
              return_states=True, key=jax.random.PRNGKey(0),
              sim_config=cfg)

eye_L    = np.array(st.plant[:, 0])   # left eye yaw (deg)
eye_R    = np.array(st.plant[:, 3])   # right eye yaw (deg)
vergence = eye_L - eye_R              # positive = converged

# Brain state: first element of _IDX_VERG is x_verg[0] (horizontal integrator)
x_verg0  = np.array(st.brain[:, _IDX_VERG.start])   # x_verg horizontal state
acc_plant= np.array(st.acc_plant[:, 0])

# Target disparity in brain state (delayed) — not directly stored, derive from sens
# Use the cyclopean cascade: C_target_disp @ x_vis
from oculomotor.models.sensory_models.sensory_model import C_target_disp, _IDX_VIS
x_vis    = np.array(st.sensory[:, _IDX_VIS])
disp_del = (C_target_disp @ x_vis.T).T   # (T, 3) — delayed disparity

print()
print(f"{'t':>6} | {'vergence':>9} | {'x_verg[0]':>10} | {'disp_del[0]':>12} | {'eye_L':>7} | {'eye_R':>7} | {'acc':>7}")
print("-" * 75)
for i in [499, 999, 1499, 1999, 2999, 4999, 6999]:
    if i >= T:
        continue
    print(f"{t[i]:6.2f} | {vergence[i]:9.3f} | {x_verg0[i]:10.3f} | "
          f"{disp_del[i,0]:12.4f} | {eye_L[i]:7.3f} | {eye_R[i]:7.3f} | {acc_plant[i]:7.4f}")

print()
print(f"SS (t>4s): vergence mean={vergence[4000:].mean():.3f}  std={vergence[4000:].std():.4f}")
print(f"Expected vergence: {VERG_DEMAND_DEG:.3f} deg")
print(f"SS acc: mean={acc_plant[4000:].mean():.4f}  (expected {ACC_DEMAND_D:.3f} D)")

# Also check: what is the INITIAL u_verg just after t=0?
# u_verg = x_verg + K_phasic * e_fus where e_fus uses delayed disparity
# At t=0 (start of recording, after warmup), disparity should be near 0 (settled)
print()
print(f"Initial conditions (t=0.5s):")
i05 = int(0.5 / 0.001)
print(f"  vergence  = {vergence[i05]:.4f} deg  (should be ~{VERG_DEMAND_DEG:.2f})")
print(f"  x_verg[0] = {x_verg0[i05]:.4f} deg")
print(f"  disp_del  = {disp_del[i05, 0]:.4f} deg  (should be ~0 after settling)")
print(f"  eye_L = {eye_L[i05]:.4f},  eye_R = {eye_R[i05]:.4f}")
print(f"  acc_plant = {acc_plant[i05]:.4f} D  (rising toward {ACC_DEMAND_D:.2f})")
