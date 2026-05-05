"""Diagnostic: dump peak values from OCR cascade to find what's slamming to -50."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from oculomotor.sim.simulator import (PARAMS_DEFAULT, simulate, with_brain, with_sensory,
                                       SimConfig, _IDX_GRAV, _IDX_NI, _IDX_SG)
from oculomotor.sim import kinematics as km
from oculomotor.analysis import vs_net, ni_net

DT = 0.001
TILT_DEG = 30.0
TILT_VEL = 20.0
HOLD_T   = 30.0
tilt_dur = TILT_DEG / TILT_VEL

params = with_brain(
    with_sensory(PARAMS_DEFAULT,
                 sigma_canal=0.0, sigma_slip=0.0, sigma_pos=0.0, sigma_vel=0.0),
    sigma_acc=0.0,
)
t = np.arange(0.0, tilt_dur + HOLD_T, DT)
T = len(t)
hv_roll = np.zeros((T, 3))
hv_roll[t < tilt_dur, 2] = TILT_VEL  # roll axis
head_km = km.build_kinematics(t, rot_vel=hv_roll)

st = simulate(params, t,
              head=head_km,
              scene_present_array=np.ones(T),
              target_present_array=np.ones(T),
              sim_config=SimConfig(warmup_s=0.0),
              return_states=True)

eye_L_tor = np.array(st.plant[:, 2])
eye_R_tor = np.array(st.plant[:, 5])
ni_tor    = ni_net(st)[:, 2]
vs_tor    = vs_net(st)[:, 2]
g_est_x   = np.array(st.brain[:, _IDX_GRAV])[:, 0]
ocr_sig   = -float(params.brain.g_ocr) * g_est_x
# x_sg layout: [e_held(3) | z_opn | z_acc | z_trig | x_ebn_R(3) | x_ebn_L(3) | x_ibn_R(3) | x_ibn_L(3)]
x_sg = np.array(st.brain[:, _IDX_SG])
e_held_tor = x_sg[:, 2]   # e_held torsion component
z_acc      = x_sg[:, 4]
ebn_R_tor  = x_sg[:, 6 + 2]   # right EBN torsion
ebn_L_tor  = x_sg[:, 9 + 2]   # left EBN torsion

print(f"Tilt: {TILT_DEG}° at {TILT_VEL}°/s, hold {HOLD_T}s, lit + target on, noise OFF")
print(f"Tilt phase: 0 → {tilt_dur:.2f}s; hold: {tilt_dur:.2f} → {tilt_dur+HOLD_T:.2f}s")
print()
print(f"{'signal':<22}{'min':>10}{'max':>10}{'mean(last 5s)':>16}{'final':>10}")
print("-" * 68)
def stat(name, x):
    last5 = x[t > tilt_dur + HOLD_T - 5.0]
    print(f"{name:<22}{x.min():>10.3f}{x.max():>10.3f}{last5.mean():>16.3f}{x[-1]:>10.3f}")

stat('OCR signal',         ocr_sig)
stat('VS torsion',         vs_tor)
stat('NI net torsion',     ni_tor)
stat('Eye L torsion',      eye_L_tor)
stat('Eye R torsion',      eye_R_tor)
stat('e_held torsion',     e_held_tor)
stat('z_acc (sacc trig)',  z_acc)
stat('EBN_R torsion',      ebn_R_tor)
stat('EBN_L torsion',      ebn_L_tor)

# u_burst is what NI integrates. Reconstruct from EBN activations: act = g_burst*(1-exp(-relu(e)/e_sat))
gb = float(params.brain.g_burst)
es = float(params.brain.e_sat_sac)
def _act(x):
    xr = np.maximum(x, 0.0)
    return gb * (1.0 - np.exp(-xr / es))
act_R = _act(ebn_R_tor)
act_L = _act(ebn_L_tor)
u_burst_tor = act_R - act_L
stat('act EBN_R torsion',  act_R)
stat('act EBN_L torsion',  act_L)
stat('u_burst torsion',    u_burst_tor)

# Integrate u_burst over hold to estimate NI contribution from saccades alone
hold_mask = t > tilt_dur
u_burst_hold = u_burst_tor[hold_mask]
print(f"\nu_burst torsion integral over hold (deg) = {u_burst_hold.sum() * DT:.2f}")
print(f"  mean u_burst over hold = {u_burst_hold.mean():.3f} deg/s (positive→eye+, negative→eye−)")

print()
print("Argmax/argmin times (s relative to tilt-end):")
for name, x in [('eye_L', eye_L_tor), ('NI', ni_tor), ('VS', vs_tor)]:
    print(f"  {name}: argmin at t-tilt={t[x.argmin()]-tilt_dur:.3f}s (val {x.min():.2f}),"
          f" argmax at t-tilt={t[x.argmax()]-tilt_dur:.3f}s (val {x.max():.2f})")
