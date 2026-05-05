"""Diagnostic: VOR post-rotational TC sweep over tau_vs.

Measure post-rotational eye velocity decay TC after step rotation in dark.
Target: 15–20 s (Raphan-Cohen velocity-storage extension).
Currently measured at 35.5 s — too long.

Step rotation: 30°/s yaw, then sudden stop. Post-stop, eye velocity decays.
Fit single-exponential to log(spv) over the decay phase.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax.numpy as jnp
from oculomotor.sim.simulator import (PARAMS_DEFAULT, simulate, with_brain, with_sensory,
                                       SimConfig)
from oculomotor.sim import kinematics as km
from oculomotor.analysis import extract_spv_states, fit_tc

DT = 0.001
SPIN_VEL = 30.0
T_PRE   = 30.0   # constant rotation, let VS charge
T_POST  = 60.0   # post-rotation, in dark
TOTAL   = T_PRE + T_POST
t = np.arange(0.0, TOTAL, DT)
T = len(t)

# Square step: constant SPIN_VEL for [0, T_PRE), then 0
yaw_vel = np.where(t < T_PRE, SPIN_VEL, 0.0).astype(np.float32)
head_vel = np.stack([yaw_vel, np.zeros(T), np.zeros(T)], axis=1)
head_km = km.build_kinematics(t, rot_vel=head_vel)

print(f"VOR step: {SPIN_VEL}°/s yaw for {T_PRE}s, then stop. Dark.")
print(f"Target post-rotational TC: 15–20 s (Raphan-Cohen velocity storage extension)")
print()

# Format: (label, tau_vs, tau_vs_adapt)
combos = [
    ('tau_vs=20 adapt=600      ', 20.0,  600.0),  # current default
    ('tau_vs=20 adapt=6000     ', 20.0, 6000.0),  # effectively disabled
    ('tau_vs=20 adapt=60       ', 20.0,   60.0),  # PAN-like
    ('tau_vs=15 adapt=6000     ', 15.0, 6000.0),
    ('tau_vs=12 adapt=6000     ', 12.0, 6000.0),
    ('tau_vs=10 adapt=6000     ', 10.0, 6000.0),
    ('tau_vs=15 adapt=600      ', 15.0,  600.0),
    ('tau_vs=12 adapt=600      ', 12.0,  600.0),
]

print(f"{'config':<24}{'fit τ (s)':>12}{'eye vel @ stop':>16}"
      f"{'eye vel +5s':>14}{'eye vel +20s':>14}")
print("-" * 80)

for name, tau_vs, tau_adapt in combos:
    params = with_brain(
        with_sensory(PARAMS_DEFAULT,
                     sigma_canal=0.0, sigma_slip=0.0, sigma_pos=0.0, sigma_vel=0.0),
        tau_vs=tau_vs, tau_vs_adapt=tau_adapt, sigma_acc=0.0,
    )
    st = simulate(params, t, head=head_km,
                  scene_present_array=np.zeros(T),
                  target_present_array=np.zeros(T),
                  sim_config=SimConfig(warmup_s=0.0),
                  return_states=True)
    spv = extract_spv_states(st, t)[:, 0]   # yaw axis

    # Fit exponential decay over [T_PRE+1s, T_PRE+30s]
    fit_start = T_PRE + 1.0
    fit_end   = T_PRE + 30.0
    tau_fit, _, _ = fit_tc(t, np.abs(spv), fit_start, fit_end)
    if tau_fit is None: tau_fit = float('nan')

    # Reference values
    spv_at_stop = float(spv[int(T_PRE / DT) - 1])
    spv_p5      = float(spv[int((T_PRE + 5.0) / DT)])
    spv_p20     = float(spv[int((T_PRE + 20.0) / DT)])

    print(f"{name:<24}{tau_fit:>12.2f}{spv_at_stop:>16.2f}"
          f"{spv_p5:>14.2f}{spv_p20:>14.2f}")
