"""Run OKN (C/D) and VVOR (E/F) exactly as in the Raphan notebook and
print the signal chain at onset to locate the oscillation source."""
import numpy as np
import sys
sys.path.insert(0, "src")
import jax
import jax.numpy as jnp

from oculomotor.sim.simulator import simulate, PARAMS_DEFAULT as THETA, _IDX_EC, _IDX_VIS
from oculomotor.analysis import vs_net as _vs_net3
from oculomotor.models.brain_models import efference_copy as ec_mod
from oculomotor.models.sensory_models.sensory_model import C_slip
from oculomotor.sim import stimuli as stim_mod

DT = 1.0 / 200.0
BASELINE = 1.0
n_base = int(BASELINE / DT)
t_base = jnp.arange(-n_base, 0) * DT

# ── C/D: OKN + OKAN ─────────────────────────────────────────────────
V_CD  = 30.0
ON_CD = 30.0
OFF_CD = 50.0
t_cd_stim = jnp.arange(0.0, ON_CD + OFF_CD, DT)
t_cd = jnp.concatenate([t_base, t_cd_stim])
T_cd = len(t_cd)
v_sc = jnp.zeros((T_cd, 3)).at[:, 0].set(
    jnp.where((t_cd >= 0.0) & (t_cd < ON_CD), V_CD, 0.0))
sc_pres_cd = jnp.where((t_cd >= 0.0) & (t_cd < ON_CD), 1.0, 0.0)

print("Running C/D OKN...")
st_cd = simulate(THETA, t_cd,
                 v_scene_array=v_sc,
                 scene_present_array=sc_pres_cd,
                 target_present_array=jnp.zeros(T_cd),
                 max_steps=int((BASELINE+ON_CD+OFF_CD)/0.001)+500,
                 return_states=True)
t_cd_np = np.array(t_cd)

# Extract signals
vs_cd   = _vs_net3(st_cd)[:, 0]
ep_cd   = (np.array(st_cd.plant)[:, 0] + np.array(st_cd.plant)[:, 3]) / 2.0
ev_cd   = np.gradient(ep_cd, t_cd_np[1]-t_cd_np[0])

x_ec_traj    = np.array(st_cd.brain)[:, _IDX_EC]
motor_ec_cd  = np.array(jax.vmap(ec_mod.read_delayed)(x_ec_traj))[:, 0]
x_vis_L      = np.array(st_cd.sensory)[:, _IDX_VIS]
slip_raw_cd  = (np.array(C_slip) @ x_vis_L.T)[0, :]

slip_corr_cd = slip_raw_cd + motor_ec_cd

# Print onset (0-3s)
print("\n=== C/D OKN onset signal chain (30 deg/s scene rightward, no head) ===")
print(f"{'t':>5s}  {'VS_net':>8s}  {'slip_raw':>9s}  {'slip_corr':>10s}  {'motor_ec':>9s}  {'eye_vel':>8s}")
mask = (t_cd_np >= 0.0) & (t_cd_np <= 3.0)
t_sub = t_cd_np[mask]
for ti in np.arange(0.0, 3.05, 0.1):
    idx = np.argmin(np.abs(t_sub - ti))
    m_idx = np.where(mask)[0][idx]
    print(f"{ti:5.1f}  {vs_cd[m_idx]:8.2f}  {slip_raw_cd[m_idx]:9.2f}  "
          f"{slip_corr_cd[m_idx]:10.2f}  {motor_ec_cd[m_idx]:9.2f}  {ev_cd[m_idx]:8.2f}")

# Check oscillation: look for frequency content in VS and slip_corr at onset
print("\n--- Frequency analysis of first 10 s of C/D (detect oscillation frequency) ---")
mask10 = (t_cd_np >= 0.0) & (t_cd_np <= 10.0)
for name, sig in [("VS_net", vs_cd), ("slip_raw", slip_raw_cd), ("slip_corr", slip_corr_cd)]:
    s = sig[mask10]
    N = len(s)
    fft = np.abs(np.fft.rfft(s - s.mean()))
    freqs = np.fft.rfftfreq(N, DT)
    # Find top 3 peaks
    top3 = np.argsort(fft)[-3:][::-1]
    peaks = [(freqs[i], fft[i]) for i in top3 if freqs[i] > 0.1]
    print(f"  {name:12s}: dominant freqs = {[(f'{f:.2f}Hz', f'{a:.1f}') for f,a in peaks]}")

# ── E/F: VVOR ────────────────────────────────────────────────────────
V_EF   = 30.0
ROT_EF = 30.0
CST_EF = 50.0
t_ef_stim, hv_ef_stim = stim_mod.rotation_step(V_EF, rotate_dur=ROT_EF, coast_dur=CST_EF, dt=DT)
t_ef = jnp.concatenate([t_base, jnp.array(t_ef_stim)])
hv_ef = jnp.concatenate([jnp.zeros((n_base, 3)), jnp.array(hv_ef_stim)])
T_ef  = len(t_ef)
sc_pres_ef = jnp.where((t_ef >= 0.0) & (t_ef < ROT_EF), 1.0, 0.0)

print("\n\nRunning E/F VVOR...")
st_ef = simulate(THETA, t_ef,
                 head_vel_array=hv_ef,
                 scene_present_array=sc_pres_ef,
                 target_present_array=jnp.zeros(T_ef),
                 max_steps=int((BASELINE+ROT_EF+CST_EF)/0.001)+500,
                 return_states=True)
t_ef_np = np.array(t_ef)

vs_ef = _vs_net3(st_ef)[:, 0]
ep_ef = (np.array(st_ef.plant)[:, 0] + np.array(st_ef.plant)[:, 3]) / 2.0
ev_ef = np.gradient(ep_ef, t_ef_np[1]-t_ef_np[0])
x_vis_L_ef  = np.array(st_ef.sensory)[:, _IDX_VIS]
slip_raw_ef = (np.array(C_slip) @ x_vis_L_ef.T)[0, :]
x_ec_ef     = np.array(st_ef.brain)[:, _IDX_EC]
motor_ec_ef = np.array(jax.vmap(ec_mod.read_delayed)(x_ec_ef))[:, 0]

print("\n=== E/F VVOR signal chain (5-30s steady rotation in light, 30 deg/s) ===")
print(f"{'t':>5s}  {'VS_net':>8s}  {'slip_raw':>9s}  {'eye_vel':>8s}  {'head_vel':>9s}")
hv_ef_np = np.array(hv_ef)[:, 0]
for ti in np.arange(5.0, 30.5, 5.0):
    idx = np.argmin(np.abs(t_ef_np - ti))
    print(f"{ti:5.1f}  {vs_ef[idx]:8.2f}  {slip_raw_ef[idx]:9.2f}  {ev_ef[idx]:8.2f}  {hv_ef_np[idx]:9.2f}")

# Frequency analysis during steady VVOR (5-30s)
print("\n--- E/F VVOR frequency analysis (5–30 s) ---")
mask_ef = (t_ef_np >= 5.0) & (t_ef_np <= 30.0)
for name, sig in [("VS_net", vs_ef), ("slip_raw", slip_raw_ef), ("eye_vel", ev_ef)]:
    s = sig[mask_ef] - sig[mask_ef].mean()
    N = len(s)
    fft = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(N, DT)
    top3 = np.argsort(fft)[-3:][::-1]
    peaks = [(freqs[i], fft[i]) for i in top3 if freqs[i] > 0.05]
    total_power = np.sum(fft[1:]**2)
    signal_power = np.sum(fft[freqs < 0.5]**2)
    print(f"  {name:12s}: dominant freqs = {[(f'{f:.2f}Hz', f'{a:.0f}') for f,a in peaks]}, "
          f"HF power fraction = {1-signal_power/total_power:.3f}")
