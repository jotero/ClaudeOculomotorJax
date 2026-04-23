"""Diagnostic: isolate OKR/VVOR oscillation sources.

1. VOR in dark (no visual) — sanity check that canal→VS is correct
2. OKR only (head still, scene moves) — isolate visual→VS pathway
3. VVOR (head + scene) — combined response
"""
import numpy as np
import sys
sys.path.insert(0, "src")

from oculomotor.sim.simulator import simulate, PARAMS_DEFAULT, with_brain, with_sensory
from oculomotor.sim.stimuli   import rotation_sinusoid, rotation_step, scene_dark, scene_stationary, scene_motion
from oculomotor.analysis      import vs_net

DT = 0.001
FREQ = 0.1   # Hz
AMP  = 60.0  # deg/s peak
TOTAL = 30.0

t_sin, h_sin = rotation_sinusoid(amplitude_deg_s=AMP, frequency_hz=FREQ, duration=TOTAL)
_, v_dark, sp_dark = scene_dark(TOTAL)
_, v_still, sp_still = scene_stationary(TOTAL)
h_zero = np.zeros((len(t_sin), 3), dtype=np.float32)

# ── Scenario A: VOR in dark ─────────────────────────────────────────────────
p_a = with_brain(PARAMS_DEFAULT, g_burst=0.0)
st_a = simulate(p_a, t_sin, head_vel_array=h_sin,
                v_scene_array=v_dark, scene_present_array=sp_dark,
                return_states=True)
vs_a = vs_net(st_a)[:, 0]
ev_a = np.gradient((st_a.plant[:, 0] + st_a.plant[:, 3]) / 2.0, DT)

# ── Scenario B: OKR only (head still, scene moves sinusoidally) ─────────────
v_sc_b = np.column_stack([
    AMP * np.sin(2*np.pi*FREQ*t_sin), np.zeros(len(t_sin)), np.zeros(len(t_sin))
]).astype(np.float32)
sp_b = np.ones(len(t_sin), dtype=np.float32)

p_b = with_brain(PARAMS_DEFAULT, g_burst=0.0)
st_b = simulate(p_b, t_sin, head_vel_array=h_zero,
                v_scene_array=v_sc_b, scene_present_array=sp_b,
                return_states=True)
vs_b = vs_net(st_b)[:, 0]
ev_b = np.gradient((st_b.plant[:, 0] + st_b.plant[:, 3]) / 2.0, DT)

# ── Scenario C: VVOR (head + stationary lit scene) ──────────────────────────
p_c = with_brain(PARAMS_DEFAULT, g_burst=0.0)
st_c = simulate(p_c, t_sin, head_vel_array=h_sin,
                v_scene_array=v_still, scene_present_array=sp_still,
                return_states=True)
vs_c = vs_net(st_c)[:, 0]
ev_c = np.gradient((st_c.plant[:, 0] + st_c.plant[:, 3]) / 2.0, DT)

# ── Print summary ────────────────────────────────────────────────────────────
mask = t_sin > 10.0  # skip warm-up
h_std  = np.std(h_sin[mask, 0])

print("=" * 70)
print("0.1 Hz sinusoidal test   (head or scene vel = 60 deg/s peak)")
print(f"  Head vel std (signal) = {h_std:.1f} deg/s")
print("=" * 70)
print(f"{'Scenario':<35s} {'VS std':>7s}  {'EyeVel std':>10s}  {'VS/Head ratio':>13s}")
print("-" * 70)

for label, vs, ev in [
    ("A: VOR dark        (canal→VS)",    vs_a, ev_a),
    ("B: OKR only        (visual→VS)",   vs_b, ev_b),
    ("C: VVOR            (canal+visual)", vs_c, ev_c),
]:
    vs_std = np.std(vs[mask])
    ev_std = np.std(ev[mask])
    ratio  = vs_std / (h_std + 1e-9)
    print(f"  {label:<33s} {vs_std:>7.1f}  {ev_std:>10.1f}  {ratio:>13.2f}")

print()
print("Per-frequency check (amplitude at 0.1 Hz via FFT):")
N = int(np.sum(mask))
for label, sig in [("A VS", vs_a[mask]), ("B VS", vs_b[mask]),
                   ("C VS", vs_c[mask]), ("A EV", ev_a[mask]),
                   ("C EV", ev_c[mask])]:
    fft = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(N, DT)
    idx = np.argmin(np.abs(freqs - FREQ))
    amp_at_f = 2 * np.abs(fft[idx]) / N
    print(f"  {label}: amplitude at {FREQ} Hz = {amp_at_f:.1f} deg/s")

# ── Quick OKR transient check: step scene, no head, no saccades ────────────
print()
print("=" * 70)
print("Step OKN check: 30 deg/s scene, head still, no saccades")
print("  Expect: eye_vel_initial ≈ 0 → builds up, VS drives it negative (OKR)")
print("=" * 70)
T2 = 5.0
t2, v_sc2, sp2 = scene_motion(30.0, on_dur=5.0, total_dur=T2)
h2 = np.zeros((len(t2), 3), dtype=np.float32)
p2 = with_brain(PARAMS_DEFAULT, g_burst=0.0)
st2 = simulate(p2, t2, head_vel_array=h2, v_scene_array=v_sc2,
               scene_present_array=sp2, return_states=True)
vs2 = vs_net(st2)[:, 0]
ev2 = np.gradient((st2.plant[:, 0] + st2.plant[:, 3]) / 2.0, DT)
ep2 = (st2.plant[:, 0] + st2.plant[:, 3]) / 2.0

print(f"{'t':>5s}  {'head_vel':>9s}  {'VS_net':>8s}  {'eye_pos':>8s}  {'eye_vel':>8s}")
for ti in [0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 4.9]:
    idx = int(ti / DT)
    print(f"{ti:5.1f}  {h2[idx,0]:9.2f}  {vs2[idx]:8.2f}  {ep2[idx]:8.2f}  {ev2[idx]:8.2f}")

# ── Test with K_vis varied ─────────────────────────────────────────────────
print()
print("=" * 70)
print("K_vis sensitivity (A=VOR dark, C=VVOR — VS amplitude at 0.1 Hz)")
print("=" * 70)
for kv in [0.02, 0.05, 0.10, 0.20, 0.50]:
    p_kv = with_brain(PARAMS_DEFAULT, K_vis=kv, g_burst=0.0)
    st_kv_a = simulate(p_kv, t_sin, head_vel_array=h_sin,
                       v_scene_array=v_dark, scene_present_array=sp_dark,
                       return_states=True)
    st_kv_c = simulate(p_kv, t_sin, head_vel_array=h_sin,
                       v_scene_array=v_still, scene_present_array=sp_still,
                       return_states=True)
    vs_kv_a = vs_net(st_kv_a)[:, 0][mask]
    vs_kv_c = vs_net(st_kv_c)[:, 0][mask]
    fft_a = np.fft.rfft(vs_kv_a)
    fft_c = np.fft.rfft(vs_kv_c)
    freqs = np.fft.rfftfreq(len(vs_kv_a), DT)
    idx_f = np.argmin(np.abs(freqs - FREQ))
    amp_a = 2 * np.abs(fft_a[idx_f]) / len(vs_kv_a)
    amp_c = 2 * np.abs(fft_c[idx_f]) / len(vs_kv_c)
    # Also check for oscillation at other frequencies (spurious power)
    idx_osc = np.argmax(np.abs(fft_c[:len(fft_c)//2]))  # dominant freq in VVOR
    dom_freq = freqs[idx_osc]
    dom_amp  = 2 * np.abs(fft_c[idx_osc]) / len(vs_kv_c)
    print(f"  K_vis={kv:.2f}: VOR-dark VS={amp_a:5.1f}  VVOR VS={amp_c:5.1f}  "
          f"dominant_freq={dom_freq:.3f} Hz (amp={dom_amp:.1f})")
