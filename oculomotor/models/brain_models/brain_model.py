"""Brain model — velocity storage, neural integrator, saccade generator, efference copy.

Aggregates velocity_storage, neural_integrator, saccade_generator, and
efference_copy into a single SSM with one state vector and one step() function.

Signal flow:
    y_canals         (6,)   canal afferents         → VS
    raw_slip_delayed (3,)   delayed raw retinal slip → VS (after EC correction)
    e_cmd            (3,)   motor error command      → SG

Efference copy correction (inside step, before VS):
    u_burst_delayed  = ec.read_delayed(x_ec)          from current EC state
    e_slip_corrected = raw_slip_delayed − u_burst_delayed
    VS receives e_slip_corrected

Internal flow:
    VS  →  w_est  →  u_vel (with u_burst)  →  NI  →  u_p  →  (plant)
    SG  →  u_burst  (saccade velocity command → EC delay → next-step correction)
    EC  →  delays u_burst by tau_vis for next-step slip cancellation

State vector  x_brain = [x_vs (3) | x_ni (3) | x_sg (9) | x_ec (120)]  — N_STATES = 135

Index constants (relative to x_brain):
    _IDX_VS   — velocity storage states  (3,)
    _IDX_NI   — neural integrator states (3,)
    _IDX_SG   — saccade generator states (9,)
    _IDX_EC   — efference copy states    (120,)

Outputs of step():
    dx_brain  (135,)  state derivative
    u_p       (3,)    pulse-step motor command → plant
    u_burst   (3,)    saccade burst velocity command
"""

from typing import NamedTuple

import jax.numpy as jnp

from oculomotor.models.brain_models import velocity_storage as vs
from oculomotor.models.brain_models import neural_integrator as ni
from oculomotor.models.brain_models import saccade_generator as sg
from oculomotor.models.brain_models import efference_copy as ec
from oculomotor.models.sensory_models.sensory_model import SensoryOutput  # noqa: F401 (re-exported)


# ── Brain parameters ────────────────────────────────────────────────────────────

class BrainParams(NamedTuple):
    """Learnable central parameters — fit to patient eye-movement data."""

    # Velocity storage — Raphan, Matsuo & Cohen (1979)
    tau_vs:                float = 20.0   # storage / OKAN TC (s); ~20 s monkey (Cohen 1977)
    K_vs:                  float = 0.1    # canal-to-VS gain (1/s); controls charging speed
    K_vis:                 float = 0.3    # visual-to-VS gain (1/s); OKR / OKAN charging
    g_vis:                 float = 0.3    # visual feedthrough (unitless); fast OKR onset

    # Neural integrator — Robinson (1975)
    tau_i:                 float = 25.0   # leak TC (s); healthy >20 s (Cannon & Robinson 1985)
    tau_p:                 float = 0.15   # plant TC copy — NI feedthrough for lag cancellation
    tau_vis:               float = 0.08   # visual delay copy — EC delay must match retinal delay
                                          # should match PlantParams.tau_p in healthy subjects;
                                          # may differ in pathology (imperfect internal model)

    # Saccade generator — Robinson (1975) local-feedback burst model
    g_burst:               float = 700.0  # burst ceiling (deg/s); 0 disables saccades
    e_sat_sac:             float = 7.0    # main-sequence saturation (deg)
    k_sac:                 float = 200.0  # trigger sigmoid steepness (1/deg)
    threshold_sac:         float = 0.5    # retinal error trigger threshold (deg)
    threshold_stop:        float = 0.1    # burst-stop threshold (deg)
    threshold_sac_release: float = 0.4    # OPN latch release threshold
    tau_reset_fast:        float = 0.05   # inter-saccade x_copy reset TC (s)
    tau_ref:               float = 0.15   # refractory (OPN) decay TC (s); ~150 ms ISI
    tau_ref_charge:        float = 0.001  # OPN charge TC (s)
    k_ref:                 float = 50.0   # bistable OPN gate steepness (1/z_ref)
    threshold_ref:         float = 0.1    # OPN threshold
    tau_hold:              float = 0.005  # sample-and-hold tracking TC (s)
    tau_sac:               float = 0.001  # saccade latch TC (s)
    tau_acc:               float = 0.080  # accumulator rise TC (s)
    tau_drain:             float = 0.120  # accumulator drain TC (s)
    threshold_acc:         float = 0.5    # accumulator trigger threshold
    k_acc:                 float = 50.0   # accumulator sigmoid steepness

    # Orbital reset — centering saccade policy at orbital limit
    alpha_reset:           float = 1.0    # centering-saccade gain; 0=suppress, 1=active centering


# ── State layout ───────────────────────────────────────────────────────────────

N_STATES = vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES   # 3+3+9+120 = 135

# Index constants — relative to x_brain
_IDX_VS = slice(0,             vs.N_STATES)                                   # (3,)
_IDX_NI = slice(vs.N_STATES,   vs.N_STATES + ni.N_STATES)                    # (3,)
_IDX_SG = slice(vs.N_STATES + ni.N_STATES,
                vs.N_STATES + ni.N_STATES + sg.N_STATES)                      # (9,)
_IDX_EC = slice(vs.N_STATES + ni.N_STATES + sg.N_STATES,
                vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES)        # (120,)


# ── Step function ──────────────────────────────────────────────────────────────

def step(x_brain, sensory_out, brain_params):
    """Single ODE step for the brain subsystem.

    Args:
        x_brain:     (135,)        brain state [x_vs | x_ni | x_sg | x_ec]
        sensory_out: SensoryOutput bundled canal afferents + delayed visual signals
                       .canal         (6,)    canal afferent rates
                       .slip_delayed  (3,)    delayed retinal slip (no EC correction yet)
                       .pos_visible   (3,)    delayed position error, gated by visual field
                       .e_cmd         (3,)    motor error command for the saccade generator
                       .scene_present scalar  0=dark, 1=lit — gates EC slip correction
        brain_params: BrainParams   model parameters

    Returns:
        dx_brain:  (135,)  dx_brain/dt
        u_p:       (3,)    pulse-step motor command → plant
        u_burst:   (3,)    saccade burst velocity command
    """
    x_vs = x_brain[_IDX_VS]
    x_ni = x_brain[_IDX_NI]
    x_sg = x_brain[_IDX_SG]
    x_ec = x_brain[_IDX_EC]

    # ── Efference copy correction: add delayed burst to delayed slip ──────────
    # u_burst_delayed matches the phase of slip_delayed (both delayed by tau_vis).
    # Cancels burst-driven eye motion from retinal slip before VS.
    # Gated by scene_present: correction only applies when a visual scene is visible.
    u_burst_delayed  = ec.read_delayed(x_ec)
    e_slip_corrected = sensory_out.slip_delayed + sensory_out.scene_present * u_burst_delayed

    # ── Velocity storage: canal + corrected slip → velocity estimate ──────────
    dx_vs, w_est = vs.step(x_vs, jnp.concatenate([sensory_out.canal, e_slip_corrected]), brain_params)

    # ── Saccade generator ─────────────────────────────────────────────────────
    dx_sg, u_burst = sg.step(x_sg, sensory_out.e_cmd, brain_params)

    # ── Neural integrator: combined eye-velocity command ──────────────────────
    u_vel      = -w_est + u_burst
    dx_ni, u_p = ni.step(x_ni, u_vel, brain_params)

    # ── Efference copy: advance delay cascade with current burst ──────────────
    dx_ec, _ = ec.step(x_ec, u_burst, brain_params)

    # ── Pack state derivative ─────────────────────────────────────────────────
    dx_brain = jnp.concatenate([dx_vs, dx_ni, dx_sg, dx_ec])

    return dx_brain, u_p, u_burst
