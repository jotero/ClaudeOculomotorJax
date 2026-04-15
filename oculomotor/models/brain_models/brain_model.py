"""Brain model — velocity storage, neural integrator, saccade generator, efference copy.

Aggregates velocity_storage, neural_integrator, saccade_generator, and
efference_copy into a single SSM with one state vector and one step() function.

Signal flow:
    y_canals         (6,)   canal afferents         → VS
    raw_slip_delayed (3,)   delayed raw retinal slip → VS (after EC correction)
    e_cmd            (3,)   motor error command      → SG

Efference copy correction (inside step, before VS):
    saccade_ec       = ec.read_delayed(x_ec)           from current EC state
    e_slip_corrected = raw_slip_delayed + scene_present * saccade_ec
    VS receives e_slip_corrected

Internal flow:
    VS  →  w_est  →  u_vel (with u_burst)  →  NI  →  motor_cmd  →  (plant)
    SG  →  u_burst  (saccade velocity command → EC delay → next-step correction)
    EC  →  delays u_burst by tau_vis for next-step slip cancellation

State vector  x_brain = [x_vs (3) | x_ni (3) | x_sg (9) | x_ec (120)]  — N_STATES = 135

Index constants (relative to x_brain):
    _IDX_VS   — velocity storage states  (3,)
    _IDX_NI   — neural integrator states (3,)
    _IDX_SG   — saccade generator states (9,)
    _IDX_EC   — efference copy states    (120,)

Outputs of step():
    dx_brain   (135,)  state derivative
    motor_cmd  (3,)    pulse-step motor command → plant
    u_burst    (3,)    saccade burst velocity command
"""

from typing import NamedTuple

import jax.numpy as jnp

from oculomotor.models.brain_models import velocity_storage    as vs
from oculomotor.models.brain_models import neural_integrator   as ni
from oculomotor.models.brain_models import saccade_generator   as sg
from oculomotor.models.brain_models import efference_copy      as ec
from oculomotor.models.brain_models import gravity_estimator   as ge
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

    # Otolith / gravity estimation — Laurens & Angelaki (2011, 2017)
    K_grav:                float = 0.5    # otolith correction gain (1/s); TC = 1/K_grav ≈ 2 s
    K_gd:                  float = 0.0    # gravity dumping gain (1/s); 0 = disabled
    g_ocr:                 float = 0.0    # OCR gain (dimensionless); 0 = disabled until verified


# ── State layout ───────────────────────────────────────────────────────────────

N_STATES = vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES + ge.N_STATES   # 3+3+9+120+3 = 138

# Index constants — relative to x_brain
_IDX_VS   = slice(0,             vs.N_STATES)                                       # (3,)
_IDX_NI   = slice(vs.N_STATES,   vs.N_STATES + ni.N_STATES)                        # (3,)
_IDX_SG   = slice(vs.N_STATES + ni.N_STATES,
                  vs.N_STATES + ni.N_STATES + sg.N_STATES)                          # (9,)
_IDX_EC   = slice(vs.N_STATES + ni.N_STATES + sg.N_STATES,
                  vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES)            # (120,)
_IDX_GRAV = slice(vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES,
                  vs.N_STATES + ni.N_STATES + sg.N_STATES + ec.N_STATES + ge.N_STATES)  # (3,)


def make_x0():
    """Default initial brain state — gravity estimator pointing down (upright head)."""
    x0 = jnp.zeros(N_STATES)
    return x0.at[_IDX_GRAV].set(ge.X0)


# ── Step function ──────────────────────────────────────────────────────────────

def step(x_brain, sensory_out, brain_params):
    """Single ODE step for the brain subsystem.

    Args:
        x_brain:     (138,)        brain state [x_vs | x_ni | x_sg | x_ec | x_grav]
        sensory_out: SensoryOutput bundled canal afferents + delayed visual signals
                       .canal         (6,)    canal afferent rates
                       .slip_delayed  (3,)    delayed retinal slip (no EC correction yet)
                       .pos_visible   (3,)    delayed position error, gated by visual field
                       .e_cmd         (3,)    motor error command for the saccade generator
                       .f_otolith     (3,)    specific force in head frame (m/s²)
                       .scene_present scalar  0=dark, 1=lit — gates EC slip correction
        brain_params: BrainParams   model parameters

    Returns:
        dx_brain:   (138,)  dx_brain/dt
        motor_cmd:  (3,)    pulse-step motor command → plant
    """
    x_vs   = x_brain[_IDX_VS]
    x_ni   = x_brain[_IDX_NI]
    x_sg   = x_brain[_IDX_SG]
    x_ec   = x_brain[_IDX_EC]
    x_grav = x_brain[_IDX_GRAV]

    # ── Efference copy correction ──────────────────────────────────────────────
    saccade_ec       = ec.read_delayed(x_ec)
    e_slip_corrected = sensory_out.slip_delayed + sensory_out.scene_present * saccade_ec

    # ── Velocity storage: canal + corrected slip + gravity estimate → ω̂ ───────
    # g_hat from the current state (one-step lag — standard in ODE evaluation)
    dx_vs, w_est = vs.step(
        x_vs,
        jnp.concatenate([sensory_out.canal, e_slip_corrected, x_grav]),
        brain_params)

    # ── Gravity estimator: cross-product transport + otolith correction ────────
    dx_grav, g_hat = ge.step(
        x_grav,
        jnp.concatenate([w_est, sensory_out.f_otolith]),
        brain_params)

    # ── Saccade generator ─────────────────────────────────────────────────────
    dx_sg, u_burst = sg.step(x_sg, sensory_out.e_cmd, brain_params)

    # ── OCR / somatogravic: gravity-driven eye position command ───────────────
    # g_hat = specific force (+x upright).  Tilt signals are normalised components:
    #   tilt_roll  = g_hat[1]/|g_hat| < 0 for CW head roll → eye rolls CCW (−z) ✓
    #   tilt_pitch = g_hat[2]/|g_hat| > 0 for nose-up tilt → eye pitches down (−y) ✓
    # (g_ocr = 0 by default; set to ~0.3 to enable)
    g_norm  = jnp.linalg.norm(g_hat) + 1e-9
    ocr_pos = brain_params.g_ocr * (180.0 / jnp.pi) * jnp.array([
        0.0,
        -g_hat[2] / g_norm,   # pitch: nose-up tilt → eye pitches down (−y)
         g_hat[1] / g_norm,   # roll:  CW tilt (g_hat[1]<0) → eye CCW (−z)
    ])

    # ── Neural integrator ─────────────────────────────────────────────────────
    dx_ni, motor_cmd_ni = ni.step(x_ni, -w_est + u_burst, brain_params)

    # Add OCR position offset directly to motor command (bypasses NI leak)
    motor_cmd = motor_cmd_ni + ocr_pos

    # ── Efference copy: advance delay cascade ─────────────────────────────────
    dx_ec, _ = ec.step(x_ec, u_burst, brain_params)

    # ── Pack state derivative ─────────────────────────────────────────────────
    dx_brain = jnp.concatenate([dx_vs, dx_ni, dx_sg, dx_ec, dx_grav])

    return dx_brain, motor_cmd
