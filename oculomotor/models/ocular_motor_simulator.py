"""Full oculomotor simulator: Canal Array → VS → NI → Plant, with OKR.

Signal flow (3-D):

    VOR pathway:
        w_head (3,) → [Canal Array] → y_canals (6,)
                    → PINV_SENS (3×6) → u_canal (3,)

    OKR pathway  [see models/visual_delay.py and models/okr.py]:
        w_scene   (3,)   scene angular velocity
        w_eye     (3,)   eye angular velocity (from plant)
        e_slip    (3,)   retinal slip = w_scene − w_head − w_eye
                         ≈ 0 in dark VOR (w_eye ≈ −w_head when compensating)
                         ≠ 0 during OKN (w_scene ≠ 0, w_head = w_eye = 0)
        x_vis     (12,)  visual delay cascade state (4 stages × 3 axes)
        e_delayed (3,)   = C · x_vis  — delayed retinal slip
        u_okr     (3,)   = D · e_delayed  — OKR VS drive

    Combined VS input:
        u_vs = u_canal − u_okr

    Downstream:
        u_vs → [VS] → x_vs (3,) slow storage state
        u_ni = x_vs + u_vs               (Robinson/Raphan feedthrough)
        u_ni → [NI] → x_ni (3,)         position command
        u_p  = C·x_ni + D·u_ni          pulse-step
        u_p  → [Plant] → q (3,)         eye rotation vector (deg)

Global state vector (27 states):
    x = [x1_c0..x1_c5 | x2_c0..x2_c5 | x_vs (3) | x_ni (3) | x_p (3) | x_vis (12) | x_okr (3)]
         ──────── canal (12) ──────────   ── VS ──   ── NI ──   plant    vis delay     OKR store

Head velocity input:
    Accepts 1-D (T,) horizontal-only array — padded to (T, 3) internally.
    Accepts 3-D (T, 3) array directly for full 3-D stimulation.
    Accepts a Stimulus object (oculomotor.sim.stimulus) for full 6-DOF + visual input.

Scene angular velocity input:
    Accepts (T, 3) array or None (dark = zeros).
    Or pass a Stimulus object that carries w_scene.

Output:
    simulate() returns eye rotation vector, shape (T, 3).
    Use readout module to convert to Fick angles, Listing deviation, etc.
"""

import jax.numpy as jnp
import diffrax

from oculomotor.models import canal as canal_ssm
from oculomotor.models import velocity_storage as vs_ssm
from oculomotor.models import neural_integrator as ni_ssm
from oculomotor.models import plant as plant_ssm
from oculomotor.models import visual_delay as vd_ssm
from oculomotor.models import okr as okr_ssm

# ── Canonical model parameters ─────────────────────────────────────────────────

THETA_DEFAULT = {
    'tau_c':      5.0,    # canal adaptation time constant (s); HP corner ≈ 0.03 Hz
    'tau_s':      0.005,  # canal inertia time constant (s); LP corner ≈ 32 Hz
    'g_vor':      1.0,    # VOR gain (unitless)
    'tau_i':      25.0,   # neural integrator time constant (s)
    'tau_p':      0.15,   # plant time constant (s)
    'tau_vs':     50.0,   # VS prior time constant (s); τ_eff = 1/(1/τ_vs + K_vs) ≈ 20 s
    'K_vs':       0.03,   # VS Kalman gain (1/s)
    'g_okr':      0.7,    # OKR gain (unitless); retinal-slip → VS drive
    'tau_vis':    0.08,   # visual delay time constant (s); ~80 ms
    'tau_okan':   25.0,   # OKR store time constant (s); gives sustained OKN + OKAN
}

# ── State vector layout ────────────────────────────────────────────────────────

_NC      = canal_ssm.N_STATES    # 12  (x1+x2 per canal, 6 canals)
_NVS     = vs_ssm.N_STATES       #  3
_NNI     = ni_ssm.N_STATES       #  3
_NP      = plant_ssm.N_STATES    #  3
_NVis    = vd_ssm.N_STATES       # 12  visual delay cascade
_NOKR    = okr_ssm.N_STATES      #  3  OKR slow store
_N_TOTAL = _NC + _NVS + _NNI + _NP + _NVis + _NOKR   # 27

_IDX_C   = slice(0,                                    _NC)
_IDX_VS  = slice(_NC,                                  _NC + _NVS)
_IDX_NI  = slice(_NC + _NVS,                           _NC + _NVS + _NNI)
_IDX_P   = slice(_NC + _NVS + _NNI,                    _NC + _NVS + _NNI + _NP)
_IDX_VIS = slice(_NC + _NVS + _NNI + _NP,              _NC + _NVS + _NNI + _NP + _NVis)
_IDX_OKR = slice(_NC + _NVS + _NNI + _NP + _NVis,      _N_TOTAL)

_DT_SOLVE = 0.005  # Heun fixed step (s); must satisfy dt < 2*tau_s (inertia TC)


# ── ODE vector field ───────────────────────────────────────────────────────────

def vor_vector_field(t, x, args):
    """ODE right-hand side: chains all SSMs.

    Compatible with diffrax: signature f(t, x, args).

    Args:
        t:    scalar time (s)
        x:    global state array, shape (_N_TOTAL,) = (24,)
        args: (theta, hv_interp, vs_interp, canal_gains)
              hv_interp   evaluates to (3,) head angular velocity at time t
              vs_interp   evaluates to (3,) visual scene velocity at time t
              canal_gains — (N_CANALS,) per-canal gains

    Returns:
        dx_dt: shape (_N_TOTAL,)
    """
    theta, hv_interp, vs_interp, canal_gains = args

    x_c   = x[_IDX_C]
    x_vs  = x[_IDX_VS]
    x_ni  = x[_IDX_NI]
    x_p   = x[_IDX_P]
    x_vis = x[_IDX_VIS]
    x_okr = x[_IDX_OKR]

    # ══ SENSING (world inputs → retinal slip) ════════════════════════════════

    # Physical angular velocities (deg/s)
    w_head  = hv_interp.evaluate(t)                             # (3,) head angular velocity
    w_scene = vs_interp.evaluate(t)                             # (3,) scene angular velocity

    # Canal transduction: bandpass dynamics + nonlinear output + PINV mixing
    dx_c     = canal_ssm.get_A(theta) @ x_c + canal_ssm.get_B(theta) @ w_head
    y_canals = canal_ssm.canal_nonlinearity(x_c, canal_gains)
    u_canal  = canal_ssm.PINV_SENS @ y_canals                   # (3,) head vel estimate (deg/s)

    # Eye angular velocity from current state (no OKR — delayed, second-order effect here)
    u_ni_est = vs_ssm.C @ x_vs + vs_ssm.D @ u_canal
    u_p_est  = ni_ssm.C @ x_ni + ni_ssm.get_D(theta) @ u_ni_est
    w_eye    = plant_ssm.get_A(theta) @ x_p + plant_ssm.get_B(theta) @ u_p_est  # (3,)

    # Retinal slip: scene − head − eye (physical gaze stabilization error, deg/s)
    # Uses w_head (not canal estimate) — retina measures actual motion
    # ≈ 0 in dark VOR (perfect compensation: w_eye ≈ −w_head)
    # ≠ 0 during OKN (w_scene ≠ 0, w_head = w_eye = 0)
    e_slip = w_scene - w_head - w_eye                           # (3,) deg/s

    # Visual delay cascade: low-pass filter chain approximating pure delay τ_vis
    dx_vis    = vd_ssm.get_A(theta) @ x_vis + vd_ssm.get_B(theta) @ e_slip
    e_delayed = vd_ssm.C @ x_vis                                # (3,) delayed retinal slip

    # OKR slow store: charges from delayed slip, holds drive when slip → 0
    dx_okr = okr_ssm.get_A(theta) @ x_okr + okr_ssm.get_B(theta) @ e_delayed

    # ══ NEURAL PROCESSING (sensory + visual → motor commands) ════════════════

    # OKR drive: direct (fast) + store (sustained) — no algebraic loop
    u_okr = okr_ssm.get_C(theta) @ x_okr + okr_ssm.get_D(theta) @ e_delayed  # (3,)

    # Velocity Storage: canal estimate + OKR → stored slow-phase velocity
    u_vs  = u_canal - u_okr
    dx_vs = vs_ssm.get_A(theta) @ x_vs + vs_ssm.get_B(theta) @ u_vs

    # Neural Integrator: velocity → position command (pulse-step)
    u_ni  = vs_ssm.C @ x_vs + vs_ssm.D @ u_vs
    dx_ni = ni_ssm.get_A(theta) @ x_ni + ni_ssm.get_B(theta) @ u_ni

    # Plant: eye rotation vector driven by pulse-step command
    u_p  = ni_ssm.C @ x_ni + ni_ssm.get_D(theta) @ u_ni
    dx_p = plant_ssm.get_A(theta) @ x_p + plant_ssm.get_B(theta) @ u_p

    return jnp.concatenate([dx_c, dx_vs, dx_ni, dx_p, dx_vis, dx_okr])


# ── Simulation entry point ─────────────────────────────────────────────────────

def simulate(theta, t_array_or_stimulus, head_vel_array=None,
             v_scene_array=None,
             canal_gains=None,
             max_steps=10000, dt_solve=None):
    """Integrate the oculomotor ODE and return eye rotation vector.

    Args:
        theta:                dict with keys tau_c, tau_s, g_vor, tau_i, tau_p,
                              tau_vs, K_vs, and optionally g_okr, tau_vis.
        t_array_or_stimulus:  1-D time array (s), shape (T,)  — OR —
                              a Stimulus object (oculomotor.sim.stimulus).
                              When a Stimulus is passed, head_vel_array and
                              v_scene_array are ignored and taken from the object.
        head_vel_array:       head angular velocity, shape (T,) or (T, 3).
                              If 1-D, treated as horizontal (yaw); pitch/roll = 0.
                              Ignored when t_array_or_stimulus is a Stimulus.
        v_scene_array:        visual scene angular velocity, shape (T, 3) or None.
                              None (default) = dark — no visual drive.
                              Ignored when t_array_or_stimulus is a Stimulus.
        canal_gains:          per-canal gains, length N_CANALS. Default None=all 1.
                              Set element to 0.0 to simulate individual canal loss.
        max_steps:            ODE solver step budget (≥ duration / dt_solve).
        dt_solve:             Heun fixed step size (s). Default: _DT_SOLVE (0.005).

    Returns:
        eye_rot: eye rotation vector (deg), shape (T, 3)
                 columns: [yaw, pitch, roll]
                 Use oculomotor.models.readout for Fick/Helmholtz/etc.
    """
    dt = _DT_SOLVE if dt_solve is None else dt_solve

    # ── Accept Stimulus object ────────────────────────────────────────────────
    if hasattr(t_array_or_stimulus, 'omega') and hasattr(t_array_or_stimulus, 't'):
        stim           = t_array_or_stimulus
        t_array        = stim.t
        head_vel_array = stim.omega
        v_scene_array  = stim.v_scene
    else:
        t_array = t_array_or_stimulus

    # ── Pad 1-D head velocity to 3-D ─────────────────────────────────────────
    if jnp.ndim(head_vel_array) == 1:
        hv3 = jnp.stack([head_vel_array,
                          jnp.zeros_like(head_vel_array),
                          jnp.zeros_like(head_vel_array)], axis=1)
    else:
        hv3 = head_vel_array

    # ── Visual scene velocity (zeros if dark) ─────────────────────────────────
    T = len(t_array)
    if v_scene_array is None:
        vs3 = jnp.zeros((T, 3))
    elif jnp.ndim(v_scene_array) == 1:
        vs3 = jnp.stack([v_scene_array,
                          jnp.zeros_like(v_scene_array),
                          jnp.zeros_like(v_scene_array)], axis=1)
    else:
        vs3 = jnp.asarray(v_scene_array)

    hv_interp = diffrax.LinearInterpolation(ts=t_array, ys=hv3)
    vs_interp = diffrax.LinearInterpolation(ts=t_array, ys=vs3)
    x0        = jnp.zeros(_N_TOTAL)

    if canal_gains is None:
        gains_array = jnp.ones(canal_ssm.N_CANALS)
    else:
        gains_array = jnp.array(list(canal_gains), dtype=jnp.float32)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(vor_vector_field),
        diffrax.Heun(),
        t0=t_array[0],
        t1=t_array[-1],
        dt0=dt,
        y0=x0,
        args=(theta, hv_interp, vs_interp, gains_array),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=t_array),
        max_steps=max_steps,
    )

    return solution.ys[:, _IDX_P]   # (T, 3) eye rotation vector
