"""Full oculomotor simulator: Canal Array → VS → NI → Plant, with OKR and Saccades.

Signal flow (3-D):

    VOR pathway:
        w_head (3,) → [Canal Array] → y_canals (6,)
                    → PINV_SENS (3×6) → u_canal (3,)

    OKR pathway  [see models/visual_delay.py and models/okr.py]:
        x_vis     (12,)  visual delay cascade state (4 stages × 3 axes)
        e_delayed (3,)   = C · x_vis  — delayed retinal slip (read from state)
        u_okr     (3,)   OKR drive from delayed slip + store

    Visual delay update (end of step, after motor pathway):
        w_eye     (3,)   = dx_p  — exact eye velocity (u_p now known)
        e_slip    (3,)   = w_scene − w_head − w_eye  — exact retinal slip at t
        dx_vis           updates x_vis with current e_slip

    Combined VS input:
        u_vs = u_canal − u_okr  (handled inside vs.step via stacked input)

    Saccade pathway  [see models/saccade_generator.py]:
        p_target  (3,)   Cartesian target position → theta_target (deg) via atan2
        e_motor   (3,)   = theta_target − q_eye  (motor error in deg)
        u_burst   (3,)   saccade velocity command from local-feedback burst model

    Downstream (VOR + Saccade combined):
        u_vs → [VS] → x_vs (3,)  slow storage state
        u_ni = x_vs + u_vs               (Robinson/Raphan feedthrough)
        u_ni + u_burst → [NI] → x_ni (3,)   position command (burst charges NI)
        u_p  = C·x_ni + D·(u_ni + u_burst)   pulse-step (burst feeds plant directly)
        u_p  → [Plant] → q (3,)              eye rotation vector (deg)

Global state vector (51 states):
    x = [x_c (12) | x_vs (3) | x_ni (3) | x_p (3) | x_vis (24) | x_okr (3) | x_sg (3)]
         ─ canal ─   ─ VS ─   ─  NI  ─   plant     vis delay     OKR store   saccade gen
                                           (12 slip + 12 pos error)

Head velocity input:
    Accepts 1-D (T,) horizontal-only array — padded to (T, 3) internally.
    Accepts 3-D (T, 3) array directly for full 3-D stimulation.
    Accepts a Stimulus object (oculomotor.sim.stimulus) for full 6-DOF + visual input.

Scene angular velocity input:
    Accepts (T, 3) array or None (dark = zeros).
    Or pass a Stimulus object that carries w_scene.

Target position input:
    Accepts (T, 3) Cartesian array or None (saccades disabled, g_burst must be 0).
    Pass a fixed (3,) array for a constant target (replicated across time).

Output:
    simulate() returns eye rotation vector, shape (T, 3).
    Use readout module to convert to Fick angles, Listing deviation, etc.
"""

import jax.numpy as jnp
import diffrax

from oculomotor.models import canal
from oculomotor.models import velocity_storage as vs
from oculomotor.models import neural_integrator as ni
from oculomotor.models import plant
from oculomotor.models import visual_delay
from oculomotor.models import okr
from oculomotor.models import saccade_generator as sg

# ── Canonical model parameters ─────────────────────────────────────────────────

THETA_DEFAULT = {
    'tau_c':        5.0,    # canal adaptation time constant (s); HP corner ≈ 0.03 Hz
    'tau_s':        0.005,  # canal inertia time constant (s); LP corner ≈ 32 Hz
    'g_vor':        1.0,    # VOR gain (unitless)
    'tau_i':        25.0,   # neural integrator time constant (s)
    'tau_p':        0.15,   # plant time constant (s)
    'tau_vs':       50.0,   # VS prior time constant (s); τ_eff = 1/(1/τ_vs + K_vs) ≈ 20 s
    'K_vs':         0.03,   # VS Kalman gain (1/s)
    'g_okr':        0.7,    # OKR gain (unitless); retinal-slip → VS drive
    'tau_vis':      0.08,   # visual delay time constant (s); ~80 ms
    'tau_okan':     25.0,   # OKR store time constant (s); gives sustained OKN + OKAN
    # saccade generator (disabled by default; set g_burst > 0 to enable)
    'g_burst':        0.0,   # burst ceiling (deg/s); 0 = saccades disabled
    'threshold_sac':  0.5,   # saccade threshold (deg)
    'k_sac':         15.0,   # sigmoid steepness (1/deg)
    'e_sat_sac':     10.0,   # tanh saturation amplitude (deg)
    'tau_reset_sac':  0.1,   # resettable integrator decay TC (s)
}

# ── State vector layout ────────────────────────────────────────────────────────

_NC      = canal.N_STATES           # 12  (x1+x2 per canal, 6 canals)
_NVS     = vs.N_STATES              #  3
_NNI     = ni.N_STATES              #  3
_NP      = plant.N_STATES           #  3
_NVis    = visual_delay.N_STATES    # 24  visual delay (12 slip + 12 pos error)
_NOKR    = okr.N_STATES             #  3  OKR slow store
_NSG     = sg.N_STATES              #  3  saccade generator (x_reset_int)
_N_TOTAL = _NC + _NVS + _NNI + _NP + _NVis + _NOKR + _NSG    # 51

_IDX_C   = slice(0,                                         _NC)
_IDX_VS  = slice(_NC,                                       _NC + _NVS)
_IDX_NI  = slice(_NC + _NVS,                                _NC + _NVS + _NNI)
_IDX_P   = slice(_NC + _NVS + _NNI,                         _NC + _NVS + _NNI + _NP)
_IDX_VIS = slice(_NC + _NVS + _NNI + _NP,                   _NC + _NVS + _NNI + _NP + _NVis)
_IDX_OKR = slice(_NC + _NVS + _NNI + _NP + _NVis,           _NC + _NVS + _NNI + _NP + _NVis + _NOKR)
_IDX_SG  = slice(_NC + _NVS + _NNI + _NP + _NVis + _NOKR,   _N_TOTAL)

_DT_SOLVE = 0.005  # Heun fixed step (s); must satisfy dt < 2*tau_s (inertia TC)


# ── ODE vector field ───────────────────────────────────────────────────────────

def vor_vector_field(t, x, args):
    """ODE right-hand side: chains all SSMs.

    Compatible with diffrax: signature f(t, x, args).

    Args:
        t:    scalar time (s)
        x:    global state vector, shape (_N_TOTAL,)
        args: (theta, hv_interp, vs_interp, target_interp, canal_gains)

    Returns:
        dx_dt: shape (_N_TOTAL,)
    """
    theta, hv_interp, vs_interp, target_interp, canal_gains = args

    x_c   = x[_IDX_C]
    x_vs  = x[_IDX_VS]
    x_ni  = x[_IDX_NI]
    x_p   = x[_IDX_P]
    x_vis = x[_IDX_VIS]
    x_okr = x[_IDX_OKR]
    x_sg  = x[_IDX_SG]

    w_head   = hv_interp.evaluate(t)       # (3,) head angular velocity (deg/s)
    w_scene  = vs_interp.evaluate(t)       # (3,) scene angular velocity (deg/s)
    p_target = target_interp.evaluate(t)   # (3,) Cartesian target position

    # ── Vestibular sensing ────────────────────────────────────────────────────
    dx_c, y_canals = canal.step(x_c, w_head, theta, canal_gains)

    # ── Read delayed slip from visual delay state ────────────────────────────
    e_slip_delayed = visual_delay.C_slip @ x_vis   # (3,) delayed retinal slip

    # ── OKR and velocity storage ───────────────────────────────────────────────
    dx_okr, u_okr = okr.step(x_okr, e_slip_delayed, theta)
    dx_vs, u_ni   = vs.step(x_vs, jnp.concatenate([y_canals, u_okr]), theta)

    # ── Saccade generator (instantaneous motor error) ─────────────────────────
    theta_target   = sg.target_to_angle(p_target)   # (3,) target angle (deg)
    e_motor        = theta_target - x_p              # (3,) instantaneous motor error
    dx_sg, u_burst = sg.step(x_sg, e_motor, theta)

    # ── Neural integrator + plant (VOR + saccade combined) ───────────────────
    dx_ni, u_p = ni.step(x_ni, u_ni, theta)
    tau_p  = theta['tau_p']
    u_p    = u_p + tau_p * u_burst    # velocity pulse → plant (cancels LP lag)
    dx_ni  = dx_ni + u_burst          # burst integrates into real NI → holds position

    dx_p, _ = plant.step(x_p, u_p, theta)

    # ── Update visual delay: retinal slip + position error ────────────────────
    w_eye  = dx_p
    e_slip = w_scene - w_head - w_eye
    dx_vis, _, _ = visual_delay.step(x_vis, e_slip, e_motor, theta)

    return jnp.concatenate([dx_c, dx_vs, dx_ni, dx_p, dx_vis, dx_okr, dx_sg])


# ── Simulation entry point ─────────────────────────────────────────────────────

def simulate(theta, t_array_or_stimulus, head_vel_array=None,
             v_scene_array=None,
             p_target_array=None,
             canal_gains=None,
             max_steps=10000, dt_solve=None):
    """Integrate the oculomotor ODE and return eye rotation vector.

    Args:
        theta:                dict with keys tau_c, tau_s, g_vor, tau_i, tau_p,
                              tau_vs, K_vs, and optionally g_okr, tau_vis,
                              g_burst, tau_fb_sac, threshold_sac, k_sac, tau_ref.
        t_array_or_stimulus:  1-D time array (s), shape (T,)  — OR —
                              a Stimulus object (oculomotor.sim.stimulus).
        head_vel_array:       head angular velocity, shape (T,) or (T, 3).
                              If 1-D, treated as horizontal (yaw); pitch/roll = 0.
        v_scene_array:        visual scene angular velocity, shape (T, 3) or None.
                              None (default) = dark — no visual drive.
        p_target_array:       Cartesian target position, shape (T, 3) or (3,) or None.
                              None = straight ahead [0, 0, 1] — saccades inactive
                              unless g_burst > 0 in theta.
                              Shape (3,) = constant target (replicated over time).
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

    T = len(t_array)

    # ── Pad 1-D head velocity to 3-D ─────────────────────────────────────────
    if jnp.ndim(head_vel_array) == 1:
        hv3 = jnp.stack([head_vel_array,
                          jnp.zeros_like(head_vel_array),
                          jnp.zeros_like(head_vel_array)], axis=1)
    else:
        hv3 = head_vel_array

    # ── Visual scene velocity (zeros if dark) ─────────────────────────────────
    if v_scene_array is None:
        vs3 = jnp.zeros((T, 3))
    elif jnp.ndim(v_scene_array) == 1:
        vs3 = jnp.stack([v_scene_array,
                          jnp.zeros_like(v_scene_array),
                          jnp.zeros_like(v_scene_array)], axis=1)
    else:
        vs3 = jnp.asarray(v_scene_array)

    # ── Target position (straight ahead = [0, 0, 1] if not specified) ─────────
    if p_target_array is None:
        pt3 = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (T, 1))
    elif jnp.ndim(jnp.asarray(p_target_array)) == 1:
        pt3 = jnp.tile(jnp.asarray(p_target_array), (T, 1))
    else:
        pt3 = jnp.asarray(p_target_array)

    hv_interp     = diffrax.LinearInterpolation(ts=t_array, ys=hv3)
    vs_interp     = diffrax.LinearInterpolation(ts=t_array, ys=vs3)
    target_interp = diffrax.LinearInterpolation(ts=t_array, ys=pt3)
    x0            = jnp.zeros(_N_TOTAL)

    if canal_gains is None:
        gains_array = jnp.ones(canal.N_CANALS)
    else:
        gains_array = jnp.array(list(canal_gains), dtype=jnp.float32)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(vor_vector_field),
        diffrax.Heun(),
        t0=t_array[0],
        t1=t_array[-1],
        dt0=dt,
        y0=x0,
        args=(theta, hv_interp, vs_interp, target_interp, gains_array),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=t_array),
        max_steps=max_steps,
    )

    return solution.ys[:, _IDX_P]   # (T, 3) eye rotation vector
