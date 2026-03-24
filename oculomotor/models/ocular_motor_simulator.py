"""Full oculomotor simulator: Canal Array → VS → NI → Plant, with OKR.

Signal flow (3-D):

    VOR pathway:
        ω_head (3,) → [Canal Array] → y_canals (6,)
                    → PINV_SENS (3×6) → u_canal (3,)

    OKR pathway  [see models/okr.py]:
        v_scene  (3,)       visual scene velocity
        x_vis    (12,)      gamma-delay cascade state (4 stages × 3 axes)
        e        (3,)       retinal slip = v_scene − x_vis[last stage]
        u_okr    (3,)       OKR VS drive = D(θ) · e = g_okr · e

    Combined VS input:
        u_vs = u_canal − u_okr
        (OKR subtracts: rightward scene → rightward eye, same as leftward head VOR)

    Downstream:
        u_vs → [VS] → x_vs (3,) slow storage state
        u_ni = x_vs + u_vs               (Robinson/Raphan feedthrough)
        u_ni → [NI] → x_ni (3,)         position command
        u_p  = C·x_ni + D·u_ni          pulse-step
        u_p  → [Plant] → q (3,)         eye rotation vector (deg)

    OKR state update:
        v_eye = dx_p                     (plant derivative = eye velocity)
        dx_vis = A·x_vis + B·v_eye       (LP delay toward eye velocity)

Global state vector (33 states):
    x = [x1_c0..x1_c5 | x2_c0..x2_c5 | x_vs (3) | x_ni (3) | x_p (3) | x_vis (12)]
         ──────── canal (12) ──────────   ── VS ──   ── NI ──   plant    OKR cascade

Head velocity input:
    Accepts 1-D (T,) horizontal-only array — padded to (T, 3) internally.
    Accepts 3-D (T, 3) array directly for full 3-D stimulation.
    Accepts a Stimulus object (oculomotor.sim.stimulus) for full 6-DOF + visual input.

Scene velocity input:
    Accepts (T, 3) array or None (dark = zeros).
    Or pass a Stimulus object that carries v_scene.

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
from oculomotor.models import okr as okr_ssm

# ── State vector layout ────────────────────────────────────────────────────────

_NC      = canal_ssm.N_STATES    # 12  (x1+x2 per canal, 6 canals)
_NVS     = vs_ssm.N_STATES       #  3
_NNI     = ni_ssm.N_STATES       #  3
_NP      = plant_ssm.N_STATES    #  3
_NVis    = okr_ssm.N_STATES      #  3  OKR visual delay state
_N_TOTAL = _NC + _NVS + _NNI + _NP + _NVis   # 24

_IDX_C   = slice(0,                           _NC)
_IDX_VS  = slice(_NC,                         _NC + _NVS)
_IDX_NI  = slice(_NC + _NVS,                  _NC + _NVS + _NNI)
_IDX_P   = slice(_NC + _NVS + _NNI,           _NC + _NVS + _NNI + _NP)
_IDX_VIS = slice(_NC + _NVS + _NNI + _NP,     _N_TOTAL)

_DT_SOLVE = 0.005  # Heun fixed step (s); must satisfy dt < 2*tau_s (inertia TC)


# ── ODE vector field ───────────────────────────────────────────────────────────

def vor_vector_field(t, x, args):
    """ODE right-hand side: chains all SSMs including OKR.

    Compatible with diffrax: signature f(t, x, args).

    Args:
        t:    scalar time (s)
        x:    global state array, shape (_N_TOTAL,) = (24,)
        args: (theta, hv_interp, vs_interp, canal_floor, canal_gains)
              hv_interp  evaluates to (3,) head angular velocity at time t
              vs_interp  evaluates to (3,) visual scene velocity at time t
              canal_floor  — inhibition floor (deg/s); large value = linear
              canal_gains  — (N_CANALS,) per-canal gains

    Returns:
        dx_dt: shape (_N_TOTAL,)
    """
    theta, hv_interp, vs_interp, canal_floor, canal_gains = args

    x_c   = x[_IDX_C]
    x_vs  = x[_IDX_VS]
    x_ni  = x[_IDX_NI]
    x_p   = x[_IDX_P]
    x_vis = x[_IDX_VIS]

    # ── Canal array: bandpass dynamics ──────────────────────────────────────
    u_c  = hv_interp.evaluate(t)
    dx_c = canal_ssm.get_A(theta) @ x_c + canal_ssm.get_B(theta) @ u_c

    # ── Canal outputs → VOR velocity estimate ────────────────────────────────
    y_canals = canal_ssm.canal_outputs(x_c, canal_floor, canal_gains)
    u_canal  = canal_ssm.PINV_SENS @ y_canals              # (3,)

    # ── OKR: retinal slip → VS input ─────────────────────────────────────────
    v_scene_t = vs_interp.evaluate(t)
    u_okr     = okr_ssm.okr_drive(x_vis, v_scene_t, theta)  # (3,) = g_okr · e
    u_vs      = u_canal - u_okr                              # combined VS drive

    # ── Velocity Storage ─────────────────────────────────────────────────────
    dx_vs = vs_ssm.get_A(theta) @ x_vs + vs_ssm.get_B(theta) @ u_vs

    # ── Neural Integrator ─────────────────────────────────────────────────────
    u_ni  = vs_ssm.C @ x_vs + vs_ssm.D @ u_vs
    dx_ni = ni_ssm.get_A(theta) @ x_ni + ni_ssm.get_B(theta) @ u_ni

    # ── Plant ─────────────────────────────────────────────────────────────────
    u_p  = ni_ssm.C @ x_ni + ni_ssm.get_D(theta) @ u_ni
    dx_p = plant_ssm.get_A(theta) @ x_p + plant_ssm.get_B(theta) @ u_p

    # ── OKR visual delay: gamma-cascade (4-stage, τ/4 each) ──────────────────
    dx_vis = okr_ssm.cascade_deriv(x_vis, dx_p, theta)

    return jnp.concatenate([dx_c, dx_vs, dx_ni, dx_p, dx_vis])


# ── Simulation entry point ─────────────────────────────────────────────────────

def simulate(theta, t_array_or_stimulus, head_vel_array=None,
             v_scene_array=None,
             canal_floor=1e6, canal_gains=None,
             max_steps=10000, dt_solve=None):
    """Integrate the oculomotor ODE and return eye rotation vector.

    Args:
        theta:                dict with keys tau_c, tau_s, g_vor, tau_i, tau_p,
                              tau_vs, K_vs, and optionally g_okr, tau_okr_del.
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
        canal_floor:          inhibition depth (deg/s). Default 1e6 ≈ linear.
                              Use 80.0 for physiological HIT nonlinearity.
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
        args=(theta, hv_interp, vs_interp, jnp.float32(canal_floor), gains_array),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=t_array),
        max_steps=max_steps,
    )

    return solution.ys[:, _IDX_P]   # (T, 3) eye rotation vector
