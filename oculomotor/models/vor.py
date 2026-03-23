"""Full VOR model: Canal Array → Velocity Storage → Neural Integrator → Plant.

Signal flow (3-D):
    ω_head (3,) → [Canal Array] → y_canals (6,)
                → PINV_SENS (3×6) → u_vs (3,)
                → [VS] → x_vs (3,) slow storage state
                ↓                      ↓
                u_vs (direct)        x_vs (storage)
                └──────────────────┘ → u_ni = x_vs + u_vs   (Robinson/Raphan feedthrough)
                → [NI]  x_ni (3,): position cmd, D: velocity feedthrough
                → u_p = C·x_ni + D·u_ni   (pulse-step, 3,)
                → [Plant] → q (3,) eye rotation vector (yaw, pitch, roll deg)

Global state vector (21 states):
    x = [x1_c0..x1_c5 | x2_c0..x2_c5 | x_vs (3) | x_ni (3) | x_p (3)]
         ──── canal (12) ────────────    ── VS ──   ── NI ──   plant

Head velocity input:
    Accepts 1-D (T,) horizontal-only array — padded to (T, 3) internally.
    Accepts 3-D (T, 3) array directly for full 3-D stimulation.

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

# ── State vector layout ────────────────────────────────────────────────────────

_NC      = canal_ssm.N_STATES    # 12  (x1+x2 per canal, 6 canals)
_NVS     = vs_ssm.N_STATES       #  3
_NNI     = ni_ssm.N_STATES       #  3
_NP      = plant_ssm.N_STATES    #  3
_N_TOTAL = _NC + _NVS + _NNI + _NP   # 21

_IDX_C  = slice(0,                  _NC)
_IDX_VS = slice(_NC,                _NC + _NVS)
_IDX_NI = slice(_NC + _NVS,         _NC + _NVS + _NNI)
_IDX_P  = slice(_NC + _NVS + _NNI,  _N_TOTAL)

_DT_SOLVE = 0.005  # Heun fixed step (s); must satisfy dt < 2*tau_s (inertia TC)


# ── ODE vector field ───────────────────────────────────────────────────────────

def vor_vector_field(t, x, args):
    """ODE right-hand side: chains all four SSMs.

    Compatible with diffrax: signature f(t, x, args).

    Args:
        t:    scalar time (s)
        x:    global state array, shape (_N_TOTAL,) = (21,)
        args: (theta, hv_interp, canal_floor, canal_gains)
              hv_interp evaluates to (3,) head angular velocity at time t
              canal_floor  — inhibition floor (deg/s); large value = linear
              canal_gains  — (N_CANALS,) per-canal gains

    Returns:
        dx_dt: shape (_N_TOTAL,)
    """
    theta, hv_interp, canal_floor, canal_gains = args

    x_c  = x[_IDX_C]
    x_vs = x[_IDX_VS]
    x_ni = x[_IDX_NI]
    x_p  = x[_IDX_P]

    # ── Canal array: bandpass dynamics ──────────────────────────────────────
    u_c  = hv_interp.evaluate(t)                           # (3,) head angular vel
    dx_c = canal_ssm.get_A(theta) @ x_c + canal_ssm.get_B(theta) @ u_c

    # ── Canal outputs (nonlinear) → pseudo-inverse mixing → VS input ────────
    y_canals = canal_ssm.canal_outputs(x_c, canal_floor, canal_gains)
    u_vs     = canal_ssm.PINV_SENS @ y_canals              # (3,) velocity estimate

    # ── Velocity Storage ─────────────────────────────────────────────────────
    dx_vs = vs_ssm.get_A(theta) @ x_vs + vs_ssm.get_B(theta) @ u_vs

    # ── Neural Integrator ─────────────────────────────────────────────────────
    # VS output = slow storage state + direct canal feedthrough (D=I).
    u_ni  = vs_ssm.C @ x_vs + vs_ssm.D @ u_vs
    dx_ni = ni_ssm.get_A(theta) @ x_ni + ni_ssm.get_B(theta) @ u_ni

    # ── Plant: receives pulse-step signal ─────────────────────────────────────
    u_p  = ni_ssm.C @ x_ni + ni_ssm.get_D(theta) @ u_ni
    dx_p = plant_ssm.get_A(theta) @ x_p + plant_ssm.get_B(theta) @ u_p

    return jnp.concatenate([dx_c, dx_vs, dx_ni, dx_p])


# ── Simulation entry point ─────────────────────────────────────────────────────

def simulate(theta, t_array, head_vel_array,
             canal_floor=1e6, canal_gains=None,
             max_steps=10000, dt_solve=None):
    """Integrate the VOR ODE and return eye rotation vector at each time point.

    Args:
        theta:           dict with keys tau_c, tau_s, g_vor, tau_i, tau_p, tau_vs, K_vs
        t_array:         1-D time array (s), shape (T,)
        head_vel_array:  head angular velocity, shape (T,) or (T, 3).
                         If 1-D, treated as horizontal (yaw); pitch and roll = 0.
        canal_floor:     inhibition depth (deg/s). Default 1e6 ≈ linear.
                         Use 80.0 for physiological HIT nonlinearity.
        canal_gains:     per-canal gains, length N_CANALS. Default None = all 1.
                         Set element to 0.0 to simulate individual canal loss.
        max_steps:       ODE solver step budget (must be ≥ duration / dt_solve).
        dt_solve:        Heun fixed step size (s). Default: _DT_SOLVE (0.005 s).

    Returns:
        eye_rot: eye rotation vector (deg), shape (T, 3)
                 columns: [yaw, pitch, roll]
                 Use oculomotor.models.readout for Fick/Helmholtz/etc.
    """
    dt = _DT_SOLVE if dt_solve is None else dt_solve

    # Pad 1-D head velocity to 3-D (horizontal only)
    if jnp.ndim(head_vel_array) == 1:
        hv3 = jnp.stack([head_vel_array,
                          jnp.zeros_like(head_vel_array),
                          jnp.zeros_like(head_vel_array)], axis=1)
    else:
        hv3 = head_vel_array   # (T, 3)

    hv_interp   = diffrax.LinearInterpolation(ts=t_array, ys=hv3)
    x0          = jnp.zeros(_N_TOTAL)

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
        args=(theta, hv_interp, jnp.float32(canal_floor), gains_array),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=t_array),
        max_steps=max_steps,
    )

    return solution.ys[:, _IDX_P]   # (T, 3) eye rotation vector
