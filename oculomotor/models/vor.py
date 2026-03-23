"""Full VOR model: Canal Array → Velocity Storage → Neural Integrator → Plant.

Signal flow:
    head_vel → [Canal Array] → y_canals
             → mixing (PINV_SENS_1D) → u_vs
             → [VS] → x_vs (slow storage state)
             ↓                 ↓
             u_vs (direct)   x_vs (storage)
             └───────────────┘ → u_ni = x_vs + u_vs   (Robinson/Raphan feedthrough)
             → [NI]  x_ni: state (position cmd), D: velocity feedthrough
             → u_p = C·x_ni + D·u_ni   (pulse-step)
             → [Plant] → eye_pos

Each canal produces an HP-filtered projection of head velocity onto its
sensitivity axis.  A pseudo-inverse mixing step (canal.PINV_SENS_1D) converts
the canal array output into an angular-velocity estimate for VS — matching the
Laurens internal-model formulation and ready for 3-D extension.

Global state vector (7 states for horizontal pair):
    x = [x1_c0, x1_c1, x2_c0, x2_c1 | x_vs | x_ni | x_plant]
    Canal: x1 = adaptation LP (τ_c), x2 = inertia LP (τ_s); output = x2
"""

import jax.numpy as jnp
import diffrax

from oculomotor.models import canal as canal_ssm
from oculomotor.models import velocity_storage as vs_ssm
from oculomotor.models import neural_integrator as ni_ssm
from oculomotor.models import plant as plant_ssm

# ── State vector layout ───────────────────────────────────────────────────────

_NC      = canal_ssm.N_STATES    # 4  (x1+x2 per canal: adaptation LP + inertia LP)
_NVS     = vs_ssm.N_STATES       # 1
_NNI     = ni_ssm.N_STATES       # 1
_NP      = plant_ssm.N_STATES    # 1
_N_TOTAL = _NC + _NVS + _NNI + _NP  # 7

_IDX_C  = slice(0,                  _NC)
_IDX_VS = slice(_NC,                _NC + _NVS)
_IDX_NI = slice(_NC + _NVS,         _NC + _NVS + _NNI)
_IDX_P  = slice(_NC + _NVS + _NNI,  _N_TOTAL)

_DT_SOLVE = 0.005  # Heun fixed step (s); must satisfy dt < 2*tau_s (inertia TC)


# ── ODE vector field ──────────────────────────────────────────────────────────

def vor_vector_field(t, x, args):
    """ODE right-hand side: chains all four SSMs.

    Compatible with diffrax: signature f(t, x, args).

    Args:
        t:    scalar time (s)
        x:    global state array, shape (_N_TOTAL,)
        args: (theta, hv_interp, canal_floor, canal_gains)
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

    # ── Canal array: HP dynamics ─────────────────────────────────────────
    u_c  = jnp.atleast_1d(hv_interp.evaluate(t))          # (1,) head vel
    dx_c = canal_ssm.get_A(theta) @ x_c + canal_ssm.get_B(theta) @ u_c

    # ── Canal outputs (nonlinear) → pseudo-inverse mixing → VS input ─────
    y_canals = canal_ssm.canal_outputs(x_c, canal_floor, canal_gains)
    u_vs     = canal_ssm.PINV_SENS_1D @ y_canals           # (1,) velocity estimate

    # ── Velocity Storage ─────────────────────────────────────────────────
    dx_vs = vs_ssm.get_A(theta) @ x_vs + vs_ssm.get_B(theta) @ u_vs

    # ── Neural Integrator ────────────────────────────────────────────────
    # VS output = slow storage state + direct canal feedthrough (D=1).
    # At HIT frequencies x_vs≈0 so canal passes directly; at low freq
    # x_vs dominates (velocity storage extension).
    u_ni  = vs_ssm.C @ x_vs + vs_ssm.D @ u_vs
    dx_ni = ni_ssm.get_A(theta) @ x_ni + ni_ssm.get_B(theta) @ u_ni

    # ── Plant: receives pulse-step signal (position + velocity burst) ─────
    # y_ni = C @ x_ni + D @ u_ni  cancels the plant's low-pass lag so that
    # eye_pos = −g_vor × head_pos at all VOR frequencies.
    u_p  = ni_ssm.C @ x_ni + ni_ssm.get_D(theta) @ u_ni
    dx_p = plant_ssm.get_A(theta) @ x_p + plant_ssm.get_B(theta) @ u_p

    return jnp.concatenate([dx_c, dx_vs, dx_ni, dx_p])


# ── Simulation entry point ────────────────────────────────────────────────────

def simulate(theta, t_array, head_vel_array,
             canal_floor=1e6, canal_gains=(1.0, 1.0),
             max_steps=10000, dt_solve=None):
    """Integrate the VOR ODE and return eye position at each time point.

    Args:
        theta:           dict with keys tau_c, g_vor, tau_i, tau_p, tau_vs, K_vs
        t_array:         1-D time array (s), shape (T,)
        head_vel_array:  1-D horizontal head velocity (deg/s), shape (T,)
        canal_floor:     inhibition depth limit (deg/s). Default 1e6 ≈ linear
                         (no half-wave clipping). Use 80.0 for physiological HIT.
        canal_gains:     per-canal gains, length N_CANALS. Default (1, 1) = normal.
                         Set one element to 0.0 to simulate unilateral canal loss.
        max_steps:       ODE solver step budget (must be ≥ duration / dt_solve).
        dt_solve:        Heun fixed step size (s). Default: _DT_SOLVE (0.05 s).
                         Use a smaller value (e.g. 0.005) for fast stimuli such as
                         HIT where tau_p = 0.15 s requires fine stepping for accuracy.

    Returns:
        eye_pos: 1-D array (deg), shape (T,)
    """
    dt          = _DT_SOLVE if dt_solve is None else dt_solve
    hv_interp   = diffrax.LinearInterpolation(ts=t_array, ys=head_vel_array)
    x0          = jnp.zeros(_N_TOTAL)
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

    return (plant_ssm.C @ solution.ys[:, _IDX_P].T).squeeze(0)
