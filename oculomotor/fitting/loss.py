"""Loss functions for VOR parameter fitting."""

import jax
import jax.numpy as jnp

from oculomotor.models.ocular_motor_simulator import simulate


def _unconstrain_to_params(phi, tau_c, tau_s):
    """Map unconstrained vector phi → physical parameter dict theta.

    tau_c and tau_s are fixed (known) and passed separately; not optimized.

    phi index  parameter   transform                    constraint
    ─────────────────────────────────────────────────────────────────
    0          g_vor       sigmoid × 2                  ∈ (0, 2)
    1          tau_i       softplus                     > 0
    2          tau_p       0.05 + softplus              > 0.05 (Heun stability)
    3          tau_vs      softplus                     > 0
    4          K_vs        softplus                     > 0
    """
    softplus = lambda x: jnp.log1p(jnp.exp(x))
    return {
        'tau_c':  tau_c,
        'tau_s':  tau_s,
        'g_vor':  jax.nn.sigmoid(phi[0]) * 2.0,
        'tau_i':  softplus(phi[1]),
        'tau_p':  0.05 + softplus(phi[2]),
        'tau_vs': softplus(phi[3]),
        'K_vs':   softplus(phi[4]),
    }


def params_to_phi(theta):
    """Map physical parameter dict theta → unconstrained vector phi (5 elements).

    tau_c and tau_s are not included — they are treated as known fixed parameters.
    """
    def softplus_inv(y):
        return jnp.log(jnp.expm1(y))

    def sigmoid_inv_times2(val):
        p = val / 2.0
        return jnp.log(p / (1.0 - p))

    return jnp.array([
        sigmoid_inv_times2(theta['g_vor']),
        softplus_inv(theta['tau_i']),
        softplus_inv(theta['tau_p'] - 0.05),   # inverse of 0.05 + softplus(·)
        softplus_inv(theta['tau_vs']),
        softplus_inv(theta['K_vs']),
    ])


@jax.jit
def _condition_val_and_grad(phi, tau_c, tau_s, t, head_vel, eye_obs):
    """JIT-compiled value+grad for a single stimulus condition.

    Compiled once per unique (t.shape, head_vel.shape, eye_obs.shape) signature.
    The 6 sinusoidal conditions all share the same shape → one compilation.
    The step condition has a different shape → one more compilation.

    tau_c and tau_s are passed as static-valued JAX scalars (fixed known parameters).
    """
    def single_loss(phi):
        theta = _unconstrain_to_params(phi, tau_c, tau_s)
        eye_pred = simulate(theta, t, head_vel)[:, 0]   # horizontal component
        # Normalise by observed variance so all conditions contribute equally.
        # Without this, the 0.05 Hz condition (95 deg amplitude) swamps the
        # 2 Hz condition (1 deg amplitude) which is the primary tau_p signal.
        obs_var = jnp.var(eye_obs) + 1e-6
        return jnp.mean((eye_pred - eye_obs) ** 2) / obs_var

    return jax.value_and_grad(single_loss)(phi)


def mse_loss(phi, tau_c, tau_s, stimuli, observations):
    """Full MSE loss, averaged over stimulus conditions.

    NOTE: not JIT-compiled as a whole (to avoid fusing all ODE solves).
    Uses _condition_val_and_grad which caches per shape.
    """
    total = 0.0
    for (t, head_vel), eye_obs in zip(stimuli, observations):
        l, _ = _condition_val_and_grad(phi, tau_c, tau_s, t, head_vel, eye_obs)
        total = total + float(l)
    return total / len(stimuli)
