"""Parameter recovery / fitting pipeline.

Primary method: scipy L-BFGS-B with JAX gradients (~100 evaluations to converge).
Secondary method: Adam gradient descent (kept for diagnostic/trajectory plotting).
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from scipy.optimize import minimize

from oculomotor.fitting.loss import (
    _condition_val_and_grad,
    _unconstrain_to_params,
    params_to_phi,
)


def _accumulate_grad(phi, tau_c, tau_s, stimuli, observations):
    """Accumulate loss and gradient over all conditions."""
    total_loss = 0.0
    total_grad = jnp.zeros_like(phi)
    n = len(stimuli)
    for (t, hv), obs in zip(stimuli, observations):
        l, g = _condition_val_and_grad(phi, tau_c, tau_s, t, hv, obs)
        total_loss = total_loss + l
        total_grad = total_grad + g
    return total_loss / n, total_grad / n


def fit(stimuli, observations, theta_init, tau_c, tau_s, n_steps=500, learning_rate=1e-2,
        grad_clip=1.0, print_every=50, method='adam'):
    """Run parameter recovery.

    Args:
        stimuli: list of (t, head_vel) tuples
        observations: list of observed eye_pos arrays
        theta_init: dict with initial parameter guesses (tau_c, tau_s excluded)
        tau_c: fixed canal adaptation time constant (not optimized)
        tau_s: fixed canal inertia time constant (not optimized)
        n_steps: Adam steps (ignored for lbfgs)
        learning_rate: Adam peak LR with cosine decay (ignored for lbfgs)
        grad_clip: gradient clip norm (ignored for lbfgs)
        print_every: print interval (0 = silent)
        method: 'lbfgs' (default, fast) or 'adam' (for trajectory plots)

    Returns:
        theta_fit: recovered parameter dict (includes tau_c, tau_s = fixed values)
        history: dict with keys 'loss', 'g_vor', 'tau_i', 'tau_p', 'tau_vs', 'K_vs'
    """
    if method == 'lbfgs':
        return _fit_lbfgs(stimuli, observations, theta_init, tau_c, tau_s,
                          max_iter=500, print_every=print_every)
    else:
        return _fit_adam(stimuli, observations, theta_init, tau_c, tau_s,
                         n_steps=n_steps, learning_rate=learning_rate,
                         grad_clip=grad_clip, print_every=print_every)


def _fit_lbfgs(stimuli, observations, theta_init, tau_c, tau_s, max_iter=500, print_every=50):
    """scipy L-BFGS-B with JAX gradients.  Converges in ~100-200 evaluations."""
    phi_init = params_to_phi(theta_init)
    _PARAM_KEYS = ('g_vor', 'tau_i', 'tau_p', 'tau_vs', 'K_vs')
    history = {'loss': []} | {k: [] for k in _PARAM_KEYS}
    call_count = [0]
    tau_c_j = jnp.array(tau_c, dtype=jnp.float32)
    tau_s_j = jnp.array(tau_s, dtype=jnp.float32)

    def fun_and_grad(phi_np):
        phi = jnp.array(phi_np, dtype=jnp.float32)
        loss_val, grad = _accumulate_grad(phi, tau_c_j, tau_s_j, stimuli, observations)
        loss_f = float(loss_val)
        grad_np = np.array(grad, dtype=np.float64)
        theta = _unconstrain_to_params(phi, tau_c_j, tau_s_j)
        history['loss'].append(loss_f)
        for k in _PARAM_KEYS:
            history[k].append(float(theta[k]))
        call_count[0] += 1
        if print_every and call_count[0] % print_every == 0:
            print(f"eval {call_count[0]:4d}  loss={loss_f:.6f}  "
                  f"g_vor={float(theta['g_vor']):.3f}  "
                  f"tau_i={float(theta['tau_i']):.3f}  "
                  f"tau_p={float(theta['tau_p']):.4f}  "
                  f"tau_vs={float(theta['tau_vs']):.1f}  "
                  f"K_vs={float(theta['K_vs']):.4f}")
        return loss_f, grad_np

    phi_init_np = np.array(phi_init, dtype=np.float64)
    result = minimize(fun_and_grad, phi_init_np, method='L-BFGS-B', jac=True,
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    if print_every:
        print(f"L-BFGS-B finished: {result.message}  evals={call_count[0]}")
    theta_fit = _unconstrain_to_params(jnp.array(result.x, dtype=jnp.float32), tau_c_j, tau_s_j)
    return theta_fit, history


def _fit_adam(stimuli, observations, theta_init, tau_c, tau_s, n_steps=2000, learning_rate=1e-2,
              grad_clip=1.0, print_every=200):
    """Adam gradient descent — useful for watching parameter trajectories."""
    phi_init = params_to_phi(theta_init)
    lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=n_steps, alpha=0.1)
    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip),
                            optax.adam(lr_schedule))
    opt_state = optimizer.init(phi_init)

    _PARAM_KEYS = ('g_vor', 'tau_i', 'tau_p', 'tau_vs', 'K_vs')
    phi = phi_init
    history = {'loss': []} | {k: [] for k in _PARAM_KEYS}
    tau_c_j = jnp.array(tau_c, dtype=jnp.float32)
    tau_s_j = jnp.array(tau_s, dtype=jnp.float32)

    for i in range(n_steps + 1):
        loss_val, grads = _accumulate_grad(phi, tau_c_j, tau_s_j, stimuli, observations)
        theta = _unconstrain_to_params(phi, tau_c_j, tau_s_j)
        history['loss'].append(float(loss_val))
        for k in _PARAM_KEYS:
            history[k].append(float(theta[k]))
        if print_every and i % print_every == 0:
            print(f"step {i:5d}  loss={float(loss_val):.6f}  "
                  f"g_vor={float(theta['g_vor']):.3f}  "
                  f"tau_i={float(theta['tau_i']):.3f}  "
                  f"tau_p={float(theta['tau_p']):.4f}  "
                  f"tau_vs={float(theta['tau_vs']):.1f}  "
                  f"K_vs={float(theta['K_vs']):.4f}")
        if i < n_steps:
            updates, opt_state = optimizer.update(grads, opt_state)
            phi = optax.apply_updates(phi, updates)

    return _unconstrain_to_params(phi, tau_c_j, tau_s_j), history
