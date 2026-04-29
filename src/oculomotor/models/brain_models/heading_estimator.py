"""Heading estimator SSM — linear velocity of the head in head-fixed frame.

Integrates the otolith-derived linear acceleration (a_est = GIA − ĝ, from the
gravity estimator) to track the head's linear velocity in the head-fixed frame.
Angular heading velocity is already available from velocity storage (w_est) and
needs no additional state here.

State:  x_head = [v_lin (3,)]                                               (3,)
    v_lin  linear velocity estimate, head-fixed frame (m/s)

Dynamics:
    dv_lin/dt = a_est − v_lin / τ_head

where:
    a_est   (3,)  estimated linear acceleration from gravity estimator = GIA − g_est
    τ_head        leaky integrator TC (s); prevents drift from accumulating

This is an approximation to the true integral ∫ a_est dt:
    - A perfect integrator reconstructs velocity exactly but accumulates unbounded drift.
    - The leak term provides a DC restoring force: v_lin → 0 when a_est = 0.
    - τ_head trades off low-frequency velocity tracking against drift rejection.

Frequency response (v_lin / a_est):
    Below 1/(2π·τ_head): gain → τ_head² · ω  (strongly attenuated — drift-rejected)
    Above 1/(2π·τ_head): gain ≈ τ_head / jω   (integrator — correct reconstruction)
    τ_head = 2 s → corner frequency ≈ 0.08 Hz; covers behavioural range (0.1–5 Hz).

At rest (a_est = 0): v_lin → 0.
Sustained lateral acceleration A: v_lin → A · τ_head (m/s) at steady state.

Axis convention — head-fixed frame (LEFT-HANDED: x=right, y=up, z=forward):
    v_lin[0]: rightward velocity  (m/s)
    v_lin[1]: upward velocity     (m/s)
    v_lin[2]: forward velocity    (m/s)

Planned extensions:
    - Convert v_lin to world frame (requires orientation tracking or quaternion
      state) for path integration / spatial navigation.
    - Combine with angular velocity w_est to drive LVOR for near targets:
        LVOR compensation = v_lin × vergence_angle / target_distance
    - Add a second slower leak (τ_head_slow >> τ_head) for self-motion perception
      at DC (sustained translation feels like a tilt via somatogravic illusion;
      v_lin complements g_est to disentangle tilt vs. translation).

Parameters (in BrainParams):
    tau_head (s)  linear velocity integration TC. Default 2 s.
"""

import jax.numpy as jnp

N_STATES  = 3   # [v_lin (3,)]
N_INPUTS  = 6   # [g_est (3,) | gia (3,)]  from gravity estimator
N_OUTPUTS = 3   # [v_lin (3,)]

X0 = jnp.zeros(3)   # zero velocity at rest


def step(x_head, u, brain_params):
    """Single ODE step: leaky linear-velocity integrator.

    Args:
        x_head:       (3,)  [v_lin (3,)]  current linear velocity estimate (m/s)
        u:            (6,)  [g_est (3,) | gia (3,)]  from gravity estimator
                            g_est: gravity estimate, head frame (m/s²); upright rest: [0, +9.81, 0]
                            gia:   gravitoinertial acceleration from otolith, head frame (m/s²)
        brain_params: BrainParams  (reads tau_head)

    Returns:
        dx_head: (3,)  dv_lin/dt  (m/s²)
        v_lin:   (3,)  current linear velocity estimate (m/s), passed through
    """
    v_lin = x_head   # (3,) current linear velocity, head frame (m/s)
    g_est = u[:3]    # (3,) gravity estimate from GE (m/s²)
    gia   = u[3:]    # (3,) gravitoinertial acceleration from otolith (m/s²)

    # Linear acceleration = GIA minus gravity estimate.
    # At rest: gia ≈ g_est → a_est ≈ 0. During translation: a_est ≠ 0.
    a_est = gia - g_est

    # Leaky integration: accumulate linear acceleration; leak prevents drift.
    dx = a_est - v_lin / brain_params.tau_head

    return dx, v_lin
