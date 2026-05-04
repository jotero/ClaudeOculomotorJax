"""Self-motion estimation — single SSM module: VS + Gravity + Heading.

Bayesian-style observer of head motion + gravity (Laurens & Angelaki 2011, 2017).
Three internal subsystems share state and are stepped in a fixed order; the
public surface is one ``step()`` function plus index constants for the
combined state and bundled-input layouts.

Internal subsystems (kept as private helpers, no external import path):

  Velocity storage (VS) — bilateral push-pull VN populations + null adaptation
                         angular velocity estimate ω̂ (deg/s) → NI
  Gravity estimator (GE) — Laurens cross-product transport + linear-acc estimate
                          gravity vector ĝ (m/s²) and a_lin (m/s², head frame)
  Heading estimator (HE) — leaky integration of a_lin + visual translational flow
                          head linear velocity v_lin (m/s, head frame) → T-VOR

State layout (relative to x_self_motion, total 21 states):

    x_vs   (9,)   _IDX_VS    [0:9]   — VS L pop + R pop + null
    x_grav (9,)   _IDX_GRAV  [9:18]  — gravity estimate, a_lin, rf state
    x_head (3,)   _IDX_HEAD  [18:21] — head linear velocity v_lin

Bundled inputs to step() (length N_INPUTS=16):

    canal           (6,)   _IDX_INPUT_CANAL          — canal afferents (deg/s)
    scene_slip      (3,)   _IDX_INPUT_SLIP           — retinal scene slip (deg/s)
    gia             (3,)   _IDX_INPUT_GIA            — otolith GIA (m/s², head frame)
    scene_lin_vel   (3,)   _IDX_INPUT_SCENE_LIN_VEL  — scene translational flow (m/s)
    scene_visible   (1,)   _IDX_INPUT_SCENE_VISIBLE  — gates HE visual fusion

Outputs from step():

    dx_self_motion (21,)   ODE derivative of the combined state
    w_est          (3,)    angular velocity estimate (deg/s)
    g_est          (3,)    gravity estimate (m/s², head frame)
    v_lin          (3,)    head linear velocity (m/s, head frame)
    a_lin_est      (3,)    linear-acc estimate (m/s², head frame) — for T-VOR direct

References:
    Raphan, Matsuo & Cohen (1979) — Velocity storage in vestibular nystagmus.
    Cannon & Robinson (1985)      — NI leak / GEN model.
    Laurens & Angelaki (2011) JNS — VS-GE coupling; rf rotational feedback.
    Laurens, Meng & Angelaki (2013) PLoS Comp Bio — translation prior τ_a_lin.
    Paige & Tomko (1991) JN       — empirical T-VOR dark gain.
"""

import jax.numpy as jnp

from oculomotor.models.sensory_models.sensory_model import PINV_SENS, N_CANALS
from oculomotor.models.sensory_models.retina        import ypr_to_xyz, xyz_to_ypr


# ─────────────────────────────────────────────────────────────────────────────
# Module-wide constants
# ─────────────────────────────────────────────────────────────────────────────

G0 = 9.81   # standard gravity (m/s²)

# Default initial GE substate: ĝ upright, â=0, rf=0
GRAV_X0 = jnp.array([0.0, G0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0])

# VS-internal scaling — population-health normaliser
_B_NOMINAL = 100.0   # healthy resting bias (deg/s)

# GE-internal — fast first-order tracking of rf for the VS↔GE algebraic-loop break
_TAU_RF_STATE = 0.005   # 5 ms

# Default initial state for the whole subsystem (used by brain_model.make_x0)
X0 = jnp.concatenate([
    jnp.zeros(9),     # VS — populations and null start at 0; brain_model overrides
                      # with b_vs equilibrium for both L and R pops.
    GRAV_X0,          # GE
    jnp.zeros(3),     # HE — v_lin starts at 0
])


# ─────────────────────────────────────────────────────────────────────────────
# State + input layout
# ─────────────────────────────────────────────────────────────────────────────

# VS sub-state slices (within x_vs, which is the first 9 of x_self_motion)
_VS_IDX_A    = slice(0, 3)   # population A (preferred-direction)
_VS_IDX_B    = slice(3, 6)   # population B (opposite preferred)
_VS_IDX_POP  = slice(0, 6)   # both populations
_VS_IDX_NULL = slice(6, 9)   # null adaptation

# GE sub-state slices (within x_grav, which is x_self_motion[9:18])
_GE_IDX_G  = slice(0, 3)   # g_est  — gravity estimate
_GE_IDX_A  = slice(3, 6)   # a_lin  — linear-acc estimate
_GE_IDX_RF = slice(6, 9)   # rf     — rotational feedback state

# Top-level state slices (relative to x_self_motion)
N_STATES  = 9 + 9 + 3   # 21
_IDX_VS   = slice(0, 9)
_IDX_GRAV = slice(9, 18)
_IDX_HEAD = slice(18, 21)

# Bundled-input layout (length 16)
N_INPUTS  = 6 + 3 + 3 + 3 + 1
N_OUTPUTS = 3   # primary output for SSM convention is w_est; auxiliaries via tuple

_IDX_INPUT_CANAL         = slice(0, 6)
_IDX_INPUT_SLIP          = slice(6, 9)
_IDX_INPUT_GIA           = slice(9, 12)
_IDX_INPUT_SCENE_LIN_VEL = slice(12, 15)
_IDX_INPUT_SCENE_VISIBLE = 15


# ─────────────────────────────────────────────────────────────────────────────
# Velocity Storage — bilateral push-pull (Raphan, Matsuo & Cohen 1979)
# ─────────────────────────────────────────────────────────────────────────────

def _vs_step(x_vs, canal, slip, rf, brain_params):
    """VS internal step.

    Two populations (A/B) with opposite preferred directions; net = x_A − x_B.
    Push-pull on canal and visual inputs; per-axis tau (yaw/pitch/roll fractions);
    null adaptation extends effective TC for sustained stimuli; rf (Laurens)
    couples gravity-estimate mismatch back as a perceived rotation.

    Args:
        x_vs:   (9,) [x_A | x_B | x_null] (deg/s)
        canal:  (6,) canal afferents (deg/s)
        slip:   (3,) scene retinal slip [yaw,pitch,roll] (deg/s)
        rf:     (3,) Laurens rotational feedback [yaw,pitch,roll] (deg/s)
        brain_params: BrainParams

    Returns:
        dx_vs: (9,) state derivative
        w_est: (3,) angular velocity estimate [yaw,pitch,roll] (deg/s)
    """
    x_null = x_vs[_VS_IDX_NULL]
    x_pop  = x_vs[_VS_IDX_POP]   # (6,) [x_A | x_B]

    canal_in = jnp.clip(canal, -brain_params.v_max_vor, brain_params.v_max_vor)
    u_lin    = jnp.concatenate([canal_in, slip])   # (9,) linear inputs

    # Set point: per-population resting bias plus a slow null-adapted shift.
    SP    = brain_params.b_vs + jnp.concatenate([x_null / 2.0, -x_null / 2.0])
    g_pop = brain_params.b_vs / _B_NOMINAL   # (6,) population health: 1=healthy, 0=silent

    # Per-axis tau_vs (yaw + pitch_frac + roll_frac), shared across both populations.
    tau3   = jnp.array([brain_params.tau_vs,
                        brain_params.tau_vs * brain_params.tau_vs_pitch_frac,
                        brain_params.tau_vs * brain_params.tau_vs_roll_frac])
    inv_t6 = jnp.concatenate([1.0 / tau3, 1.0 / tau3])
    A = -jnp.diag(inv_t6)

    # B (6×9): canal + visual, push-pull across populations.
    B_top = jnp.concatenate([ g_pop[:3, None] * brain_params.K_vs * PINV_SENS,
                             -brain_params.K_vis * jnp.eye(3)], axis=1)
    B_bot = jnp.concatenate([-g_pop[3:, None] * brain_params.K_vs * PINV_SENS,
                              brain_params.K_vis * jnp.eye(3)], axis=1)
    B = jnp.concatenate([B_top, B_bot], axis=0)

    # C (3×6): net = x_A − x_B
    C = jnp.concatenate([jnp.eye(3), -jnp.eye(3)], axis=1)

    # D (3×9): canal + visual feedthrough on the net output
    D = jnp.concatenate([brain_params.g_vor * PINV_SENS,
                        -brain_params.g_vis * jnp.eye(3)], axis=1)

    # Rotational feedback (Laurens & Angelaki 2011) — push-pull across populations
    rf6 = brain_params.K_gd * jnp.concatenate([rf, -rf])

    dx_pop  = A @ (x_pop - SP) + B @ u_lin - rf6
    w_est   = C @ x_pop + D @ u_lin
    dx_null = (w_est - x_null) / brain_params.tau_vs_adapt

    return jnp.concatenate([dx_pop, dx_null]), w_est


# ─────────────────────────────────────────────────────────────────────────────
# Gravity Estimator — Laurens & Angelaki cross-product dynamics
# ─────────────────────────────────────────────────────────────────────────────

def _ge_step(x_grav, w_est, gia, brain_params):
    """GE internal step.

    Tracks gravity ĝ (slow, anchored to GIA) and translation â (transient,
    decays toward 0 in absence of evidence). VS angular velocity transports ĝ
    in the head frame. rf is the Laurens rotational feedback fed BACK into VS
    next step.

    Args:
        x_grav: (9,) [ĝ | â | rf] (head frame, m/s² for first six, deg/s for rf)
        w_est:  (3,) VS net angular velocity [yaw,pitch,roll] (deg/s)
        gia:    (3,) otolith GIA (m/s², head frame)
        brain_params: BrainParams

    Returns:
        dx_grav: (9,) state derivative
        g_est:   (3,) gravity estimate (passed through from state)
    """
    g_est    = x_grav[_GE_IDX_G]
    a_lin    = x_grav[_GE_IDX_A]
    rf_state = x_grav[_GE_IDX_RF]

    # Residual: GIA minus the brain's two estimates. The two states (ĝ, â)
    # compete for it via their own gains (K_grav, K_lin). Once â captures
    # the translation component, residual → 0 and ĝ stops drifting toward
    # transient acceleration. Translation prior (decay on â) keeps it from
    # locking on a sustained DC.
    residual = gia - g_est - a_lin

    # Transport: rotate ĝ with VS angular velocity (VN → uvula/nodulus pathway).
    w_rad_xyz = jnp.radians(ypr_to_xyz(w_est))
    transport = -jnp.cross(w_rad_xyz, g_est)

    # Gravity correction: pulled toward residual with somatogravic gain K_grav.
    dg = transport + brain_params.K_grav * residual

    # Linear acceleration: tracks residual, decays toward 0 on TC τ_a_lin
    # (deterministic stand-in for L&A's translation-duration prior).
    da = brain_params.K_lin * residual - a_lin / brain_params.tau_a_lin

    # Rotational feedback (Laurens 2011): GIA × G_down / G0². Zero at SS;
    # active when ĝ lags GIA. Stored in state (1-step delayed) so brain_model
    # can read it next step without an algebraic loop.
    rf_new = xyz_to_ypr(jnp.cross(gia, -g_est)) / (G0 ** 2)
    drf    = (rf_new - rf_state) / _TAU_RF_STATE

    return jnp.concatenate([dg, da, drf]), g_est


# ─────────────────────────────────────────────────────────────────────────────
# Heading Estimator — leaky integration of a_lin + visual flow
# ─────────────────────────────────────────────────────────────────────────────

def _he_step(x_head, a_lin, scene_lin_vel, scene_visible, brain_params):
    """HE internal step.

    Vestibular path: leaky integral of a_lin (the GE's translation-attributed
    component of GIA, NOT raw gia − g_est — avoids gravity-mismatch drift).
    Visual path: scene flow pulls v_lin toward −scene_lin_vel; gated by
    scene visibility (zeroed in dark).

    Args:
        x_head:        (3,) v_lin estimate (m/s, head frame)
        a_lin:         (3,) GE's linear-acc estimate (m/s², head frame)
        scene_lin_vel: (3,) cyclopean scene flow (m/s, head frame)
        scene_visible: scalar in [0, 1] — visual fusion gate
        brain_params:  BrainParams (reads tau_head, K_he_vis)

    Returns:
        dx_head: (3,) dv_lin/dt (m/s²)
        v_lin:   (3,) v_lin (passed through from state)
    """
    v_lin    = x_head
    v_visual = -scene_lin_vel
    K_vis    = brain_params.K_he_vis * scene_visible

    dx = a_lin - v_lin / brain_params.tau_head + K_vis * (v_visual - v_lin)
    return dx, v_lin


# ─────────────────────────────────────────────────────────────────────────────
# Public step() — orchestrates VS → GE → HE
# ─────────────────────────────────────────────────────────────────────────────

def step(x_self_motion, u, brain_params):
    """Single ODE step for the unified self-motion observer.

    Internal sequencing (matches Laurens & Angelaki 2017):
      1. VS uses rf_state from the (1-step delayed) ODE state — breaks the
         VS↔GE algebraic loop with negligible lag (τ_rf_state ≈ 5 ms).
      2. GE then runs with the freshly-computed w_est from VS.
      3. HE consumes a_lin from GE state (1-step delayed) for its own
         leaky integration toward v_lin.

    Args:
        x_self_motion: (21,) bundled state — see _IDX_VS / _IDX_GRAV / _IDX_HEAD
        u:             (16,) bundled input  — see _IDX_INPUT_* above
        brain_params:  BrainParams

    Returns:
        dx_self_motion : (21,) state derivative
        w_est          : (3,)  angular velocity estimate (deg/s)
        g_est          : (3,)  gravity estimate (m/s², head frame)
        v_lin          : (3,)  head linear velocity (m/s, head frame)
        a_lin_est      : (3,)  linear-acc estimate (m/s², head frame) — for T-VOR direct
    """
    # State split
    x_vs   = x_self_motion[_IDX_VS]
    x_grav = x_self_motion[_IDX_GRAV]
    x_head = x_self_motion[_IDX_HEAD]

    # GE state read (rf and a_lin are 1-step delayed via ODE state)
    rf_state  = x_grav[_GE_IDX_RF]
    a_lin_est = x_grav[_GE_IDX_A]

    # Input split
    canal         = u[_IDX_INPUT_CANAL]
    slip          = u[_IDX_INPUT_SLIP]
    gia           = u[_IDX_INPUT_GIA]
    scene_lin_vel = u[_IDX_INPUT_SCENE_LIN_VEL]
    scene_visible = u[_IDX_INPUT_SCENE_VISIBLE]

    # 1. VS — angular velocity estimate
    dx_vs, w_est = _vs_step(x_vs, canal, slip, rf_state, brain_params)

    # 2. GE — gravity + linear-acc estimates (rf updated for next step)
    dx_grav, g_est = _ge_step(x_grav, w_est, gia, brain_params)

    # 3. HE — head linear velocity (consumes the prior step's a_lin to avoid
    #         needing the freshly-computed â here)
    dx_head, v_lin = _he_step(x_head, a_lin_est, scene_lin_vel, scene_visible, brain_params)

    dx_self_motion = jnp.concatenate([dx_vs, dx_grav, dx_head])
    return dx_self_motion, w_est, g_est, v_lin, a_lin_est


__all__ = [
    "step", "X0", "GRAV_X0", "G0",
    "N_STATES", "N_INPUTS", "N_OUTPUTS",
    "_IDX_VS", "_IDX_GRAV", "_IDX_HEAD",
    "_IDX_INPUT_CANAL", "_IDX_INPUT_SLIP", "_IDX_INPUT_GIA",
    "_IDX_INPUT_SCENE_LIN_VEL", "_IDX_INPUT_SCENE_VISIBLE",
]
