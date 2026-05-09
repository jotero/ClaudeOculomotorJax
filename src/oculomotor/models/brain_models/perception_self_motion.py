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

from typing import NamedTuple

import jax
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

# ── State + registries ────────────────────────────────────────────────────────
# VS pops are called A/B internally (population A=preferred, B=opposite), but
# exposed externally as L/R to match the codebase-wide convention
# (model L pop ≡ A; net = L − R = A − B).

class State(NamedTuple):
    """Self-motion state: VS pops + null + GE observer + heading."""
    vs_L:    jnp.ndarray   # (3,)  left  VN pop  [≡ internal pop A]
    vs_R:    jnp.ndarray   # (3,)  right VN pop  [≡ internal pop B]
    vs_null: jnp.ndarray   # (3,)  VS adaptation register (slow null drift)
    g_est:   jnp.ndarray   # (3,)  gravity estimate (head frame)  [VN/cb gravity cells]
    a_lin:   jnp.ndarray   # (3,)  linear-accel estimate          [VN linear-accel cells]
    rf:      jnp.ndarray   # (3,)  rotational feedback            [Laurens observer]
    v_lin:   jnp.ndarray   # (3,)  head linear velocity estimate  [MST / heading cells]


class Activations(NamedTuple):
    """Self-motion firing rates (VS pops + GE/HE observer)."""
    vs_R:  jnp.ndarray
    vs_L:  jnp.ndarray
    g_est: jnp.ndarray
    a_lin: jnp.ndarray
    rf:    jnp.ndarray
    v_lin: jnp.ndarray


class Decoded(NamedTuple):
    """VS net head angular velocity readout."""
    vs_net: jnp.ndarray   # (3,) signed = vs_L − vs_R   head ang vel estimate (deg/s)


class Weights(NamedTuple):
    """VS adaptation register (long-term: learned weight)."""
    vs_null: jnp.ndarray   # (3,) signed slow-null


def rest_state():
    """Initial state — VS pops at b_vs equilibrium, GE at gravity vertical."""
    return State(
        vs_L    = jnp.zeros(3),
        vs_R    = jnp.zeros(3),
        vs_null = jnp.zeros(3),
        g_est   = X0[_GE_IDX_G] if 'X0' in globals() else jnp.array([G0, 0.0, 0.0]),
        a_lin   = jnp.zeros(3),
        rf      = jnp.zeros(3),
        v_lin   = jnp.zeros(3),
    )


def read_activations(state):
    """Project self-motion State → Activations (firing rates only)."""
    return Activations(
        vs_R  = state.vs_R,
        vs_L  = state.vs_L,
        g_est = state.g_est,
        a_lin = state.a_lin,
        rf    = state.rf,
        v_lin = state.v_lin,
    )


def decode_states(acts):
    return Decoded(vs_net=acts.vs_L - acts.vs_R)


def read_weights(state):
    return Weights(vs_null=state.vs_null)

# Bundled-input layout (length 19) — RAW sensory inputs + ec_d_scene; gating
# (post-delay EC subtraction + magnitude/directional gate on scene slip and
# scene_linear_vel) is now done INSIDE step() using ec_d_scene.
N_INPUTS  = 6 + 3 + 3 + 3 + 1 + 3
N_OUTPUTS = 3   # primary output for SSM convention is w_est; auxiliaries via tuple

_IDX_INPUT_CANAL         = slice(0, 6)
_IDX_INPUT_SLIP          = slice(6, 9)     # RAW delayed scene slip (eye frame)
_IDX_INPUT_GIA           = slice(9, 12)
_IDX_INPUT_SCENE_LIN_VEL = slice(12, 15)   # RAW delayed scene linear velocity
_IDX_INPUT_SCENE_VISIBLE = 15
_IDX_INPUT_EC_D_SCENE    = slice(16, 19)   # delayed EC (cascade-matched to scene slip)


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

    # Canal saturation now applied at the sensor side (canal.step + sensory_model.read_outputs);
    # canal afferents arrive here already clipped at canal_v_max.
    u_lin    = jnp.concatenate([canal, slip])      # (9,) linear inputs

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

    # Kalman-derived state-dependent gain modulation.
    # The bilinear gravity-transport term [ω]_× couples (ω, g) in the EKF Riccati
    # equation; the resulting Kalman gain has the structure:
    #   K_grav,eff = K_grav · √(1 + ρ)         (boost when rotation ⊥ gravity)
    #   K_lin,eff  = K_lin  / √(1 + ρ)         (suppress same regime)
    # where ρ = |ω̂ × ĝ_hat| / w_canal_gate is the "rotation-perpendicular-to-
    # gravity" Bayes factor. Rotation parallel to gravity (e.g. upright yaw)
    # gives ρ → 0 → no gating, since parallel rotations don't change head-frame
    # gravity and produce no spurious otolith residual to misattribute.
    g_hat       = g_est / (jnp.linalg.norm(g_est) + 1e-9)
    w_xyz       = ypr_to_xyz(w_est)
    w_perp_g    = jnp.linalg.norm(jnp.cross(w_xyz, g_hat))   # deg/s
    rho         = w_perp_g / brain_params.w_canal_gate
    gate_factor = jnp.sqrt(1.0 + rho)
    K_grav_eff  = brain_params.K_grav * gate_factor
    K_lin_eff   = brain_params.K_lin  / gate_factor

    # Gravity correction: pulled toward residual with state-modulated K_grav.
    dg = transport + K_grav_eff * residual

    # Linear acceleration: tracks residual (gated), decays toward 0 on TC τ_a_lin
    # (deterministic stand-in for L&A's translation-duration prior).
    da = K_lin_eff * residual - a_lin / brain_params.tau_a_lin

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

def step(state, u, brain_params):
    """Single ODE step for the unified self-motion observer.

    Internal sequencing (matches Laurens & Angelaki 2017):
      0. Post-delay EC subtraction + magnitude / directional gates on the
         RAW delayed scene slip (and on scene_linear_vel by Hill K_mag).
      1. VS uses rf_state from the (1-step delayed) ODE state — breaks the
         VS↔GE algebraic loop with negligible lag (τ_rf_state ≈ 5 ms).
      2. GE then runs with the freshly-computed w_est from VS.
      3. HE consumes a_lin from GE state (1-step delayed) for its own
         leaky integration toward v_lin.

    Args:
        state:         sm.State  bundled state (vs_L/R/null + g_est + a_lin + rf + v_lin)
        u:             (19,) bundled input  — see _IDX_INPUT_* above
                              [canal | scene_slip(raw) | gia | scene_lin_vel(raw)
                               | scene_visible | ec_d_scene]
        brain_params:  BrainParams — reads VS/GE/HE params + v_crit_ec_gate,
                                      n_ec_gate, alpha_ec_dir, bias_ec_dir

    Returns:
        dstate         : sm.State  state derivative
        w_est          : (3,)  angular velocity estimate (deg/s)
        g_est          : (3,)  gravity estimate (m/s², head frame)
        v_lin          : (3,)  head linear velocity (m/s, head frame)
        a_lin_est      : (3,)  linear-acc estimate (m/s², head frame) — for T-VOR direct
    """
    # Repack state into the flat sub-blocks the internal helpers expect.
    x_vs   = jnp.concatenate([state.vs_L, state.vs_R, state.vs_null])
    x_grav = jnp.concatenate([state.g_est, state.a_lin, state.rf])
    x_head = state.v_lin

    # GE state read (rf and a_lin are 1-step delayed via ODE state)
    rf_state  = state.rf
    a_lin_est = state.a_lin

    # Input split — RAW sensory + delayed EC
    canal         = u[_IDX_INPUT_CANAL]
    scene_slip    = u[_IDX_INPUT_SLIP]
    gia           = u[_IDX_INPUT_GIA]
    scene_lin_vel = u[_IDX_INPUT_SCENE_LIN_VEL]
    scene_visible = u[_IDX_INPUT_SCENE_VISIBLE]
    ec_d_scene    = u[_IDX_INPUT_EC_D_SCENE]

    # 0. Post-delay EC subtraction + saccadic-suppression gates on scene path.
    #    Visibility-gating ensures EC contribution = 0 when scene was invisible.
    scene_slip_corr = scene_slip + scene_visible * ec_d_scene
    K_mag_scene     = 1.0 / (1.0 + (jnp.linalg.norm(ec_d_scene) / brain_params.v_crit_ec_gate)
                              ** brain_params.n_ec_gate)
    ec_norm_s       = jnp.linalg.norm(ec_d_scene) + 1e-9
    slip_dot_s      = jnp.dot(scene_slip, ec_d_scene / ec_norm_s)
    K_dir_s         = jax.nn.sigmoid((slip_dot_s + brain_params.bias_ec_dir)
                                      * brain_params.alpha_ec_dir)
    slip_gated      = K_mag_scene * K_dir_s * scene_slip_corr
    scene_lin_gated = K_mag_scene * scene_lin_vel

    # 1. VS — angular velocity estimate
    dx_vs, w_est = _vs_step(x_vs, canal, slip_gated, rf_state, brain_params)

    # 2. GE — gravity + linear-acc estimates (rf updated for next step)
    dx_grav, g_est = _ge_step(x_grav, w_est, gia, brain_params)

    # 3. HE — head linear velocity (consumes the prior step's a_lin to avoid
    #         needing the freshly-computed â here)
    dx_head, v_lin = _he_step(x_head, a_lin_est, scene_lin_gated, scene_visible, brain_params)

    dstate = State(
        vs_L    = dx_vs[_VS_IDX_A],
        vs_R    = dx_vs[_VS_IDX_B],
        vs_null = dx_vs[_VS_IDX_NULL],
        g_est   = dx_grav[_GE_IDX_G],
        a_lin   = dx_grav[_GE_IDX_A],
        rf      = dx_grav[_GE_IDX_RF],
        v_lin   = dx_head,
    )
    return dstate, w_est, g_est, v_lin, a_lin_est


# ── Legacy flat-array adapters (deleted once brain_model migrates to BrainState) ─

N_STATES  = 9 + 9 + 3   # 21
_IDX_VS   = slice(0, 9)
_IDX_GRAV = slice(9, 18)
_IDX_HEAD = slice(18, 21)


def from_array(x_self_motion):
    """(21,) flat array → sm.State."""
    x_vs   = x_self_motion[_IDX_VS]
    x_grav = x_self_motion[_IDX_GRAV]
    return State(
        vs_L    = x_vs[_VS_IDX_A],
        vs_R    = x_vs[_VS_IDX_B],
        vs_null = x_vs[_VS_IDX_NULL],
        g_est   = x_grav[_GE_IDX_G],
        a_lin   = x_grav[_GE_IDX_A],
        rf      = x_grav[_GE_IDX_RF],
        v_lin   = x_self_motion[_IDX_HEAD],
    )


def to_array(state):
    """sm.State → (21,) flat array."""
    return jnp.concatenate([
        state.vs_L, state.vs_R, state.vs_null,
        state.g_est, state.a_lin, state.rf,
        state.v_lin,
    ])


__all__ = [
    "step", "State", "Activations", "Decoded", "Weights",
    "rest_state", "read_activations", "decode_states", "read_weights",
    "from_array", "to_array",
    "X0", "GRAV_X0", "G0",
    "N_STATES", "N_INPUTS", "N_OUTPUTS",
    "_IDX_VS", "_IDX_GRAV", "_IDX_HEAD",
    "_IDX_INPUT_CANAL", "_IDX_INPUT_SLIP", "_IDX_INPUT_GIA",
    "_IDX_INPUT_SCENE_LIN_VEL", "_IDX_INPUT_SCENE_VISIBLE",
]
