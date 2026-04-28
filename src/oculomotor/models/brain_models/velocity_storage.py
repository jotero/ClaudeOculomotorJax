"""Velocity Storage SSM — push-pull bilateral architecture.

Two populations (Left, Right) model the bilateral vestibular nucleus (VN)
Type I / Type II neuron organization.

State:  x_vs = [x_L (3,) | x_R (3,)]                        (6,)
Input:  u    = [y_canals (6,) | e_slip_delayed (3,) | g_est (3,)]  (12,)
Output: w_est = ω̂  (3,)  — net head-velocity estimate → NI

ABCD system (linear core):
────────────────────────────────────────────────────────────────────────
    dx/dt  = A @ (x − b)  +  B @ u_lin  +  gd(x, g_est)
    w_est  = C @ x        +  D @ u_lin

    u_lin  = [y_canals (6,) | e_slip_delayed (3,)]  (9,)   linear inputs
    g_est  is handled separately in gd() — nonlinear, outside ABCD core

    A  (6×6)  =  −(1/τ_vs) · I₆
                 diagonal; one TC for both populations.
                 Future: full 6×6 → cross-axis coupling, asymmetric decay.

    B  (6×9)  =  ⎡  +K_vs·PINV_SENS  │  −K_vis·I₃ ⎤   ← left  population
                 ⎣  −K_vs·PINV_SENS  │  +K_vis·I₃ ⎦   ← right population
                 Push-pull: opposite canal sign, opposite visual sign.
                 Future: full 6×9 → per-neuron canal and visual weights.

    C  (3×6)  =  [I₃ | −I₃]
                 Reads net signal x_L − x_R.
                 Likely kept fixed (defines L/R convention).

    D  (3×9)  =  [PINV_SENS | −g_vis·I₃]
                 Canal and slip feedthrough on the net output.
                 Future: fit D → data-driven feedthrough gains.

    b  (6,)   =  b_vs · 1₆
                 Equilibrium (intrinsic VN resting state, deg/s units).
                 Future: fit per-neuron bias vector.

────────────────────────────────────────────────────────────────────────
Path from scalar params → matrix params (system identification):

    Stage 1 (now):   A = −(1/τ)·I,  B built from K_vs/K_vis scalars
    Stage 2:         A = diag(a),    B = free (6×9)   [per-axis TCs]
    Stage 3:         A free (6×6),   B free (6×9)     [full coupling]

    At each stage the fitting code is identical — only the pytree changes.
    Nonlinearities (gravity coupling, future rectification) remain outside
    the ABCD core and are held fixed during linear fitting.

────────────────────────────────────────────────────────────────────────
Healthy symmetric regime:
    At rest:  x_L = x_R = b  →  C @ x = 0   (no spontaneous nystagmus) ✓
    Head right (+):  x_L ↑, x_R ↓  →  w_est > 0  →  eye moves left ✓
    Scene right (+): x_L ↓, x_R ↑  →  w_est < 0  →  eye follows right ✓
    C @ x = x_L − x_R is identical to old scalar x_vs in linear regime.

Lesion modelling (once rectification is added):
    Left VN lesion:  b_L → 0  →  spontaneous rightward nystagmus ✓
    Left neuritis:   canal_gains_L → 0  →  asymmetric canal drive ✓

Null-point adaptation (mirrors neural_integrator):
────────────────────────────────────────────────────────────────────────
    State layout:  x_vs = [x_L (3,) | x_R (3,) | x_null (3,)]    (9,)

    dx_null/dt = (w_est − x_null) / τ_vs_adapt

    x_null tracks the current VS output.  The VS leak target shifts toward
    x_null (via b_eff), so sustained activation extends effective TC.
    With τ_vs_adapt >> τ_vs (default 600 s), effect is negligible in normal
    demos — only becomes significant for very-long stimuli (PAN modelling).

    For PAN: set τ_vs_adapt ≈ 30–60 s.  The null adaptation accumulates
    during spontaneous nystagmus, providing the slow oscillatory drive.

Parameters:
    b_vs       (deg/s) — intrinsic VN resting bias.          Default 100.
    τ_vs       (s)     — storage / OKAN TC.                  Default 20 s.
    K_vs       (1/s)   — canal→VS gain.                      Default 0.1.
    K_vis      (1/s)   — visual→VS gain.                     Default 1.0.
    g_vis              — visual feedthrough.                  Default 0.3.
    K_gd       (1/s)   — gravity dumping gain.               Default 0 (disabled).
    τ_vs_adapt (s)     — null adaptation TC.                  Default 600 s.
"""

import jax.numpy as jnp
from oculomotor.models.sensory_models.sensory_model import PINV_SENS, N_CANALS

N_STATES  = 9   # x_L(3) + x_R(3) + x_null(3)
N_INPUTS  = N_CANALS + 3 + 3 + 3   # 6 canal afferents + 3 slip + 3 g_est + 3 f_oto
N_OUTPUTS = 3   # w_est (3,)

# Sub-index slices (relative to x_vs)
_IDX_L    = slice(0, 3)
_IDX_R    = slice(3, 6)
_IDX_NULL = slice(6, 9)

B_NOMINAL = 100.0   # healthy resting bias (deg/s); used to derive per-population input scaling


def step(x_vs, u, brain_params):
    """Single ODE step: bilateral VS dynamics + null adaptation + net velocity output.

    Args:
        x_vs:         (9,)   VS state [x_L (3,) | x_R (3,) | x_null (3,)]
                             axes: [x=up/yaw, y=left/interaural, z=fwd/naso-occipital] (deg/s)
        u:            (15,)  [y_canals (6,) | e_slip_delayed (3,) | g_est (3,) | f_oto (3,)]
                             y_canals:       6 semicircular canal afferents (deg/s)
                             e_slip_delayed: retinal slip, head frame [x=up, y=left, z=fwd] (deg/s)
                             g_est:          gravity estimate from GE state, head frame (m/s²)
                                             [x=up/yaw, y=left/interaural, z=fwd/naso-occipital]
                                             upright rest: [+9.81, 0, 0]
                             f_oto:          raw GIA from otolith, same head frame (m/s²)
                                             = gravity + linear acceleration (specific force)
                                             Used for gravity dumping: cross(f_oto, g_est).
        brain_params: BrainParams

    Returns:
        dx:    (9,)  dx_vs/dt  — same head frame as x_vs
        w_est: (3,)  angular velocity estimate → NI (deg/s), head frame [x=up, y=left, z=fwd]
    """
    x_L    = x_vs[_IDX_L]    # (3,) left  pop
    x_R    = x_vs[_IDX_R]    # (3,) right pop
    x_null = x_vs[_IDX_NULL] # (3,) adapted null
    x_pop  = x_vs[:6]        # (6,) bilateral populations (for ABCD)

    canal_in = jnp.clip(u[:N_CANALS], -brain_params.v_max_vor, brain_params.v_max_vor)  # (6,)
    slip_in  = u[N_CANALS:N_CANALS+3]    # (3,) head frame [x=up, y=left, z=fwd]
    g_est    = u[N_CANALS+3:N_CANALS+6]  # (3,) gravity estimate, head frame (m/s²)
    f_oto    = u[N_CANALS+6:]            # (3,) raw GIA from otolith, head frame (m/s²)
    u_lin    = jnp.concatenate([canal_in, slip_in])   # (9,) linear inputs

    # Population equilibria: b_vs bias ± half-null shift
    b6     = jnp.broadcast_to(jnp.asarray(brain_params.b_vs, dtype=jnp.float32), (6,))
    b_eff  = b6 + jnp.concatenate([x_null / 2.0, -x_null / 2.0])  # (6,)

    # Per-population health [0–1]: derived from b_vs relative to healthy nominal.
    # b_vs = 100 (healthy scalar) → g_pop = [1,…,1] → full canal drive.
    # b_vs[3:6] = 0  (left VN infarct) → g_pop[3:6] = 0 → canal drive and bias both zero.
    # b_vs[3:6] = 70 (left UVH)        → g_pop[3:6] = 0.7 → partial drive.
    # This ensures x initialised to b_vs is already at equilibrium — no spurious transient.
    g_pop  = b6 / B_NOMINAL                                         # (6,)

    # ── ABCD matrices ──────────────────────────────────────────────────────────

    # A (6×6): per-axis decay  [yaw, pitch, roll] × 2 populations.
    tau3   = jnp.array([brain_params.tau_vs,
                         brain_params.tau_vs * brain_params.tau_vs_pitch_frac,
                         brain_params.tau_vs * brain_params.tau_vs_roll_frac])
    inv_t6 = jnp.concatenate([1.0 / tau3, 1.0 / tau3])   # (6,) same per-axis TC for both pops
    A = -jnp.diag(inv_t6)

    # B (6×9): push-pull canal and visual inputs, scaled by per-population health
    # Future work: correctly combining canal (head-frame) and optokinetic (eye-frame)
    # inputs may require applying current eye position (R_eye) to one of them so both
    # are expressed in the same frame before summation.
    B_top = jnp.concatenate([ g_pop[:3, None] * brain_params.K_vs * PINV_SENS, -brain_params.K_vis * jnp.eye(3)], axis=1)
    B_bot = jnp.concatenate([-g_pop[3:, None] * brain_params.K_vs * PINV_SENS,  brain_params.K_vis * jnp.eye(3)], axis=1)
    B = jnp.concatenate([B_top, B_bot], axis=0)

    # C (3×6): read net signal x_L − x_R
    C = jnp.concatenate([jnp.eye(3), -jnp.eye(3)], axis=1)

    # D (3×9): feedthrough on net output — canal + visual
    D = jnp.concatenate([brain_params.g_vor * PINV_SENS, -brain_params.g_vis * jnp.eye(3)], axis=1)

    # ── Gravity dumping — mismatch between GIA and gravity estimate ─────────────
    # cross(f_oto, g_est) / |g_est|² measures the angular error between the raw
    # otolith measurement and the gravity estimate.  Zero when they are aligned
    # (upright head OR stable tilt with well-calibrated g_est); nonzero when VS
    # transport has rotated g_est away from f_oto during post-rotatory nystagmus.
    # Push-pull: +gd drives L pop down, -gd drives R pop up (net = 2×gd_signal).
    g_norm_sq  = jnp.dot(g_est, g_est) + 1e-9
    gd_signal  = brain_params.K_gd * jnp.cross(f_oto, g_est) / g_norm_sq
    gd         = jnp.concatenate([gd_signal, -gd_signal])

    # ── Bilateral dynamics: leak toward adapted bias ───────────────────────────
    dx_pop = A @ (x_pop - b_eff) + B @ u_lin + gd

    # ── Net output ─────────────────────────────────────────────────────────────
    w_est = C @ x_pop + D @ u_lin   # (C @ b_eff = 0 since b_eff_L − b_eff_R = x_null − x_null = 0)

    # ── Null adaptation: null tracks w_est ────────────────────────────────────
    dx_null = (w_est - x_null) / brain_params.tau_vs_adapt

    return jnp.concatenate([dx_pop, dx_null]), w_est
