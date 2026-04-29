"""Velocity Storage SSM — push-pull bilateral architecture.

Two populations (A, B) model the bilateral vestibular nucleus (VN).
Each population is defined by its preferred direction in head coordinates:

    x_A[0]  yaw:   rotation to the RIGHT   (+y axis)
    x_A[1]  pitch: nose UP                 (−x axis)
    x_A[2]  roll:  right ear DOWN          (+z axis)
    x_B[0]  yaw:   rotation to the LEFT    (−y axis)
    x_B[1]  pitch: nose DOWN               (+x axis)
    x_B[2]  roll:  left ear DOWN           (−z axis)

Both populations are excitatory; push-pull arises from opposite preferred directions.
PINV_SENS converts 6 canal afferents (canal coordinates) → 3 head-frame axes [yaw, pitch, roll].

State:  x_vs = [x_A (3,) | x_B (3,) | x_null (3,)]          (9,)
Input:  u    = [y_canals (6,) | e_slip_delayed (3,) | g_est (3,) | GIA (3,)]  (15,)
Output: w_est = ω̂  (3,)  — net head-velocity estimate → NI

ABCD system (linear core):
────────────────────────────────────────────────────────────────────────
    dx/dt  = A @ (x − b)  +  B @ u_lin  −  rf(GIA, g_est)
    w_est  = C @ x        +  D @ u_lin

    u_lin  = [y_canals (6,) | e_slip_delayed (3,)]  (9,)   linear inputs
    rf     is the rotational feedback — nonlinear, outside ABCD core

    A  (6×6)  =  −(1/τ_vs) · I₆
                 diagonal; one TC per axis, shared across both populations.
                 Future: full 6×6 → cross-axis coupling, asymmetric decay.

    B  (6×9)  =  ⎡  +K_vs·PINV_SENS  │  −K_vis·I₃ ⎤   ← pop A (preferred +)
                 ⎣  −K_vs·PINV_SENS  │  +K_vis·I₃ ⎦   ← pop B (preferred −)
                 Push-pull: canal and visual signs flip between populations.
                 Future: full 6×9 → per-neuron canal and visual weights.

    C  (3×6)  =  [I₃ | −I₃]
                 Net output: x_A − x_B (positive = pop A direction).

    D  (3×9)  =  [PINV_SENS | −g_vis·I₃]
                 Canal and slip feedthrough on the net output.
                 Future: fit D → data-driven feedthrough gains.

    SP (6,)  =  b_vs · 1₆  +  null shift
                 Set point: equilibrium the A-matrix leak drives toward.
                 At rest SP_A = SP_B → no spontaneous output.
                 Null adaptation shifts SP slowly, extending effective TC.
                 Future: fit per-population set point.

────────────────────────────────────────────────────────────────────────
Current simplification — 2 abstract populations (A/B):

    The 6 states represent 2 populations × 3 head-frame axes [yaw, pitch, roll].
    Pop A encodes motion in the positive direction of each axis; Pop B the negative.
    PINV_SENS maps 6 canal afferents → 3 head-frame axes, collapsing canal identity.

Future: 6 canal-matched populations (one per semicircular canal):

    Each VN population maps 1:1 to one canal (LA, RA, LP, RP, LH, RH).
    Its preferred direction is the canal's sensitivity vector (one row of SENS).
    This restores canal identity and enables canal-specific VN lesion modelling:
        - Individual canal plugging (e.g. BPPV repositioning)
        - Unilateral neuritis affecting specific canal afferents
        - SVN (anterior-canal) vs IVN (posterior-canal) lesions
        - Left vs right MVN lesions for horizontal VOR
    Anatomically: LH/RH → MVN (horizontal); LA/RA → SVN; LP/RP → IVN.
    State layout becomes x_vs = [x_LH | x_RH | x_LA | x_RA | x_LP | x_RP | x_null]  (21,)
    B matrix is replaced by a 6×6 identity-like structure in canal coordinates.

Path from scalar params → matrix params (system identification):

    Stage 1 (now):   A = −(1/τ)·I,  B built from K_vs/K_vis scalars
    Stage 2:         A = diag(a),    B = free (6×9)   [per-axis TCs]
    Stage 3:         6-population model, one per canal; B = canal sensitivity matrix

    At each stage the fitting code is identical — only the pytree changes.
    Nonlinearities (rotational feedback, future rectification) remain outside
    the ABCD core and are held fixed during linear fitting.

────────────────────────────────────────────────────────────────────────
Healthy symmetric regime:
    At rest:      x_A = x_B = b  →  C @ x = 0   (no spontaneous nystagmus) ✓
    Head right:   x_A ↑, x_B ↓  →  w_est > 0  →  eye moves left ✓
    Scene right:  x_A ↓, x_B ↑  →  w_est < 0  →  eye follows right ✓

Lesion modelling (once rectification is added):
    b_vs[0:3] → 0  (pop A infarct) →  spontaneous nystagmus in pop-B direction ✓
    canal_gains[0:3] → 0  (partial pop A hypofunction) →  asymmetric drive ✓

Null-point adaptation:
────────────────────────────────────────────────────────────────────────
    dx_null/dt = (w_est − x_null) / τ_vs_adapt

    x_null tracks the VS output; the bias target SP shifts toward x_null,
    extending the effective TC for sustained stimuli.
    Default τ_vs_adapt = 600 s → negligible for normal demos.
    For PAN: set τ_vs_adapt ≈ 30–60 s.

Parameters:
    b_vs       (deg/s) — resting bias (both populations).     Default 100.
    τ_vs       (s)     — storage / OKAN TC (yaw).             Default 20 s.
    K_vs       (1/s)   — canal→VS gain.                       Default 0.1.
    K_vis      (1/s)   — visual→VS gain.                      Default 1.0.
    g_vis              — visual feedthrough.                   Default 0.3.
    K_gd               — rotational feedback gain.             Default 0 (disabled).
    τ_vs_adapt (s)     — null adaptation TC.                   Default 600 s.
"""

import jax.numpy as jnp
from oculomotor.models.sensory_models.sensory_model import PINV_SENS, N_CANALS

N_STATES  = 9   # x_A(3) + x_B(3) + x_null(3)
N_INPUTS  = N_CANALS + 3 + 3   # 6 canal afferents + 3 slip + 3 rf (rotational feedback)
N_OUTPUTS = 3   # w_est (3,)

_IDX_NULL = slice(6, 9)   # null adaptation state within x_vs

B_NOMINAL = 100.0   # healthy resting bias (deg/s); used to derive per-population input scaling


def step(x_vs, u, brain_params):
    """Single ODE step: bilateral VS dynamics + null adaptation + net velocity output.

    Args:
        x_vs:         (9,)   VS state [x_A (3,) | x_B (3,) | x_null (3,)], (deg/s)
        u:            (12,)  [y_canals (6,) | e_slip_delayed (3,) | grav_mismatch (3,)]
                             y_canals:       6 canal afferents (deg/s)
                             e_slip_delayed: scene retinal slip, head frame [yaw, pitch, roll] (deg/s)
                             rf:             rotational feedback from gravity estimator [yaw, pitch, roll]
                                             = xyz_to_ypr(GIA × (−g_est)) / G0²  (Laurens & Angelaki 2011)
        brain_params: BrainParams

    Returns:
        dx:    (9,)  dx_vs/dt  (deg/s²)
        w_est: (3,)  angular velocity estimate → NI (deg/s), [yaw, pitch, roll]
    """
    x_null = x_vs[_IDX_NULL] # (3,) adapted null
    x_pop  = x_vs[:6]        # (6,) [x_A | x_B] both populations (for ABCD)

    canal_in = jnp.clip(u[:N_CANALS], -brain_params.v_max_vor, brain_params.v_max_vor)  # (6,)
    slip_in  = u[N_CANALS:N_CANALS+3]   # (3,) [yaw, pitch, roll] (deg/s)
    rf       = u[N_CANALS+3:]           # (3,) rotational feedback from GE [yaw, pitch, roll]
    u_lin    = jnp.concatenate([canal_in, slip_in])   # (9,) linear inputs

    # b_vs is (6,) float32 — normalised by simulate() before ODE entry.
    SP    = brain_params.b_vs + jnp.concatenate([x_null / 2.0, -x_null / 2.0])  # (6,) set point
    g_pop = brain_params.b_vs / B_NOMINAL   # (6,) population health: 1=healthy, 0=silent

    # ── ABCD matrices ──────────────────────────────────────────────────────────

    # A (6×6): per-axis decay [yaw, pitch, roll] × 2 populations.
    tau3   = jnp.array([brain_params.tau_vs,
                         brain_params.tau_vs * brain_params.tau_vs_pitch_frac,
                         brain_params.tau_vs * brain_params.tau_vs_roll_frac])
    inv_t6 = jnp.concatenate([1.0 / tau3, 1.0 / tau3])   # same TC for both pops
    A = -jnp.diag(inv_t6)

    # B (6×9): canal and visual inputs, push-pull across populations.
    # Future: canal (head-frame) and optokinetic (eye-frame) may need R_eye alignment.
    B_top = jnp.concatenate([ g_pop[:3, None] * brain_params.K_vs * PINV_SENS, -brain_params.K_vis * jnp.eye(3)], axis=1)
    B_bot = jnp.concatenate([-g_pop[3:, None] * brain_params.K_vs * PINV_SENS,  brain_params.K_vis * jnp.eye(3)], axis=1)
    B = jnp.concatenate([B_top, B_bot], axis=0)

    # C (3×6): net output x_A − x_B
    C = jnp.concatenate([jnp.eye(3), -jnp.eye(3)], axis=1)

    # D (3×9): canal and visual feedthrough on net output
    D = jnp.concatenate([brain_params.g_vor * PINV_SENS, -brain_params.g_vis * jnp.eye(3)], axis=1)

    # ── Rotational feedback — Laurens & Angelaki (2011) ───────────────────────
    # rf pre-computed in gravity_estimator: xyz_to_ypr(GIA × (−g_est)) / G0²
    # Zero at steady state (GIA ≈ −g_est); active when gravity estimate lags GIA.
    rf6 = brain_params.K_gd * jnp.concatenate([rf, -rf])   # push-pull across populations

    # ── Bilateral dynamics ─────────────────────────────────────────────────────
    dx_pop = A @ (x_pop - SP) + B @ u_lin - rf6

    # ── Net output ─────────────────────────────────────────────────────────────
    w_est = C @ x_pop + D @ u_lin   # C @ SP = 0 since SP_A − SP_B = 0

    # ── Null adaptation ────────────────────────────────────────────────────────
    dx_null = (w_est - x_null) / brain_params.tau_vs_adapt

    return jnp.concatenate([dx_pop, dx_null]), w_est
