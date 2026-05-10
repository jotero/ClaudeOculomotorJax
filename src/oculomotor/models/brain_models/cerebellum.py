"""Cerebellum — single-rule prediction-error correction.

Framework (see docs/cerebellum.md):

    u_cereb = K_cereb · ( s_actual − s_pred )                                  (Eq 1)

The brainstem retains its own reactive feedback on the actual sensed signal;
the cerebellum applies a separate gain on the prediction error against an
internal forward model.  Each region differs only in WHAT it predicts and
WHERE it injects the residual.

Regions are split by prediction class:

  Constant prediction (setpoint shift; cerebellum.md §3 limit) ──────────────
    flocculus            (FL)    →  VS net  (gaze-stabilisation slip setpoint)
    paraflocculus        (VPF)   →  NI net  (gaze-holding setpoint)
    vermis_fastigial     (V/CFN) →  NI net  (tonic gaze offset)
    nodulus_uvula        (NU)    →  VS axis (gravity-axis setpoint)

  Dynamic prediction (forward model; cerebellum.md §4.3) ───────────────────
    pursuit_paraflocc    (PU)    →  pursuit drive
        s_actual = target_slip   (delayed retinal target velocity, eye frame)
        s_pred   = −ec_d_target  (predicted slip from self-motion only —
                                  stationary-target prior; ec_d_target lives
                                  in efference_copy.py and is rotated/saturated/
                                  cascaded to match the target slip path)
        pred_err = target_slip + ec_d_target   (residual ≈ true v_target)
        drive    = K_pu · K_mag · K_dir · pred_err
            K_mag (Hill on |ec_d_target|)  closes during fast self-motion
            K_dir (sigmoid on slip · ec_d alignment)  suppresses opposite-sign
                  residuals (self-motion case) and passes aligned/zero-slip
                  residuals (steady pursuit, fixation).

The drive's sign matches doc Eq 1; consuming subsystems wire it according to
their own dynamics:
  - For populations whose s_actual IS the population state (gaze hold), the
    consumer adds −drive to dx/dt (or equivalently sets K negative) to pin
    state to s_pred.
  - For populations whose s_actual is an upstream signal (pursuit), the
    consumer adds +drive forward as integrator input.

Lesion model: g_cereb · g_<region> · K_<region> → 0 zeros the drive while
leaving pred_err as a pure diagnostic.

State:  none  (N_STATES = 0).  The pursuit region's "forward model" is just
the already-cascaded ec_d_target — its delay buffer lives in
efference_copy.py, not here.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


N_STATES = 0


# ─── Params ──────────────────────────────────────────────────────────────────

class CerebellumParams(NamedTuple):
    """Cerebellum parameters — flat NamedTuple, hand-set, no learning."""

    # Lesion knobs (1.0 = intact, 0.0 = fully lesioned)
    g_cereb: float = 1.0
    g_flocculus: float = 1.0
    g_paraflocculus: float = 1.0
    g_vermis_fastigial: float = 1.0
    g_nodulus_uvula: float = 1.0
    g_pursuit: float = 1.0

    # Per-region constant prediction (3-vector — yaw/pitch/roll or axis).
    # The pursuit region does NOT use a constant theta_bias; its prediction is
    # generated dynamically from ec_d_target.
    theta_bias_fl:   tuple = (0.0, 0.0, 0.0)
    theta_bias_vpf:  tuple = (0.0, 0.0, 0.0)
    theta_bias_vcfn: tuple = (0.0, 0.0, 0.0)
    theta_bias_nu:   tuple = (0.0, 0.0, 0.0)

    # Per-region prediction-error gain (1/s for setpoint regions; dimensionless
    # for pursuit since pred_err and target_slip are both deg/s).
    K_fl:   float = 0.5
    K_vpf:  float = 0.5
    K_vcfn: float = 0.5
    K_nu:   float = 0.5
    K_pu:   float = 1.0   # Stage 1 default: drive_pu equals the pre-refactor
                          # pt.target_slip_for_pursuit (K_mag·K_dir·pred_err) so
                          # behavior is byte-equivalent before downstream tuning.


def with_cerebellum(theta: CerebellumParams, **overrides) -> CerebellumParams:
    """Return a copy of `theta` with the named fields replaced."""
    return theta._replace(**overrides)


# ─── Inputs / Outputs ────────────────────────────────────────────────────────

class CerebellumInputs(NamedTuple):
    """Brainstem signals the cerebellum reads."""
    x_vs_net:        jnp.ndarray   # (3,)  VS net signal     (deg/s)
    x_ni_net:        jnp.ndarray   # (3,)  NI net signal     (deg)
    x_vs_axis:       jnp.ndarray   # (3,)  VS axis signal    (axis units)
    target_slip:     jnp.ndarray   # (3,)  delayed retinal target velocity, eye frame (deg/s)
    target_visible:  jnp.ndarray   # scalar delayed cyclopean target visibility ∈ [0,1]
    ec_d_target:     jnp.ndarray   # (3,)  delayed EC matched to target_vel cascade (deg/s)


class _RegionOut(NamedTuple):
    """Internal: single-region readout (bias, pred_err, drive)."""
    bias: jnp.ndarray
    pred_err: jnp.ndarray
    drive: jnp.ndarray


class CerebellumOut(NamedTuple):
    """Per-region readouts and drives (each field shape (3,))."""
    # Constant-prediction regions
    bias_fl:       jnp.ndarray
    pred_err_fl:   jnp.ndarray
    drive_fl:      jnp.ndarray
    bias_vpf:      jnp.ndarray
    pred_err_vpf:  jnp.ndarray
    drive_vpf:     jnp.ndarray
    bias_vcfn:     jnp.ndarray
    pred_err_vcfn: jnp.ndarray
    drive_vcfn:    jnp.ndarray
    bias_nu:       jnp.ndarray
    pred_err_nu:   jnp.ndarray
    drive_nu:      jnp.ndarray
    # Dynamic-prediction region (pursuit)
    pred_err_pu:   jnp.ndarray   # raw pred error, ungated (target_slip + ec_d_target)
    drive_pu:      jnp.ndarray   # K_pu · K_mag · K_dir · pred_err  → pursuit integrator


# ─── Region rule ─────────────────────────────────────────────────────────────

def _region(K_eff, s_actual, s_pred) -> _RegionOut:
    """One cerebellar region per doc Eq 1: u_cereb = K · (s_actual − s_pred).

        bias     = K_eff · s_pred                  (diagnostic only)
        pred_err = s_actual − s_pred
        drive    = K_eff · pred_err

    Sign convention: positive K_eff transmits the prediction error forward.
    Use negative K_eff (or subtract `drive` at the consumer) for setpoint
    regions where the cerebellar contribution must pull the population state
    toward s_pred (cerebellum.md §3, Eq 3).
    """
    s_pred   = jnp.asarray(s_pred, dtype=jnp.float32)
    pred_err = s_actual - s_pred
    bias     = K_eff * s_pred
    drive    = K_eff * pred_err
    return _RegionOut(bias=bias, pred_err=pred_err, drive=drive)


def _pursuit_region(K_eff, target_slip, target_visible, ec_d_target,
                    v_crit, n_hill, alpha_dir, bias_dir) -> _RegionOut:
    """Dynamic-prediction region for smooth pursuit.

    Forward model: stationary-target prior → predicted slip = −self-motion EC.
    Visibility-gated so EC contribution is zero whenever the target was
    invisible at the delayed time (raw target_slip is also zero in that case).
    Torsion is zeroed because the retina is 2-D.

        s_pred   = − target_visible · ec_d_target_no_torsion
        pred_err = target_slip − s_pred  =  target_slip + visible·ec_d_target
        drive    = K_eff · K_mag · K_dir · pred_err

    K_mag closes during fast self-motion; K_dir suppresses opposite-sign
    residuals (true self-motion case) and passes aligned/zero-slip residuals
    (steady pursuit, fixation).  Both gates were previously in
    perception_target.py and are part of the cerebellar trust signal — they
    decide WHEN to believe the prediction error.

    Args:
        K_eff:            effective per-region gain (g_cereb · g_pu · K_pu)
        target_slip:      (3,)  delayed retinal target velocity, eye frame (deg/s)
        target_visible:   scalar delayed cyclopean target visibility ∈ [0,1]
        ec_d_target:      (3,)  delayed EC matched to target_vel cascade (deg/s)
        v_crit, n_hill:   Hill magnitude-gate parameters (BrainParams)
        alpha_dir, bias_dir: sigmoid directional-gate parameters (BrainParams)
    """
    ec_no_torsion = ec_d_target.at[2].set(0.0)
    s_pred        = -target_visible * ec_no_torsion
    pred_err      = target_slip - s_pred           # = target_slip + visible·ec_no_torsion

    # Hill magnitude gate on |ec_d_target| (full vector incl. torsion is fine
    # for the magnitude — it just measures self-motion speed).
    K_mag = 1.0 / (1.0 + (jnp.linalg.norm(ec_d_target) / v_crit) ** n_hill)

    # Directional gate: signed projection of raw delayed slip onto ec_d
    # direction.  Slip and ec_d anti-parallel (self-motion residual) → close
    # gate.  Aligned or near-zero slip (real motion / pursuit / fixation) →
    # keep gate open.
    ec_norm  = jnp.linalg.norm(ec_d_target) + 1e-9
    ec_hat   = ec_d_target / ec_norm
    slip_dot = jnp.dot(target_slip, ec_hat)
    K_dir    = jax.nn.sigmoid((slip_dot + bias_dir) * alpha_dir)

    drive = K_eff * K_mag * K_dir * pred_err
    bias  = K_eff * s_pred                         # diagnostic only
    return _RegionOut(bias=bias, pred_err=pred_err, drive=drive)


# ─── Step ────────────────────────────────────────────────────────────────────

def step(x_cereb, inputs: CerebellumInputs, theta: CerebellumParams,
         brain_params=None):
    """Single ODE step for the cerebellum.

    Args:
        x_cereb:      (0,) zero-length state — cerebellum is stateless
        inputs:       CerebellumInputs
        theta:        CerebellumParams (cerebellum-specific gains/biases/lesions)
        brain_params: BrainParams — optional; required only when the pursuit
                      region is engaged (it reads the shared Hill / directional
                      gate parameters v_crit_ec_gate, n_ec_gate, alpha_ec_dir,
                      bias_ec_dir from BrainParams).  Pass None for standalone
                      smoke tests of the constant-prediction regions.

    Returns:
        dx:  (0,) zero-length state derivative
        out: CerebellumOut
    """
    g = theta.g_cereb
    fl  = _region(g * theta.g_flocculus        * theta.K_fl,
                  inputs.x_vs_net,  theta.theta_bias_fl)
    vpf = _region(g * theta.g_paraflocculus    * theta.K_vpf,
                  inputs.x_ni_net,  theta.theta_bias_vpf)
    vc  = _region(g * theta.g_vermis_fastigial * theta.K_vcfn,
                  inputs.x_ni_net,  theta.theta_bias_vcfn)
    nu  = _region(g * theta.g_nodulus_uvula    * theta.K_nu,
                  inputs.x_vs_axis, theta.theta_bias_nu)

    if brain_params is None:
        # Standalone path (tests / no pursuit wiring).  Pursuit drive is zero;
        # pred_err still reports the raw residual so it remains diagnosable.
        zero3 = jnp.zeros(3, dtype=jnp.float32)
        ec_no_t = inputs.ec_d_target.at[2].set(0.0)
        pe_pu   = inputs.target_slip + inputs.target_visible * ec_no_t
        pu_out  = _RegionOut(bias=zero3, pred_err=pe_pu, drive=zero3)
    else:
        pu_out = _pursuit_region(
            g * theta.g_pursuit * theta.K_pu,
            inputs.target_slip, inputs.target_visible, inputs.ec_d_target,
            brain_params.v_crit_ec_gate, brain_params.n_ec_gate,
            brain_params.alpha_ec_dir, brain_params.bias_ec_dir,
        )

    out = CerebellumOut(
        bias_fl=fl.bias,    pred_err_fl=fl.pred_err,    drive_fl=fl.drive,
        bias_vpf=vpf.bias,  pred_err_vpf=vpf.pred_err,  drive_vpf=vpf.drive,
        bias_vcfn=vc.bias,  pred_err_vcfn=vc.pred_err,  drive_vcfn=vc.drive,
        bias_nu=nu.bias,    pred_err_nu=nu.pred_err,    drive_nu=nu.drive,
        pred_err_pu=pu_out.pred_err, drive_pu=pu_out.drive,
    )

    dx = jnp.zeros((0,), dtype=jnp.float32)
    return dx, out


# ─── Smoke tests ─────────────────────────────────────────────────────────────
# Run with:
#   python -X utf8 -m oculomotor.models.brain_models.cerebellum

if __name__ == "__main__":
    import numpy as np

    def _close(a, b, tol=1e-6):
        return np.allclose(np.asarray(a), np.asarray(b), atol=tol)

    def _check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            raise AssertionError(name)

    print("Cerebellum smoke tests")
    print("=" * 60)

    x0 = jnp.zeros((0,), dtype=jnp.float32)
    zero3 = jnp.zeros(3, dtype=jnp.float32)
    inputs0 = CerebellumInputs(
        x_vs_net=zero3, x_ni_net=zero3, x_vs_axis=zero3,
        target_slip=zero3, target_visible=jnp.float32(0.0), ec_d_target=zero3,
    )

    # Shape contract
    print("\n[1] Shape contract")
    dx, out = step(x0, inputs0, CerebellumParams())
    _check("dx.shape == (0,)", dx.shape == (0,))
    for f in out._fields:
        _check(f"out.{f}.shape == (3,)", getattr(out, f).shape == (3,))

    # Zero-input identity
    print("\n[2] Zero-input identity")
    for f in out._fields:
        _check(f"{f} == 0", _close(getattr(out, f), zero3))

    # Pure-bias case (FL): drive == K · pred_err == K · (s_actual − s_pred) == K · (0 − θ_bias) == −θ_bias
    print("\n[3] Pure-bias case (FL): drive = K · (s_actual − s_pred)")
    theta = CerebellumParams(theta_bias_fl=(1.0, 0.0, 0.0), K_fl=1.0)
    _, out = step(x0, inputs0, theta)
    _check("bias_fl == (1, 0, 0)",      _close(out.bias_fl,     [1, 0, 0]))
    _check("pred_err_fl == (-1, 0, 0)", _close(out.pred_err_fl, [-1, 0, 0]))
    _check("drive_fl == (-1, 0, 0)",    _close(out.drive_fl,    [-1, 0, 0]))
    _check("other regions unchanged",   _close(out.bias_vpf, zero3) and _close(out.drive_nu, zero3))

    # Pure-residual case (FL)
    print("\n[4] Pure-residual case (FL)")
    theta = CerebellumParams(K_fl=1.0)
    inputs = inputs0._replace(x_vs_net=jnp.array([1.0, 0.0, 0.0]))
    _, out = step(x0, inputs, theta)
    _check("bias_fl == 0",             _close(out.bias_fl,     zero3))
    _check("pred_err_fl == (1, 0, 0)", _close(out.pred_err_fl, [1, 0, 0]))
    _check("drive_fl == (1, 0, 0)",    _close(out.drive_fl,    [1, 0, 0]))

    # Algebraic identity drive == K_eff · pred_err
    print("\n[5] Algebraic identity (constant-prediction regions)")
    theta = CerebellumParams(
        theta_bias_fl=(0.7, -0.3, 0.4),    theta_bias_vpf=(2.0, 1.0, -1.5),
        theta_bias_vcfn=(0.5, 0.5, 0.5),   theta_bias_nu=(-0.1, 0.2, -0.3),
        K_fl=0.4, K_vpf=0.6, K_vcfn=0.8, K_nu=1.2,
    )
    inputs = inputs0._replace(
        x_vs_net=jnp.array([0.2, 0.1, -0.4]),
        x_ni_net=jnp.array([1.5, 0.7, -2.0]),
        x_vs_axis=jnp.array([0.0, 0.5, 0.0]),
    )
    _, out = step(x0, inputs, theta)
    K_eff = {
        "fl":   theta.g_cereb * theta.g_flocculus        * theta.K_fl,
        "vpf":  theta.g_cereb * theta.g_paraflocculus    * theta.K_vpf,
        "vcfn": theta.g_cereb * theta.g_vermis_fastigial * theta.K_vcfn,
        "nu":   theta.g_cereb * theta.g_nodulus_uvula    * theta.K_nu,
    }
    for r in ("fl", "vpf", "vcfn", "nu"):
        pe, drv = (getattr(out, f"{k}_{r}") for k in ("pred_err", "drive"))
        _check(f"drive_{r} == K_eff · pred_err_{r}",
               _close(drv, K_eff[r] * pe))

    # Global lesion: drive zero, pred_err unchanged
    print("\n[6] Global lesion (g_cereb = 0)")
    theta_lesion = CerebellumParams(
        g_cereb=0.0,
        theta_bias_fl=(1.0, 1.0, 1.0), theta_bias_vpf=(2.0, 2.0, 2.0),
        K_fl=10.0, K_vpf=10.0, K_vcfn=10.0, K_nu=10.0, K_pu=10.0,
    )
    inputs = inputs0._replace(
        x_vs_net=jnp.array([3.0, 4.0, 5.0]),
        x_ni_net=jnp.array([6.0, 7.0, 8.0]),
        x_vs_axis=jnp.array([9.0, 10.0, 11.0]),
    )
    _, out = step(x0, inputs, theta_lesion)
    for r in ("fl", "vpf", "vcfn", "nu"):
        _check(f"drive_{r} == 0",  _close(getattr(out, f"drive_{r}"), zero3))
    _check("pred_err_fl unchanged by lesion", not _close(out.pred_err_fl, zero3))

    # Per-region lesion
    print("\n[7] Per-region lesion identity")
    theta_with_bias = CerebellumParams(
        theta_bias_fl=(1.0, 0.0, 0.0),    theta_bias_vpf=(0.0, 1.0, 0.0),
        theta_bias_vcfn=(0.0, 0.0, 1.0),  theta_bias_nu=(1.0, 1.0, 1.0),
        K_fl=1.0, K_vpf=1.0, K_vcfn=1.0, K_nu=1.0,
    )
    region_to_field = {
        "g_flocculus": "fl", "g_paraflocculus": "vpf",
        "g_vermis_fastigial": "vcfn", "g_nodulus_uvula": "nu",
    }
    for knob, suf in region_to_field.items():
        theta_one = with_cerebellum(theta_with_bias, **{knob: 0.0})
        _, out = step(x0, inputs0, theta_one)
        _check(f"drive_{suf} == 0 ({knob}=0)", _close(getattr(out, f"drive_{suf}"), zero3))
        for other in ("fl", "vpf", "vcfn", "nu"):
            if other == suf:
                continue
            _check(f"drive_{other} != 0 ({knob}=0)",
                   not _close(getattr(out, f"drive_{other}"), zero3))

    # Pursuit region: standalone path (brain_params=None) gives raw pred_err, zero drive
    print("\n[8] Pursuit region — standalone path (no brain_params)")
    inputs_pu = inputs0._replace(
        target_slip=jnp.array([3.0, 0.0, 0.0]),
        target_visible=jnp.float32(1.0),
        ec_d_target=jnp.array([2.0, 0.0, 5.0]),   # torsion 5 should be zeroed
    )
    _, out = step(x0, inputs_pu, CerebellumParams(K_pu=1.0))
    _check("pred_err_pu == target_slip + visible · ec (torsion zeroed)",
           _close(out.pred_err_pu, [5.0, 0.0, 0.0]))
    _check("drive_pu == 0 (no brain_params)", _close(out.drive_pu, zero3))

    # Pursuit region: full path with brain_params
    print("\n[9] Pursuit region — full path (with brain_params gates)")
    class _BP(NamedTuple):
        v_crit_ec_gate: float = 50.0
        n_ec_gate:      float = 6.0
        alpha_ec_dir:   float = 0.4
        bias_ec_dir:    float = 15.0
    bp = _BP()
    # Quiet self-motion: K_mag ≈ 1, K_dir ≈ sigmoid(15·0.4)·≈1 (slip_dot ≈ 0)
    inputs_quiet = inputs0._replace(
        target_slip=jnp.array([10.0, 0.0, 0.0]),
        target_visible=jnp.float32(1.0),
        ec_d_target=jnp.array([0.0, 0.0, 0.0]),
    )
    _, out = step(x0, inputs_quiet, CerebellumParams(K_pu=1.0), brain_params=bp)
    # K_dir saturates at sigmoid(bias·alpha) = sigmoid(15·0.4) ≈ 0.9975 when slip_dot=0,
    # so drive ≈ 0.9975 · pred_err (not exactly equal — gate is asymptotic).
    K_dir_open = float(jax.nn.sigmoid(jnp.float32(bp.bias_ec_dir * bp.alpha_ec_dir)))
    _check("drive_pu ≈ K_dir_open · pred_err when ec_d_target=0",
           _close(out.drive_pu, K_dir_open * out.pred_err_pu, tol=1e-4))
    # Lesion: g_pursuit=0
    _, out_les = step(x0, inputs_quiet, CerebellumParams(K_pu=1.0, g_pursuit=0.0), brain_params=bp)
    _check("drive_pu == 0 (g_pursuit=0)", _close(out_les.drive_pu, zero3))
    _check("pred_err_pu unchanged by lesion",
           _close(out_les.pred_err_pu, out.pred_err_pu))
    # Fast self-motion: K_mag closes
    inputs_fast = inputs0._replace(
        target_slip=jnp.array([10.0, 0.0, 0.0]),
        target_visible=jnp.float32(1.0),
        ec_d_target=jnp.array([200.0, 0.0, 0.0]),   # huge self-motion
    )
    _, out_fast = step(x0, inputs_fast, CerebellumParams(K_pu=1.0), brain_params=bp)
    K_mag_expected = 1.0 / (1.0 + (200.0 / 50.0) ** 6.0)
    _check("|drive_pu| < 0.1·|pred_err_pu| when |ec_d| >> v_crit",
           jnp.linalg.norm(out_fast.drive_pu) < 0.1 * jnp.linalg.norm(out_fast.pred_err_pu))
    _check(f"K_mag matches Hill formula ({K_mag_expected:.4f})",
           K_mag_expected < 0.001)

    # with_cerebellum
    print("\n[10] with_cerebellum helper")
    base = CerebellumParams()
    edited = with_cerebellum(base, K_fl=2.5, theta_bias_vpf=(0.1, 0.2, 0.3),
                              K_pu=0.7, g_pursuit=0.5)
    _check("K_fl updated",            edited.K_fl == 2.5)
    _check("theta_bias_vpf updated",  edited.theta_bias_vpf == (0.1, 0.2, 0.3))
    _check("K_pu updated",            edited.K_pu == 0.7)
    _check("g_pursuit updated",       edited.g_pursuit == 0.5)
    _check("g_cereb unchanged",       edited.g_cereb == base.g_cereb)
    _check("theta_bias_fl unchanged", edited.theta_bias_fl == base.theta_bias_fl)

    # JIT
    print("\n[11] JIT compatibility (with brain_params)")
    step_jit = jax.jit(step)
    _, out_jit = step_jit(x0, inputs0, CerebellumParams(), bp)
    _check("JIT compile + run", out_jit is not None)
    _check("JIT output shape",  out_jit.bias_fl.shape == (3,))
    _check("JIT pursuit shape", out_jit.drive_pu.shape == (3,))

    print("\n" + "=" * 60)
    print("All smoke tests passed.")
