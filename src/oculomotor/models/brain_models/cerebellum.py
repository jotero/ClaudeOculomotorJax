"""Cerebellum — single-rule prediction-error correction across four regions.

Framework (see docs/cerebellum.md):

    u_cereb = K_cereb · ( θ_bias − x_target )

The brainstem retains its own reactive feedback on the actual sensed error;
the cerebellum applies a separate gain on the *prediction* error against an
internal forward model. In this constant-prediction scaffold the forward
model is just a hand-set θ_bias per region (no learning).

The drive decomposes into two diagnostically distinct components:

    bias     = K_eff · θ_bias                        (constant component;
                                                      "what setpoint is being
                                                      enforced")
    pred_err = x_target − θ_bias                     (signed residual, no gain;
                                                      "how far off the
                                                      brainstem is right now")
    drive    = bias − K_eff · pred_err               (drive applied to
                                                      brainstem)

Both bias and pred_err are exposed in CerebellumOut even though they share
the same mechanism. Lesion (g_cereb · g_<region> · K_<region> → 0) zeros
bias and drive; pred_err is unaffected and remains a pure diagnostic.

Four regions, four targets:

    flocculus            (FL)    →  VS net signal      (cancel canal-pinv drift)
    paraflocculus        (VPF)   →  NI net signal      (gaze-holding setpoint)
    vermis_fastigial     (V/CFN) →  NI net signal      (tonic gaze offset)
    nodulus_uvula        (NU)    →  VS axis            (gravity-axis setpoint)

State:  none  (N_STATES = 0)
Input:  CerebellumInputs(x_vs_net, x_ni_net, x_vs_axis)
Output: CerebellumOut (12 fields — bias, pred_err, drive per region)

Standalone — not yet wired into brain_model.py / simulator.py.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


N_STATES = 0
N_INPUTS = 9     # x_vs_net(3) + x_ni_net(3) + x_vs_axis(3)
N_OUTPUTS = 36   # 4 regions × (bias(3) + pred_err(3) + drive(3))


# ─── Params ──────────────────────────────────────────────────────────────────

class CerebellumParams(NamedTuple):
    """Cerebellum parameters — flat NamedTuple, hand-set, no learning."""

    # Lesion knobs (1.0 = intact, 0.0 = fully lesioned)
    g_cereb: float = 1.0
    g_flocculus: float = 1.0
    g_paraflocculus: float = 1.0
    g_vermis_fastigial: float = 1.0
    g_nodulus_uvula: float = 1.0

    # Per-region prediction (constant 3-vector — yaw/pitch/roll or axis)
    theta_bias_fl:   tuple = (0.0, 0.0, 0.0)
    theta_bias_vpf:  tuple = (0.0, 0.0, 0.0)
    theta_bias_vcfn: tuple = (0.0, 0.0, 0.0)
    theta_bias_nu:   tuple = (0.0, 0.0, 0.0)

    # Per-region prediction-error gain (1/s)
    K_fl:   float = 0.5
    K_vpf:  float = 0.5
    K_vcfn: float = 0.5
    K_nu:   float = 0.5


def with_cerebellum(theta: CerebellumParams, **overrides) -> CerebellumParams:
    """Return a copy of `theta` with the named fields replaced.

    Mirrors `with_brain` / `with_sensory` in the rest of the codebase.
    """
    return theta._replace(**overrides)


# ─── Inputs / Outputs ────────────────────────────────────────────────────────

class CerebellumInputs(NamedTuple):
    """Brainstem signals the cerebellum reads (each (3,))."""
    x_vs_net:  jnp.ndarray   # VS net signal     (deg/s)
    x_ni_net:  jnp.ndarray   # NI net signal     (deg)
    x_vs_axis: jnp.ndarray   # VS axis signal    (axis units)


class _RegionOut(NamedTuple):
    """Internal: single-region readout (bias, pred_err, drive)."""
    bias: jnp.ndarray
    pred_err: jnp.ndarray
    drive: jnp.ndarray


class CerebellumOut(NamedTuple):
    """Per-region readouts and drives (each field shape (3,))."""
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


# ─── Region rule ─────────────────────────────────────────────────────────────

def _region(K_eff, theta_bias, x_target) -> _RegionOut:
    """Apply the cerebellar rule for one region.

        bias     = K_eff · θ_bias
        pred_err = x_target − θ_bias
        drive    = bias − K_eff · pred_err
    """
    theta_bias = jnp.asarray(theta_bias, dtype=jnp.float32)
    bias     = K_eff * theta_bias
    pred_err = x_target - theta_bias
    drive    = bias - K_eff * pred_err
    return _RegionOut(bias=bias, pred_err=pred_err, drive=drive)


# ─── Step ────────────────────────────────────────────────────────────────────

def step(x_cereb, inputs: CerebellumInputs, theta: CerebellumParams):
    """Single ODE step for the cerebellum.

    Args:
        x_cereb: (0,) zero-length state — cerebellum is stateless
        inputs:  CerebellumInputs
        theta:   CerebellumParams

    Returns:
        dx:  (0,) zero-length state derivative
        out: CerebellumOut
    """
    g = theta.g_cereb
    fl  = _region(g * theta.g_flocculus        * theta.K_fl,
                  theta.theta_bias_fl,   inputs.x_vs_net)
    vpf = _region(g * theta.g_paraflocculus    * theta.K_vpf,
                  theta.theta_bias_vpf,  inputs.x_ni_net)
    vc  = _region(g * theta.g_vermis_fastigial * theta.K_vcfn,
                  theta.theta_bias_vcfn, inputs.x_ni_net)
    nu  = _region(g * theta.g_nodulus_uvula    * theta.K_nu,
                  theta.theta_bias_nu,   inputs.x_vs_axis)

    out = CerebellumOut(
        bias_fl=fl.bias,     pred_err_fl=fl.pred_err,     drive_fl=fl.drive,
        bias_vpf=vpf.bias,   pred_err_vpf=vpf.pred_err,   drive_vpf=vpf.drive,
        bias_vcfn=vc.bias,   pred_err_vcfn=vc.pred_err,   drive_vcfn=vc.drive,
        bias_nu=nu.bias,     pred_err_nu=nu.pred_err,     drive_nu=nu.drive,
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
    inputs0 = CerebellumInputs(x_vs_net=zero3, x_ni_net=zero3, x_vs_axis=zero3)

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

    # Pure-bias case
    print("\n[3] Pure-bias case (FL)")
    theta = CerebellumParams(theta_bias_fl=(1.0, 0.0, 0.0), K_fl=1.0)
    _, out = step(x0, inputs0, theta)
    _check("bias_fl == (1, 0, 0)",      _close(out.bias_fl,     [1, 0, 0]))
    _check("pred_err_fl == (-1, 0, 0)", _close(out.pred_err_fl, [-1, 0, 0]))
    _check("drive_fl == (2, 0, 0)",     _close(out.drive_fl,    [2, 0, 0]))
    _check("other regions unchanged",   _close(out.bias_vpf, zero3) and _close(out.drive_nu, zero3))

    # Pure-residual case
    print("\n[4] Pure-residual case (FL)")
    theta = CerebellumParams(K_fl=1.0)
    inputs = CerebellumInputs(
        x_vs_net=jnp.array([1.0, 0.0, 0.0]), x_ni_net=zero3, x_vs_axis=zero3,
    )
    _, out = step(x0, inputs, theta)
    _check("bias_fl == 0",             _close(out.bias_fl,     zero3))
    _check("pred_err_fl == (1, 0, 0)", _close(out.pred_err_fl, [1, 0, 0]))
    _check("drive_fl == (-1, 0, 0)",   _close(out.drive_fl,    [-1, 0, 0]))

    # Algebraic identity drive == bias - K_eff · pred_err
    print("\n[5] Algebraic identity")
    theta = CerebellumParams(
        theta_bias_fl=(0.7, -0.3, 0.4),    theta_bias_vpf=(2.0, 1.0, -1.5),
        theta_bias_vcfn=(0.5, 0.5, 0.5),   theta_bias_nu=(-0.1, 0.2, -0.3),
        K_fl=0.4, K_vpf=0.6, K_vcfn=0.8, K_nu=1.2,
    )
    inputs = CerebellumInputs(
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
        bias, pe, drv = (getattr(out, f"{k}_{r}") for k in ("bias", "pred_err", "drive"))
        _check(f"drive_{r} == bias_{r} - K_eff · pred_err_{r}",
               _close(drv, bias - K_eff[r] * pe))

    # Global lesion: bias and drive zero, pred_err unchanged
    print("\n[6] Global lesion (g_cereb = 0)")
    theta_lesion = CerebellumParams(
        g_cereb=0.0,
        theta_bias_fl=(1.0, 1.0, 1.0), theta_bias_vpf=(2.0, 2.0, 2.0),
        K_fl=10.0, K_vpf=10.0, K_vcfn=10.0, K_nu=10.0,
    )
    inputs = CerebellumInputs(
        x_vs_net=jnp.array([3.0, 4.0, 5.0]),
        x_ni_net=jnp.array([6.0, 7.0, 8.0]),
        x_vs_axis=jnp.array([9.0, 10.0, 11.0]),
    )
    _, out = step(x0, inputs, theta_lesion)
    for r in ("fl", "vpf", "vcfn", "nu"):
        _check(f"bias_{r} == 0",   _close(getattr(out, f"bias_{r}"),  zero3))
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
        _check(f"bias_{suf} == 0  ({knob}=0)",  _close(getattr(out, f"bias_{suf}"),  zero3))
        _check(f"drive_{suf} == 0 ({knob}=0)", _close(getattr(out, f"drive_{suf}"), zero3))
        for other in ("fl", "vpf", "vcfn", "nu"):
            if other == suf:
                continue
            _check(f"bias_{other} != 0 ({knob}=0)",
                   not _close(getattr(out, f"bias_{other}"), zero3))

    # with_cerebellum
    print("\n[8] with_cerebellum helper")
    base = CerebellumParams()
    edited = with_cerebellum(base, K_fl=2.5, theta_bias_vpf=(0.1, 0.2, 0.3))
    _check("K_fl updated",            edited.K_fl == 2.5)
    _check("theta_bias_vpf updated",  edited.theta_bias_vpf == (0.1, 0.2, 0.3))
    _check("g_cereb unchanged",       edited.g_cereb == base.g_cereb)
    _check("theta_bias_fl unchanged", edited.theta_bias_fl == base.theta_bias_fl)

    # JIT
    print("\n[9] JIT compatibility")
    step_jit = jax.jit(step)
    _, out_jit = step_jit(x0, inputs0, CerebellumParams())
    _check("JIT compile + run", out_jit is not None)
    _check("JIT output shape",  out_jit.bias_fl.shape == (3,))

    print("\n" + "=" * 60)
    print("All smoke tests passed.")
