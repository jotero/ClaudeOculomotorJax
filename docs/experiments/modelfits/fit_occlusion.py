"""Fit a small set of vergence parameters to averaged occlusion-experiment data.

The averaged data has 6 traces (vergence + vergence velocity) per condition:

    conv_continuous, conv_flashing, conv_dark
    div_continuous,  div_flashing,  div_dark

These come from the bench `bench_experiments.py` (or experimental recordings with
the same paradigm — fixation at near + far targets, monocular occlusion with
continuous / flashing / dark conditions, averaged across L/R viewing eyes).

──── Fittable parameters (7) ────────────────────────────────────────────────────

    tonic_verg          — resting vergence baseline (deg)
    dark_tonic_verg     — resting vergence in dark (deg)
    tau_verg_tonic      — slow tonic adapter TC (s)
    AC_A                — AC/A ratio (pd / D)
    K_verg              — fast vergence integrator gain
    tau_verg            — fast vergence integrator TC (s)
    K_verg_tonic        — slow integrator coupling gain

Reparameterisation: tonic_verg + dark_tonic_verg are raw (signed); positive-only
quantities (TCs, gains, AC/A) use softplus to enforce positivity smoothly.

──── Loss ───────────────────────────────────────────────────────────────────────

    L = Σ_conditions (MSE_vergence + λ · MSE_vergence_velocity)

    λ = 0.1 by default — vergence units (deg) and vergence velocity (deg/s)
    differ by ~1/τ ratio so a small λ keeps both terms balanced.

──── Synthetic recovery test ────────────────────────────────────────────────────

    1. Pick a "ground truth" θ.
    2. Forward → synthetic data.
    3. Perturb θ by ~30 % and run optimisation.
    4. Check final params recover within ~5 % and loss drops 100×.

Use the synthetic test before fitting real data — it tells you whether the
paradigm is identifiable for the chosen subset and whether the optimiser
converges from your starting point.

──── Data file format (npz) ─────────────────────────────────────────────────────

    np.savez('averaged.npz',
        t = t_array,                                        # (T,) seconds
        conv_continuous_v = ..., conv_continuous_vv = ...,  # (T,) each
        conv_flashing_v   = ..., conv_flashing_vv   = ...,
        conv_dark_v       = ..., conv_dark_vv       = ...,
        div_continuous_v  = ..., div_continuous_vv  = ...,
        div_flashing_v    = ..., div_flashing_vv    = ...,
        div_dark_v        = ..., div_dark_vv        = ...,
    )

──── Usage ──────────────────────────────────────────────────────────────────────

    # Synthetic recovery test (no data needed):
    python -X utf8 experiments/modelfits/fit_occlusion.py --synthetic

    # Fit averaged experimental data:
    python -X utf8 experiments/modelfits/fit_occlusion.py --data path/to/averaged.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

# Make `oculomotor` importable when run from project root or any cwd.
_PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJ_ROOT / 'src'))

from oculomotor.sim.simulator import (         # noqa: E402
    PARAMS_DEFAULT, SimConfig, simulate, with_brain, with_sensory,
)
from oculomotor.sim.kinematics import build_target, TargetTrajectory   # noqa: E402

# TargetTrajectory is a plain dataclass; register as a pytree so jit can trace
# through it (its fields are JAX arrays).
jax.tree_util.register_pytree_node(
    TargetTrajectory,
    lambda tt: ((tt.t, tt.lin_pos, tt.lin_vel), None),
    lambda _, children: TargetTrajectory(*children),
)


# ── Configuration mirroring bench_experiments.py ───────────────────────────────

DT       = 0.001
T_END    = 12.0    # was 15 — fitting only needs the post-occlusion drift window
T_FIX    = 2.0
T_ON     = 0.080
T_PERIOD = 0.980
WARMUP_S = 8.0     # was 30 — short warmup OK because we re-fit tonic anyway;
                   # 8 s ≈ 1.6 × τ_verg_tonic gets vergence to within ~80 % of SS

# Two experimental configs — initial fixation distance + lens — held FIXED at fit time.
# Only the 7 dynamics parameters are fitted; the geometry of the experiment is given.
_CONV_DIST_M = 0.25
_CONV_LENS_D = -3.0
_DIV_DIST_M  = 10.0
_DIV_LENS_D  = 1.0

# Six conditions in the order they appear in the data:
#   (config_tag, cond_label, list-of-(occlusion-eye, weight) for averaging)
# For fitting we run a SINGLE occlusion eye per condition (not the L+R average).
# The model is L/R symmetric so a single run is statistically equivalent for
# vergence/version means; this halves runtime per gradient step.
_CONDITIONS = [
    ('conv', 'continuous', [('left', 1.0)]),
    ('conv', 'pulsed',     [('left', 1.0)]),
    ('conv', 'dark',       [('left', 1.0)]),
    ('div',  'continuous', [('left', 1.0)]),
    ('div',  'pulsed',     [('left', 1.0)]),
    ('div',  'dark',       [('left', 1.0)]),
]

LAMBDA_VEL = 0.1   # weight on vergence-velocity MSE

# Smoothing — applied to model predictions before computing loss, so the model
# is compared against the smoothed averaged traces on equal footing. Default
# is a 500 ms moving-average (boxcar) filter to match a typical experimental
# preprocessing pipeline. Set width to 0 to disable. Override at the CLI with
# --smooth-window-ms.
SMOOTH_WINDOW_S = 0.500

# Noiseless base (deterministic simulation for fitting)
_THETA_BASE = with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_pos=0.0, sigma_vel=0.0)
_THETA_BASE = with_brain(_THETA_BASE, sigma_acc=0.0)


# ── Stimulus arrays (numpy — recomputed once, deterministic per condition) ─────

def _make_flags(t_np: np.ndarray, cond: str, occ_eye: str):
    """Per-eye target-presence + strobe arrays — copy of bench_experiments logic."""
    rel_t = t_np - T_FIX
    on_off = np.where(rel_t < 0, 1.0, 0.0).astype(np.float32)
    if cond == 'continuous':
        viewing = np.ones_like(t_np, dtype=np.float32)
        no_strobe = np.zeros_like(viewing)
    elif cond == 'pulsed':
        phase = np.mod(np.maximum(rel_t, 0.0), T_PERIOD)
        viewing = np.where(rel_t < 0, 1.0, (phase < T_ON).astype(np.float32)).astype(np.float32)
        no_strobe = np.where(rel_t < 0, 0.0, 1.0).astype(np.float32)
    elif cond == 'dark':
        viewing = on_off
        no_strobe = np.zeros_like(viewing)
    else:
        raise ValueError(f'unknown cond {cond}')
    off = on_off
    if occ_eye == 'left':
        tL, tR = off, viewing
    else:
        tL, tR = viewing, off
    return tL, tR, no_strobe


def _build_inputs(t_np: np.ndarray, cond: str, occ_eye: str, dist_m: float, lens_d: float):
    """Build all (constant) input arrays for one column."""
    T = len(t_np)
    pt = np.tile(np.array([0.0, 0.0, dist_m]), (T, 1)).astype(np.float32)
    tL, tR, ts = _make_flags(t_np, cond, occ_eye)
    lens_arr = np.full((T,), lens_d, dtype=np.float32)
    return pt, tL, tR, ts, lens_arr


# ── Forward pass: params → 6 averaged (vergence, vergence_velocity) traces ─────

def _make_theta(p: dict, dist_m: float, lens_d: float, dark: bool):
    """Construct a Params NamedTuple with the fitted overrides applied.

    Positive-only params come in as raw floats and get softplus-projected here.
    """
    tonic = p['dark_tonic_verg'] if dark else p['tonic_verg']
    return with_brain(
        _THETA_BASE,
        tonic_verg     = tonic,
        tau_verg_tonic = jax.nn.softplus(p['raw_tau_verg_tonic']) + 1.0,   # ≥1 s
        AC_A           = jax.nn.softplus(p['raw_AC_A']),
        K_verg         = jax.nn.softplus(p['raw_K_verg']),
        tau_verg       = jax.nn.softplus(p['raw_tau_verg']) + 0.1,         # ≥0.1 s
        K_verg_tonic   = jax.nn.softplus(p['raw_K_verg_tonic']),
    )


def _simulate_column(theta, t_jnp, target, tL_jnp, tR_jnp, ts_jnp, lens_jnp):
    """Run one occlusion column and return per-eye eye-yaw time series.

    `target` is a pre-built TargetMotion (built outside jit because build_target
    uses numpy operations on t).
    """
    states = simulate(
        theta, t_jnp,
        target                 = target,
        scene_present_array    = jnp.zeros(t_jnp.shape[0]),
        target_present_L_array = tL_jnp,
        target_present_R_array = tR_jnp,
        target_strobed_array   = ts_jnp,
        lens_L_array           = lens_jnp,
        lens_R_array           = lens_jnp,
        return_states          = True,
        sim_config             = SimConfig(warmup_s=WARMUP_S),
    )
    return states.plant.left[:, 0], states.plant.right[:, 0]   # eye_L_yaw, eye_R_yaw


def forward(p: dict, stim_arrays: dict):
    """Compute the 6 averaged (vergence_velocity, version_velocity) traces.

    Args:
        p:           fitted-parameter dict (values are jax-friendly scalars)
        stim_arrays: precomputed (t_jnp, condition→inputs) bundle from `prepare_stim`
    Returns:
        (verg_vel, vers_vel): each (6, T) — rows match _CONDITIONS order.
            verg_vel = d/dt (eye_L - eye_R)             (vergence velocity, deg/s)
            vers_vel = d/dt (eye_L + eye_R) / 2          (version velocity,  deg/s)
    """
    t_jnp = stim_arrays['t']
    rows_verg_vel, rows_vers_vel = [], []
    for tag, cond, weights in _CONDITIONS:
        dist_m = _CONV_DIST_M if tag == 'conv' else _DIV_DIST_M
        lens_d = _CONV_LENS_D if tag == 'conv' else _DIV_LENS_D
        is_dark = (cond == 'dark')
        theta = _make_theta(p, dist_m, lens_d, dark=is_dark)
        eye_L_avg = jnp.zeros_like(t_jnp)
        eye_R_avg = jnp.zeros_like(t_jnp)
        for occ_eye, w in weights:
            key = (tag, cond, occ_eye)
            inputs = stim_arrays['col'][key]
            eye_L, eye_R = _simulate_column(theta, t_jnp, *inputs)
            eye_L_avg = eye_L_avg + w * eye_L
            eye_R_avg = eye_R_avg + w * eye_R
        d_L = jnp.gradient(eye_L_avg, DT)
        d_R = jnp.gradient(eye_R_avg, DT)
        rows_verg_vel.append(d_L - d_R)
        rows_vers_vel.append(0.5 * (d_L + d_R))
    return jnp.stack(rows_verg_vel), jnp.stack(rows_vers_vel)


def prepare_stim(t_np: np.ndarray) -> dict:
    """Pre-build the per-column input arrays once (they don't depend on θ).

    Includes the TargetMotion struct (built via build_target outside jit, since
    build_target uses numpy ops that can't be traced).
    """
    t_jnp = jnp.array(t_np)
    cols = {}
    for tag, cond, weights in _CONDITIONS:
        dist_m = _CONV_DIST_M if tag == 'conv' else _DIV_DIST_M
        lens_d = _CONV_LENS_D if tag == 'conv' else _DIV_LENS_D
        for occ_eye, _ in weights:
            pt, tL, tR, ts, lens = _build_inputs(t_np, cond, occ_eye, dist_m, lens_d)
            target = build_target(t_jnp, lin_pos=jnp.array(pt))   # outside jit
            cols[(tag, cond, occ_eye)] = (
                target, jnp.array(tL), jnp.array(tR), jnp.array(ts), jnp.array(lens)
            )
    return {'t': t_jnp, 'col': cols}


# ── Loss ───────────────────────────────────────────────────────────────────────

def _boxcar_kernel(window_s: float) -> jnp.ndarray:
    """Uniform 1-D moving-average kernel sampled at the simulation rate (DT)."""
    if window_s <= 0:
        return jnp.array([1.0])
    n = int(round(window_s / DT))
    n = max(n, 1)
    return jnp.ones(n) / n


_SMOOTH_KERNEL = None  # lazily built so the CLI can override SMOOTH_WINDOW_S first


def _smooth(traces: jnp.ndarray) -> jnp.ndarray:
    """Apply a moving-average filter along axis -1 (time) of (N, T) traces."""
    global _SMOOTH_KERNEL
    if _SMOOTH_KERNEL is None:
        _SMOOTH_KERNEL = _boxcar_kernel(SMOOTH_WINDOW_S)
    if _SMOOTH_KERNEL.shape[0] == 1:
        return traces
    # Centred 'same'-style convolution with edge-replication padding so the
    # output length matches the input.
    n = _SMOOTH_KERNEL.shape[0]
    pad_l = (n - 1) // 2
    pad_r = n - 1 - pad_l
    padded = jnp.pad(traces, ((0, 0), (pad_l, pad_r)), mode='edge')
    return jax.vmap(lambda x: jnp.convolve(x, _SMOOTH_KERNEL, mode='valid'))(padded)


def _resample_to_data_grid(pred_traces: jnp.ndarray, t_data: jnp.ndarray) -> jnp.ndarray:
    """Linearly resample (6, T_model) traces onto the data time grid.

    The data time origin (t_data = 0) is defined as the moment of occlusion
    onset, which corresponds to model time T_FIX. So model_t = t_data + T_FIX.
    """
    t_model = jnp.arange(pred_traces.shape[1]) * DT       # model time grid (s)
    target_t = t_data + T_FIX                              # data time → model time
    return jax.vmap(lambda x: jnp.interp(target_t, t_model, x))(pred_traces)


def loss_fn(p: dict, stim_arrays: dict, data_verg: jnp.ndarray,
            data_vers: jnp.ndarray, mask: jnp.ndarray, t_data: jnp.ndarray):
    pred_verg, pred_vers = forward(p, stim_arrays)
    # Match the smoothing applied to the experimental averages so the model
    # isn't penalised for high-frequency content the data has already lost.
    pred_verg = _smooth(pred_verg)
    pred_vers = _smooth(pred_vers)
    # Resample model output onto the data time grid (data t = 0 ≡ model T_FIX).
    pred_verg = _resample_to_data_grid(pred_verg, t_data)
    pred_vers = _resample_to_data_grid(pred_vers, t_data)
    # Masked MSE — NaNs in data become False in mask.
    n_eff = jnp.maximum(mask.sum(), 1).astype(jnp.float32)
    err_verg = jnp.where(mask, (pred_verg - data_verg) ** 2, 0.0).sum() / n_eff
    err_vers = jnp.where(mask, (pred_vers - data_vers) ** 2, 0.0).sum() / n_eff
    return err_verg + err_vers


# ── Initial parameter dict ─────────────────────────────────────────────────────

def _inv_softplus(y: float) -> float:
    """Inverse of softplus(x) = log(1+e^x). Used to set raw values to known TCs."""
    return float(np.log(np.expm1(y)))   # log(e^y − 1)


def initial_params() -> dict:
    """Initial fitted-parameter dict — defaults from the model's PARAMS_DEFAULT.

    Positive params are stored as raw values; softplus is applied at use time.
    """
    bp = PARAMS_DEFAULT.brain
    return {
        'tonic_verg':         jnp.float32(bp.tonic_verg),
        'dark_tonic_verg':    jnp.float32(bp.tonic_verg),                # init same as light
        'raw_tau_verg_tonic': jnp.float32(_inv_softplus(bp.tau_verg_tonic - 1.0)),
        'raw_AC_A':           jnp.float32(_inv_softplus(bp.AC_A)),
        'raw_K_verg':         jnp.float32(_inv_softplus(bp.K_verg)),
        'raw_tau_verg':       jnp.float32(_inv_softplus(bp.tau_verg - 0.1)),
        'raw_K_verg_tonic':   jnp.float32(_inv_softplus(bp.K_verg_tonic)),
    }


def readable_params(p: dict) -> dict:
    """Project raw → human-readable values."""
    return {
        'tonic_verg':     float(p['tonic_verg']),
        'dark_tonic_verg':float(p['dark_tonic_verg']),
        'tau_verg_tonic': float(jax.nn.softplus(p['raw_tau_verg_tonic']) + 1.0),
        'AC_A':           float(jax.nn.softplus(p['raw_AC_A'])),
        'K_verg':         float(jax.nn.softplus(p['raw_K_verg'])),
        'tau_verg':       float(jax.nn.softplus(p['raw_tau_verg']) + 0.1),
        'K_verg_tonic':   float(jax.nn.softplus(p['raw_K_verg_tonic'])),
    }


# ── Training loop ──────────────────────────────────────────────────────────────

def fit(p_init: dict, stim_arrays: dict, data_verg: jnp.ndarray,
        data_vers: jnp.ndarray, mask: jnp.ndarray, t_data: jnp.ndarray,
        steps: int = 300, lr: float = 3e-3, verbose: bool = True,
        grad_clip: float = 1.0) -> dict:
    """Run optax.adam for `steps` iterations and return the fitted params.

    Gradients flow through `simulate()` via diffrax (jit'd internally). The
    numpy preprocessing inside `simulate` is not traced — only the params
    (theta) are differentiable from JAX's perspective, which is exactly what
    we want.

    Gradients can be large near the closed-loop edge of stability so we
    chain a global-norm clip before Adam. NaN gradients (when the simulator
    transiently goes singular for a candidate θ) are zeroed for that step,
    avoiding a one-shot poisoning of the parameter state.
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(lr),
    )
    opt_state = optimizer.init(p_init)
    grad_fn = jax.value_and_grad(loss_fn)

    def _replace_nan(g):
        return jax.tree_util.tree_map(lambda x: jnp.where(jnp.isfinite(x), x, 0.0), g)

    import time as _time
    p = p_init
    if verbose:
        print('  step    0  starting (first call traces+compiles the simulator — '
              'this takes 1–3 min on CPU; subsequent steps are much faster) ...',
              flush=True)
    t0 = _time.time()
    for step in range(steps):
        loss_val, grads = grad_fn(p, stim_arrays, data_verg, data_vers, mask, t_data)
        grads = _replace_nan(grads)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        if verbose and (step % 5 == 0 or step == steps - 1):
            elapsed = _time.time() - t0
            print(f'  step {step:4d}  loss = {float(loss_val):.4e}  '
                  f'(elapsed {elapsed:6.1f} s)', flush=True)
    return p


# ── Synthetic recovery test ────────────────────────────────────────────────────

def _perturb_params(p: dict, frac: float, key) -> dict:
    """Multiplicatively perturb each scalar by ±frac (uniform). Tonic gets ±frac·5° additive."""
    out = {}
    for i, k in enumerate(p):
        sub = jax.random.fold_in(key, i)
        if k in ('tonic_verg', 'dark_tonic_verg'):
            out[k] = p[k] + jax.random.uniform(sub, (), minval=-5.0 * frac, maxval=5.0 * frac)
        else:
            out[k] = p[k] + jax.random.uniform(sub, (), minval=-frac, maxval=frac)
    return out


def synthetic_test(steps: int = 200):
    print('── Synthetic recovery test ──────────────────────────────────────')
    t_np = np.arange(0.0, T_END, DT, dtype=np.float32)
    stim = prepare_stim(t_np)

    # Ground truth: PARAMS_DEFAULT with mild overrides so it's not the trivial init.
    p_true = initial_params()
    p_true = {**p_true,
              'tonic_verg':      jnp.float32(8.0),
              'dark_tonic_verg': jnp.float32(20.0)}
    print('  Ground truth:', readable_params(p_true))

    print('  Generating synthetic data (with the same smoothing the model will see) ...')
    # Build a fake data grid covering [-T_FIX, +T_END - T_FIX] @ 5 ms
    t_data = jnp.arange(-T_FIX, T_END - T_FIX, 0.005, dtype=jnp.float32)
    pred_verg, pred_vers = forward(p_true, stim)
    pred_verg = _resample_to_data_grid(_smooth(pred_verg), t_data)
    pred_vers = _resample_to_data_grid(_smooth(pred_vers), t_data)
    mask = jnp.ones_like(pred_verg, dtype=bool)

    p_init = _perturb_params(p_true, frac=0.6, key=jax.random.PRNGKey(0))
    print('  Perturbed init:', readable_params(p_init))

    print('  Fitting ...')
    p_fit = fit(p_init, stim, pred_verg, pred_vers, mask, t_data, steps=steps)
    print('  Recovered:    ', readable_params(p_fit))

    truth = readable_params(p_true)
    fit_v = readable_params(p_fit)
    print('\n  Recovery summary:')
    for k in truth:
        rel = (fit_v[k] - truth[k]) / (abs(truth[k]) + 1e-9)
        print(f'    {k:<18s}  truth={truth[k]:+8.4f}  fit={fit_v[k]:+8.4f}  rel-err={rel:+.2%}')


# ── Real data fit ──────────────────────────────────────────────────────────────

# Map data column name (MATLAB-friendly) → (config_tag, condition_label)
_COLUMN_MAP = {
    'FromConvergenceNoFlashing': ('conv', 'continuous'),
    'FromConvergenceFlashing':   ('conv', 'pulsed'),
    'FromConvergenceNoTarget':   ('conv', 'dark'),
    'FromDivergenceNoFlashing':  ('div',  'continuous'),
    'FromDivergenceFlashing':    ('div',  'pulsed'),
    'FromDivergenceNoTarget':    ('div',  'dark'),
}


def _read_csv(path: str) -> tuple[list[str], np.ndarray]:
    """Read a CSV with header and float entries (NaN-aware)."""
    import csv
    with open(path, 'r') as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = []
        for r in rdr:
            row = [float(x) if x.strip() and x.strip().lower() != 'nan' else np.nan
                   for x in r]
            rows.append(row)
    return header, np.array(rows, dtype=np.float64)


def load_csv_pair(verg_csv: str, vers_csv: str, sign_flip: bool = True):
    """Load matched vergence + version CSVs and align with model time.

    Returns:
        (data_verg, data_vers, mask, t_data) — first three are (6, N_data) JAX
        arrays; t_data is (N_data,) numpy. Rows are in _CONDITIONS order.
        Data is cropped to t_data ∈ [−T_FIX, T_END − T_FIX] (the simulation
        window after the warmup) and sign-flipped if requested.
    """
    h_v,  a_v  = _read_csv(verg_csv)
    h_vs, a_vs = _read_csv(vers_csv)

    if 'Time' not in h_v or 'Time' not in h_vs:
        raise ValueError('Both CSVs must have a "Time" column.')
    t_v  = a_v[:,  h_v.index('Time')]
    t_vs = a_vs[:, h_vs.index('Time')]
    if not np.allclose(t_v, t_vs, atol=1e-6):
        raise ValueError('Time columns differ between vergence.csv and version.csv.')

    keep = (t_v >= -T_FIX) & (t_v < T_END - T_FIX)
    t_kept = t_v[keep].astype(np.float32)

    rows_verg, rows_vers = [], []
    for tag, cond, _ in _CONDITIONS:
        col_name = next(n for n, (t_, c_) in _COLUMN_MAP.items()
                        if t_ == tag and c_ == cond)
        if col_name not in h_v or col_name not in h_vs:
            raise ValueError(f'Column {col_name!r} missing from one of the CSVs.')
        rows_verg.append(a_v [keep, h_v .index(col_name)])
        rows_vers.append(a_vs[keep, h_vs.index(col_name)])

    verg = np.stack(rows_verg).astype(np.float32)
    vers = np.stack(rows_vers).astype(np.float32)
    if sign_flip:
        verg, vers = -verg, -vers

    mask = np.isfinite(verg) & np.isfinite(vers)
    verg = np.where(mask, verg, 0.0)
    vers = np.where(mask, vers, 0.0)

    return jnp.array(verg), jnp.array(vers), jnp.array(mask), jnp.array(t_kept)


def fit_csv(verg_csv: str, vers_csv: str, steps: int = 500, sign_flip: bool = True):
    print(f'── Fitting from {verg_csv} + {vers_csv} ─────────────────────')
    data_verg, data_vers, mask, t_data = load_csv_pair(verg_csv, vers_csv, sign_flip=sign_flip)
    print(f'  data shape: {data_verg.shape}, valid samples: {int(mask.sum())} / {mask.size}')
    print(f'  data t range: [{float(t_data.min()):+.2f}, {float(t_data.max()):+.2f}] s')

    t_np = np.arange(0.0, T_END, DT, dtype=np.float32)
    stim = prepare_stim(t_np)

    p_init = initial_params()
    print('  Initial:', readable_params(p_init))
    p_fit = fit(p_init, stim, data_verg, data_vers, mask, t_data, steps=steps)
    print('  Final:  ', readable_params(p_fit))
    return p_fit


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    global SMOOTH_WINDOW_S, _SMOOTH_KERNEL
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--synthetic', action='store_true',
                    help='Run a synthetic-recovery test (no data file needed).')
    ap.add_argument('--vergence-csv', type=str, default=None,
                    help='Path to vergence-velocity CSV (rows = time samples, '
                         'columns = condition names + Time).')
    ap.add_argument('--version-csv', type=str, default=None,
                    help='Path to version-velocity CSV (same structure / time as the vergence CSV).')
    ap.add_argument('--no-sign-flip', action='store_true',
                    help='By default the loader multiplies data by -1 to match the '
                         'model sign convention. Pass this flag to disable.')
    ap.add_argument('--steps', type=int, default=300,
                    help='Number of Adam iterations (default 300).')
    ap.add_argument('--smooth-window-ms', type=float, default=SMOOTH_WINDOW_S * 1000.0,
                    help='Moving-average (boxcar) window (ms) applied to model '
                         'predictions to match the smoothing of the experimental '
                         'averages. 0 disables. Default 500 ms.')
    args = ap.parse_args()

    SMOOTH_WINDOW_S = args.smooth_window_ms / 1000.0
    _SMOOTH_KERNEL = None   # rebuild on next use

    if args.synthetic:
        synthetic_test(steps=args.steps)
    elif args.vergence_csv and args.version_csv:
        fit_csv(args.vergence_csv, args.version_csv,
                steps=args.steps, sign_flip=not args.no_sign_flip)
    else:
        ap.error('Pass --synthetic, or both --vergence-csv and --version-csv.')


if __name__ == '__main__':
    main()
