"""Cerebellum — forward-model prediction-error correction + EC delay cascades.

Anatomical split (Leigh & Zee; Cannon & Robinson 1985; Lisberger):

  Flocculus (FL)            → NI integrator extension (gaze-holding).
                              Reads NI net + null state, outputs positive
                              feedback that cancels the brainstem NI leak,
                              extending effective TC from intrinsic ~1–3 s
                              to ~25 s (or ∞ for K_cereb_fl = 1).
                              Lesion → gaze-evoked nystagmus.

  Ventral paraflocculus     → smooth-pursuit forward model.  Reads
  (VPF / vermis VI–VII)       cyclopean target_vel + delayed EC; outputs
                              gated prediction drive to the pursuit
                              integrator.
                              Lesion → reduced pursuit gain.

Both regions share the EC delay cascades (scene + target), which are
maintained here as part of cerebellum state because they ARE the cerebellum's
internal forward-model output (predicted retinal self-motion contribution at
the same delay as the actual retinal signal).

Architectural rationale (cerebellum.md): the cerebellum maintains an
internal forward model of how the eye responds to its own motor commands.
The EC delay cascades ARE that forward model's output (predicted retinal
self-motion contribution at the same delay as the actual retinal signal),
so they belong here rather than in a separate efference_copy module.

Pursuit region (cerebellum.md §4.3)
───────────────────────────────────
    s_pred   = − target_visible · ec_d_target_no_torsion       (stationary-
                                                                target prior)
    pred_err = target_slip − s_pred                            (diagnostic)
    drive    = gate · K_cereb_pu · (−s_pred)
             = gate · K_cereb_pu · target_visible · ec_d_target_no_torsion

    gate     = K_mag(|ec_d_target|) · K_dir(slip · ec_d alignment)

The brainstem direct path k·target_slip is added downstream in
brain_model.step.  The combined pursuit input becomes:

    pursuit_in = K_pursuit_direct · target_slip + cb drive

Implicit cancellation: target_slip ≈ −eye_velocity (delayed) and
ec_d_target ≈ +eye_velocity (delayed by the same cascade), so the two terms
naturally cancel during fast eye movements without an engineered cancel
term.

Lesion (K_cereb_pu = 0): drive vanishes.  Pursuit falls back to the
brainstem direct path on raw self-motion-contaminated slip → reduced
slow-phase gain (the classic flocculus-lesion phenotype).

EC cascades
───────────
    Scene path:   tau_vis_smooth_motion        (~20 ms LP)   — matches
                  perception_cyclopean's scene_angular_vel cascade shape
    Target path:  tau_vis_smooth_target_vel    (~150 ms LP)  — matches
                  perception_cyclopean's target_vel cascade shape

Both share the same `tau_vis_sharp` 6-stage gamma cascade (matches the
per-eye retina sharp cascade).  The motor command (ec_vel, head frame) is
rotated into eye frame using ec_pos, then saturated by v_max_okr (scene)
and v_max_pursuit (target) to mirror the retinal saturation.

State
─────
    scene  — (_N_PER_PATH,)  scene-path EC cascade buffer (21 states)
    target — (_N_PER_PATH,)  target-path EC cascade buffer (21 states)

Activations (read at top of brain_model.step from BrainState)
────────────────────────────────────────────────────────────
    ec_scene   — (3,) delayed scene EC (= state.scene[-3:])
    ec_target  — (3,) delayed target EC (= state.target[-3:])
    pred_err   — (3,) target_slip − s_pred  (diagnostic)
    drive      — (3,) gated cerebellar pursuit drive

Params: read flat from BrainParams — K_cereb_pu (0 = lesion), the shared
trust-gate params (v_crit_ec_gate, n_ec_gate, alpha_ec_dir, bias_ec_dir),
and the cascade params (tau_vis_sharp, tau_vis_smooth_motion,
tau_vis_smooth_target_vel, v_max_okr, v_max_pursuit).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oculomotor.models.plant_models.readout import rotation_matrix
from oculomotor.models.sensory_models.retina import (
    cascade_lp_step, ypr_to_xyz, xyz_to_ypr, velocity_saturation,
)


# ── Cascade geometry ──────────────────────────────────────────────────────────
# 6-stage sharp gamma + 1-stage smoothing LP, on 3 axes per cascade.

_N_SHARP   = 6
_N_LP      = 1
_N_AXES    = 3
N_PER_PATH = (_N_SHARP + _N_LP) * _N_AXES   # 21 states per cascade

N_STATES   = 2 * N_PER_PATH + 3             # scene + target + MN forward-model LP


# ── State + Activations ───────────────────────────────────────────────────────

class State(NamedTuple):
    """Cerebellum state — EC cascade buffers + MN forward-model LP.

    `mn_lp` is the forward model of the FCP motor-neuron membrane LP
    (tau_mn ≈ 5 ms).  The actual eye velocity that the retina sees has
    already been smoothed by MN dynamics (motor command → MN → nerves →
    plant), so the EC cascade input — which is meant to mirror the slip
    cascade input — must also be MN-smoothed.  The LP sits BEFORE the
    saturation so that ec_vel_eye undergoes the same MN-then-saturate
    pipeline that the slip cascade implicitly applies (eye_vel_actual is
    already MN-LP'd, then sees retinal saturation).
    """
    scene:  jnp.ndarray   # (21,) cascade buffer matching scene_angular_vel
    target: jnp.ndarray   # (21,) cascade buffer matching target_vel
    mn_lp:  jnp.ndarray   # (3,)  forward-model MN-membrane LP (tau_mn)


class Activations(NamedTuple):
    """Cerebellar readouts."""
    # EC delay cascade outputs (delayed self-motion predictions)
    ec_scene:    jnp.ndarray   # (3,) delayed EC for scene path
    ec_target:   jnp.ndarray   # (3,) delayed EC for target path
    # Ventral paraflocculus (VPF) — smooth pursuit forward model
    pred_err:    jnp.ndarray   # (3,) target_slip − s_pred  (diagnostic)
    vpf_drive:   jnp.ndarray   # (3,) gated cerebellar pursuit drive → pursuit_in
    # Flocculus (FL) — NI integrator extension (Cannon & Robinson 1985)
    fl_drive:    jnp.ndarray   # (3,) leak-cancellation feedback → u_ni_in


def rest_state():
    """Zero state — both EC cascades + MN LP empty."""
    return State(scene=jnp.zeros(N_PER_PATH),
                 target=jnp.zeros(N_PER_PATH),
                 mn_lp=jnp.zeros(3))


# ── Activation read (thin: just state tail reads) ─────────────────────────────

def read_activations(brain_state, brain_params):
    """Read cerebellar activations from BrainState.

    Trivial wrapper that re-runs `step` against the current state with
    `ec_vel = ec_pos = 0` to extract activations only — the cascade
    derivative is discarded.  All forward-model and trust-gate computation
    lives in `step` (where the cerebellum's per-step processing belongs);
    this function just exists so brain_model.read_activations can populate
    `acts.cb` from state at the top of brain_model.step.
    """
    ni_net  = brain_state.ni.L - brain_state.ni.R
    ni_null = brain_state.ni.null
    _, acts = step(brain_state.cb,
                   ec_vel=jnp.zeros(3), ec_pos=jnp.zeros(3),
                   ni_net=ni_net, ni_null=ni_null,
                   target_slip=brain_state.pc.target_vel,
                   target_visible=brain_state.pc.target_visible[-1],
                   brain_params=brain_params)
    return acts


# ── State step (advances the EC cascades) ─────────────────────────────────────

def step(state, ec_vel, ec_pos, ni_net, ni_null,
         target_slip, target_visible, brain_params):
    """Cerebellum per-step processing.

    Three jobs:
      1. **Flocculus (FL) — NI integrator extension** (Cannon & Robinson
         1985).  Reads NI net + null state; outputs leak-cancellation
         feedback added back to NI input downstream:
             fl_drive = K_cereb_fl · (ni_net − ni_null) / tau_i_per_axis

      2. **VPF — pursuit forward model** (cerebellum.md §4.3):
             s_pred    = − target_visible · ec_d_target_no_torsion
             pred_err  = target_slip − s_pred                  (diagnostic)
             gate      = K_mag(|ec_d_target|) · K_dir(slip · ec_d alignment)
             vpf_drive = gate · K_cereb_pu · (−s_pred)

      3. **EC cascade advance** — scene + target cascades match the
         perception_cyclopean slip cascades (gamma/LP TCs + retinal
         velocity-saturation ceiling).  MN forward-model LP smooths the
         ec_vel before saturation so the EC mirrors the actual eye velocity
         the retina sees through the MN pathway.

    Args:
        state:           cerebellum.State
        ec_vel:          (3,)    version velocity efference (head frame, deg/s)
        ec_pos:          (3,)    eye position (head frame, deg)
        ni_net:          (3,)    NI net signal (decoded.ni.net) — for FL
        ni_null:         (3,)    NI null adaptation state — for FL
        target_slip:     (3,)    delayed cyclopean retinal target velocity
        target_visible:  scalar  delayed cyclopean target visibility ∈ [0,1]
        brain_params:    BrainParams

    Returns:
        dstate: cerebellum.State  state derivative
        acts:   cerebellum.Activations  EC tails + VPF drive + FL drive
    """
    bp = brain_params

    # ── Frame transform: ec_vel head → eye frame ─────────────────────────
    # All cerebellum computation downstream lives in eye frame (matching
    # the slip cascade and cyclopean signals).
    R_eye      = rotation_matrix(ypr_to_xyz(ec_pos))
    ec_vel_eye = xyz_to_ypr(R_eye.T @ ypr_to_xyz(ec_vel))

    # ── Flocculus (FL): NI leak cancellation ──────────────────────────────
    # Cannon & Robinson 1985: floccular feedback to NPH/MVN turns the leaky
    # brainstem NI into a near-perfect integrator.  Output is added to the
    # NI input downstream (in brain_model.step):
    #     u_ni_in_total = u_ni_in + fl_drive
    #     dx_net/dt     = −(x_net − x_null)/tau_i + u_ni_in_total
    #                   = u_ni_in   when fl_drive = leak amount, K_cereb_fl=1
    # K_cereb_fl = 0  →  lesion: NI reverts to intrinsic leak (gaze-evoked
    # nystagmus); K_cereb_fl = 1  →  perfect leak cancellation.
    tau_i_axes = jnp.array([bp.tau_i,
                             bp.tau_i * bp.tau_i_pitch_frac,
                             bp.tau_i * bp.tau_i_roll_frac])
    fl_drive = bp.K_cereb_fl * (ni_net - ni_null) / tau_i_axes

    # ── VPF: pursuit forward model (uses cerebellum's own state for ec_target) ──
    ec_scene  = state.scene[-3:]
    ec_target = state.target[-3:]
    ec_no_torsion = ec_target.at[2].set(0.0)
    s_pred        = -target_visible * ec_no_torsion
    pred_err      = target_slip - s_pred                   # diagnostic only

    # Trust gate: closes during fast self-motion (Hill on |ec_target|) or
    # when residual is anti-parallel to ec_d (sigmoid on slip · ec_d alignment).
    K_mag = 1.0 / (1.0 + (jnp.linalg.norm(ec_target) / bp.v_crit_ec_gate)
                   ** bp.n_ec_gate)
    ec_norm  = jnp.linalg.norm(ec_target) + 1e-9
    ec_hat   = ec_target / ec_norm
    slip_dot = jnp.dot(target_slip, ec_hat)
    K_dir    = jax.nn.sigmoid((slip_dot + bp.bias_ec_dir) * bp.alpha_ec_dir)
    gate     = K_mag * K_dir

    # VPF's pursuit drive = gated forward-model prediction.
    vpf_drive = gate * bp.K_cereb_pu * (-s_pred)

    # ── EC cascade advance ────────────────────────────────────────────────
    # Forward-model MN LP applied BEFORE saturation.  The actual eye velocity
    # that the retina sees has already been smoothed by MN dynamics (motor
    # command → MN → nerves → plant → eye), so the slip cascade input is
    # implicitly MN-LP'd.  For the EC cascade input to mirror it, we apply
    # the same MN LP here before the retinal velocity saturation.
    dmn_lp   = (ec_vel_eye - state.mn_lp) / bp.tau_mn
    mn_lp_in = state.mn_lp                                # state-based input

    ec_vel_scene_in  = velocity_saturation(mn_lp_in, bp.v_max_okr)      # NOT/AOS
    ec_vel_target_in = velocity_saturation(mn_lp_in, bp.v_max_pursuit)  # MT/MST

    dstate = State(
        scene  = cascade_lp_step(state.scene,  ec_vel_scene_in,
                                  bp.tau_vis_sharp,
                                  bp.tau_vis_smooth_motion,
                                  _N_SHARP, _N_AXES, _N_LP),
        target = cascade_lp_step(state.target, ec_vel_target_in,
                                  bp.tau_vis_sharp,
                                  bp.tau_vis_smooth_target_vel,
                                  _N_SHARP, _N_AXES, _N_LP),
        mn_lp  = dmn_lp,
    )
    acts = Activations(ec_scene=ec_scene, ec_target=ec_target,
                       pred_err=pred_err, vpf_drive=vpf_drive,
                       fl_drive=fl_drive)
    return dstate, acts


# ── Smoke tests ───────────────────────────────────────────────────────────────
# Run with:
#   python -X utf8 -m oculomotor.models.brain_models.cerebellum

if __name__ == "__main__":
    import numpy as np

    from oculomotor.models.brain_models.brain_model import BrainParams, rest_brain_state

    def _close(a, b, tol=1e-6):
        return np.allclose(np.asarray(a), np.asarray(b), atol=tol)

    def _check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            raise AssertionError(name)

    print("Cerebellum smoke tests")
    print("=" * 60)

    bp     = BrainParams()
    state0 = rest_brain_state()
    zero3  = jnp.zeros(3, dtype=jnp.float32)

    # Shape contract
    print("\n[1] Shape contract")
    acts = read_activations(state0, bp)
    _check("ec_scene shape",  acts.ec_scene.shape  == (3,))
    _check("ec_target shape", acts.ec_target.shape == (3,))
    _check("pred_err shape",  acts.pred_err.shape  == (3,))
    _check("vpf_drive shape", acts.vpf_drive.shape == (3,))
    _check("fl_drive shape",  acts.fl_drive.shape  == (3,))

    # Zero-state identity
    print("\n[2] Zero-state identity")
    _check("ec_scene == 0",  _close(acts.ec_scene, zero3))
    _check("ec_target == 0", _close(acts.ec_target, zero3))
    _check("pred_err == 0",  _close(acts.pred_err, zero3))
    _check("drive == 0",     _close(acts.vpf_drive, zero3))

    # State injection: target_slip drives pred_err
    print("\n[3] Pursuit forward model: pred_err uses cyclopean target_vel + EC")
    state_t = state0._replace(pc=state0.pc._replace(
        target_vel=jnp.array([10.0, 0.0, 0.0]),
        target_visible=state0.pc.target_visible.at[-1].set(1.0),
    ))
    acts = read_activations(state_t, bp)
    _check("pred_err == target_slip when ec=0", _close(acts.pred_err, [10.0, 0.0, 0.0]))
    _check("drive == 0 when ec=0",               _close(acts.vpf_drive, zero3))

    # EC cascade injection
    print("\n[4] EC cascade tail drives pursuit drive (torsion zeroed)")
    cb_state = state0.cb._replace(target=state0.cb.target.at[-3].set(2.0).at[-1].set(5.0))
    state_e  = state_t._replace(cb=cb_state)
    acts = read_activations(state_e, bp)
    _check("ec_target reads cascade tail", _close(acts.ec_target, [2.0, 0.0, 5.0]))
    _check("drive yaw ≈ 2",                abs(float(acts.vpf_drive[0]) - 2.0) < 0.1)
    _check("drive torsion zeroed",         abs(float(acts.vpf_drive[2])) < 1e-5)
    _check("pred_err yaw = ts + visible · ec_no_torsion = 12",
           _close(acts.pred_err, [12.0, 0.0, 0.0]))

    # Lesion
    print("\n[5] Cerebellar pursuit lesion (K_cereb_pu = 0)")
    bp_les = BrainParams(K_cereb_pu=0.0)
    acts_les = read_activations(state_e, bp_les)
    _check("vpf_drive == 0 under lesion", _close(acts_les.vpf_drive, zero3))
    _check("pred_err unchanged",        _close(acts_les.pred_err, [12.0, 0.0, 0.0]))
    _check("ec_target unchanged",       _close(acts_les.ec_target, [2.0, 0.0, 5.0]))

    # FL — leak-cancellation drive scales with (ni_net − ni_null)/tau_i
    print("\n[5b] Flocculus NI leak cancellation (K_cereb_fl)")
    state_ni = state0._replace(ni=state0.ni._replace(
        L=state0.ni.L.at[0].set(10.0),    # ni_net = +10 deg yaw
        R=state0.ni.R.at[0].set(0.0),
        null=state0.ni.null.at[0].set(2.0),  # x_null = +2 deg yaw
    ))
    acts_fl = read_activations(state_ni, BrainParams(K_cereb_fl=1.0))
    expected_fl_yaw = 1.0 * (10.0 - 2.0) / 25.0      # = 0.32
    _check("fl_drive yaw = K · (ni_net − ni_null)/tau_i",
           _close(acts_fl.fl_drive, [expected_fl_yaw, 0.0, 0.0], tol=1e-3))
    # Lesion: fl_drive = 0
    acts_fl_les = read_activations(state_ni, BrainParams(K_cereb_fl=0.0))
    _check("fl_drive == 0 under floccular lesion", _close(acts_fl_les.fl_drive, zero3))

    # step shape contract — returns (dstate, activations)
    print("\n[6] step shape contract")
    cb_rest = rest_state()
    dstate, acts_step = step(cb_rest,
                              ec_vel=jnp.array([100.0, 0.0, 0.0]),
                              ec_pos=jnp.zeros(3),
                              ni_net=zero3, ni_null=zero3,
                              target_slip=zero3,
                              target_visible=jnp.float32(1.0),
                              brain_params=bp)
    _check("dstate scene shape",  dstate.scene.shape  == (N_PER_PATH,))
    _check("dstate target shape", dstate.target.shape == (N_PER_PATH,))
    _check("step returns acts",   acts_step is not None)

    # JIT
    print("\n[7] JIT compatibility")
    read_jit = jax.jit(read_activations)
    step_jit = jax.jit(step)
    acts_jit = read_jit(state0, bp)
    dstate_jit, _ = step_jit(state0.cb, jnp.zeros(3), jnp.zeros(3),
                              zero3, zero3,
                              zero3, jnp.float32(1.0), bp)
    _check("read_activations JIT",  acts_jit is not None)
    _check("step JIT",              dstate_jit is not None)

    print("\n" + "=" * 60)
    print("All smoke tests passed.")
