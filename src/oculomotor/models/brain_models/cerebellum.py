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

Pursuit region (cerebellum.md §4.3) — pure prediction error
────────────────────────────────────────────────────────────
    predicted_self_slip = − K_cereb_pu · target_visible · ec_d_target_no_torsion
    pred_err            = target_slip − predicted_self_slip
                        = target_slip + K_cereb_pu · target_visible · ec_d_target_no_torsion
    vpf_drive           = pred_err                              → pursuit

No gates.  Pursuit's internal Smith predictor handles steady-state self-
sustaining: at SS, pred_err ≈ +V (eye velocity), pursuit memory matches,
and the Smith stage cancels the surplus.

Lesion (K_cereb_pu = 0): no EC correction → pursuit driven by raw self-
motion-contaminated slip → classic flocculus phenotype (reduced effective
gain when chasing stationary targets during head motion).

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
    pred_err   — (3,) target_slip + K_cereb_pu · vis · ec_no_torsion
    vpf_drive  — (3,) cerebellar pursuit drive = pred_err

Params: read flat from BrainParams — K_cereb_pu (0 = lesion) and the
cascade params (tau_vis_sharp, tau_vis_smooth_motion,
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

_N_SAT_PER_PATH = (_N_SHARP + _N_LP) * 1    # 7 scalar cascade states for sat-flag
N_STATES   = 2 * N_PER_PATH + 6 + 2 * _N_SAT_PER_PATH
                                            # scene + target EC + two MN LPs + two
                                            # scalar saturation-flag cascades


# ── State + Activations ───────────────────────────────────────────────────────

class State(NamedTuple):
    """Cerebellum state — EC cascade buffers + two-stage MN forward-model LP.

    `mn_lp1` and `mn_lp2` are the two cascaded MN-membrane LPs that mirror
    the FCP two-stage motoneuron pathway.  In real anatomy, one half of the
    horizontal yaw command reaches its muscle through a single MN stage (LR
    direct via the ipsilateral cranial nerve) and the other half through
    two MN stages (MR via abducens-internuclear → MLF → CN3 medial-rectus
    MN).  Both stages share `tau_mn`.

    The effective MN-stage delay is per-axis: yaw uses 0.5·(mn_lp1+mn_lp2)
    (mix of 1-stage and 2-stage paths) while pitch/roll use mn_lp1 alone
    (CN3/CN4 vertical/torsional MNs receive premotor directly in the model
    — no MLF stage on the vertical axis).
    """
    scene:      jnp.ndarray   # (21,) cascade buffer matching scene_angular_vel
    target:     jnp.ndarray   # (21,) cascade buffer matching target_vel
    mn_lp1:     jnp.ndarray   # (3,)  first MN-membrane LP (AIN/ABN side)
    mn_lp2:     jnp.ndarray   # (3,)  second MN-membrane LP (CN3-MR side, via MLF)
    sat_scene:  jnp.ndarray   # (7,)  scene saturation-flag scalar cascade
    sat_target: jnp.ndarray   # (7,)  target saturation-flag scalar cascade


class Activations(NamedTuple):
    """Cerebellar readouts."""
    # EC delay cascade outputs (delayed self-motion predictions)
    ec_scene:        jnp.ndarray   # (3,) delayed EC for scene path
    ec_target:       jnp.ndarray   # (3,) delayed EC for target path
    # Ventral paraflocculus (VPF) — smooth pursuit forward model
    pred_err:        jnp.ndarray   # (3,) target_slip + ec_correction (PE)
    vpf_drive:       jnp.ndarray   # (3,) gated cerebellar pursuit drive → pursuit_in
    # Saccadic-suppression gates (1 − cascaded saturation flag).  Multiplied
    # onto the corresponding PE so retinal evidence is down-weighted during
    # high-speed self-motion (when both slip and EC have saturated).
    saccadic_suppression_scene:  jnp.ndarray   # scalar  → multiplies scene PE in sm.step
    saccadic_suppression_target: jnp.ndarray   # scalar  → multiplies target PE (vpf_drive)
    # Flocculus (FL) — leak cancellation (Cannon & Robinson 1985)
    fl_drive:        jnp.ndarray   # (3,) NI leak-cancellation feedback → u_ni_in
    fl_vs_drive:     jnp.ndarray   # (3,) VS leak-cancellation feedback → VS dx_pop
    # Nodulus + uvula (NU) — VS axis alignment with gravity (cerebellum.md §4.4)
    nu_drive:    jnp.ndarray   # (3,) gravity-axis dumping signal → VS


def rest_state():
    """Zero state — both EC cascades + both MN LPs + sat-flag cascades empty."""
    return State(scene=jnp.zeros(N_PER_PATH),
                 target=jnp.zeros(N_PER_PATH),
                 mn_lp1=jnp.zeros(3),
                 mn_lp2=jnp.zeros(3),
                 sat_scene=jnp.zeros(_N_SAT_PER_PATH),
                 sat_target=jnp.zeros(_N_SAT_PER_PATH))


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
    vs_net  = brain_state.sm.vs_L - brain_state.sm.vs_R
    vs_null = brain_state.sm.vs_null
    rf      = brain_state.sm.rf
    # w_est is computed inside sm.step from canal+visual inputs; we don't
    # have it at read_activations time, so pass zeros.  Activations (pred_err,
    # vpf_drive, fl_drive, fl_vs_drive, nu_drive) depend only on state, not
    # on saturation, so the dummy w_est here doesn't affect them — only the
    # discarded dstate (cascade derivative) is affected.
    _, acts = step(brain_state.cb,
                   ec_vel=jnp.zeros(3), ec_pos=jnp.zeros(3),
                   ni_net=ni_net, ni_null=ni_null,
                   vs_net=vs_net, vs_null=vs_null, rf=rf,
                   w_est=jnp.zeros(3),
                   target_slip=brain_state.pc.target_vel,
                   target_visible=brain_state.pc.target_visible[-1],
                   brain_params=brain_params)
    return acts


# ── State step (advances the EC cascades) ─────────────────────────────────────

def step(state, ec_vel, ec_pos, ni_net, ni_null,
         vs_net, vs_null, rf, w_est,
         target_slip, target_visible, brain_params):
    """Cerebellum per-step processing.

    Three jobs:
      1. **Flocculus (FL) — NI integrator extension** (Cannon & Robinson
         1985).  Reads NI net + null state; outputs leak-cancellation
         feedback added back to NI input downstream:
             fl_drive = K_cereb_fl · (ni_net − ni_null) / tau_i_per_axis

      2. **VPF — pursuit forward model** (cerebellum.md §4.3):
             predicted_self_slip = − K_cereb_pu · vis · ec_d_target_no_torsion
             pred_err            = target_slip − predicted_self_slip
             vpf_drive           = pred_err                      (no gates)

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

    # ── Flocculus (FL): VS leak cancellation ──────────────────────────────
    # Same Cannon-Robinson architecture, applied to the velocity-storage
    # integrator instead of the position integrator.  Extends the effective
    # tau_vs from the brainstem-only ~5 s (≈ canal TC) to the cerebellum-
    # extended ~20 s.  Lesion (K_cereb_fl = 0) → shortened OKAN, reduced
    # velocity storage TC.
    tau_vs_axes = jnp.array([bp.tau_vs,
                              bp.tau_vs * bp.tau_vs_pitch_frac,
                              bp.tau_vs * bp.tau_vs_roll_frac])
    fl_vs_drive = bp.K_cereb_fl_vs * (vs_net - vs_null) / tau_vs_axes

    # ── Nodulus + uvula (NU): VS axis dumping toward gravity ─────────────
    # Cohen, Raphan, Wearne — drives the VS rotation-axis component
    # perpendicular to gravity toward zero (tilt suppression of post-rotatory
    # nystagmus, OVAR sustained nystagmus).  Replaces the old K_gd · rf
    # computation that lived inside sm._vs_step; rf comes from the
    # gravity-estimator state and represents the perpendicular component.
    # Lesion (K_cereb_nu = 0) → no dumping → prolonged tau_vs + PAN
    # (periodic alternating nystagmus) + loss of tilt suppression.
    nu_drive = bp.K_cereb_nu * bp.K_gd * rf

    # ── VPF: pursuit forward model (uses cerebellum's own state for ec_target) ──
    ec_scene  = state.scene[-3:]
    ec_target = state.target[-3:]
    ec_no_torsion = ec_target.at[2].set(0.0)

    # ── Pure prediction error (no gates) ──────────────────────────────────────
    # vpf_drive = pred_err = slip + EC_correction
    #
    # The cerebellum predicts the slip that would arise from current self-
    # motion alone (predicted_self_slip = −ec_correction).  The prediction
    # error is the discrepancy between actual slip and that prediction:
    #     PE = slip − predicted_self_slip = slip + ec_correction
    # Pursuit drives on the PE; the pursuit module's internal Smith predictor
    # handles steady-state self-sustaining.
    #
    # K_cereb_pu scales the cerebellum's EC contribution (lesion → K=0 → no
    # self-motion cancellation → raw slip drives pursuit; classic flocculus
    # phenotype with reduced effective gain during head motion).
    ec_correction = bp.K_cereb_pu * target_visible * ec_no_torsion
    pred_err      = target_slip + ec_correction
    vpf_drive     = pred_err

    # ── EC cascade advance ────────────────────────────────────────────────
    # Order mirrors the vision pathway from motor command to retinal cascade:
    #   MN lag → rotation (head→eye frame) → saturation → visual delays.
    # The motor command lives in the head frame, the MN pool low-passes it
    # there, the plant then turns that into eye rotation (frame transform
    # via ec_pos), and only then does the retinal saturation see it.

    # 1. Two-stage MN forward-model LP (head frame).  Mirrors the FCP MN
    #    pathway: half the conjugate yaw command reaches its muscle through
    #    a single MN stage (ABN → LR direct via cranial nerve) and the
    #    other half through two stages (AIN → MLF → CN3_MR).  Vertical and
    #    torsional commands go through a single CN3/CN4 MN — no MLF.
    #    Per-axis weighting:
    #        yaw (axis 0): 0.5 · (mn_lp1 + mn_lp2)
    #        pitch/roll  : mn_lp1   (no MLF stage modelled)
    d_mn1 = (ec_vel       - state.mn_lp1) / bp.tau_mn
    d_mn2 = (state.mn_lp1 - state.mn_lp2) / bp.tau_mn
    w_mlf    = bp.mlf_axis_weight                          # (3,) — yaw 0.5, others 0
    mn_lp_in = (1.0 - w_mlf) * state.mn_lp1 + w_mlf * state.mn_lp2

    # 2. Rotation: head-frame velocity → eye-frame velocity using current
    #    ec_pos.  Same transform the retina applies to actual eye velocity.
    R_eye        = rotation_matrix(ypr_to_xyz(ec_pos))
    mn_lp_in_eye = xyz_to_ypr(R_eye.T @ ypr_to_xyz(mn_lp_in))
    w_est_eye    = xyz_to_ypr(R_eye.T @ ypr_to_xyz(w_est))

    # 3. Retinal velocity saturation, offset by −w_est_eye: gain rolloff is
    #    computed on the residual (motor_cmd + w_est ≈ eye_vel_in_world).
    #    During perfect VOR the residual is zero so the cascade input passes
    #    through unchanged.
    v_offset_eye     = -w_est_eye
    ec_vel_scene_in  = velocity_saturation(mn_lp_in_eye, bp.v_max_okr,
                                            v_offset=v_offset_eye)      # NOT/AOS
    ec_vel_target_in = velocity_saturation(mn_lp_in_eye, bp.v_max_pursuit,
                                            v_offset=v_offset_eye)      # MT/MST

    # 4. Saccadic-suppression flags = 1 − cosine-rolloff gain.  Detects when
    #    the residual eye-velocity (mn_lp_in_eye + w_est_eye) is fast enough
    #    that the retinal-pathway saturation is engaged.  Two flags because
    #    target (v_max_pursuit) and scene (v_max_okr) have different
    #    thresholds.  Flag ∈ [0, 1]:  0 = no saturation, 1 = full clamp.
    v_rel  = mn_lp_in_eye - v_offset_eye           # = mn_lp_in_eye + w_est_eye
    speed  = jnp.linalg.norm(v_rel)

    def _sat_flag(spd, v_sat):
        v_zero = 2.0 * v_sat
        t      = jnp.clip((spd - v_sat) / (v_zero - v_sat), 0.0, 1.0)
        gain   = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
        return 1.0 - gain

    sat_scene_now  = _sat_flag(speed, bp.v_max_okr)
    sat_target_now = _sat_flag(speed, bp.v_max_pursuit)

    # 5. Visual delay cascade for the saturation flags — same gamma + LP TCs
    #    as their corresponding EC paths, so the gate is delay-aligned with
    #    the slip / EC it modulates.
    d_sat_scene  = cascade_lp_step(state.sat_scene,  sat_scene_now,
                                    bp.tau_vis_sharp,
                                    bp.tau_vis_smooth_motion,
                                    _N_SHARP, 1, _N_LP)
    d_sat_target = cascade_lp_step(state.sat_target, sat_target_now,
                                    bp.tau_vis_sharp,
                                    bp.tau_vis_smooth_target_vel,
                                    _N_SHARP, 1, _N_LP)

    # 6. Visual delay cascade for the slip/EC signal (gamma sharp + LP) below.

    # Cascaded saturation flags (LP tails) → gates applied to PEs downstream.
    # Contrast-amplification: `clip((raw_gate − thr)/(1 − thr), 0, 1) ** k`
    # pushes intermediate values toward 0 so only near-zero cascaded_sat
    # produces an open gate.  Threshold 0 + steepness 1 reproduces the raw
    # linear `1 − sat_d` gate.
    sat_d_scene  = state.sat_scene[-1]
    sat_d_target = state.sat_target[-1]
    thr   = bp.saccadic_suppression_threshold
    k     = bp.saccadic_suppression_steepness

    def _strengthen(raw):
        x = jnp.clip((raw - thr) / (1.0 - thr + 1e-9), 0.0, 1.0)
        return x ** k

    saccadic_suppression_scene  = _strengthen(1.0 - sat_d_scene)
    saccadic_suppression_target = _strengthen(1.0 - sat_d_target)

    # Apply target gate inside cerebellum (pursuit pathway).
    vpf_drive = saccadic_suppression_target * vpf_drive

    dstate = State(
        scene  = cascade_lp_step(state.scene,  ec_vel_scene_in,
                                  bp.tau_vis_sharp,
                                  bp.tau_vis_smooth_motion,
                                  _N_SHARP, _N_AXES, _N_LP),
        target = cascade_lp_step(state.target, ec_vel_target_in,
                                  bp.tau_vis_sharp,
                                  bp.tau_vis_smooth_target_vel,
                                  _N_SHARP, _N_AXES, _N_LP),
        mn_lp1     = d_mn1,
        mn_lp2     = d_mn2,
        sat_scene  = d_sat_scene,
        sat_target = d_sat_target,
    )
    acts = Activations(ec_scene=ec_scene, ec_target=ec_target,
                       pred_err=pred_err, vpf_drive=vpf_drive,
                       saccadic_suppression_scene=saccadic_suppression_scene,
                       saccadic_suppression_target=saccadic_suppression_target,
                       fl_drive=fl_drive, fl_vs_drive=fl_vs_drive,
                       nu_drive=nu_drive)
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

    # Pure prediction-error formulation: vpf_drive = pred_err = slip + K·vis·ec.
    print("\n[3] Pure PE: pred_err = target_slip + K_cereb_pu · vis · ec_no_torsion")
    state_t = state0._replace(pc=state0.pc._replace(
        target_vel=jnp.array([10.0, 0.0, 0.0]),
        target_visible=state0.pc.target_visible.at[-1].set(1.0),
    ))
    acts = read_activations(state_t, bp)
    _check("pred_err == target_slip when ec=0", _close(acts.pred_err, [10.0, 0.0, 0.0]))
    _check("vpf_drive == pred_err",             _close(acts.vpf_drive, acts.pred_err))

    # EC cascade injection
    print("\n[4] EC cascade tail: pred_err = ts + vis · ec_no_torsion, drive = pred_err")
    cb_state = state0.cb._replace(target=state0.cb.target.at[-3].set(2.0).at[-1].set(5.0))
    state_e  = state_t._replace(cb=cb_state)
    acts = read_activations(state_e, bp)
    _check("ec_target reads cascade tail", _close(acts.ec_target, [2.0, 0.0, 5.0]))
    _check("pred_err yaw = 10 + 1 · 2 = 12",
           _close(acts.pred_err, [12.0, 0.0, 0.0]))
    _check("vpf_drive torsion zeroed",     abs(float(acts.vpf_drive[2])) < 1e-5)
    _check("vpf_drive yaw = pred_err yaw = 12",
           abs(float(acts.vpf_drive[0]) - 12.0) < 1e-3)

    # Lesion: K_cereb_pu = 0 → no EC correction → pred_err = raw target_slip.
    print("\n[5] Cerebellar pursuit lesion (K_cereb_pu = 0) → no EC correction")
    bp_les = BrainParams(K_cereb_pu=0.0)
    acts_les = read_activations(state_e, bp_les)
    _check("pred_err = target_slip (no EC correction)",
           _close(acts_les.pred_err, [10.0, 0.0, 0.0]))
    _check("vpf_drive = pred_err under lesion",
           _close(acts_les.vpf_drive, acts_les.pred_err))
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
                              vs_net=zero3, vs_null=zero3, rf=zero3,
                              w_est=zero3,
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
                              zero3, zero3, zero3, zero3, zero3, zero3,
                              zero3, jnp.float32(1.0), bp)
    _check("read_activations JIT",  acts_jit is not None)
    _check("step JIT",              dstate_jit is not None)

    print("\n" + "=" * 60)
    print("All smoke tests passed.")
