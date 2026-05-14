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
per-eye retina sharp cascade).  The predicted eye velocity (eye_vel_pred,
head frame) is rotated into eye frame using x_p_pred, then saturated by
v_max_okr (scene) and v_max_pursuit (target) to mirror the retinal
saturation.

Motor-pathway forward model
───────────────────────────
The cerebellum runs a *literal copy* of the chain the eye actually sees.
In simulator.py the plant is driven by the FCP `nerves` output (NOT by
motor_cmd_ni directly), so the chain is:

    NI  →  pulse-step  ([NI_net + tau_p·ec_vel,  vergence ≈ 0])
        →  fcp.step     (14-state MN forward model — M_NUCLEUS encode, ×2
                          reciprocal compensation, NERVE_MAX f-I clips, MLF
                          cross-projection, tau_mn LP, nerve caps, r_baseline)
                          → nerves_pred (12,)
        →  per-eye plant LP  (M_PLANT_EYE_{L,R} @ nerves_pred → tau_p LP)
        →  eye  (x_p_pred = ½·(plant_pred_L + plant_pred_R), head frame)

This carries the FCP *nonlinearities* (rectification floors, f-I ceilings,
MLF caps) — at large saccade amplitudes the real eye velocity is well below
the linear command→velocity map, so a linear forward model would over-
predict it and the EC would over-cancel the retinal slip.  The head→eye
rotation that builds the EC uses x_p_pred — the *actual* eye position the
retina rotates through, which lags NI_net by several degrees mid-saccade
(rotating the EC through NI_net would inject a large eccentricity-dependent
frame mismatch).  At rest all forward-model states start at 0 and warm up
within a few tau_mn / tau_p (covered by the simulator's warmup phase).

State
─────
    scene        — (_N_PER_PATH,)  scene-path EC cascade buffer (21 states)
    target       — (_N_PER_PATH,)  target-path EC cascade buffer (21 states)
    fcp_pred     — (14,)           FCP MN-membrane forward-model state
    plant_pred_L — (3,)            left-eye plant forward-model rotation vec
    plant_pred_R — (3,)            right-eye plant forward-model rotation vec
    sat_*        — (7,) ×2         saturation-flag delay cascades

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

from oculomotor.models.brain_models import final_common_pathway as fcp
from oculomotor.models.plant_models.readout import rotation_matrix
from oculomotor.models.plant_models.muscle_geometry import M_PLANT_EYE_L, M_PLANT_EYE_R
from oculomotor.models.sensory_models.retina import (
    cascade_lp_step, ypr_to_xyz, xyz_to_ypr, velocity_saturation,
)


# ── Cascade geometry ──────────────────────────────────────────────────────────
# 6-stage sharp gamma + 1-stage smoothing LP, on 3 axes per cascade.

_N_SHARP   = 6
_N_LP      = 1
_N_AXES    = 3
N_PER_PATH = (_N_SHARP + _N_LP) * _N_AXES   # 21 states per cascade

_N_FCP   = fcp.N_STATES                     # 14 MN-membrane states (forward-model copy)
_N_PLANT = 3                                # per-eye plant rotation vector
_N_SAT_PER_PATH = (_N_SHARP + _N_LP) * 1    # 7 scalar cascade states for sat-flag
N_STATES   = 2 * N_PER_PATH + _N_FCP + 2 * _N_PLANT + 2 * _N_SAT_PER_PATH
                                            # scene + target EC +
                                            # FCP forward model (14) +
                                            # per-eye plant forward model (3 + 3) +
                                            # two scalar saturation-flag cascades


# ── State + Activations ───────────────────────────────────────────────────────

class State(NamedTuple):
    """Cerebellum state — EC cascade buffers + motor-pathway forward model.

    The forward model is a *literal copy* of the actual motor chain the eye
    (and hence the retina) sees — the same `fcp.step` + first-order plant the
    simulator uses, fed the cerebellum's own copy of the motor command:

        NI  →  pulse-step  ([NI_net + tau_p·ec_vel,  vergence≈0])
            →  fcp.step      (14-state MN forward model: M_NUCLEUS encode,
                              ×2 reciprocal compensation, NERVE_MAX clips,
                              MLF cross-projection, tau_mn LP, nerve caps,
                              r_baseline tonic)  → nerves_pred (12,)
            →  per-eye plant LP   (M_PLANT_EYE_{L,R} @ nerves_pred → tau_p LP)
            →  eye  (x_p_pred = ½·(plant_pred_L + plant_pred_R), head frame)

    Crucially this carries the FCP's *nonlinearities* (rectification floors,
    f-I ceilings, MLF caps) — at large saccade amplitudes the actual eye
    velocity is well below the linear command→velocity map, and a linear
    forward model would over-predict it (and hence the EC would over-cancel
    the retinal slip).  The head→eye rotation that builds the EC uses
    `x_p_pred` — the *actual* eye position the retina rotates through, which
    lags NI_net by several degrees mid-saccade.

    At rest all forward-model states start at zero and warm up within a few
    tau_mn / tau_p of t=0 (covered by the simulator's warmup phase).
    """
    scene:        jnp.ndarray   # (21,) cascade buffer matching scene_angular_vel
    target:       jnp.ndarray   # (21,) cascade buffer matching target_vel
    fcp_pred:     jnp.ndarray   # (14,) FCP MN-membrane forward-model state
    plant_pred_L: jnp.ndarray   # (3,)  left-eye plant forward-model rotation vec
    plant_pred_R: jnp.ndarray   # (3,)  right-eye plant forward-model rotation vec
    sat_scene:    jnp.ndarray   # (7,)  scene saturation-flag scalar cascade
    sat_target:   jnp.ndarray   # (7,)  target saturation-flag scalar cascade


class Activations(NamedTuple):
    """Cerebellar readouts."""
    # EC delay cascade outputs (delayed self-motion predictions)
    ec_scene:        jnp.ndarray   # (3,) delayed EC for scene path
    ec_target:       jnp.ndarray   # (3,) delayed EC for target path
    # Cerebellar EC corrections (gated by the saccadic-suppression factor).
    # Both have the form `sat · visibility · ec_no_torsion` and sum cleanly
    # with the brainstem direct path (= sat · raw_slip) in brain_model.
    pred_err:        jnp.ndarray   # (3,) target_slip + ec_correction (PE, diagnostic)
    vpf_drive:       jnp.ndarray   # (3,) gated target EC → pursuit (Ventral paraflocculus)
    fl_okr_drive:    jnp.ndarray   # (3,) gated scene EC → VS (Flocculus / vermis OKR)
    # Saccadic-suppression gates (1 − cascaded saturation flag).  Applied to
    # the RAW retinal input (cyc.scene_angular_vel / cyc.target_vel) directly
    # in brain_model; cerebellum's own EC corrections above are pre-gated by
    # the same factor so brainstem-direct + cerebellar sum stays consistent.
    saccadic_suppression_scene:  jnp.ndarray   # scalar  → multiplies cyc.scene_angular_vel
    saccadic_suppression_target: jnp.ndarray   # scalar  → multiplies cyc.target_vel
    # Flocculus (FL) — leak cancellation (Cannon & Robinson 1985)
    fl_drive:        jnp.ndarray   # (3,) NI leak-cancellation feedback → u_ni_in
    fl_vs_drive:     jnp.ndarray   # (3,) VS leak-cancellation feedback → VS dx_pop
    # Nodulus + uvula (NU) — VS axis alignment with gravity (cerebellum.md §4.4)
    nu_drive:    jnp.ndarray   # (3,) gravity-axis dumping signal → VS


def rest_state():
    """Zero state — both EC cascades + motor forward model + sat cascades."""
    return State(scene=jnp.zeros(N_PER_PATH),
                 target=jnp.zeros(N_PER_PATH),
                 fcp_pred=jnp.zeros(_N_FCP),
                 plant_pred_L=jnp.zeros(_N_PLANT),
                 plant_pred_R=jnp.zeros(_N_PLANT),
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
                   scene_visible=brain_state.pc.scene_visible[-1],
                   brain_params=brain_params)
    return acts


# ── State step (advances the EC cascades) ─────────────────────────────────────

def step(state, ec_vel, ec_pos, ni_net, ni_null,
         vs_net, vs_null, rf, w_est,
         target_slip, target_visible, scene_visible, brain_params):
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
         velocity-saturation ceiling).  Pipeline: motor-pathway forward
         model (pulse-step → fcp.step [MN + MLF + clips] → per-eye plant LP
         → x_p_pred) → head→eye rotation through x_p_pred → retinal velocity
         saturation → visual delay cascade.  Using x_p_pred (≈ the actual
         eye position the retina rotates through, which lags NI_net and
         carries the FCP nonlinearities) rather than ec_pos (= NI_net) for
         the rotation keeps the EC matched to the retinal slip mid-saccade.

    Args:
        state:           cerebellum.State
        ec_vel:          (3,)    version velocity efference (head frame, deg/s)
        ec_pos:          (3,)    eye position command = NI_net (head frame, deg)
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

    # ── Cerebellar pursuit contribution: CB2 ──────────────────────────────────
    # The pursuit input downstream is:
    #     pursuit_in = K_pursuit_direct · target_slip + K_cereb_pu · vpf_drive
    # where the first term is the brainstem direct (raw retinal slip) path and
    # the second is THIS cerebellar contribution.  Defining
    #     vpf_drive = sat_gate · (target_visible · ec_no_torsion)
    #               − (1 − sat_gate) · target_slip
    # makes the SUM at default gains (K_pursuit_direct = K_cereb_pu = 1) equal
    # to the previous full output:
    #     pursuit_in = slip + sat·vis·ec − (1−sat)·slip = sat·(slip + vis·ec)
    # The first half (sat · vis · ec) is the EC correction (cancels self-
    # motion contamination of slip).  The second half (−(1−sat) · slip) is
    # the cerebellum's *cancellation of the brainstem direct path* during
    # saccades — so when the gate closes, the brainstem slip is silenced.
    #
    # Lesion (K_cereb_pu = 0): pursuit driven by raw slip alone — no EC
    # correction (reduced gain during head motion) AND no saccadic
    # suppression of pursuit (the classic flocculus phenotype).
    ec_correction = target_visible * ec_no_torsion
    pred_err      = target_slip + ec_correction              # diagnostic only

    # ── EC cascade advance ────────────────────────────────────────────────
    # Order mirrors the path from motor command to retinal cascade:
    #   pulse-step sum -> FCP (MN + MLF + clips) -> plant LP -> rotation
    #   (head->eye) -> retinal saturation -> visual delays.

    # 1. Motor-pathway forward model (head frame) — a literal copy of the
    #    chain the eye actually sees: the same fcp.step + first-order plant
    #    the simulator uses, fed the cerebellum's own copy of the motor
    #    command.  Carrying the FCP nonlinearities (rectification floors,
    #    NERVE_MAX f-I ceilings, MLF caps, x2 reciprocal compensation,
    #    r_baseline) matters: at large saccade amplitudes the real eye
    #    velocity is well below the linear command->velocity map, so a linear
    #    forward model over-predicts it and the EC over-cancels the slip.
    #        premotor_pred = [NI_net + tau_p*(ec_vel + fl_drive),  verg ~ 0]  (6,)
    #        d_fcp, nerves_pred = fcp.step(read_activations(fcp_pred),
    #                                       premotor_pred, brain_params)
    #        d_plant_pred_{L,R} = (M_PLANT_EYE_{L,R} @ nerves_pred[half]
    #                              - plant_pred_{L,R}) / tau_p
    #        x_p_pred     = 0.5*(plant_pred_L + plant_pred_R)   (conjugate eye pos)
    #        eye_vel_pred = 0.5*(d_plant_pred_L + d_plant_pred_R)
    #    The version command is `NI_net + tau_p*u_ni_in` where u_ni_in =
    #    ec_vel + fl_drive (+ Listing + direct-VOR terms, ~0 for a stationary-
    #    head saccade — not added here).  fl_drive (the floccular leak-cancel
    #    feedback) is included so the forward-model eye velocity matches the
    #    real one to <0.5 deg/s.  Vergence in the forward model is taken as 0
    #    (saccade-conjugate scope); the orbital wall clip is omitted.
    premotor_pred = jnp.concatenate(
        [ec_pos + bp.tau_p * (ec_vel + fl_drive), jnp.zeros(3)])    # [version, verg=0]
    fcp_acts_pred = fcp.read_activations(fcp.State(mn=state.fcp_pred))
    d_fcp_state, nerves_pred = fcp.step(fcp_acts_pred, premotor_pred, bp)
    d_fcp_pred = d_fcp_state.mn
    motor_cmd_pred_L = M_PLANT_EYE_L @ nerves_pred[:6]
    motor_cmd_pred_R = M_PLANT_EYE_R @ nerves_pred[6:]
    d_plant_pred_L   = (motor_cmd_pred_L - state.plant_pred_L) / bp.tau_p
    d_plant_pred_R   = (motor_cmd_pred_R - state.plant_pred_R) / bp.tau_p
    x_p_pred     = 0.5 * (state.plant_pred_L + state.plant_pred_R)   # conjugate eye pos (head frame)
    eye_vel_pred = 0.5 * (d_plant_pred_L  + d_plant_pred_R)         # conjugate eye vel (head frame)

    # 2. Rotation: head-frame velocity → eye-frame velocity using the
    #    predicted plant position x_p_pred.  Same transform the retina
    #    applies to the actual eye velocity through the actual eye position —
    #    so using x_p_pred (not ec_pos = NI_net) keeps the frame-mix term in
    #    the EC matched to the retina even mid-saccade.
    R_eye         = rotation_matrix(ypr_to_xyz(x_p_pred))
    eye_vel_pred_eye = xyz_to_ypr(R_eye.T @ ypr_to_xyz(eye_vel_pred))
    w_est_eye        = xyz_to_ypr(R_eye.T @ ypr_to_xyz(w_est))

    # 3. Retinal velocity saturation, offset by −w_est_eye: gain rolloff is
    #    computed on the residual (eye_vel_pred + w_est ≈ eye_vel_in_world).
    #    During perfect VOR the residual is zero so the cascade input passes
    #    through unchanged.
    v_offset_eye     = -w_est_eye
    ec_vel_scene_in  = velocity_saturation(eye_vel_pred_eye, bp.v_max_okr,
                                            v_offset=v_offset_eye)      # NOT/AOS
    ec_vel_target_in = velocity_saturation(eye_vel_pred_eye, bp.v_max_pursuit,
                                            v_offset=v_offset_eye)      # MT/MST

    # 4. Saccadic-suppression flags = 1 − cosine-rolloff gain.  Detects when
    #    the residual eye-velocity (eye_vel_pred_eye + w_est_eye) is fast
    #    enough that the retinal-pathway saturation is engaged.  Two flags
    #    because target (v_max_pursuit) and scene (v_max_okr) have different
    #    thresholds.  Flag ∈ [0, 1]:  0 = no saturation, 1 = full clamp.
    v_rel  = eye_vel_pred_eye - v_offset_eye       # = eye_vel_pred_eye + w_est_eye
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

    # Cerebellar EC corrections — pre-gated by saccadic suppression.
    # brain_model assembles:
    #     pursuit_in     = K_pursuit_direct · sat_target · cyc.target_vel + K_cereb_pu  · vpf_drive
    #     slip_pe_for_vs = K_vor_direct     · sat_scene  · cyc.scene_ang  + K_cereb_okr · fl_okr_drive
    # so both the brainstem-direct path and the cerebellar EC scale with the
    # same gate.  The gate must be MILD (`saccadic_suppression_threshold ≈ 0`,
    # `steepness ≈ 1` → gate ≈ 1 − cascaded_sat) — an aggressive strengthen
    # would throttle pursuit between catch-up saccades, where the gate cascade
    # has not fully reopened.
    vpf_drive    = saccadic_suppression_target * target_visible * ec_no_torsion
    fl_okr_drive = saccadic_suppression_scene  * scene_visible  * ec_scene

    dstate = State(
        scene  = cascade_lp_step(state.scene,  ec_vel_scene_in,
                                  bp.tau_vis_sharp,
                                  bp.tau_vis_smooth_motion,
                                  _N_SHARP, _N_AXES, _N_LP),
        target = cascade_lp_step(state.target, ec_vel_target_in,
                                  bp.tau_vis_sharp,
                                  bp.tau_vis_smooth_target_vel,
                                  _N_SHARP, _N_AXES, _N_LP),
        fcp_pred     = d_fcp_pred,
        plant_pred_L = d_plant_pred_L,
        plant_pred_R = d_plant_pred_R,
        sat_scene    = d_sat_scene,
        sat_target   = d_sat_target,
    )
    acts = Activations(ec_scene=ec_scene, ec_target=ec_target,
                       pred_err=pred_err, vpf_drive=vpf_drive,
                       fl_okr_drive=fl_okr_drive,
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

    # Cerebellum outputs vpf_drive = sat · target_visible · ec_no_torsion
    # (the gated EC correction).  Brain_model assembles the full CB2:
    #   CB2 = vpf_drive − (1 − sat) · cyc.target_vel
    #   pursuit_in = K_pursuit_direct · cyc.target_vel + K_cereb_pu · CB2
    print("\n[3] vpf_drive ≈ 0 when no EC (cerebellum has no correction to offer)")
    state_t = state0._replace(pc=state0.pc._replace(
        target_vel=jnp.array([10.0, 0.0, 0.0]),
        target_visible=state0.pc.target_visible.at[-1].set(1.0),
    ))
    acts = read_activations(state_t, bp)
    _check("pred_err == target_slip when ec=0", _close(acts.pred_err, [10.0, 0.0, 0.0]))
    _check("vpf_drive ≈ 0 when ec=0",  _close(acts.vpf_drive, zero3, tol=1e-3))

    # EC cascade injection
    print("\n[4] EC cascade tail: vpf_drive = sat · vis · ec_no_torsion")
    cb_state = state0.cb._replace(target=state0.cb.target.at[-3].set(2.0).at[-1].set(5.0))
    state_e  = state_t._replace(cb=cb_state)
    acts = read_activations(state_e, bp)
    _check("ec_target reads cascade tail", _close(acts.ec_target, [2.0, 0.0, 5.0]))
    _check("pred_err yaw = 10 + 1 · 2 = 12",
           _close(acts.pred_err, [12.0, 0.0, 0.0]))
    _check("vpf_drive torsion zeroed", abs(float(acts.vpf_drive[2])) < 1e-5)
    # With sat ≈ 1 at rest: vpf_drive = sat · vis · ec_no_torsion = 1 · 1 · 2 = 2
    _check("vpf_drive yaw ≈ 2 (gate open)",
           abs(float(acts.vpf_drive[0]) - 2.0) < 1e-3)

    # Cerebellum-internal output is invariant to brain-side K_cereb_pu;
    # the lesion is applied downstream in brain_model.
    print("\n[5] vpf_drive invariant under K_cereb_pu (lesion handled in brain_model)")
    bp_les = BrainParams(K_cereb_pu=0.0)
    acts_les = read_activations(state_e, bp_les)
    _check("pred_err unchanged", _close(acts_les.pred_err, [12.0, 0.0, 0.0]))
    _check("vpf_drive unchanged",
           _close(acts_les.vpf_drive, acts.vpf_drive))
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
                              scene_visible=jnp.float32(1.0),
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
                              zero3, jnp.float32(1.0), jnp.float32(1.0), bp)
    _check("read_activations JIT",  acts_jit is not None)
    _check("step JIT",              dstate_jit is not None)

    print("\n" + "=" * 60)
    print("All smoke tests passed.")
