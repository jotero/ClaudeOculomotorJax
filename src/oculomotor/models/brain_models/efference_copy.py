"""Efference copy — post-delay EC cascades for scene and target paths.

Architecture
────────────
The brain produces a version-velocity motor command (saccade burst + pursuit
+ T-VOR) that drives the eye AND contaminates the retinal signals.  To cancel
this contamination at the brain's perception stages, the command must be
added back to the delayed retinal signals — but AFTER the same visual delay,
so the EC and the slip arrive at the perception stage at the same time.

Two cascades run in parallel because the two slip paths in perception_cyclopean
use different smoothing TCs (scene = tighter, target = much smoother).  Each
cascade's impulse response is matched to its slip path's, so post-delay
subtraction cancels self-generated motion cleanly:

    Scene path:   tau_vis_smooth_motion        (~0.02 s LP)
    Target path:  tau_vis_smooth_target_vel    (~0.15 s LP)

Both paths share the same `tau_vis_sharp` gamma cascade upfront (matches the
per-eye retina sharp cascade); only the post-fusion LP stage differs.

Frame
─────
The motor command (`ec_vel`) is in head frame.  The retinal slip it must
cancel is in eye frame.  We rotate ec_vel into eye frame before feeding it
into the cascade, using the current eye position (`ec_pos`).  The pre-rotation
is then saturated by `v_max_okr` to match the retinal velocity-saturation
ceiling on scene_angular_vel.

State
─────
`State`:
    scene  — (_N_PER_PATH,)  scene-path EC cascade buffer (21 states)
    target — (_N_PER_PATH,)  target-path EC cascade buffer (21 states)

Outputs of `update`:
    dstate       — efference_copy.State   state derivative
    ec_d_scene   — (3,) delayed EC matched to scene_angular_vel cascade shape
    ec_d_target  — (3,) delayed EC matched to target_vel        cascade shape
"""

from typing import NamedTuple

import jax.numpy as jnp

from oculomotor.models.plant_models.readout import rotation_matrix
from oculomotor.models.sensory_models.retina import (
    cascade_lp_step, ypr_to_xyz, xyz_to_ypr, velocity_saturation,
)


# ── Cascade geometry ──────────────────────────────────────────────────────────
# The cascade has two stages: a sharp gamma cascade (shared with the per-eye
# retina sharp cascade) followed by a single LP smoothing stage.  Total length
# per path = (_N_SHARP + _N_LP) × _N_AXES.

_N_SHARP    = 6              # sharp cascade stages (matches retina)
_N_LP       = 1              # smoothing LP stages
_N_AXES     = 3
N_PER_PATH  = (_N_SHARP + _N_LP) * _N_AXES   # 21 states per cascade


# ── State NamedTuple ──────────────────────────────────────────────────────────

class State(NamedTuple):
    """Efference-copy cascade buffers — one per slip path."""
    scene:  jnp.ndarray   # (21,) cascade buffer matching scene_angular_vel impulse shape
    target: jnp.ndarray   # (21,) cascade buffer matching target_vel       impulse shape


def rest_state():
    """Zero state — both cascades empty."""
    return State(scene=jnp.zeros(N_PER_PATH), target=jnp.zeros(N_PER_PATH))


# ── Activations registry (delayed EC outputs) ────────────────────────────────

class Activations(NamedTuple):
    """Delayed EC values — what downstream perception stages read.

    Each is the cascade tail (last 3 elements of the buffer); they have the
    same impulse-response shape as the corresponding slip cascade in
    perception_cyclopean, so post-delay subtraction cancels self-generated
    motion.
    """
    scene:  jnp.ndarray   # (3,) delayed EC matched to scene_angular_vel cascade
    target: jnp.ndarray   # (3,) delayed EC matched to target_vel        cascade


def read_activations(state):
    """Project EC cascade state → Activations (delayed EC outputs)."""
    return Activations(scene=state.scene[-3:], target=state.target[-3:])


# ── Cascade advance ───────────────────────────────────────────────────────────

def step(state, ec_vel, ec_pos, brain_params):
    """Advance both EC cascades by one ODE step.

    The scene-path and target-path cascades each match their corresponding
    perception_cyclopean slip cascade — same gamma/LP TCs AND same retinal
    velocity-saturation ceiling (v_max_okr for scene = NOT/AOS; v_max_pursuit
    for target = MT/MST).  Equal saturation is essential: a saccade burst that
    saturates the scene_angular_vel cascade in perception_cyclopean must
    saturate the scene EC too, otherwise the EC overshoots and post-delay
    subtraction injects spurious slip.

    Args:
        state:        efference_copy.State
        ec_vel:       (3,) version velocity efference (head frame, deg/s)
        ec_pos:       (3,) eye position (head frame, deg) — current rotation
                          used to bring ec_vel into eye frame before feeding
                          the cascades
        brain_params: BrainParams (reads tau_vis_sharp,
                                   tau_vis_smooth_motion, tau_vis_smooth_target_vel,
                                   v_max_okr, v_max_pursuit)

    Returns:
        dstate: efference_copy.State  state derivative (cascade dx)
    """
    R_eye      = rotation_matrix(ypr_to_xyz(ec_pos))
    ec_vel_eye = xyz_to_ypr(R_eye.T @ ypr_to_xyz(ec_vel))
    ec_vel_scene_in  = velocity_saturation(ec_vel_eye, brain_params.v_max_okr)      # NOT/AOS
    ec_vel_target_in = velocity_saturation(ec_vel_eye, brain_params.v_max_pursuit)  # MT/MST
    return State(
        scene  = cascade_lp_step(state.scene,  ec_vel_scene_in,
                                  brain_params.tau_vis_sharp,
                                  brain_params.tau_vis_smooth_motion,
                                  _N_SHARP, _N_AXES, _N_LP),
        target = cascade_lp_step(state.target, ec_vel_target_in,
                                  brain_params.tau_vis_sharp,
                                  brain_params.tau_vis_smooth_target_vel,
                                  _N_SHARP, _N_AXES, _N_LP),
    )
