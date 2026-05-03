"""Heading estimator SSM — linear velocity of the head in head-fixed frame.

Kalman-like fusion of two velocity estimates:
  - VESTIBULAR: leaky integral of a_est = (GIA − g_est) gives velocity from otolith.
  - VISUAL:    translational scene flow (already in head frame, with per-eye parallax
               folded in by the retina) gives velocity directly: v_visual = −scene_lin_vel.

The visual signal pulls v_lin toward the visually-derived velocity; in dark
(scene_lin_vel = 0) it pulls v_lin → 0, suppressing the slow drift that an
isolated vestibular integrator can build up from gravity-mismatch transients
during pure rotations.

State:  x_head = [v_lin (3,)]                                               (3,)

Dynamics:
    dv_lin/dt = a_est − v_lin / τ_head + K_he_vis · (v_visual − v_lin)

Parameters (in BrainParams):
    tau_head (s)    leaky integrator TC. Default 2 s.
    K_he_vis (1/s)  visual-velocity pull gain. 0 = vestibular only (legacy).
                    ~0.5–1.0 gives the visual estimate moderate authority.

Axis convention — head-fixed frame (LEFT-HANDED: x=right, y=up, z=forward):
    v_lin[0]: rightward velocity (m/s)
    v_lin[1]: upward velocity    (m/s)
    v_lin[2]: forward velocity   (m/s)
"""

import jax.numpy as jnp


N_STATES  = 3   # [v_lin (3,)]
N_INPUTS  = 10  # [a_lin (3,) | scene_lin_vel (3,) | scene_visible (1,) | scene_disp_rate (3,)]
N_OUTPUTS = 3   # [v_lin (3,)]

X0 = jnp.zeros(3)


def step(x_head, u, brain_params):
    """Single ODE step: vestibular integration + visual velocity fusion.

    Args:
        x_head:       (3,)  v_lin estimate (m/s, head frame)
        u:            (10,) [a_lin (3,) | scene_lin_vel (3,) | scene_visible (1,) |
                             scene_disp_rate (3,)]
                            a_lin:           linear acceleration estimate (m/s², head frame)
                                             from gravity_estimator — the brain's translation-
                                             attributed component of (gia − g_est).  Using a_lin
                                             instead of raw (gia − g_est) avoids integrating
                                             gravity-mismatch artifacts into v_lin (Laurens-style).
                            scene_lin_vel:   cyclopean translational scene flow (m/s, head frame)
                            scene_visible:   scalar in [0,1] — visual fusion gate
                            scene_disp_rate: per-eye scene-flow differential (m/s, head frame).
                                             0 in uniform/depthless scene → constrains v_lin[z]
                                             toward 0 (no z-motion evidence).
        brain_params: BrainParams  (reads tau_head, K_he_vis, K_he_disp)

    Returns:
        dx_head: (3,)  dv_lin/dt  (m/s²)
        v_lin:   (3,)  v_lin (m/s), passed through
    """
    v_lin           = x_head
    a_lin           = u[:3]
    scene_lin_vel   = u[3:6]
    scene_visible   = u[6]
    scene_disp_rate = u[7:10]

    # Vestibular: use the gravity estimator's a_lin directly (the brain's translation-
    # attributed component), not raw (gia − g_est).  This is the Laurens-Angelaki
    # decomposition: g_est captures slow gravity, a_lin captures transient acceleration.
    # HE then integrates only the translation part, not the gravity-mismatch artifact.
    a_est = a_lin

    # Visual: scene flow is opposite to head motion in the head frame.  Gated by
    # scene visibility — in dark (scene_visible = 0) the visual pull turns off
    # entirely so v_lin is governed by vestibular integration with the long τ_head
    # leak only (so DARK T-VOR can build up during sustained translation).
    v_visual = -scene_lin_vel
    K_vis    = brain_params.K_he_vis * scene_visible

    # Disparity-rate visual evidence for v_lin[z] (heading-z cue).
    # scene_disp_rate is the per-eye scene-flow differential: 0 in a uniform/depthless
    # scene → "no depth-rate change" → constrains v_lin[z] toward 0.  In a real depth-
    # structured scene with z-motion, the differential would be non-zero and would
    # augment the vestibular v_lin[z] estimate via parallax.
    # We pull v_lin[z] toward the disparity-rate-implied estimate (here 0 in our model
    # because scene_disp_rate is 0 with no depth), gated by scene_visible.
    K_disp     = brain_params.K_he_disp * scene_visible
    visual_z   = -scene_disp_rate[0]   # in m/s; for our depthless scene this is 0
    z_damping  = jnp.array([0.0, 0.0, K_disp * (visual_z - v_lin[2])])

    # Leaky integration of vestibular accel + (gated) visual velocity pull + disp-rate damping.
    dx = a_est - v_lin / brain_params.tau_head + K_vis * (v_visual - v_lin) + z_damping

    return dx, v_lin
