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
N_INPUTS  = 10  # [g_est (3,) | gia (3,) | scene_lin_vel (3,) | scene_visible (1,)]
N_OUTPUTS = 3   # [v_lin (3,)]

X0 = jnp.zeros(3)


def step(x_head, u, brain_params):
    """Single ODE step: vestibular integration + visual velocity fusion.

    Args:
        x_head:       (3,)  v_lin estimate (m/s, head frame)
        u:            (10,) [g_est (3,) | gia (3,) | scene_lin_vel (3,) | scene_visible (1,)]
                            g_est:         gravity estimate (m/s², head frame)
                            gia:           gravitoinertial accel (m/s², head frame)
                            scene_lin_vel: cyclopean translational scene flow (m/s, head frame)
                            scene_visible: scalar in [0,1] — visual fusion gate
        brain_params: BrainParams  (reads tau_head, K_he_vis)

    Returns:
        dx_head: (3,)  dv_lin/dt  (m/s²)
        v_lin:   (3,)  v_lin (m/s), passed through
    """
    v_lin         = x_head
    g_est         = u[:3]
    gia           = u[3:6]
    scene_lin_vel = u[6:9]
    scene_visible = u[9]

    # Vestibular: linear acceleration after gravity removal.
    a_est = gia - g_est

    # Visual: scene flow is opposite to head motion in the head frame.  Gated by
    # scene visibility — in dark (scene_visible = 0) the visual pull turns off
    # entirely so v_lin is governed by vestibular integration with the long τ_head
    # leak only (so DARK T-VOR can build up during sustained translation).
    v_visual = -scene_lin_vel
    K_vis    = brain_params.K_he_vis * scene_visible

    # Leaky integration of vestibular accel + (gated) visual velocity pull.
    dx = a_est - v_lin / brain_params.tau_head + K_vis * (v_visual - v_lin)

    return dx, v_lin
