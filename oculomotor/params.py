"""Model and simulation parameters — structured pytree-compatible containers.

Three levels of parameters:

    SimConfig   — simulation settings (dt_solve, etc.); not model biology,
                  not JAX leaves, not learnable.

    PhysParams  — physical / sensor parameters: canal mechanics, plant TC,
                  visual delay.  Constrained by physiology; fixed during
                  typical patient fitting but could be freed for data with
                  known peripheral pathology.

    BrainParams — learnable central parameters: VS, NI, SG, orbital limits.
                  These are the primary targets for parameter estimation
                  from eye-movement recordings.

    Params      — top-level container: Params(phys, brain).
                  A JAX pytree (NamedTuple) — works transparently with
                  jax.grad, jax.jit, optax, and diffrax.

Typical fitting loop:
    params    = default_params()
    opt       = optax.adam(1e-3)
    opt_state = opt.init(params.brain)          # only brain leaves
    grads     = jax.grad(loss)(params.brain)
    updates, opt_state = opt.update(grads, opt_state)
    params    = params._replace(brain=optax.apply_updates(params.brain, updates))

Convenience updaters (replace the old dict-spread idiom):
    sac_params  = with_brain(default_params(), g_burst=600.0)
    dark_params = with_brain(sac_params, g_burst=0.0)
"""

from typing import NamedTuple

import jax.numpy as jnp


# ── 1. Simulation config (not biology, not a pytree leaf) ──────────────────────

class SimConfig(NamedTuple):
    """Solver / run settings — not model parameters, not learnable.

    Fields:
        dt_solve: Heun fixed step (s).  Must satisfy dt < 2 * tau_stage_vis.
                  With N_STAGES=40 and tau_vis=0.08 s → tau_stage = 0.002 s
                  → dt_max = 0.004 s.  Default 0.001 s gives 4× safety margin.
    """
    dt_solve: float = 0.001


# ── 2. Physical parameters (sensor + plant) ────────────────────────────────────

class PhysParams(NamedTuple):
    """Physical / sensor parameters — typically fixed during fitting.

    Vary with peripheral pathology (canal paresis, plant surgery) but not
    with central adaptation.
    """
    
    # Semicircular canals — Steinhausen torsion-pendulum (Fernandez & Goldberg 1971)
    tau_c:       float       = 5.0    # cupula adaptation TC (s); HP corner ≈ 0.03 Hz
    tau_s:       float       = 0.005  # endolymph inertia TC (s); LP corner ≈ 32 Hz
    canal_gains: jnp.ndarray = None   # (6,) per-canal scale; 1=intact, 0=paresis; set via default_params()
    
    # Ocular plant — Robinson (1964)
    tau_p:       float       = 0.15   # plant TC (s); Robinson 1981, Goldstein 1983

    # Visual pathway
    tau_vis:     float       = 0.08   # gamma-cascade mean delay (s); Lisberger & Movshon 1999


# ── 3. Learnable brain parameters ──────────────────────────────────────────────

class BrainParams(NamedTuple):
    """Learnable central parameters — fit to patient eye-movement data."""

    # Velocity storage — Raphan, Matsuo & Cohen (1979)
    tau_vs:                float = 20.0   # storage / OKAN TC (s); ~20 s monkey (Cohen 1977)
    K_vs:                  float = 0.1    # canal-to-VS gain (1/s); controls charging speed
    K_vis:                 float = 0.3    # visual-to-VS gain (1/s); OKR / OKAN charging
    g_vis:                 float = 0.3    # visual feedthrough (unitless); fast OKR onset

    # Neural integrator — Robinson (1975)
    tau_i:                 float = 25.0   # leak TC (s); healthy >20 s, ~25 s rhesus (Cannon & Robinson 1985)

    # Saccade generator — Robinson (1975) local-feedback burst model
    g_burst:               float = 700.0  # burst ceiling (deg/s); 0 disables saccades
    e_sat_sac:             float = 7.0    # main-sequence saturation (deg)
    k_sac:                 float = 200.0  # trigger sigmoid steepness (1/deg)
    threshold_sac:         float = 0.5    # retinal error trigger threshold (deg)
    threshold_stop:        float = 0.1    # burst-stop threshold (deg)
    threshold_sac_release: float = 0.4    # OPN latch release threshold
    tau_reset_fast:        float = 0.05   # inter-saccade x_copy reset TC (s)
    tau_ref:               float = 0.15   # refractory (OPN) decay TC (s); ~150 ms ISI
    tau_ref_charge:        float = 0.001  # OPN charge TC (s)
    k_ref:                 float = 50.0   # bistable OPN gate steepness (1/z_ref)
    threshold_ref:         float = 0.1    # OPN threshold
    tau_hold:              float = 0.005  # sample-and-hold tracking TC (s)
    tau_sac:               float = 0.001  # saccade latch TC (s)
    tau_acc:               float = 0.080  # accumulator rise TC (s)
    tau_drain:             float = 0.120  # accumulator drain TC (s)
    threshold_acc:         float = 0.5    # accumulator trigger threshold
    k_acc:                 float = 50.0   # accumulator sigmoid steepness

    # Orbital limits & target selector
    orbital_limit:         float = 50.0   # mechanical half-range (deg); plant soft-limit + orbital gate
    k_orbital:             float = 1.0    # orbital gate sigmoid steepness (1/deg)
    alpha_reset:           float = 1.0    # centering-saccade gain at orbital limit
    visual_field_limit:    float = 90.0   # eccentricity beyond which target is invisible (deg)


# ── 4. Top-level container ─────────────────────────────────────────────────────

class Params(NamedTuple):
    """Top-level parameter container — a JAX pytree.

    jax.tree_util.tree_leaves(params) returns all float / array fields from
    both phys and brain, in field order.  optax.apply_updates works directly
    on params.brain for partial-parameter optimisation.
    """
    phys:  PhysParams
    brain: BrainParams


# ── Factory (canal_gains can't be a NamedTuple default — it's a jnp array) ────

def default_params() -> Params:
    """Healthy primate default parameters."""
    return Params(
        phys  = PhysParams(canal_gains=jnp.ones(6)),
        brain = BrainParams(),
    )


SIM_CONFIG_DEFAULT = SimConfig()
PARAMS_DEFAULT     = default_params()


# ── Convenience updaters ───────────────────────────────────────────────────────

def with_brain(params: Params, **kwargs) -> Params:
    """Return a new Params with brain fields updated.

    Replaces the old dict-spread idiom:
        OLD: {**THETA_DEFAULT, 'g_burst': 600.0}
        NEW: with_brain(default_params(), g_burst=600.0)

    Example:
        sac_params  = with_brain(default_params(), g_burst=600.0)
        dark_params = with_brain(sac_params, g_burst=0.0)
    """
    return params._replace(brain=params.brain._replace(**kwargs))


def with_phys(params: Params, **kwargs) -> Params:
    """Return a new Params with phys fields updated.

    Example:
        lesion = with_phys(default_params(), canal_gains=jnp.array([0,0,1,1,1,1.]))
    """
    return params._replace(phys=params.phys._replace(**kwargs))
