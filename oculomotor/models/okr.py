"""OKR SSM — optokinetic reflex drive.

Converts delayed retinal slip (from visual_delay module) into a VS drive
signal via a slow store (first-order low-pass on the delayed slip).

Signal flow
───────────
    e_delayed (3,)   — delayed retinal slip, output of visual_delay cascade

    Two parallel pathways:

    Direct (fast):
        u_direct = g_okr · e_delayed          responds immediately to slip

    Store (slow — charges from slip, holds drive when slip → 0):
        dx_okr   = (-x_okr + e_delayed) / τ_okan
        u_store  = g_okr · x_okr

    Combined:
        u_okr = u_direct + u_store = g_okr · (e_delayed + x_okr)

    u_okr enters velocity storage as −u_okr (sign convention in simulator):
        u_vs = u_canal − u_okr

Build-up and OKAN
─────────────────
    OKN onset: direct term drives VS immediately; store charges slowly.
    Sustained OKN: store maintains drive even as retinal slip decreases.
    OKAN: when lights go out (e_delayed → 0), x_okr decays with τ_okan,
          continuing to drive nystagmus for ~τ_okan seconds.

Parameters
──────────
    g_okr   — OKR gain (unitless).         Default: 0.7.
    τ_okan  — store time constant (s).     Default: 25.0 s.
"""

import jax.numpy as jnp

N_STATES = 3   # one slow-store state per spatial axis


def get_A(theta):
    """(3 × 3) store decay matrix: −I / τ_okan."""
    tau = theta.get('tau_okan', 25.0)
    return -jnp.eye(3) / tau


def get_B(theta):
    """(3 × 3) input matrix: e_delayed drives store at rate 1/τ_okan."""
    tau = theta.get('tau_okan', 25.0)
    return jnp.eye(3) / tau


def get_C(theta):
    """(3 × 3) output matrix: g_okr scales stored drive."""
    return theta.get('g_okr', 0.0) * jnp.eye(3)


def get_D(theta):
    """(3 × 3) direct feedthrough: g_okr scales instantaneous delayed slip."""
    return theta.get('g_okr', 0.0) * jnp.eye(3)
