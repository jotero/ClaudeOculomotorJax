"""Accommodation plant — lens and ciliary muscle biomechanics.

First-order low-pass filter mapping the neural accommodation command (D)
to actual optical power (D).  The lens + ciliary muscle act as a viscoelastic
element: Beers & van der Heijde (1994) show the transfer function is
well-approximated by a first-order system with τ_plant ≈ 0.156 s for
young eyes (Schor & Bharadwaj 2006; Read et al. 2022 J Vision 22(9):4).

Dynamics:
    dx_plant = (u_neural − x_plant) / τ_plant

State:   x_plant  (1,)  actual lens accommodation (D)
Input:   u_neural  scalar  total neural command (D) from accommodation controller
Output:  x_plant  scalar  current lens optical power (D) → fed back to neural controller
                           → used for AC/A vergence drive

Parameters:
    tau_acc_plant (s)  lens / ciliary muscle TC; ~0.156 s young adults [Schor & Bharadwaj 2006]

References:
    Beers AP, van der Heijde GL (1994) Optom Vis Sci 71:587–589
    Schor CM, Bharadwaj SR (2006) J Neurophysiol 95:3459–3474
    Read et al. (2022) J Vision 22(9):4 — Eq. 9 and Figure 10
"""

import jax.numpy as jnp

N_STATES  = 1   # [x_plant (D)] — actual lens accommodation
N_INPUTS  = 1   # u_neural (D)  — neural command from accommodation controller
N_OUTPUTS = 1   # x_plant (D)   — current optical power


def step(x_plant, u_neural, tau_acc_plant):
    """Single ODE step for the accommodation plant.

    Args:
        x_plant:       (1,)   current lens accommodation (D)
        u_neural:      scalar  total neural command (x_fast + x_slow + tonic_acc) (D)
        tau_acc_plant: scalar  lens / ciliary muscle TC (s)

    Returns:
        dx_plant: (1,)   state derivative (D/s)
        x_plant_out: scalar  current accommodation (D) — for AC/A and blur feedback
    """
    dx = (u_neural - x_plant[0]) / tau_acc_plant
    return jnp.array([dx]), x_plant[0]
