"""Accommodation SSM — Schor dual-interaction model + plant dynamics.

Read, Kaspiris-Rousellis, Wood, Wu, Vlaskamp & Schor (2022) J Vision 22(9):4.
Schor CM (1979) Vision Res 19:1359–1367; Schor CM & Ciuffreda KJ (1983) textbook.

Dual neural controller (fast + slow integrators) driven by a DELAYED DEFOCUS signal,
cross-coupled to vergence via AC/A and CA/C ratios.

Defocus signal (computed in ODE, delayed through cyclopean cascade):
    defocus = 1/z + refractive_error − x_plant   (diopters)
    Positive = near target is closer than current focus (need more accommodation).
    The delay (~80 ms) is set by tau_vis; accommodation TCs (2.5–30 s) are >> delay,
    so the closed-loop behaviour is essentially the same as zero delay.

    AC/A  (accommodative convergence / accommodation):
        When the eye accommodates to bring a near target into focus, vergence
        is reflexively driven inward.  Provides convergence even when binocular
        disparity is absent (e.g. one eye suppressed in IXT).

        ac_a_drive (deg) = AC_A (pd/D) × 0.5729 (deg/pd) × (x_plant − tonic_acc) (D)

        Using INCREMENT above dark focus (x_plant − tonic_acc) so that at dark
        focus the AC/A drive is zero and x_verg settles to tonic_verg.

    CA/C  (convergence accommodation / accommodation):
        When vergence is driven by disparity, the lens is co-driven to maintain
        focus at the new depth.  Reduces blur during vergence movements.

        The CA/C drives the PLANT directly (bypassing the blur controller):

            A_cac (D) = CA_C (D/pd) × (Δvergence / 0.5729)
            Δvergence = x_verg_yaw − tonic_verg  (delta from dark vergence)

        A_cac is returned so the ODE can add it to u_plant: u_plant = u_neural + A_cac.
        The blur controller sees the raw defocus (e_blur = defocus, NOT defocus − A_cac):
        x_plant already includes A_cac (via u_plant), so defocus naturally reflects the
        residual blur after the CA/C contribution.  Subtracting A_cac from e_blur would
        double-count the CA/C and flip its sign (working backwards with gain G ≈ 13).

        At dark vergence (x_verg_yaw = tonic_verg): A_cac = 0 → controller
        holds at tonic_acc unperturbed.

Architecture (Read & Schor 2022, Fig. 10):
    Blur error uses the PLANT output via the defocus signal, giving the
    second-order closed-loop response:
        ζ ≈ 1/√2  when  τ_fast ≈ 2·K_fast·τ_plant

    e_blur  = defocus              (raw blur; CA/C bypasses blur controller via u_plant)
    dx_fast = −x_fast / τ_fast + K_fast · e_blur
    dx_slow = −x_slow / τ_slow + K_slow · e_blur
    u_neural = x_fast + x_slow + tonic_acc          (→ accommodation_plant.step)
    u_plant  = u_neural + A_cac                     (done in ODE / brain_model)

    Plant dynamics live in plant_models/accommodation_plant.py:
    dx_plant = (u_plant − x_plant) / τ_plant

Tonic baseline:
    tonic_acc (D) is the dark-focus resting level (~1 D ≈ 1 m for young adults).
    x_fast and x_slow represent DEVIATIONS from this baseline; both are zero
    at rest.  The plant settles at tonic_acc when defocus = 0 and A_cac = 0
    (i.e., at dark focus with dark vergence).

State:
    x_acc = [x_fast (D), x_slow (D)]   (2 neural states; plant state in ODE/SimState)

Input:
    defocus     (D)  = delay(1/z + refractive_error − x_plant)
    x_verg_yaw  (deg)                   — current vergence yaw state (CA/C feedback)

Output:
    u_neural (D)  — total neural command = x_fast + x_slow + tonic_acc → accommodation plant
    A_cac    (D)  — CA/C feedforward drive; added to u_neural in ODE before plant

Unit note:
    1 prism diopter (pd) = arctan(0.01) rad ≈ 0.5729 deg.
    AC/A ratio is conventionally in pd/D; multiply by 0.5729 to get deg/D.
    CA/C ratio is in D/pd; divide vergence by 0.5729 to convert deg → pd.

Parameters (BrainParams fields):
    tonic_acc      (D)    dark-focus resting level;    default 1.0 D
    tonic_verg     (deg)  dark vergence baseline;       default ~3.67°
    tau_acc_plant  (s)    lens/ciliary muscle TC;       default 0.156 s  [Schor & Bharadwaj 2006]
    tau_acc_fast   (s)    fast neural integrator TC;    default 2.5 s    [Read & Schor 2022]
    tau_acc_slow   (s)    slow adaptation TC;           default 30 s
    K_acc_fast     (1/s)  fast blur gain
    K_acc_slow     (1/s)  slow integration gain
    AC_A           (pd/D) AC/A ratio;                   default 5.0
    CA_C           (D/pd) CA/C ratio;                   default 0.4

References:
    Read et al. (2022) J Vision 22(9):4   — Smith predictor, plant TC, τ_fast constraint
    Schor CM (1979) Vision Res 19:1359–1367
    Schor CM, Bharadwaj SR (2006) J Neurophysiol 95:3459–3474  — plant TC
    Hung GK, Semmlow JL (1980) IEEE Trans Biomed Eng 27:722–728
    Semmlow JL, Hung GK, Ciuffreda KJ (1986) Invest Ophthalmol Vis Sci 27:558
"""

import jax.numpy as jnp

N_STATES  = 2   # [x_fast (D), x_slow (D)] — neural integrators only; plant state in ODE/SimState
N_INPUTS  = 2   # defocus (D), x_verg_yaw (deg)
N_OUTPUTS = 2   # u_neural (D), A_cac (D)

# Unit glossary for this module:
#   D   = optical diopters (1/metres)  — used for defocus and accommodation
#   pd  = prism diopters (arctan(0.01) rad ≈ 0.5729 deg)  — used for vergence magnitude
#   deg = degrees                       — eye rotation / vergence angle
#
# AC/A ratio is in pd/D  →  × _DEG_PER_PD [deg/pd] × accommodation [D]  = deg
# CA/C ratio is in D/pd  →  vergence [deg] / _DEG_PER_PD [deg/pd] = pd  → × CA_C [D/pd] = D
#
# IMPORTANT: pd ≠ D.  One prism diopter (pd) is a vergence unit; one optical diopter (D)
# is a focus/power unit.  The cross-coupling ratios AC/A and CA/C bridge the two systems.
_DEG_PER_PD = 0.5729   # deg per prism diopter; 1 pd = arctan(0.01) rad ≈ 0.5729 deg


def step(x_acc, defocus, x_verg_yaw, brain_params):
    """Single ODE step for the dual neural accommodation controller.

    Blur error is derived from the pre-delayed defocus signal (= acc_demand + RE − x_plant),
    so x_plant is already accounted for — no separate x_plant argument needed.

    CA/C drives the plant directly (bypasses the blur controller): A_cac is returned
    so the ODE/brain_model can compute u_plant = u_neural + A_cac.

    CA/C uses DELTA vergence (deviation from dark vergence = tonic_verg) so that at
    rest the CA/C contribution is zero and the controller holds at tonic_acc.

    The defocus input is gated by defocus_visible (scene OR target visible) inside
    cyclopean_vision; when nothing is visible, defocus = 0 and integrators hold.

    Args:
        x_acc:        (2,)   [x_fast, x_slow] neural integrator states (D)
        defocus:      scalar  delayed cyclopean defocus (D) from sensory_out.defocus
                              = delay(acc_demand + refractive_error − x_plant)
                              Positive = need more accommodation.
        x_verg_yaw:   scalar  vergence yaw state (deg) → CA/C feedback
        brain_params: BrainParams  (reads tonic_acc, tonic_verg, tau_acc_fast, tau_acc_slow,
                                          K_acc_fast, K_acc_slow, CA_C)

    Returns:
        dx_acc:   (2,)   state derivative (D/s)
        u_neural: scalar  total neural command = x_fast + x_slow + tonic_acc (D)
        A_cac:    scalar  CA/C feedforward drive (D); add to u_neural before plant
    """
    x_fast, x_slow = x_acc[0], x_acc[1]

    # CA/C: DELTA vergence from dark vergence drives accommodation feedforward.
    # At rest (x_verg_yaw = tonic_verg): A_cac = 0 → controller holds at tonic_acc.
    # When converging to near (x_verg_yaw > tonic_verg): A_cac > 0 → plant driven
    # toward greater accommodation; neural controller sees reduced blur error.
    delta_verg = x_verg_yaw - brain_params.tonic_verg   # (deg)
    A_cac = brain_params.CA_C * (delta_verg / _DEG_PER_PD)   # D

    # Blur error: raw optical defocus.  CA/C bypasses the blur controller by adding
    # A_cac directly to u_plant in the ODE; x_plant already includes that contribution,
    # so defocus naturally reflects the residual blur.  Do NOT subtract A_cac here —
    # that would flip the sign of CA/C (with gain G≈13, the neural reduction −G·A_cac
    # dominates over the direct +A_cac to the plant, driving accommodation backwards).
    e_blur = defocus

    dx_fast = -x_fast / brain_params.tau_acc_fast + brain_params.K_acc_fast * e_blur
    dx_slow = -x_slow / brain_params.tau_acc_slow + brain_params.K_acc_slow * e_blur

    # Total neural command includes tonic bias (dark-focus baseline)
    u_neural = x_fast + x_slow + brain_params.tonic_acc

    return jnp.array([dx_fast, dx_slow]), u_neural, A_cac


def ac_a_drive(x_plant, brain_params):
    """Convert actual lens accommodation INCREMENT (D above dark focus) to vergence drive (deg).

    Uses (x_plant − tonic_acc) so that at dark focus the AC/A drive is zero
    and x_verg settles to tonic_verg with no extra bias.

    Args:
        x_plant:      scalar  actual lens accommodation — plant output (D)
        brain_params: BrainParams  (reads AC_A, tonic_acc)

    Returns:
        drive: scalar  vergence drive (deg); positive = converging
    """
    return brain_params.AC_A * _DEG_PER_PD * (x_plant - brain_params.tonic_acc)
