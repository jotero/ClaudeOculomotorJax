"""Readout functions — convert plant state (rotation vector) to measurement coordinates.

The plant state is a rotation vector q ∈ ℝ³ (deg):
    q[0] = yaw   (rightward positive)
    q[1] = pitch (upward positive)
    q[2] = roll  (CW from subject's view positive)

All functions accept either a single vector q (3,) or a trajectory Q (T, 3)
via standard NumPy/JAX broadcasting.

Rotation vector → rotation matrix via Rodrigues' formula (numerically stable
at small angles via sinc = sin(θ)/θ).  For |q| < 60° (all physiological
movements) the approximation q ≈ small-angle Euler angles is < 2% error.

Measurement conventions
-----------------------
Fick angles       — standard clinical / video-oculography
                    H: rotate about fixed vertical (yaw first)
                    V: rotate about new horizontal (pitch second)
                    T: torsion about line of sight (last)
Helmholtz angles  — used in some search-coil systems
                    V: rotate about fixed horizontal (pitch first)
                    H: rotate about new vertical (yaw second)
Rotation vector   — identity (Tweed-Vilis 1987 analysis)
Listing deviation — torsional component; = 0 iff eye is in Listing's plane
"""

import jax.numpy as jnp


# ── Simple axis readouts ───────────────────────────────────────────────────────

def horizontal(q):
    """Horizontal (yaw) eye position, deg.  Works on (3,) or (T, 3)."""
    return q[..., 0]


def vertical(q):
    """Vertical (pitch) eye position, deg.  Works on (3,) or (T, 3)."""
    return q[..., 1]


def torsion(q):
    """Torsional (roll) eye position, deg.  Works on (3,) or (T, 3)."""
    return q[..., 2]


def rotation_vector(q):
    """Identity — returns q as-is (Tweed-Vilis representation)."""
    return q


# ── Rotation matrix via Rodrigues' formula ─────────────────────────────────────

def rotation_matrix(q_deg):
    """Rotation matrix R from a rotation vector q (deg) via Rodrigues' formula.

    Args:
        q_deg: (3,) rotation vector in degrees

    Returns:
        R: (3, 3) rotation matrix
    """
    q    = jnp.radians(q_deg)
    th   = jnp.linalg.norm(q)
    sinc = jnp.sinc(th / jnp.pi)              # sin(th)/th,       = 1   at th=0
    cosc = jnp.sinc(th / (2 * jnp.pi))**2 / 2 # (1-cos(th))/th², = 0.5 at th=0
    qx, qy, qz = q[0], q[1], q[2]
    skew = jnp.array([
        [  0., -qz,  qy],
        [ qz,   0., -qx],
        [-qy,  qx,   0.],
    ])
    return jnp.eye(3) + sinc * skew + cosc * (skew @ skew)


# ── Fick angles ────────────────────────────────────────────────────────────────

def fick_angles(q_deg):
    """Convert rotation vector (deg) to Fick angles (deg): [horizontal, vertical, torsion].

    Fick convention: R = R_z(H) @ R_y(V) @ R_x(T)
    Extracted from the rotation matrix columns.

    Args:
        q_deg: (3,) rotation vector in degrees

    Returns:
        [H, V, T] in degrees
    """
    R = rotation_matrix(q_deg)
    H = jnp.degrees(jnp.arctan2( R[1, 0],  R[0, 0]))
    V = jnp.degrees(jnp.arctan2(-R[2, 0],  jnp.sqrt(R[2, 1]**2 + R[2, 2]**2)))
    T = jnp.degrees(jnp.arctan2( R[2, 1],  R[2, 2]))
    return jnp.array([H, V, T])


# ── Helmholtz angles ───────────────────────────────────────────────────────────

def helmholtz_angles(q_deg):
    """Convert rotation vector (deg) to Helmholtz angles (deg): [horizontal, vertical, torsion].

    Helmholtz convention: R = R_y(H) @ R_x(V) @ R_z(T)

    Args:
        q_deg: (3,) rotation vector in degrees

    Returns:
        [H, V, T] in degrees
    """
    R = rotation_matrix(q_deg)
    V = jnp.degrees(jnp.arctan2(-R[2, 0],  jnp.sqrt(R[0, 0]**2 + R[1, 0]**2)))
    H = jnp.degrees(jnp.arctan2( R[2, 1],  R[2, 2]))   # approximate for small angles
    T = jnp.degrees(jnp.arctan2( R[0, 1], -R[1, 1]))
    return jnp.array([H, V, T])


# ── Listing's plane ────────────────────────────────────────────────────────────

def listing_deviation(q_deg):
    """Torsional deviation from Listing's plane (deg).

    Listing's law: for visually-guided fixations, the torsional component
    of the rotation vector is zero (eye stays in Listing's plane).
    VOR can drive eyes out of Listing's plane (torsional VOR).

    Returns:
        scalar, deg — positive = CW torsion beyond Listing's plane
    """
    return q_deg[..., 2]
