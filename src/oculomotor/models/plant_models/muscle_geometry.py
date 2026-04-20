"""Extraocular muscle geometry and motor nucleus connectivity.

Two-stage encode from brain version/vergence command to per-muscle nerve activations:

    Stage 1 — Motor nuclei  (M_NUCLEUS, 12×6):
        [version (3,), vergence (3,)] → 12 nucleus activations
        Nuclei: ABN_L/R, CN4_L/R, CN3_MR_L/R, CN3_SR_L/R, CN3_IR_L/R, CN3_IO_L/R

    Stage 2 — Nucleus-to-nerve projections  (M_NERVE_PROJ, 12×12):
        12 nucleus activations → 12 nerve outputs [6 left-eye | 6 right-eye]
        Captures bilateral anatomy: ABN → ipsilateral LR + contralateral MR via MLF;
        CN4 → contralateral SO; all CN3 subdivisions → ipsilateral muscle.

    Decode (plant, per eye)  — M_PLANT_EYE_{L,R}  (3×6):
        6 nerve activations → 3-D effective motor command
        M_PLANT_EYE = pinv(M_NERVE) so M_PLANT_EYE @ M_NERVE = I₃.

Healthy (all gains = 1) round-trip is transparent:
    M_PLANT_EYE_L @ (M_NERVE_PROJ @ M_NUCLEUS)[:6, :] @ [version, vergence]
    = version + 0.5 * vergence  (left  eye motor command)
    M_PLANT_EYE_R @ (M_NERVE_PROJ @ M_NUCLEUS)[6:, :] @ [version, vergence]
    = version - 0.5 * vergence  (right eye motor command)

─────────────────────────────────────────────────────────────────────────────
Lesion guide
─────────────────────────────────────────────────────────────────────────────
CN VI nerve palsy (R):   g_nerve[LR_R+6] = 0    (LR only; MR_L via MLF intact)
CN VI nucleus palsy (R): g_nucleus[ABN_R] = 0   (LR_R + conjugate MR_L both affected)
CN III nerve palsy (R):  g_nerve[[MR_R,SR_R,IR_R,IO_R]+6] = 0  (LR_R + SO_R intact)
CN IV nerve palsy (R):   g_nerve[SO_R+6] = 0
INO (right MLF):         requires zeroing M_NERVE_PROJ[MR_L, ABN_R] directly
                         (g_nucleus/g_nerve cannot isolate a single inter-nuclear projection)

─────────────────────────────────────────────────────────────────────────────
Muscle / nucleus index constants
─────────────────────────────────────────────────────────────────────────────
Per-eye muscle indices (0–5):    LR, MR, SR, IR, SO, IO
Nerve output indices (0–11):     L eye 0–5, R eye 6–11  (same muscle order)
Nucleus indices (0–11):          ABN_L/R, CN4_L/R, CN3_MR/SR/IR/IO L/R
"""

import numpy as np
import jax.numpy as jnp


# ── Orbital angles ─────────────────────────────────────────────────────────────

_ALPHA_VR  = np.radians(23.0)   # vertical recti: nasal tilt from sagittal plane
_ALPHA_OBL = np.radians(54.0)   # obliques: angle from sagittal after trochlea/pulley


# ── Per-eye muscle geometry  M_NERVE_{L,R}  (6 × 3) ──────────────────────────
# Row order: LR(0), MR(1), SR(2), IR(3), SO(4), IO(5)
# Columns: [yaw, pitch, roll]

_M_NERVE_R_np = np.array([
    [+1.0,                   0.0,                   0.0             ],  # LR: abduction
    [-1.0,                   0.0,                   0.0             ],  # MR: adduction
    [ 0.0, +np.cos(_ALPHA_VR),  -np.sin(_ALPHA_VR) ],  # SR: elevation + intorsion
    [ 0.0, -np.cos(_ALPHA_VR),  +np.sin(_ALPHA_VR) ],  # IR: depression + extorsion
    [ 0.0, -np.sin(_ALPHA_OBL), -np.cos(_ALPHA_OBL)],  # SO: depression + intorsion (CN IV)
    [ 0.0, +np.sin(_ALPHA_OBL), +np.cos(_ALPHA_OBL)],  # IO: elevation + extorsion
], dtype=np.float32)

# Left eye: mirror yaw (col 0) and roll (col 2)
_M_NERVE_L_np = _M_NERVE_R_np * np.array([-1.0, +1.0, -1.0], dtype=np.float32)

# Per-eye decode: M_PLANT_EYE = pinv(M_NERVE)  →  M_PLANT_EYE @ M_NERVE = I₃
_M_PLANT_EYE_R_np = np.linalg.pinv(_M_NERVE_R_np).astype(np.float32)  # (3, 6)
_M_PLANT_EYE_L_np = np.linalg.pinv(_M_NERVE_L_np).astype(np.float32)  # (3, 6)


# ── Muscle / nerve output index constants ─────────────────────────────────────

# Per-eye muscle indices (used by both M_NERVE and combined nerve-output array)
LR = 0   # Lateral  Rectus  (CN VI)
MR = 1   # Medial   Rectus  (CN III)
SR = 2   # Superior Rectus  (CN III)
IR = 3   # Inferior Rectus  (CN III)
SO = 4   # Superior Oblique (CN IV)
IO = 5   # Inferior Oblique (CN III)

# Combined 12-D nerve output row indices  [L eye 0–5 | R eye 6–11]
LR_L, MR_L, SR_L, IR_L, SO_L, IO_L = 0, 1, 2, 3, 4, 5
LR_R, MR_R, SR_R, IR_R, SO_R, IO_R = 6, 7, 8, 9, 10, 11


# ── Motor nucleus index constants (0–11) ──────────────────────────────────────

ABN_L, ABN_R       =  0,  1   # Abducens nucleus  (CN VI)  — horizontal gaze push-pull
CN4_L, CN4_R       =  2,  3   # Trochlear nucleus (CN IV)  — SO, contralateral projection
CN3_MR_L, CN3_MR_R =  4,  5   # CN III — medial rectus subnucleus
CN3_SR_L, CN3_SR_R =  6,  7   # CN III — superior rectus subnucleus
CN3_IR_L, CN3_IR_R =  8,  9   # CN III — inferior rectus subnucleus
CN3_IO_L, CN3_IO_R = 10, 11   # CN III — inferior oblique subnucleus

N_NUCLEI = 12
N_NERVES = 12   # = 6 left-eye + 6 right-eye


# ── Stage 2 — M_NERVE_PROJ  (12 × 12)  nucleus → nerve ───────────────────────
# Bilateral anatomy:
#   ABN_L → LR_L  (direct, CN VI nerve)  +  MR_R  (via MLF to right CN3_MR)
#   ABN_R → LR_R  (direct, CN VI nerve)  +  MR_L  (via MLF to left  CN3_MR)
#   CN4_L → SO_R  (CN IV nerve decussates in dorsal midbrain)
#   CN4_R → SO_L
#   CN3 subdivisions → ipsilateral muscle (CN III nerve, uncrossed)

_M_NERVE_PROJ_np = np.zeros((N_NERVES, N_NUCLEI), dtype=np.float32)

# ABN_L: ipsilateral LR_L + contralateral MR_R via MLF
_M_NERVE_PROJ_np[LR_L, ABN_L] = 1.0
_M_NERVE_PROJ_np[MR_R, ABN_L] = 1.0

# ABN_R: ipsilateral LR_R + contralateral MR_L via MLF
_M_NERVE_PROJ_np[LR_R, ABN_R] = 1.0
_M_NERVE_PROJ_np[MR_L, ABN_R] = 1.0

# CN4_L → SO_R (contralateral);  CN4_R → SO_L (contralateral)
_M_NERVE_PROJ_np[SO_R, CN4_L] = 1.0
_M_NERVE_PROJ_np[SO_L, CN4_R] = 1.0

# CN3 subdivisions → ipsilateral muscles (vergence drive + MLF target)
_M_NERVE_PROJ_np[MR_L, CN3_MR_L] = 1.0
_M_NERVE_PROJ_np[MR_R, CN3_MR_R] = 1.0
_M_NERVE_PROJ_np[SR_L, CN3_SR_L] = 1.0
_M_NERVE_PROJ_np[SR_R, CN3_SR_R] = 1.0
_M_NERVE_PROJ_np[IR_L, CN3_IR_L] = 1.0
_M_NERVE_PROJ_np[IR_R, CN3_IR_R] = 1.0
_M_NERVE_PROJ_np[IO_L, CN3_IO_L] = 1.0
_M_NERVE_PROJ_np[IO_R, CN3_IO_R] = 1.0


# ── Stage 1 — M_NUCLEUS  (12 × 6)  [version, vergence] → nuclei ──────────────
# Derived so that the healthy round-trip reproduces the original split:
#   left  nerve = M_NERVE_L @ (version + 0.5 * vergence)
#   right nerve = M_NERVE_R @ (version − 0.5 * vergence)
#
# M_NERVE_PROJ @ M_NUCLEUS = M_FULL
#   where M_FULL (12×6) = [[M_NERVE_L | 0.5*M_NERVE_L],
#                           [M_NERVE_R | −0.5*M_NERVE_R]]
#
# Since M_NERVE_PROJ is square and full-rank, M_NUCLEUS = inv(M_NERVE_PROJ) @ M_FULL.
# Key results (verified analytically):
#   ABN_R:      [+1, 0, 0, −0.5, 0, 0]  — version yaw + vergence inhibits LR
#   ABN_L:      [−1, 0, 0, −0.5, 0, 0]
#   CN3_MR_L/R: [ 0, 0, 0, +1,   0, 0]  — pure convergence drive, no version
#   CN4_L/R:    version pitch/roll + vergence pitch/roll (vertical gaze)
#   CN3_SR/IR/IO: version pitch/roll ± vergence pitch/roll

_M_FULL_np = np.vstack([
    np.hstack([_M_NERVE_L_np,  0.5 * _M_NERVE_L_np]),   # left  nerves: version | vergence
    np.hstack([_M_NERVE_R_np, -0.5 * _M_NERVE_R_np]),   # right nerves: version | vergence
]).astype(np.float32)   # (12, 6)

_M_NUCLEUS_np = np.linalg.solve(_M_NERVE_PROJ_np, _M_FULL_np).astype(np.float32)  # (12, 6)


# ── JAX arrays (immutable; safe inside jit) ────────────────────────────────────

M_NERVE_R     = jnp.array(_M_NERVE_R_np)        # (6, 3)  right-eye muscle geometry
M_NERVE_L     = jnp.array(_M_NERVE_L_np)        # (6, 3)  left-eye  muscle geometry
M_PLANT_EYE_R = jnp.array(_M_PLANT_EYE_R_np)   # (3, 6)  right-eye decode (plant)
M_PLANT_EYE_L = jnp.array(_M_PLANT_EYE_L_np)   # (3, 6)  left-eye  decode (plant)

M_NUCLEUS    = jnp.array(_M_NUCLEUS_np)          # (12, 6) brain → nuclei
M_NERVE_PROJ = jnp.array(_M_NERVE_PROJ_np)       # (12,12) nuclei → nerves (bilateral anatomy)

G_NUCLEUS_DEFAULT = jnp.ones(N_NUCLEI, dtype=jnp.float32)  # healthy: all = 1
G_NERVE_DEFAULT   = jnp.ones(N_NERVES, dtype=jnp.float32)  # healthy: all = 1

# Backward-compat alias (was used in previous g_muscle_L/R architecture)
G_MUSCLE_DEFAULT  = jnp.ones(6, dtype=jnp.float32)
