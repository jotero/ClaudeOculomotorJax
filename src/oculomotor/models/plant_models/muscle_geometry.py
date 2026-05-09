"""Extraocular muscle geometry and motor nucleus connectivity.

Two-stage encode from brain version/vergence command to per-muscle nerve activations:

    Stage 1 — Motor nuclei  (M_NUCLEUS, 14×6):
        [version (3,), vergence (3,)] → 14 nucleus activations
        Nuclei: ABN_L/R, CN4_L/R, CN3_MR_L/R, CN3_SR_L/R, CN3_IR_L/R, CN3_IO_L/R,
                AIN_L/R  (abducens internuclear neurons; pure version, project via MLF)

        Encoding:
            ABN_L/R       — version + vergence drive to ipsilateral LR motoneurons
                             (matches real LR firing pattern: more during divergence)
            AIN_L/R       — pure version drive (encodes saccade burst only); axons
                             cross midline and ascend in the contralateral MLF
            CN3_MR_L/R    — vergence-only drive to ipsilateral MR motoneurons
                             (MR vergence command from supraoculomotor area)
            Other CN3, CN4 — version + vergence components for the verticals/obliques

    Stage 2 — Nucleus-to-nerve projections  (M_NERVE_PROJ, 12×14):
        14 nucleus activations → 12 nerve outputs [6 left-eye | 6 right-eye]
        Anatomy modelled:
            ABN_L → LR_L                       (ipsilateral CN VI motoneurons)
            ABN_R → LR_R
            AIN_L → MR_R   weighted g_mlf_R    (right MLF — crosses to contralateral)
            AIN_R → MR_L   weighted g_mlf_L    (left  MLF)
            CN4_L → SO_R, CN4_R → SO_L         (CN IV decussates dorsally)
            CN3_*  → ipsilateral muscle        (vergence drive direct, no decussation)
        MR motoneurons are now an explicit summation of vergence (CN3_MR direct)
        and version (AIN via MLF), making INO a clean lesion of g_mlf_L or g_mlf_R.

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
CN VI nerve palsy (R):   g_nerve[LR_R] = 0         (LR_R only; MR_L unaffected)
CN VI nucleus palsy (R): g_nucleus[ABN_R] = 0      (LR_R paralysed; MR_L unaffected — only
                                                     ABN motoneurons silenced; AIN_R intact
                                                     so left MR still adducts. Set
                                                     g_nucleus[AIN_R]=0 too for full
                                                     abducens nucleus lesion.)
CN III nerve palsy (R):  g_nerve[[MR_R,SR_R,IR_R,IO_R]+6] = 0  (LR_R + SO_R intact)
CN IV nerve palsy (R):   g_nerve[SO_R+6] = 0
INO (right MLF):         g_mlf_R = 0  (right MLF carries AIN_L → MR_R; lesion blocks
                                        right MR adduction on leftward gaze.
                                        Convergence preserved via CN3_MR_R direct.)
INO (left MLF):          g_mlf_L = 0  (left MLF carries AIN_R → MR_L; lesion blocks
                                        left MR adduction on rightward gaze.)

─────────────────────────────────────────────────────────────────────────────
Muscle / nucleus index constants
─────────────────────────────────────────────────────────────────────────────
Per-eye muscle indices (0–5):    LR, MR, SR, IR, SO, IO
Nerve output indices (0–11):     L eye 0–5, R eye 6–11  (same muscle order)
Nucleus indices (0–13):          ABN_L/R, CN4_L/R, CN3_MR/SR/IR/IO L/R, AIN_L/R
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
#
# For reference: with symmetric 45°/45° angles (Q=0, pitch-roll decouple):
#
#   M_PLANT_EYE_R  (row=axis, col=muscle: LR    MR    SR     IR     SO     IO)
#     yaw:        [+1/2, -1/2,   0,     0,     0,     0   ]
#     pitch:      [  0,    0,  +√2/4, -√2/4, -√2/4, +√2/4]
#     roll:       [  0,    0,  -√2/4, +√2/4, -√2/4, +√2/4]
#
#   M_PLANT_EYE_L  (yaw and roll rows negated vs R):
#     yaw:        [-1/2, +1/2,   0,     0,     0,     0   ]
#     pitch:      [  0,    0,  +√2/4, -√2/4, -√2/4, +√2/4]
#     roll:       [  0,    0,  +√2/4, -√2/4, +√2/4, -√2/4]
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


# ── Motor nucleus index constants (0–13) ──────────────────────────────────────

ABN_L, ABN_R       =  0,  1   # Abducens nucleus motoneurons (CN VI) → ipsilateral LR
CN4_L, CN4_R       =  2,  3   # Trochlear nucleus (CN IV)  — SO, contralateral projection
CN3_MR_L, CN3_MR_R =  4,  5   # CN III — medial rectus subnucleus (vergence drive)
CN3_SR_L, CN3_SR_R =  6,  7   # CN III — superior rectus subnucleus
CN3_IR_L, CN3_IR_R =  8,  9   # CN III — inferior rectus subnucleus
CN3_IO_L, CN3_IO_R = 10, 11   # CN III — inferior oblique subnucleus
AIN_L, AIN_R       = 12, 13   # Abducens internuclear neurons → contralateral MR via MLF

N_NUCLEI = 14
N_NERVES = 12   # = 6 left-eye + 6 right-eye


# ── Stage 1 — M_NUCLEUS  (14 × 6)  [version, vergence] → nuclei ──────────────
# Each row encodes only the input that nucleus actually receives:
#   ABN_L/R      — version + vergence drive (motoneuron pool firing pattern: LR
#                   fires more during divergence, so vrg_yaw column is non-zero)
#   AIN_L/R      — pure version drive (saccadic burst); MLF carries this only
#   CN3_MR_L/R   — vergence-only drive (supraoculomotor → MR motoneurons)
#   Other CN3, CN4 — derived so the healthy round-trip recovers the
#                     [ver ± ½·vrg] left/right motor commands at the plant.
#
# Healthy round-trip target  (M_NERVE_PROJ @ M_NUCLEUS = M_FULL):
#   M_FULL (12 × 6) = [[M_NERVE_L | +0.5·M_NERVE_L],
#                       [M_NERVE_R | −0.5·M_NERVE_R]]

_M_FULL_np = np.vstack([
    np.hstack([_M_NERVE_L_np,  0.5 * _M_NERVE_L_np]),   # left  nerves: version | vergence
    np.hstack([_M_NERVE_R_np, -0.5 * _M_NERVE_R_np]),   # right nerves: version | vergence
]).astype(np.float32)   # (12, 6)

_M_NUCLEUS_np = np.zeros((N_NUCLEI, 6), dtype=np.float32)

# ABN motoneurons: drive ipsilateral LR (version + vergence components).
_M_NUCLEUS_np[ABN_L] = np.concatenate([_M_NERVE_L_np[LR, :],  0.5 * _M_NERVE_L_np[LR, :]])
_M_NUCLEUS_np[ABN_R] = np.concatenate([_M_NERVE_R_np[LR, :], -0.5 * _M_NERVE_R_np[LR, :]])

# AIN: pure version drive (no vergence column). Same sign as ABN motoneuron
# version output — both populations decode horizontal version from the same
# saccadic burst command.
_M_NUCLEUS_np[AIN_L, :3] = _M_NERVE_L_np[LR, :]   # = [-1, 0, 0]
_M_NUCLEUS_np[AIN_R, :3] = _M_NERVE_R_np[LR, :]   # = [+1, 0, 0]

# CN3_MR: vergence-only drive (version arrives at MR via MLF from contralateral AIN).
# MR_L total = AIN_R (+ver) + CN3_MR_L (+½·vrg) = +ver + ½·vrg ✓
_M_NUCLEUS_np[CN3_MR_L, 3:] = +0.5 * _M_NERVE_L_np[MR, :]   # → [+½, 0, 0]
_M_NUCLEUS_np[CN3_MR_R, 3:] = -0.5 * _M_NERVE_R_np[MR, :]   # → [+½, 0, 0]

# CN4 (contralateral SO):
#   SO_L drive = M_NERVE_L[SO] @ [ver + 0.5·vrg]; CN4_R is its sole projector.
#   SO_R drive = M_NERVE_R[SO] @ [ver − 0.5·vrg]; CN4_L is its sole projector.
_M_NUCLEUS_np[CN4_R] = np.concatenate([_M_NERVE_L_np[SO, :],  0.5 * _M_NERVE_L_np[SO, :]])
_M_NUCLEUS_np[CN4_L] = np.concatenate([_M_NERVE_R_np[SO, :], -0.5 * _M_NERVE_R_np[SO, :]])

# Remaining CN3 subdivisions (SR, IR, IO): direct ipsilateral, both version + vergence.
for nuc, mus in ((CN3_SR_L, SR), (CN3_IR_L, IR), (CN3_IO_L, IO)):
    _M_NUCLEUS_np[nuc] = np.concatenate([_M_NERVE_L_np[mus, :],  0.5 * _M_NERVE_L_np[mus, :]])
for nuc, mus in ((CN3_SR_R, SR), (CN3_IR_R, IR), (CN3_IO_R, IO)):
    _M_NUCLEUS_np[nuc] = np.concatenate([_M_NERVE_R_np[mus, :], -0.5 * _M_NERVE_R_np[mus, :]])


# ── Stage 2 — M_NERVE_PROJ_BASE  (12 × 14)  nucleus → nerve ──────────────────
# Healthy connectivity at unit MLF gain. The FCP step injects g_mlf_L/R into
# the (MR_L, AIN_R) and (MR_R, AIN_L) entries at runtime — see fcp.step.

_M_NERVE_PROJ_np = np.zeros((N_NERVES, N_NUCLEI), dtype=np.float32)

# ABN motoneurons → ipsilateral LR (CN VI nerve, uncrossed).
_M_NERVE_PROJ_np[LR_L, ABN_L] = 1.0
_M_NERVE_PROJ_np[LR_R, ABN_R] = 1.0

# AIN → contralateral MR via MLF (g_mlf_L/R injected per-step in fcp).
# Default to 1.0 here so the static matrix preserves the healthy round-trip
# for any code that uses it directly (e.g. tests, doc).
_M_NERVE_PROJ_np[MR_R, AIN_L] = 1.0     # right MLF: AIN_L → MR_R
_M_NERVE_PROJ_np[MR_L, AIN_R] = 1.0     # left  MLF: AIN_R → MR_L

# CN4 → contralateral SO (CN IV decussates dorsally).
_M_NERVE_PROJ_np[SO_R, CN4_L] = 1.0
_M_NERVE_PROJ_np[SO_L, CN4_R] = 1.0

# CN3 vergence/version drive → ipsilateral muscle (CN III, uncrossed).
_M_NERVE_PROJ_np[MR_L, CN3_MR_L] = 1.0
_M_NERVE_PROJ_np[MR_R, CN3_MR_R] = 1.0
_M_NERVE_PROJ_np[SR_L, CN3_SR_L] = 1.0
_M_NERVE_PROJ_np[SR_R, CN3_SR_R] = 1.0
_M_NERVE_PROJ_np[IR_L, CN3_IR_L] = 1.0
_M_NERVE_PROJ_np[IR_R, CN3_IR_R] = 1.0
_M_NERVE_PROJ_np[IO_L, CN3_IO_L] = 1.0
_M_NERVE_PROJ_np[IO_R, CN3_IO_R] = 1.0


# ── Sanity check: healthy round-trip preserves the version+½·vergence command ──
# M_NERVE_PROJ @ M_NUCLEUS must equal M_FULL (left/right ± ½·vrg split).
_check = _M_NERVE_PROJ_np @ _M_NUCLEUS_np
assert np.allclose(_check, _M_FULL_np, atol=1e-6), \
    f"FCP healthy round-trip broken: max diff {np.abs(_check - _M_FULL_np).max():.3e}"


# ── JAX arrays (immutable; safe inside jit) ────────────────────────────────────

M_NERVE_R     = jnp.array(_M_NERVE_R_np)             # (6, 3)  right-eye muscle geometry
M_NERVE_L     = jnp.array(_M_NERVE_L_np)             # (6, 3)  left-eye  muscle geometry
M_PLANT_EYE_R = jnp.array(_M_PLANT_EYE_R_np)        # (3, 6)  right-eye decode (plant)
M_PLANT_EYE_L = jnp.array(_M_PLANT_EYE_L_np)        # (3, 6)  left-eye  decode (plant)

M_NUCLEUS    = jnp.array(_M_NUCLEUS_np)    # (14, 6) brain → nuclei
M_NERVE_PROJ = jnp.array(_M_NERVE_PROJ_np) # (12,14) nuclei → nerves (g_mlf=1 default)

# g_nucleus is (12,) — one gain per anatomical nucleus.  AIN_L/AIN_R share their
# gain with ABN_L/ABN_R respectively, since motoneurons and internuclear neurons
# are anatomically intermingled in the abducens nucleus and any real lesion
# affects both populations together.  The FCP expands (12,) → (14,) at runtime.
N_GAINS_NUCLEUS   = 12
G_NUCLEUS_DEFAULT = jnp.ones(N_GAINS_NUCLEUS, dtype=jnp.float32)  # healthy: all = 1
G_NERVE_DEFAULT   = jnp.ones(N_NERVES, dtype=jnp.float32)         # healthy: all = 1

# Backward-compat alias (was used in previous g_muscle_L/R architecture)
G_MUSCLE_DEFAULT  = jnp.ones(6, dtype=jnp.float32)
