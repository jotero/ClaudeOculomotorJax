"""Visual-flags segment builder — the only stimulus helper still in this module.

All kinematic stimulus construction (head, scene, target) has moved to
``oculomotor.sim.kinematics``.  This module retains only the per-eye
visibility-flag builder used by the LLM pipeline.
"""

import numpy as np
import jax.numpy as jnp


def build_visual_flags(
    segments,   # list[VisualFlagsSegment]
    total_T: int,
    dt: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert VisualFlagsSegments to per-eye visual flag arrays.

    scene_present_L/R and target_present_L/R default to the segment's scene_present /
    target_present value when the per-eye override fields are None — enabling monocular
    occlusion and cover-test scenarios.

    Returns:
        scene_present_L:  (T,) float32 in [0, 1] — L eye scene visibility
        scene_present_R:  (T,) float32 in [0, 1] — R eye scene visibility
        target_present_L: (T,) float32 in [0, 1] — L eye target visibility
        target_present_R: (T,) float32 in [0, 1] — R eye target visibility
        target_strobed:   (T,) float32 in {0, 1} — 1 = stroboscopic (velocity absent)
    """
    spL_chunks, spR_chunks, tpL_chunks, tpR_chunks, ts_chunks = [], [], [], [], []
    for seg in segments:
        T   = max(1, round(seg.duration_s / dt))
        sp  = float(seg.scene_present)
        spL = float(seg.scene_present_L)  if seg.scene_present_L  is not None else sp
        spR = float(seg.scene_present_R)  if seg.scene_present_R  is not None else sp
        tp  = float(seg.target_present)
        tpL = float(seg.target_present_L) if seg.target_present_L is not None else tp
        tpR = float(seg.target_present_R) if seg.target_present_R is not None else tp
        ts  = float(getattr(seg, 'target_strobed', False))
        spL_chunks.append(np.full(T, spL, dtype=np.float32))
        spR_chunks.append(np.full(T, spR, dtype=np.float32))
        tpL_chunks.append(np.full(T, tpL, dtype=np.float32))
        tpR_chunks.append(np.full(T, tpR, dtype=np.float32))
        ts_chunks.append(np.full(T, ts,  dtype=np.float32))

    spL = np.concatenate(spL_chunks)
    spR = np.concatenate(spR_chunks)
    tpL = np.concatenate(tpL_chunks)
    tpR = np.concatenate(tpR_chunks)
    ts  = np.concatenate(ts_chunks)

    def _fit1d(arr, T):
        if len(arr) >= T: return arr[:T]
        return np.concatenate([arr, np.full(T - len(arr), arr[-1], dtype=np.float32)])

    return (_fit1d(spL, total_T), _fit1d(spR, total_T),
            _fit1d(tpL, total_T), _fit1d(tpR, total_T),
            _fit1d(ts,  total_T))
