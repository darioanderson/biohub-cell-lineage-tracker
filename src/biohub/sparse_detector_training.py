"""Sparse-label-safe target and loss construction for cell detection."""

from __future__ import annotations

import numpy as np


def build_sparse_detection_target(
    normalized_frame: np.ndarray,
    annotated_zyx: np.ndarray,
    teacher_zyx: np.ndarray,
    downsample=(1, 4, 4),
    background_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Use annotated centers and explicit background labels only.

    Darkness does not establish background. DoG detections are predictions and
    must not silently become confirmed positive labels.
    """
    frame = np.asarray(normalized_frame, dtype=np.float32)
    if frame.ndim != 3:
        raise ValueError("normalized_frame must have shape (Z, Y, X)")
    if np.asarray(teacher_zyx).size:
        raise ValueError('Unverified teacher detections cannot be confirmed labels')
    ds = np.asarray(downsample, dtype=int)
    if ds.shape != (3,) or np.any(ds < 1):
        raise ValueError("downsample must contain three positive integers")
    reduced = frame[:: ds[0], :: ds[1], :: ds[2]]
    target = np.zeros(reduced.shape, dtype=np.float32)
    known = np.zeros(reduced.shape, dtype=bool)
    if background_mask is not None:
        background = np.asarray(background_mask)
        if background.shape != frame.shape or background.dtype != np.bool_:
            raise ValueError('background_mask must be boolean and match the full frame')
        known = background[::ds[0], ::ds[1], ::ds[2]].copy()
    for points in (annotated_zyx,):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not len(points):
            continue
        if not np.isfinite(points).all() or np.any(points < 0) or np.any(points >= np.asarray(frame.shape)):
            raise ValueError('Annotated centers must be finite and inside the frame')
        index = np.rint(points / ds).astype(int)
        index = np.clip(index, 0, np.asarray(reduced.shape) - 1)
        target[tuple(index.T)] = 1.0
        known[tuple(index.T)] = True
    return target, known.astype(np.float32)


def masked_balanced_bce(logits, target, known_mask):
    """Balanced BCE over confirmed positives and safe background only."""
    import torch
    import torch.nn.functional as functional

    target = target.to(dtype=logits.dtype)
    known_mask = known_mask.to(dtype=logits.dtype)
    positive = known_mask * target
    negative = known_mask * (1.0 - target)
    positive_count = positive.flatten(1).sum(1).clamp(min=1.0)
    negative_count = negative.flatten(1).sum(1).clamp(min=1.0)
    shape = (logits.shape[0],) + (1,) * (logits.ndim - 1)
    weights = positive / positive_count.reshape(shape) + negative / negative_count.reshape(shape)
    per_voxel = functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (per_voxel * weights).flatten(1).sum(1).mean()
