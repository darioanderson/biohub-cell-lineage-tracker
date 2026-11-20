"""Adapters between Biohub point detections and Trackastra instance masks."""

from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import ball
from skimage.segmentation import watershed


def make_isotropic_xy(volume: np.ndarray, factor: int = 4) -> np.ndarray:
    """Block-average XY so the competition's 4:1 anisotropy becomes isotropic."""
    if factor < 1:
        raise ValueError("factor must be positive")
    nz, ny, nx = volume.shape
    ny2, nx2 = ny // factor, nx // factor
    cropped = np.asarray(volume)[:, : ny2 * factor, : nx2 * factor]
    return cropped.reshape(nz, ny2, factor, nx2, factor).mean(axis=(2, 4))


def _nearest_free_marker(markers: np.ndarray, point: np.ndarray, max_radius: int = 2) -> tuple[int, int, int]:
    center = np.clip(np.rint(point).astype(int), 0, np.asarray(markers.shape) - 1)
    for radius in range(max_radius + 1):
        offsets = sorted(product(range(-radius, radius + 1), repeat=3), key=lambda d: (sum(v * v for v in d), d))
        for offset in offsets:
            candidate = center + np.asarray(offset)
            if np.any(candidate < 0) or np.any(candidate >= markers.shape):
                continue
            index = tuple(int(v) for v in candidate)
            if markers[index] == 0:
                return index
    raise ValueError(f"Could not place a unique Trackastra marker near {tuple(point)}")


def centroids_to_watershed_mask(
    volume_iso: np.ndarray,
    centroids_zyx: np.ndarray,
    *,
    xy_downsample: int = 4,
    max_radius: int = 5,
    foreground_percentile: float = 75.0,
) -> np.ndarray:
    """Create deterministic image-supported instances with one label per centroid."""
    if len(centroids_zyx) >= np.iinfo(np.uint16).max:
        raise ValueError("Trackastra uint16 mask supports fewer than 65535 detections per frame")
    markers = np.zeros(volume_iso.shape, dtype=np.int32)
    for label, raw_point in enumerate(np.asarray(centroids_zyx, dtype=float), start=1):
        iso_point = np.array([raw_point[0], raw_point[1] / xy_downsample, raw_point[2] / xy_downsample])
        markers[_nearest_free_marker(markers, iso_point)] = label
    if not len(centroids_zyx):
        return markers.astype(np.uint16)
    seed_mask = markers > 0
    bounded = binary_dilation(seed_mask, structure=ball(max_radius))
    foreground = ((volume_iso > np.percentile(volume_iso, foreground_percentile)) & bounded) | seed_mask
    labels = watershed(-np.asarray(volume_iso, dtype=np.float32), markers, mask=foreground, compactness=0.01)
    present = set(np.unique(labels))
    missing = [label for label in range(1, len(centroids_zyx) + 1) if label not in present]
    if missing:
        raise RuntimeError(f"Watershed lost Trackastra instance labels: {missing[:10]}")
    return labels.astype(np.uint16)


def graph_edges_to_detection_ids(graph, frame_ids: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    """Translate Trackastra graph edges back to Biohub detection IDs."""
    lookup: dict[object, int] = {}
    for node, attrs in graph.nodes(data=True):
        t, label = int(attrs["time"]), int(attrs["label"])
        if 0 <= t < len(frame_ids) and 1 <= label <= len(frame_ids[t]):
            lookup[node] = int(frame_ids[t][label - 1])
    return [
        (lookup[source], lookup[target])
        for source, target in graph.edges()
        if source in lookup and target in lookup
    ]
