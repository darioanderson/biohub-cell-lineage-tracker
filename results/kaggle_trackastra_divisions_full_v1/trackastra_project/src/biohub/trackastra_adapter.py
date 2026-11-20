"""Adapters between Biohub point detections and Trackastra instance masks."""

from __future__ import annotations

from itertools import product
from collections import defaultdict
from typing import Mapping, Sequence

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


def add_trackastra_division_proposals(
    baseline_edges: Sequence[tuple[int, int]],
    trackastra_edges: Sequence[tuple[int, int]],
    node_time: Mapping[int, int],
    node_xyz: Mapping[int, np.ndarray],
    scale_zyx: np.ndarray,
    *,
    parent_gate_um: float,
    sister_gate_um: float,
    midpoint_gate_um: float,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Add only Trackastra splits anchored by a baseline continuation.

    The baseline continuation graph is preserved byte-for-byte.  A Trackastra
    1->2 proposal may add one unmatched daughter only when the other proposed
    daughter is already the baseline continuation and all physical gates pass.
    """
    result = list(dict.fromkeys((int(a), int(b)) for a, b in baseline_edges))
    children: dict[int, list[int]] = defaultdict(list)
    incoming: dict[int, int] = {}
    for source, target in result:
        children[source].append(target)
        incoming[target] = source
    proposed: dict[int, list[int]] = defaultdict(list)
    for source, target in trackastra_edges:
        if int(target) not in proposed[int(source)]:
            proposed[int(source)].append(int(target))

    diagnostics = {"trackastra_splits": 0, "accepted_splits": 0, "rejected_conflict": 0, "rejected_geometry": 0}
    scale = np.asarray(scale_zyx, dtype=float)
    for source in sorted(proposed):
        candidates = proposed[source]
        if len(candidates) != 2:
            continue
        diagnostics["trackastra_splits"] += 1
        baseline_children = children.get(source, [])
        if len(baseline_children) != 1 or baseline_children[0] not in candidates:
            diagnostics["rejected_conflict"] += 1
            continue
        continuation = baseline_children[0]
        daughter = candidates[0] if candidates[1] == continuation else candidates[1]
        if daughter in incoming or any(node not in node_time or node not in node_xyz for node in (source, continuation, daughter)):
            diagnostics["rejected_conflict"] += 1
            continue
        if node_time[continuation] != node_time[source] + 1 or node_time[daughter] != node_time[source] + 1:
            diagnostics["rejected_conflict"] += 1
            continue
        parent = np.asarray(node_xyz[source], dtype=float) * scale
        first = np.asarray(node_xyz[continuation], dtype=float) * scale
        second = np.asarray(node_xyz[daughter], dtype=float) * scale
        if (
            np.linalg.norm(first - parent) > parent_gate_um
            or np.linalg.norm(second - parent) > parent_gate_um
            or np.linalg.norm(first - second) > sister_gate_um
            or np.linalg.norm(0.5 * (first + second) - parent) > midpoint_gate_um
        ):
            diagnostics["rejected_geometry"] += 1
            continue
        result.append((source, daughter))
        children[source].append(daughter)
        incoming[daughter] = source
        diagnostics["accepted_splits"] += 1
    return result, diagnostics
