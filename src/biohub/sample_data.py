"""Synthetic and bundled sample volumes for offline development."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticGroundTruth:
    """Rendered volume plus exact observation and lineage truth."""

    volume: np.ndarray
    nodes: pd.DataFrame
    edges: pd.DataFrame
    divisions: pd.DataFrame
    scenario: str


SYNTHETIC_SCENARIOS = (
    "stationary",
    "uniform_translation",
    "acceleration",
    "crossing",
    "dense_packing",
    "entering_leaving",
    "missed_observation",
    "temporary_occlusion",
    "noise",
    "anisotropic_z",
    "one_mitosis",
    "multiple_mitoses",
    "no_division",
    "low_snr",
)


def _linear(start, velocity, frames):
    start = np.asarray(start, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    return {t: start + t * velocity for t in frames}


def synthetic_ground_truth(
    scenario: str = "uniform_translation",
    t: int = 8,
    shape: Tuple[int, int, int] = (16, 48, 48),
    seed: int = 42,
) -> SyntheticGroundTruth:
    """Create a controlled 4D scenario with exact graph and lineage labels."""
    if scenario not in SYNTHETIC_SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SYNTHETIC_SCENARIOS}")
    if t < 4:
        raise ValueError("Synthetic scenarios require at least four frames")

    frames = range(t)
    trajectories: Dict[int, Dict[int, np.ndarray]] = {}
    parents: Dict[int, int] = {}
    lineage: Dict[int, int] = {}
    amplitudes: Dict[int, float] = {}

    if scenario in {"stationary", "no_division", "low_snr"}:
        starts = ([6, 14, 14], [9, 31, 30], [7, 15, 34])
        for track, start in enumerate(starts, 1):
            trajectories[track] = _linear(start, [0, 0, 0], frames)
    elif scenario == "uniform_translation":
        for track, start in enumerate(([5, 12, 12], [9, 29, 30], [7, 34, 15]), 1):
            trajectories[track] = _linear(start, [0.15, 0.7, 0.5], frames)
    elif scenario == "acceleration":
        for track, start in enumerate(([5, 10, 10], [9, 30, 28]), 1):
            start = np.asarray(start, dtype=np.float64)
            velocity = np.asarray([0.05, 0.25, 0.20]) * track
            acceleration = np.asarray([0.01, 0.08, 0.05])
            trajectories[track] = {
                frame: start + velocity * frame + 0.5 * acceleration * frame ** 2
                for frame in frames
            }
    elif scenario == "crossing":
        trajectories[1] = _linear([7, 12, 10], [0, 0, 3.5], frames)
        trajectories[2] = _linear([7, 12, 38], [0, 0, -3.5], frames)
    elif scenario == "dense_packing":
        track = 1
        for yy in (20, 24, 28):
            for xx in (20, 24, 28):
                trajectories[track] = _linear([7, yy, xx], [0, 0.2, -0.1], frames)
                track += 1
    elif scenario == "entering_leaving":
        trajectories[1] = _linear([7, 12, 8], [0, 0.5, 2.0], range(0, t - 2))
        trajectories[2] = _linear([8, 34, 42], [0, -0.4, -2.0], range(2, t))
    elif scenario in {"missed_observation", "temporary_occlusion"}:
        hidden = t // 2
        visible = [frame for frame in frames if frame != hidden]
        trajectories[1] = _linear([7, 12, 12], [0, 0.7, 0.6], visible)
        trajectories[2] = _linear([9, 31, 30], [0, -0.3, 0.2], frames)
    elif scenario in {"noise", "anisotropic_z"}:
        for track, start in enumerate(([5, 12, 12], [9, 29, 30], [7, 34, 15]), 1):
            trajectories[track] = _linear(start, [0.1, 0.4, 0.3], frames)
    elif scenario in {"one_mitosis", "multiple_mitoses"}:
        divide_t = t // 2
        trajectories[1] = _linear([7, 18, 16], [0, 0.4, 0.7], range(divide_t))
        parent_pos = trajectories[1][divide_t - 1]
        trajectories[2] = _linear(parent_pos + [0, -1.2, 1.0], [0, -0.4, 0.8], range(divide_t, t))
        trajectories[3] = _linear(parent_pos + [0, 1.2, -1.0], [0, 0.5, -0.7], range(divide_t, t))
        parents.update({2: 1, 3: 1})
        if scenario == "multiple_mitoses":
            trajectories[4] = _linear([9, 32, 32], [0, -0.3, -0.5], range(divide_t))
            parent_pos = trajectories[4][divide_t - 1]
            trajectories[5] = _linear(parent_pos + [0, -1, -1], [0, -0.5, -0.4], range(divide_t, t))
            trajectories[6] = _linear(parent_pos + [0, 1, 1], [0, 0.4, 0.5], range(divide_t, t))
            parents.update({5: 4, 6: 4})

    for track in trajectories:
        root = track
        while root in parents:
            root = parents[root]
        lineage[track] = root
        amplitudes[track] = 160.0 if scenario == "low_snr" else 1400.0

    node_rows: List[dict] = []
    edge_rows: List[dict] = []
    node_for: Dict[Tuple[int, int], int] = {}
    node_id = 1
    for frame in frames:
        for track in sorted(trajectories):
            if frame not in trajectories[track]:
                continue
            z, y, x = trajectories[track][frame]
            node_for[(track, frame)] = node_id
            node_rows.append({
                "node_id": node_id, "track_id": track, "lineage_id": lineage[track],
                "t": frame, "z": float(z), "y": float(y), "x": float(x),
                "is_division_parent": False,
            })
            node_id += 1

    for track, positions in trajectories.items():
        times = sorted(positions)
        for a, b in zip(times, times[1:]):
            if b - a <= 2:
                edge_rows.append({"source_id": node_for[(track, a)], "target_id": node_for[(track, b)]})
    for child, parent_track in parents.items():
        child_t = min(trajectories[child])
        parent_t = max(tt for tt in trajectories[parent_track] if tt < child_t)
        edge_rows.append({"source_id": node_for[(parent_track, parent_t)], "target_id": node_for[(child, child_t)]})

    nodes = pd.DataFrame(node_rows)
    edges = pd.DataFrame(edge_rows, columns=["source_id", "target_id"])
    child_counts = edges.groupby("source_id").size() if len(edges) else pd.Series(dtype=int)
    division_ids = sorted(int(idx) for idx, count in child_counts.items() if count >= 2)
    if division_ids:
        nodes.loc[nodes["node_id"].isin(division_ids), "is_division_parent"] = True
    divisions = pd.DataFrame({"parent_node_id": division_ids})

    rng = np.random.default_rng(seed)
    noise_high = 180 if scenario == "noise" else (80 if scenario == "low_snr" else 18)
    volume = rng.integers(0, noise_high, size=(t, *shape), dtype=np.uint16)
    zz, yy, xx = np.indices(shape, dtype=np.float32)
    for row in nodes.itertuples(index=False):
        z_sigma = 0.65 if scenario == "anisotropic_z" else 1.2
        blob = np.exp(-(((zz-row.z)/z_sigma)**2 + ((yy-row.y)/1.7)**2 + ((xx-row.x)/1.7)**2) / 2)
        rendered = np.rint(amplitudes[int(row.track_id)] * blob).astype(np.uint16)
        volume[int(row.t)] = np.clip(volume[int(row.t)].astype(np.uint32) + rendered, 0, 65535).astype(np.uint16)
    return SyntheticGroundTruth(volume, nodes, edges, divisions, scenario)


def synthetic_volume(
    t: int = 12,
    z: int = 32,
    y: int = 96,
    x: int = 96,
    n_cells: int = 18,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a fluorescent-cell-like 4D array `(T, Z, Y, X)`.

    Bright Gaussian blobs drift slowly and occasionally divide.
    """
    rng = np.random.default_rng(seed)
    vol = np.zeros((t, z, y, x), dtype=np.uint16)
    cells = []
    for _ in range(n_cells):
        cells.append(
            {
                "pos": np.array(
                    [rng.integers(4, z - 4), rng.integers(8, y - 8), rng.integers(8, x - 8)],
                    dtype=np.float64,
                ),
                "amp": rng.uniform(800, 2200),
            }
        )

    for frame in range(t):
        new_cells = []
        for cell in cells:
            cell["pos"] += rng.normal(0, 0.35, size=3)
            cell["pos"] = np.clip(cell["pos"], [2, 4, 4], [z - 3, y - 4, x - 4])
            cz, cy, cx = cell["pos"].astype(int)
            zz, yy, xx = np.ogrid[:z, :y, :x]
            blob = np.exp(
                -((zz - cz) ** 2 / 4.0 + (yy - cy) ** 2 / 1.5 + (xx - cx) ** 2 / 1.5)
            )
            vol[frame] += (cell["amp"] * blob).astype(np.uint16)
            if frame > 0 and frame % 5 == 0 and rng.random() < 0.08 and len(cells) + len(new_cells) < n_cells + 8:
                daughter = {"pos": cell["pos"] + rng.normal(0.8, 0.2, size=3), "amp": cell["amp"] * 0.9}
                new_cells.append(daughter)
        cells.extend(new_cells)
        vol[frame] += rng.integers(0, 30, size=(z, y, x), dtype=np.uint16)
    return vol


def load_sample_volume(path: Path | None = None) -> np.ndarray:
    path = path or Path("data/sample/synthetic_volume.npy")
    if path.exists():
        return np.load(path)
    return synthetic_volume()


def ensure_sample_data(path: Path | None = None) -> Path:
    path = path or Path("data/sample/synthetic_volume.npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        np.save(path, synthetic_volume())
    return path
