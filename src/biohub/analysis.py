"""Structured analysis results and in-memory pipeline runner."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict
from uuid import uuid4
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from biohub.config import Config
from biohub.data import load_volume, read_physical_metadata, read_zarr_meta
from biohub.detector import get_detector
from biohub.tracking import (
    annotate_lineage_ids,
    close_frame_gaps,
    count_divisions,
    graph_invariant_errors,
    link_frames,
    link_sequence_global,
    prune_isolated_nodes,
)


@dataclass
class FrameResult:
    t: int
    coords: np.ndarray
    scores: np.ndarray
    intensities: np.ndarray
    node_ids: List[int]
    projection_mask: Optional[np.ndarray] = None


@dataclass
class AnalysisResult:
    dataset_name: str
    shape: Tuple[int, ...]
    nodes: pd.DataFrame
    edges: pd.DataFrame
    frames: List[FrameResult] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_divisions(self) -> int:
        if self.edges.empty:
            return 0
        counts = self.edges.groupby("source_id").size()
        return int((counts >= 2).sum())


def _rows_to_frames(node_rows: List[dict], frame_ids, frame_centroids, frame_intensities) -> List[FrameResult]:
    frames = []
    for t, (ids, xyz, ints) in enumerate(zip(frame_ids, frame_centroids, frame_intensities)):
        frames.append(FrameResult(t=t, coords=xyz, scores=np.zeros(len(xyz)), intensities=ints, node_ids=ids))
    return frames


def run_tracking_pipeline(
    source: Union[Path, Sequence[np.ndarray]],
    cfg: Config,
    dataset_name: str = "dataset",
    max_frames: Optional[int] = None,
    progress_callback=None,
) -> AnalysisResult:
    """
    Run detection, linking, and lineage reconstruction.

    Parameters
    ----------
    source
        Path to a `.zarr` dataset directory or a sequence of `(Z,Y,X)` volumes.
    """
    started = time.perf_counter()
    time_verified = spatial_verified = False
    if isinstance(source, Path):
        shape, dtype = read_zarr_meta(source)
        voxel_size, frame_interval = read_physical_metadata(source)
        time_verified, spatial_verified = frame_interval is not None, voxel_size is not None
        if voxel_size is not None or frame_interval is not None:
            cfg = cfg.copy_with(
                voxel_size_zyx_um=voxel_size or cfg.voxel_size_zyx_um,
                frame_interval_s=frame_interval or cfg.frame_interval_s,
            )
        n_t = shape[0] if max_frames is None else min(shape[0], max_frames)
        volume_iter = (
            (t, load_volume(source, t, shape, dtype)) for t in range(n_t)
        )
        full_shape = shape
    else:
        volumes = source if hasattr(source, '__len__') and hasattr(source, '__getitem__') else list(source)
        if max_frames is not None:
            volumes = volumes[:max_frames]
        n_t = len(volumes)
        if n_t == 0:
            raise ValueError("No volumes provided.")
        z, y, x = volumes[0].shape
        full_shape = (n_t, z, y, x)
        volume_iter = enumerate(volumes)

    detector = get_detector(cfg)

    node_rows: List[dict] = []
    edge_rows: List[dict] = []
    frame_ids: List[List[int]] = []
    frame_centroids: List[np.ndarray] = []
    frame_intensities: List[np.ndarray] = []
    frame_image_features: List[np.ndarray] = []
    frame_results: List[FrameResult] = []
    node_id = 1
    prev_count: Optional[int] = None
    velocity_by_id: Dict[int, np.ndarray] = {}
    position_by_id: Dict[int, np.ndarray] = {}

    for t, vol in volume_iter:
        if progress_callback:
            progress_callback(.7 * t / max(n_t, 1), f'Detecting frame {t+1}/{n_t}')
        coords, scores, intens = detector.detect(vol, cfg, prev_count=prev_count)
        if cfg.linker_backend == "image_learned":
            from biohub.learned_linker import extract_local_image_features
            image_features = extract_local_image_features(vol, coords)
        else:
            image_features = np.empty((len(coords), 0), dtype=np.float32)
        del vol
        gc.collect()

        if len(coords):
            order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
            coords, scores, intens = coords[order], scores[order], intens[order]
            image_features = image_features[order]

        curr_ids = list(range(node_id, node_id + len(coords)))
        node_id += len(coords)
        frame_ids.append(curr_ids)
        frame_centroids.append(coords)
        frame_intensities.append(intens)
        frame_image_features.append(image_features)
        prev_count = len(coords)

        for nid, pt, raw_score, intensity in zip(curr_ids, coords, scores, intens):
            z, y, x = [float(v) for v in pt]
            node_rows.append(
                {
                    "dataset": dataset_name,
                    "row_type": "node",
                    "node_id": int(nid),
                    "t": int(t),
                    "z": z,
                    "y": y,
                    "x": x,
                    "source_id": -1,
                    "target_id": -1,
                    "raw_detection_score": float(raw_score),
                    "intensity": float(intensity),
                }
            )
            position_by_id[int(nid)] = pt.astype(np.float64)

        frame_results.append(
            FrameResult(t=int(t), coords=coords, scores=scores, intensities=intens, node_ids=curr_ids,
                        projection_mask=getattr(detector,'last_projection_masks',None))
        )

    if cfg.tracking_backend == "global":
        for s, u in link_sequence_global(frame_ids, frame_centroids, frame_intensities, cfg):
            edge_rows.append({
                "dataset": dataset_name, "row_type": "edge", "node_id": -1,
                "t": -1, "z": -1, "y": -1, "x": -1,
                "source_id": int(s), "target_id": int(u),
            })
    for t in range(1, n_t) if cfg.tracking_backend == "framewise" else ():
        if progress_callback:
            progress_callback(.7 + .3 * t / max(n_t, 1), f'Linking frame {t+1}/{n_t}')
        prev_ids = frame_ids[t - 1]
        curr_ids = frame_ids[t]
        prev_xyz = frame_centroids[t - 1]
        curr_xyz = frame_centroids[t]
        prev_int = frame_intensities[t - 1]
        curr_int = frame_intensities[t]
        prev_vel = np.asarray(
            [velocity_by_id.get(pid, np.zeros(3, dtype=np.float64)) for pid in prev_ids],
            dtype=np.float64,
        )
        links = link_frames(
            prev_ids,
            prev_xyz,
            curr_ids,
            curr_xyz,
            cfg,
            prev_intensity=prev_int,
            curr_intensity=curr_int,
            prev_velocity=prev_vel if cfg.use_rich_linking else None,
            prev_image_features=frame_image_features[t - 1] if cfg.linker_backend == "image_learned" else None,
            curr_image_features=frame_image_features[t] if cfg.linker_backend == "image_learned" else None,
        )
        for s, u in links:
            edge_rows.append(
                {
                    "dataset": dataset_name,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": int(s),
                    "target_id": int(u),
                }
            )
            if s in position_by_id and u in position_by_id:
                velocity_by_id[u] = (position_by_id[u] - position_by_id[s]) * cfg.scale_array

    pair_edges = [(int(e["source_id"]), int(e["target_id"])) for e in edge_rows]
    for s, u in close_frame_gaps(frame_ids, frame_centroids, frame_intensities, pair_edges, cfg) if cfg.tracking_backend == "framewise" else ():
        edge_rows.append(
            {
                "dataset": dataset_name,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": int(s),
                "target_id": int(u),
            }
        )

    if cfg.prune_isolated_nodes:
        node_rows, edge_rows, prune_stats = prune_isolated_nodes(node_rows, edge_rows, cfg)
    else:
        prune_stats = {"removed_isolated": 0, "flagged_isolated": 0}

    annotate_lineage_ids(node_rows, edge_rows)
    allowed_gap = cfg.global_max_frame_gap if cfg.tracking_backend == "global" else (2 if cfg.gap_close_enabled else 1)
    invariant_errors = graph_invariant_errors(node_rows, edge_rows, allowed_gap=allowed_gap)
    if invariant_errors:
        raise ValueError("Invalid tracking graph: " + "; ".join(invariant_errors))

    node_columns = ['dataset','row_type','node_id','t','z','y','x','source_id','target_id',
                    'raw_detection_score','intensity','track_id','lineage_id','generation','qc_status','filter_reason']
    nodes = pd.DataFrame(node_rows) if node_rows else pd.DataFrame(columns=node_columns)
    edges = pd.DataFrame(edge_rows) if edge_rows else pd.DataFrame(
        columns=["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    )

    stats = {
        "dataset": dataset_name,
        "shape": full_shape,
        "nodes": len(nodes),
        "edges": len(edges),
        "divisions": count_divisions(edge_rows) if edge_rows else 0,
        "removed_isolated": prune_stats.get("removed_isolated", 0),
        "flagged_isolated": prune_stats.get("flagged_isolated", 0),
        "run_id": uuid4().hex,
        "runtime_s": time.perf_counter() - started,
        "effective_config": {k: str(v) if isinstance(v, Path) else v for k,v in asdict(cfg).items()},
        "time_interval_verified": time_verified,
        "spatial_calibration_verified": spatial_verified,
    }
    if progress_callback:
        progress_callback(1., 'Complete')

    return AnalysisResult(
        dataset_name=dataset_name,
        shape=full_shape,
        nodes=nodes,
        edges=edges,
        frames=frame_results,
        stats=stats,
    )
