"""Systematic hyperparameter search on training data (v1.5)."""

from __future__ import annotations

import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from biohub.config import Config, MATCH_GATE_UM, SCALE
from biohub.cache import cached_detection
from biohub.data import find_geff_path, list_datasets, load_geff_graph, load_volume, read_physical_metadata, read_zarr_meta
from biohub.detection import detect_cells
from biohub.tracking import link_frames


def _f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return 2.0 * precision * recall / denom if denom else 0.0


def evaluate_sparse_graph(
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    pred_edges: Sequence[Tuple[int, int]],
    pred_id_to_gt: Dict[int, int],
    evaluable_times: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Evaluate links and divisions only where sparse GEFF labels support it.

    A predicted edge is evaluable only when both predicted endpoints matched
    annotated GT nodes. Edges with either endpoint unmatched are unknown, not
    false positives. GT recall is restricted to annotated edges whose endpoint
    frames were present in ``gt_nodes``.
    """
    node_time = {
        int(row.node_id): int(row.t)
        for row in gt_nodes[["node_id", "t"]].itertuples(index=False)
    }
    annotated_times = (
        set(node_time.values())
        if evaluable_times is None
        else {int(t) for t in evaluable_times}
    )
    raw_gt_edges = [] if gt_edges is None else [
        (int(s), int(t))
        for s, t in gt_edges[["source_id", "target_id"]].to_numpy()
    ]
    gt_set = {
        (s, t)
        for s, t in raw_gt_edges
        if s in node_time and t in node_time
        and node_time[s] in annotated_times and node_time[t] in annotated_times
    }

    evaluable_pairs: List[Tuple[int, int]] = []
    unknown_edges = 0
    for s, t in pred_edges:
        gs, gt = pred_id_to_gt.get(s), pred_id_to_gt.get(t)
        if gs is None or gt is None or gs not in node_time or gt not in node_time:
            unknown_edges += 1
            continue
        evaluable_pairs.append((int(gs), int(gt)))

    predicted_pair_set = set(evaluable_pairs)
    edge_tp = sum(pair in gt_set for pair in evaluable_pairs)
    edge_fp = len(evaluable_pairs) - edge_tp
    edge_fn = len(gt_set - predicted_pair_set)
    edge_precision = edge_tp / (edge_tp + edge_fp) if edge_tp + edge_fp else 0.0
    edge_recall = edge_tp / (edge_tp + edge_fn) if edge_tp + edge_fn else 0.0

    gt_children: Dict[int, set[int]] = defaultdict(set)
    for s, t in gt_set:
        gt_children[s].add(t)
    gt_division_parents = {s for s, children in gt_children.items() if len(children) >= 2}

    pred_children: Dict[int, List[int]] = defaultdict(list)
    for s, t in pred_edges:
        pred_children[int(s)].append(int(t))
    raw_divisions = {s: children for s, children in pred_children.items() if len(children) >= 2}
    division_tp = 0
    division_fp = 0
    unknown_divisions = 0
    confirmed_gt_parents: set[int] = set()
    for source, children in raw_divisions.items():
        gt_source = pred_id_to_gt.get(source)
        gt_child_ids = [pred_id_to_gt.get(child) for child in children]
        if gt_source is None or any(child is None for child in gt_child_ids):
            unknown_divisions += 1
            continue
        matched_children = {int(child) for child in gt_child_ids if child is not None}
        if (
            int(gt_source) in gt_division_parents
            and len(matched_children & gt_children[int(gt_source)]) >= 2
        ):
            division_tp += 1
            confirmed_gt_parents.add(int(gt_source))
        else:
            division_fp += 1
    division_fn = len(gt_division_parents - confirmed_gt_parents)
    division_precision = (
        division_tp / (division_tp + division_fp)
        if division_tp + division_fp else 0.0
    )
    division_recall = (
        division_tp / (division_tp + division_fn)
        if division_tp + division_fn else 0.0
    )

    return {
        "edge_precision": float(edge_precision),
        "edge_recall": float(edge_recall),
        "edge_f1": float(_f1(edge_precision, edge_recall)),
        "edge_tp": float(edge_tp),
        "edge_fp": float(edge_fp),
        "edge_fn": float(edge_fn),
        "edge_unknown": float(unknown_edges),
        "edge_pred_evaluable": float(len(evaluable_pairs)),
        "edge_pred_unknown": float(unknown_edges),
        "edge_gt_evaluable": float(len(gt_set)),
        "division_precision": float(division_precision),
        "division_recall": float(division_recall),
        "division_f1": float(_f1(division_precision, division_recall)),
        "division_tp": float(division_tp),
        "division_fp": float(division_fp),
        "division_fn": float(division_fn),
        "division_unknown": float(unknown_divisions),
        "divisions": float(len(raw_divisions)),
    }


def _match_nodes_to_gt(
    pred_xyz: np.ndarray,
    gt_frame: pd.DataFrame,
    scale: np.ndarray,
    gate_um: float = MATCH_GATE_UM,
) -> Dict[int, int]:
    """Map predicted index -> GT node ID with optimal one-to-one matching."""
    if len(pred_xyz) == 0 or len(gt_frame) == 0:
        return {}
    gt_xyz = gt_frame[["z", "y", "x"]].to_numpy(dtype=np.float64)
    # Python integers preserve unsigned 64-bit IDs. NumPy's platform `int`
    # can be 32-bit on Windows and silently truncates real GEFF identifiers.
    gt_ids = [int(value) for value in gt_frame["node_id"].to_numpy()]
    pred_phys = pred_xyz.astype(np.float64) * scale
    gt_phys = gt_xyz * scale
    from scipy.optimize import linear_sum_assignment

    dist = np.sqrt(((pred_phys[:, None, :] - gt_phys[None, :, :]) ** 2).sum(axis=2))
    row_idx, col_idx = linear_sum_assignment(np.where(dist <= gate_um, dist, 1e6))
    mapping: Dict[int, int] = {}
    for i, j in zip(row_idx, col_idx):
        if dist[i, j] <= gate_um:
            mapping[int(i)] = int(gt_ids[j])
    return mapping


def score_config_on_sample(
    train_dir: Path,
    name: str,
    cfg: Config,
    n_frames: int = 4,
) -> Dict[str, float]:
    """Proxy score using production linking and sparse-label-safe metrics."""
    geff_path = find_geff_path(train_dir, name)
    if geff_path is None:
        return {
            "label_available": 0.0,
            "recall": np.nan,
            "edge_proxy": np.nan,
            "divisions": np.nan,
            "score": np.nan,
        }
    gt_nodes, gt_edges = load_geff_graph(geff_path)
    if gt_nodes is None:
        return {
            "label_available": 0.0,
            "recall": np.nan,
            "edge_proxy": np.nan,
            "divisions": np.nan,
            "score": np.nan,
        }

    zarr_path = train_dir / f"{name}.zarr"
    shape, dtype = read_zarr_meta(zarr_path)
    voxel_size, frame_interval = read_physical_metadata(zarr_path)
    if voxel_size is not None or frame_interval is not None:
        cfg = cfg.copy_with(
            voxel_size_zyx_um=voxel_size or cfg.voxel_size_zyx_um,
            frame_interval_s=frame_interval or cfg.frame_interval_s,
        )
    scale = cfg.scale_array

    recalls = []
    pred_edges: List[Tuple[int, int]] = []
    prev_xyz = np.empty((0, 3))
    prev_int = np.empty((0,))
    prev_image_features = None
    prev_vel = None
    prev_ids: List[int] = []
    pred_id = 1
    pred_id_to_gt: Dict[int, int] = {}

    labeled_ts = gt_nodes["t"].to_numpy()
    start_t = int(labeled_ts.min()) if len(labeled_ts) else 0
    start_t = min(start_t, max(shape[0] - n_frames, 0))
    frame_range = range(start_t, min(start_t + n_frames, shape[0]))

    for loop_i, t in enumerate(frame_range):
        vol = load_volume(zarr_path, t, shape, dtype)
        prev_count = len(prev_xyz) if loop_i else None
        coords, _, intens = cached_detection(
            zarr_path,
            t,
            cfg,
            prev_count,
            lambda: detect_cells(vol, cfg, prev_count=prev_count),
        )
        if cfg.linker_backend == "image_learned":
            from biohub.learned_linker import extract_local_image_features
            curr_image_features = extract_local_image_features(vol, coords)
        else:
            curr_image_features = None
        gt_t = gt_nodes[gt_nodes["t"] == t]
        # Hungarian (one-to-one) assignment FIRST; one prediction can never
        # satisfy two GT cells.
        frame_map = _match_nodes_to_gt(coords, gt_t, scale) if len(coords) and len(gt_t) else {}
        if len(gt_t):
            recalls.append(len(frame_map) / len(gt_t))
        curr_ids = []
        for i in range(len(coords)):
            pid = pred_id
            pred_id += 1
            curr_ids.append(pid)
            if i in frame_map:
                pred_id_to_gt[pid] = frame_map[i]

        if t > 0 and len(prev_ids) and len(curr_ids):
            links = link_frames(
                prev_ids,
                prev_xyz,
                curr_ids,
                coords,
                cfg,
                prev_intensity=prev_int,
                curr_intensity=intens,
                prev_velocity=prev_vel,
                prev_image_features=prev_image_features,
                curr_image_features=curr_image_features,
            )
            pred_edges.extend((int(s), int(u)) for s, u in links)

            if len(links):
                vel = {}
                pos = {pid: prev_xyz[i] for i, pid in enumerate(prev_ids)}
                pos.update({pid: coords[i] for i, pid in enumerate(curr_ids)})
                for s, u in links:
                    if s in pos and u in pos:
                        vel[u] = (pos[u] - pos[s]) * scale
                prev_vel = np.asarray(
                    [vel.get(pid, np.zeros(3)) for pid in curr_ids],
                    dtype=np.float64,
                )

        prev_xyz, prev_int, prev_ids = coords, intens, curr_ids
        prev_image_features = curr_image_features

    recall = float(np.mean(recalls)) if recalls else 0.0
    graph_metrics = evaluate_sparse_graph(
        gt_nodes,
        gt_edges,
        pred_edges,
        pred_id_to_gt,
        evaluable_times=list(frame_range),
    )
    score = (
        0.45 * recall
        + 0.45 * graph_metrics["edge_f1"]
        + 0.10 * graph_metrics["division_f1"]
    )
    return {
        "label_available": 1.0,
        "recall": recall,
        "edge_proxy": graph_metrics["edge_f1"],
        "score": score,
        **graph_metrics,
    }


def hyperparameter_search(
    train_dir: Path,
    cfg: Config,
    sample_limit: int = 3,
    frames: int = 4,
) -> Tuple[Config, pd.DataFrame]:
    """
    Grid search key hyperparameters on training samples.

    Returns best config and a results table.
    """
    names = list_datasets(train_dir)[:sample_limit]
    if not names:
        return cfg, pd.DataFrame()

    grid = {
        "thresh_rel": [0.26, 0.28, 0.30, 0.32],
        "max_link_dist_um": [10.0, 11.0, 12.0],
        "div_parent_dist_um": [11.0, 12.0],
        "div_sister_dist_um": [7.0, 7.5, 8.0],
        "nms_radius_um": [2.5, 2.65],
    }

    rows: List[dict] = []
    best_cfg = cfg
    best_score = -1.0

    base = {
        "thresh_rel": cfg.thresh_rel,
        "max_link_dist_um": cfg.max_link_dist_um,
        "div_parent_dist_um": cfg.div_parent_dist_um,
        "div_sister_dist_um": cfg.div_sister_dist_um,
        "nms_radius_um": cfg.nms_radius_um,
    }

    def _iter_grid(d, keys, i=0, cur=None):
        cur = cur or {}
        if i == len(keys):
            yield cur
            return
        k = keys[i]
        for v in d[k]:
            cur[k] = v
            yield from _iter_grid(d, keys, i + 1, cur)

    for params in _iter_grid(grid, list(grid.keys())):
        t0 = time.time()
        trial = cfg.copy_with(**params)
        scores = [score_config_on_sample(train_dir, n, trial, frames) for n in names]
        runtime_s = time.time() - t0
        score_values = [s["score"] for s in scores]
        mean_score = float(np.nanmean(score_values)) if np.isfinite(score_values).any() else np.nan
        row = {
            **params,
            "mean_score": mean_score,
            "runtime_s": round(runtime_s, 2),
            "n_samples": len(names),
            "valid_scores": int(np.isfinite(score_values).sum()),
        }
        row.update({f"{k}_{i}": s[k] for i, s in enumerate(scores) for k in ("recall", "edge_proxy")})
        rows.append(row)
        if np.isfinite(mean_score) and mean_score > best_score:
            best_score = mean_score
            best_cfg = trial.copy_with(**params)

    results = pd.DataFrame(rows).sort_values("mean_score", ascending=False)
    return best_cfg, results


def single_knob_candidates() -> List[Dict[str, float]]:
    """
    Low-risk one-parameter candidates for post-v4 leaderboard optimization.

    These avoid v1.5 regression flags (gap close, soft prune, division symmetry)
    and keep each trial attributable to exactly one parameter.
    """
    return [
        {},
        {"max_link_dist_um": 9.5},
        {"max_link_dist_um": 10.0},
        {"max_link_dist_um": 10.5},
        {"max_link_dist_um": 11.5},
        {"thresh_rel": 0.26},
        {"thresh_rel": 0.28},
        {"thresh_rel": 0.32},
        {"thresh_rel": 0.34},
        {"nms_radius_um": 2.50},
        {"nms_radius_um": 2.80},
        {"div_parent_dist_um": 11.0},
        {"div_parent_dist_um": 13.0},
        {"div_sister_dist_um": 7.0},
        {"div_sister_dist_um": 8.0},
    ]


def _candidate_name(params: Dict[str, float]) -> Tuple[str, str, float | None]:
    if not params:
        return "baseline", "baseline", None
    if len(params) != 1:
        raise ValueError(f"Expected one-parameter candidate, got {params}")
    knob, value = next(iter(params.items()))
    return f"{knob}={value}", knob, float(value)


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Return mean and bootstrap CI after dropping non-finite values."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return np.nan, np.nan, np.nan
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(random_state)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return mean, float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def paired_bootstrap_delta(
    baseline: Sequence[float],
    candidate: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap a paired candidate-minus-baseline delta by dataset."""
    base = np.asarray(baseline, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if base.shape != cand.shape:
        raise ValueError("baseline and candidate must contain aligned per-dataset scores")
    valid = np.isfinite(base) & np.isfinite(cand)
    delta = cand[valid] - base[valid]
    return bootstrap_ci(delta, n_boot=n_boot, ci=ci, random_state=random_state)


def single_knob_sweep(
    train_dir: Path,
    cfg: Config,
    sample_limit: int = 5,
    frames: int = 6,
    candidates: Optional[Iterable[Dict[str, float]]] = None,
) -> Tuple[Config, pd.DataFrame]:
    """
    Score low-risk one-knob candidates on sparse train labels.

    The public leaderboard is still the final judge, but this avoids blind
    manual trials by ranking candidates with the same proxy used for local
    hyperparameter search.
    """
    names = list_datasets(train_dir)[:sample_limit]
    if not names:
        return cfg, pd.DataFrame()

    rows: List[dict] = []
    candidate_scores: Dict[str, np.ndarray] = {}
    best_cfg = cfg
    best_score = -1.0
    candidate_list = list(candidates) if candidates is not None else single_knob_candidates()

    for params in candidate_list:
        candidate, knob, value = _candidate_name(params)
        trial = cfg.copy_with(**params)
        # Explicit threshold trials must not be overwritten by later density calibration.
        if knob == "thresh_rel":
            trial = trial.copy_with(run_density_calibration=False)

        t0 = time.time()
        scores = [score_config_on_sample(train_dir, n, trial, frames) for n in names]
        runtime_s = time.time() - t0
        score_values = np.array([s["score"] for s in scores], dtype=np.float64)
        candidate_scores[candidate] = score_values
        recall_values = np.array([s["recall"] for s in scores], dtype=np.float64)
        edge_values = np.array([s["edge_proxy"] for s in scores], dtype=np.float64)
        division_values = np.array([s["divisions"] for s in scores], dtype=np.float64)
        label_values = np.array([s["label_available"] for s in scores], dtype=np.float64)
        valid_scores = int(np.isfinite(score_values).sum())
        mean_score = float(np.nanmean(score_values)) if valid_scores else np.nan
        mean_recall = float(np.nanmean(recall_values)) if np.isfinite(recall_values).any() else np.nan
        mean_edge = float(np.nanmean(edge_values)) if np.isfinite(edge_values).any() else np.nan
        mean_divisions = float(np.nanmean(division_values)) if np.isfinite(division_values).any() else np.nan
        row = {
            "candidate": candidate,
            "knob": knob,
            "value": value,
            "mean_score": mean_score,
            "mean_recall": mean_recall,
            "mean_edge_proxy": mean_edge,
            "mean_divisions": mean_divisions,
            "runtime_s": round(runtime_s, 2),
            "n_samples": len(names),
            "valid_scores": valid_scores,
            "labels_found": int(label_values.sum()),
            "frames": frames,
            "run_density_calibration": trial.run_density_calibration,
        }
        for k in (
            "thresh_rel",
            "max_link_dist_um",
            "nms_radius_um",
            "div_parent_dist_um",
            "div_sister_dist_um",
        ):
            row[k] = getattr(trial, k)
        rows.append(row)
        if np.isfinite(mean_score) and mean_score > best_score:
            best_score = mean_score
            best_cfg = trial

    results = pd.DataFrame(rows)
    baseline_rows = results[results["candidate"] == "baseline"]
    if len(baseline_rows) and np.isfinite(float(baseline_rows.iloc[0]["mean_score"])):
        baseline_score = float(baseline_rows.iloc[0]["mean_score"])
        deltas = {
            name: paired_bootstrap_delta(candidate_scores["baseline"], values)
            for name, values in candidate_scores.items()
        }
        results["delta_vs_baseline"] = results["candidate"].map(lambda name: deltas[name][0])
        results["delta_ci_low"] = results["candidate"].map(lambda name: deltas[name][1])
        results["delta_ci_high"] = results["candidate"].map(lambda name: deltas[name][2])
        results["significant_improvement"] = results["delta_ci_low"] > 0.0
    else:
        results["delta_vs_baseline"] = np.nan
        results["delta_ci_low"] = np.nan
        results["delta_ci_high"] = np.nan
        results["significant_improvement"] = False
    results = results.sort_values("mean_score", ascending=False, na_position="last")
    return best_cfg, results
