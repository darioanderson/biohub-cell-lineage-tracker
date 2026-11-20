"""Train and evaluate a parent linker on real detector output.

This experiment deliberately keeps every detector output in the assignment
problem.  Only candidate edges whose endpoints have unambiguous sparse GEFF
identities receive training labels; unmatched detections remain in the tracker
and are reported as unknown rather than being invented as background labels.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from biohub.cache import cached_detection
from biohub.config import Config
from biohub.data import (
    find_geff_path,
    list_datasets,
    load_geff_graph,
    load_volume,
    read_physical_metadata,
    read_zarr_meta,
)
from biohub.detection import detect_cells
from biohub.learned_linker import extract_local_image_features
from biohub.tracking import _candidate_components, link_frames
from biohub.tuning import _match_nodes_to_gt, bootstrap_ci, evaluate_sparse_graph, paired_bootstrap_delta


@dataclass
class DetectedFrame:
    t: int
    ids: list[int]
    xyz: np.ndarray
    intensity: np.ndarray
    image_features: np.ndarray
    pred_to_gt: dict[int, int]
    gt_count: int


@dataclass
class DetectedSample:
    name: str
    embryo: str
    gt_nodes: pd.DataFrame
    gt_edges: pd.DataFrame
    frames: list[DetectedFrame]


def embryo_id(name: str) -> str:
    """Return the biological-prefix grouping used by the competition files."""
    return name.split("_", 1)[0]


def _geometry_features(source_phys: np.ndarray, target_phys: np.ndarray, rank: int, source_count: int) -> np.ndarray:
    delta = target_phys - source_phys
    dist = float(np.linalg.norm(delta))
    return np.r_[delta, np.abs(delta), dist, dist * dist, float(rank), float(source_count)]


def build_detected_sample(train_dir: Path, name: str, cfg: Config) -> DetectedSample:
    """Run the production detector and preserve matched and unmatched detections."""
    zarr_path = train_dir / f"{name}.zarr"
    geff_path = find_geff_path(train_dir, name)
    if geff_path is None:
        raise FileNotFoundError(f"No GEFF graph for {name}")
    gt_nodes, gt_edges = load_geff_graph(geff_path)
    if gt_nodes is None or gt_edges is None:
        raise RuntimeError(f"Could not read GEFF graph for {name}")
    shape, dtype = read_zarr_meta(zarr_path)
    voxel, interval = read_physical_metadata(zarr_path)
    if voxel is not None or interval is not None:
        cfg = cfg.copy_with(
            voxel_size_zyx_um=voxel or cfg.voxel_size_zyx_um,
            frame_interval_s=interval or cfg.frame_interval_s,
        )

    frames: list[DetectedFrame] = []
    next_id = 1
    prev_count = None
    # Sparse labels determine what can be scored, but every intervening frame is
    # processed so the detector sees the real temporal count history.
    labeled_times = gt_nodes["t"].astype(int)
    first_t, last_t = int(labeled_times.min()), int(labeled_times.max())
    for t in range(first_t, min(last_t + 1, shape[0])):
        volume = load_volume(zarr_path, t, shape, dtype)
        xyz, _, intensity = cached_detection(
            zarr_path,
            t,
            cfg,
            prev_count,
            lambda: detect_cells(volume, cfg, prev_count=prev_count),
        )
        prev_count = len(xyz)
        ids = list(range(next_id, next_id + len(xyz)))
        next_id += len(xyz)
        gt_frame = gt_nodes[gt_nodes["t"] == t]
        index_map = _match_nodes_to_gt(xyz, gt_frame, cfg.scale_array)
        pred_to_gt = {ids[i]: gt for i, gt in index_map.items()}
        frames.append(
            DetectedFrame(
                t=t,
                ids=ids,
                xyz=xyz,
                intensity=intensity,
                image_features=extract_local_image_features(volume, xyz),
                pred_to_gt=pred_to_gt,
                gt_count=len(gt_frame),
            )
        )
    return DetectedSample(name, embryo_id(name), gt_nodes, gt_edges, frames)


def build_training_pairs(sample: DetectedSample, cfg: Config) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create sparse-safe labels from consecutive real detection frames."""
    gt_parent: dict[int, list[int]] = defaultdict(list)
    for source, target in sample.gt_edges[["source_id", "target_id"]].to_numpy():
        gt_parent[int(target)].append(int(source))
    features, labels = [], []
    eligible_targets = covered_targets = unknown_candidates = 0
    scale = cfg.scale_array
    for prev, curr in zip(sample.frames, sample.frames[1:]):
        if curr.t != prev.t + 1:
            continue
        prev_phys = prev.xyz * scale
        curr_phys = curr.xyz * scale
        # Match production inference exactly: feature rank and source count are
        # computed inside each sparse connected candidate component.
        for source_rows, target_cols in _candidate_components(
            prev_phys, curr_phys, cfg.learned_linker_gate_um
        ):
            for target_i in target_cols:
                target_id = curr.ids[target_i]
                target_gt = curr.pred_to_gt.get(target_id)
                parents = gt_parent.get(target_gt, []) if target_gt is not None else []
                if len(parents) != 1:
                    continue
                eligible_targets += 1
                parent_gt = parents[0]
                distances = np.linalg.norm(prev_phys[source_rows] - curr_phys[target_i], axis=1)
                order = np.argsort(distances, kind="stable")
                target_examples: list[tuple[np.ndarray, int]] = []
                positive_seen = False
                for rank, local_source in enumerate(order):
                    source_i = source_rows[int(local_source)]
                    if distances[local_source] > cfg.learned_linker_gate_um:
                        continue
                    source_id = prev.ids[source_i]
                    source_gt = prev.pred_to_gt.get(source_id)
                    if source_gt is None:
                        unknown_candidates += 1
                        continue
                    geometry = _geometry_features(
                        prev_phys[source_i], curr_phys[target_i], rank, len(source_rows)
                    )
                    src_img = prev.image_features[source_i]
                    dst_img = curr.image_features[target_i]
                    feature = np.r_[geometry, src_img, dst_img, np.abs(src_img - dst_img)]
                    is_parent = int(source_gt == parent_gt)
                    target_examples.append((feature, is_parent))
                    positive_seen |= bool(is_parent)
                # If the detector missed the true parent, every remaining known
                # candidate is an ambiguous alternative rather than a verified
                # negative.  Dropping the whole target prevents false-negative
                # leakage into the supervised edge loss.
                if positive_seen:
                    covered_targets += 1
                    features.extend(feature for feature, _ in target_examples)
                    labels.extend(label for _, label in target_examples)
    x = np.asarray(features, dtype=np.float32).reshape(-1, 25)
    y = np.asarray(labels, dtype=np.float32)
    return x, y, {
        "eligible_targets": eligible_targets,
        "covered_targets": covered_targets,
        "missed_parent_targets": eligible_targets - covered_targets,
        "unknown_candidates": unknown_candidates,
        "labeled_candidates": len(y),
        "positive_edges": int(y.sum()) if len(y) else 0,
    }


def train_linker(x: np.ndarray, y: np.ndarray, output: Path, epochs: int, seed: int) -> Path:
    import torch
    from torch import nn

    if not len(y) or not np.any(y == 1) or not np.any(y == 0):
        raise ValueError("Training fold needs both positive and negative labeled candidates")
    np.random.seed(seed)
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-6] = 1.0
    xt = torch.from_numpy(((x - mean) / std).astype(np.float32))
    yt = torch.from_numpy(y[:, None])
    model = nn.Sequential(nn.Linear(25, 48), nn.ReLU(), nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1.0)]))
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        losses = []
        for start in range(0, len(y), 4096):
            idx = torch.from_numpy(order[start : start + 4096])
            optimizer.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print({"epoch": epoch + 1, "loss": float(np.mean(losses))}, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Tensor metadata keeps the checkpoint compatible with torch.load(...,
    # weights_only=True); numpy arrays are rejected by recent safe loaders.
    torch.save(
        {"model": model.state_dict(), "mean": torch.from_numpy(mean), "std": torch.from_numpy(std)},
        output,
    )
    return output


def evaluate_sample(sample: DetectedSample, cfg: Config) -> dict[str, float]:
    pred_edges: list[tuple[int, int]] = []
    mapping: dict[int, int] = {}
    recalls = []
    unmatched = detections = 0
    prev_velocity = None
    for frame in sample.frames:
        mapping.update(frame.pred_to_gt)
        detections += len(frame.ids)
        unmatched += len(frame.ids) - len(frame.pred_to_gt)
        if frame.gt_count:
            recalls.append(len(frame.pred_to_gt) / frame.gt_count)
    for prev, curr in zip(sample.frames, sample.frames[1:]):
        if curr.t != prev.t + 1:
            continue
        links = link_frames(
            prev.ids, prev.xyz, curr.ids, curr.xyz, cfg,
            prev_intensity=prev.intensity, curr_intensity=curr.intensity,
            prev_velocity=prev_velocity,
            prev_image_features=prev.image_features, curr_image_features=curr.image_features,
        )
        pred_edges.extend((int(a), int(b)) for a, b in links)
        positions = {pid: prev.xyz[i] for i, pid in enumerate(prev.ids)}
        positions.update({pid: curr.xyz[i] for i, pid in enumerate(curr.ids)})
        velocity = {b: (positions[b] - positions[a]) * cfg.scale_array for a, b in links}
        prev_velocity = np.asarray([velocity.get(pid, np.zeros(3)) for pid in curr.ids])
    graph = evaluate_sparse_graph(
        sample.gt_nodes, sample.gt_edges, pred_edges, mapping,
        evaluable_times=[frame.t for frame in sample.frames],
    )
    recall = float(np.mean(recalls)) if recalls else 0.0
    return {
        "recall": recall,
        "detections": float(detections),
        "matched_detections": float(len(mapping)),
        "unmatched_detections_unknown": float(unmatched),
        "score": 0.45 * recall + 0.45 * graph["edge_f1"] + 0.10 * graph["division_f1"],
        **graph,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/detected_linker_joint"))
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Detector cache location; use /kaggle/temp to avoid thousands of output artifacts.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 uses every paired training sample")
    parser.add_argument(
        "--samples-per-embryo", type=int, default=0,
        help="Optional stratified cap for runtime pilots; 0 uses every sample.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config().competition_v10_dog_preset().copy_with(
        use_detection_cache=True,
        detection_cache_dir=args.cache_dir or args.output_dir / "detection_cache",
        use_competition_preset=False,
    )
    names = [n for n in list_datasets(args.train_dir) if find_geff_path(args.train_dir, n)]
    if args.samples_per_embryo:
        selected, counts = [], defaultdict(int)
        for name in names:
            group = embryo_id(name)
            if counts[group] < args.samples_per_embryo:
                selected.append(name)
                counts[group] += 1
        names = selected
    if args.limit:
        names = names[: args.limit]
    samples = []
    inventory = defaultdict(int)
    status_path = args.output_dir / "status.json"
    status_path.write_text(json.dumps({
        "state": "detecting",
        "selected_samples": len(names),
        "completed_samples": 0,
        "sample_names": names,
    }, indent=2))
    for index, name in enumerate(names, 1):
        sample = build_detected_sample(args.train_dir, name, cfg)
        samples.append(sample)
        inventory[sample.embryo] += 1
        status_path.write_text(json.dumps({
            "state": "detecting",
            "selected_samples": len(names),
            "completed_samples": index,
            "last_sample": name,
            "inventory": dict(inventory),
        }, indent=2))
        print("DETECTED", index, "/", len(names), name, flush=True)
    embryos = sorted(inventory)
    if len(embryos) < 2:
        raise ValueError(f"Need at least two embryo groups, found {dict(inventory)}")

    rows, training_manifest = [], []
    for fold_i, heldout in enumerate(embryos):
        status_path.write_text(json.dumps({
            "state": "training",
            "heldout_embryo": heldout,
            "completed_folds": fold_i,
            "inventory": dict(inventory),
        }, indent=2))
        train_samples = [s for s in samples if s.embryo != heldout]
        test_samples = [s for s in samples if s.embryo == heldout]
        pairs = [build_training_pairs(s, cfg) for s in train_samples]
        x = np.concatenate([p[0] for p in pairs if len(p[0])])
        y = np.concatenate([p[1] for p in pairs if len(p[1])])
        checkpoint = train_linker(x, y, args.output_dir / f"fold_{heldout}.pt", args.epochs, args.seed + fold_i)
        training_manifest.append({
            "heldout_embryo": heldout,
            "training_embryos": sorted({s.embryo for s in train_samples}),
            "training_samples": len(train_samples),
            "training_candidates": len(y),
            "positive_edges": int(y.sum()),
            "sample_pair_diagnostics": {s.name: pair[2] for s, pair in zip(train_samples, pairs)},
        })
        classical = cfg.copy_with(linker_backend="classical")
        learned = cfg.copy_with(linker_backend="image_learned_joint", image_linker_path=checkpoint)
        for sample in test_samples:
            for arm, arm_cfg in (("current_hungarian", classical), ("detected_image_joint_linker", learned)):
                result = evaluate_sample(sample, arm_cfg)
                rows.append({"embryo": heldout, "dataset": sample.name, "arm": arm, **result})
                print("EVAL", heldout, sample.name, arm, result, flush=True)

    details = pd.DataFrame(rows)
    details.to_csv(args.output_dir / "details.csv", index=False)
    metrics = ["score", "recall", "edge_precision", "edge_recall", "edge_f1", "division_precision", "division_recall"]
    summary = {"inventory": dict(inventory), "independent_embryos": len(embryos), "metrics": {}, "training": training_manifest}
    for metric in metrics:
        pivot = details.pivot(index=["embryo", "dataset"], columns="arm", values=metric).dropna()
        base = pivot["current_hungarian"].to_numpy()
        candidate = pivot["detected_image_joint_linker"].to_numpy()
        summary["metrics"][metric] = {
            "baseline": bootstrap_ci(base),
            "candidate": bootstrap_ci(candidate),
            "paired_sample_delta": paired_bootstrap_delta(base, candidate),
        }
        embryo_delta = (pivot["detected_image_joint_linker"] - pivot["current_hungarian"]).groupby(level="embryo").mean().to_numpy()
        summary["metrics"][metric]["paired_embryo_delta"] = bootstrap_ci(embryo_delta)
    score_low = summary["metrics"]["score"]["paired_embryo_delta"][1]
    summary["decision"] = "adopt" if score_low > 0 else "reject"
    summary["decision_rule"] = "Adopt only when the embryo-level paired 95% CI lower bound for composite score is > 0."
    summary["limitations"] = [
        "Unmatched detections are retained in assignment but are unknown under sparse GEFF labels, not asserted false positives.",
        f"Only {len(embryos)} embryo prefixes are independent biological groups; volumes are not independent organisms.",
        "This is an offline sparse-label proxy, not an official competition score or submission.",
    ]
    summary["seconds"] = time.time() - started
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    status_path.write_text(json.dumps({
        "state": "completed",
        "decision": summary["decision"],
        "seconds": summary["seconds"],
        "inventory": dict(inventory),
    }, indent=2))
    print("DETECTED LINKER EXPERIMENT COMPLETED", json.dumps(summary, default=float), flush=True)


if __name__ == "__main__":
    main()
