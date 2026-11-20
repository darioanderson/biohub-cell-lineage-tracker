"""Evaluate zero-shot Trackastra on the same real detections as the production linker."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from biohub.config import Config
from biohub.data import find_geff_path, list_datasets, load_volume, read_zarr_meta
from biohub.trackastra_adapter import (
    centroids_to_watershed_mask,
    graph_edges_to_detection_ids,
    make_isotropic_xy,
)
from biohub.tuning import bootstrap_ci, evaluate_sparse_graph, paired_bootstrap_delta
from scripts.evaluate_detected_linker import build_detected_sample, embryo_id, evaluate_sample


def evaluate_trackastra(sample, edges: list[tuple[int, int]]) -> dict[str, float]:
    mapping = {pred: gt for frame in sample.frames for pred, gt in frame.pred_to_gt.items()}
    recalls = [len(frame.pred_to_gt) / frame.gt_count for frame in sample.frames if frame.gt_count]
    graph = evaluate_sparse_graph(
        sample.gt_nodes,
        sample.gt_edges,
        edges,
        mapping,
        evaluable_times=[frame.t for frame in sample.frames],
    )
    recall = float(np.mean(recalls)) if recalls else 0.0
    detections = sum(len(frame.ids) for frame in sample.frames)
    return {
        "recall": recall,
        "detections": float(detections),
        "matched_detections": float(len(mapping)),
        "unmatched_detections_unknown": float(detections - len(mapping)),
        "score": 0.45 * recall + 0.45 * graph["edge_f1"] + 0.10 * graph["division_f1"],
        **graph,
    }


def run_trackastra(model, train_dir: Path, sample, mode: str) -> tuple[list[tuple[int, int]], dict]:
    zarr_path = train_dir / f"{sample.name}.zarr"
    shape, dtype = read_zarr_meta(zarr_path)
    frames = sample.frames
    first_t = frames[0].t
    isotropic_shape = (shape[1], shape[2] // 4, shape[3] // 4)
    images = np.zeros((len(frames), *isotropic_shape), dtype=np.uint16)
    masks = np.zeros_like(images, dtype=np.uint16)
    for local_t, frame in enumerate(frames):
        volume = load_volume(zarr_path, frame.t, shape, dtype)
        isotropic = make_isotropic_xy(volume, factor=4)
        images[local_t] = np.clip(isotropic, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        masks[local_t] = centroids_to_watershed_mask(isotropic, frame.xyz)
    with contextlib.redirect_stderr(io.StringIO()):
        graph, _ = model.track(images, masks, mode=mode)
    edges = graph_edges_to_detection_ids(graph, [frame.ids for frame in frames])
    return edges, {
        "first_evaluable_t": int(first_t),
        "frames": len(frames),
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/trackastra_full"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=("greedy", "ilp", "greedy_nodiv"), default="greedy")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--samples-per-embryo", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from trackastra.model import Trackastra

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    cfg = Config().competition_v10_dog_preset().copy_with(
        use_detection_cache=True,
        detection_cache_dir=args.cache_dir or args.output_dir / "detection_cache",
        use_competition_preset=False,
    )
    names = [name for name in list_datasets(args.train_dir) if find_geff_path(args.train_dir, name)]
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
    inventory = defaultdict(int)
    for name in names:
        inventory[embryo_id(name)] += 1
    if len(inventory) < 2:
        raise ValueError(f"Need at least two embryo groups, found {dict(inventory)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Trackastra.from_folder(args.model_dir, device=device)
    rows, diagnostics = [], {}
    classical = cfg.copy_with(linker_backend="classical")
    for index, name in enumerate(names, 1):
        status_path.write_text(json.dumps({
            "state": "evaluating", "completed_samples": index - 1,
            "selected_samples": len(names), "next_sample": name,
            "inventory": dict(inventory), "device": device,
        }, indent=2))
        sample = build_detected_sample(args.train_dir, name, cfg)
        baseline = evaluate_sample(sample, classical)
        candidate_edges, trackastra_info = run_trackastra(model, args.train_dir, sample, args.mode)
        candidate = evaluate_trackastra(sample, candidate_edges)
        diagnostics[name] = trackastra_info
        for arm, result in (("current_hungarian", baseline), ("trackastra_ctc", candidate)):
            rows.append({"embryo": sample.embryo, "dataset": name, "arm": arm, **result})
        status_path.write_text(json.dumps({
            "state": "evaluating", "completed_samples": index,
            "selected_samples": len(names), "last_sample": name,
            "inventory": dict(inventory), "device": device,
        }, indent=2))
        print("EVAL", index, "/", len(names), name, baseline["score"], candidate["score"], flush=True)

    details = pd.DataFrame(rows)
    details.to_csv(args.output_dir / "details.csv", index=False)
    metrics = ["score", "recall", "edge_precision", "edge_recall", "edge_f1", "division_precision", "division_recall"]
    summary = {
        "inventory": dict(inventory), "independent_embryos": len(inventory),
        "model": "Trackastra ctc (zero-shot)", "mode": args.mode,
        "model_dir": str(args.model_dir), "device": device,
        "metrics": {}, "sample_diagnostics": diagnostics,
    }
    for metric in metrics:
        pivot = details.pivot(index=["embryo", "dataset"], columns="arm", values=metric).dropna()
        base = pivot["current_hungarian"].to_numpy()
        candidate = pivot["trackastra_ctc"].to_numpy()
        summary["metrics"][metric] = {
            "baseline": bootstrap_ci(base),
            "candidate": bootstrap_ci(candidate),
            "paired_sample_delta": paired_bootstrap_delta(base, candidate),
        }
        embryo_delta = (pivot["trackastra_ctc"] - pivot["current_hungarian"]).groupby(level="embryo").mean().to_numpy()
        summary["metrics"][metric]["paired_embryo_delta"] = bootstrap_ci(embryo_delta)
    summary["decision_rule"] = "Adopt only when the embryo-level paired 95% CI lower bound for composite score is > 0."
    summary["decision"] = "adopt" if summary["metrics"]["score"]["paired_embryo_delta"][1] > 0 else "reject"
    summary["limitations"] = [
        "Trackastra ctc is externally pretrained and is not fine-tuned on Biohub.",
        "Only two embryo prefixes are independent biological groups.",
        "The comparison uses identical DoG detections; unmatched sparse detections remain unknown.",
        "This is an offline sparse-label proxy, not an official competition submission score.",
    ]
    summary["seconds"] = time.time() - started
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    status_path.write_text(json.dumps({
        "state": "completed", "decision": summary["decision"],
        "seconds": summary["seconds"], "inventory": dict(inventory),
    }, indent=2))
    print("TRACKASTRA EXPERIMENT COMPLETED", json.dumps(summary, default=float), flush=True)


if __name__ == "__main__":
    main()
