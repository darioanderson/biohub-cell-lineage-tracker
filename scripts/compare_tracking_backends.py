"""Compare framewise and global linkers on identical exact synthetic observations.

This is a deterministic engineering benchmark, not a substitute for held-out
real-data evaluation. It isolates tracking from detector errors and writes the
measured accuracy and runtime for every controlled scenario.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from biohub.config import Config
from biohub.sample_data import SYNTHETIC_SCENARIOS, synthetic_ground_truth
from biohub.tracking import link_frames, link_sequence_global


def _metrics(predicted, truth):
    predicted, truth = set(predicted), set(truth)
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else float(not truth)
    recall = tp / len(truth) if truth else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"edge_precision": precision, "edge_recall": recall, "edge_f1": f1}


def _frames(nodes, n_frames):
    ids, xyz, intensities = [], [], []
    for frame in range(n_frames):
        subset = nodes[nodes["t"].eq(frame)].sort_values("node_id")
        ids.append(subset["node_id"].astype(int).tolist())
        xyz.append(subset[["z", "y", "x"]].to_numpy(dtype=np.float64))
        intensities.append(np.ones(len(subset), dtype=np.float64))
    return ids, xyz, intensities


def compare(output: Path, frames: int = 8) -> dict:
    cfg = Config(
        voxel_size_zyx_um=(1.0, 1.0, 1.0),
        max_link_dist_um=5.0,
        global_max_frame_gap=2,
        detect_divisions=True,
        div_min_count_gain=1,
        div_parent_dist_um=8.0,
        div_sister_dist_um=12.0,
        div_midpoint_dist_um=5.0,
    )
    rows = []
    for scenario in SYNTHETIC_SCENARIOS:
        gt = synthetic_ground_truth(scenario, t=frames)
        ids, xyz, intensity = _frames(gt.nodes, frames)
        truth = set(map(tuple, gt.edges[["source_id", "target_id"]].to_numpy(dtype=int)))
        for backend in ("framewise", "global"):
            start = time.perf_counter()
            if backend == "framewise":
                predicted = []
                for frame in range(1, frames):
                    predicted.extend(link_frames(
                        ids[frame - 1], xyz[frame - 1], ids[frame], xyz[frame], cfg,
                        prev_intensity=intensity[frame - 1], curr_intensity=intensity[frame],
                    ))
            else:
                predicted = link_sequence_global(ids, xyz, intensity, cfg.copy_with(tracking_backend="global"))
            elapsed_ms = 1000.0 * (time.perf_counter() - start)
            rows.append({
                "scenario": scenario,
                "backend": backend,
                **_metrics(predicted, truth),
                "predicted_edges": len(predicted),
                "truth_edges": len(truth),
                "runtime_ms": elapsed_ms,
            })
    summary = {
        backend: {
            "mean_edge_f1": float(np.mean([r["edge_f1"] for r in rows if r["backend"] == backend])),
            "total_runtime_ms": float(sum(r["runtime_ms"] for r in rows if r["backend"] == backend)),
        }
        for backend in ("framewise", "global")
    }
    payload = {
        "scope": "exact synthetic observations; detector excluded; not real-data evidence",
        "rows": rows,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/tracking_backend_comparison.json"))
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(compare(args.output, args.frames)["summary"], indent=2))
