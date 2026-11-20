"""Reproducible dense-versus-sparse linker profile on identical coordinates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub.config import Config
from biohub.tracking import build_link_cost, link_frames, neighbor_signature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=800)
    parser.add_argument("--output", type=Path, default=Path("results/linking_profile.json"))
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    prev = rng.uniform(0, 400, size=(args.cells, 3))
    curr = prev + rng.normal(0, 0.4, size=prev.shape)
    cfg = Config(voxel_size_zyx_um=(1, 1, 1), max_link_dist_um=3, detect_divisions=False)

    start = time.perf_counter()
    pp, cp = prev * cfg.scale_array, curr * cfg.scale_array
    ps, cs = neighbor_signature(pp, cfg), neighbor_signature(cp, cfg)
    dense_cost, _ = build_link_cost(pp, cp, cfg, prev_sig=ps, curr_sig=cs)
    rows, cols = linear_sum_assignment(dense_cost)
    dense_edges = {(int(r), int(c)) for r, c in zip(rows, cols) if dense_cost[r, c] < 1e6}
    dense_seconds = time.perf_counter() - start

    start = time.perf_counter()
    sparse_edges = set(link_frames(range(args.cells), prev, range(args.cells), curr, cfg))
    sparse_seconds = time.perf_counter() - start
    if sparse_edges != dense_edges:
        raise AssertionError("Sparse assignments differ from dense reference")

    signature_width = cfg.neighborhood_k * 3
    report = {
        "cells_per_frame": args.cells,
        "matched_edges": len(sparse_edges),
        "dense_seconds": dense_seconds,
        "sparse_seconds": sparse_seconds,
        "speedup": dense_seconds / sparse_seconds,
        "dense_distance_and_cost_bytes": args.cells * args.cells * 8 * 2,
        "old_signature_diff_bytes": args.cells * args.cells * signature_width * 8,
        "note": "Byte counts are exact array-size estimates; process RSS was not inferred.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
