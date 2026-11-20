"""Scientist-facing observation, track, division, lineage, and QC tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from biohub.analysis import AnalysisResult
from biohub.config import Config
from biohub.manifest import write_run_manifest


def build_scientific_tables(result: AnalysisResult, cfg: Config) -> Dict[str, pd.DataFrame]:
    observations = result.nodes.copy()
    required = {"node_id", "track_id", "lineage_id", "t", "z", "y", "x"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"AnalysisResult lacks scientific identity columns: {sorted(missing)}")
    scale = cfg.scale_array
    observations["time_s"] = observations["t"].astype(float) * cfg.frame_interval_s
    observations[["z_um", "y_um", "x_um"]] = observations[["z", "y", "x"]].astype(float) * scale

    track_rows = []
    for track_id, group in observations.sort_values("t").groupby("track_id", sort=True):
        xyz = group[["z_um", "y_um", "x_um"]].to_numpy(float)
        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1) if len(xyz) > 1 else np.array([])
        displacement = float(np.linalg.norm(xyz[-1] - xyz[0])) if len(xyz) else 0.0
        path_length = float(steps.sum())
        elapsed = float((group["t"].max() - group["t"].min()) * cfg.frame_interval_s)
        track_rows.append({
            "track_id": int(track_id),
            "lineage_id": int(group["lineage_id"].iloc[0]),
            "start_t": int(group["t"].min()),
            "end_t": int(group["t"].max()),
            "observations": int(len(group)),
            "lifetime_s": elapsed,
            "displacement_um": displacement,
            "path_length_um": path_length,
            "mean_speed_um_s": path_length / elapsed if elapsed > 0 else 0.0,
            "persistence": displacement / path_length if path_length > 0 else 0.0,
        })
    tracks = pd.DataFrame(track_rows)

    by_node = observations.set_index("node_id")
    division_rows = []
    if not result.edges.empty:
        for source, daughters in result.edges.groupby("source_id")["target_id"].apply(list).items():
            if len(daughters) < 2 or int(source) not in by_node.index:
                continue
            parent = by_node.loc[int(source)]
            daughter_tracks = sorted({int(by_node.loc[int(d), "track_id"]) for d in daughters if int(d) in by_node.index})
            division_rows.append({
                "parent_node_id": int(source),
                "parent_track_id": int(parent["track_id"]),
                "lineage_id": int(parent["lineage_id"]),
                "t": int(parent["t"]),
                "time_s": float(parent["t"] * cfg.frame_interval_s),
                "daughter_track_ids": ";".join(map(str, daughter_tracks)),
                "n_daughters": len(daughter_tracks),
            })
    divisions = pd.DataFrame(division_rows)

    lineage_rows = []
    for lineage_id, group in observations.groupby("lineage_id", sort=True):
        lineage_rows.append({
            "lineage_id": int(lineage_id),
            "tracks": int(group["track_id"].nunique()),
            "observations": int(len(group)),
            "start_t": int(group["t"].min()),
            "end_t": int(group["t"].max()),
            "divisions": int((divisions["lineage_id"] == lineage_id).sum()) if len(divisions) else 0,
        })
    lineages = pd.DataFrame(lineage_rows)
    qc = observations.loc[
        observations.get("qc_status", pd.Series("pass", index=observations.index)).ne("pass"),
        [c for c in ["node_id", "track_id", "lineage_id", "t", "qc_status", "filter_reason"] if c in observations],
    ].copy()
    return {"observations": observations, "tracks": tracks, "divisions": divisions, "lineages": lineages, "qc_flags": qc}


def export_scientific_bundle(
    result: AnalysisResult,
    cfg: Config,
    output_dir: Path,
    input_paths=(),
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_scientific_tables(result, cfg)
    paths: Dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    metrics = output_dir / "metrics.json"
    metrics.write_text(json.dumps(result.stats, indent=2, default=str))
    paths["metrics"] = metrics
    manifest = write_run_manifest(output_dir / "run_manifest.json", cfg, input_paths)
    paths["run_manifest"] = manifest
    return paths
