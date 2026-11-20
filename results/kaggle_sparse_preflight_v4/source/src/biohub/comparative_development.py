"""Comparative developmental analysis built on validated lineage graphs.

This module deliberately separates observed graph events from biological labels.
A disappearing track is an unresolved termination until an external annotation
or assay establishes apoptosis, and cross-embryo comparisons require canonical
cell identities or an explicit validated correspondence table.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def _require_columns(table: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{name} lacks required columns: {sorted(missing)}")


def division_phenotypes(
    observations: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    voxel_size_zyx_um: Sequence[float] = (1.0, 1.0, 1.0),
    embryo_axis_zyx: Sequence[float] = (0.0, 0.0, 1.0),
    identity_column: str | None = None,
) -> pd.DataFrame:
    """Measure joint two-daughter geometry and intensity/size asymmetry."""
    _require_columns(observations, ["node_id", "t", "z", "y", "x"], "observations")
    if edges.empty and not {"source_id", "target_id"}.issubset(edges.columns):
        edges = pd.DataFrame(columns=["source_id", "target_id"])
    _require_columns(edges, ["source_id", "target_id"], "edges")
    if identity_column is not None:
        _require_columns(observations, [identity_column], "observations")
    scale = np.asarray(voxel_size_zyx_um, dtype=float)
    axis = np.asarray(embryo_axis_zyx, dtype=float)
    if scale.shape != (3,) or np.any(scale <= 0):
        raise ValueError("voxel_size_zyx_um must contain three positive values")
    if axis.shape != (3,) or not np.linalg.norm(axis):
        raise ValueError("embryo_axis_zyx must be a non-zero three-vector")
    axis /= np.linalg.norm(axis)

    by_node = observations.set_index("node_id", drop=False)
    rows: list[dict] = []
    for source, targets in edges.groupby("source_id", sort=True)["target_id"]:
        daughters = sorted({int(target) for target in targets})
        if len(daughters) != 2 or int(source) not in by_node.index:
            continue
        if any(daughter not in by_node.index for daughter in daughters):
            continue
        parent = by_node.loc[int(source)]
        first, second = (by_node.loc[daughter] for daughter in daughters)
        first_xyz = first[["z", "y", "x"]].to_numpy(float) * scale
        second_xyz = second[["z", "y", "x"]].to_numpy(float) * scale
        division_axis = second_xyz - first_xyz
        separation = float(np.linalg.norm(division_axis))
        orientation = np.nan
        if separation:
            # A spindle axis is undirected, so 0 and 180 degrees are equivalent.
            orientation = float(np.degrees(np.arccos(np.clip(abs(np.dot(division_axis / separation, axis)), 0, 1))))
        row = {
            "parent_node_id": int(source),
            "daughter_a_node_id": daughters[0],
            "daughter_b_node_id": daughters[1],
            "t": int(parent["t"]),
            "daughter_separation_um": separation,
            "midpoint_residual_um": float(
                np.linalg.norm(0.5 * (first_xyz + second_xyz) - parent[["z", "y", "x"]].to_numpy(float) * scale)
            ),
            "division_axis_z": float(division_axis[0]),
            "division_axis_y": float(division_axis[1]),
            "division_axis_x": float(division_axis[2]),
            "orientation_to_embryo_axis_deg": orientation,
        }
        if identity_column is not None:
            row["canonical_parent_id"] = parent[identity_column]
        for value_column, output_prefix in (("intensity", "intensity"), ("volume_um3", "size")):
            if value_column in observations:
                a, b = float(first[value_column]), float(second[value_column])
                total = abs(a) + abs(b)
                row[f"daughter_{output_prefix}_ratio"] = max(abs(a), abs(b)) / max(min(abs(a), abs(b)), 1e-12)
                row[f"daughter_{output_prefix}_asymmetry"] = abs(a - b) / total if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def classify_track_terminations(
    observations: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    final_t: int | None = None,
    identity_column: str | None = None,
    annotated_deaths: Iterable[str] = (),
) -> pd.DataFrame:
    """Classify track ends without equating disappearance with cell death."""
    _require_columns(observations, ["node_id", "track_id", "t"], "observations")
    if edges.empty and not {"source_id", "target_id"}.issubset(edges.columns):
        edges = pd.DataFrame(columns=["source_id", "target_id"])
    _require_columns(edges, ["source_id", "target_id"], "edges")
    if identity_column is not None:
        _require_columns(observations, [identity_column], "observations")
    if final_t is None:
        final_t = int(observations["t"].max())
    known = {str(value) for value in annotated_deaths}
    children: dict[int, list[int]] = defaultdict(list)
    for source, target in edges[["source_id", "target_id"]].itertuples(index=False):
        children[int(source)].append(int(target))

    rows = []
    for track_id, group in observations.groupby("track_id", sort=True):
        last = group.sort_values(["t", "node_id"]).iloc[-1]
        node_id, end_t = int(last["node_id"]), int(last["t"])
        n_children = len(set(children.get(node_id, [])))
        identity = str(last[identity_column]) if identity_column is not None else None
        if n_children >= 2:
            classification = "division"
        elif n_children == 1:
            classification = "linked_track_boundary"
        elif end_t >= final_t:
            classification = "right_censored"
        elif identity is not None and identity in known:
            classification = "annotated_death_candidate"
        else:
            classification = "unresolved_termination"
        rows.append({
            "track_id": int(track_id),
            "last_node_id": node_id,
            "end_t": end_t,
            "classification": classification,
            "requires_manual_or_assay_confirmation": classification in {
                "annotated_death_candidate", "unresolved_termination"
            },
            **({"canonical_cell_id": identity} if identity_column is not None else {}),
        })
    return pd.DataFrame(rows)


def build_interdivision_atlas(
    division_events: pd.DataFrame,
    *,
    identity_column: str = "canonical_parent_id",
    time_column: str = "time_s",
) -> pd.DataFrame:
    """Build a robust normal-reference atlas from canonically identified cells."""
    _require_columns(division_events, [identity_column, time_column], "division_events")
    if division_events[identity_column].isna().any():
        raise ValueError("canonical identities cannot be missing from an atlas")
    rows = []
    for identity, group in division_events.groupby(identity_column, sort=True):
        values = group[time_column].to_numpy(float)
        median = float(np.median(values))
        rows.append({
            "canonical_parent_id": identity,
            "n_control_embryos": int(len(values)),
            "median_division_time_s": median,
            "mad_division_time_s": float(np.median(np.abs(values - median))),
            "q05_division_time_s": float(np.quantile(values, 0.05)),
            "q95_division_time_s": float(np.quantile(values, 0.95)),
        })
    return pd.DataFrame(rows)


def build_canonical_event_table(
    observations: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    frame_interval_s: float,
    identity_column: str = "canonical_cell_id",
    annotated_deaths: Iterable[str] = (),
) -> pd.DataFrame:
    """Create comparable division/termination events from named cells."""
    if frame_interval_s <= 0:
        raise ValueError("frame_interval_s must be positive")
    _require_columns(observations, ["node_id", "track_id", "t", identity_column], "observations")
    if observations[identity_column].isna().any():
        raise ValueError("canonical identities cannot be missing from an event table")
    if edges.empty and not {"source_id", "target_id"}.issubset(edges.columns):
        edges = pd.DataFrame(columns=["source_id", "target_id"])
    _require_columns(edges, ["source_id", "target_id"], "edges")
    by_node = observations.set_index("node_id")
    rows: list[dict] = []
    for source, targets in edges.groupby("source_id")["target_id"]:
        if len(set(int(target) for target in targets)) >= 2 and int(source) in by_node.index:
            parent = by_node.loc[int(source)]
            rows.append({
                "canonical_cell_id": str(parent[identity_column]),
                "event_type": "division",
                "time_s": float(parent["t"] * frame_interval_s),
                "evidence_status": "graph_candidate",
            })
    ends = classify_track_terminations(
        observations, edges, identity_column=identity_column,
        annotated_deaths=annotated_deaths,
    )
    for end in ends.itertuples(index=False):
        if end.classification not in {"annotated_death_candidate", "unresolved_termination"}:
            continue
        rows.append({
            "canonical_cell_id": end.canonical_cell_id,
            "event_type": (
                "death_candidate" if end.classification == "annotated_death_candidate" else "termination"
            ),
            "time_s": float(end.end_t * frame_interval_s),
            "evidence_status": end.classification,
        })
    result = pd.DataFrame(rows, columns=[
        "canonical_cell_id", "event_type", "time_s", "evidence_status"
    ])
    if result.duplicated(["canonical_cell_id", "event_type"]).any():
        raise ValueError("canonical event table contains duplicate cell/event pairs")
    return result.sort_values(["time_s", "canonical_cell_id", "event_type"], ignore_index=True)


def compare_lineage_events(
    reference_events: pd.DataFrame,
    observed_events: pd.DataFrame,
    *,
    identity_column: str = "canonical_cell_id",
    event_column: str = "event_type",
    time_column: str = "time_s",
    time_tolerance_s: float = 0.0,
) -> pd.DataFrame:
    """Report missing, unexpected, and time-shifted events by biological identity."""
    if time_tolerance_s < 0:
        raise ValueError("time_tolerance_s must be non-negative")
    required = [identity_column, event_column, time_column]
    _require_columns(reference_events, required, "reference_events")
    _require_columns(observed_events, required, "observed_events")
    keys = [identity_column, event_column]
    if reference_events.duplicated(keys).any() or observed_events.duplicated(keys).any():
        raise ValueError("each canonical cell/event pair must be unique")
    merged = reference_events[keys + [time_column]].merge(
        observed_events[keys + [time_column]], on=keys, how="outer",
        suffixes=("_reference", "_observed"), indicator=True,
    )
    ref_time = f"{time_column}_reference"
    obs_time = f"{time_column}_observed"
    rows = []
    for _, row in merged.iterrows():
        values = row.to_dict()
        presence = values["_merge"]
        reference_time = values.get(ref_time, np.nan)
        observed_time = values.get(obs_time, np.nan)
        if presence == "left_only":
            status, delta, first_time = "missing_observed_event", np.nan, float(reference_time)
        elif presence == "right_only":
            status, delta, first_time = "unexpected_observed_event", np.nan, float(observed_time)
        else:
            delta = float(observed_time - reference_time)
            status = "within_tolerance" if abs(delta) <= time_tolerance_s else "timing_shift"
            first_time = float(min(reference_time, observed_time))
        if status != "within_tolerance":
            rows.append({
                "canonical_cell_id": values[identity_column],
                "event_type": values[event_column],
                "reference_time_s": reference_time,
                "observed_time_s": observed_time,
                "time_delta_s": delta,
                "divergence_type": status,
                "first_divergence_time_s": first_time,
            })
    return pd.DataFrame(rows).sort_values(
        ["first_divergence_time_s", "canonical_cell_id", "event_type"], ignore_index=True
    ) if rows else pd.DataFrame(columns=[
        "canonical_cell_id", "event_type", "reference_time_s", "observed_time_s",
        "time_delta_s", "divergence_type", "first_divergence_time_s",
    ])


def add_local_density(
    observations: pd.DataFrame,
    *,
    radius_um: float,
    coordinate_columns: Sequence[str] = ("z_um", "y_um", "x_um"),
) -> pd.DataFrame:
    """Add same-frame neighbor counts for density-stratified accuracy analysis."""
    if radius_um <= 0:
        raise ValueError("radius_um must be positive")
    _require_columns(observations, ["t", *coordinate_columns], "observations")
    result = observations.copy()
    result["neighbors_within_radius"] = 0
    for _, indices in result.groupby("t", sort=False).groups.items():
        positions = result.loc[indices, list(coordinate_columns)].to_numpy(float)
        tree = cKDTree(positions)
        counts = np.asarray([len(matches) - 1 for matches in tree.query_ball_point(positions, radius_um)])
        result.loc[indices, "neighbors_within_radius"] = counts
    result["density_radius_um"] = float(radius_um)
    return result
