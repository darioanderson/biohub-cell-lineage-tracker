"""Audited manual corrections for lineage graphs."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Iterable, Mapping

import pandas as pd

from biohub.analysis import AnalysisResult
from biohub.tracking import annotate_lineage_ids, graph_invariant_errors


AUDIT_COLUMNS = [
    "timestamp_utc", "reviewer", "action", "node_id", "source_id", "target_id", "reason"
]


def _integer(value, name: str) -> int:
    if value is None or pd.isna(value):
        raise ValueError(f"{name} is required")
    return int(value)


def apply_curation(
    result: AnalysisResult,
    actions: Iterable[Mapping[str, object]],
) -> tuple[AnalysisResult, pd.DataFrame]:
    """Apply reviewed graph edits to a copy and return an immutable audit table.

    Supported actions are ``approve_node``, ``reject_node``, ``approve_edge``,
    ``reject_edge``, and ``relink``. Approval records review without changing
    topology. Rejections and relinks are materialized in the returned copy; the
    original ``AnalysisResult`` is never mutated.
    """
    nodes = result.nodes.copy(deep=True)
    edges = result.edges.copy(deep=True)
    node_ids = set(nodes["node_id"].astype(int)) if not nodes.empty else set()
    records: list[dict] = []

    for raw in actions:
        action = str(raw.get("action", "")).strip().lower()
        reviewer = str(raw.get("reviewer", "")).strip() or "anonymous"
        reason = str(raw.get("reason", "")).strip()
        node_id = raw.get("node_id")
        source_id = raw.get("source_id")
        target_id = raw.get("target_id")

        if action in {"approve_node", "reject_node"}:
            node_id = _integer(node_id, "node_id")
            if node_id not in node_ids:
                raise ValueError(f"Unknown node_id {node_id}")
            if action == "reject_node":
                nodes = nodes[nodes["node_id"].astype(int) != node_id].copy()
                if not edges.empty:
                    edges = edges[
                        (edges["source_id"].astype(int) != node_id)
                        & (edges["target_id"].astype(int) != node_id)
                    ].copy()
                node_ids.remove(node_id)
        elif action in {"approve_edge", "reject_edge", "relink"}:
            source_id = _integer(source_id, "source_id")
            target_id = _integer(target_id, "target_id")
            if source_id not in node_ids or target_id not in node_ids:
                raise ValueError(f"Unknown edge endpoint {source_id}->{target_id}")
            pair = (
                (edges["source_id"].astype(int) == source_id)
                & (edges["target_id"].astype(int) == target_id)
                if not edges.empty else pd.Series(dtype=bool)
            )
            if action == "approve_edge" and (edges.empty or not pair.any()):
                raise ValueError(f"Edge {source_id}->{target_id} does not exist")
            if action == "reject_edge":
                if edges.empty or not pair.any():
                    raise ValueError(f"Edge {source_id}->{target_id} does not exist")
                edges = edges[~pair].copy()
            if action == "relink":
                node_time = nodes.set_index("node_id")["t"].astype(int)
                if node_time.loc[target_id] <= node_time.loc[source_id]:
                    raise ValueError("A relink must point forward in time")
                if not edges.empty:
                    edges = edges[edges["target_id"].astype(int) != target_id].copy()
                template = {column: -1 for column in edges.columns}
                template.update({
                    "dataset": result.dataset_name,
                    "row_type": "edge",
                    "source_id": source_id,
                    "target_id": target_id,
                })
                edges = pd.concat([edges, pd.DataFrame([template])], ignore_index=True)
        else:
            raise ValueError(f"Unsupported curation action: {action!r}")

        records.append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reviewer": reviewer,
            "action": action,
            "node_id": int(node_id) if node_id is not None and not pd.isna(node_id) else None,
            "source_id": int(source_id) if source_id is not None and not pd.isna(source_id) else None,
            "target_id": int(target_id) if target_id is not None and not pd.isna(target_id) else None,
            "reason": reason,
        })

    node_rows = nodes.to_dict("records")
    edge_rows = edges.to_dict("records")
    errors = graph_invariant_errors(node_rows, edge_rows)
    if errors:
        raise ValueError("Curated graph is invalid: " + "; ".join(errors))
    annotate_lineage_ids(node_rows, edge_rows)
    nodes = pd.DataFrame(node_rows)
    edges = pd.DataFrame(edge_rows, columns=result.edges.columns if len(result.edges.columns) else None)
    stats = dict(result.stats)
    stats.update({
        "nodes": len(nodes),
        "edges": len(edges),
        "divisions": int((edges.groupby("source_id").size() >= 2).sum()) if not edges.empty else 0,
        "curation_actions": len(records),
    })
    curated = replace(result, nodes=nodes, edges=edges, stats=stats)
    return curated, pd.DataFrame(records, columns=AUDIT_COLUMNS)

