"""Scientist-facing lineage navigation helpers.

The tracker stores one ``node_id`` per observation, while ``track_id`` and
``lineage_id`` carry biological continuity.  This module converts the raw graph
into relationships that are easier to inspect in a UI without changing the
underlying tracking result.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import pandas as pd

from biohub.analysis import AnalysisResult


def _graph_maps(result: AnalysisResult) -> tuple[dict[int, int], dict[int, list[int]]]:
    """Return target->parent and source->children maps in deterministic order."""
    parent: dict[int, int] = {}
    children: dict[int, list[int]] = defaultdict(list)
    if not result.edges.empty:
        for row in result.edges.itertuples(index=False):
            source = int(row.source_id)
            target = int(row.target_id)
            parent[target] = source
            children[source].append(target)

    if not result.nodes.empty:
        node_time = result.nodes.set_index("node_id")["t"].astype(int).to_dict()
        for source in list(children):
            children[source] = sorted(
                set(children[source]), key=lambda nid: (node_time.get(nid, 10**9), nid)
            )
    return parent, dict(children)


def _node_row(result: AnalysisResult, node_id: int) -> pd.Series:
    rows = result.nodes.loc[result.nodes["node_id"].astype(int).eq(int(node_id))]
    if rows.empty:
        raise ValueError(f"Unknown node_id {node_id}")
    return rows.iloc[0]


def node_event_type(result: AnalysisResult, node_id: int) -> str:
    """Return a concise topology-based event label for one observation."""
    parent, children = _graph_maps(result)
    node_id = int(node_id)
    n_children = len(children.get(node_id, []))
    source = parent.get(node_id)

    if n_children >= 2:
        return "division_parent"
    if source is not None and len(children.get(source, [])) >= 2:
        return "daughter"
    if source is None and n_children == 0:
        return "isolated"
    if source is None:
        return "lineage_root"
    if n_children == 0:
        return "terminal"
    return "continuation"


def node_relationships(result: AnalysisResult, node_id: int) -> dict[str, object]:
    """Describe the graph and track context surrounding one observation.

    Ancestors are ordered root -> direct parent.  Descendants are ordered by
    frame then node ID.  The function deliberately reports graph facts rather
    than interpreting a terminal observation as death/apoptosis.
    """
    node_id = int(node_id)
    row = _node_row(result, node_id)
    parent, children = _graph_maps(result)
    nodes = result.nodes.set_index("node_id", drop=False)

    ancestors_near_to_far: list[int] = []
    cur = parent.get(node_id)
    visited: set[int] = set()
    while cur is not None and cur not in visited:
        visited.add(cur)
        ancestors_near_to_far.append(int(cur))
        cur = parent.get(int(cur))
    ancestors = list(reversed(ancestors_near_to_far))

    descendants: list[int] = []
    queue: deque[int] = deque(children.get(node_id, []))
    seen: set[int] = set()
    while queue:
        current = int(queue.popleft())
        if current in seen:
            continue
        seen.add(current)
        descendants.append(current)
        queue.extend(children.get(current, []))
    descendants.sort(
        key=lambda nid: (
            int(nodes.loc[nid, "t"]) if nid in nodes.index else 10**9,
            nid,
        )
    )

    direct_parent = parent.get(node_id)
    siblings = (
        [nid for nid in children.get(int(direct_parent), []) if int(nid) != node_id]
        if direct_parent is not None
        else []
    )
    direct_children = list(children.get(node_id, []))
    leaves = [nid for nid in descendants if not children.get(int(nid), [])]

    track_id = int(row["track_id"]) if "track_id" in row and pd.notna(row["track_id"]) else None
    lineage_id = int(row["lineage_id"]) if "lineage_id" in row and pd.notna(row["lineage_id"]) else None

    previous_track_node = None
    next_track_node = None
    if track_id is not None and "track_id" in result.nodes.columns:
        track = result.nodes[result.nodes["track_id"].astype(int).eq(track_id)].sort_values(["t", "node_id"])
        ids = track["node_id"].astype(int).tolist()
        if node_id in ids:
            idx = ids.index(node_id)
            previous_track_node = ids[idx - 1] if idx > 0 else None
            next_track_node = ids[idx + 1] if idx + 1 < len(ids) else None

    child_track_ids: list[int] = []
    if "track_id" in result.nodes.columns:
        for child in direct_children:
            if child in nodes.index:
                child_track_ids.append(int(nodes.loc[child, "track_id"]))
        child_track_ids = sorted(set(child_track_ids))

    parent_track_id = None
    if direct_parent is not None and "track_id" in result.nodes.columns and direct_parent in nodes.index:
        parent_track_id = int(nodes.loc[direct_parent, "track_id"])

    return {
        "node_id": node_id,
        "track_id": track_id,
        "lineage_id": lineage_id,
        "generation": int(row["generation"]) if "generation" in row and pd.notna(row["generation"]) else None,
        "event_type": node_event_type(result, node_id),
        "parent_node_id": int(direct_parent) if direct_parent is not None else None,
        "parent_track_id": parent_track_id,
        "children": [int(x) for x in direct_children],
        "child_track_ids": child_track_ids,
        "siblings": [int(x) for x in siblings],
        "ancestors": [int(x) for x in ancestors],
        "descendants": [int(x) for x in descendants],
        "leaf_descendants": [int(x) for x in leaves],
        "root_node_id": int(ancestors[0]) if ancestors else node_id,
        "previous_track_node": previous_track_node,
        "next_track_node": next_track_node,
    }


def local_family_node_ids(
    result: AnalysisResult,
    node_id: int,
    ancestor_depth: int = 3,
    descendant_depth: int = 3,
    include_siblings: bool = True,
) -> set[int]:
    """Return a bounded local family neighborhood around ``node_id``."""
    node_id = int(node_id)
    _node_row(result, node_id)
    parent, children = _graph_maps(result)
    chosen: set[int] = {node_id}

    cur = node_id
    for _ in range(max(0, int(ancestor_depth))):
        source = parent.get(cur)
        if source is None:
            break
        source = int(source)
        chosen.add(source)
        if include_siblings:
            chosen.update(int(x) for x in children.get(source, []) if int(x) != cur)
        cur = source

    frontier = {node_id}
    for _ in range(max(0, int(descendant_depth))):
        next_frontier: set[int] = set()
        for source in frontier:
            direct = {int(x) for x in children.get(int(source), [])}
            chosen.update(direct)
            next_frontier.update(direct)
        frontier = next_frontier
        if not frontier:
            break

    if include_siblings:
        source = parent.get(node_id)
        if source is not None:
            chosen.update(int(x) for x in children.get(int(source), []))
    return chosen


def lineage_summary(result: AnalysisResult, lineage_id: int) -> dict[str, int]:
    """Return compact counts for one lineage."""
    if "lineage_id" not in result.nodes.columns:
        raise ValueError("AnalysisResult has no lineage_id column")
    sub = result.nodes[result.nodes["lineage_id"].astype(int).eq(int(lineage_id))].copy()
    if sub.empty:
        raise ValueError(f"Unknown lineage_id {lineage_id}")
    ids = set(sub["node_id"].astype(int))
    parent, children = _graph_maps(result)
    divisions = sum(len([c for c in children.get(nid, []) if c in ids]) >= 2 for nid in ids)
    terminals = sum(len([c for c in children.get(nid, []) if c in ids]) == 0 for nid in ids)
    roots = sum(parent.get(nid) not in ids for nid in ids)
    qc_flagged = 0
    if "qc_status" in sub.columns:
        qc_flagged = int(sub["qc_status"].fillna("pass").astype(str).ne("pass").sum())
    return {
        "lineage_id": int(lineage_id),
        "observations": int(len(sub)),
        "tracks": int(sub["track_id"].nunique()) if "track_id" in sub.columns else 0,
        "divisions": int(divisions),
        "terminals": int(terminals),
        "roots": int(roots),
        "start_t": int(sub["t"].min()),
        "end_t": int(sub["t"].max()),
        "qc_flagged": qc_flagged,
    }


def lineage_event_table(result: AnalysisResult, lineage_id: int) -> pd.DataFrame:
    """Return one readable row per observation in a selected lineage."""
    if "lineage_id" not in result.nodes.columns:
        raise ValueError("AnalysisResult has no lineage_id column")
    sub = result.nodes[result.nodes["lineage_id"].astype(int).eq(int(lineage_id))].copy()
    if sub.empty:
        return pd.DataFrame()
    parent, children = _graph_maps(result)
    rows: list[dict[str, object]] = []
    for row in sub.sort_values(["t", "generation", "track_id", "node_id"]).itertuples(index=False):
        nid = int(row.node_id)
        rows.append({
            "frame": int(row.t),
            "node_id": nid,
            "track_id": int(row.track_id) if hasattr(row, "track_id") else None,
            "generation": int(row.generation) if hasattr(row, "generation") else None,
            "event": node_event_type(result, nid),
            "parent": parent.get(nid),
            "children": ", ".join(str(x) for x in children.get(nid, [])) or "—",
            "qc_status": getattr(row, "qc_status", "pass"),
        })
    return pd.DataFrame(rows)


def relationship_table(result: AnalysisResult, node_id: int) -> pd.DataFrame:
    """Return a compact relationship list for display/export."""
    rel = node_relationships(result, node_id)
    rows: list[dict[str, object]] = []

    def add(relation: str, values: object) -> None:
        if values is None:
            rows.append({"relation": relation, "node_ids": "—"})
        elif isinstance(values, (list, tuple, set)):
            rows.append({"relation": relation, "node_ids": ", ".join(map(str, values)) or "—"})
        else:
            rows.append({"relation": relation, "node_ids": str(values)})

    add("Lineage root", rel["root_node_id"])
    add("Ancestors (root → parent)", rel["ancestors"])
    add("Parent", rel["parent_node_id"])
    add("Siblings", rel["siblings"])
    add("Children", rel["children"])
    add("Descendants", rel["descendants"])
    add("Previous observation in track", rel["previous_track_node"])
    add("Next observation in track", rel["next_track_node"])
    return pd.DataFrame(rows)


def node_role_ids(
    result: AnalysisResult,
    focus_node_id: int,
    scope: str = "selected_lineage",
) -> dict[str, list[int]]:
    """Return deterministic node-ID lists for scientist-facing lineage roles.

    ``ancestor`` and ``descendant`` are always defined relative to the focus
    node.  ``division_parent`` and ``terminal`` can be restricted to the focus
    lineage or computed across the entire dataset.
    """
    focus_node_id = int(focus_node_id)
    focus = _node_row(result, focus_node_id)
    parent, children = _graph_maps(result)
    rel = node_relationships(result, focus_node_id)

    valid_scopes = {"selected_lineage", "entire_dataset"}
    if scope not in valid_scopes:
        raise ValueError(f"scope must be one of {sorted(valid_scopes)}, got {scope!r}")

    nodes = result.nodes.copy()
    if scope == "selected_lineage":
        if "lineage_id" not in nodes.columns or pd.isna(focus.get("lineage_id")):
            raise ValueError("Selected-lineage scope requires lineage_id annotations")
        lineage_id = int(focus["lineage_id"])
        nodes = nodes[nodes["lineage_id"].astype(int).eq(lineage_id)]

    allowed = set(nodes["node_id"].astype(int))
    node_time = result.nodes.set_index("node_id")["t"].astype(int).to_dict()

    def ordered(values: Iterable[int]) -> list[int]:
        return sorted(
            {int(v) for v in values if int(v) in allowed},
            key=lambda nid: (node_time.get(nid, 10**9), nid),
        )

    division_parents = ordered(
        nid for nid in allowed if len([c for c in children.get(int(nid), []) if int(c) in allowed]) >= 2
    )
    terminals = ordered(
        nid for nid in allowed if len([c for c in children.get(int(nid), []) if int(c) in allowed]) == 0
    )

    return {
        "selected": [focus_node_id],
        "division_parent": division_parents,
        "ancestor": ordered(rel["ancestors"]),
        "descendant": ordered(rel["descendants"]),
        "terminal": terminals,
    }


def node_role_table(
    result: AnalysisResult,
    focus_node_id: int,
    scope: str = "selected_lineage",
) -> pd.DataFrame:
    """Build one searchable row per node participating in any role.

    A node may have multiple roles, for example an ancestor can also be a
    division parent and a descendant can also be terminal.  The ``roles``
    column keeps all applicable labels instead of forcing one category.
    """
    focus_node_id = int(focus_node_id)
    role_ids = node_role_ids(result, focus_node_id, scope=scope)
    parent, children = _graph_maps(result)
    nodes = result.nodes.set_index("node_id", drop=False)

    pretty = {
        "selected": "selected",
        "division_parent": "division parent",
        "ancestor": "ancestor",
        "descendant": "descendant",
        "terminal": "terminal",
    }

    role_membership: dict[int, list[str]] = defaultdict(list)
    for role, ids in role_ids.items():
        for nid in ids:
            role_membership[int(nid)].append(pretty[role])

    role_order = ["selected", "division parent", "ancestor", "descendant", "terminal"]
    rows: list[dict[str, object]] = []
    for nid in sorted(
        role_membership,
        key=lambda n: (int(nodes.loc[n, "t"]) if n in nodes.index else 10**9, n),
    ):
        if nid not in nodes.index:
            continue
        row = nodes.loc[nid]
        roles = sorted(
            set(role_membership[nid]),
            key=lambda role: role_order.index(role) if role in role_order else 999,
        )
        direct_children = [int(x) for x in children.get(nid, [])]
        rows.append(
            {
                "node_id": int(nid),
                "roles": ", ".join(roles),
                "track_id": int(row["track_id"]) if "track_id" in row and pd.notna(row["track_id"]) else None,
                "lineage_id": int(row["lineage_id"]) if "lineage_id" in row and pd.notna(row["lineage_id"]) else None,
                "generation": int(row["generation"]) if "generation" in row and pd.notna(row["generation"]) else None,
                "frame": int(row["t"]),
                "parent": parent.get(nid),
                "children": ", ".join(map(str, direct_children)) if direct_children else "—",
                "qc_status": row.get("qc_status", "pass"),
            }
        )
    return pd.DataFrame(rows)
