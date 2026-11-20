"""Frame-to-frame linking and division detection (v1.2 / v1.3)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.spatial import cKDTree

from biohub.config import Config


def neighbor_signature(xyz_phys: np.ndarray, cfg: Config) -> np.ndarray:
    """K-nearest-neighbor offset vectors as a local context descriptor."""
    n = len(xyz_phys)
    if n == 0:
        return np.empty((0, cfg.neighborhood_k * 3), dtype=np.float64)
    k = min(cfg.neighborhood_k, max(n - 1, 0))
    if k == 0:
        return np.zeros((n, cfg.neighborhood_k * 3), dtype=np.float64)
    tree = cKDTree(xyz_phys)
    dists, idx = tree.query(xyz_phys, k=k + 1)
    sig = np.zeros((n, cfg.neighborhood_k * 3), dtype=np.float64)
    for i in range(n):
        nbrs = idx[i, 1:] if np.ndim(idx) > 1 else []
        for j, nb in enumerate(nbrs[: cfg.neighborhood_k]):
            if dists[i, j + 1] <= cfg.neighborhood_radius_um:
                sig[i, j * 3 : (j + 1) * 3] = xyz_phys[nb] - xyz_phys[i]
    return sig


def _intensity_cost(prev_i: np.ndarray, curr_j: np.ndarray) -> np.ndarray:
    if prev_i.size == 0 or curr_j.size == 0:
        return np.zeros((len(prev_i), len(curr_j)), dtype=np.float64)
    p = prev_i[:, None].astype(np.float64)
    c = curr_j[None, :].astype(np.float64)
    denom = np.maximum(np.maximum(p, c), 1.0)
    return np.abs(p - c) / denom


def _signature_cost(prev_sig: np.ndarray, curr_sig: np.ndarray) -> np.ndarray:
    if len(prev_sig) == 0 or len(curr_sig) == 0:
        return np.zeros((len(prev_sig), len(curr_sig)), dtype=np.float64)
    diff = prev_sig[:, None, :] - curr_sig[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def build_link_cost(
    prev_phys: np.ndarray,
    curr_phys: np.ndarray,
    cfg: Config,
    prev_intensity: Optional[np.ndarray] = None,
    curr_intensity: Optional[np.ndarray] = None,
    prev_velocity: Optional[np.ndarray] = None,
    prev_sig: Optional[np.ndarray] = None,
    curr_sig: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Physical distance matrix plus optional rich-linking terms."""
    dist = np.sqrt(((prev_phys[:, None, :] - curr_phys[None, :, :]) ** 2).sum(axis=2))
    cost = dist.copy()

    if cfg.use_rich_linking:
        if prev_velocity is not None and len(prev_velocity) == len(prev_phys):
            predicted = prev_phys[:, None, :] + prev_velocity[:, None, :]
            motion_resid = np.sqrt(((curr_phys[None, :, :] - predicted) ** 2).sum(axis=2))
            cost = cost + cfg.motion_lambda * motion_resid
        if prev_intensity is not None and curr_intensity is not None:
            cost = cost + cfg.intensity_lambda * _intensity_cost(prev_intensity, curr_intensity)
        if prev_sig is not None and curr_sig is not None:
            sig_cost = _signature_cost(prev_sig, curr_sig)
            med = float(np.median(sig_cost[sig_cost > 0])) if np.any(sig_cost > 0) else 1.0
            cost = cost + cfg.neighborhood_lambda * (sig_cost / max(med, 1e-6))

    big = 1e6
    cost = np.where(dist <= cfg.max_link_dist_um, cost, big)
    return cost, dist


def _candidate_components(
    prev_phys: np.ndarray,
    curr_phys: np.ndarray,
    gate_um: float,
) -> List[Tuple[List[int], List[int]]]:
    """Connected components of the sparse gated bipartite candidate graph."""
    tree = cKDTree(curr_phys)
    row_to_cols = [sorted(map(int, cols)) for cols in tree.query_ball_point(prev_phys, gate_um)]
    col_to_rows: Dict[int, List[int]] = defaultdict(list)
    for row, cols in enumerate(row_to_cols):
        for col in cols:
            col_to_rows[col].append(row)
    components: List[Tuple[List[int], List[int]]] = []
    seen_rows: set[int] = set()
    for start in range(len(prev_phys)):
        if start in seen_rows or not row_to_cols[start]:
            continue
        rows: set[int] = set()
        cols: set[int] = set()
        pending_rows = [start]
        while pending_rows:
            row = pending_rows.pop()
            if row in rows:
                continue
            rows.add(row)
            for col in row_to_cols[row]:
                if col not in cols:
                    cols.add(col)
                    pending_rows.extend(col_to_rows[col])
        seen_rows.update(rows)
        components.append((sorted(rows), sorted(cols)))
    return components


def _component_assignments(
    prev_phys: np.ndarray,
    curr_phys: np.ndarray,
    cfg: Config,
    prev_intensity: Optional[np.ndarray],
    curr_intensity: Optional[np.ndarray],
    prev_velocity: Optional[np.ndarray],
    prev_image_features: Optional[np.ndarray],
    curr_image_features: Optional[np.ndarray],
) -> List[Tuple[int, int]]:
    """Solve Hungarian assignment per sparse candidate component."""
    learned_backends = {"learned", "image_learned", "learned_joint", "image_learned_joint"}
    gate = cfg.learned_linker_gate_um if cfg.linker_backend in learned_backends else cfg.max_link_dist_um
    components = _candidate_components(prev_phys, curr_phys, gate)
    prev_sig = neighbor_signature(prev_phys, cfg) if cfg.use_rich_linking and cfg.linker_backend == "classical" else None
    curr_sig = neighbor_signature(curr_phys, cfg) if cfg.use_rich_linking and cfg.linker_backend == "classical" else None
    assignments: List[Tuple[int, int]] = []
    for rows, cols in components:
        row_ix = np.asarray(rows, dtype=int)
        col_ix = np.asarray(cols, dtype=int)
        if cfg.linker_backend in learned_backends:
            from biohub.learned_linker import build_learned_link_cost

            cost, _ = build_learned_link_cost(
                prev_phys[row_ix],
                curr_phys[col_ix],
                cfg.image_linker_path if cfg.linker_backend in {"image_learned", "image_learned_joint"} else cfg.learned_linker_path,
                gate_um=gate,
                prev_image_features=None if prev_image_features is None else prev_image_features[row_ix],
                curr_image_features=None if curr_image_features is None else curr_image_features[col_ix],
            )
        elif cfg.linker_backend in {"classical", "classical_joint"}:
            cost, _ = build_link_cost(
                prev_phys[row_ix], curr_phys[col_ix], cfg,
                prev_intensity=None if prev_intensity is None else prev_intensity[row_ix],
                curr_intensity=None if curr_intensity is None else curr_intensity[col_ix],
                prev_velocity=(
                    None
                    if prev_velocity is None or len(prev_velocity) != len(prev_phys)
                    else prev_velocity[row_ix]
                ),
                prev_sig=None if prev_sig is None else prev_sig[row_ix],
                curr_sig=None if curr_sig is None else curr_sig[col_ix],
            )
        else:
            raise ValueError(f"Unknown linker_backend: {cfg.linker_backend!r}")
        local_rows, local_cols = linear_sum_assignment(cost)
        for row, col in zip(local_rows, local_cols):
            if cost[row, col] < 1e6:
                assignments.append((rows[int(row)], cols[int(col)]))
    return assignments


def _joint_hypothesis_assignments(
    prev_phys: np.ndarray,
    curr_phys: np.ndarray,
    cfg: Config,
    prev_intensity: Optional[np.ndarray],
    curr_intensity: Optional[np.ndarray],
    prev_velocity: Optional[np.ndarray],
    prev_image_features: Optional[np.ndarray],
    curr_image_features: Optional[np.ndarray],
) -> List[Tuple[int, int]]:
    """Select continuation and 1->2 division hypotheses in one objective.

    A division is one binary optimization variable that claims its parent and
    both daughters together.  This avoids committing to a 1:1 edge before the
    daughter pair has been evaluated.
    """
    learned = cfg.linker_backend in {"learned_joint", "image_learned_joint"}
    gate = cfg.learned_linker_gate_um if learned else cfg.max_link_dist_um
    prev_sig = neighbor_signature(prev_phys, cfg) if cfg.use_rich_linking and not learned else None
    curr_sig = neighbor_signature(curr_phys, cfg) if cfg.use_rich_linking and not learned else None
    selected: List[Tuple[int, int]] = []
    unmatched = float(cfg.joint_hypothesis_unmatched_cost)

    for rows, cols in _candidate_components(prev_phys, curr_phys, gate):
        row_ix = np.asarray(rows, dtype=int)
        col_ix = np.asarray(cols, dtype=int)
        if learned:
            from biohub.learned_linker import build_learned_link_cost

            cost, dist = build_learned_link_cost(
                prev_phys[row_ix], curr_phys[col_ix],
                cfg.image_linker_path if cfg.linker_backend == "image_learned_joint" else cfg.learned_linker_path,
                gate_um=gate,
                prev_image_features=None if prev_image_features is None else prev_image_features[row_ix],
                curr_image_features=None if curr_image_features is None else curr_image_features[col_ix],
            )
        else:
            cost, dist = build_link_cost(
                prev_phys[row_ix], curr_phys[col_ix], cfg,
                prev_intensity=None if prev_intensity is None else prev_intensity[row_ix],
                curr_intensity=None if curr_intensity is None else curr_intensity[col_ix],
                prev_velocity=None if prev_velocity is None or len(prev_velocity) != len(prev_phys) else prev_velocity[row_ix],
                prev_sig=None if prev_sig is None else prev_sig[row_ix],
                curr_sig=None if curr_sig is None else curr_sig[col_ix],
            )

        hypotheses: List[Tuple[int, Tuple[int, ...], float]] = []
        for r in range(len(rows)):
            for c in range(len(cols)):
                if cost[r, c] < 1e6:
                    hypotheses.append((r, (c,), 2.0 * unmatched - float(cost[r, c])))
            if not cfg.detect_divisions:
                continue
            for a in range(len(cols)):
                for b in range(a + 1, len(cols)):
                    if dist[r, a] > cfg.div_parent_dist_um or dist[r, b] > cfg.div_parent_dist_um:
                        continue
                    sister = float(np.linalg.norm(curr_phys[col_ix[a]] - curr_phys[col_ix[b]]))
                    if sister > cfg.div_sister_dist_um:
                        continue
                    midpoint = 0.5 * (curr_phys[col_ix[a]] + curr_phys[col_ix[b]])
                    midpoint_residual = float(np.linalg.norm(prev_phys[row_ix[r]] - midpoint))
                    if cfg.div_use_midpoint_gate and midpoint_residual > cfg.div_midpoint_dist_um:
                        continue
                    combined_displacement = float(dist[r, a] + dist[r, b])
                    asymmetry = abs(float(dist[r, a] - dist[r, b])) / max(combined_displacement, 1e-6)
                    geometry = (
                        midpoint_residual / cfg.div_midpoint_dist_um
                        + sister / cfg.div_sister_dist_um
                        + 0.5 * asymmetry
                    )
                    joint_cost = float(cost[r, a] + cost[r, b]) + cfg.joint_division_geometry_weight * geometry
                    hypotheses.append((r, (a, b), 3.0 * unmatched - joint_cost))

        useful = [hypothesis for hypothesis in hypotheses if hypothesis[2] > 0]
        if not useful:
            continue
        constraints = np.zeros((len(rows) + len(cols), len(useful)), dtype=np.float64)
        objective = np.empty(len(useful), dtype=np.float64)
        for h, (r, targets, benefit) in enumerate(useful):
            constraints[r, h] = 1.0
            for target in targets:
                constraints[len(rows) + target, h] = 1.0
            objective[h] = -benefit + 1e-12 * h
        result = milp(
            c=objective,
            integrality=np.ones(len(useful), dtype=int),
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint(constraints, 0.0, 1.0),
            options={"presolve": True},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"Joint linking optimization failed: {result.message}")
        for value, (r, targets, _) in zip(result.x, useful):
            if value > 0.5:
                selected.extend((rows[r], cols[target]) for target in targets)
    return selected


def link_frames(
    prev_ids: Sequence[int],
    prev_xyz: np.ndarray,
    curr_ids: Sequence[int],
    curr_xyz: np.ndarray,
    cfg: Config,
    prev_intensity: Optional[np.ndarray] = None,
    curr_intensity: Optional[np.ndarray] = None,
    prev_velocity: Optional[np.ndarray] = None,
    next_xyz: Optional[np.ndarray] = None,
    prev_image_features: Optional[np.ndarray] = None,
    curr_image_features: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    """Hungarian assignment with rich costs plus division pass."""
    if len(prev_ids) == 0 or len(curr_ids) == 0:
        return []

    scale = cfg.scale_array
    prev_phys = prev_xyz.astype(np.float64) * scale[None, :]
    curr_phys = curr_xyz.astype(np.float64) * scale[None, :]

    joint_backend = cfg.linker_backend in {"classical_joint", "learned_joint", "image_learned_joint"}
    assignment_fn = _joint_hypothesis_assignments if joint_backend else _component_assignments
    assignments = assignment_fn(
        prev_phys, curr_phys, cfg, prev_intensity, curr_intensity, prev_velocity,
        prev_image_features, curr_image_features,
    )

    edges: List[Tuple[int, int]] = []
    parent_children: Dict[int, List[int]] = defaultdict(list)
    matched_curr = set()

    for r, c in assignments:
        edges.append((int(prev_ids[r]), int(curr_ids[c])))
        parent_children[int(r)].append(int(c))
        matched_curr.add(int(c))

    allow_div = cfg.detect_divisions
    if allow_div and cfg.div_min_count_gain > 0 and len(curr_ids) - len(prev_ids) < cfg.div_min_count_gain:
        allow_div = False

    if allow_div and not joint_backend:
        parent_tree = cKDTree(prev_phys)
        for c in range(len(curr_ids)):
            if c in matched_curr:
                continue
            best_p, best_score = None, np.inf
            candidate_parents = parent_tree.query_ball_point(curr_phys[c], cfg.div_parent_dist_um)
            for p in candidate_parents:
                if len(parent_children.get(p, [])) != 1:
                    continue
                parent_dist = float(np.linalg.norm(prev_phys[p] - curr_phys[c]))
                sister = parent_children[p][0]
                sister_dist = float(np.linalg.norm(curr_phys[c] - curr_phys[sister]))
                if sister_dist > cfg.div_sister_dist_um:
                    continue
                if cfg.div_use_midpoint_gate:
                    midpoint = 0.5 * (curr_phys[c] + curr_phys[sister])
                    mid_dist = float(np.linalg.norm(prev_phys[p] - midpoint))
                    if mid_dist > cfg.div_midpoint_dist_um:
                        continue
                score = float(parent_dist + 0.25 * sister_dist)
                if cfg.div_symmetry_weight > 0:
                    d_parent_sister = float(np.linalg.norm(prev_phys[p] - curr_phys[sister]))
                    d_parent_child = parent_dist
                    if d_parent_sister > 1e-6:
                        asym = abs(d_parent_child - d_parent_sister) / d_parent_sister
                        score += cfg.div_symmetry_weight * asym
                if score < best_score:
                    best_p, best_score = p, score
            if best_p is not None:
                edges.append((int(prev_ids[best_p]), int(curr_ids[c])))
                parent_children[best_p].append(c)
                matched_curr.add(c)

    return edges


def close_frame_gaps(
    frame_ids: List[List[int]],
    frame_centroids: List[np.ndarray],
    frame_intensities: List[np.ndarray],
    existing_edges: List[Tuple[int, int]],
    cfg: Config,
) -> List[Tuple[int, int]]:
    """
    Link detections across one skipped frame (t-2 -> t) for unmatched nodes.

    Recovers short occlusions without full multi-frame optimization.
    """
    if not cfg.gap_close_enabled or len(frame_ids) < 3:
        return []

    scale = cfg.scale_array
    gap_edges: List[Tuple[int, int]] = []
    used_targets: set[int] = set()
    for e in existing_edges:
        used_targets.add(int(e[1]))

    for t in range(2, len(frame_ids)):
        prev2_ids = frame_ids[t - 2]
        prev1_ids = set(frame_ids[t - 1])
        curr_ids = frame_ids[t]
        if not prev2_ids or not curr_ids:
            continue

        linked_from_prev2 = {
            int(s)
            for s, u in existing_edges + gap_edges
            if int(s) in prev2_ids and int(u) in prev1_ids
        }
        linked_to_curr = {
            int(u)
            for s, u in existing_edges + gap_edges
            if int(u) in curr_ids and int(s) in prev1_ids
        }

        unmatched_prev_idx = [i for i, nid in enumerate(prev2_ids) if int(nid) not in linked_from_prev2]
        unmatched_curr_idx = [
            j
            for j, nid in enumerate(curr_ids)
            if int(nid) not in linked_to_curr and int(nid) not in used_targets
        ]
        if not unmatched_prev_idx or not unmatched_curr_idx:
            continue

        prev_xyz = frame_centroids[t - 2][unmatched_prev_idx]
        curr_xyz = frame_centroids[t][unmatched_curr_idx]
        prev_int = frame_intensities[t - 2][unmatched_prev_idx]
        curr_int = frame_intensities[t][unmatched_curr_idx]
        prev_node_ids = [prev2_ids[i] for i in unmatched_prev_idx]
        curr_node_ids = [curr_ids[j] for j in unmatched_curr_idx]

        trial_cfg = cfg.copy_with(max_link_dist_um=cfg.gap_close_dist_um)
        links = link_frames(
            prev_node_ids,
            prev_xyz,
            curr_node_ids,
            curr_xyz,
            trial_cfg,
            prev_intensity=prev_int,
            curr_intensity=curr_int,
            prev_velocity=None,
        )
        for s, u in links:
            gap_edges.append((int(s), int(u)))
            used_targets.add(int(u))

    return gap_edges


def link_sequence_global(
    frame_ids: List[List[int]],
    frame_centroids: List[np.ndarray],
    frame_intensities: List[np.ndarray],
    cfg: Config,
) -> List[Tuple[int, int]]:
    """Optimize continuation and gap links jointly over an entire sequence.

    Each observation has separate source and target copies, allowing at most one
    predecessor and one continuation successor. Forward-only candidate edges make
    the result acyclic. A constrained adjacent-frame pass can add one second child
    to a continuing parent when the configured division geometry is satisfied.
    """
    import networkx as nx

    graph = nx.Graph()
    scale = cfg.scale_array
    max_gap = min(int(cfg.global_max_frame_gap), max(len(frame_ids) - 1, 1))
    for source_t in range(len(frame_ids) - 1):
        source_xyz = frame_centroids[source_t].astype(np.float64) * scale
        if not len(source_xyz):
            continue
        for gap in range(1, max_gap + 1):
            target_t = source_t + gap
            if target_t >= len(frame_ids):
                break
            target_xyz = frame_centroids[target_t].astype(np.float64) * scale
            if not len(target_xyz):
                continue
            gate = cfg.max_link_dist_um * gap
            tree = cKDTree(target_xyz)
            for source_i, neighbors in enumerate(tree.query_ball_point(source_xyz, gate)):
                for target_i in neighbors:
                    dist = float(np.linalg.norm(source_xyz[source_i] - target_xyz[target_i]))
                    cost = dist / gate + cfg.global_gap_penalty * (gap - 1)
                    if len(frame_intensities[source_t]) and len(frame_intensities[target_t]):
                        a = float(frame_intensities[source_t][source_i])
                        b = float(frame_intensities[target_t][target_i])
                        cost += cfg.intensity_lambda * abs(a - b) / max(a, b, 1.0)
                    benefit = 2.0 * cfg.global_unmatched_cost - cost
                    if benefit > 0:
                        graph.add_edge(
                            ("source", int(frame_ids[source_t][source_i])),
                            ("target", int(frame_ids[target_t][target_i])),
                            weight=float(benefit),
                        )
    matching = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=False, weight="weight")
    edges = []
    for left, right in matching:
        if left[0] == "target":
            left, right = right, left
        if left[0] == "source" and right[0] == "target":
            edges.append((int(left[1]), int(right[1])))

    if cfg.detect_divisions:
        id_location = {
            int(node_id): (t, i)
            for t, ids in enumerate(frame_ids)
            for i, node_id in enumerate(ids)
        }
        children: Dict[int, List[int]] = defaultdict(list)
        used_targets = set()
        for source, target in edges:
            children[source].append(target)
            used_targets.add(target)
        for t in range(1, len(frame_ids)):
            if cfg.div_min_count_gain > 0 and len(frame_ids[t]) - len(frame_ids[t - 1]) < cfg.div_min_count_gain:
                continue
            prev_phys = frame_centroids[t - 1].astype(np.float64) * scale
            curr_phys = frame_centroids[t].astype(np.float64) * scale
            if not len(prev_phys) or not len(curr_phys):
                continue
            parent_tree = cKDTree(prev_phys)
            for child_i, child_id in enumerate(frame_ids[t]):
                child_id = int(child_id)
                if child_id in used_targets:
                    continue
                best_parent = None
                best_score = np.inf
                for parent_i in parent_tree.query_ball_point(curr_phys[child_i], cfg.div_parent_dist_um):
                    parent_id = int(frame_ids[t - 1][parent_i])
                    if len(children.get(parent_id, [])) != 1:
                        continue
                    sister_id = children[parent_id][0]
                    sister_t, sister_i = id_location[sister_id]
                    if sister_t != t:
                        continue
                    sister_dist = float(np.linalg.norm(curr_phys[child_i] - curr_phys[sister_i]))
                    if sister_dist > cfg.div_sister_dist_um:
                        continue
                    parent_dist = float(np.linalg.norm(prev_phys[parent_i] - curr_phys[child_i]))
                    if cfg.div_use_midpoint_gate:
                        midpoint = 0.5 * (curr_phys[child_i] + curr_phys[sister_i])
                        if float(np.linalg.norm(prev_phys[parent_i] - midpoint)) > cfg.div_midpoint_dist_um:
                            continue
                    score = parent_dist + 0.25 * sister_dist
                    if cfg.div_symmetry_weight > 0:
                        sister_parent_dist = float(np.linalg.norm(prev_phys[parent_i] - curr_phys[sister_i]))
                        if sister_parent_dist > 1e-6:
                            score += cfg.div_symmetry_weight * abs(parent_dist - sister_parent_dist) / sister_parent_dist
                    if score < best_score:
                        best_parent, best_score = parent_id, score
                if best_parent is not None:
                    edges.append((best_parent, child_id))
                    children[best_parent].append(child_id)
                    used_targets.add(child_id)
    return sorted(edges)


def prune_isolated_nodes(
    node_rows: List[dict],
    edge_rows: List[dict],
    cfg: Optional[Config] = None,
) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    """Annotate isolated detections without deleting raw observations.

    The historical function name is retained for API compatibility.  Nodes that
    would previously have been removed now receive a QC status and reason; edges
    are returned unchanged so graph evidence is never silently discarded.
    """
    incident = set()
    for e in edge_rows:
        incident.add(int(e["source_id"]))
        incident.add(int(e["target_id"]))

    if cfg is not None and cfg.prune_soft_neighbors:
        by_time: Dict[int, List[dict]] = defaultdict(list)
        for r in node_rows:
            by_time[int(r["t"])].append(r)
        scale = cfg.scale_array
        gate = cfg.prune_neighbor_dist_um

        def _nearby(r: dict) -> bool:
            if int(r["node_id"]) in incident:
                return True
            t = int(r["t"])
            pos = np.array([r["z"], r["y"], r["x"]], dtype=np.float64) * scale
            for dt in (-1, 1):
                for other in by_time.get(t + dt, []):
                    op = np.array([other["z"], other["y"], other["x"]], dtype=np.float64) * scale
                    if float(np.linalg.norm(pos - op)) <= gate:
                        return True
            return False

        accepted_ids = {int(r["node_id"]) for r in node_rows if _nearby(r)}
    else:
        accepted_ids = incident

    flagged = 0
    for row in node_rows:
        if int(row["node_id"]) in accepted_ids:
            row["qc_status"] = "pass"
            row["filter_reason"] = ""
        else:
            row["qc_status"] = "flagged"
            row["filter_reason"] = "isolated_no_temporal_edge"
            flagged += 1
    return node_rows, edge_rows, {
        "removed_isolated": 0,
        "flagged_isolated": flagged,
        "kept_nodes": len(node_rows),
        "kept_edges": len(edge_rows),
    }


def annotate_lineage_ids(node_rows: List[dict], edge_rows: List[dict]) -> None:
    """Add observation, track, lineage, and division-parent semantics in place."""
    by_id = {int(row["node_id"]): row for row in node_rows}
    children: Dict[int, List[int]] = defaultdict(list)
    parent: Dict[int, int] = {}
    for edge in edge_rows:
        source = int(edge["source_id"])
        target = int(edge["target_id"])
        children[source].append(target)
        parent[target] = source

    next_track = 1
    next_lineage = 1
    ordered = sorted(node_rows, key=lambda r: (int(r["t"]), int(r["node_id"])))
    for row in ordered:
        node_id = int(row["node_id"])
        source = parent.get(node_id)
        if source is None or source not in by_id:
            row["track_id"] = next_track
            row["lineage_id"] = next_lineage
            row["generation"] = 0
            next_track += 1
            next_lineage += 1
        else:
            source_row = by_id[source]
            row["lineage_id"] = int(source_row["lineage_id"])
            if len(children[source]) >= 2:
                row["track_id"] = next_track
                row["generation"] = int(source_row["generation"]) + 1
                next_track += 1
            else:
                row["track_id"] = int(source_row["track_id"])
                row["generation"] = int(source_row["generation"])
        row["is_division_parent"] = len(children[node_id]) >= 2
        row["division_status"] = "predicted_candidate" if row["is_division_parent"] else "not_applicable"
        row.setdefault("qc_status", "pass")
        row.setdefault("filter_reason", "")


def graph_invariant_errors(
    node_rows: List[dict], edge_rows: List[dict], allowed_gap: int = 1,
) -> List[str]:
    """Return deterministic violations of the lineage graph contract."""
    by_id = {int(row["node_id"]): row for row in node_rows}
    indegree: Counter[int] = Counter()
    outdegree: Counter[int] = Counter()
    adjacency: Dict[int, List[int]] = defaultdict(list)
    errors: List[str] = []
    for edge in edge_rows:
        source = int(edge["source_id"])
        target = int(edge["target_id"])
        if source not in by_id or target not in by_id:
            errors.append(f"dangling edge {source}->{target}")
            continue
        indegree[target] += 1
        outdegree[source] += 1
        adjacency[source].append(target)
        dt = int(by_id[target]["t"]) - int(by_id[source]["t"])
        if dt <= 0:
            errors.append(f"non-forward edge {source}->{target}")
        elif dt > allowed_gap:
            errors.append(f"gap exceeds {allowed_gap} at {source}->{target}")
    errors.extend(f"in-degree >1 at {node}" for node, degree in indegree.items() if degree > 1)
    errors.extend(f"out-degree >2 at {node}" for node, degree in outdegree.items() if degree > 2)
    for source, targets in adjacency.items():
        if len(targets) >= 2:
            source_t = int(by_id[source]["t"])
            target_times = {int(by_id[target]["t"]) for target in targets}
            if target_times != {source_t + 1}:
                errors.append(f"invalid daughter timing at {source}")

    state: Dict[int, int] = {}
    def visit(node: int) -> bool:
        if state.get(node) == 1:
            return True
        if state.get(node) == 2:
            return False
        state[node] = 1
        if any(visit(child) for child in adjacency.get(node, [])):
            return True
        state[node] = 2
        return False
    if any(visit(node) for node in by_id if state.get(node, 0) == 0):
        errors.append("cycle detected")
    return errors


def count_divisions(edge_rows: List[dict]) -> int:
    src_counts = Counter(int(e["source_id"]) for e in edge_rows)
    return sum(1 for c in src_counts.values() if c >= 2)
