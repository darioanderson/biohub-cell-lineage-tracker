import networkx as nx
import numpy as np

from biohub.trackastra_adapter import (
    add_trackastra_division_proposals,
    centroids_to_watershed_mask,
    edge_list_sha256,
    graph_edge_scores_to_detection_ids,
    graph_edges_to_detection_ids,
    make_isotropic_xy,
)


def test_trackastra_mask_preserves_one_label_per_detection() -> None:
    volume = np.zeros((7, 32, 32), dtype=np.float32)
    volume[3, 8:13, 8:13] = 10
    volume[3, 20:25, 20:25] = 20
    iso = make_isotropic_xy(volume, factor=4)
    mask = centroids_to_watershed_mask(iso, np.array([[3, 10, 10], [3, 22, 22]]))
    assert set(np.unique(mask)) == {0, 1, 2}


def test_trackastra_graph_edges_map_to_original_detection_ids() -> None:
    graph = nx.DiGraph()
    graph.add_node("a", time=0, label=2)
    graph.add_node("b", time=1, label=1)
    graph.add_edge("a", "b")
    assert graph_edges_to_detection_ids(graph, [[10, 11], [20]]) == [(11, 20)]


def test_trackastra_graph_edge_scores_map_to_original_detection_ids() -> None:
    graph = nx.DiGraph()
    graph.add_node("a", time=0, label=2)
    graph.add_node("b", time=1, label=1)
    graph.add_edge("a", "b", weight=0.75)
    assert graph_edge_scores_to_detection_ids(graph, [[10, 11], [20]]) == {(11, 20): 0.75}


def test_trackastra_division_adds_only_unmatched_gated_daughter() -> None:
    times = {1: 0, 2: 1, 3: 1, 4: 0}
    xyz = {
        1: np.array([0.0, 0.0, 0.0]),
        2: np.array([0.0, -0.5, 0.0]),
        3: np.array([0.0, 0.5, 0.0]),
        4: np.array([10.0, 10.0, 10.0]),
    }
    edges, info = add_trackastra_division_proposals(
        [(1, 2)], [(1, 2), (1, 3)], times, xyz, np.ones(3),
        parent_gate_um=2.0, sister_gate_um=2.0, midpoint_gate_um=1.0,
    )
    assert edges == [(1, 2), (1, 3)]
    assert info["accepted_splits"] == 1

    conflicted, conflict_info = add_trackastra_division_proposals(
        [(1, 2), (4, 3)], [(1, 2), (1, 3)], times, xyz, np.ones(3),
        parent_gate_um=2.0, sister_gate_um=2.0, midpoint_gate_um=1.0,
    )
    assert conflicted == [(1, 2), (4, 3)]
    assert conflict_info["rejected_conflict"] == 1


def test_trackastra_division_rejects_geometrically_false_proposal() -> None:
    baseline = [(10, 11), (1, 2)]
    times = {10: 0, 11: 1, 1: 0, 2: 1, 3: 1}
    xyz = {
        10: np.array([5.0, 5.0, 5.0]),
        11: np.array([5.1, 5.0, 5.0]),
        1: np.array([0.0, 0.0, 0.0]),
        2: np.array([0.0, 0.5, 0.0]),
        3: np.array([0.0, 20.0, 0.0]),
    }
    edges, info = add_trackastra_division_proposals(
        baseline, [(1, 2), (1, 3)], times, xyz, np.ones(3),
        parent_gate_um=2.0, sister_gate_um=2.0, midpoint_gate_um=1.0,
    )
    assert edges == baseline
    assert info["rejected_geometry"] == 1
    assert info["accepted_splits"] == 0


def test_validated_division_preserves_unrelated_continuations_byte_for_byte() -> None:
    baseline = [(10, 11), (1, 2), (20, 21)]
    times = {10: 0, 11: 1, 1: 0, 2: 1, 3: 1, 20: 0, 21: 1}
    xyz = {
        10: np.array([5.0, 5.0, 5.0]),
        11: np.array([5.1, 5.0, 5.0]),
        1: np.array([0.0, 0.0, 0.0]),
        2: np.array([0.0, -0.5, 0.0]),
        3: np.array([0.0, 0.5, 0.0]),
        20: np.array([8.0, 8.0, 8.0]),
        21: np.array([8.1, 8.0, 8.0]),
    }
    before = edge_list_sha256(baseline)
    edges, info = add_trackastra_division_proposals(
        baseline, [(1, 2), (1, 3)], times, xyz, np.ones(3),
        parent_gate_um=2.0, sister_gate_um=2.0, midpoint_gate_um=1.0,
    )
    assert edges[: len(baseline)] == baseline
    assert edge_list_sha256(edges[: len(baseline)]) == before
    assert edges == baseline + [(1, 3)]
    assert info["baseline_edges_removed"] == 0
    assert info["baseline_prefix_unchanged"] == 1
    assert info["added_edge_count"] == 1


def test_division_merge_rejects_duplicate_baseline_instead_of_rewriting_it() -> None:
    with np.testing.assert_raises_regex(ValueError, "duplicate edges"):
        add_trackastra_division_proposals(
            [(1, 2), (1, 2)], [], {1: 0, 2: 1},
            {1: np.zeros(3), 2: np.zeros(3)}, np.ones(3),
            parent_gate_um=2.0, sister_gate_um=2.0, midpoint_gate_um=1.0,
        )


def test_division_requires_both_trackastra_edges_to_pass_score_gate() -> None:
    baseline = [(1, 2)]
    times = {1: 0, 2: 1, 3: 1}
    xyz = {1: np.zeros(3), 2: np.array([0.0, -0.5, 0.0]), 3: np.array([0.0, 0.5, 0.0])}
    edges, info = add_trackastra_division_proposals(
        baseline, [(1, 2), (1, 3)], times, xyz, np.ones(3),
        parent_gate_um=2.0, sister_gate_um=2.0, midpoint_gate_um=1.0,
        edge_scores={(1, 2): 0.9, (1, 3): 0.59}, min_edge_score=0.6,
    )
    assert edges == baseline
    assert info["rejected_score"] == 1


def test_division_requires_both_daughters_to_persist() -> None:
    baseline = [(1, 2), (2, 4), (3, 5), (4, 6)]
    times = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3}
    xyz = {node: np.array([0.0, node * 0.1, 0.0]) for node in times}
    rejected, info = add_trackastra_division_proposals(
        baseline, [(1, 2), (1, 3)], times, xyz, np.ones(3),
        parent_gate_um=2.0, sister_gate_um=2.0, midpoint_gate_um=1.0,
        persistence_frames=2,
    )
    assert rejected == baseline
    assert info["rejected_temporal"] == 1

    accepted, accepted_info = add_trackastra_division_proposals(
        baseline + [(5, 7)], [(1, 2), (1, 3)],
        {**times, 7: 3}, {**xyz, 7: np.array([0.0, 0.7, 0.0])}, np.ones(3),
        parent_gate_um=2.0, sister_gate_um=2.0, midpoint_gate_um=1.0,
        persistence_frames=2,
    )
    assert accepted[-1] == (1, 3)
    assert accepted_info["accepted_splits"] == 1
