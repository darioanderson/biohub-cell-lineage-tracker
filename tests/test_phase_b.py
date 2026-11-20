import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biohub.analysis import AnalysisResult
from biohub.config import Config
from biohub.export import to_lineage_csv
from biohub.submission import _node_row
from biohub.tracking import annotate_lineage_ids, graph_invariant_errors, prune_isolated_nodes
from biohub.visualization import plot_track_overlay


def _node(node_id, t, x=0.0):
    return {"node_id": node_id, "t": t, "z": 1.25, "y": 2.5, "x": x}


def _edge(source, target):
    return {"source_id": source, "target_id": target}


def test_global_node_ids_render_real_link():
    volume = np.zeros((3, 8, 8), dtype=np.float32)
    a = np.array([[1.0, 2.0, 2.0]])
    b = np.array([[1.0, 4.0, 5.0]])
    fig = plot_track_overlay(
        volume, a, b, [(101, 205)], 1,
        frame_a_ids=[101], frame_b_ids=[205],
    )
    # One link plus the two-layer scale bar proves the global-ID edge resolved.
    assert len(fig.axes[0].lines) == 3
    assert fig.axes[0].lines[0].get_xdata().tolist() == [2.0, 5.0]
    plt.close(fig)


def test_lineage_and_track_ids_split_at_true_division_only():
    nodes = [_node(1, 0), _node(2, 1), _node(3, 2), _node(4, 2)]
    edges = [_edge(1, 2), _edge(2, 3), _edge(2, 4)]
    annotate_lineage_ids(nodes, edges)
    by_id = {row["node_id"]: row for row in nodes}
    assert by_id[1]["track_id"] == by_id[2]["track_id"]
    assert by_id[3]["track_id"] != by_id[2]["track_id"]
    assert by_id[4]["track_id"] != by_id[3]["track_id"]
    assert len({row["lineage_id"] for row in nodes}) == 1
    assert by_id[2]["is_division_parent"] is True
    assert by_id[2]["division_status"] == "predicted_candidate"
    assert by_id[1]["is_division_parent"] is False
    assert by_id[1]["generation"] == by_id[2]["generation"] == 0
    assert by_id[3]["generation"] == by_id[4]["generation"] == 1


def test_isolated_qc_is_nondestructive():
    nodes = [_node(1, 0), _node(2, 1), _node(3, 1)]
    edges = [_edge(1, 2)]
    out_nodes, out_edges, stats = prune_isolated_nodes(nodes, edges, Config())
    assert len(out_nodes) == 3
    assert out_edges == edges
    assert stats["removed_isolated"] == 0
    assert stats["flagged_isolated"] == 1
    isolated = next(row for row in out_nodes if row["node_id"] == 3)
    assert isolated["qc_status"] == "flagged"
    assert isolated["filter_reason"] == "isolated_no_temporal_edge"


def test_graph_invariants_detect_each_contract_violation():
    nodes = [_node(1, 0), _node(2, 1), _node(3, 1), _node(4, 2), _node(5, 2)]
    assert graph_invariant_errors(nodes, [_edge(1, 2), _edge(1, 3)]) == []
    errors = graph_invariant_errors(
        nodes,
        [_edge(1, 2), _edge(3, 2), _edge(1, 3), _edge(1, 4), _edge(4, 99), _edge(2, 1)],
    )
    assert any("in-degree >1" in error for error in errors)
    assert any("out-degree >2" in error for error in errors)
    assert any("dangling edge" in error for error in errors)
    assert any("non-forward edge" in error for error in errors)
    assert any("gap exceeds" in error for error in errors)
    assert "cycle detected" in errors


def test_graph_invariants_reject_nonadjacent_division_daughters():
    nodes = [_node(1, 0), _node(2, 2), _node(3, 2)]
    errors = graph_invariant_errors(
        nodes, [_edge(1, 2), _edge(1, 3)], allowed_gap=2,
    )
    assert "invalid daughter timing at 1" in errors


def test_centroids_remain_float_until_export():
    row = _node_row("sample", 1, 0, [1.25, 2.5, 3.75])
    assert row["z"] == 1.25 and row["x"] == 3.75
    node = {**row, "track_id": 1, "lineage_id": 1, "is_division_parent": False}
    result = AnalysisResult("sample", (1, 4, 4, 4), pd.DataFrame([node]), pd.DataFrame())
    exported = to_lineage_csv(result)
    assert exported.loc[0, ["z", "y", "x"]].tolist() == [1, 2, 4]
