import pandas as pd
import pytest

from biohub.analysis import AnalysisResult
from biohub.curation import apply_curation


def _result():
    nodes = pd.DataFrame([
        {"dataset": "x", "node_id": 1, "t": 0, "z": 0., "y": 0., "x": 0.},
        {"dataset": "x", "node_id": 2, "t": 1, "z": 0., "y": 1., "x": 0.},
        {"dataset": "x", "node_id": 3, "t": 1, "z": 0., "y": 2., "x": 0.},
    ])
    edges = pd.DataFrame([{"dataset": "x", "source_id": 1, "target_id": 2}])
    return AnalysisResult("x", (2, 1, 4, 4), nodes, edges, stats={})


def test_relink_is_audited_and_does_not_mutate_original():
    original = _result()
    curated, audit = apply_curation(original, [{
        "action": "relink", "source_id": 1, "target_id": 3,
        "reviewer": "scientist", "reason": "visual review",
    }])
    assert original.edges[["source_id", "target_id"]].values.tolist() == [[1, 2]]
    assert curated.edges[["source_id", "target_id"]].values.tolist() == [[1, 2], [1, 3]]
    assert curated.n_divisions == 1
    assert audit.loc[0, "reviewer"] == "scientist"


def test_reject_node_removes_incident_edges_from_curated_copy():
    curated, audit = apply_curation(_result(), [{"action": "reject_node", "node_id": 2}])
    assert curated.nodes["node_id"].tolist() == [1, 3]
    assert curated.edges.empty
    assert audit.loc[0, "action"] == "reject_node"


def test_relink_rejects_backward_time_edge():
    with pytest.raises(ValueError, match="forward in time"):
        apply_curation(_result(), [{"action": "relink", "source_id": 2, "target_id": 1}])
