import pandas as pd

from biohub.analysis import AnalysisResult
from biohub.lineage_explorer import (
    lineage_event_table,
    lineage_summary,
    local_family_node_ids,
    node_relationships,
)
from biohub.tracking import annotate_lineage_ids


def _branched_result() -> AnalysisResult:
    nodes = [
        {"node_id": 1, "t": 0, "z": 1.0, "y": 10.0, "x": 10.0},
        {"node_id": 2, "t": 1, "z": 1.0, "y": 11.0, "x": 10.0},
        {"node_id": 3, "t": 2, "z": 1.0, "y": 12.0, "x": 10.0},
        {"node_id": 4, "t": 3, "z": 1.0, "y": 11.0, "x": 9.0},
        {"node_id": 5, "t": 3, "z": 1.0, "y": 13.0, "x": 11.0},
        {"node_id": 6, "t": 4, "z": 1.0, "y": 10.0, "x": 9.0},
    ]
    edges = [
        {"source_id": 1, "target_id": 2},
        {"source_id": 2, "target_id": 3},
        {"source_id": 3, "target_id": 4},
        {"source_id": 3, "target_id": 5},
        {"source_id": 4, "target_id": 6},
    ]
    annotate_lineage_ids(nodes, edges)
    return AnalysisResult(
        "demo", (5, 4, 32, 32), pd.DataFrame(nodes), pd.DataFrame(edges), stats={}
    )


def test_node_relationships_explain_division_structure():
    result = _branched_result()
    rel = node_relationships(result, 3)
    assert rel["event_type"] == "division_parent"
    assert rel["parent_node_id"] == 2
    assert rel["children"] == [4, 5]
    assert rel["ancestors"] == [1, 2]
    assert rel["descendants"] == [4, 5, 6]
    assert rel["root_node_id"] == 1


def test_daughter_relationships_include_sibling_and_track_continuity():
    result = _branched_result()
    rel = node_relationships(result, 4)
    assert rel["event_type"] == "daughter"
    assert rel["parent_node_id"] == 3
    assert rel["siblings"] == [5]
    assert rel["next_track_node"] == 6
    assert rel["previous_track_node"] is None


def test_local_family_is_depth_limited_but_keeps_sibling_context():
    result = _branched_result()
    ids = local_family_node_ids(result, 4, ancestor_depth=1, descendant_depth=1)
    assert ids == {3, 4, 5, 6}


def test_lineage_summary_and_event_table_report_readable_structure():
    result = _branched_result()
    lineage_id = int(result.nodes.loc[result.nodes["node_id"] == 3, "lineage_id"].iloc[0])
    summary = lineage_summary(result, lineage_id)
    assert summary["observations"] == 6
    assert summary["tracks"] == 3
    assert summary["divisions"] == 1
    assert summary["terminals"] == 2

    events = lineage_event_table(result, lineage_id)
    division = events.loc[events["node_id"] == 3].iloc[0]
    assert division["event"] == "division_parent"
    assert division["parent"] == 2
    assert division["children"] == "4, 5"


def test_node_role_ids_lists_focus_relative_and_lineage_wide_roles():
    from biohub.lineage_explorer import node_role_ids

    result = _branched_result()
    roles = node_role_ids(result, 4, scope="selected_lineage")
    assert roles["selected"] == [4]
    assert roles["division_parent"] == [3]
    assert roles["ancestor"] == [1, 2, 3]
    assert roles["descendant"] == [6]
    assert roles["terminal"] == [5, 6]


def test_node_role_table_preserves_multiple_roles_per_node():
    from biohub.lineage_explorer import node_role_table

    result = _branched_result()
    table = node_role_table(result, 4, scope="selected_lineage")
    node3 = table.loc[table["node_id"] == 3].iloc[0]
    node6 = table.loc[table["node_id"] == 6].iloc[0]
    assert "division parent" in node3["roles"]
    assert "ancestor" in node3["roles"]
    assert "descendant" in node6["roles"]
    assert "terminal" in node6["roles"]
    assert node3["children"] == "4, 5"
