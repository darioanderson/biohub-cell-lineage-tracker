import numpy as np

from biohub.sample_data import SYNTHETIC_SCENARIOS, synthetic_ground_truth
from biohub.tracking import graph_invariant_errors


def test_all_controlled_scenarios_return_exact_graph_truth():
    for name in SYNTHETIC_SCENARIOS:
        result = synthetic_ground_truth(name, t=6, shape=(12, 40, 40), seed=7)
        assert result.volume.shape == (6, 12, 40, 40)
        assert {"node_id", "track_id", "lineage_id", "t", "z", "y", "x", "is_division_parent"}.issubset(result.nodes.columns)
        assert graph_invariant_errors(
            result.nodes.to_dict("records"), result.edges.to_dict("records"), allowed_gap=2,
        ) == []


def test_one_mitosis_has_one_exact_parent_and_two_daughters():
    result = synthetic_ground_truth("one_mitosis", t=8)
    assert result.divisions["parent_node_id"].tolist() == [4]
    daughters = result.edges.loc[result.edges["source_id"].eq(4), "target_id"].tolist()
    assert len(daughters) == 2
    parent = result.nodes.set_index("node_id").loc[4]
    child_rows = result.nodes.set_index("node_id").loc[daughters]
    assert parent["is_division_parent"]
    assert child_rows["track_id"].nunique() == 2
    assert np.all(child_rows["lineage_id"].to_numpy() == parent["lineage_id"])


def test_no_division_control_contains_no_divisions():
    result = synthetic_ground_truth("no_division", t=6)
    assert result.divisions.empty
    assert not result.nodes["is_division_parent"].any()


def test_crossing_trajectories_keep_identity_through_crossing():
    result = synthetic_ground_truth("crossing", t=8)
    tracks = result.nodes.groupby("track_id")
    assert set(tracks.groups) == {1, 2}
    assert tracks.get_group(1).sort_values("t")["x"].is_monotonic_increasing
    assert tracks.get_group(2).sort_values("t")["x"].is_monotonic_decreasing


def test_occlusion_truth_contains_one_explicit_gap_edge():
    result = synthetic_ground_truth("temporary_occlusion", t=8)
    times = result.nodes.set_index("node_id")["t"]
    gaps = [int(times.loc[v] - times.loc[u]) for u, v in result.edges.to_numpy()]
    assert gaps.count(2) == 1


def test_acceleration_truth_has_increasing_displacements():
    result = synthetic_ground_truth("acceleration", t=8)
    track = result.nodes[result.nodes["track_id"] == 1].sort_values("t")
    steps = np.diff(track["y"].to_numpy())
    assert np.all(np.diff(steps) > 0)
