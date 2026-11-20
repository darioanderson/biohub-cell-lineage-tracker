import numpy as np

from biohub.config import Config
from biohub.tracking import graph_invariant_errors, link_sequence_global


def test_global_tracker_bridges_one_missing_frame_without_conflicts():
    ids = [[1], [], [2]]
    xyz = [np.array([[0., 0., 0.]]), np.empty((0, 3)), np.array([[0., 1., 0.]])]
    intensity = [np.array([10.]), np.empty(0), np.array([11.])]
    cfg = Config(
        tracking_backend="global", global_max_frame_gap=2,
        voxel_size_zyx_um=(1., 1., 1.), max_link_dist_um=2.,
    )
    edges = link_sequence_global(ids, xyz, intensity, cfg)
    assert edges == [(1, 2)]


def test_global_tracker_enforces_one_predecessor_and_successor():
    ids = [[1, 2], [3], [4]]
    xyz = [
        np.array([[0., 0., 0.], [0., 5., 0.]]),
        np.array([[0., .2, 0.]]),
        np.array([[0., .4, 0.]]),
    ]
    intensity = [np.ones(2), np.ones(1), np.ones(1)]
    edges = link_sequence_global(ids, xyz, intensity, Config(
        tracking_backend="global", global_max_frame_gap=2,
        voxel_size_zyx_um=(1., 1., 1.), max_link_dist_um=2.,
    ))
    nodes = [
        {"node_id": node, "t": t}
        for t, frame in enumerate(ids) for node in frame
    ]
    edge_rows = [{"source_id": a, "target_id": b} for a, b in edges]
    assert graph_invariant_errors(nodes, edge_rows) == []
    assert sum(b == 3 for _, b in edges) <= 1
    assert sum(a == 3 for a, _ in edges) <= 1


def test_global_tracker_adds_only_one_geometrically_valid_second_daughter():
    ids = [[1], [2, 3, 4]]
    xyz = [
        np.array([[0., 0., 0.]]),
        np.array([[0., -0.5, 0.], [0., 0.5, 0.], [0., 8., 0.]]),
    ]
    intensity = [np.ones(1), np.ones(3)]
    cfg = Config(
        tracking_backend="global", global_max_frame_gap=1,
        voxel_size_zyx_um=(1., 1., 1.), max_link_dist_um=2.,
        detect_divisions=True, div_min_count_gain=1,
        div_parent_dist_um=2., div_sister_dist_um=2.,
        div_use_midpoint_gate=True, div_midpoint_dist_um=1.,
    )
    edges = link_sequence_global(ids, xyz, intensity, cfg)
    assert len([edge for edge in edges if edge[0] == 1]) == 2
    assert (1, 4) not in edges
    nodes = [{"node_id": node, "t": t} for t, frame in enumerate(ids) for node in frame]
    assert graph_invariant_errors(nodes, [
        {"source_id": source, "target_id": target} for source, target in edges
    ]) == []
