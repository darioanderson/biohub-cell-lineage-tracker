import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub.config import Config
from biohub.tracking import build_link_cost, link_frames, neighbor_signature


def _dense_reference(prev, curr, cfg):
    pp = prev * cfg.scale_array
    cp = curr * cfg.scale_array
    cost, _ = build_link_cost(
        pp, cp, cfg,
        prev_sig=neighbor_signature(pp, cfg),
        curr_sig=neighbor_signature(cp, cfg),
    )
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] < 1e6]


def test_sparse_components_match_dense_hungarian_reference():
    rng = np.random.default_rng(4)
    prev = rng.uniform(0, 40, size=(35, 3))
    curr = prev + rng.normal(0, .3, size=(35, 3))
    curr = np.vstack([curr, rng.uniform(100, 120, size=(5, 3))])
    cfg = Config(voxel_size_zyx_um=(1, 1, 1), max_link_dist_um=3, detect_divisions=False)
    expected = {(r, c) for r, c in _dense_reference(prev, curr, cfg)}
    actual_edges = link_frames(range(35), prev, range(40), curr, cfg)
    assert set(actual_edges) == expected


def test_sparse_linking_handles_large_disconnected_problem_without_global_matrix():
    n = 2500
    prev = np.column_stack([np.zeros(n), np.arange(n) * 5.0, np.zeros(n)])
    curr = prev + np.array([0.0, 0.2, 0.0])
    cfg = Config(voxel_size_zyx_um=(1, 1, 1), max_link_dist_um=1, use_rich_linking=False, detect_divisions=False)
    edges = link_frames(range(n), prev, range(n), curr, cfg)
    assert len(edges) == n
    assert edges[0] == (0, 0) and edges[-1] == (n - 1, n - 1)


def test_stale_velocity_shape_is_ignored_safely():
    prev = np.array([[0., 0., 0.], [0., 10., 0.]])
    curr = prev + .2
    cfg = Config(voxel_size_zyx_um=(1, 1, 1), detect_divisions=False)
    edges = link_frames([1, 2], prev, [3, 4], curr, cfg, prev_velocity=np.zeros((1, 3)))
    assert edges == [(1, 3), (2, 4)]
