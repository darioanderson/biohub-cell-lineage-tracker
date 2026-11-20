from __future__ import annotations

import numpy as np

from biohub.analysis import run_tracking_pipeline
from biohub.config import Config
from biohub.detection import detect_cells
from biohub.export import to_lineage_csv
from biohub.sample_data import synthetic_ground_truth
from biohub.tracking import close_frame_gaps, link_frames


def _link_cfg(**kwargs):
    values = dict(
        voxel_size_zyx_um=(1., 1., 1.), use_rich_linking=False,
        max_link_dist_um=3., detect_divisions=False,
    )
    values.update(kwargs)
    return Config(**values)


def test_link_frames_identity_and_translation() -> None:
    previous = np.array([[2., 3., 4.], [8., 9., 10.]])
    ids_a, ids_b = [11, 12], [21, 22]
    assert link_frames(ids_a, previous, ids_b, previous.copy(), _link_cfg()) == [(11, 21), (12, 22)]
    shifted = previous + np.array([0., .5, 1.])
    assert link_frames(ids_a, previous, ids_b, shifted, _link_cfg()) == [(11, 21), (12, 22)]


def test_link_gate_is_never_exceeded() -> None:
    edges = link_frames(
        [1], np.array([[0., 0., 0.]]), [2], np.array([[0., 0., 3.01]]),
        _link_cfg(max_link_dist_um=3.),
    )
    assert edges == []


def test_division_and_one_to_one_control() -> None:
    cfg = _link_cfg(
        detect_divisions=True, div_parent_dist_um=3., div_sister_dist_um=3.,
        div_midpoint_dist_um=2.,
    )
    split = link_frames(
        [1], np.array([[0., 0., 0.]]), [2, 3],
        np.array([[0., -.5, 0.], [0., .5, 0.]]), cfg,
    )
    assert set(split) == {(1, 2), (1, 3)}
    continuation = link_frames(
        [1], np.array([[0., 0., 0.]]), [2], np.array([[0., .5, 0.]]), cfg,
    )
    assert continuation == [(1, 2)]


def test_joint_division_is_selected_as_one_hypothesis() -> None:
    cfg = _link_cfg(
        linker_backend="classical_joint", detect_divisions=True,
        div_parent_dist_um=3., div_sister_dist_um=3., div_midpoint_dist_um=2.,
        joint_hypothesis_unmatched_cost=2., joint_division_geometry_weight=.1,
    )
    edges = link_frames(
        [1], np.array([[0., 0., 0.]]), [2, 3],
        np.array([[0., -.5, 0.], [0., .5, 0.]]), cfg,
    )
    assert set(edges) == {(1, 2), (1, 3)}

    blocked = link_frames(
        [1], np.array([[0., 0., 0.]]), [2, 3],
        np.array([[0., -.5, 0.], [0., .5, 0.]]),
        cfg.copy_with(div_min_count_gain=2),
    )
    assert len(blocked) == 1


def test_gap_closing_recovers_one_skipped_frame() -> None:
    cfg = _link_cfg(gap_close_enabled=True, gap_close_dist_um=3.)
    edges = close_frame_gaps(
        [[1], [], [2]],
        [np.array([[0., 0., 0.]]), np.empty((0, 3)), np.array([[0., 1., 0.]])],
        [np.array([10.]), np.empty(0), np.array([10.])],
        [], cfg,
    )
    assert edges == [(1, 2)]


def test_detection_known_blob() -> None:
    center = np.array([5., 12., 13.])
    zz, yy, xx = np.indices((12, 24, 24))
    volume = (2000 * np.exp(-((zz-center[0])**2/2 + (yy-center[1])**2/3 + (xx-center[2])**2/3))).astype(np.uint16)
    cfg = Config(
        xy_ds=1, min_z_hard=0, border_z=0, border_yx=0,
        use_dense_cluster_pass=False, use_adaptive_frame_threshold=False,
        nms_radius_um=1., min_rel_contrast=.01,
    )
    coords, _, _ = detect_cells(volume, cfg)
    assert np.min(np.linalg.norm(coords - center, axis=1)) <= 1.0


def test_complete_pipeline_is_byte_deterministic() -> None:
    truth = synthetic_ground_truth("uniform_translation", t=4, shape=(12, 32, 32), seed=9)
    cfg = Config(xy_ds=2, min_z_hard=0, border_z=0, border_yx=0)
    first = to_lineage_csv(run_tracking_pipeline(list(truth.volume), cfg, dataset_name="same"))
    second = to_lineage_csv(run_tracking_pipeline(list(truth.volume), cfg, dataset_name="same"))
    assert first.to_csv().encode() == second.to_csv().encode()
