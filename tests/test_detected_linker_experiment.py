import numpy as np
import pandas as pd

from biohub.config import Config
from scripts.evaluate_detected_linker import DetectedFrame, DetectedSample, build_training_pairs, embryo_id


def test_detected_pairs_keep_unknowns_and_detector_misses_out_of_labels():
    prev = DetectedFrame(
        t=0,
        ids=[1, 2, 3, 6],
        xyz=np.array([[0.0, 0, 0], [1.0, 1, 1], [5.0, 5, 5], [8.8, 9, 9]]),
        intensity=np.ones(4),
        image_features=np.zeros((4, 5)),
        pred_to_gt={1: 10, 2: 11, 6: 11},  # detection 3 is sparse-label unknown
        gt_count=3,  # GT 12 was missed by the detector
    )
    curr = DetectedFrame(
        t=1,
        ids=[4, 5],
        xyz=np.array([[0.1, 0, 0], [9.0, 9, 9]]),
        intensity=np.ones(2),
        image_features=np.zeros((2, 5)),
        pred_to_gt={4: 20, 5: 21},
        gt_count=2,
    )
    nodes = pd.DataFrame(
        {
            "node_id": [10, 11, 12, 20, 21],
            "t": [0, 0, 0, 1, 1],
            "z": [0, 1, 8, 0.1, 9],
            "y": [0, 1, 8, 0, 9],
            "x": [0, 1, 8, 0, 9],
        }
    )
    edges = pd.DataFrame({"source_id": [10, 12], "target_id": [20, 21]})
    sample = DetectedSample("44b6_example", "44b6", nodes, edges, [prev, curr])

    features, labels, diagnostics = build_training_pairs(sample, Config())

    assert embryo_id(sample.name) == "44b6"
    assert features.shape[1] == 25
    assert labels.tolist().count(1.0) == 1
    assert labels.tolist() == [1.0, 0.0, 0.0]
    assert len(features) == 3
    assert diagnostics["unknown_candidates"] > 0
    assert diagnostics["missed_parent_targets"] == 1
    assert diagnostics["eligible_targets"] == 2


def test_missed_true_parent_drops_all_alternative_candidates():
    prev = DetectedFrame(
        t=0,
        ids=[1],
        xyz=np.array([[0.0, 0.0, 0.0]]),
        intensity=np.ones(1),
        image_features=np.zeros((1, 5)),
        pred_to_gt={1: 11},
        gt_count=2,
    )
    curr = DetectedFrame(
        t=1,
        ids=[2],
        xyz=np.array([[0.1, 0.0, 0.0]]),
        intensity=np.ones(1),
        image_features=np.zeros((1, 5)),
        pred_to_gt={2: 20},
        gt_count=1,
    )
    nodes = pd.DataFrame(
        {"node_id": [10, 11, 20], "t": [0, 0, 1], "z": [0, 0, 0.1], "y": [0, 0, 0], "x": [0, 0, 0]}
    )
    edges = pd.DataFrame({"source_id": [10], "target_id": [20]})
    sample = DetectedSample("44b6_missed", "44b6", nodes, edges, [prev, curr])

    features, labels, diagnostics = build_training_pairs(sample, Config())

    assert features.shape == (0, 25)
    assert labels.shape == (0,)
    assert diagnostics["eligible_targets"] == 1
    assert diagnostics["covered_targets"] == 0
    assert diagnostics["missed_parent_targets"] == 1
    assert diagnostics["labeled_candidates"] == 0
