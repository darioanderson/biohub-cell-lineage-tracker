from __future__ import annotations

import numpy as np
import pandas as pd

from biohub.tuning import (
    _match_nodes_to_gt,
    evaluate_sparse_graph,
    paired_bootstrap_delta,
)


def test_one_prediction_cannot_match_two_gt_cells() -> None:
    gt = pd.DataFrame(
        {"node_id": [10, 20], "t": [0, 0], "z": [0, 0], "y": [0, 0], "x": [0.0, 2.0]}
    )
    pred = np.array([[0.0, 0.0, 1.0]])
    mapping = _match_nodes_to_gt(pred, gt, np.ones(3), gate_um=1.5)
    assert len(mapping) == 1
    assert len(mapping) / len(gt) == 0.5


def test_sparse_unmatched_edge_is_unknown_not_false_positive() -> None:
    gt_nodes = pd.DataFrame(
        {"node_id": [10, 20], "t": [0, 1], "z": [0, 0], "y": [0, 0], "x": [0, 1]}
    )
    gt_edges = pd.DataFrame({"source_id": [10], "target_id": [20]})
    metrics = evaluate_sparse_graph(
        gt_nodes,
        gt_edges,
        pred_edges=[(1, 2), (1, 3)],
        pred_id_to_gt={1: 10, 2: 20},
    )
    assert metrics["edge_tp"] == 1
    assert metrics["edge_fp"] == 0
    assert metrics["edge_unknown"] == 1
    assert metrics["edge_pred_evaluable"] == 1
    assert metrics["edge_pred_unknown"] == 1
    assert metrics["edge_gt_evaluable"] == 1
    assert metrics["edge_precision"] == 1.0


def test_gt_edges_outside_processed_window_are_not_false_negatives() -> None:
    gt_nodes = pd.DataFrame(
        {
            "node_id": [10, 20, 30, 40],
            "t": [0, 1, 10, 11],
            "z": [0, 0, 0, 0],
            "y": [0, 0, 0, 0],
            "x": [0, 1, 0, 1],
        }
    )
    gt_edges = pd.DataFrame({"source_id": [10, 30], "target_id": [20, 40]})
    metrics = evaluate_sparse_graph(
        gt_nodes,
        gt_edges,
        pred_edges=[(1, 2)],
        pred_id_to_gt={1: 10, 2: 20},
        evaluable_times=[0, 1],
    )
    assert metrics["edge_tp"] == 1
    assert metrics["edge_fn"] == 0
    assert metrics["edge_recall"] == 1.0


def test_division_count_cannot_inflate_score_without_gt_support() -> None:
    gt_nodes = pd.DataFrame(
        {
            "node_id": [10, 20, 30],
            "t": [0, 1, 1],
            "z": [0, 0, 0],
            "y": [0, 0, 0],
            "x": [0, -1, 1],
        }
    )
    gt_edges = pd.DataFrame({"source_id": [10], "target_id": [20]})
    metrics = evaluate_sparse_graph(
        gt_nodes,
        gt_edges,
        pred_edges=[(1, 2), (1, 3)],
        pred_id_to_gt={1: 10, 2: 20, 3: 30},
    )
    assert metrics["divisions"] == 1
    assert metrics["division_tp"] == 0
    assert metrics["division_fp"] == 1
    assert metrics["division_f1"] == 0.0


def test_paired_bootstrap_uses_per_dataset_delta() -> None:
    baseline = [0.10, 0.80, 0.25, 0.65]
    candidate = [0.20, 0.90, 0.35, 0.75]
    mean, low, high = paired_bootstrap_delta(
        baseline, candidate, n_boot=1000, random_state=7
    )
    assert np.isclose(mean, 0.10)
    assert low > 0.0
    assert high > 0.0
