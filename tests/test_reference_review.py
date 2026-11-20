import numpy as np
from app.reference_review import match_centres


def test_duplicate_prediction_can_only_match_one_reference():
    pairs, errors = match_centres([[0,0,0]], [[0,0,0],[0,0,0]])
    assert pairs.shape == (1,2)
    assert errors.tolist() == [0]


def test_distance_gate_and_empty_sets():
    pairs, _ = match_centres([[0,0,0]], [[0,0,3.01]])
    assert not len(pairs)
    pairs, errors = match_centres(np.empty((0,3)), [[0,0,0]])
    assert pairs.shape == (0,2) and errors.shape == (0,)
