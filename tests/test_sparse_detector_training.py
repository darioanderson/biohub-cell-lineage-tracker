import numpy as np
import pytest

from biohub.sparse_detector_training import build_sparse_detection_target, masked_balanced_bce


def test_bright_unannotated_voxels_are_unknown_not_background():
    frame = np.zeros((3, 8, 8), dtype=np.float32)
    frame[1, 4, 4] = 0.9
    target, known = build_sparse_detection_target(
        frame, annotated_zyx=np.empty((0, 3)), teacher_zyx=np.empty((0, 3)),
        downsample=(1, 1, 1),
    )
    assert target[1, 4, 4] == 0
    assert known[1, 4, 4] == 0
    assert known[0, 0, 0] == 0


def test_annotations_positive_and_explicit_background_only():
    frame = np.ones((3, 8, 8), dtype=np.float32)
    background=np.zeros(frame.shape,dtype=bool);background[0,0,0]=True
    target, known = build_sparse_detection_target(
        frame, annotated_zyx=np.array([[1, 4, 4]]),
        teacher_zyx=np.empty((0,3)), downsample=(1, 1, 1), background_mask=background,
    )
    assert target[1, 4, 4] == known[1, 4, 4] == 1
    assert target[0,0,0] == 0 and known[0,0,0] == 1
    assert known[1, 5, 5] == 0


def test_unverified_teacher_labels_and_bad_masks_fail():
    frame=np.zeros((2,3,4))
    with pytest.raises(ValueError,match='teacher'):
        build_sparse_detection_target(frame,[],[[0,0,0]])
    with pytest.raises(ValueError,match='boolean'):
        build_sparse_detection_target(frame,[],[],background_mask=np.zeros(frame.shape))


def test_unknown_voxels_do_not_change_masked_loss():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 1, 2, 2, 2))
    target = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    target[0, 0, 0, 0, 0] = 1
    mask[0, 0, 0, 0, 0] = 1
    mask[0, 0, 1, 1, 1] = 1
    first = masked_balanced_bce(logits, target, mask)
    logits[0, 0, 0, 1, 1] = 100
    second = masked_balanced_bce(logits, target, mask)
    assert torch.allclose(first, second)
