from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from biohub.learned_linker import build_learned_link_cost


def test_bundled_checkpoint_produces_finite_gated_costs() -> None:
    checkpoint = Path("models/full_annotation_linker/final_all_annotations.pt")
    prev = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 5.0]])
    curr = np.array([[0.0, 0.5, 0.5], [0.0, 100.0, 100.0]])
    cost, distance = build_learned_link_cost(prev, curr, checkpoint, gate_um=20.0)
    assert cost.shape == (2, 2)
    assert distance.shape == (2, 2)
    assert np.isfinite(cost[:, 0]).all()
    assert (cost[:, 1] == 1e6).all()


def test_image_checkpoint_requires_and_uses_25_features(tmp_path) -> None:
    model = nn.Sequential(nn.Linear(25, 64), nn.ReLU(), nn.Dropout(.05), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    checkpoint = tmp_path / "image.pt"
    torch.save({"model": model.state_dict(), "mean": [0.0] * 25, "std": [1.0] * 25}, checkpoint)
    prev = np.array([[0.0, 0.0, 0.0]])
    curr = np.array([[0.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="requires"):
        build_learned_link_cost(prev, curr, checkpoint)
    cost, _ = build_learned_link_cost(
        prev, curr, checkpoint,
        prev_image_features=np.ones((1, 5), np.float32),
        curr_image_features=np.ones((1, 5), np.float32),
    )
    assert np.isfinite(cost).all()
