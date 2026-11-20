from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


_PATH = Path(__file__).parents[1] / "scripts" / "train_sparse_temporal_detector.py"
_SPEC = importlib.util.spec_from_file_location("train_sparse_temporal_detector", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _DummyTemporal(torch.nn.Module):
    def forward(self, pair):
        return torch.zeros((pair.shape[0], pair.shape[1], 1, *pair.shape[-3:]))


def test_loss_logits_select_only_annotated_frame() -> None:
    pair = torch.zeros((1, 2, 1, 3, 4, 5))
    logits = _MODULE._annotated_frame_logits(_DummyTemporal(), pair)
    assert logits.shape == (1, 1, 3, 4, 5)


def test_atomic_json_creates_parent_directory(tmp_path) -> None:
    path = tmp_path / "new" / "status.json"
    _MODULE._atomic_json(path, {"state": "ok"})
    assert path.read_text().strip().startswith("{")


def test_real_detector_can_backpropagate_sparse_supervision():
    from biohub.learned_detector import build_temporal_detection_model
    from biohub.sparse_detector_training import masked_balanced_bce
    model=build_temporal_detection_model(out_channels=4,layers=(4,8))
    pair=torch.randn((1,2,1,8,8,8))
    logits=_MODULE._annotated_frame_logits(model,pair)
    assert logits.shape==(1,1,8,8,8)
    target=torch.zeros_like(logits);mask=torch.zeros_like(logits)
    target[0,0,4,4,4]=1;mask[0,0,4,4,4]=1;mask[0,0,0,0,0]=1
    loss=masked_balanced_bce(logits,target,mask);loss.backward()
    assert torch.isfinite(loss)
    assert model.detect_head.weight.grad is not None
