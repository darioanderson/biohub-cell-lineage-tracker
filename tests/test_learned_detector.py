import json

import numpy as np
import pytest

from biohub.config import Config
from biohub.detector import get_detector
from biohub.learned_detector import extract_probability_peaks, physical_peak_footprint


def test_physical_nms_uses_anisotropic_voxel_spacing():
    footprint = physical_peak_footprint(2.0, (2.0, 1.0, 0.5))
    assert footprint.shape == (3, 5, 9)
    assert footprint[1, 2, 4]
    assert not footprint[0, 0, 0]


def test_probability_peak_extraction_returns_scores_and_suppresses_neighbors():
    probability = np.zeros((5, 9, 9), dtype=np.float32)
    probability[2, 4, 4] = 0.9
    probability[2, 4, 5] = 0.8
    probability[2, 7, 7] = 0.7
    coords, scores = extract_probability_peaks(
        probability, threshold=0.5, pool_um=1.5, voxel_size_zyx_um=(1.0, 1.0, 1.0)
    )
    assert coords.tolist() == [[2.0, 4.0, 4.0], [2.0, 7.0, 7.0]]
    assert np.allclose(scores, [0.9, 0.7])


def test_learned_backend_never_silently_falls_back_when_checkpoint_is_missing(tmp_path):
    cfg = Config(detector_backend="learned", learned_detector_path=tmp_path / "missing.pth")
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        get_detector(cfg)


def test_organizer_compatible_checkpoint_loads_and_runs(tmp_path):
    torch = pytest.importorskip("torch")
    from biohub.temporal_unet import TemporalUNet3D

    unet = TemporalUNet3D(
        in_channels=1, out_channels=4, layers=(4, 8), gradient_checkpointing=False
    )
    head = torch.nn.Conv3d(4, 1, kernel_size=1)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(-10.0)
    state = {f"unet.{key}": value for key, value in unet.state_dict().items()}
    state.update({f"detect_head.{key}": value for key, value in head.state_dict().items()})
    checkpoint = tmp_path / "edge_predictor_best.pth"
    torch.save(state, checkpoint)
    (tmp_path / "config.json").write_text(json.dumps({
        "unet_layers": [4, 8], "unet_out_channels": 4, "downsample": [1, 1, 1]
    }))
    cfg = Config(
        detector_backend="learned", learned_detector_path=checkpoint,
        learned_detector_downsample=(1, 1, 1), learned_detector_threshold=0.5,
        learned_detector_pool_um=2.0,
    )
    coords, scores, intensity = get_detector(cfg).detect(
        np.zeros((8, 8, 8), dtype=np.uint16), cfg
    )
    assert coords.shape == (0, 3)
    assert scores.size == 0 and intensity.size == 0
