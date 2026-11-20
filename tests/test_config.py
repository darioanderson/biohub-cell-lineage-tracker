import pytest

from biohub.config import Config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"voxel_size_zyx_um": (1, 0, 1)},
        {"frame_interval_s": 0},
        {"thresh_rel": 1.1},
        {"border_keep_quantile": -0.1},
        {"max_link_dist_um": -1},
        {"dog_sigma_small_um": 3, "dog_sigma_large_um": 2},
        {"detector_backend": "imaginary"},
        {"learned_detector_downsample": (1, 0, 4)},
        {"learned_detector_threshold": 1.1},
    ],
)
def test_invalid_config_fails_immediately(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
