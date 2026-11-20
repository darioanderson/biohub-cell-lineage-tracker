from pathlib import Path

import numpy as np

from biohub.cache import cached_detection, detector_config_hash
from biohub.config import Config


def test_detection_cache_reuses_exact_input_and_config(tmp_path):
    zarr = tmp_path / "sample.zarr"
    chunk = zarr / "0" / "c" / "0" / "0" / "0"
    chunk.parent.mkdir(parents=True)
    (zarr / "zarr.json").write_text("{}")
    (zarr / "0" / "zarr.json").write_text("{}")
    chunk.write_bytes(b"real compressed frame bytes")
    cfg = Config(detection_cache_dir=tmp_path / "cache")
    calls = []

    def compute():
        calls.append(1)
        return np.ones((1, 3)), np.ones(1), np.ones(1)

    first = cached_detection(zarr, 0, cfg, None, compute)
    second = cached_detection(zarr, 0, cfg, None, compute)
    assert len(calls) == 1
    assert np.array_equal(first[0], second[0])


def test_detector_hash_changes_for_detector_parameter_only():
    base = Config()
    assert detector_config_hash(base) != detector_config_hash(base.copy_with(thresh_rel=.4))
    assert detector_config_hash(base) == detector_config_hash(base.copy_with(max_link_dist_um=99))
