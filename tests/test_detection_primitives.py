from __future__ import annotations

import numpy as np

from biohub.config import Config
from biohub.detection import physical_nms, refine_centroid


def test_physical_nms_uses_micrometre_spacing() -> None:
    coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 10., 0.]])
    scores = np.array([3., 2., 1.])
    kept, _ = physical_nms(coords, scores, radius_um=2.0, scale=np.array([1.625, .40625, .40625]))
    assert kept.tolist() == [[0., 0., 0.], [0., 10., 0.]]


def test_scale_conversion_is_exact() -> None:
    voxel = np.array([2., 4., 8.])
    assert np.allclose(voxel * Config().scale_array, [3.25, 1.625, 3.25])


def test_refine_centroid_recovers_known_gaussian() -> None:
    center = np.array([4.2, 8.3, 7.7])
    zz, yy, xx = np.indices((10, 16, 16))
    volume = np.exp(-((zz-center[0])**2 + (yy-center[1])**2 + (xx-center[2])**2) / 4)
    found, _ = refine_centroid(volume, np.rint(center), Config(refine_radius_z=3, refine_radius_yx=4))
    assert np.linalg.norm(found - center) < 0.5
