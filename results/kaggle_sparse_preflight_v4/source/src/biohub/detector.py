"""Detector backends."""

from __future__ import annotations

from typing import Optional, Protocol, Tuple

import numpy as np

from biohub.config import Config
from biohub.detection import detect_cells


class CellDetector(Protocol):
    def detect(
        self,
        vol: np.ndarray,
        cfg: Config,
        prev_count: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...


class PeakDetector:
    """Classical anisotropy-aware peak detector (Gaussian or DoG band-pass)."""

    def detect(
        self,
        vol: np.ndarray,
        cfg: Config,
        prev_count: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return detect_cells(vol, cfg, prev_count=prev_count)


class DogPeakDetector(PeakDetector):
    """Peak detector with DoG band-pass preprocessing enabled."""

    def detect(
        self,
        vol: np.ndarray,
        cfg: Config,
        prev_count: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = cfg.copy_with(use_dog_bandpass=True, detector_backend="peaks_dog")
        return detect_cells(vol, cfg, prev_count=prev_count)


def get_detector(cfg: Config) -> CellDetector:
    if cfg.detector_backend == "learned":
        from biohub.learned_detector import TemporalUNetDetector

        return TemporalUNetDetector(cfg)
    if cfg.detector_backend == "peaks_dog" or cfg.use_dog_bandpass:
        return DogPeakDetector()
    return PeakDetector()
