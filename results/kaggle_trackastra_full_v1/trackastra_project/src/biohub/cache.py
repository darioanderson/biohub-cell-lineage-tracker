"""Content-addressed detection cache for repeated controlled experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

from biohub.config import Config


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def detector_config_hash(cfg: Config) -> str:
    fields = asdict(cfg)
    relevant = {
        key: _jsonable(value)
        for key, value in fields.items()
        if key.startswith(("detector_", "dog_", "thresh_", "min_", "dense_", "refine_", "nms_", "border_", "max_frame_", "voxel_size_"))
    }
    encoded = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _frame_checksum(zarr_path: Path, t: int) -> str:
    digest = hashlib.sha256()
    for path in (zarr_path / "zarr.json", zarr_path / "0" / "zarr.json", zarr_path / "0" / "c" / str(t) / "0" / "0" / "0"):
        if not path.exists():
            continue
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def cached_detection(
    zarr_path: Path,
    t: int,
    cfg: Config,
    prev_count: int | None,
    compute: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load or atomically store one detector result under an input/config hash."""
    if not cfg.use_detection_cache:
        return compute()
    payload = f"{_frame_checksum(zarr_path, t)}:{detector_config_hash(cfg)}:{prev_count}"
    key = hashlib.sha256(payload.encode()).hexdigest()
    cache_dir = Path(cfg.detection_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            return saved["coords"], saved["scores"], saved["intensities"]
    coords, scores, intensities = compute()
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, coords=coords, scores=scores, intensities=intensities)
    temp.replace(path)
    return coords, scores, intensities
