"""Checkpoint-backed temporal 3D U-Net cell detector.

The checkpoint layout and network match the Biohub competition organizers'
BSD-licensed baseline. The trained backend is opt-in; no classical fallback is
used when a learned checkpoint is requested because that would conceal errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from skimage.feature import peak_local_max

from biohub.config import Config


def build_temporal_detection_model(out_channels: int = 32, layers=(32, 64, 128)):
    """Build the checkpoint-compatible U-Net and detection head."""
    import torch.nn as nn

    from biohub.temporal_unet import TemporalUNet3D

    class DetectionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = TemporalUNet3D(
                in_channels=1, out_channels=out_channels, layers=layers,
                gradient_checkpointing=False,
            )
            self.detect_head = nn.Conv3d(out_channels, 1, kernel_size=1)

        def forward(self, pair):
            features = self.unet(pair)
            return self.detect_head(features[:, 0])

    return DetectionModel()


def physical_peak_footprint(pool_um: float, voxel_size_zyx_um) -> np.ndarray:
    """Return an ellipsoidal NMS footprint defined in physical units."""
    scale = np.asarray(voxel_size_zyx_um, dtype=np.float64)
    if scale.shape != (3,) or np.any(scale <= 0) or pool_um <= 0:
        raise ValueError("pool_um and all voxel sizes must be positive")
    radius = np.maximum(1, np.ceil(pool_um / scale).astype(int))
    grids = np.ogrid[tuple(slice(-r, r + 1) for r in radius)]
    norm = sum((grid * scale[axis] / pool_um) ** 2 for axis, grid in enumerate(grids))
    return np.asarray(norm <= 1.0, dtype=bool)


def extract_probability_peaks(
    probabilities: np.ndarray,
    threshold: float,
    pool_um: float,
    voxel_size_zyx_um,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract deterministic 3D local maxima and their probability scores."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape (Z, Y, X)")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    coords = peak_local_max(
        probabilities,
        threshold_abs=float(threshold),
        footprint=physical_peak_footprint(pool_um, voxel_size_zyx_um),
        exclude_border=False,
    )
    if not len(coords):
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)
    scores = probabilities[tuple(coords.T)].astype(np.float64)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], -scores))
    return coords[order].astype(np.float64), scores[order]


class TemporalUNetDetector:
    """Load organizer-compatible temporal U-Net weights and detect cell centers."""

    def __init__(self, cfg: Config) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError("The learned detector requires the training dependencies") from exc

        path = Path(cfg.learned_detector_path)
        if not path.is_file():
            raise FileNotFoundError(f"Learned detector checkpoint not found: {path}")
        metadata_path = path.parent / "config.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        self.downsample = tuple(metadata.get("downsample", cfg.learned_detector_downsample))
        layers = tuple(metadata.get("unet_layers", (32, 64, 128)))
        out_channels = int(metadata.get("unet_out_channels", 32))
        if cfg.learned_detector_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for the learned detector but is unavailable")
        device_name = (
            "cuda" if cfg.learned_detector_device == "auto" and torch.cuda.is_available()
            else "cpu" if cfg.learned_detector_device == "auto"
            else cfg.learned_detector_device
        )
        self.device = torch.device(device_name)

        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        state = {str(key).replace("unet.module.", "unet.", 1): value for key, value in state.items()}
        detector_state = {
            key: value for key, value in state.items()
            if key.startswith("unet.") or key.startswith("detect_head.")
        }
        model = build_temporal_detection_model(out_channels, layers)
        missing, unexpected = model.load_state_dict(detector_state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"Checkpoint is incompatible with detector config; missing={missing}, unexpected={unexpected}"
            )
        self.model = model.to(self.device).eval()
        self._torch = torch

    def detect(
        self,
        vol: np.ndarray,
        cfg: Config,
        prev_count: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = self._torch
        dz, dy, dx = self.downsample
        raw = np.asarray(vol[::dz, ::dy, ::dx], dtype=np.float32)
        low, high = np.quantile(raw, (0.001, 0.999))
        frame = np.clip((raw - low) / max(float(high - low), 1e-6), 0.0, None)
        tensor = torch.from_numpy(frame).to(self.device)
        pair = torch.stack([tensor, tensor], dim=0)[None, :, None]
        with torch.no_grad():
            logits = self.model(pair)[0, 0]
            probabilities = torch.sigmoid(logits).cpu().numpy()
        ds_scale = cfg.scale_array * np.asarray(self.downsample, dtype=np.float64)
        coords_ds, scores = extract_probability_peaks(
            probabilities, cfg.learned_detector_threshold,
            cfg.learned_detector_pool_um, ds_scale,
        )
        coords = coords_ds * np.asarray(self.downsample, dtype=np.float64)
        if len(coords) > cfg.max_nodes_per_frame:
            keep = np.argsort(scores)[-cfg.max_nodes_per_frame:]
            coords, scores = coords[keep], scores[keep]
        if len(coords):
            index = np.rint(coords).astype(int)
            index = np.clip(index, 0, np.asarray(vol.shape) - 1)
            intensities = np.asarray(vol[tuple(index.T)], dtype=np.float64)
        else:
            intensities = np.empty(0, dtype=np.float64)
        return coords, scores, intensities
