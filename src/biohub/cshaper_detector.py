"""Inference adapter for the saved Kaggle CShaper center detector."""
from pathlib import Path

import numpy as np
from skimage.feature import peak_local_max


class CShaperDetector:
    def __init__(self, cfg):
        import torch
        from torch import nn
        from torch.nn import functional as functional

        def block(inputs, outputs):
            return nn.Sequential(
                nn.Conv3d(inputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.SiLU(),
                nn.Conv3d(outputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.SiLU(),
            )

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = block(1, 8)
                self.deep = block(8, 16)
                self.dec = block(24, 8)
                self.head = nn.Conv3d(8, 1, 1)

            def forward(self, image):
                first = self.enc(image)
                second = self.deep(functional.avg_pool3d(first, (1, 2, 2)))
                second = functional.interpolate(second, size=first.shape[2:], mode="trilinear", align_corners=False)
                return self.head(self.dec(torch.cat([first, second], dim=1)))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Network().to(self.device)
        checkpoint = torch.load(Path(cfg.learned_detector_path), map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def detect(self, vol, cfg, prev_count=None):
        import torch

        if vol.ndim != 3 or min(vol.shape) < 2:
            raise ValueError("CShaper inference requires a 3D volume with all dimensions >= 2.")
        low, high = np.percentile(vol, (1, 99.8))
        image = np.clip((vol.astype(np.float32)-low)/max(1, high-low), 0, 1).astype(np.float32)
        scores = np.empty_like(image)
        with torch.inference_mode():
            for start in range(0, len(image), 8):
                low_z, high_z = max(0, start-4), min(len(image), start+12)
                end = min(len(image), start+8)
                tensor = torch.as_tensor(image[low_z:high_z], dtype=torch.float32, device=self.device)[None, None]
                scores[start:end] = self.model(tensor).sigmoid()[0, 0, start-low_z:end-low_z].cpu().numpy()
        points = peak_local_max(scores, threshold_abs=0.5, min_distance=2, exclude_border=False)
        confidence = scores[tuple(points.T)]
        intensity = vol[tuple(points.T)].astype(float)
        return points.astype(float), confidence.astype(float), intensity
