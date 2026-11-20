"""Optional learned parent-link cost used by the production linker.

The bundled checkpoint was trained on all 199 competition GEFF annotation
graphs. It uses annotated coordinates only; it is not an image detector.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np


@lru_cache(maxsize=4)
def _load_model(path: str):
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "The learned linker requires PyTorch; install requirements-training.txt"
        ) from exc

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    state = checkpoint["model"]
    linear_indices = sorted(int(key.split(".")[0]) for key in state if key.endswith(".weight"))
    modules = OrderedDict()
    last = linear_indices[-1]
    for index in range(last + 1):
        weight = state.get(f"{index}.weight")
        if weight is not None:
            modules[str(index)] = nn.Linear(weight.shape[1], weight.shape[0])
        else:
            modules[str(index)] = nn.ReLU()
    model = nn.Sequential(modules)
    model.load_state_dict(state)
    model.eval()
    return model, mean, std


def build_learned_link_cost(
    prev_phys: np.ndarray,
    curr_phys: np.ndarray,
    checkpoint_path: Path,
    gate_um: float = 20.0,
    prev_image_features: Optional[np.ndarray] = None,
    curr_image_features: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return learned BCE costs and physical distances for Hungarian linking."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Learned-linker checkpoint not found: {path}")
    if not len(prev_phys) or not len(curr_phys):
        shape = (len(prev_phys), len(curr_phys))
        return np.empty(shape), np.empty(shape)
    if len(prev_phys) * len(curr_phys) > 8_000_000:
        raise ValueError('Learned linking exceeds the bounded dense-matrix budget (8 million pairs). Use spatially partitioned candidates or a smaller region; no fallback links were generated.')

    import torch

    model, mean, std = _load_model(str(path.resolve()))
    # Retain the public dense distance/cost contract, but never build N*M*F features.
    dist = np.empty((len(prev_phys), len(curr_phys)), dtype=np.float64)
    for target in range(len(curr_phys)):
        dist[:, target] = np.linalg.norm(curr_phys[target] - prev_phys, axis=1)
    input_width = len(mean)
    if input_width not in (10, 25):
        raise ValueError(f"Unsupported learned-linker feature width: {input_width}")
    if input_width == 25 and (prev_image_features is None or curr_image_features is None):
        raise ValueError("Image-aware checkpoint requires source and target image features")
    cost = np.full(dist.shape, 1e6, dtype=np.float64)
    for target in range(len(curr_phys)):
        candidates = np.flatnonzero(dist[:, target] <= gate_um)
        if not len(candidates):
            continue
        order = np.argsort(dist[:, target], kind="stable")
        rank = np.empty(len(prev_phys), dtype=np.float32)
        rank[order] = np.arange(len(prev_phys), dtype=np.float32)
        d = (curr_phys[target] - prev_phys[candidates]).astype(np.float32)
        r = dist[candidates, target].astype(np.float32)
        geometry = np.column_stack(
            [d, np.abs(d), r, r * r, rank[candidates], np.full(len(candidates), len(prev_phys))]
        )
        if input_width == 25:
            src = np.asarray(prev_image_features, dtype=np.float32)[candidates]
            dst = np.repeat(np.asarray(curr_image_features[target], dtype=np.float32)[None, :], len(candidates), axis=0)
            features = np.column_stack([geometry, src, dst, np.abs(src - dst)])
        else:
            features = geometry.astype(np.float32)
        flat = ((features.astype(np.float32) - mean) / std).astype(np.float32)
        with torch.no_grad():
            logits = model(torch.from_numpy(flat)).squeeze(1).numpy()
        cost[candidates, target] = np.logaddexp(0.0, -logits)
    return cost.astype(np.float64, copy=False), dist.astype(np.float64, copy=False)


def extract_local_image_features(frame: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Match the five patch features used by the Kaggle image-aware linker."""
    if not len(coords):
        return np.empty((0, 5), dtype=np.float32)
    sampled = frame[::4, ::8, ::8].astype(np.float32)
    lo, hi = np.percentile(sampled, [1, 99.8])
    denom = max(float(hi - lo), 1.0)
    result = np.zeros((len(coords), 5), dtype=np.float32)
    for index, point in enumerate(coords):
        z, y, x = np.rint(point).astype(int)
        z = int(np.clip(z, 0, frame.shape[0] - 1))
        y = int(np.clip(y, 0, frame.shape[1] - 1))
        x = int(np.clip(x, 0, frame.shape[2] - 1))
        patch = frame[
            max(0, z - 1):min(frame.shape[0], z + 2),
            max(0, y - 3):min(frame.shape[1], y + 4),
            max(0, x - 3):min(frame.shape[2], x + 4),
        ].astype(np.float32)
        patch = np.clip((patch - lo) / denom, 0, 2)
        result[index] = [
            patch.mean(), patch.std(), patch.max(), np.percentile(patch, 75),
            float((patch > .5).mean()),
        ]
    return result
