"""Train a temporal 3D U-Net on all paired Biohub volumes without fake labels.

Sparse annotations are positives. Explicit background masks provide negatives;
all other voxels are unknown, including dark voxels and unverified DoG peaks.
Validation is leave-one-embryo-out and reports candidate versus DoG recall on the
same annotated cells. A final checkpoint is trained on all samples only after the
cross-validation epochs have completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from biohub.config import Config
from biohub.background_review import load_reviewed_background
from biohub.data import (
    find_geff_path, list_datasets, load_geff_graph, load_volume,
    read_physical_metadata, read_zarr_meta,
)
from biohub.detection import detect_cells
from biohub.learned_detector import (
    build_temporal_detection_model, extract_probability_peaks,
)
from biohub.sparse_detector_training import (
    build_sparse_detection_target, masked_balanced_bce,
)


@dataclass(frozen=True)
class FrameSpec:
    name: str
    embryo: str
    t: int
    annotated: np.ndarray


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def inventory(train_dir: Path) -> tuple[list[FrameSpec], dict]:
    specs, sample_counts = [], {}
    names = [name for name in list_datasets(train_dir) if find_geff_path(train_dir, name)]
    for name in names:
        nodes, _ = load_geff_graph(find_geff_path(train_dir, name))
        if nodes is None or nodes.empty:
            continue
        embryo = name.split("_", 1)[0]
        sample_counts[embryo] = sample_counts.get(embryo, 0) + 1
        for t, group in nodes.groupby("t", sort=True):
            specs.append(FrameSpec(
                name=name, embryo=embryo, t=int(t),
                annotated=group[["z", "y", "x"]].to_numpy(np.float64),
            ))
    return specs, {
        "paired_samples": len(names), "embryos": sample_counts,
        "annotated_frames": len(specs),
    }


def _normalise(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    low, high = np.quantile(volume, (0.001, 0.999))
    return np.clip((volume - low) / max(float(high - low), 1e-6), 0.0, None)


def _teacher_cache_path(cache_dir: Path, spec: FrameSpec, cfg: Config) -> Path:
    key = hashlib.sha256(
        f"{spec.name}:{spec.t}:{cfg.dog_sigma_small_um}:{cfg.dog_sigma_large_um}:{cfg.thresh_rel}".encode()
    ).hexdigest()[:16]
    return cache_dir / f"{key}.npy"


def _load_frame(train_dir: Path, spec: FrameSpec):
    path = train_dir / f"{spec.name}.zarr"
    shape, dtype = read_zarr_meta(path)
    volume = load_volume(path, spec.t, shape, dtype)
    neighbor_t = min(spec.t + 1, shape[0] - 1)
    if neighbor_t == spec.t and spec.t:
        neighbor_t = spec.t - 1
    neighbor = load_volume(path, neighbor_t, shape, dtype)
    voxel, _ = read_physical_metadata(path)
    return volume, neighbor, voxel or Config().voxel_size_zyx_um


def _teacher(train_dir: Path, spec: FrameSpec, volume: np.ndarray, cache_dir: Path, cfg: Config):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _teacher_cache_path(cache_dir, spec, cfg)
    if path.is_file():
        return np.load(path)
    coords, _, _ = detect_cells(volume, cfg)
    np.save(path, coords)
    return coords


def _recall_and_error(pred: np.ndarray, truth: np.ndarray, scale, gate=7.0):
    if not len(truth):
        return 1.0, 0.0
    if not len(pred):
        return 0.0, float("nan")
    dist = np.linalg.norm(
        (pred[:, None, :] - truth[None, :, :]) * np.asarray(scale)[None, None, :], axis=2
    )
    rows, cols = linear_sum_assignment(dist)
    accepted = dist[rows, cols] <= gate
    matched = dist[rows[accepted], cols[accepted]]
    return float(len(matched) / len(truth)), float(matched.mean()) if len(matched) else float("nan")


def _model_peaks(model, pair, downsample, scale, threshold, pool_um):
    with torch.no_grad():
        probabilities = torch.sigmoid(_annotated_frame_logits(model,pair)[0, 0]).cpu().numpy()
    coords, scores = extract_probability_peaks(
        probabilities, threshold, pool_um,
        np.asarray(scale) * np.asarray(downsample),
    )
    return coords * np.asarray(downsample), scores


def _pair_tensor(volume, neighbor, downsample, device):
    ds = tuple(int(v) for v in downsample)
    frames = [_normalise(item)[::ds[0], ::ds[1], ::ds[2]] for item in (volume, neighbor)]
    return torch.from_numpy(np.stack(frames))[None, :, None].to(device)


def _annotated_frame_logits(model, pair):
    """Return only t=0 logits; the neighbouring frame has no target here."""
    output = model(pair)
    if output.ndim == 5 and output.shape[1] == 1:
        return output
    if output.ndim != 6 or output.shape[1] < 1:
        raise ValueError(f"Detector must return (B,1,Z,Y,X) or (B,T,1,Z,Y,X), got {tuple(output.shape)}")
    if output.shape[2] != 1:
        raise ValueError('Detector output must have one detection channel')
    return output[:, 0]


def train_fold(
    train_dir: Path, train_specs: list[FrameSpec], val_specs: list[FrameSpec],
    output_dir: Path, cache_dir: Path, epochs: int, seed: int,
    downsample=(1, 4, 4), threshold=0.5, pool_um=5.0, background_dir=None,
):
    if background_dir is None:
        raise ValueError('Explicit verified background masks are required for supervised detector training')
    if epochs < 1:
        raise ValueError('epochs must be positive')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_temporal_detection_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dog_cfg = Config().competition_v10_dog_preset()
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best_score, best_epoch = [], -np.inf, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(train_specs)
        random.shuffle(order)
        losses = []
        for index, spec in enumerate(order, 1):
            volume, neighbor, scale = _load_frame(train_dir, spec)
            background = load_reviewed_background(
                Path(background_dir) / f'{spec.name}_{spec.t}.json',
                volume, spec.name, spec.t, spec.annotated)
            normalized = _normalise(volume)
            target, known = build_sparse_detection_target(
                normalized, spec.annotated, np.empty((0,3)), downsample=downsample,
                background_mask=background,
            )
            if not np.any(known * (1-target)):
                raise ValueError(f'No verified background supervision after sampling for {spec.name}:{spec.t}')
            pair = _pair_tensor(volume, neighbor, downsample, device)
            # Only the first temporal output has this FrameSpec's sparse labels.
            # Supervising the neighbour output with the same target would invent
            # labels and silently broadcast the loss across two different frames.
            logits = _annotated_frame_logits(model, pair)
            target_t = torch.from_numpy(target)[None, None].to(device)
            known_t = torch.from_numpy(known)[None, None].to(device)
            loss = masked_balanced_bce(logits, target_t, known_t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            if index % 25 == 0 or index == len(order):
                print(f"TRAIN epoch={epoch} frame={index}/{len(order)} loss={np.mean(losses[-25:]):.6f}", flush=True)

        model.eval()
        candidate_recall, baseline_recall, errors, counts = [], [], [], []
        for index, spec in enumerate(val_specs, 1):
            volume, neighbor, scale = _load_frame(train_dir, spec)
            teacher = _teacher(train_dir, spec, volume, cache_dir, dog_cfg)
            pair = _pair_tensor(volume, neighbor, downsample, device)
            predicted, _ = _model_peaks(model, pair, downsample, scale, threshold, pool_um)
            recall, error = _recall_and_error(predicted, spec.annotated, scale)
            base, _ = _recall_and_error(teacher, spec.annotated, scale)
            candidate_recall.append(recall)
            baseline_recall.append(base)
            if np.isfinite(error):
                errors.append(error)
            counts.append(len(predicted))
            if index % 25 == 0 or index == len(val_specs):
                print(f"VALIDATE epoch={epoch} frame={index}/{len(val_specs)}", flush=True)
        row = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "candidate_recall": float(np.mean(candidate_recall)) if candidate_recall else None,
            "dog_recall": float(np.mean(baseline_recall)) if baseline_recall else None,
            "localization_error_um": float(np.mean(errors)) if errors else None,
            "mean_predicted_cells": float(np.mean(counts)) if counts else None,
        }
        history.append(row)
        score = -row["train_loss"]
        # Fixed last epoch: the held-out embryo must not choose its checkpoint.
        if epoch == epochs:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), output_dir / "edge_predictor_last.pth")
            (output_dir / "config.json").write_text(json.dumps({
                "unet_out_channels": 32, "unet_layers": [32, 64, 128],
                "downsample": list(downsample), "threshold": threshold,
                "pool_kernel_um": pool_um,
            }, indent=2))
        _atomic_json(output_dir / "history.json", {"rows": history, "best_epoch": best_epoch})
        print("EPOCH", json.dumps(row), flush=True)
    return {"best_epoch": best_epoch, "best_score": best_score, "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sparse_temporal_detector"))
    parser.add_argument("--cache-dir", type=Path, default=Path("results/teacher_cache"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--background-dir', type=Path, required=True,
                        help='Reviewed image-bound background records named SAMPLE_TIME.json')
    args = parser.parse_args()
    started = time.time()
    specs, manifest = inventory(args.train_dir)
    missing=[str(args.background_dir / f'{s.name}_{s.t}.json') for s in specs
             if not (args.background_dir / f'{s.name}_{s.t}.json').is_file()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} verified background masks; first: {missing[0]}')
    embryos = sorted({spec.embryo for spec in specs})
    if len(embryos) < 2:
        raise RuntimeError(f"Need at least two embryo prefixes; found {embryos}")
    _atomic_json(args.output_dir / "status.json", {
        "state": "cross_validation", "inventory": manifest, "embryos": embryos,
    })
    folds = {}
    for heldout in embryos:
        folds[heldout] = train_fold(
            args.train_dir,
            [s for s in specs if s.embryo != heldout],
            [s for s in specs if s.embryo == heldout],
            args.output_dir / f"fold_{heldout}", args.cache_dir,
            args.epochs, args.seed + len(folds),
            background_dir=args.background_dir,
        )
    selected_epochs = args.epochs
    final = train_fold(
        args.train_dir, specs, [], args.output_dir / "final_all_199", args.cache_dir,
        selected_epochs, args.seed + 100,
        background_dir=args.background_dir,
    )
    summary = {
        "inventory": manifest, "folds": folds, "final": final,
        "selected_final_epochs": selected_epochs,
        "runtime_seconds": time.time() - started,
        "label_policy": "annotated centers positive; explicit verified background masks negative; all other voxels ignored; fixed last epoch",
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    _atomic_json(args.output_dir / "status.json", {"state": "complete", **summary})
    print("COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
