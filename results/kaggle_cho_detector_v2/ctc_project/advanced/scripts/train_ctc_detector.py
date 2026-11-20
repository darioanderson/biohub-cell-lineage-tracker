"""Reproducible real-data centroid experiment. Not an official CTC evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from torch import nn
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[2]
SPACING = np.array([1.0, .202, .202])
DS_SPACING = SPACING * [1, 2, 2]
TRAIN = list(range(0, 60, 3))
VALID = [70, 77, 84, 91]
THRESHOLDS = [.15, .25, .4, .55]


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def centers(mask):
    labels = np.unique(mask)
    labels = labels[labels > 0]
    if not len(labels):
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(ndi.center_of_mass(mask > 0, mask, labels), dtype=np.float32)


def normalize(raw):
    z, y, x = raw.shape
    a = raw[:, :y//2*2, :x//2*2].astype(np.float32)
    a = a.reshape(z, y//2, 2, x//2, 2).mean((2, 4))
    lo, hi = np.percentile(a, [1, 99.8])
    return np.clip((a-lo)/max(hi-lo, 1), 0, 1).astype(np.float32)


def heatmap(shape, points):
    target = np.zeros(shape, np.float32)
    sigma = 1.0 / DS_SPACING
    for point in points:
        p = (point - [0, .5, .5]) / [1, 2, 2]
        low = np.maximum(np.floor(p-3*sigma).astype(int), 0)
        high = np.minimum(np.ceil(p+3*sigma+1).astype(int), shape)
        slices = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
        grid = np.ogrid[tuple(slice(int(a), int(b)) for a, b in zip(low, high))]
        d2 = sum(((g-c)/s)**2 for g, c, s in zip(grid, p, sigma))
        target[slices] = np.maximum(target[slices], np.exp(-.5*d2))
    return target


def load_frame(data, seq, frame):
    raw_path = data / seq / f't{frame:03d}.tif'
    label_path = data / f'{seq}_GT' / 'TRA' / f'man_track{frame:03d}.tif'
    raw = tifffile.imread(raw_path)
    labels = tifffile.imread(label_path)
    if raw.ndim != 3 or raw.shape != labels.shape:
        raise ValueError(f'Raw/label shape mismatch: {raw_path}')
    points = centers(labels)
    return raw, normalize(raw), points


def block(a, b):
    return nn.Sequential(nn.Conv3d(a, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU(),
                         nn.Conv3d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU())


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = block(1, 8)
        self.deep = block(8, 16)
        self.dec = block(24, 8)
        self.head = nn.Conv3d(8, 1, 1)

    def forward(self, x):
        a = self.enc(x)
        b = self.deep(F.avg_pool3d(a, (1, 2, 2)))
        b = F.interpolate(b, size=a.shape[2:], mode='trilinear', align_corners=False)
        return self.head(self.dec(torch.cat([a, b], 1)))


def crop(a, target, points, rng):
    size = np.array([8, 64, 64])
    pad = [(0, max(0, int(s-n))) for s, n in zip(size, a.shape)]
    a = np.pad(a, pad, mode='edge')
    target = np.pad(target, pad, mode='constant')
    if len(points) and rng.random() < .7:
        p = (points[rng.integers(len(points))] - [0, .5, .5]) / [1, 2, 2]
        start = p.astype(int) - size//2 + rng.integers(-8, 9, 3)
    else:
        start = np.array([rng.integers(max(1, n-s+1)) for n, s in zip(a.shape, size)])
    start = np.clip(start, 0, np.array(a.shape)-size)
    sl = tuple(slice(int(k), int(k+s)) for k, s in zip(start, size))
    x, y = a[sl], target[sl]
    # XY flips preserve the anisotropic Z spacing.
    for axis in [1, 2]:
        if rng.random() < .5:
            x, y = np.flip(x, axis), np.flip(y, axis)
    return x.copy(), y.copy()


@torch.inference_mode()
def predict(model, a, device):
    model.eval()
    # CHO volumes fit comfortably; refuse unexpected large volumes rather than OOM silently.
    if a.size > 8_000_000:
        raise ValueError('Volume too large for this initial full-volume inference adapter')
    x = torch.from_numpy(a)[None, None].to(device)
    return model(x).sigmoid()[0, 0].cpu().numpy()


def decode(prob, threshold):
    peaks = peak_local_max(prob, min_distance=2, threshold_abs=threshold, exclude_border=False)
    if not len(peaks):
        return np.empty((0, 3), np.float32)
    scores = prob[tuple(peaks.T)]
    ordered = peaks[np.argsort(-scores, kind='stable')].astype(float)
    physical = ordered * DS_SPACING
    suppressed = np.zeros(len(ordered), dtype=bool)
    selected = []
    for index, point in enumerate(ordered):
        if suppressed[index]:
            continue
        selected.append(point)
        suppressed |= ((physical-physical[index])**2).sum(axis=1) < 2.5**2
    return np.asarray(selected) * [1, 2, 2] + [0, .5, .5]


def counts(pred, truth, gate=3.):
    if not len(pred) or not len(truth):
        return [0, len(pred), len(truth)]
    distances = np.linalg.norm((pred[:, None]-truth[None])*SPACING, axis=2)
    # Invalid assignments cost more than all valid assignments combined: maximize valid matches first.
    penalty = (min(distances.shape)+1)*(gate+1)
    cost = np.where(distances <= gate, distances, penalty)
    ri, ci = linear_sum_assignment(cost)
    tp = int((distances[ri, ci] <= gate).sum())
    return [tp, len(pred)-tp, len(truth)-tp]


def metrics(rows):
    tp, fp, fn = np.sum(rows, axis=0).astype(int).tolist()
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp/max(tp+fp, 1),
            'recall': tp/max(tp+fn, 1), 'f1': 2*tp/max(2*tp+fp+fn, 1)}


def validate(model, frames, device):
    predictions = [(predict(model, a, device), truth) for _, a, truth in frames]
    values = [(metrics([counts(decode(p, t), g) for p, g in predictions]), t) for t in THRESHOLDS]
    return max(values, key=lambda item: item[0]['f1'])


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'last.pt').exists() and not args.resume:
        raise ValueError('Existing run: use --resume or choose a new output directory')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise RuntimeError('CUDA required for this configured training run; GPU check failed')
    rng = np.random.default_rng(42)
    model = Detector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    train_frames = [load_frame(args.data, '01', t) for t in TRAIN]
    train_data = [(a, heatmap(a.shape, g), g) for _, a, g in train_frames]
    valid = [load_frame(args.data, '01', t) for t in VALID]
    provenance = {'train_sequence': '01', 'train_frames': TRAIN, 'validation_sequence': '01',
                  'validation_frames': VALID, 'held_out_sequence': '02', 'seed': 42,
                  'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
                  'script_sha256': digest(__file__), 'parameters': sum(p.numel() for p in model.parameters()),
                  'dataset_archive_sha256': digest(args.data.parent/'Fluo-N3DH-CHO.zip'),
                  'metric': 'local centroid F1 at 3 um; NOT official SEG or TRA',
                  'max_epochs': args.epochs, 'steps_per_epoch': args.steps,
                  'input_hashes': {str(p.relative_to(args.data)): digest(p)
                      for t in TRAIN+VALID for p in [args.data/'01'/f't{t:03d}.tif',
                          args.data/'01_GT'/'TRA'/f'man_track{t:03d}.tif']}}
    start, best, stale = 0, -1., 0
    if args.resume:
        ck = torch.load(out/'last.pt', map_location=device, weights_only=False)
        if ck['script_sha256'] != provenance['script_sha256']:
            raise ValueError('Training script changed; use a new experiment directory')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start, best, stale = ck['epoch'], ck['best'], ck['stale']
        rng.bit_generator.state = ck['numpy_rng']
        torch.set_rng_state(ck['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
    else:
        atomic_json(out/'provenance.json', provenance)
        initial, threshold = validate(model, valid, device)
        atomic_json(out/'untrained_validation.json', {'metrics': initial, 'threshold': threshold})
    print(json.dumps({'status': 'training', **provenance}), flush=True)
    started = time.time()
    for epoch in range(start+1, args.epochs+1):
        model.train()
        losses = []
        for _ in range(args.steps):
            a, target, g = train_data[rng.integers(len(train_data))]
            x, y = crop(a, target, g, rng)
            x = torch.from_numpy(x)[None, None].to(device)
            y = torch.from_numpy(y)[None, None].to(device)
            optimizer.zero_grad(set_to_none=True)
            prob = model(x).sigmoid()
            loss = ((prob-y).square()*(1+30*y)).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        score, threshold = validate(model, valid, device)
        improved = score['f1'] > best + 1e-5
        stale = 0 if improved else stale+1
        if improved:
            best = score['f1']
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'threshold': threshold,
                        'validation': score, 'script_sha256': provenance['script_sha256']}, out/'best.tmp')
            (out/'best.tmp').replace(out/'best.pt')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                    'best': best, 'stale': stale, 'numpy_rng': rng.bit_generator.state,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    'script_sha256': provenance['script_sha256']}, out/'last.tmp')
        (out/'last.tmp').replace(out/'last.pt')
        row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'validation': score,
               'threshold': threshold, 'best_validation_f1': best, 'seconds': time.time()-started}
        with (out/'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
        atomic_json(out/'status.json', {'state': 'training', 'pid': os.getpid(), **row})
        print(json.dumps(row), flush=True)
        if epoch >= 15 and stale >= 12:
            break
    atomic_json(out/'status.json', {'state': 'training_complete', 'pid': os.getpid(), **row,
                                   'best_checkpoint_sha256': digest(out/'best.pt')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=PROJECT/'data'/'Fluo-N3DH-CHO')
    parser.add_argument('--output', type=Path, default=PROJECT/'experiments'/'cho-centroid-001')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output/'status.json', {'state': 'failed', 'error': str(exc)})
        raise
