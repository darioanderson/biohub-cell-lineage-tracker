"""Compare frozen baseline with trained detector on a declared split, using identical matching."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from train_ctc_detector import (PROJECT, SPACING, VALID, Detector, atomic_json,
                                counts, decode, digest, load_frame, metrics, predict)

# Import original code, not a copy modified during training. Prevent pyc writes in baseline.
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT/'baseline'/'src'))
from biohub.config import Config
from biohub.detection import detect_cells


class CHOConfig(Config):
    @property
    def scale_array(self):
        return SPACING.copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['validation', 'held-out'], default='validation')
    p.add_argument('--checkpoint', type=Path, default=PROJECT/'experiments'/'cho-centroid-001'/'best.pt')
    p.add_argument('--baseline-only', action='store_true')
    a = p.parse_args()
    if a.split == 'held-out' and a.baseline_only:
        raise ValueError('Held-out evaluation is reserved for the completed paired comparison')
    data = PROJECT/'data'/'Fluo-N3DH-CHO'
    seq = '01' if a.split == 'validation' else '02'
    frame_ids = VALID if seq == '01' else sorted(int(f.stem[1:]) for f in (data/seq).glob('t*.tif'))[::5]
    name = f'{a.split}-'+('baseline' if a.baseline_only else a.checkpoint.parent.name)+'.json'
    report = PROJECT/'comparison'/name
    if a.split == 'held-out' and report.exists():
        raise ValueError('Held-out report already exists; do not repeatedly tune against this split')
    model = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not a.baseline_only:
        ck = torch.load(a.checkpoint, map_location=device, weights_only=True)
        model = Detector().to(device)
        model.load_state_dict(ck['model'])
    configs = {'original_defaults_with_CHO_spacing': CHOConfig(),
               'original_with_shallow_volume_adapter': CHOConfig(min_z_hard=0, border_z=0, xy_ds=2)}
    rows, times, image_hashes = [], {}, {}
    for t in frame_ids:
        raw, image, truth = load_frame(data, seq, t)
        image_hashes[str(t)] = {'raw': digest(data/seq/f't{t:03d}.tif'),
                               'labels': digest(data/f'{seq}_GT'/'TRA'/f'man_track{t:03d}.tif')}
        for name, cfg in configs.items():
            started = time.perf_counter()
            coords, _, _ = detect_cells(raw, cfg)
            times[name] = times.get(name, 0) + time.perf_counter()-started
            rows.append({'frame': t, 'method': name, 'counts': counts(coords, truth)})
        if model is not None:
            started = time.perf_counter()
            coords = decode(predict(model, image, device), ck['threshold'])
            name = 'learned_3d_centroid'
            times[name] = times.get(name, 0) + time.perf_counter()-started
            rows.append({'frame': t, 'method': name, 'counts': counts(coords, truth)})
    result = {'split': a.split, 'sequence': seq, 'frames': frame_ids, 'rows': rows,
              'metrics': {name: metrics([r['counts'] for r in rows if r['method'] == name]) for name in times},
              'inference_seconds': times, 'input_hashes': image_hashes,
              'checkpoint_sha256': digest(a.checkpoint) if model is not None else None,
              'metric_definition': 'Pooled one-to-one centroid F1 at 3 um; NOT official SEG/TRA',
              'limitations': ['CHO is not zebrafish', 'Detection only: tracking and division accuracy not evaluated',
                  'Baseline adapter is not extensively tuned', 'No independent leaderboard evaluation']}
    atomic_json(report, result)
    print(json.dumps(result['metrics'], indent=2))
    print('Saved', report)


if __name__ == '__main__':
    main()
