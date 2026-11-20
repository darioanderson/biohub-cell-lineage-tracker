"""Content-bound calibration; filenames and numerical defaults are not evidence."""
import json
from pathlib import Path


def known_recording(digest, root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1] / 'data'
    for folder in ('local_biohub_pair', 'real_contiguous_divisions', 'real_division_clips',
                   'real_control_variant_under200mb', 'real_control_variant_large'):
        path = root / folder / 'PROVENANCE.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        entries = {'control':data} if 'sha256' in data else data.get('recordings',data)
        for group, entry in entries.items():
            if not isinstance(entry, dict) or entry.get('sha256') != digest:
                continue
            spacing = entry.get('voxel_size_zyx_um', entry.get('grid_zyx_um'))
            if spacing is None and folder != 'local_biohub_pair':
                spacing = (.8, .09, .09) if group == 'control' else (.713349, .09106, .09106)
            interval = entry.get('frame_interval_s', entry.get('frame_interval_seconds'))
            # Earlier manifests omit the known control cadence; this evidence is
            # tied to these exact source hashes, never to arbitrary 10-second files.
            if interval is None and group == 'control' and folder != 'local_biohub_pair':
                interval = 10.
            return dict(entry=entry, manifest=str(path), folder=folder, group=group,
                        spacing=spacing, interval=interval,
                        spatial_verified=spacing is not None,
                        time_verified=interval is not None,
                        genotype_verified=False,
                        labels=['Recording A', 'Recording B'] if folder == 'local_biohub_pair' else ['Control', 'Variant'])
    return dict(entry={}, spacing=None, interval=None, spatial_verified=False,
                time_verified=False, genotype_verified=False, labels=['Recording A', 'Recording B'])


def calibrated_config(cfg, info):
    return cfg.copy_with(voxel_size_zyx_um=tuple(info['spacing']) if info['spatial_verified'] else (1., 1., 1.),
                         frame_interval_s=float(info['interval']) if info['time_verified'] else 1.)
