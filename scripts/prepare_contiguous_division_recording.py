"""Make a bounded-size, contiguous real movie spanning published divisions.

Uses only the already downloaded Emb8 archive. XY is reduced by averaging
2x2 pixels, preserving all Z slices and all timepoints in the chosen window.
Published reference names are never used to manufacture model predictions.
"""
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/real_control_variant_large/source'
OUT = ROOT / 'data/real_contiguous_divisions'


def main():
    OUT.mkdir(exist_ok=True)
    frames = list(range(55, 91))
    destination = OUT / 'CONTROL_Emb8_t055_090_XY2.npy'
    movie = np.lib.format.open_memmap(destination, mode='w+', dtype=np.uint8,
                                    shape=(len(frames), 56, 256, 356))
    with zipfile.ZipFile(SOURCE / 'Emb8_Raw_Nucleus.zip') as archive:
        members = {}
        for member in archive.infolist():
            match = re.search(r'-t(\d+)-p(\d+)\.tif$', member.filename)
            if match and int(match[1]) in frames:
                key = (int(match[1]), int(match[2]))
                assert key not in members, f'Duplicate source plane {key}'
                members[key] = member
        assert len(members) == len(frames) * 56
        for index, frame in enumerate(frames):
            for z in range(56):
                image = tifffile.imread(io.BytesIO(archive.read(members[frame, z+1])))
                assert image.shape == (512, 712) and image.dtype == np.uint8
                reduced = image.reshape(256, 2, 356, 2).mean(axis=(1, 3))
                movie[index, z] = np.rint(reduced).astype(np.uint8)
            print(f'Prepared source frame {frame}', flush=True)
    movie.flush()
    del movie
    assert destination.stat().st_size < 200_000_000
    tracing = pd.read_csv(SOURCE / 'Emb8_CellTracing.csv')
    reference = tracing[tracing.time.isin(frames)].copy()
    reference['clip_frame'] = reference.time - frames[0]
    reference.to_csv(OUT / 'published_reference_native_coordinates.csv', index=False)
    counts = reference.groupby('time').size().reindex(frames).tolist()
    metadata = dict(
        file=destination.name, shape=[36,56,256,356], bytes=destination.stat().st_size,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        source='https://doi.org/10.6084/m9.figshare.29064530', sample='Emb8',
        organism='C. elegans', genotype='wild-type', source_frames=frames,
        processing='2x2 XY mean, rounded uint8; all Z planes and consecutive timepoints retained',
        voxel_size_zyx_um=[0.8,0.18,0.18], frame_interval_s=10.0,
        reference_counts=counts,
        published_divisions=[dict(parent='AB',daughters=['ABa','ABp'],source_frame=64,clip_frame=9),
                             dict(parent='P1',daughters=['EMS','P2'],source_frame=82,clip_frame=27)],
        reference_coordinate_note='CSV coordinates remain in the source grid; XY must be divided by 2 for this clip',
        model_inference_performed=False,
        limitations='Real reference divisions are present; model recovery is not guaranteed. This is a window, not the full embryogenesis, and is not Biohub zebrafish data.')
    (OUT / 'PROVENANCE.json').write_text(json.dumps(metadata, indent=2))
    (OUT / 'README.txt').write_text(
        'Real contiguous wild-type C. elegans movie, 36 frames, source 55-90.\n'
        'Two published reference divisions: AB at clip frame 9; P1 at clip frame 27 (zero-based).\n'
        'Image spacing Z,Y,X: 0.8,0.18,0.18 um. Cadence: 10 seconds.\n'
        'XY reduced by 2x2 mean for size; no frames omitted.\n'
        'Reference CSV is source-coordinate data, NOT model output.\n'
        'No guarantee that the current detector/tracker recovers these divisions.\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
