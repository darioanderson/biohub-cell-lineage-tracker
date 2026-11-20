"""Convert existing local zebrafish data; never assign invented genotypes."""
import hashlib
import json
from pathlib import Path

import numpy as np
from biohub.data import read_zarr_meta, read_physical_metadata, load_volume

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('C:/Users/Admin/Desktop/biohub_data/train')
OUT = ROOT / 'data/local_biohub_pair'


def main():
    OUT.mkdir(exist_ok=True)
    records = {}
    for slot, name in [('control', '44b6_0113de3b'), ('variant', '44b6_0b24845f')]:
        source = SOURCE / f'{name}.zarr'
        shape, dtype = read_zarr_meta(source)
        path = OUT / f'{name}_all100.npy'
        if not path.exists():
            temporary = path.with_suffix('.partial.npy')
            movie = np.lib.format.open_memmap(temporary, mode='w+', dtype=dtype, shape=shape)
            for t in range(shape[0]):
                movie[t] = load_volume(source, t, shape, dtype)
            movie.flush()
            del movie
            temporary.replace(path)
        movie = np.load(path, mmap_mode='r', allow_pickle=False)
        assert movie.shape == shape and movie.dtype == dtype
        # Check all converted pixels against the source, one frame at a time.
        for t in range(shape[0]):
            assert np.array_equal(movie[t], load_volume(source, t, shape, dtype))
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8*1024**2), b''):
                digest.update(block)
        spacing, interval = read_physical_metadata(source)
        records[slot] = dict(file=path.name, sha256=digest.hexdigest(),
            shape=list(shape), bytes=path.stat().st_size, source_zarr=str(source),
            reference_geff=str(SOURCE/f'{name}.geff'), genotype='unverified',
            voxel_size_zyx_um=spacing, frame_interval_s=interval,
            source_frames=list(range(shape[0])), all_pixels_verified=True)
        print(name, shape, 'all frames verified', flush=True)
    (OUT/'PROVENANCE.json').write_text(json.dumps(dict(recordings=records,
        organism='zebrafish', purpose='Exploratory recording comparison, not verified control versus variant',
        same_embryo_prefix=True, stage_alignment_verified=False,
        genotype_verified=False, no_download_performed=True), indent=2))


if __name__ == '__main__':
    main()
