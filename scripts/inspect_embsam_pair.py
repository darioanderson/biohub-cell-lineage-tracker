"""Bounded download and inspection; never invent missing time or Z axes."""
import hashlib
import json
from pathlib import Path
import struct
import zipfile
import zlib

import tifffile
from remote_zip import RangeReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/embsam_control_rnai_candidate'
URL = ('https://drive.usercontent.google.com/download?'
       'id=1M6Ety0AljmbgjvD8QFWGgniSxC6ObkKB&export=download&confirm=t')


def main():
    reader = RangeReader(URL, budget=140*1024**2)
    archive = zipfile.ZipFile(reader)
    records = {}
    for group, name in [('control', 'Emb20'), ('perturbed', 'Emb21')]:
        info = archive.getinfo(f'Emb10_30/{name}/{name}.tif')
        path = OUT / f'{name}.tif'
        if not path.exists():
            reader.seek(info.header_offset)
            header = struct.unpack('<4s5H3L2H', reader.read(30))
            assert header[0] == b'PK\x03\x04'
            reader.read(header[-2] + header[-1])
            decoder = zlib.decompressobj(-15) if info.compress_type == 8 else None
            assert info.compress_type in (0, 8)
            remaining, crc, total = info.compress_size, 0, 0
            temporary = path.with_suffix('.download')
            with temporary.open('wb') as target:
                while remaining:
                    block = reader.read(min(8*1024**2, remaining))
                    remaining -= len(block)
                    raw = decoder.decompress(block) if decoder else block
                    target.write(raw)
                    crc, total = zlib.crc32(raw, crc), total + len(raw)
                    print(name, 'compressed bytes remaining', remaining, flush=True)
                if decoder:
                    raw = decoder.flush()
                    target.write(raw)
                    crc, total = zlib.crc32(raw, crc), total + len(raw)
            assert crc == info.CRC and total == info.file_size
            temporary.replace(path)
        with tifffile.TiffFile(path) as image:
            series = image.series[0]
            # Decode all pixels to check integrity, without synthesizing axes.
            pixels = series.asarray()
            records[group] = dict(
                file=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                shape=list(pixels.shape), axes=series.axes, pages=len(image.pages),
                description=image.pages[0].description,
                imagej_metadata=str(image.imagej_metadata),
                decoded_all_pixels=True,
                app_4d_compatibility='unverified; source axes must be interpreted before conversion')
        print(group, json.dumps(records[group]), flush=True)
        (OUT / 'INSPECTION.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
