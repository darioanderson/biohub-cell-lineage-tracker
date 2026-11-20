"""Prepare native-grid real test clips. Compression is matched; sampling/stage is not.

Emb8 is an uncompressed wild-type embryo (EmbSAM); MT2 is a wee-1.1
mutant (CMap). Neither resampling nor stage alignment is silently performed.
"""
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import zipfile
import zlib
import numpy as np
import tifffile
from remote_zip import RangeReader

OUT=Path(__file__).resolve().parents[1]/'data/real_control_variant_large'
OUT.mkdir(parents=True,exist_ok=True)
CACHE=OUT/'source'
CACHE.mkdir(exist_ok=True)
FRAMES=30
metadata=dict(purpose='Larger real microscopy model tests; not a verified stage-aligned experiment',
 control=dict(sample='Emb8',genotype='wild-type',strain='ZZY0861',compression='uncompressed',frame_interval_seconds=10,
              source='https://doi.org/10.6084/m9.figshare.29064530',grid_zyx_um=[.80,.09,.09]),
 variant=dict(sample='MT_wee-1.1_2',genotype='wee-1.1 mutant',compression='uncompressed',
              source='https://doi.org/10.6084/m9.figshare.27872238',frame_interval_seconds_verified=False),
 stage_alignment_verified=False,common_physical_grid=False,resampling_performed=False)


def save_frames(z,pattern,path):
    members=[]
    for info in z.infolist():
        match=re.search(pattern,info.filename)
        if match and 1<=int(match[1])<=FRAMES:members.append((int(match[1]),int(match[2]),info))
    assert members,'No matching nucleus TIFFs'
    groups={t:sorted([(p,info) for frame,p,info in members if frame==t]) for t in range(1,FRAMES+1)}
    shape=tifffile.imread(io.BytesIO(z.read(groups[1][0][1]))).shape
    n_slices=len(groups[1])
    array=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=(FRAMES,n_slices,*shape))
    for t,items in groups.items():
        assert len(items)==n_slices and [p for p,_ in items]==list(range(1,n_slices+1))
        for zi,(_,info) in enumerate(items):
            image=tifffile.imread(io.BytesIO(z.read(info)))
            assert image.dtype==np.uint8 and image.shape==shape
            array[t-1,zi]=image
        array.flush()
        print(path.name,'frame',t,'of',FRAMES,flush=True)
    del array


url='https://drive.usercontent.google.com/download?id=1I9-Tbwl9e1Hn0nJyTYDmd2GHWdccy0Il&export=download&confirm=t'
r=RangeReader(url,budget=2*1024**3)
outer=zipfile.ZipFile(r)
info=outer.getinfo('Emb8/Emb8_Raw_Nucleus.zip')
archive_path=CACHE/'Emb8_Raw_Nucleus.zip'
if not archive_path.exists():
    r.seek(info.header_offset)
    fields=struct.unpack('<4s5H3L2H',r.read(30))
    r.read(fields[-2]+fields[-1])
    decoder=zlib.decompressobj(-15)
    staging=archive_path.with_suffix('.part')
    remaining=info.compress_size
    crc=0
    with staging.open('wb') as target:
        while remaining:
            chunk=r.read(min(16*1024*1024,remaining))
            data=decoder.decompress(chunk)
            target.write(data);crc=zlib.crc32(data,crc)
            remaining-=len(chunk)
            print('Control nucleus archive',round(100*(1-remaining/info.compress_size),1),'%',flush=True)
        last=decoder.flush();target.write(last);crc=zlib.crc32(last,crc)
    assert crc==info.CRC and staging.stat().st_size==info.file_size
    staging.replace(archive_path)
with zipfile.ZipFile(archive_path) as z:
    print('Control TIFF examples',z.namelist()[:6],flush=True)
    path=OUT/'CONTROL_WT_Emb8_t001_030.npy'
    if not path.exists():save_frames(z,r'-t(\d+)-p(\d+)\.tif$',path)

# Extract a contiguous small range covering only the desired mutant timepoints.
r=RangeReader('https://ndownloader.figshare.com/files/51665312')
z=zipfile.ZipFile(r)
members=[i for i in z.infolist() if (m:=re.search(r'-t(\d+)-p(\d+)\.tif$',i.filename)) and 1<=int(m[1])<=FRAMES]
assert len(members)==94*FRAMES
start=min(i.header_offset for i in members)
end=max(i.header_offset+30+len(i.filename.encode())+len(i.extra)+i.compress_size for i in members)
r.seek(start)
chunks=[]
remaining=end-start
while remaining:
    chunk=r.read(min(16*1024*1024,remaining));chunks.append(chunk);remaining-=len(chunk)
blob=b''.join(chunks)
class SelectedZip:
    def infolist(self):return members
    def read(self,info):
        pos=info.header_offset-start
        fields=struct.unpack('<4s5H3L2H',blob[pos:pos+30]);assert fields[0]==b'PK\x03\x04'
        pos+=30+fields[-2]+fields[-1]
        compressed=blob[pos:pos+info.compress_size]
        raw=zlib.decompress(compressed,-15) if info.compress_type==8 else compressed
        assert len(raw)==info.file_size and zlib.crc32(raw)==info.CRC
        return raw
path=OUT/'VARIANT_wee-1.1_MT2_t001_030.npy'
if not path.exists():save_frames(SelectedZip(),r'-t(\d+)-p(\d+)\.tif$',path)
for group,filename in [('control','CONTROL_WT_Emb8_t001_030.npy'),('variant','VARIANT_wee-1.1_MT2_t001_030.npy')]:
    path=OUT/filename
    array=np.load(path,mmap_mode='r',allow_pickle=False)
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(8*1024*1024),b''):digest.update(chunk)
    metadata[group].update(file=filename,shape=list(array.shape),bytes=path.stat().st_size,sha256=digest.hexdigest())
(OUT/'PROVENANCE.json').write_text(json.dumps(metadata,indent=2))
(OUT/'READ_ME_FIRST.txt').write_text('REAL DATA, 30 native 3D timepoints per recording. Both embryos are uncompressed.\nControl is EmbSAM Emb8 wild type. Variant is CMap MT_wee-1.1_2.\nThese are from different recordings/studies; sampling intervals, voxel grids and biological stages are NOT aligned.\nUse Files on this PC in the app. Do NOT confirm alignment or claim all differences are due to genotype.\nThese are exploratory tests of inference, not a validated genetic comparison.\nSee PROVENANCE.json.\n')
print(json.dumps(metadata,indent=2),flush=True)
