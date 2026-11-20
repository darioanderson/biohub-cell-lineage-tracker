"""Fetch small real wild-type/mutant clips; explicitly NOT a matched experiment.

Only selected TIFF archive ranges are fetched, never the full mutant archive.
"""
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import zipfile
import zlib

import nibabel as nib
import numpy as np
import requests
import tifffile
from bs4 import BeautifulSoup

OUT = Path(__file__).resolve().parents[1]/'data/real_control_variant'
OUT.mkdir(parents=True,exist_ok=True)
SESSION = requests.Session()
SOURCE = 'https://ndownloader.figshare.com/files/51665312'
SIZE = 853385415
TRANSFERRED = 0


def bounded_get(url, limit=32*1024*1024, **kwargs):
    with SESSION.get(url, stream=True,timeout=(20,90),**kwargs) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_content(1024*1024):
            data.extend(chunk)
            if len(data)>limit:
                raise ValueError('Download exceeds the small-file budget.')
        return bytes(data)


def archive_range(start, length):
    global TRANSFERRED
    assert 0<=length<=32*1024*1024 and TRANSFERRED+length<=128*1024*1024
    if not length:
        return b''
    with SESSION.get(SOURCE,headers={'Range':f'bytes={start}-{start+length-1}'},stream=True,timeout=(20,90)) as response:
        if response.status_code != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
            raise ValueError('Server did not honor byte ranges; refusing full archive download.')
        raw = response.raw.read(length+1)
        assert len(raw)==length
    TRANSFERRED+=length
    return raw


class RemoteZip(io.RawIOBase):
    def __init__(self): self.pos=0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else SIZE+offset
        return self.pos
    def read(self,length=-1):
        length=SIZE-self.pos if length<0 else min(length,SIZE-self.pos)
        raw=archive_range(self.pos,length)
        self.pos+=len(raw)
        return raw


def listing(folder):
    html=bounded_get('https://drive.google.com/embeddedfolderview',params={'id':folder}).decode()
    found={}
    for anchor in BeautifulSoup(html,'html.parser').select('a'):
        title=anchor.select_one('.flip-entry-title')
        match=re.search(r'/(?:folders|d)/([\w-]+)',anchor.get('href',''))
        if title and match: found[title.get_text(strip=True)]=match[1]
    return found


provenance={'purpose':'Real-data input/model testing only; NOT a matched control/variant experiment',
    'stage_alignment_verified':False,'physical_grid_alignment_verified':False,
    'control_conditions':'CShaper Sample18, mechanically compressed wild-type embryo',
    'variant_conditions':'CMap MT_wee-1.1_2, uncompressed wee-1.1 mutant embryo',
    'papers':['https://doi.org/10.1038/s41467-020-19863-x','https://doi.org/10.1038/s41467-025-58878-0'],
    'source_files':[]}

# The archive's tif/ channel is the nucleus TIFF series (tifR/ is the membrane
# convention in CMap). Keep all slices, native intensities and native grid.
archive=zipfile.ZipFile(RemoteZip())
members=[item for item in archive.infolist() if re.fullmatch(r'tif/241106wee1plc1p4_L1-t00[123]-p\d+\.tif',item.filename)]
assert len(members)==3*94, len(members)
start=min(item.header_offset for item in members)
end=max(item.header_offset+30+len(item.filename.encode())+len(item.extra)+item.compress_size for item in members)
blob=archive_range(start,end-start)
volumes=[]
for frame in (1,2,3):
    slices=[]
    selected=sorted([item for item in members if f'-t{frame:03d}-' in item.filename],key=lambda item:int(re.search(r'-p(\d+)',item.filename)[1]))
    for item in selected:
        offset=item.header_offset-start
        fields=struct.unpack('<4s5H3L2H',blob[offset:offset+30])
        assert fields[0]==b'PK\x03\x04'
        payload=offset+30+fields[-2]+fields[-1]
        compressed=blob[payload:payload+item.compress_size]
        raw=zlib.decompress(compressed,-15) if item.compress_type==zipfile.ZIP_DEFLATED else compressed
        assert len(raw)==item.file_size and zlib.crc32(raw)==item.CRC
        image=tifffile.imread(io.BytesIO(raw))
        assert image.shape==(512,712)
        slices.append(image)
        provenance['source_files'].append(dict(group='variant',archive_url=SOURCE,member=item.filename,sha256=hashlib.sha256(raw).hexdigest()))
    volumes.append(np.stack(slices))
variant_path=OUT/'VARIANT_wee-1.1_MT2_t001_003.npy'
np.save(variant_path,np.stack(volumes),allow_pickle=False)
print('Variant saved',variant_path,variant_path.stat().st_size,flush=True)

root=listing('151pBI7d_JfkOz4eQI4R4F2u_cKtlftBi')
children=listing(root['Sample18'])
images=listing(children['RawNuc'])
volumes=[]
for frame in (1,2,3):
    name=f'Sample18_{frame:03d}_rawNuc.nii.gz'
    key=images[name]
    path=OUT/name
    if not path.exists():
        raw=bounded_get('https://drive.usercontent.google.com/download',params={'id':key,'export':'download','confirm':'t'})
        assert raw.startswith(bytes((31,139)))
        path.write_bytes(raw)
    nii=nib.load(path)
    volumes.append(np.asarray(nii.dataobj).transpose(2,1,0))
    provenance['source_files'].append(dict(group='control',file=name,drive_id=key,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),header_spacing=list(map(float,nii.header.get_zooms()[::-1]))))
control_path=OUT/'CONTROL_WT_Sample18_t001_003.npy'
np.save(control_path,np.stack(volumes),allow_pickle=False)
for group,path in [('control',control_path),('variant',variant_path)]:
    array=np.load(path,mmap_mode='r',allow_pickle=False)
    provenance[group+'_output']=dict(file=path.name,shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
provenance['mutant_archive_bytes_transferred']=TRANSFERRED
(OUT/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
(OUT/'READ_ME_FIRST.txt').write_text('REAL MICROSCOPY, NOT DEMO DATA.\nControl: mechanically compressed wild-type Sample18.\nVariant: uncompressed wee-1.1 mutant MT2.\nBoth files contain the first three recorded 3D frames, NOT verified same biological stages.\nNative voxel grids differ. No registration or resampling was performed.\nUse to test loading and inference only. DO NOT confirm stage/spacing alignment or interpret count differences as genetic effects.\nThe study-matched uncompressed wild-type download link was inaccessible when checked.\nSee PROVENANCE.json for exact sources, hashes and limitations.\n')
print(json.dumps({group:provenance[group+'_output'] for group in ['control','variant']},indent=2),flush=True)
