"""Package one bounded CHO detector experiment with frozen original control."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
import argparse


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validation-only', action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    folder=root/'notebooks/kaggle_cho_detector';folder.mkdir(exist_ok=True)
    files={f'advanced/scripts/{n}':root/'scripts'/n for n in
           ['train_ctc_detector.py','compare_ctc_detector.py','test_ctc_training.py']}
    files.update({p.relative_to(root.parent).as_posix():p for p in (root.parent/'baseline/src').rglob('*.py')})
    buffer=io.BytesIO()
    manifest={'source_hashes':{n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()},
      'dataset_url':'https://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-CHO.zip',
      'dataset_sha256':'48d7e32b6408dddd04f1b6e4153e91f19181b405f4c75610629a25e05d40fe77',
      'epochs_max':30,'steps_per_epoch':30,'seed':42,'train_sequence':'01',
      'train_frames':list(range(0,60,3)),'validation_frames':[70,77,84,91],
      'heldout_sequence':'02','heldout_frame_step':5,
      'threshold_grid':[.15,.25,.4,.55],
      'evaluation_split':'validation' if args.validation_only else 'held-out',
      'limitations':['CHO centroid detection only, not Biohub tracking or official CTC scores',
                     'Validation selects epoch/threshold on sequence 01 only; evaluate sequence 02 once',
                     'Original controls include shallow-volume adapter; not exhaustively tuned']}
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for name,path in files.items():z.write(path,name)
        z.writestr('frozen_protocol.json',json.dumps(manifest,indent=2))
    code="""import base64,io,zipfile,hashlib,json,subprocess,sys
from pathlib import Path
import requests
root=Path('/kaggle/working/ctc_project');root.mkdir(exist_ok=True)
subprocess.run([sys.executable,'-m','pip','install','--no-deps','--only-binary=:all:','imagecodecs==2024.12.30'],check=True)
import imagecodecs
assert imagecodecs.lzw_decode(imagecodecs.lzw_encode(b'CTC codec preflight'))==b'CTC codec preflight'
payload=PAYLOAD
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:z.extractall(root)
protocol=json.loads((root/'frozen_protocol.json').read_text())
for name,expected in protocol['source_hashes'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
data=root/'data';data.mkdir(exist_ok=True)
archive=data/'Fluo-N3DH-CHO.zip'
response=requests.get(protocol['dataset_url'],stream=True,timeout=(30,120));response.raise_for_status()
size=0;digest=hashlib.sha256()
with archive.open('wb') as f:
    for chunk in response.iter_content(1024*1024):
        size+=len(chunk)
        if size>128*1024*1024:raise RuntimeError('Archive exceeds predeclared 128 MiB limit')
        f.write(chunk);digest.update(chunk)
assert digest.hexdigest()==protocol['dataset_sha256'],'Dataset bytes differ from verified local source'
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        assert (data/name).resolve().is_relative_to(data.resolve()),name
    z.extractall(data)
scripts=root/'advanced/scripts'
import tifffile
for name in ['01/t000.tif','01_GT/TRA/man_track000.tif']:
    sample=tifffile.imread(data/'Fluo-N3DH-CHO'/name)
    assert sample.ndim==3,(name,sample.shape)
print('LZW raw-image and annotation preflight passed',flush=True)
(root/'comparison').mkdir(exist_ok=True)
subprocess.run([sys.executable,str(scripts/'test_ctc_training.py')],check=True)
subprocess.run([sys.executable,str(scripts/'train_ctc_detector.py'),'--epochs','30','--steps','30'],check=True)
subprocess.run([sys.executable,str(scripts/'compare_ctc_detector.py'),'--split',protocol['evaluation_split']],check=True)
print('CHO DETECTOR TRAINING AND COMPARISON COMPLETED',protocol['evaluation_split'],flush=True)
""".replace('PAYLOAD',repr(base64.b64encode(buffer.getvalue()).decode()))
    nb={'cells':[{'cell_type':'markdown','id':'protocol','metadata':{},'source':['# CHO detector feasibility experiment\n',
                 'Human tracking markers. Evaluation split: '+manifest['evaluation_split']+'. No Biohub claims.']},
                 {'cell_type':'code','id':'run','metadata':{},'source':code,'execution_count':None,'outputs':[]}],
        'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'nbformat':4,'nbformat_minor':5}
    (folder/'run.ipynb').write_text(json.dumps(nb))
    (folder/'frozen_protocol.json').write_text(json.dumps(manifest,indent=2))
    (folder/'kernel-metadata.json').write_text(json.dumps({'id':'dushi2/biohub-cho-detector-feasibility',
        'title':'Biohub CHO Detector Feasibility','code_file':'run.ipynb','language':'python',
        'kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,
        'dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]},indent=2))
    print('Packaged',len(files),'verified source files')


if __name__=='__main__':main()
