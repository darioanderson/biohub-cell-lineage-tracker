"""Freeze a bounded private Kaggle sparse-label preflight."""
from pathlib import Path
import base64,io,json,zipfile,hashlib
root=Path(__file__).resolve().parents[1]
folder=root/'notebooks/kaggle_sparse_preflight';folder.mkdir(exist_ok=True)
files=list((root/'src/biohub').glob('*.py'))+[root/'scripts/biohub_sparse_preflight.py']
buf=io.BytesIO();hashes={}
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:
        name=p.relative_to(root).as_posix();z.write(p,name);hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
code='''import base64,io,zipfile,sys,subprocess,hashlib,json
from pathlib import Path
import importlib.util
for module,requirement in [('numcodecs','numcodecs==0.16.3'),('blosc2','blosc2==2.7.1')]:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable,'-m','pip','install','--only-binary=:all:',requirement],check=True)
import numcodecs,blosc2
print('Codec versions',numcodecs.__version__,blosc2.__version__,flush=True)
root=Path('/kaggle/working/source');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:z.extractall(root)
hashes=HASHES
for n,h in hashes.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
(root/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
sys.path.insert(0,str(root/'src'))
import runpy
runpy.run_path(str(root/'scripts/biohub_sparse_preflight.py'),run_name='__main__')
'''.replace('PAYLOAD',repr(base64.b64encode(buf.getvalue()).decode())).replace('HASHES',repr(hashes))
compile(code,'notebook','exec')
(folder/'run.ipynb').write_text(json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'cells':[{'cell_type':'code','id':'preflight','metadata':{},'source':code,'outputs':[],'execution_count':None}]}))
(folder/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
(folder/'kernel-metadata.json').write_text(json.dumps({'id':'dushi2/biohub-sparse-detector-preflight','title':'Biohub Sparse Detector Preflight','code_file':'run.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'competition_sources':['biohub-cell-tracking-during-development'],'dataset_sources':[],'kernel_sources':[],'model_sources':[]},indent=2))
print('Packaged',len(files),'source files')
