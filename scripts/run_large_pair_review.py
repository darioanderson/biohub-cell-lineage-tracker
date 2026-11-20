"""Run the fixed local model on both real 30-frame clips and preserve evidence."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from biohub.analysis import run_tracking_pipeline
from biohub.config import Config
from biohub.cshaper_detector import CShaperDetector
from app.result_review import frame_measurements

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/real_control_variant_large'
OUT=ROOT/'results/large_pair_review_20261119'
OUT.mkdir(exist_ok=True)
checkpoint=ROOT/'results/kaggle_ce_cshaper_v2/detector.pt'
cfg=Config(detector_backend='cshaper',learned_detector_path=checkpoint,voxel_size_zyx_um=(1.,1.,1.))
original=CShaperDetector.detect
progress={'frame':0,'group':''}
def logged_detect(self,*args,**kwargs):
    value=original(self,*args,**kwargs)
    print(progress['group'],'frame',progress['frame'],'raw detections',len(value[0]),flush=True)
    progress['frame']+=1
    return value
CShaperDetector.detect=logged_detect
provenance=json.loads((DATA/'PROVENANCE.json').read_text())
summary=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             detector='Kaggle-trained CShaper centre detector',linker='classical',
             physical_calibration_used=False,stage_alignment_verified=False,
             note='Exploratory inference on independent studies; not a validated genetic comparison',recordings={})
for group in ('control','variant'):
    path=DATA/provenance[group]['file']
    progress.update(frame=0,group=group)
    (OUT/'status.json').write_text(json.dumps({'state':'running','group':group}))
    movie=np.load(path,mmap_mode='r',allow_pickle=False)
    try:
        result=run_tracking_pipeline(list(movie),cfg,dataset_name=group.title())
    except Exception as error:
        (OUT/'status.json').write_text(json.dumps({'state':'failed','group':group,'error':str(error)}))
        raise
    result.nodes.to_csv(OUT/(group+'_nodes.csv'),index=False)
    result.edges.to_csv(OUT/(group+'_edges.csv'),index=False)
    frame_measurements(result).to_csv(OUT/(group+'_frame_metrics.csv'),index=False)
    summary['recordings'][group]=dict(file=str(path),sha256=provenance[group]['sha256'],shape=list(movie.shape),
        observations=result.n_nodes,links=result.n_edges,division_proposals=result.n_divisions)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print('FINISHED',group,summary['recordings'][group],flush=True)
(OUT/'status.json').write_text(json.dumps({'state':'complete','both_recordings':True}))
