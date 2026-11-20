"""Full local zebrafish inference with honest sparse-reference checks.

No training, cloud launch, genotype assignment, or annotation-derived predictions.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from biohub.analysis import run_tracking_pipeline, AnalysisResult, FrameResult
import pandas as pd
from biohub.config import Config
from app.recording_metadata import known_recording, calibrated_config
from app.biohub_pair_reference import evaluate_reference


def main():
    data=ROOT/'data/local_biohub_pair'
    manifest=json.loads((data/'PROVENANCE.json').read_text())
    out=ROOT/'results/submission_biohub_20261119';out.mkdir(exist_ok=True)
    records={}
    for slot,label in [('control','Recording A'),('variant','Recording B')]:
        entry=manifest['recordings'][slot]
        cfg=calibrated_config(Config(detector_backend='peaks'),known_recording(entry['sha256']))
        movie=np.load(data/entry['file'],mmap_mode='r',allow_pickle=False)
        (out/'status.json').write_text(json.dumps(dict(state='running',recording=label,frames=len(movie))))
        def progress(fraction,text):
            if 'Complete' in text or any(f' {n}/' in text for n in [1,10,20,30,40,50,60,70,80,90,100]):
                print(label,text,flush=True)
        stats_file=out/(slot+'_stats.json')
        if stats_file.exists() and json.loads(stats_file.read_text()).get('input_sha256')==entry['sha256']:
            stats=json.loads(stats_file.read_text())
            nodes=pd.read_csv(out/(slot+'_nodes.csv'));edges=pd.read_csv(out/(slot+'_edges.csv'))
            result=AnalysisResult(label,tuple(movie.shape),nodes,edges,[None]*len(movie),stats)
            print(label,'reusing completed input-bound inference; retrying reference checks only',flush=True)
        else:
            result=run_tracking_pipeline(movie,cfg,dataset_name=label,progress_callback=progress)
        result.stats.update(time_interval_verified=True,spatial_calibration_verified=True,input_sha256=entry['sha256'],
            source='Original classical Biohub detector and linker; not a trained neural model',
            input_paths=[str(data/entry['file'])],genotype_verified=False)
        result.nodes.to_csv(out/(slot+'_nodes.csv'),index=False)
        result.edges.to_csv(out/(slot+'_edges.csv'),index=False)
        (out/(slot+'_stats.json')).write_text(json.dumps(result.stats,indent=2,default=str))
        metrics=evaluate_reference(result,entry['reference_geff'],cfg.scale_array)
        records[label]=dict(frames=len(movie),nodes=result.n_nodes,edges=result.n_edges,
            division_proposals=result.n_divisions,runtime_s=result.stats['runtime_s'],sparse_reference=metrics,
            input_sha256=entry['sha256'],run_id=result.stats['run_id'])
        (out/'summary.json').write_text(json.dumps(dict(recordings=records,genetic_comparison_validated=False,
            training_performed=False,same_prefix=True),indent=2,default=str))
        print(label,json.dumps(records[label]),flush=True)
    (out/'status.json').write_text(json.dumps(dict(state='complete',limitations='Sparse-reference conditional metrics; genotype unknown; no genetic inference')))


if __name__=='__main__':main()
