"""Exercise real saved prediction tables through every Streamlit review page.

This checks presentation and session binding, not a new model-accuracy result.
"""
import json
from pathlib import Path
import sys
import argparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from biohub.analysis import AnalysisResult, FrameResult
from biohub.config import Config
from app.recording_metadata import known_recording, calibrated_config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--biohub',action='store_true');args=parser.parse_args()
    source=ROOT/('results/submission_biohub_20261119' if args.biohub else 'results/submission_demo_20261119')
    data=ROOT/('data/local_biohub_pair' if args.biohub else 'data/real_control_variant_under200mb')
    manifest=json.loads((data/'PROVENANCE.json').read_text())['recordings']
    cfg=Config(detector_backend='peaks' if args.biohub else 'cellpose_projection',learned_detector_path=ROOT/('results/kaggle_ce_cshaper_v2/detector.pt' if args.biohub else 'models/cellpose_nuclei/nucleitorch_0'))
    labels=['Recording A','Recording B'] if args.biohub else ['Control','Variant']
    pair=dict(cfg=cfg,volumes=[],results=[],hashes=[],filenames=[],configs=[],metadata=[],reviews={},labels=labels)
    for group in ('control','variant'):
        entry=manifest[group];info=known_recording(entry['sha256'])
        movie=np.load(data/entry['file'],mmap_mode='r',allow_pickle=False)
        nodes=pd.read_csv(source/(group+'_nodes.csv'));edges=pd.read_csv(source/(group+'_edges.csv'))
        frames=[]
        for t in range(len(movie)):
            n=nodes[nodes.t.eq(t)]
            frames.append(FrameResult(t,n[['z','y','x']].to_numpy(),n.raw_detection_score.to_numpy(),n.intensity.to_numpy(),n.node_id.astype(int).tolist()))
        stats=json.loads((source/(group+'_stats.json')).read_text())
        stats.update(time_interval_verified=info['time_verified'],spatial_calibration_verified=info['spatial_verified'],
                     run_id='ui_verification_'+group,input_sha256=entry['sha256'])
        result=AnalysisResult(labels[len(pair['results'])],movie.shape,nodes,edges,frames,stats)
        pair['volumes'].append(movie);pair['results'].append(result);pair['hashes'].append(entry['sha256'])
        pair['filenames'].append(entry['file']);pair['configs'].append(calibrated_config(cfg,info));pair['metadata'].append(info)
    test=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
    test.session_state['analysis_workspace']='Control and variant'
    test.session_state['paired_input_mode']='Files on this PC'
    test.session_state['pair_model_engine']='Biohub original pipeline — classical detection' if args.biohub else 'Cellpose nuclei — experimental projection review'
    test.session_state['local_control_path']=str(data/manifest['control']['file'])
    test.session_state['local_variant_path']=str(data/manifest['variant']['file'])
    test.session_state['real_pair_output']=pair
    if args.biohub:
        from app.saved_runs import restore_biohub_run
        restore_biohub_run(pair,source)
    checks=[]
    for group in labels:
        test.session_state['pair_view']=group
        for page in ('Home','Dataset','Volume','Pipeline','Detection','Lineage','Exports','Compare Lineages'):
            print('Checking',group,page,flush=True)
            test.session_state['main_nav']=page
            test.run(timeout=60)
            errors=[e.message for e in test.exception]
            checks.append(dict(recording=group,page=page,errors=errors))
            if errors:raise RuntimeError(f'{group}/{page}: {errors}')
    out=ROOT/'results/submission_verification_20261119';out.mkdir(exist_ok=True)
    (out/('biohub_ui_checks.json' if args.biohub else 'ui_checks.json')).write_text(json.dumps(dict(source='Actual saved real-microscopy predictions; no new inference',checks=checks),indent=2))
    print('PASS: all 16 real-output page/recording combinations',flush=True)


if __name__=='__main__':main()
