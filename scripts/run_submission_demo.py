"""Real small-clip inference with an explicitly experimental pretrained detector."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import argparse
import numpy as np
from biohub.config import Config
from biohub.analysis import run_tracking_pipeline
from app.paired_movies import recording_config
from app.result_review import frame_measurements
from app.pair_dashboard import reference_counts_for_pair

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--large',action='store_true');args=parser.parse_args()
data=root/('data/real_control_variant_large' if args.large else 'data/real_control_variant_under200mb')
out=root/('results/submission_large_demo_20261119' if args.large else 'results/submission_demo_20261119')
out.mkdir(exist_ok=True)
manifest=json.loads((data/'PROVENANCE.json').read_text())
checkpoint=root/'models/cellpose_nuclei/nucleitorch_0'
cfg=Config(detector_backend='cellpose_projection',learned_detector_path=checkpoint)
pair=dict(results=[],volumes=[],hashes=[],filenames=[],configs=[])
summary=dict(detector='Pretrained Cellpose nuclei, 2D maximum projection with estimated depth',
             checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             source='https://cellpose.readthedocs.io/en/v3.1.1.1/models.html',
             genetic_diagnosis_validated=False,stage_alignment_verified=False,training_performed=False,recordings={})
try:
    for group in ['control','variant']:
        entry=manifest.get('recordings',manifest)[group]
        volume=np.load(data/Path(entry['file']).name,mmap_mode='r',allow_pickle=False)
        config=recording_config(cfg,entry['sha256'])
        (out/'status.json').write_text(json.dumps(dict(state='running',group=group)))
        result=run_tracking_pipeline(list(volume),config,dataset_name=group.title())
        result.stats.update(source='Pretrained Cellpose nuclei projection; classical tracking',
            limitations='Projection overlaps and mitosis cause misses. Not a 3D segmentation or genetic classifier.',
            spacing_note='Source spacing used for linking; stage and acquisition times remain unaligned.')
        result.stats['time_interval_verified']=config.frame_interval_s==10.
        result.nodes.to_csv(out/(group+'_nodes.csv'),index=False)
        result.edges.to_csv(out/(group+'_edges.csv'),index=False)
        frame_measurements(result).to_csv(out/(group+'_frame_metrics.csv'),index=False)
        (out/(group+'_stats.json')).write_text(json.dumps(result.stats,indent=2,default=str))
        pair['results'].append(result);pair['volumes'].append(volume);pair['hashes'].append(entry['sha256'])
        pair['filenames'].append(Path(entry['file']).name);pair['configs'].append(config)
        summary['recordings'][group]=dict(input_sha256=entry['sha256'],file=Path(entry['file']).name,shape=list(volume.shape),
            detections_per_frame=[len(result.nodes[result.nodes.t.eq(t)]) for t in range(len(volume))],
            observations=result.n_nodes,links=result.n_edges,division_proposals=result.n_divisions)
        print(group,summary['recordings'][group],flush=True)
    reference_counts_for_pair(pair).to_csv(out/'reference_counts.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'status.json').write_text(json.dumps(dict(state='complete_with_known_errors',genetic_validation=False)))
except Exception as error:
    (out/'status.json').write_text(json.dumps(dict(state='failed',error=str(error))))
    raise
