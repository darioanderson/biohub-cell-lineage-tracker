"""Frozen candidate, additional volumes, annotation-stratified exploratory test."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from biohub.config import Config
from biohub.data import read_physical_metadata
from biohub.joint_division import joint_divisions
from scripts.evaluate_detected_linker import build_detected_sample
from scripts.evaluate_trackastra import run_classical_edges, run_trackastra, evaluate_trackastra
from scripts.division_candidate_audit import PredictionCapture


def audit_edits(edits, mapping, gt_edges):
    def label(edge):
        if edge is None:return 'not_applicable'
        a,b=edge
        if a not in mapping or b not in mapping:return 'unknown'
        return 'supported' if (mapping[a],mapping[b]) in gt_edges else 'inconsistent'
    rows=[{**e,'added_support':label(e['added']),'removed_support':label(e['removed'])} for e in edits]
    return {'counts':dict(Counter(f"added_{e['added_support']}__removed_{e['removed_support']}" for e in rows)),
            'known_edges_removed':sum(e['removed_support']=='supported' for e in rows),
            'known_edges_added':sum(e['added_support']=='supported' for e in rows),
            'edits':rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('train_dir',type=Path);parser.add_argument('model_dir',type=Path)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text())
    root=Path(__file__).resolve().parents[1]
    for name,expected in manifest['source_hashes'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Source changed: {name}')
    (out/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2))
    import torch
    from trackastra.model import Trackastra
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no silent CPU fallback')
    model=PredictionCapture(Trackastra.from_folder(args.model_dir,device='cuda'))
    cfg=Config().competition_v10_dog_preset().copy_with(use_competition_preset=False,
        detection_cache_dir=Path('/kaggle/temp/joint_expansion_cache'))
    rows=[]
    for item in manifest['samples']:
        name=item['name'];start=time.perf_counter()
        (out/'status.json').write_text(json.dumps({'state':'running','sample':name,'completed_arms':len(rows)}))
        voxel,interval=read_physical_metadata(args.train_dir/f'{name}.zarr')
        local=cfg.copy_with(voxel_size_zyx_um=voxel or cfg.voxel_size_zyx_um,frame_interval_s=interval or cfg.frame_interval_s)
        sample=build_detected_sample(args.train_dir,name,local)
        baseline=run_classical_edges(sample,local)
        preprocessing_seconds=time.perf_counter()-start
        start=time.perf_counter();run_trackastra(model,args.train_dir,sample,'greedy')
        from biohub.trackastra_adapter import graph_edge_scores_to_detection_ids
        weights=graph_edge_scores_to_detection_ids(model.raw,[f.ids for f in sample.frames])
        model_seconds=time.perf_counter()-start
        times={n:f.t for f in sample.frames for n in f.ids}
        xyz={n:f.xyz[i] for f in sample.frames for i,n in enumerate(f.ids)}
        mapping={n:g for f in sample.frames for n,g in f.pred_to_gt.items()}
        gt_edges=set(map(tuple,sample.gt_edges[['source_id','target_id']].to_numpy()))
        for arm in ['current_hungarian','joint_trackastra','joint_geometry']:
            start=time.perf_counter();graph=baseline;info={'edits':[]}
            if arm!='current_hungarian':
                graph,info=joint_divisions(baseline,weights,times,xyz,local.scale_array,
                    **manifest['parameters'],score_mode='geometry' if arm=='joint_geometry' else 'trackastra')
            metrics=evaluate_trackastra(sample,graph)
            audit=audit_edits(info['edits'],mapping,gt_edges)
            rows.append({'sample':name,'embryo':sample.embryo,'stratum':item['stratum'],'arm':arm,
                         'metrics':metrics,'audit':audit,'seconds':time.perf_counter()-start,
                         'preprocessing_seconds':preprocessing_seconds,'model_seconds':model_seconds})
            (out/'details.json').write_text(json.dumps(rows,indent=2))
            print('SCORED',name,arm,json.dumps(metrics),flush=True)
        (out/f'{name}.edges.json').write_text(json.dumps({'baseline':baseline,'mapping':mapping}))
    (out/'status.json').write_text(json.dumps({'state':'complete','completed_arms':len(rows),
         'adoption_verdict':None,'limitation':'Additional volumes in previously used embryos; not independent biological validation'}))


if __name__=='__main__':main()
