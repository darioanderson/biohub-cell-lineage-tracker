"""Exploratory fixed ablation on saved graphs; no adoption or precision claim."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from biohub.joint_division import joint_divisions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    arms={'trackastra_original_gate':{},
          'trackastra_triangle_bound':{'sister_gate':24.},
          'geometry_triangle_bound':{'sister_gate':24.,'score_mode':'geometry'}}
    report={'purpose':'Exploratory engineering ablation on previously inspected samples, not held-out validation',
            'rationale':'24 um is twice the existing 12 um parent gate (triangle bound); no grid search',
            'limitations':['Full GT edge tables are absent: edge accuracy and division precision unavailable',
                          'Only six annotated divisions; samples are annotation-enriched'],
            'arms':arms,'samples':[],'input_hashes':{}}
    for p in sorted(args.input.glob('*.graphs.json.gz')):
        name=p.name.replace('.graphs.json.gz','')
        report['input_hashes'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        s=json.loads(gzip.open(p,'rt').read())
        gt=json.loads((args.input/f'{name}.json').read_text())
        times={n:f['t'] for f in s['frames'] for n in f['ids']}
        xyz={n:np.array(x) for f in s['frames'] for n,x in zip(f['ids'],f['xyz'])}
        mapping={int(g):int(n) for f in s['frames'] for n,g in f['pred_to_gt'].items()}
        weights={(a,b):w for a,b,w in s['raw_weights']}
        for arm,kw in arms.items():
            start=time.perf_counter()
            edges,info=joint_divisions(s['baseline'],weights,times,xyz,np.array(s['scale']),**kw)
            pairs=set(edges);baseline=set(map(tuple,s['baseline']));events=[]
            for e in gt['ceiling']['events']:
                needed={(mapping.get(e['parent_gt']),mapping.get(d)) for d in e['daughter_gt']}
                events.append({'parent_gt':e['parent_gt'],'baseline_complete':needed<=baseline,'candidate_complete':needed<=pairs})
            row={'sample':name,'arm':arm,'seconds':time.perf_counter()-start,
                 'annotated_divisions':events,'graph_audit':info}
            report['samples'].append(row)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2))
            print(name,arm,'edits',info['accepted'],'GT splits',sum(e['candidate_complete'] for e in events),flush=True)


if __name__=='__main__':main()
