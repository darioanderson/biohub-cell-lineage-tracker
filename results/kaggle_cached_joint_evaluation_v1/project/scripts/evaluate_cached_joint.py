"""Small exploratory paired evaluation; uses unchanged sparse evaluator."""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from biohub.data import find_geff_path, load_geff_graph
from biohub.joint_division import joint_divisions
from scripts.evaluate_trackastra import evaluate_trackastra


def main():
    out=Path('/kaggle/working/cached_joint_evaluation');out.mkdir(exist_ok=True)
    roots=list(Path('/kaggle/input').rglob('selection.json'))
    choices=[p.parent for p in roots if len(list(p.parent.glob('*.graphs.json.gz')))==4]
    if len(choices)!=1:raise RuntimeError(f'Expected one four-volume cache, found {choices}')
    train=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')
    rows=[]
    for p in sorted(choices[0].glob('*.graphs.json.gz')):
        name=p.name.replace('.graphs.json.gz','');s=json.loads(gzip.open(p,'rt').read())
        nodes,edges=load_geff_graph(find_geff_path(train,name))
        frames=[SimpleNamespace(t=f['t'],ids=f['ids'],pred_to_gt={int(n):g for n,g in f['pred_to_gt'].items()},
                                gt_count=int((nodes.t==f['t']).sum())) for f in s['frames']]
        sample=SimpleNamespace(gt_nodes=nodes,gt_edges=edges,frames=frames)
        times={n:f['t'] for f in s['frames'] for n in f['ids']}
        xyz={n:np.array(x) for f in s['frames'] for n,x in zip(f['ids'],f['xyz'])}
        weights={(a,b):w for a,b,w in s['raw_weights']}
        for arm,kw in [('current_hungarian',None),('joint_trackastra',{'sister_gate':24.}),
                       ('joint_geometry',{'sister_gate':24.,'score_mode':'geometry'})]:
            start=time.perf_counter(); graph=s['baseline'];info={}
            if kw is not None:graph,info=joint_divisions(graph,weights,times,xyz,np.array(s['scale']),**kw)
            metrics=evaluate_trackastra(sample,graph)
            row={'sample':name,'embryo':name.split('_')[0],'arm':arm,'metrics':metrics,'seconds':time.perf_counter()-start,'audit':info}
            rows.append(row)
            (out/'details.json').write_text(json.dumps(rows,indent=2))
            print('SCORED',name,arm,json.dumps(metrics),flush=True)
    (out/'status.json').write_text(json.dumps({'state':'complete','adoption_verdict':None,
         'limitation':'Four previously inspected annotation-enriched volumes; exploratory only. No threshold selection.'}))


if __name__=='__main__':main()
