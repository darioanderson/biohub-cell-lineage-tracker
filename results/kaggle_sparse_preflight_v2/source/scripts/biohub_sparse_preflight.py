"""Real Biohub I/O and sparse-supervision gradient checks, not detector training."""
import json
from pathlib import Path
import numpy as np
import torch
from biohub.data import list_datasets,find_geff_path,load_geff_graph,read_zarr_meta,load_volume
from biohub.learned_detector import build_temporal_detection_model
from biohub.sparse_detector_training import build_sparse_detection_target,masked_balanced_bce


def main():
    out=Path('/kaggle/working/sparse_preflight');out.mkdir(exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for GPU compatibility check')
    candidates=sorted(Path('/kaggle/input').rglob('train'))
    train=next((p for p in candidates if list_datasets(p)),None)
    if train is None:raise FileNotFoundError('Mounted Biohub train directory not found')
    names=[n for n in list_datasets(train) if find_geff_path(train,n)]
    groups=sorted(set(n.split('_')[0] for n in names))
    if len(groups)<2:raise ValueError('Need two distinct embryo groups')
    selected=[next(n for n in names if n.startswith(g+'_')) for g in groups[:2]]
    results=[]
    torch.manual_seed(42)
    for name in selected:
        nodes,_=load_geff_graph(find_geff_path(train,name))
        if nodes.empty:raise ValueError('No annotations: '+name)
        t=int(nodes.t.min());points=nodes[nodes.t==t][['z','y','x']].to_numpy(float)
        shape,dtype=read_zarr_meta(train/f'{name}.zarr')
        raw=load_volume(train/f'{name}.zarr',t,shape,dtype)
        center=np.rint(points[0]).astype(int);size=np.minimum([16,64,64],raw.shape)
        start=np.clip(center-size//2,0,np.array(raw.shape)-size)
        sl=tuple(slice(int(a),int(a+b)) for a,b in zip(start,size))
        patch=raw[sl].astype(np.float32);low,high=np.percentile(patch,[1,99])
        patch=np.clip((patch-low)/max(high-low,1),0,1)
        local=points-start;local=local[np.all((local>=0)&(local<size),axis=1)]
        target,known=build_sparse_detection_target(patch,local,np.empty((0,3)),downsample=(1,1,1))
        positives=int((target*known).sum());negatives=int(((1-target)*known).sum())
        assert positives>0 and negatives==0
        # Explicitly a positive-only diagnostic gradient, never an optimizer step.
        net=build_temporal_detection_model(out_channels=8,layers=(8,16,32)).cuda()
        pair=torch.from_numpy(np.stack([patch,patch]))[None,:,None].cuda()
        logits=net(pair);logits.retain_grad()
        y=torch.from_numpy(target)[None,None].cuda();mask=torch.from_numpy(known)[None,None].cuda()
        loss=masked_balanced_bce(logits,y,mask);loss.backward()
        unknown_grad=float(logits.grad[mask==0].abs().max().item())
        assert unknown_grad==0 and torch.isfinite(loss)
        results.append({'sample':name,'embryo':name.split('_')[0],'frame':t,'raw_shape':list(raw.shape),
            'patch_shape':list(patch.shape),'annotated_positives':positives,'verified_negatives':negatives,
            'unknown_output_gradient_max':unknown_grad,'diagnostic_loss':float(loss.detach()),
            'optimizer_steps':0,'temporal_input':'same frame duplicated for shape/gradient test only'})
        print(json.dumps(results[-1]),flush=True)
        del net,pair,logits,loss,raw;torch.cuda.empty_cache()
    report={'state':'compatibility_passed_training_blocked','paired_volume_inventory':len(names),
        'selected_before_metrics':selected,'device':torch.cuda.get_device_name(),'checks':results,
        'reason':'No verified background supervision supplied. Positive-only BCE permits an all-positive solution.',
        'next_requirement':'Provide verified negative regions or implement and validate a different sparse-label learning objective.',
        'trained_checkpoint':None,'full_training_started':False}
    (out/'status.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
