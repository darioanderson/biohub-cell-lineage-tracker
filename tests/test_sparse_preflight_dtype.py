import importlib.util
from pathlib import Path
import numpy as np
import torch
from biohub.learned_detector import build_temporal_detection_model
from biohub.sparse_detector_training import masked_balanced_bce

spec=importlib.util.spec_from_file_location('preflight',Path(__file__).parents[1]/'scripts/biohub_sparse_preflight.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_percentile_normalization_and_real_patch_backward(monkeypatch):
    # Force float64 percentile scalars even on environments that preserve float32.
    monkeypatch.setattr(module.np,'percentile',lambda *a,**k:np.array([1.,100.],dtype=np.float64))
    patch=module.normalise_patch(np.ones((16,64,64),dtype=np.uint16)*50)
    assert patch.dtype==np.float32
    torch.set_num_threads(4)
    model=build_temporal_detection_model(out_channels=8,layers=(8,16,32))
    pair=torch.from_numpy(np.stack([patch,patch]))[None,:,None]
    logits=model(pair);logits.retain_grad()
    target=torch.zeros_like(logits);known=torch.zeros_like(logits)
    target[0,0,8,32,32]=known[0,0,8,32,32]=1
    loss=masked_balanced_bce(logits,target,known);loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[known==0].abs().max()==0
    assert logits.grad[known==1].abs().sum()>0
