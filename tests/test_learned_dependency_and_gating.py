import builtins
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from biohub.learned_detector import TemporalUNetDetector
from biohub import learned_linker


def test_missing_checkpoint_precedes_torch_import(tmp_path, monkeypatch):
    original = builtins.__import__
    def without_torch(name, *args, **kwargs):
        if name == 'torch':
            raise ImportError('unavailable')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_torch)
    with pytest.raises(FileNotFoundError, match='checkpoint'):
        TemporalUNetDetector(SimpleNamespace(learned_detector_path=tmp_path/'missing.pt'))


@pytest.mark.parametrize('width', [10,25])
def test_only_gated_features_scored_with_original_ranks(tmp_path, monkeypatch, width):
    p=tmp_path/'model.pt';p.touch()
    batches=[]
    class Model:
        def __call__(self,x):
            batches.append(x.numpy().copy())
            return x.sum(1,keepdim=True)*.001
    monkeypatch.setattr(learned_linker,'_load_model',lambda p:(Model(),np.zeros(width,np.float32),np.ones(width,np.float32)))
    prev=np.array([[0.,0.,0.],[0.,2.,0.],[0.,100.,0.]])
    curr=np.array([[0.,1.,0.],[0.,200.,0.]])
    src=np.arange(15,dtype=np.float32).reshape(3,5);dst=np.ones((2,5),np.float32)
    cost,dist=learned_linker.build_learned_link_cost(prev,curr,p,2,src,dst)
    assert len(batches)==1 and batches[0].shape==(2,width)
    np.testing.assert_array_equal(batches[0][:,8],[0,1])
    np.testing.assert_array_equal(batches[0][:,9],[3,3])
    d=(curr[0]-prev[:2]).astype(np.float32);r=np.linalg.norm(d,axis=1)
    expected=np.column_stack([d,abs(d),r,r*r,[0,1],[3,3]]).astype(np.float32)
    if width==25:expected=np.column_stack([expected,src[:2],dst[[0,0]],abs(src[:2]-dst[0])])
    np.testing.assert_array_equal(batches[0],expected)
    np.testing.assert_allclose(cost[:2,0],np.logaddexp(0,-expected.sum(1)*.001),rtol=1e-6)
    assert np.all(cost[dist>2]==1e6)
