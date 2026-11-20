"""Compare optimized solver with preserved pre-change implementation."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.sparse import issparse
from biohub.config import Config
import biohub.tracking as current


def load_reference():
    path=Path(__file__).resolve().parents[1]/'references/joint_solver_before_optimization/tracking.py'
    spec=importlib.util.spec_from_file_location('joint_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('seed',range(8))
@pytest.mark.parametrize('divisions',[False,True])
def test_exact_assignment_and_constraint_parity(seed,divisions,monkeypatch):
    old=load_reference();rng=np.random.default_rng(seed)
    prev=rng.uniform(0,6,(9,3));curr=prev+rng.normal(0,.2,(9,3))
    curr=np.vstack((curr,prev[:2]+.5))
    cfg=Config(voxel_size_zyx_um=(1,1,1),linker_backend='classical_joint',
        max_link_dist_um=3,div_parent_dist_um=2,div_sister_dist_um=2,
        div_midpoint_dist_um=2,detect_divisions=divisions)
    matrices=[];original_milp=current.milp
    def spy(**kwargs):
        a=kwargs['constraints'].A
        matrices.append((a.toarray() if issparse(a) else a.copy(),issparse(a)))
        return original_milp(**kwargs)
    monkeypatch.setattr(old,'milp',spy);monkeypatch.setattr(current,'milp',spy)
    velocity=rng.normal(0,.1,prev.shape)
    expected=old._joint_hypothesis_assignments(prev,curr,cfg,None,None,velocity,None,None)
    split=len(matrices)
    actual=current._joint_hypothesis_assignments(prev,curr,cfg,None,None,velocity,None,None)
    assert actual==expected
    assert len(matrices)==2*split
    for (dense,_),(sparse,is_sparse) in zip(matrices[:split],matrices[split:]):
        assert is_sparse
        np.testing.assert_array_equal(dense,sparse)
