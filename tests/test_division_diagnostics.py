from types import SimpleNamespace
import numpy as np
import pandas as pd
from biohub.config import Config
from scripts.diagnose_divisions import division_ceiling, inspect_conflicts


def sample(merged=False):
    nodes = pd.DataFrame([[1,0,0,0,0],[2,1,0,0,-1],[3,1,0,0,1]], columns=['node_id','t','z','y','x'])
    frames = [SimpleNamespace(t=0,ids=[10],xyz=np.array([[0.,0,0]]),pred_to_gt={10:1}),
              SimpleNamespace(t=1,ids=[20] if merged else [20,30],
                              xyz=np.array([[0.,0,0]]) if merged else np.array([[0.,0,-1],[0,0,1]]),
                              pred_to_gt={20:2} if merged else {20:2,30:3})]
    return SimpleNamespace(gt_nodes=nodes,gt_edges=pd.DataFrame([[1,2],[1,3]],columns=['source_id','target_id']),frames=frames)


def test_merged_detection_cannot_cover_two_daughters():
    report = division_ceiling(sample(True),np.ones(3),gate_um=2)
    assert report['gt_divisions'] == 1
    assert report['raw_two_distinct_daughters'] == 0
    assert report['matched_triplet_fraction'] == 0


def test_distinct_daughters_and_missing_parent():
    data = sample()
    assert division_ceiling(data,np.ones(3),gate_um=2)['matched_triplet_fraction'] == 1
    data.frames[0].pred_to_gt = {}
    report = division_ceiling(data,np.ones(3),gate_um=2)
    assert report['matched_daughters'] == 1
    assert report['matched_triplet'] == 0


def test_conflicts_are_not_all_owned_daughters():
    data = sample()
    cfg = Config(voxel_size_zyx_um=(1,1,1))
    proposals = [(10,20),(10,30)]
    missing_anchor = inspect_conflicts(data,[],proposals,{},cfg)
    owned = inspect_conflicts(data,[(10,20),(40,30)],proposals,{},cfg)
    assert missing_anchor['counts'] == {'parent_baseline_child_count_not_one':1}
    assert owned['counts'] == {'second_daughter_already_owned':1}
    assert owned['examples'][0]['sparse_label'] == 'gt_supported_pair'


def test_no_gt_divisions_has_no_invented_ceiling():
    data = sample()
    data.gt_edges = data.gt_edges.iloc[:1]
    assert division_ceiling(data,np.ones(3))['matched_triplet_fraction'] is None
