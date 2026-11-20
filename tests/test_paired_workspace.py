import io
import numpy as np
import pandas as pd
import pytest
from biohub.analysis import AnalysisResult
from biohub.config import Config
from app.paired_movies import activate_recording, restore_single_recording, read_movie, compare_counts


def test_recording_switch_preserves_separate_results_and_restores_single():
    original = object()
    cfg = Config()
    state = dict(result=original, volume_4d='original',cfg=cfg,dataset_name='original',overlay_gif=b'old',review_frame=90)
    first,second = object(),object()
    pair = dict(volumes=['control movie','variant movie'],results=[first,second],cfg=cfg,hashes=['a','b'],reviews={})
    activate_recording(state,pair,0)
    assert state['result'] is first and state['volume_4d']=='control movie'
    assert 'overlay_gif' not in state and 'review_frame' not in state
    edited = object()
    state['result'] = edited
    state['curation_audit'] = 'control audit'
    activate_recording(state,pair,1)
    assert state['result'] is second and state['curation_audit'] is None
    activate_recording(state,pair,0)
    assert state['result'] is edited and state['curation_audit']=='control audit'
    state['real_pair_output'] = pair
    restore_single_recording(state)
    assert state['result'] is original and state['volume_4d']=='original'


def test_new_pair_does_not_reuse_old_bound_results():
    cfg=Config()
    state=dict(result='stale',pair_active_index=0,pair_binding=['old','pair'])
    pair=dict(volumes=['new control','new variant'],results=[None,None],cfg=cfg,hashes=['new','pair'],reviews={})
    activate_recording(state,pair,0)
    assert state['result'] is None and state['volume_4d']=='new control'


def test_invalid_microscopy_is_rejected():
    for array in (np.zeros((2,3)),np.full((2,2,2,2),np.nan)):
        data=io.BytesIO();np.save(data,array)
        with pytest.raises(ValueError):read_movie(data.getvalue())


def test_comparison_only_uses_shared_frames():
    edges=pd.DataFrame(columns=['source_id','target_id'])
    control=AnalysisResult('control',(3,2,2,2),pd.DataFrame({'t':[0,1,2],'node_id':[1,2,3]}),edges,[None]*3)
    variant=AnalysisResult('variant',(2,2,2,2),pd.DataFrame({'t':[0,0,1],'node_id':[1,2,3]}),edges,[None]*2)
    comparison=compare_counts(control,variant)
    assert comparison.frame.tolist()==[0,1]
    assert comparison.detections_variant.tolist()==[2,1]


def test_each_recording_has_full_frame_limit_and_separate_config():
    cfg=Config()
    pair=dict(volumes=[np.zeros((31,2,2,2)),np.ones((4,2,2,2))],results=[None,None],
              cfg=cfg,hashes=['c','v'],reviews={})
    state=dict(cfg=cfg,preview_frames=20)
    activate_recording(state,pair,0)
    assert state['preview_frames']==31 and state['cfg'] is not cfg
    control_cfg=state['cfg']
    state['review_frame']=30
    activate_recording(state,pair,1)
    assert state['preview_frames']==4 and state['cfg'] is not control_cfg
    assert 'review_frame' not in state
    state['real_pair_output']=pair
    restore_single_recording(state)
    assert state['preview_frames']==20


def test_local_recording_is_memory_mapped_and_rejects_invalid_data(tmp_path):
    from app.paired_movies import read_local_movie
    path=tmp_path/'movie.npy'
    np.save(path,np.zeros((3,4,5,6),dtype=np.uint16))
    movie=read_local_movie(path)
    assert isinstance(movie,np.memmap) and movie.shape==(3,4,5,6)
    invalid=tmp_path/'invalid.npy'
    np.save(invalid,np.full((2,2,2,2),np.nan))
    with pytest.raises(ValueError):read_local_movie(invalid)


def test_recording_boundaries_are_not_reported_as_internal_endings():
    from app.pair_dashboard import continuity_table
    nodes=pd.DataFrame({'node_id':[1,2,3],'t':[0,1,2]})
    edges=pd.DataFrame({'source_id':[1,2],'target_id':[2,3]})
    r=AnalysisResult('fixture',(3,2,2,2),nodes,edges,[None]*3)
    table=continuity_table(r)
    assert np.isnan(table.starts.iloc[0]) and np.isnan(table.endings.iloc[-1])
    assert table.starts.iloc[1:].eq(0).all() and table.endings.iloc[:-1].eq(0).all()
