import io
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from biohub.analysis import AnalysisResult, run_tracking_pipeline
from biohub.config import Config
from biohub.scientific import build_scientific_tables


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def test_new_input_invalidates_predictions_and_review(monkeypatch):
    import app.ui as ui
    state = State(cfg=Config(),result=object(),curation_audit='old',overlay_gif=b'old',review_frame=99)
    monkeypatch.setattr(ui,'st',SimpleNamespace(session_state=state))
    data=io.BytesIO();np.save(data,np.ones((2,3,4,5),np.uint16))
    ui.load_uploaded_npy(data.getvalue(),2)
    assert state.result is None and state.curation_audit is None
    assert 'overlay_gif' not in state and 'review_frame' not in state
    assert state.input_sha256 and state.cfg.voxel_size_zyx_um==(1.,1.,1.)


def test_empty_prediction_supports_scientific_views():
    r=run_tracking_pipeline(np.zeros((2,8,24,24),np.uint16),Config())
    assert r.n_nodes==0
    tables=build_scientific_tables(r,Config())
    assert tables['observations'].empty
    assert {'node_id','track_id','lineage_id','t'}.issubset(tables['observations'])


def fixture_result():
    return AnalysisResult('test',(2,2,2,2),pd.DataFrame(dict(
        node_id=[1,2],track_id=[1,1],lineage_id=[1,1],t=[0,1],z=[0.,0.],y=[0.,0.],x=[0.,2.])),
        pd.DataFrame(dict(source_id=[1],target_id=[2])),[None,None])


def test_scientific_measurements_use_saved_calibration_not_current_widgets():
    r=fixture_result()
    r.stats={'effective_config':dict(voxel_size_zyx_um=[1.,1.,3.],frame_interval_s=2.),
             'time_interval_verified':True,'spatial_calibration_verified':True}
    t=build_scientific_tables(r,Config(voxel_size_zyx_um=(9.,9.,9.),frame_interval_s=99.))
    assert t['tracks'].iloc[0].mean_speed_um_s==3.
    r.stats['time_interval_verified']=False
    assert pd.isna(build_scientific_tables(r,Config())['tracks'].iloc[0].mean_speed_um_s)


def test_atlas_rejects_pseudoreplication_and_missing_embryos():
    from biohub.comparative_development import build_interdivision_atlas
    table=pd.DataFrame(dict(canonical_parent_id=['AB','AB'],time_s=[10.,12.],embryo_id=['same','same']))
    with pytest.raises(ValueError,match='exactly one'):build_interdivision_atlas(table)
    with pytest.raises(ValueError,match='embryo_id'):build_interdivision_atlas(table.drop(columns='embryo_id'))


def test_unknown_recording_timing_is_never_inferred_from_other_input(tmp_path):
    from app.recording_metadata import known_recording
    p=tmp_path/'local_biohub_pair';p.mkdir()
    (p/'PROVENANCE.json').write_text(json.dumps(dict(recordings=dict(control=dict(sha256='known',voxel_size_zyx_um=[1.,2.,3.],frame_interval_s=10.)))))
    assert known_recording('known',tmp_path)['time_verified']
    assert not known_recording('unknown',tmp_path)['time_verified']


def test_changed_pair_clears_alignment():
    from app.paired_movies import reset_pair_alignment
    state=dict(alignment_input_hashes=['a','b'],real_pair_aligned=True)
    reset_pair_alignment(state,['a','c'])
    assert not state['real_pair_aligned']


@pytest.mark.parametrize('backend',['peaks','cellpose_projection','cshaper'])
def test_export_manifest_names_actual_backend(backend):
    from app.paired_movies import pair_manifest
    cfg=Config(detector_backend=backend)
    r=fixture_result();r.stats['effective_config']={'frame_interval_s':3.}
    p=dict(cfg=cfg,hashes=['a','b'],filenames=['a.npy','b.npy'],volumes=[np.zeros((2,2,2,2))]*2,results=[r,r])
    manifest=pair_manifest(p)
    assert manifest['detector_backend']==backend
    assert manifest['recordings'][0]['effective_config']['frame_interval_s']==3.
    assert not manifest['genetic_diagnosis']


def test_metadata_converts_units_and_refuses_unitless_scales(tmp_path):
    from biohub.data import read_physical_metadata
    axes=[dict(name=n,unit=u) for n,u in zip('TZYX',['minute','nanometer','nanometer','nanometer'])]
    multiscale=dict(axes=axes,datasets=[dict(coordinateTransformations=[dict(type='scale',scale=[2,1000,500,500])])])
    p=tmp_path/'zarr.json';p.write_text(json.dumps(dict(attributes=dict(multiscales=[multiscale]))))
    assert read_physical_metadata(tmp_path)==((1.,.5,.5),120.)
    for a in axes:a.pop('unit')
    p.write_text(json.dumps(dict(attributes=dict(multiscales=[multiscale]))))
    assert read_physical_metadata(tmp_path)==(None,None)


def test_sparse_pair_reference_does_not_turn_unknown_into_false_positive(monkeypatch):
    import app.biohub_pair_reference as ref
    r=fixture_result()
    # One extra detection has no annotation and must remain unknown.
    r.nodes.loc[2]=[3,2,2,0,100.,100.,100.]
    r.edges.loc[1]=[3,2]
    gt=pd.DataFrame(dict(node_id=[11,12],t=[0,1],z=[0.,0.],y=[0.,0.],x=[0.,2.]))
    monkeypatch.setattr(ref,'load_geff_graph',lambda _: (gt,pd.DataFrame(dict(source_id=[11],target_id=[12]))))
    metrics=ref.evaluate_reference(r,'unused',(1.,1.,1.))
    assert metrics['edge_tp']==1 and metrics['edge_fp']==0 and metrics['edge_unknown']==1
    assert metrics['annotated_detection_recall']==1.


def test_reference_matching_preserves_large_unsigned_geff_ids():
    from biohub.tuning import _match_nodes_to_gt
    identities=[2**32+1345294339,2**63+17]
    gt=pd.DataFrame(dict(node_id=np.array(identities,dtype=np.uint64),z=[0.,1.],y=[0.,0.],x=[0.,0.]))
    matches=_match_nodes_to_gt(np.array([[0.,0.,0.],[1.,0.,0.]]),gt,np.ones(3))
    assert matches=={0:identities[0],1:identities[1]}


def test_selecting_new_dataset_clears_previous_images(monkeypatch):
    import app.ui as ui
    entry=SimpleNamespace(name='new')
    state=State(result=object(),selected_entry='old',volume_4d='old pixels',catalog=SimpleNamespace(by_label=lambda _:entry))
    monkeypatch.setattr(ui,'st',SimpleNamespace(session_state=state))
    ui.select_dataset_entry('new')
    assert state.result is None and state.volume_4d is None and state.selected_entry is entry


def test_saved_run_rejects_corrupted_artifacts(tmp_path):
    from app.saved_runs import restore_biohub_run
    (tmp_path/'status.json').write_text(json.dumps(dict(state='complete')))
    (tmp_path/'table.csv').write_text('tampered')
    (tmp_path/'SHA256.json').write_text(json.dumps({'table.csv':'wrong'}))
    with pytest.raises(ValueError,match='checksum'):restore_biohub_run({},tmp_path)


def test_learned_linker_refuses_unbounded_dense_allocation(tmp_path):
    from biohub.learned_linker import build_learned_link_cost
    checkpoint=tmp_path/'model.pt';checkpoint.touch()
    with pytest.raises(ValueError,match='bounded dense-matrix'):
        build_learned_link_cost(np.zeros((3000,3)),np.zeros((3000,3)),checkpoint)


def test_global_graph_preview_cannot_readd_all_nodes_via_edges(monkeypatch):
    import networkx as nx
    from biohub.visualization import plot_lineage_graph
    import matplotlib.pyplot as plt
    nodes=pd.DataFrame(dict(node_id=range(300),t=range(300)))
    edges=pd.DataFrame(dict(source_id=range(299),target_id=range(1,300)))
    r=AnalysisResult('large',(300,2,2,2),nodes,edges)
    sizes=[]
    def layout(graph,**kwargs):
        sizes.append(len(graph));return {n:np.array([i,0.]) for i,n in enumerate(graph)}
    monkeypatch.setattr(nx,'spring_layout',layout)
    figure=plot_lineage_graph(r,max_nodes=12)
    assert sizes==[12]
    plt.close(figure)
