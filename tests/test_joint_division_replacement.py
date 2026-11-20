import numpy as np
import pytest
from biohub.joint_division import joint_divisions, validate_graph


def fixture():
    edges=[(1,2),(4,3),(2,5),(3,6),(5,7),(6,8),(9,10)]
    times={1:0,4:0,2:1,3:1,5:2,6:2,7:3,8:3,9:0,10:1}
    xyz={n:np.array([0.,0.,0.]) for n in times}
    xyz[2]=np.array([0.,0.,-.2]);xyz[3]=np.array([0.,0.,.2])
    return edges,times,xyz


def test_joint_replacement_preserves_unrelated_edges_and_records_termination():
    edges,times,xyz=fixture()
    result,info=joint_divisions(edges,{(1,2):.8,(1,3):.8,(4,3):.1},times,xyz,np.ones(3))
    assert result==[e for e in edges if e!=(4,3)]+[(1,3)]
    assert info['edits'][0]['unresolved_termination']==4
    validate_graph(result,times)


def test_rejects_strong_competitor_and_false_geometry():
    edges,times,xyz=fixture()
    weights={(1,2):.8,(1,3):.8,(4,3):.99}
    assert joint_divisions(edges,weights,times,xyz,np.ones(3))[0]==edges
    weights[(4,3)]=.1;xyz[3]=np.array([0,0,100])
    assert joint_divisions(edges,weights,times,xyz,np.ones(3))[0]==edges


def test_requires_both_scores_and_persistence():
    edges,times,xyz=fixture()
    assert joint_divisions(edges,{(1,3):.9},times,xyz,np.ones(3))[0]==edges
    shorter=[e for e in edges if e!=(6,8)]
    assert joint_divisions(shorter,{(1,2):.8,(1,3):.8,(4,3):.1},times,xyz,np.ones(3))[0]==shorter


def test_invalid_graph_fails():
    with pytest.raises(ValueError):validate_graph([(1,2),(3,2)],{1:0,3:0,2:1})
    with pytest.raises(ValueError):validate_graph([(1,2),(2,1)],{1:0,2:1})


def test_geometry_control_ignores_model_scores():
    edges,times,xyz=fixture()
    xyz[4]=np.array([0.,0.,30.])
    first=joint_divisions(edges,{},times,xyz,np.ones(3),score_mode='geometry')
    second=joint_divisions(edges,{(1,3):.001},times,xyz,np.ones(3),score_mode='geometry')
    assert first==second
    assert first[1]['accepted']==1
