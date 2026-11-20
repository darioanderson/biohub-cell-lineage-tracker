from scripts.evaluate_joint_expansion import audit_edits


def test_audit_exposes_lost_known_edge_and_unknown_addition():
    edits=[{'added':[1,3],'removed':[2,3]},{'added':[1,4],'removed':None}]
    result=audit_edits(edits,{1:10,2:20,3:30},{(20,30)})
    assert result['known_edges_removed']==1
    assert result['known_edges_added']==0
    assert result['edits'][0]['added_support']=='inconsistent'
    assert result['edits'][1]['added_support']=='unknown'


def test_audit_counts_recovered_edge():
    result=audit_edits([{'added':[1,3],'removed':[2,3]}],{1:10,2:20,3:30},{(10,30)})
    assert result['known_edges_added']==1
    assert result['edits'][0]['removed_support']=='inconsistent'
