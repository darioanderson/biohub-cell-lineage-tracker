import pandas as pd
from scripts.division_candidate_audit import adjacent_division_count, classify_edge


def test_annotation_selection_rejects_gap_and_continuation():
    nodes = pd.DataFrame({'node_id':[1,2,3], 't':[0,1,1]})
    edges = pd.DataFrame({'source_id':[1,1], 'target_id':[2,3]})
    assert adjacent_division_count(nodes,edges) == 1
    assert adjacent_division_count(nodes,edges.iloc[:1]) == 0
    nodes.loc[2,'t'] = 2
    assert adjacent_division_count(nodes,edges) == 0


def test_stages_distinguish_model_graph_threshold_and_assignment():
    e=(1,2)
    assert classify_edge((1,None),{},{},set()) == 'missing_detection'
    assert classify_edge(e,{},{},set()) == 'no_model_weight'
    assert classify_edge(e,{e:.8},{},set()) == 'candidate_graph_pruned'
    assert classify_edge(e,{e:.4},{e:.4},set()) == 'below_default_greedy_threshold'
    assert classify_edge(e,{e:.8},{e:.8},set()) == 'greedy_conflict'
    assert classify_edge(e,{e:.8},{e:.8},{e}) == 'selected'
