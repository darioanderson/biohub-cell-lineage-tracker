"""Sparse GEFF reference checks using the unchanged production evaluator."""
from pathlib import Path
import numpy as np
import pandas as pd
from biohub.data import load_geff_graph
from biohub.tuning import _match_nodes_to_gt, evaluate_sparse_graph


def evaluate_reference(result, geff_path, scale):
    gt_nodes, gt_edges = load_geff_graph(Path(geff_path))
    times = list(range(len(result.frames)))
    gt_nodes = gt_nodes[gt_nodes.t.isin(times)]
    mapping = {}
    distances = []
    for t, pred in result.nodes.groupby('t'):
        pred = pred.reset_index(drop=True)
        gt = gt_nodes[gt_nodes.t.eq(t)]
        matches = _match_nodes_to_gt(pred[['z','y','x']].to_numpy(float), gt, np.asarray(scale))
        gt_by_id = gt.set_index('node_id')
        for i, identity in matches.items():
            mapping[int(pred.iloc[i].node_id)] = identity
            delta = (pred.iloc[i][['z','y','x']].to_numpy(float)-gt_by_id.loc[identity,['z','y','x']].to_numpy(float))*scale
            distances.append(float(np.linalg.norm(delta)))
    edges = list(result.edges[['source_id','target_id']].itertuples(index=False,name=None))
    metrics = evaluate_sparse_graph(gt_nodes,gt_edges,edges,mapping,evaluable_times=times)
    metrics.update(annotated_observations=len(gt_nodes), matched_annotations=len(set(mapping.values())),
                   annotated_detection_recall=len(set(mapping.values()))/len(gt_nodes) if len(gt_nodes) else None,
                   matched_localization_mean_um=float(np.mean(distances)) if distances else None,
                   match_gate_um=7., precision_scope='Conditional on endpoints matched to sparse annotations; unmatched detections/edges remain unknown')
    return metrics


def attach_pair_references(pair):
    checks = []
    for result, info, cfg in zip(pair['results'],pair.get('metadata',[]),pair['configs']):
        path = info.get('entry',{}).get('reference_geff')
        if not path or not Path(path).exists():
            checks.append(None)
            continue
        try:
            checks.append(dict(metrics=evaluate_reference(result,path,cfg.scale_array),source=path))
        except (RuntimeError,ValueError,OSError,ImportError) as error:
            checks.append(dict(error=str(error),source=path))
    pair['reference_checks'] = checks
