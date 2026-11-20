"""Read-only division preflight; never selects thresholds or modifies a graph."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from biohub.config import Config, MATCH_GATE_UM
from biohub.data import list_datasets, find_geff_path, read_physical_metadata, load_geff_graph
from scripts.evaluate_detected_linker import build_detected_sample, embryo_id
from scripts.evaluate_trackastra import run_classical_edges, run_trackastra


def division_ceiling(sample, scale, gate_um=MATCH_GATE_UM):
    """Report raw-neighborhood and evaluator-matched coverage independently."""
    frames = {f.t: f for f in sample.frames}
    nodes = sample.gt_nodes.set_index('node_id')
    children = defaultdict(set)
    for a, b in sample.gt_edges[['source_id', 'target_id']].to_numpy():
        children[int(a)].add(int(b))
    matched = {g for f in sample.frames for g in f.pred_to_gt.values()}
    trees = {t: cKDTree(f.xyz * scale) for t, f in frames.items()}
    events = []
    for parent, daughters in sorted(children.items()):
        if len(daughters) < 2:
            continue
        ids = [parent, *sorted(daughters)]
        if any(i not in nodes.index for i in ids):
            continue
        times = [int(nodes.loc[i, 't']) for i in ids]
        if any(t not in frames for t in times):
            continue
        row = {'parent_gt': parent, 'daughter_gt': ids[1:], 'times': times,
               'binary_adjacent': len(daughters) == 2 and all(t == times[0] + 1 for t in times[1:]),
               'matched_daughters': all(i in matched for i in ids[1:]),
               'matched_triplet': all(i in matched for i in ids)}
        neighbors = [trees[t].query_ball_point(nodes.loc[i, ['z','y','x']].to_numpy(dtype=float) * scale, gate_um)
                     for i, t in zip(ids, times)]
        # Distinct detections are essential: one merged blob cannot cover two daughters.
        row['raw_two_distinct_daughters'] = bool(len(daughters) == 2 and neighbors[1] and neighbors[2]
            and (times[1] != times[2] or len(set(neighbors[1]) | set(neighbors[2])) >= 2))
        row['raw_parent_present'] = bool(neighbors[0])
        events.append(row)
    n = len(events)
    return {'gt_divisions': n, 'match_gate_um': gate_um,
            **{k: sum(bool(r[k]) for r in events) for k in
               ('binary_adjacent', 'matched_daughters', 'matched_triplet', 'raw_two_distinct_daughters')},
            'matched_triplet_fraction': sum(r['matched_triplet'] for r in events) / n if n else None,
            'events': events}


def inspect_conflicts(sample, baseline, proposals, scores, cfg, limit=20):
    """Disaggregate the historical conflict gate, without changing its order."""
    children, proposed, incoming = defaultdict(list), defaultdict(set), defaultdict(list)
    for a, b in baseline:
        children[a].append(b)
        incoming[b].append(a)
    for a, b in proposals:
        proposed[a].add(b)
    times = {n: f.t for f in sample.frames for n in f.ids}
    xyz = {n: f.xyz[j] for f in sample.frames for j, n in enumerate(f.ids)}
    gt = {n: g for f in sample.frames for n, g in f.pred_to_gt.items()}
    gt_edges = set(map(tuple, sample.gt_edges[['source_id','target_id']].to_numpy()))
    counts, examples = Counter(), []
    for parent, daughters in sorted(proposed.items()):
        if len(daughters) != 2:
            continue
        d = sorted(daughters)
        reason = 'eligible'
        if len(children[parent]) != 1:
            reason = 'parent_baseline_child_count_not_one'
        elif children[parent][0] not in daughters:
            reason = 'baseline_continuation_not_a_proposed_daughter'
        else:
            second = next(i for i in d if i != children[parent][0])
            if incoming[second]:
                reason = 'second_daughter_already_owned'
            elif any(n not in times or n not in xyz for n in [parent, *d]):
                reason = 'missing_node'
            elif any(times[n] != times[parent] + 1 for n in d):
                reason = 'nonadjacent_time'
        geometry = None
        if all(n in xyz for n in [parent, *d]):
            p, a, b = [np.asarray(xyz[n]) * cfg.scale_array for n in [parent, *d]]
            geometry = {'parent_distances_um': [float(np.linalg.norm(a-p)), float(np.linalg.norm(b-p))],
                        'sister_um': float(np.linalg.norm(a-b)),
                        'midpoint_um': float(np.linalg.norm((a+b)/2-p))}
            passes = max(geometry['parent_distances_um']) <= cfg.div_parent_dist_um and geometry['sister_um'] <= cfg.div_sister_dist_um and geometry['midpoint_um'] <= cfg.div_midpoint_dist_um
            if reason == 'eligible' and not passes:
                reason = 'geometry'
        counts[reason] += 1
        label = 'unknown'
        if all(n in gt for n in [parent, *d]):
            label = 'gt_supported_pair' if all((gt[parent], gt[n]) in gt_edges for n in d) else 'gt_inconsistent_pair'
        # Deterministic examples spread across categories, rather than cherry-picked scores.
        if reason not in ('eligible', 'geometry') and sum(e['reason'] == reason for e in examples) < limit:
            examples.append({'parent': parent, 'daughters': d, 'reason': reason,
                'baseline_children': children[parent], 'daughter_incoming': [incoming[n] for n in d],
                'times': [times.get(n) for n in [parent,*d]], 'geometry': geometry,
                'scores': [scores.get((parent,n)) for n in d], 'sparse_label': label})
    return {'counts': dict(counts), 'examples': examples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', type=Path)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--samples-per-embryo', type=int, default=2)
    parser.add_argument('--division-bearing', action='store_true')
    parser.add_argument('--inspect-candidates', action='store_true')
    args = parser.parse_args()
    import torch
    from trackastra.model import Trackastra
    if not torch.cuda.is_available():
        raise RuntimeError('Diagnostic Trackastra run requires CUDA; refusing silent CPU fallback')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(n for n in list_datasets(args.train_dir) if find_geff_path(args.train_dir,n))
    if args.samples_per_embryo < 1:
        raise ValueError('samples-per-embryo must be positive')
    if args.division_bearing:
        from scripts.division_candidate_audit import adjacent_division_count
        inventory = {}
        for n in names:
            nodes, edges = load_geff_graph(find_geff_path(args.train_dir,n))
            inventory[n] = adjacent_division_count(nodes,edges)
        (out/'annotation_inventory.json').write_text(json.dumps(inventory,indent=2))
        names = [n for n in names if inventory[n] > 0]
    selected, counts = [], Counter()
    for n in names:
        group = embryo_id(n)
        if counts[group] < args.samples_per_embryo:
            selected.append(n)
            counts[group] += 1
    if len(counts) < 2 or any(c != args.samples_per_embryo for c in counts.values()):
        raise ValueError('Insufficient eligible volumes in at least two embryo groups')
    manifest = {'samples': selected, 'selection': 'lexicographic first two per embryo, before metrics',
                'purpose': 'diagnostic only; no threshold selection or adoption verdict',
                'device': torch.cuda.get_device_name(0), 'match_gate_um': MATCH_GATE_UM}
    manifest['selection'] = f'lexicographic first {args.samples_per_embryo} per embryo; division-bearing={args.division_bearing}; annotation-only selection before model inference'
    (out/'selection.json').write_text(json.dumps(manifest, indent=2))
    model = Trackastra.from_folder(args.model_dir, device='cuda')
    if args.inspect_candidates:
        from scripts.division_candidate_audit import PredictionCapture
        model = PredictionCapture(model)
    cfg = Config().competition_v10_dog_preset().copy_with(use_competition_preset=False)
    for n in selected:
        start = time.perf_counter()
        (out/'status.json').write_text(json.dumps({'state':'running','sample':n}))
        voxel, interval = read_physical_metadata(args.train_dir/f'{n}.zarr')
        local_cfg = cfg.copy_with(voxel_size_zyx_um=voxel or cfg.voxel_size_zyx_um,
                                  frame_interval_s=interval or cfg.frame_interval_s)
        sample = build_detected_sample(args.train_dir,n,local_cfg)
        detection_seconds = time.perf_counter()-start
        ceiling = division_ceiling(sample,local_cfg.scale_array)
        if args.division_bearing and ceiling['binary_adjacent'] == 0:
            raise RuntimeError(f'{n}: annotated divisions fell outside processed frames')
        start = time.perf_counter()
        baseline = run_classical_edges(sample,local_cfg)
        classical_seconds = time.perf_counter()-start
        start = time.perf_counter()
        proposals,scores,info = run_trackastra(model,args.train_dir,sample,'greedy')
        trackastra_seconds = time.perf_counter()-start
        report = {'sample':n,'embryo':sample.embryo,'ceiling':ceiling,
                  'conflicts':inspect_conflicts(sample,baseline,proposals,scores,local_cfg),
                  'seconds':{'detection':detection_seconds,'classical':classical_seconds,'trackastra':trackastra_seconds},'trackastra':info}
        candidate_snapshot = {}
        if args.inspect_candidates:
            report['candidate_audit'], candidate_snapshot = model.audit(sample,ceiling,proposals)
        (out/f'{n}.json').write_text(json.dumps(report,indent=2,default=float))
        snapshot = {'baseline':baseline,'trackastra':proposals,'scores':[[a,b,s] for (a,b),s in scores.items()],
                    **candidate_snapshot,
                    'scale':local_cfg.scale_array.tolist(),
                    'frames':[{'t':f.t,'ids':f.ids,'xyz':f.xyz.tolist(),'pred_to_gt':f.pred_to_gt} for f in sample.frames]}
        with gzip.open(out/f'{n}.graphs.json.gz','wt') as f:
            json.dump(snapshot,f)
        print('DIAGNOSTIC',n,json.dumps({k:v for k,v in ceiling.items() if k != 'events'}),report['conflicts']['counts'],flush=True)
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name != 'status.json'}
    (out/'hashes.json').write_text(json.dumps(hashes,indent=2))
    (out/'status.json').write_text(json.dumps({'state':'completed','samples':selected,'adoption_verdict':None}))


if __name__ == '__main__':
    main()
