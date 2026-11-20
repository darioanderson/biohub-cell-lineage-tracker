"""Restore complete, input-bound local runs without repeating inference."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from biohub.analysis import AnalysisResult, FrameResult
from biohub.manifest import config_sha256


def restore_biohub_run(pair, folder):
    folder=Path(folder)
    if json.loads((folder/'status.json').read_text()).get('state')!='complete':
        raise ValueError('Saved analysis is not complete.')
    hashes=json.loads((folder/'SHA256.json').read_text())
    for name,expected in hashes.items():
        if Path(name).name != name or hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Saved analysis checksum mismatch.')
    summary=json.loads((folder/'summary.json').read_text())
    results=[];checks=[]
    for slot,label,digest,movie,cfg in zip(('control','variant'),pair['labels'],pair['hashes'],pair['volumes'],pair['configs']):
        stats=json.loads((folder/(slot+'_stats.json')).read_text())
        if stats.get('input_sha256')!=digest:
            raise ValueError('Saved predictions belong to a different input.')
        effective=dict(stats['effective_config'])
        # The classical backend has no checkpoint; this unused path is irrelevant.
        if cfg.detector_backend=='peaks':effective['learned_detector_path']=str(cfg.learned_detector_path)
        comparable=cfg.copy_with(**effective)
        if config_sha256(comparable)!=config_sha256(cfg):
            raise ValueError('Saved predictions use different analysis settings.')
        nodes=pd.read_csv(folder/(slot+'_nodes.csv'));edges=pd.read_csv(folder/(slot+'_edges.csv'))
        if len(nodes)!=summary['recordings'][label]['nodes'] or len(edges)!=summary['recordings'][label]['edges']:
            raise ValueError('Saved prediction table sizes do not match their summary.')
        frames=[]
        for t in range(len(movie)):
            n=nodes[nodes.t.eq(t)]
            frames.append(FrameResult(t,n[['z','y','x']].to_numpy(float),n.raw_detection_score.to_numpy(float),n.intensity.to_numpy(float),n.node_id.astype(int).tolist()))
        results.append(AnalysisResult(label,tuple(movie.shape),nodes,edges,frames,stats))
        checks.append(dict(metrics=summary['recordings'][label]['sparse_reference']))
    pair.update(results=results,reviews={},checkpoint_sha256=None,reference_checks=checks,saved_run=str(folder))
