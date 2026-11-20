"""Experimental local division replacement; fixed diagnostic scoring, not production."""
from collections import defaultdict
import math
import numpy as np


def validate_graph(edges, times):
    incoming, outgoing = defaultdict(int), defaultdict(int)
    if len(set(edges)) != len(edges):
        raise ValueError('Duplicate edges')
    for a, b in edges:
        if a not in times or b not in times or times[b] <= times[a]:
            raise ValueError('Missing node or non-forward edge (cycle risk)')
        incoming[b] += 1
        outgoing[a] += 1
    if any(v > 1 for v in incoming.values()) or any(v > 2 for v in outgoing.values()):
        raise ValueError('Invalid graph degree')


def joint_divisions(baseline, weights, times, xyz, scale, *, parent_gate=12.,
                    sister_gate=7.5, midpoint_gate=9., minimum_score=.05,
                    margin=.5, persistence=2, score_mode='trackastra'):
    """Compare joint log-score evidence against displaced continuation evidence.

    Preserve the original parent continuation. Release at most one competing
    edge, recording its source as an unresolved termination rather than inventing
    a replacement. Scores are uncalibrated; this objective needs held-out tests.
    """
    if min(parent_gate,sister_gate,midpoint_gate) <= 0 or not 0 < minimum_score < 1 or margin < 0 or persistence < 0:
        raise ValueError('Invalid gates')
    if score_mode not in ('trackastra', 'geometry'):
        raise ValueError('Unknown score mode')
    original = [tuple(e) for e in baseline]
    validate_graph(original,times)
    children, incoming = defaultdict(list), {}
    for a,b in original:
        children[a].append(b); incoming[b]=a
    def persists(n,steps):
        if steps == 0:return True
        return any(times[c]==times[n]+1 and persists(c,steps-1) for c in children[n])
    def logit(w):
        w=float(np.clip(w,1e-6,1-1e-6));return math.log(w/(1-w))
    candidates=[]
    if score_mode == 'geometry':
        # Build candidates without Trackastra proposals or scores.
        from scipy.spatial import cKDTree
        frames=defaultdict(list)
        for n,t in times.items():frames[t].append(n)
        weights={}
        for t,parents in frames.items():
            daughters=frames.get(t+1,[])
            if not daughters:continue
            tree=cKDTree(np.asarray([xyz[n] for n in daughters])*scale)
            for p in parents:
                for j in tree.query_ball_point(np.asarray(xyz[p])*scale,parent_gate):
                    weights[p,daughters[j]]=.5
    for (p,d),w in sorted(weights.items()):
        if p not in times or d not in times or times[d]!=times[p]+1 or len(children[p])!=1:continue
        a=children[p][0]
        if a==d or times[a]!=times[d]:continue
        first=weights.get((p,a))
        if first is None or not np.isfinite(w) or not np.isfinite(first) or min(w,first)<minimum_score:continue
        old=incoming.get(d)
        if old is not None and (len(children[old])!=1 or (score_mode=='trackastra' and (old,d) not in weights)):continue
        competitor=weights.get((old,d),.5)
        if not np.isfinite(competitor):continue
        pp,aa,dd=[np.asarray(xyz[n])*scale for n in (p,a,d)]
        distances=[np.linalg.norm(aa-pp),np.linalg.norm(dd-pp)]
        sister=np.linalg.norm(aa-dd);mid=np.linalg.norm((aa+dd)/2-pp)
        if max(distances)>parent_gate or sister>sister_gate or mid>midpoint_gate:continue
        if not persists(a,persistence) or not persists(d,persistence):continue
        # Shared first edge cancels between hypotheses. Penalize a split and
        # displaced termination; joint geometry discourages implausible pairs.
        geometry=mid/midpoint_gate+sister/sister_gate+abs(distances[0]-distances[1])/parent_gate
        gain=logit(w)-logit(competitor)-geometry-(.5 if old is not None else 0.)
        if score_mode == 'geometry':
            alternative=float(np.linalg.norm((np.asarray(xyz[old])-np.asarray(xyz[d]))*scale)) if old is not None else parent_gate
            gain=(alternative-distances[1])/parent_gate-geometry-(.5 if old is not None else 0.)
        if gain>=margin:candidates.append((gain,p,a,d,old))
    result=list(original); touched=set(); edits=[]
    for gain,p,a,d,old in sorted(candidates,key=lambda x:(-x[0],x[1],x[3])):
        neighborhood={p,a,d}|({old} if old is not None else set())
        if touched & neighborhood:continue
        removed=(old,d) if old is not None else None
        if removed:result.remove(removed)
        result.append((p,d)); validate_graph(result,times)
        edits.append({'parent':p,'daughters':[a,d],'added':[p,d],
                      'removed':list(removed) if removed else None,
                      'unresolved_termination':old,'gain':float(gain)})
        touched.update(neighborhood)
    removed={tuple(e['removed']) for e in edits if e['removed']}
    added={tuple(e['added']) for e in edits}
    if [e for e in result if e not in added] != [e for e in original if e not in removed]:
        raise RuntimeError('Unrelated edge ordering changed')
    return result,{'accepted':len(edits),'eligible':len(candidates),'edits':edits,
                   'unrelated_edges_unchanged':True}
