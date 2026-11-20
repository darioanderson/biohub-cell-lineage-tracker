"""Instrumentation for pinned Trackastra 0.5.3; never changes its predictions."""
from collections import defaultdict
import networkx as nx
from biohub.trackastra_adapter import graph_edge_scores_to_detection_ids


def adjacent_division_count(nodes, edges):
    times = dict(zip(nodes.node_id.astype(int), nodes.t.astype(int)))
    children = defaultdict(set)
    for a, b in edges[['source_id', 'target_id']].to_numpy():
        children[int(a)].add(int(b))
    return sum(len(ds) == 2 and p in times and
               all(d in times and times[d] == times[p] + 1 for d in ds)
               for p, ds in children.items())


def classify_edge(edge, raw, candidates, selected):
    if None in edge:
        return 'missing_detection'
    if edge not in raw:
        return 'no_model_weight'
    if edge not in candidates:
        return 'candidate_graph_pruned'
    if raw[edge] < 0.5:
        return 'below_default_greedy_threshold'
    return 'selected' if edge in selected else 'greedy_conflict'


class PredictionCapture:
    def __init__(self, model):
        self.model = model
        self.raw = self.candidates = None

    def track(self, *args, **kwargs):
        from importlib.metadata import version
        if version('trackastra') != '0.5.3':
            raise RuntimeError('Candidate audit requires inspected Trackastra 0.5.3')
        from trackastra.tracking.tracking import build_graph
        original = self.model._track_from_predictions

        def capture(predictions, mode='greedy', **options):
            if options:
                raise ValueError('Audit expects the unchanged default graph gates')
            self.raw = nx.DiGraph()
            for n in predictions['nodes']:
                self.raw.add_node(n['id'], time=n['time'], label=n['label'])
            for (a, b), weight in predictions['weights']:
                self.raw.add_edge(a, b, weight=float(weight))
            self.candidates = build_graph(predictions['nodes'], predictions['weights'],
                use_distance=False, max_distance=256, max_neighbors=10, delta_t=1)
            return original(predictions, mode=mode)

        self.model._track_from_predictions = capture
        try:
            return self.model.track(*args, **kwargs)
        finally:
            self.model._track_from_predictions = original

    def audit(self, sample, ceiling, selected):
        frame_ids = [f.ids for f in sample.frames]
        raw = graph_edge_scores_to_detection_ids(self.raw, frame_ids)
        candidates = graph_edge_scores_to_detection_ids(self.candidates, frame_ids)
        mapping = {g:n for f in sample.frames for n,g in f.pred_to_gt.items()}
        selected = set(map(tuple, selected))
        events = []
        for event in ceiling['events']:
            parent = mapping.get(event['parent_gt'])
            edges = [(parent, mapping.get(d)) for d in event['daughter_gt']]
            events.append({'parent_gt':event['parent_gt'], 'binary_adjacent':event['binary_adjacent'],
                'daughter_edges':[{'edge':edge, 'model_weight':raw.get(edge),
                    'stage':classify_edge(edge, raw, candidates, selected)} for edge in edges]})
        return {'events':events, 'raw_weight_count':len(raw), 'candidate_edge_count':len(candidates)}, {
            'raw_weights':[[a,b,w] for (a,b),w in raw.items()],
            'candidate_weights':[[a,b,w] for (a,b),w in candidates.items()]}
