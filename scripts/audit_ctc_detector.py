"""Training-side validation failure audit; never reads sequence 02."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_ctc_detector import PROJECT, SPACING, VALID, Detector, load_frame, predict, decode, digest


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    checkpoint = root/'results/kaggle_cho_detector_v2/ctc_project/experiments/cho-centroid-001/best.pt'
    out = root/'results/cho_validation_failure_audit'
    out.mkdir(exist_ok=True)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    net = Detector(); net.load_state_dict(ck['model'])
    rows = []
    for frame in VALID:
        raw, image, gt = load_frame(PROJECT/'data/Fluo-N3DH-CHO', '01', frame)
        pred = decode(predict(net, image, 'cpu'), ck['threshold'])
        distances = cdist(pred*SPACING, gt*SPACING)
        penalty = (min(distances.shape)+1)*4
        ri, ci = linear_sum_assignment(np.where(distances <= 3, distances, penalty))
        valid = distances[ri, ci] <= 3
        pairs = dict(zip(ri[valid].tolist(), ci[valid].tolist()))
        neighbors = cdist(gt*SPACING, gt*SPACING); np.fill_diagonal(neighbors, np.inf)
        def border(points):
            return np.minimum(points*SPACING, (np.array(raw.shape)-1-points)*SPACING).min(axis=1)
        def xy_border(points):
            return np.minimum(points[:,1:]*SPACING[1:], (np.array(raw.shape)[1:]-1-points[:,1:])*SPACING[1:]).min(axis=1)
        row = {'frame': frame, 'tp': len(pairs), 'fp': len(pred)-len(pairs), 'fn': len(gt)-len(pairs),
               'matched_distances_um': [float(distances[i,j]) for i,j in pairs.items()],
               'truth': [{'zyx': p.tolist(), 'matched': i in pairs.values(),
                          'border_um': float(border(gt)[i]),
                          'xy_border_um': float(xy_border(gt)[i]),
                          'nearest_annotated_neighbor_um': float(neighbors[i].min()) if len(gt)>1 else None}
                         for i,p in enumerate(gt)],
               'predictions': [{'zyx':p.tolist(), 'matched': i in pairs,
                                'border_um': float(border(pred)[i]),
                                'xy_border_um': float(xy_border(pred)[i]),
                                'nearest_truth_um': float(distances[i].min()) if len(gt) else None}
                               for i,p in enumerate(pred)]}
        rows.append(row)
        fig, ax = plt.subplots(figsize=(9,9))
        ax.imshow(raw.max(axis=0), cmap='gray', vmin=np.percentile(raw,1), vmax=np.percentile(raw,99.8))
        ax.scatter(gt[:,2],gt[:,1],s=100,facecolors='none',edgecolors='lime',label='Annotated centers')
        for matched, color, label in [(True,'cyan','Matched predictions'),(False,'red','Unmatched predictions')]:
            pts=np.array([p for i,p in enumerate(pred) if (i in pairs)==matched]).reshape(-1,3)
            ax.scatter(pts[:,2],pts[:,1],s=24,c=color,marker='x',label=label)
        ax.set_title(f'Sequence 01 validation frame {frame}: TP {len(pairs)}, FP {len(pred)-len(pairs)}, FN {len(gt)-len(pairs)}\nXY maximum projection; matching uses 3D physical distances')
        ax.legend(); fig.tight_layout(); fig.savefig(out/f'frame_{frame:03d}.png',dpi=120);plt.close(fig)
    summary={'split':'sequence01 validation only', 'checkpoint_sha256':digest(checkpoint),
             'threshold':ck['threshold'], 'rows':rows,
             'limitations':['Same validation frames selected checkpoint; descriptive audit, not independent evidence',
                            'Maximum projections hide Z overlap; 3D matching unchanged',
                            'Nearest annotated neighbor is not an image-derived density estimate']}
    (out/'audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({k:sum(r[k] for r in rows) for k in ['tp','fp','fn']}))
    for group in ['truth','predictions']:
        for matched in [True,False]:
            pts=[p for r in rows for p in r[group] if p['matched']==matched]
            print(group,'matched',matched,'count',len(pts),'within 1um boundary',sum(p['border_um']<=1 for p in pts))


if __name__=='__main__': main()
