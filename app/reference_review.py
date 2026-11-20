"""Detection agreement with published generated reference centres, not manual truth."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
GATE = 3.0  # Same voxel gate as the saved CShaper detector evaluation.


def match_centres(reference, predicted, gate=GATE):
    reference = np.asarray(reference, dtype=float).reshape(-1, 3)
    predicted = np.asarray(predicted, dtype=float).reshape(-1, 3)
    if not len(reference) or not len(predicted):
        return np.empty((0, 2), dtype=int), np.empty(0)
    distance = cdist(reference, predicted)
    rows, cols = linear_sum_assignment(np.where(distance <= gate, distance, (min(distance.shape)+1)*(gate+1)))
    valid = distance[rows, cols] <= gate
    return np.column_stack((rows[valid], cols[valid])), distance[rows[valid], cols[valid]]


def reference_for(result, volume):
    """Only attach a reference to the exact corresponding image array."""
    path = ROOT/'data/cshaper_demo/reference_centres.csv'
    movie = ROOT/'data/cshaper_demo/Sample18_t080_082.npy'
    if result.dataset_name != 'Sample18_t080_082' or not path.exists() or volume is None:
        return None
    original = np.load(movie, mmap_mode='r', allow_pickle=False)
    if original.shape != volume.shape or not np.array_equal(original, volume):
        return None
    return pd.read_csv(path)


def agreement_table(result, reference):
    records = []
    for t in range(len(result.frames)):
        truth = reference[reference.t.eq(t)][['z','y','x']].to_numpy()
        predicted = result.nodes[result.nodes.t.eq(t)][['z','y','x']].to_numpy()
        pairs, errors = match_centres(truth, predicted)
        tp, n, m = len(pairs), len(truth), len(predicted)
        records.append(dict(frame=t, reference_centres=n, predicted_centres=m, matched=tp,
                            unmatched_reference=n-tp, unmatched_predictions=m-tp,
                            precision=tp/m if m else np.nan, recall=tp/n if n else np.nan,
                            f1=2*tp/(m+n) if m+n else np.nan,
                            mean_error_voxels=float(errors.mean()) if len(errors) else np.nan))
    return pd.DataFrame(records)


def render_reference_agreement(result, volume, key):
    import streamlit as st
    import plotly.graph_objects as go
    reference = reference_for(result, volume)
    if reference is None:
        st.info('No verified matching reference labels are attached to this movie; input/output visualization alone cannot measure accuracy.')
        return
    table = agreement_table(result, reference)
    matched = table.matched.sum()
    n, m = table.reference_centres.sum(), table.predicted_centres.sum()
    st.markdown('**Prediction versus matching reference labels**')
    st.caption('Published generated SegNuc masks, not independently verified manual ground truth. One-to-one 3D centre matching within 3 voxels, fixed before this review. No threshold tuning performed.')
    cols = st.columns(3)
    cols[0].metric('Precision vs reference', f'{matched/m:.1%}' if m else 'N/A')
    cols[1].metric('Recall vs reference', f'{matched/n:.1%}' if n else 'N/A')
    cols[2].metric('F1 vs reference', f'{2*matched/(n+m):.1%}' if n+m else 'N/A')
    fig = go.Figure()
    for column, label in [('reference_centres','Reference centres'),('predicted_centres','Predicted centres'),('matched','Matched')]:
        fig.add_trace(go.Bar(x=table.frame.astype(str), y=table[column], name=label))
    fig.update_layout(barmode='group', xaxis_title='Test frame', yaxis_title='Centres', height=310,
                      template='plotly_white', margin=dict(t=20,b=30),legend=dict(orientation='h'))
    st.plotly_chart(fig,use_container_width=True,key=key+'_reference_counts')
    st.dataframe(table,hide_index=True)
    st.caption('Unmatched predictions may include real nuclei missing from the generated reference. These three frames cannot establish overall model accuracy or genetic defects.')
    st.download_button('Download reference comparison',table.to_csv(index=False),file_name='reference_agreement.csv',key=key+'_reference')
