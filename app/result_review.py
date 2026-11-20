"""Interactive inspection of actual pipeline predictions."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def projection_figure(volume, nodes, z=None, mask=None):
    image = volume.max(axis=0) if z is None else volume[z]
    low, high = np.percentile(image, (1, 99.8))
    gray = (255*np.clip((image.astype(float)-low)/max(high-low, 1), 0, 1)).astype(np.uint8)
    fig = go.Figure(go.Image(z=np.repeat(gray[..., None], 3, axis=-1), hoverinfo="skip"))
    shown = nodes if z is None else nodes[nodes.z.sub(z).abs().le(1)]
    fig.add_trace(go.Scatter(x=shown.x, y=shown.y, mode="markers",
        customdata=shown[["node_id", "z"]].to_numpy(),
        marker=dict(size=11, symbol="circle-open", color=shown.z, colorscale="Turbo", line=dict(width=2), colorbar=dict(title="Z voxel")),
        hovertemplate="Cell %{customdata[0]:.0f}<br>x=%{x:.1f}, y=%{y:.1f}<br>z=%{customdata[1]:.1f}<extra></extra>"))
    if z is None and mask is not None:
        from skimage.measure import find_contours
        for identity in np.unique(mask[mask>0]):
            for contour in find_contours(mask==identity,.5):
                fig.add_trace(go.Scatter(x=contour[:,1]*2,y=contour[:,0]*2,mode='lines',
                    line=dict(color='#24d7c3',width=1.5),hoverinfo='skip',name='2D nucleus outline'))
    fig.update_layout(height=580, margin=dict(l=15,r=15,t=45,b=20),
        title="Whole-volume maximum projection" if z is None else f"Z={z} (centres within ±1 voxel)", template="plotly_white", showlegend=False)
    fig.update_xaxes(title="X (pixels)", range=[-.5,image.shape[1]-.5])
    fig.update_yaxes(title="Y (pixels)", autorange="reversed", scaleanchor="x")
    return fig


def frame_measurements(result):
    records = []
    for t in range(len(result.frames)):
        current = result.nodes[result.nodes.t.eq(t)]
        outgoing = result.edges[result.edges.source_id.isin(current.node_id)] if not result.edges.empty else result.edges
        divisions = int(outgoing.groupby("source_id").size().eq(2).sum()) if len(outgoing) else 0
        records.append(dict(frame=t, detections=len(current), outgoing_links=len(outgoing), proposed_divisions=divisions))
    return pd.DataFrame(records)


def render_real_summary(result, key="real_summary"):
    st.markdown("**Measured model output — current recording**")
    st.caption(result.dataset_name + " · " + str(result.stats.get("source", "Pipeline predictions")))
    st.caption("One recording only: comparing frames, not control versus genetic variant. Counts are predictions, not accuracy measurements.")
    from app.reference_review import render_reference_agreement
    render_reference_agreement(result, st.session_state.get('volume_4d'), key)
    table = frame_measurements(result)
    st.bar_chart(table.set_index("frame")[["detections"]])
    st.dataframe(table, hide_index=True)
    st.caption("Final-frame outgoing links are zero because no following frame is available. Divisions are tracker hypotheses, not confirmed biological events.")
    if not result.edges.empty:
        indexed = result.nodes.set_index("node_id")
        edges = result.edges[result.edges.source_id.isin(indexed.index) & result.edges.target_id.isin(indexed.index)]
        source, target = indexed.loc[edges.source_id], indexed.loc[edges.target_id]
        distance = np.linalg.norm(target[["z","y","x"]].to_numpy()-source[["z","y","x"]].to_numpy(),axis=1)
        steps = target.t.to_numpy()-source.t.to_numpy()
        motion = pd.DataFrame(dict(source_id=edges.source_id.to_numpy(), target_id=edges.target_id.to_numpy(),
            source_frame=source.t.to_numpy(), displacement_voxels=distance, voxels_per_frame=distance/steps))
        fig = go.Figure(go.Histogram(x=motion.voxels_per_frame, nbinsx=25))
        fig.update_layout(title="Predicted link movement", xaxis_title="3D voxel distance per frame (uncalibrated)",
            yaxis_title="Links",height=290,template="plotly_white",margin=dict(t=40,b=40))
        st.plotly_chart(fig,use_container_width=True,key=key+"_motion")
        st.download_button("Download measured links",motion.to_csv(index=False),file_name="predicted_link_measurements.csv",key=key+"_links")
    st.download_button("Download actual detected cells",result.nodes.to_csv(index=False),file_name="predicted_cells.csv",key=key+"_nodes")


def render_detection_review(result, volume):
    st.caption(result.dataset_name + " · " + str(result.stats.get("source", "Pipeline predictions")))
    st.caption("Hover for cell IDs and depth; zoom to inspect crowded regions. Switch to 3D to separate overlapping projected cells.")
    t = st.slider("Review frame",0,len(volume)-1,0,key="review_frame") if len(volume)>1 else 0
    nodes = result.nodes[result.nodes.t.eq(t)]
    cols = st.columns(3)
    cols[0].metric("Frame",t)
    cols[1].metric("Predicted centres",len(nodes))
    cols[2].metric("Volume depth",f"{volume.shape[1]} slices")
    view = st.radio("Image view",["Input vs model output","Whole volume","Single slice","3D tracks"],horizontal=True)
    if view == 'Input vs model output':
        raw = projection_figure(volume[t],nodes.iloc[:0])
        raw.update_layout(title='INPUT · raw microscopy (maximum projection)',height=500)
        output = projection_figure(volume[t],nodes,mask=getattr(result.frames[t],'projection_mask',None))
        output.update_layout(title='OUTPUT · predicted nucleus centres',height=500)
        first,second = st.columns(2)
        with first:
            st.plotly_chart(raw,use_container_width=True,key='raw_input_image')
        with second:
            st.plotly_chart(output,use_container_width=True,key='model_output_image')
        st.caption('Same image, contrast, crop and pixel coordinate system. Rings show predicted centres; colour indicates estimated Z depth. Cyan outlines, when available, are 2D nuclear segmentation, not 3D cell boundaries.')
        from app.reference_review import render_reference_agreement
        render_reference_agreement(result,volume,'input_output')
        return
    if view != "3D tracks":
        z = st.slider("Review Z slice",0,volume.shape[1]-1,volume.shape[1]//2) if view=="Single slice" else None
        st.plotly_chart(projection_figure(volume[t],nodes,z,mask=getattr(result.frames[t],'projection_mask',None)),use_container_width=True,key="real_detection_image")
        st.caption("Colour = Z depth, not confidence. Projection collapses depth and can hide overlaps. Markers are predicted centres, not cell boundaries. Coordinates are shown in pixels; physical calibration is not used in this display.")
    else:
        history = result.nodes[result.nodes.t.le(t)].set_index("node_id")
        fig = go.Figure()
        xyz = [[],[],[]]
        for edge in result.edges.itertuples():
            if edge.source_id in history.index and edge.target_id in history.index:
                for values,axis in zip(xyz,("x","y","z")):
                    values.extend([history.loc[edge.source_id,axis],history.loc[edge.target_id,axis],None])
        fig.add_trace(go.Scatter3d(x=xyz[0],y=xyz[1],z=xyz[2],mode="lines",line=dict(color="#8a9bb0",width=3),name="Classical links"))
        fig.add_trace(go.Scatter3d(x=nodes.x,y=nodes.y,z=nodes.z,mode="markers",text=nodes.node_id,
            marker=dict(size=5,color=nodes.z,colorscale="Turbo"),hovertemplate="Cell %{text}<br>(%{x}, %{y}, %{z})<extra></extra>",name="Current detections"))
        fig.update_layout(height=580,template="plotly_white",scene=dict(aspectmode="data",xaxis_title="X voxel",yaxis_title="Y voxel",zaxis_title="Z voxel"),margin=dict(l=0,r=0,b=0,t=30))
        st.plotly_chart(fig,use_container_width=True,key="real_detection_3d")
        st.caption("Drag to rotate. Axes are voxel indices, not calibrated physical anatomy. Lines show links through the selected frame.")
    with st.expander("Frame measurements and exports"):
        render_real_summary(result,key="detection_review")
