"""Descriptive review of two inferred recordings, with explicit alignment limits."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import json
from app.result_review import frame_measurements,projection_figure

COLORS=['#3567a5','#df8b35']
GROUPS=['Control','Variant']


def continuity_table(result):
    nodes=result.nodes
    incoming=set(result.edges.target_id) if len(result.edges) else set()
    outgoing=set(result.edges.source_id) if len(result.edges) else set()
    records=[]
    for t in range(len(result.frames)):
        ids=set(nodes[nodes.t.eq(t)].node_id)
        records.append(dict(frame=t,starts=len(ids-incoming) if t>0 else np.nan,
                            endings=len(ids-outgoing) if t<len(result.frames)-1 else np.nan))
    return pd.DataFrame(records)


def reference_counts_for_pair(pair):
    root=Path(__file__).resolve().parents[1]
    records=[]
    for index,(group,reference_file) in enumerate([('control','Emb8_CellTracing.csv'),('variant','MT2_CellTracing.csv')]):
        verified=False
        source_frames=None
        for folder in ['real_division_clips','real_control_variant_under200mb','real_control_variant_large']:
            manifest=root/'data'/folder/'PROVENANCE.json'
            if manifest.exists():
                data=json.loads(manifest.read_text());info=data.get('recordings',data).get(group,{})
                if info.get('sha256')==pair.get('hashes',[None,None])[index]:
                    verified=True
                    source_frames=info.get('source_frames')
        path=root/'data/real_control_variant_large/source'/reference_file
        if not verified or not path.exists():continue
        reference=pd.read_csv(path).groupby('time').size()
        predicted=frame_measurements(pair['results'][index]).set_index('frame').detections
        for t,value in predicted.items():
            source_t=source_frames[t] if source_frames else t+1
            if source_t in reference.index:
                records.append(dict(recording=group,frame=t,source_frame=source_t,predicted=int(value),published_reference=int(reference.loc[source_t])))
    return pd.DataFrame(records)


def render_dashboard(pair,aligned):
    GROUPS=pair.get('labels',['Control','Variant'])
    results=pair['results']
    tables=[frame_measurements(r) for r in results]
    for column,group,result,movie,name in zip(st.columns(2),GROUPS,results,pair['volumes'],pair['filenames']):
        with column:
            st.markdown('**'+group+'**')
            st.caption(name)
            st.metric('Analyzed frames',len(result.frames))
            st.caption(f'{result.n_nodes:,} predicted observations · {result.n_edges:,} links · {result.n_divisions} division proposals')
    quality=reference_counts_for_pair(pair)
    fig=go.Figure()
    for group,color,table in zip(GROUPS,COLORS,tables):
        fig.add_trace(go.Scatter(x=table.frame,y=table.detections,name=group,mode='lines+markers',line=dict(color=color,width=3)))
    fig.update_layout(title='Predicted cell counts',xaxis_title='Aligned frame' if aligned else 'Frame index within each recording (not aligned biological time)',
                      yaxis_title='Detected centres',template='plotly_white',height=350,legend=dict(orientation='h'),hovermode='x unified')
    st.plotly_chart(fig,use_container_width=True,key='pair_population')
    tabs=st.tabs(['Image comparison','Division proposals','Track continuity','Track duration','Matched-frame differences','Family trees','Reference check'])
    with tabs[0]:
        max_t=min(len(r.frames) for r in results)-1
        frame=st.slider('Review frame in both recordings',0,max_t,0,key='pair_review_frame') if max_t else 0
        show_raw=st.checkbox('Show raw images without prediction markers',key='pair_raw_images')
        st.caption('Same frame number does not imply the same developmental stage. Hover to identify predicted cells; colours show depth, not genotype or confidence.')
        for column,group,color,result,movie in zip(st.columns(2),GROUPS,COLORS,results,pair['volumes']):
            with column:
                nodes=result.nodes[result.nodes.t.eq(frame)]
                fig=projection_figure(movie[frame],nodes.iloc[:0] if show_raw else nodes,
                    mask=None if show_raw else getattr(result.frames[frame],'projection_mask',None))
                fig.update_layout(title=f'{group} · frame {frame} · {len(nodes)} predictions',height=470)
                st.plotly_chart(fig,use_container_width=True,key='pair_image_'+group)
    with tabs[1]:
        fig=go.Figure()
        for group,color,table in zip(GROUPS,COLORS,tables):
            fig.add_trace(go.Bar(x=table.frame.iloc[:-1],y=table.proposed_divisions.iloc[:-1],name=group,marker_color=color))
        fig.update_layout(title='Unverified binary division proposals',barmode='group',xaxis_title='Source frame within recording',yaxis_title='Proposals',template='plotly_white',height=300)
        st.plotly_chart(fig,use_container_width=True,key='pair_divisions')
        st.caption('Review these in the Lineage page. False detections and incorrect links can look like divisions.')
    with tabs[2]:
        fig=go.Figure()
        for group,color,result in zip(GROUPS,COLORS,results):
            table=continuity_table(result)
            for field,dash in [('starts','solid'),('endings','dash')]:
                fig.add_trace(go.Scatter(x=table.frame,y=table[field],name=group+' '+field,line=dict(color=color,dash=dash)))
        fig.update_layout(title='Track starts and endings inside the recording',xaxis_title='Frame within recording',yaxis_title='Observations without incoming/outgoing links',template='plotly_white',height=320)
        st.plotly_chart(fig,use_container_width=True,key='pair_continuity')
        st.caption('Initial starts and final endings are censored. Internal endings can be tracking errors or boundary exits; they do not establish cell death.')
    with tabs[3]:
        fig=go.Figure()
        for group,color,result in zip(GROUPS,COLORS,results):
            if 'track_id' not in result.nodes or result.nodes.empty:continue
            span=result.nodes.groupby('track_id').t.agg(['min','max'])
            fig.add_trace(go.Box(y=span['max']-span['min']+1,name=group,marker_color=color,boxpoints='all',jitter=.3))
        fig.update_layout(title='Observed track duration',yaxis_title='Frames spanned (not cell-cycle duration)',template='plotly_white',height=320)
        st.plotly_chart(fig,use_container_width=True,key='pair_durations')
        st.caption('Tracks touching a recording boundary are truncated. Numerical track IDs do not identify the same biological cell across embryos.')
    with tabs[4]:
        if not aligned:
            st.info('Paired differences require verified common developmental stage, frame interval and voxel spacing. Keep alignment unchecked for the supplied exploratory clips.')
        else:
            comparison=tables[0].merge(tables[1],on='frame',suffixes=('_control','_variant'))
            delta=comparison.detections_variant-comparison.detections_control
            fig=go.Figure(go.Bar(x=comparison.frame,y=delta,marker_color=np.where(delta>=0,COLORS[1],COLORS[0])))
            fig.update_layout(title='Variant − control detected centres',xaxis_title='Aligned frame',yaxis_title='Count difference',template='plotly_white',height=320)
            st.plotly_chart(fig,use_container_width=True,key='pair_deltas')
            st.caption('User-confirmed alignment; this is not a significance test and does not establish a genetic cause.')
    with tabs[5]:
        from biohub.visualization import plot_node_family_tree
        import matplotlib.pyplot as plt
        st.caption('Choose a predicted cell in each embryo to inspect its parents and daughters. The app does not establish that these are homologous biological cells.')
        for column,group,result in zip(st.columns(2),GROUPS,results):
            with column:
                ids=result.nodes.node_id.astype(int).tolist()
                if not ids:
                    st.info(group+': no predicted cells.')
                    continue
                selected=st.selectbox(group+' cell',ids,key='pair_family_'+group)
                fig=plot_node_family_tree(result,int(selected))
                st.pyplot(fig)
                plt.close(fig)
    with tabs[6]:
        for label,check in zip(GROUPS,pair.get('reference_checks',[])):
            if not check:continue
            st.markdown('**'+label+' · sparse Biohub reference**')
            if 'error' in check:
                st.warning('Reference check failed: '+check['error'])
            else:
                st.json(check['metrics'])
                recall=check['metrics'].get('annotated_detection_recall')
                if recall is not None and recall<.5:
                    st.warning(f'Only {recall:.1%} of annotated nuclei were recovered. These predictions are inadequate for a reliable lineage comparison.')
                if check['metrics'].get('division_tp',0)+check['metrics'].get('division_fn',0)==0:
                    st.caption('No annotated divisions are evaluable here. The displayed division proposals do not establish division accuracy.')
                st.caption('Recall covers annotated cells only. Edge/division precision is conditional on matched sparse endpoints, not full tracking accuracy. Unknown predictions are not false positives.')
        reference=quality
        if reference.empty:
            if not any(pair.get('reference_checks',[])):
                st.info('No verified reference tracing is attached to these file hashes. Visual inspection is not an accuracy score.')
        else:
            fig=go.Figure()
            for group,color in zip(['control','variant'],COLORS):
                table=reference[reference.recording.eq(group)]
                for column,dash in [('predicted','solid'),('published_reference','dash')]:
                    fig.add_trace(go.Scatter(x=table.frame,y=table[column],name=group+' '+column,line=dict(color=color,dash=dash),mode='lines+markers'))
            fig.update_layout(title='Actual predictions versus published tracing counts',xaxis_title='Frame within each recording',yaxis_title='Nuclei',template='plotly_white',height=330)
            st.plotly_chart(fig,use_container_width=True,key='pair_reference_quality')
            st.dataframe(reference,hide_index=True)
            mismatches=int(reference.predicted.ne(reference.published_reference).sum())
            if mismatches:st.warning(f'{mismatches} frame counts disagree with the published reference. Review these errors before interpreting a lineage difference.')
            st.caption('Equal counts do not prove correct locations or links. This is a descriptive count check on named test clips, not independent genetic validation.')
    st.caption('These graphs summarize model predictions, not every biological difference. The same detector and classical tracker process both recordings. Individual biological cell correspondence and genetic diagnosis are not validated.')
