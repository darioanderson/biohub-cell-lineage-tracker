"""Two uploaded microscopy movies, one fixed detector, no demo substitution."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.result_review import frame_measurements, projection_figure
from biohub.analysis import run_tracking_pipeline
from biohub.config import Config
from app.recording_metadata import known_recording, calibrated_config


def read_movie(raw):
    movie = np.load(io.BytesIO(raw), allow_pickle=False)
    if not isinstance(movie, np.ndarray) or movie.ndim != 4 or min(movie.shape) < 2:
        raise ValueError('Each file must be a microscopy movie shaped (time, Z, Y, X), with at least two frames and two voxels per spatial axis.')
    if movie.dtype.kind not in 'uif' or not np.isfinite(movie).all():
        raise ValueError('Movies must contain finite numeric image intensities.')
    if movie.nbytes > 256*1024*1024:
        raise ValueError('Please use a test clip smaller than 256 MiB when loaded.')
    return movie


def read_local_movie(path):
    file=Path(path).expanduser().resolve(strict=True)
    if file.suffix.lower()!='.npy':
        raise ValueError('Choose a .npy microscopy movie.')
    movie=np.load(file,mmap_mode='r',allow_pickle=False)
    if movie.ndim!=4 or min(movie.shape)<2 or movie.dtype.kind not in 'uif':
        raise ValueError('Expected a numeric (time, Z, Y, X) movie with at least two frames.')
    if movie.dtype.kind == 'f':
        for frame in movie:
            if not np.isfinite(frame).all():raise ValueError('Movie contains nonfinite intensities.')
    return movie


@st.cache_data(show_spinner=False)
def local_digest(path,size,mtime_ns):
    digest=hashlib.sha256()
    with open(path,'rb') as source:
        for chunk in iter(lambda:source.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def known_validation_failure(pair):
    report=Path(__file__).resolve().parents[1]/'results/large_pair_review_20261119/native_grid_failure.json'
    if not report.exists():return None
    failure=json.loads(report.read_text())
    checkpoint=pair['cfg'].learned_detector_path
    if not checkpoint.exists():return None
    same_failed_control=pair['hashes'][0]==failure.get('control_sha256')
    small_manifest=Path(__file__).resolve().parents[1]/'data/real_control_variant_under200mb/PROVENANCE.json'
    if small_manifest.exists():
        clip=json.loads(small_manifest.read_text()).get('recordings',{}).get('control',{})
        same_failed_control = same_failed_control or (pair['hashes'][0]==clip.get('sha256') and clip.get('source_sha256')==failure.get('control_sha256'))
    if (same_failed_control and
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()==failure.get('checkpoint_sha256')):
        return failure
    return None


def compare_counts(control, variant):
    return frame_measurements(control).merge(frame_measurements(variant),on='frame',suffixes=('_control','_variant'))


def recording_config(cfg,digest):
    return calibrated_config(cfg, known_recording(digest))


def pair_manifest(pair, aligned=False):
    return dict(recordings=[dict(slot=slot, label=label, file=name, sha256=digest, shape=list(movie.shape),
                                effective_config=result.stats.get('effective_config'),
                                time_interval_verified=result.stats.get('time_interval_verified',False),
                                spatial_calibration_verified=result.stats.get('spatial_calibration_verified',False),
                                run_id=result.stats.get('run_id'))
        for slot,label,name,digest,movie,result in zip(('control','variant'),pair.get('labels',['Control','Variant']),pair['filenames'],pair['hashes'],pair['volumes'],pair['results'])],
        checkpoint_sha256=pair.get('checkpoint_sha256'),detector_backend=pair['cfg'].detector_backend,
        linker_backend=pair['cfg'].linker_backend,
        stage_and_sampling_alignment='user confirmed' if aligned else 'not verified',
        biological_identity_matching=False,genetic_diagnosis=False,
        comparison_limitation='Descriptive recording comparison. No validated genetic-defect inference.')


def reset_pair_alignment(state, hashes):
    if state.get('alignment_input_hashes') != hashes:
        state['real_pair_aligned'] = False
        state['alignment_input_hashes'] = hashes


SESSION_KEYS = ('volume_4d','volume_source','selected_entry','dataset_name','result','cfg','curation_audit','preview_frames')


def activate_recording(state, pair, index):
    """Bind all ordinary pages to one recording, saving each recording's review state."""
    previous = state.get('pair_active_index')
    if previous is not None and state.get('pair_binding') == pair['hashes']:
        pair['reviews'][previous] = {key:state.get(key) for key in SESSION_KEYS}
    if previous == index and state.get('pair_binding') == pair['hashes']:
        return
    if 'single_recording_backup' not in state:
        state['single_recording_backup'] = {key:state.get(key) for key in SESSION_KEYS}
    review = pair['reviews'].get(index)
    if review is None:
        review = dict(volume_4d=pair['volumes'][index], volume_source='upload_npy', selected_entry=None,
                      dataset_name=pair.get('labels',['Control','Variant'])[index], result=pair['results'][index], cfg=pair.get('configs',[pair['cfg'],pair['cfg']])[index].copy_with(),curation_audit=None,
                      preview_frames=len(pair['volumes'][index]))
    state.update(review)
    state['pair_active_index'] = index
    state['pair_binding'] = pair['hashes'][:]
    for key in list(state):
        if key in ('overlay_gif','det_t','det_z','vol_t','vol_z','review_frame','lineage_focus_node','Review Z slice') or key.startswith(('curate_','lineage_role_jump_')):
            state.pop(key,None)


def restore_single_recording(state):
    if 'single_recording_backup' in state:
        pair = state.get('real_pair_output')
        index = state.get('pair_active_index')
        if pair and index is not None and state.get('pair_binding') == pair['hashes']:
            pair['reviews'][index] = {key:state.get(key) for key in SESSION_KEYS}
        state.update(state.pop('single_recording_backup'))
    state.pop('pair_active_index',None)
    state.pop('pair_binding',None)


def render_pair_sidebar():
    """Load each file once; selecting a recording never launches inference."""
    st.header('Control and variant')
    st.caption('Microscopy movies · .npy (time, Z, Y, X)')
    engine=st.radio('Model',['Cellpose nuclei — experimental projection review','Our Kaggle 3D detector','Biohub original pipeline — classical detection'],key='pair_model_engine')
    pretrained=engine.startswith('Cellpose')
    biohub=engine.startswith('Biohub')
    if pretrained:
        st.caption('Pretrained third-party model. Uses a 2D projection and estimates depth; overlaps and mitosis can cause misses. Not a validated genetic-defect detector.')
    model_file=Path(__file__).resolve().parents[1]/('models/cellpose_nuclei/nucleitorch_0' if pretrained else 'results/kaggle_ce_cshaper_v2/detector.pt')
    backend='peaks' if biohub else ('cellpose_projection' if pretrained else 'cshaper')
    if biohub:
        st.caption('Original Biohub peak detector and classical tracking. This option does not use a trained neural detector.')
    input_mode=st.radio('Recording source',['Upload files','Files on this PC'],key='paired_input_mode')
    if input_mode=='Files on this PC':
        zebrafish=Path(__file__).resolve().parents[1]/'data/local_biohub_pair'
        if (zebrafish/'PROVENANCE.json').exists() and st.button('Use local zebrafish recordings — all 100 frames'):
            st.session_state.local_control_path=str(zebrafish/'44b6_0113de3b_all100.npy')
            st.session_state.local_variant_path=str(zebrafish/'44b6_0b24845f_all100.npy')
            st.session_state.real_pair_aligned=False
        divisions=Path(__file__).resolve().parents[1]/'data/real_division_clips'
        if (divisions/'PROVENANCE.json').exists() and st.button('Use real division recordings (under 200 MB)'):
            st.session_state.local_control_path=str(divisions/'CONTROL_WT_Emb8_divisions_5frames.npy')
            st.session_state.local_variant_path=str(divisions/'VARIANT_wee-1.1_MT2_divisions_5frames.npy')
            st.session_state.real_pair_aligned=False
        small=Path(__file__).resolve().parents[1]/'data/real_control_variant_under200mb'
        if (small/'PROVENANCE.json').exists() and st.button('Use small real recordings (under 200 MB)'):
            st.session_state.local_control_path=str(small/'CONTROL_WT_Emb8_5frames.npy')
            st.session_state.local_variant_path=str(small/'VARIANT_wee-1.1_MT2_5frames.npy')
            st.session_state.real_pair_aligned=False
        downloaded=Path(__file__).resolve().parents[1]/'data/real_control_variant_large'
        if (downloaded/'PROVENANCE.json').exists() and st.button('Use downloaded 30-frame recordings'):
            st.session_state.local_control_path=str(downloaded/'CONTROL_WT_Emb8_t001_030.npy')
            st.session_state.local_variant_path=str(downloaded/'VARIANT_wee-1.1_MT2_t001_030.npy')
            st.session_state.real_pair_aligned=False
        paths=[st.text_input('Control file path',key='local_control_path'),st.text_input('Variant file path',key='local_variant_path')]
        local_files=[]
        for value in paths:
            try:
                path=Path(value.strip().strip('"')).expanduser().resolve(strict=True) if value.strip() else None
                local_files.append(path if path is not None and path.is_file() and path.suffix.lower()=='.npy' else None)
            except OSError:local_files.append(None)
        complete=all(path is not None for path in local_files)
        digests=[local_digest(str(path),path.stat().st_size,path.stat().st_mtime_ns) for path in local_files] if complete else None
        filenames=[path.name if path else '' for path in local_files]
        raw_inputs=None
    else:
        local_files=None
        complete,digests,filenames,raw_inputs=render_upload_inputs()
    reset_pair_alignment(st.session_state,digests)
    pair = st.session_state.get('real_pair_output')
    zebrafish_manifest=Path(__file__).resolve().parents[1]/'data/local_biohub_pair/PROVENANCE.json'
    zebrafish_hashes=set()
    if zebrafish_manifest.exists():
        zebrafish_hashes={r['sha256'] for r in json.loads(zebrafish_manifest.read_text())['recordings'].values()}
    is_zebrafish=bool(digests and any(d in zebrafish_hashes for d in digests))
    if is_zebrafish:
        st.warning('These zebrafish recordings have unverified genotypes and share embryo prefix 44b6. Control/Variant are input slots only; this is not a verified genetic comparison. Use Biohub original pipeline for these inputs.')
    matching = pair is not None and digests == pair['hashes'] and pair['cfg'].detector_backend==backend
    if st.button('Load recordings',disabled=not complete):
        try:
            if is_zebrafish and not biohub:
                raise ValueError('Select Biohub original pipeline — classical detection for these zebrafish inputs. The C. elegans detectors are not validated for them.')
            if digests[0] == digests[1]:
                raise ValueError('Please choose two different recordings.')
            volumes = [read_local_movie(path) for path in local_files] if local_files else [read_movie(item) for item in raw_inputs]
            if volumes[0].shape == volumes[1].shape and all(np.array_equal(a,b) for a,b in zip(*volumes)):
                raise ValueError('Both files contain the same movie. Please choose distinct recordings.')
            cfg = Config(detector_backend=backend, learned_detector_path=model_file,voxel_size_zyx_um=(1.,1.,1.))
            pair = dict(volumes=volumes,results=[None,None],reviews={},hashes=digests,
                        filenames=filenames,cfg=cfg,configs=[recording_config(cfg,digest) for digest in digests])
            pair['metadata']=[known_recording(digest) for digest in digests]
            pair['labels']=['Recording A','Recording B'] if is_zebrafish else ['Control','Variant']
            st.session_state.real_pair_output = pair
            st.session_state.pop('pair_binding',None)
            matching = True
        except (OSError,ValueError) as exc:
            st.error(str(exc))
    if not matching:
        if st.session_state.get('pair_active_index') is not None:
            restore_single_recording(st.session_state)
        if 'single_recording_backup' not in st.session_state:
            st.session_state.single_recording_backup = {key:st.session_state.get(key) for key in SESSION_KEYS}
        st.session_state.update(volume_4d=None,result=None,selected_entry=None,dataset_name='')
        st.info('Choose both files and press Load recordings.')
        return
    for group,movie in zip(('Control','Variant'),pair['volumes']):
        st.caption(f'{group}: {len(movie)} frames · {movie.nbytes/1024**2:.1f} MiB · {tuple(movie.shape[1:])} voxels')
    saved=Path(__file__).resolve().parents[1]/'results/submission_biohub_20261119'
    if biohub and is_zebrafish and (saved/'status.json').exists():
        if json.loads((saved/'status.json').read_text()).get('state')=='complete':
            if st.button('Load saved zebrafish analysis — no rerun'):
                try:
                    from app.saved_runs import restore_biohub_run
                    restore_biohub_run(pair,saved)
                    st.session_state.pop('pair_binding',None)
                    st.success('Saved predictions matched to these inputs and settings.')
                except (ValueError,OSError,KeyError) as error:
                    st.error(str(error))
    failure=known_validation_failure(pair)
    if failure:
        st.warning('This model failed the published reference count check on this Control recording (66 predictions versus 2 reference nuclei in its first frame). Full comparison is disabled to avoid misleading results.')
    if st.button('Load model and analyze both',type='primary',disabled=failure is not None):
        pair.update(results=[None,None],reviews={})
        st.session_state.pop('pair_binding',None)
        try:
            if not biohub and not pair['cfg'].learned_detector_path.exists():
                raise ValueError('Trained detector checkpoint is missing.')
            checkpoint_hash=None if biohub else hashlib.sha256(pair['cfg'].learned_detector_path.read_bytes()).hexdigest()
            results = []
            bar=st.progress(0.,text='Starting analysis')
            for index,(group,movie,record_cfg,info) in enumerate(zip(pair['labels'],pair['volumes'],pair['configs'],pair['metadata'])):
                import time
                started=time.perf_counter()
                with st.spinner(f'Analyzing all {len(movie)} {group.lower()} frames…'):
                    result = run_tracking_pipeline(movie,record_cfg,dataset_name=group,
                        progress_callback=lambda fraction,text:bar.progress((index+fraction)/2,text=text))
                result.stats.update(source=('Original Biohub classical peak detector and classical linking' if biohub else ('Pretrained Cellpose nuclei projection + intensity-derived depth; classical linking' if pretrained else 'Kaggle-trained CShaper detector; classical linking')),
                                    spacing_note='Physical calibration not supplied; voxel/frame units only.')
                if pretrained:
                    result.stats['limitations']='Experimental projection detections: overlaps/mitosis can be missed; depth is estimated, not 3D segmentation.'
                    result.stats['score_kind']='Cellpose raw foreground logit; not calibrated confidence'
                if record_cfg.voxel_size_zyx_um!=(1.,1.,1.):
                    result.stats['spacing_note']='Tracking uses source voxel spacing; acquisition timing and developmental stage remain unaligned.'
                result.stats['time_interval_verified']=info['time_verified']
                result.stats['spatial_calibration_verified']=info['spatial_verified']
                result.stats['input_sha256']=pair['hashes'][index]
                result.stats['input_paths']=[str(local_files[index])] if local_files else []
                result.stats['checkpoint_sha256']=checkpoint_hash
                result.stats['runtime_s']=time.perf_counter()-started
                if is_zebrafish:
                    result.stats['comparison_limitation']='Genotypes unverified; same embryo prefix. No genetic-defect conclusions.'
                results.append(result)
            if not biohub and hashlib.sha256(pair['cfg'].learned_detector_path.read_bytes()).hexdigest()!=checkpoint_hash:
                raise ValueError('Checkpoint changed during inference. Both recordings must use identical weights.')
            pair.update(results=results,reviews={},checkpoint_sha256=checkpoint_hash)
            from app.biohub_pair_reference import attach_pair_references
            with st.spinner('Checking available sparse lineage annotations…'):
                attach_pair_references(pair)
            save_dir=Path(__file__).resolve().parents[1]/'results/ui_runs'/results[0].stats['run_id']
            save_dir.mkdir(parents=True,exist_ok=False)
            for slot,result in zip(('control','variant'),results):
                result.nodes.to_csv(save_dir/(slot+'_nodes.csv'),index=False)
                result.edges.to_csv(save_dir/(slot+'_edges.csv'),index=False)
                (save_dir/(slot+'_stats.json')).write_text(json.dumps(result.stats,indent=2,default=str))
            (save_dir/'paired_run_provenance.json').write_text(json.dumps(pair_manifest(pair),indent=2,default=str))
            (save_dir/'sparse_reference_checks.json').write_text(json.dumps(pair.get('reference_checks',[]),indent=2,default=str))
            pair['saved_run']=str(save_dir)
            st.session_state.pop('pair_binding',None)
            st.success('Both analyzed. Select which recording to view below.')
        except (OSError,ValueError,RuntimeError) as exc:
            st.error(str(exc))
    labels=pair.get('labels',['Control','Variant'])
    if st.session_state.get('pair_view') not in labels:st.session_state.pop('pair_view',None)
    selected = st.radio('Show throughout the app',labels,key='pair_view')
    index = labels.index(selected)
    activate_recording(st.session_state,pair,index)
    st.caption(pair['filenames'][index])
    compatible=(not any(info.get('manifest') for info in pair.get('metadata',[])))
    if not compatible:st.session_state.real_pair_aligned=False
    st.checkbox('Same starting developmental stage, frame interval and voxel spacing',key='real_pair_aligned',disabled=not compatible)
    if not compatible:st.caption('Alignment is unavailable for the supplied exploratory recordings; it has not been established by their source metadata.')
    st.caption('Use Volume, Detection, Lineage and Exports for the selected recording. Compare Lineages shows the comparison.')


def render_upload_inputs():
    original = st.session_state.get('single_recording_backup', st.session_state)
    previous_movie = original.get('volume_4d')
    reuse = st.checkbox('Use previously loaded recording as Control', value=previous_movie is not None,
                        disabled=previous_movie is None, key='reuse_loaded_control')
    if reuse and previous_movie is not None:
        # Encode only once per loaded array, rather than copying a large movie on every rerun.
        cached = st.session_state.get('reused_control_input')
        if cached is None or cached['volume'] is not previous_movie:
            if previous_movie.nbytes>256*1024**2:
                st.info('For this large recording, select Files on this PC and enter both .npy paths. This avoids an extra in-memory upload copy.')
                return False,None,['',''],[None,None]
            buffer = io.BytesIO()
            np.save(buffer, previous_movie, allow_pickle=False)
            cached = dict(volume=previous_movie, raw=buffer.getvalue(), name=(original.get('dataset_name') or 'previous_recording')+'.npy')
            st.session_state.reused_control_input = cached
        control_bytes, control_name = cached['raw'], cached['name']
        st.caption('Control: '+control_name)
    else:
        control_file = st.file_uploader('Control recording',type='npy',key='real_control_movie')
        control_bytes = control_file.getvalue() if control_file is not None else None
        control_name = control_file.name if control_file is not None else ''
    variant_file = st.file_uploader('Variant recording',type='npy',key='real_variant_movie')
    raw_inputs = [control_bytes, variant_file.getvalue() if variant_file is not None else None]
    filenames = [control_name, variant_file.name if variant_file is not None else '']
    complete = all(item is not None for item in raw_inputs)
    digests = [hashlib.sha256(item).hexdigest() for item in raw_inputs] if complete else None
    return complete,digests,filenames,raw_inputs



def paired_movies_page():
    st.subheader('Compare Lineages')
    pair = st.session_state.get('real_pair_output')
    if st.session_state.get('analysis_workspace') != 'Control and variant' or st.session_state.get('pair_binding') != (pair or {}).get('hashes'):
        st.info('Choose Control and variant in the sidebar, load both recordings, then press Load model and analyze both.')
        return
    if pair is None or any(result is None for result in pair['results']):
        failure=known_validation_failure(pair) if pair else None
        if failure:
            st.error('Model validation failed on this recording. These predictions cannot support a control-versus-variant lineage conclusion.')
            fig=go.Figure(go.Bar(x=['Published reference','Model prediction'],y=[2,66],marker_color=['#3567a5','#df8b35']))
            fig.update_layout(title='Actual first-frame count validation',yaxis_title='Nucleus centres',template='plotly_white',height=300)
            st.plotly_chart(fig,use_container_width=True,key='failed_pair_count_check')
            st.caption('This is a real failed check. A fixed physical-scale pilot also over-detected (88 predictions); it was not adopted. No full-pair performance score is claimed.')
            return
        st.info('Load the model and analyze both recordings to see their comparison.')
        return
    aligned=bool(st.session_state.get('real_pair_aligned'))
    control,variant = pair['results']
    comparison = compare_counts(control,variant)
    st.caption('Analyzed: '+pair['filenames'][0]+' versus '+pair['filenames'][1])
    st.caption('Graphs use the original outputs of the same model run. Manual review edits are retained separately for each recording.')
    from app.pair_dashboard import render_dashboard
    render_dashboard(pair,aligned)
    with st.expander('Measurements and downloads'):
        st.dataframe(comparison,hide_index=True)
        st.download_button('Download comparison',comparison.to_csv(index=False),file_name='real_control_variant_comparison.csv')
        combined = pd.concat([result.nodes.assign(recording=group) for result,group in zip(pair['results'],('control','variant'))],ignore_index=True)
        st.download_button('Download predicted cells',combined.to_csv(index=False),file_name='real_pair_predictions.csv')
        manifest = pair_manifest(pair,aligned)
        st.download_button('Download run details',json.dumps(manifest,indent=2),file_name='paired_run_provenance.json',mime='application/json')
