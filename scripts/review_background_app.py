"""Local slice-by-slice review UI. Never automatically labels image background."""
from pathlib import Path
import json
from datetime import datetime, timezone
import tempfile

import numpy as np
import streamlit as st
import tifffile
from biohub.background_review import image_hash, load_reviewed_background
from biohub.data import find_geff_path, load_geff_graph

ROOT=Path(__file__).resolve().parents[2]
st.set_page_config(page_title='Biohub background review',layout='wide')
st.title('Review background regions')
st.write('Inspect every Z slice covered by a proposed box. Dim or unmarked cells are not background. Leave uncertain regions unmarked.')
folder=ROOT/'reviews/background'
records=sorted(folder.glob('*.json'))
if not records:
    st.info('Export a real frame with prepare_background_review.py first.');st.stop()
path=Path(st.selectbox('Image record',[str(p) for p in records]))
record=json.loads(path.read_text())
frame=tifffile.imread(path.with_suffix('.tif'))
if image_hash(frame)!=record['image_sha256']:
    st.error('Image differs from review record.');st.stop()
key=str(path)
if st.session_state.get('record_key')!=key:
    st.session_state.record_key=key;st.session_state.boxes=[]
    st.session_state.seen=set()
z=st.slider('Z slice',0,frame.shape[0]-1,0)
st.session_state.seen.add(z)
left,right=st.columns([2,1])
with left:
    percentile=st.slider('Display upper percentile (contrast only)',90.0,100.0,99.8,.1)
    lo,hi=np.percentile(frame,[1,percentile])
    gray=(np.clip((frame[z].astype(float)-lo)/max(hi-lo,1),0,1)*255).astype(np.uint8)
    rgb=np.repeat(gray[:,:,None],3,axis=2)
    for low,high in st.session_state.boxes:
        if low[0]<=z<high[0]:
            y0,x0=low[1:];y1,x1=high[1:]
            rgb[y0:y1,x0]=[255,180,0];rgb[y0:y1,x1-1]=[255,180,0]
            rgb[y0,x0:x1]=[255,180,0];rgb[y1-1,x0:x1]=[255,180,0]
    st.image(rgb,caption=f'{record["sample"]}, frame {record["frame"]}, Z={z}. Orange boxes are proposals, not verified labels.',width=640)
    st.caption(f'Raw slice range: {frame[z].min()}–{frame[z].max()}; display range: {lo:.1f}–{hi:.1f}. Coordinates are full-resolution voxels.')
with right:
    st.subheader('Propose a box')
    st.caption('Bounds are inclusive at start, exclusive at end. X runs right; Y runs down.')
    low=[];high=[]
    for axis,size in zip('ZYX',frame.shape):
        a,b=st.columns(2)
        low.append(a.number_input(axis+' start',0,int(size)-1,0,key=axis+'lo'))
        high.append(b.number_input(axis+' end',1,int(size),1,key=axis+'hi'))
    if st.button('Add unreviewed box'):
        if any(b<=a for a,b in zip(low,high)):st.error('Each end must exceed its start.')
        else:st.session_state.boxes.append([low,high]);st.rerun()
    if st.button('Remove last box') and st.session_state.boxes:
        st.session_state.boxes.pop();st.rerun()
    st.json(st.session_state.boxes)
    reviewer=st.text_input('Reviewer name')
    reason=st.text_area('Evidence / review notes')
    confirmed=st.checkbox('I inspected the full 3D extent of every box and judge it cell-free.')
    st.caption('This records your judgment; it does not establish biological correctness automatically.')
    if st.button('Validate and save reviewed record'):
        try:
            required={i for low,high in st.session_state.boxes for i in range(low[0],high[0])}
            if not confirmed or not reviewer.strip() or not reason.strip():raise ValueError('Review confirmation, name and notes are required.')
            if not required.issubset(st.session_state.seen):raise ValueError('Visit every Z slice covered by the boxes before saving.')
            candidates=[ROOT/'data/biohub-stage1/train',Path.home()/'Desktop/biohub_data/train']
            geff=next((find_geff_path(p,record['sample']) for p in candidates if find_geff_path(p,record['sample'])),None)
            if geff is None:raise ValueError('Matching local GEFF required for annotation conflict checks.')
            nodes,_=load_geff_graph(geff)
            points=nodes[nodes.t==record['frame']][['z','y','x']].to_numpy(float)
            reviewed={**record,'status':'reviewed','reviewer':reviewer.strip(),'reason':reason.strip(),
                      'reviewed_at':datetime.now(timezone.utc).isoformat(),'background_boxes_zyx':st.session_state.boxes}
            with tempfile.TemporaryDirectory() as td:
                temp=Path(td)/'review.json';temp.write_text(json.dumps(reviewed))
                mask=load_reviewed_background(temp,frame,record['sample'],record['frame'],points)
                if not mask[::1,::4,::4].any():raise ValueError('No background voxels remain after trainer downsampling.')
            dest=ROOT/'reviews/verified_background';dest.mkdir(exist_ok=True)
            target=dest/path.name
            with target.open('x',encoding='utf-8') as f:json.dump(reviewed,f,indent=2)
            st.success(f'Saved {target}. This one reviewed frame does not unlock full-data training.')
        except (ValueError,FileExistsError,RuntimeError) as exc:st.error(str(exc))
