"""Explicit reviewed background regions; absence of annotation is never background."""
import hashlib
import json
from pathlib import Path
import numpy as np


def image_hash(frame):
    a=np.ascontiguousarray(frame)
    h=hashlib.sha256(str((a.shape,a.dtype.str)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def review_template(frame, sample, time):
    return {'schema_version':1,'sample':sample,'frame':int(time),
            'image_sha256':image_hash(frame),'shape_zyx':list(frame.shape),
            'status':'unreviewed','reviewer':'','reviewed_at':'','reason':'',
            'background_boxes_zyx':[]}


def load_reviewed_background(path, frame, sample, time, annotated):
    record=json.loads(Path(path).read_text(encoding='utf-8'))
    if record.get('schema_version')!=1 or record.get('status')!='reviewed':
        raise ValueError('Background record must be explicitly reviewed, schema 1')
    for field in ['reviewer','reviewed_at','reason']:
        if not str(record.get(field,'')).strip():raise ValueError(f'Missing review {field}')
    if record.get('sample')!=sample or record.get('frame')!=int(time):
        raise ValueError('Review sample/frame mismatch')
    if record.get('shape_zyx')!=list(frame.shape) or record.get('image_sha256')!=image_hash(frame):
        raise ValueError('Review image provenance mismatch')
    mask=np.zeros(frame.shape,dtype=bool)
    for box in record.get('background_boxes_zyx',[]):
        bounds=np.asarray(box)
        if bounds.shape!=(2,3) or not np.issubdtype(bounds.dtype,np.integer):
            raise ValueError('Boxes require integer [[z0,y0,x0],[z1,y1,x1]]')
        lo,hi=bounds
        if np.any(lo<0) or np.any(hi>frame.shape) or np.any(hi<=lo):
            raise ValueError('Background box outside image or empty')
        mask[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))]=True
    if not mask.any():raise ValueError('No reviewed background regions')
    points=np.asarray(annotated).reshape(-1,3)
    if len(points):
        if not np.isfinite(points).all() or np.any(points<0) or np.any(points>=frame.shape):
            raise ValueError('Invalid annotated coordinates')
        idx=np.clip(np.rint(points).astype(int),0,np.array(frame.shape)-1)
        if mask[tuple(idx.T)].any():raise ValueError('Background overlaps annotated center')
    return mask
