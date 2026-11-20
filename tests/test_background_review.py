import json
import numpy as np
import pytest
from biohub.background_review import review_template,load_reviewed_background


def test_review_gates_and_provenance(tmp_path):
    frame=np.zeros((4,8,8),np.uint16);p=tmp_path/'review.json'
    record=review_template(frame,'sample',0)
    def read():
        p.write_text(json.dumps(record))
        return load_reviewed_background(p,frame,'sample',0,[[2,5,5]])
    with pytest.raises(ValueError,match='reviewed'):read()
    record.update(status='reviewed',reviewer='test fixture',reviewed_at='2026-09-22',reason='synthetic test',background_boxes_zyx=[[[0,0,0],[1,2,2]]])
    mask=read();assert mask.sum()==4 and not mask[2,5,5]
    record['image_sha256']='wrong'
    with pytest.raises(ValueError,match='provenance'):read()


def test_conflicting_or_out_of_bounds_boxes_rejected(tmp_path):
    frame=np.zeros((4,8,8));record=review_template(frame,'s',0)
    record.update(status='reviewed',reviewer='fixture',reviewed_at='2026-09-22',reason='test')
    for box in [[[-1,0,0],[1,1,1]],[[0,0,0],[5,8,8]],[[2,5,5],[3,6,6]]]:
        record['background_boxes_zyx']=[box];p=tmp_path/'r.json';p.write_text(json.dumps(record))
        with pytest.raises(ValueError):load_reviewed_background(p,frame,'s',0,[[2,5,5]])
