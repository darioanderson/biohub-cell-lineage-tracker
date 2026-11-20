import numpy as np
import pytest
from biohub.config import Config
from biohub.cellpose_projection import CellposeProjectionDetector


class StubModel:
    def __init__(self,mask):self.mask=mask
    def eval(self,*args,**kwargs):
        return self.mask,[None,None,np.full(self.mask.shape,4.)],None


def detector(mask):
    instance=CellposeProjectionDetector.__new__(CellposeProjectionDetector)
    instance.model=StubModel(mask)
    instance.diameter=17.
    instance.xy_stride=2
    return instance


def test_projection_coordinates_return_to_native_grid_and_depth():
    mask=np.zeros((4,4),int);mask[1:3,1:3]=1
    volume=np.zeros((6,8,8),np.uint8);volume[3]=10
    points,scores,intensities=detector(mask).detect(volume,Config())
    np.testing.assert_allclose(points,[[3,3,3]])
    assert scores.tolist()==[4.] and intensities.tolist()==[10.]


def test_empty_masks_do_not_invent_cells():
    points,scores,intensities=detector(np.zeros((4,4),int)).detect(np.zeros((6,8,8)),Config())
    assert points.shape==(0,3) and scores.size==0 and intensities.size==0


def test_missing_checkpoint_reports_path_error_before_optional_import(tmp_path):
    with pytest.raises(FileNotFoundError):
        CellposeProjectionDetector(Config(learned_detector_path=tmp_path/'missing'/'weights'))
