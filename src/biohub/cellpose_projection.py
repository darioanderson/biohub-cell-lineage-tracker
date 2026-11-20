"""Pretrained 2D Cellpose nucleus segmentation with intensity-derived depth.

This is not full 3D instance segmentation. Nuclei overlapping in XY may merge.
"""
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi


class CellposeProjectionDetector:
    def __init__(self,cfg):
        folder=Path(cfg.learned_detector_path).parent
        if not folder.is_dir() or not (folder/'nucleitorch_0').exists():
            raise FileNotFoundError('The pretrained Cellpose nuclei checkpoint is missing.')
        from cellpose import models
        import torch
        torch.set_num_threads(4)
        # Explicit local paths ensure inference never silently downloads a different model.
        self.model=models.CellposeModel(gpu=False,pretrained_model=str(folder/'nucleitorch_0'),diam_mean=17.)
        self.size_model=models.SizeModel(self.model,pretrained_size=str(folder/'size_nucleitorch_0.npy'))
        self.diameter=None
        self.xy_stride=2

    def detect(self,vol,cfg,prev_count=None):
        image=np.asarray(vol[:,::self.xy_stride,::self.xy_stride],dtype=np.float32)
        projection=image.max(axis=0)
        if self.diameter is None:
            self.diameter=float(np.asarray(self.size_model.eval(projection,channels=[0,0])[0]).item())
            if not np.isfinite(self.diameter) or self.diameter<=0:
                self.diameter=17.  # The pretrained nucleus model's documented mean size.
        masks,flows,_=self.model.eval(projection,channels=[0,0],diameter=self.diameter)
        ids=np.unique(masks);ids=ids[ids>0]
        points=[];scores=[];intensities=[]
        for identity in ids:
            region=masks==identity
            y,x=ndi.center_of_mass(region)
            profile=image[:,region].mean(axis=1)
            weights=np.maximum(profile-np.percentile(profile,20),0)
            z=float(np.dot(np.arange(len(profile)),weights)/weights.sum()) if weights.sum()>0 else float(np.argmax(profile))
            point=np.array([z,y*self.xy_stride,x*self.xy_stride])
            points.append(point)
            scores.append(float(np.asarray(flows[2])[region].mean()))
            index=tuple(np.clip(np.rint(point).astype(int),0,np.array(vol.shape)-1))
            intensities.append(float(vol[index]))
        self.last_projection_masks=masks
        return np.asarray(points,dtype=float).reshape(-1,3),np.asarray(scores),np.asarray(intensities)
