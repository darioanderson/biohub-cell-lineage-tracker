"""Measure solver changes against a preserved reference on a connected chain."""
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from biohub.config import Config
import biohub.tracking as current
from scipy.sparse import issparse


def main():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('reference',root/'references/joint_solver_before_optimization/tracking.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    rng=np.random.default_rng(42);n=160
    prev=np.column_stack((np.arange(n)*1.5,np.zeros(n),np.zeros(n)))
    curr=prev+rng.normal(0,.02,prev.shape)
    cfg=Config(voxel_size_zyx_um=(1,1,1),linker_backend='classical_joint',max_link_dist_um=2.,
        div_parent_dist_um=2.,div_sister_dist_um=2.,div_midpoint_dist_um=2.,use_rich_linking=False)
    report={'cells_per_frame':n,'data':'seeded synthetic connected chain; not an accuracy benchmark','arms':{}}
    outputs=[]
    # Warm up HiGHS once so import/startup cost is not attributed to the first arm.
    current._joint_hypothesis_assignments(prev[:2],curr[:2],cfg,None,None,None,None,None)
    for label,module in [('before',old),('optimized',current)]:
        native=module.milp;sizes=[]
        def record(**kwargs):
            a=kwargs['constraints'].A
            sizes.append(int(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes) if issparse(a) else a.nbytes)
            return native(**kwargs)
        module.milp=record
        timings=[]
        for _ in range(3):
            start=time.perf_counter();result=module._joint_hypothesis_assignments(prev,curr,cfg,None,None,None,None,None)
            timings.append(time.perf_counter()-start)
        module.milp=native;outputs.append(result)
        report['arms'][label]={'seconds':timings,'median_seconds':float(np.median(timings)),
                               'constraint_matrix_bytes_max':max(sizes),'edges':len(result)}
    assert outputs[0]==outputs[1]
    report['identical_ordered_edges']=True
    report['speedup']=report['arms']['before']['median_seconds']/report['arms']['optimized']['median_seconds']
    path=root/'results/joint_solver_optimization.json';path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
