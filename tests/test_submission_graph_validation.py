import pandas as pd
import pytest
from biohub.submission import validate_submission


def test_submission_rejects_invalid_lineage_topology(tmp_path):
    (tmp_path/'sample.zarr').mkdir()
    columns=['dataset','row_type','node_id','t','z','y','x','source_id','target_id']
    rows=[['sample','node',1,0,1,1,1,-1,-1],
          ['sample','node',2,1,1,1,1,-1,-1],
          ['sample','edge',-1,-1,-1,-1,-1,1,2]]
    good=pd.DataFrame(rows,columns=columns)
    validate_submission(good,tmp_path)
    bad=good.copy();bad.loc[2,['source_id','target_id']]=[2,1]
    with pytest.raises(AssertionError,match='non-forward'):
        validate_submission(bad,tmp_path)
    bad=good.copy();bad.loc[1,'node_id']=1
    with pytest.raises(AssertionError,match='Duplicate'):
        validate_submission(bad,tmp_path)
    bad=good.copy();bad.loc[2,'row_type']='unknown'
    with pytest.raises(AssertionError,match='row_type'):
        validate_submission(bad,tmp_path)
