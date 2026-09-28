import sys,os,copy
from pathlib import Path
import pandas as pd,numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from common.manuscript_cohorts import split_center_a_indices,split_metadata,require_split_design
os.environ['MANUSCRIPT_SPLIT_SEED']='20260902'
a=pd.DataFrame({'record_id':[f'A{i:04}' for i in range(969)],'y_SCLC':np.arange(969)%5==0})
split_center_a_indices(a)
saved={'split_design':split_metadata()}
require_split_design(saved)
for field,value in [('master_seed',7),('train_record_id_sha256','wrong'),('test_record_id_sha256','wrong'),('train_fraction',.76)]:
    bad=copy.deepcopy(saved);bad['split_design'][field]=value
    try:require_split_design(bad)
    except ValueError:pass
    else:raise AssertionError(field)
try:require_split_design({})
except ValueError:pass
else:raise AssertionError('missing provenance accepted')
b=a.copy();b['record_id']='B'+b.record_id
split_center_a_indices(b)
try:require_split_design(saved)
except ValueError:pass
else:raise AssertionError('different patient IDs accepted')
print('PASS: exact metadata accepted; wrong seed, draw, train/test identity hashes, missing provenance and different actual IDs rejected.')
