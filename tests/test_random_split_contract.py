import os,sys
from pathlib import Path
import pandas as pd,numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from common.manuscript_cohorts import split_center_a_indices,split_metadata,split_design
for n in (100,969,988):
    frame=pd.DataFrame({'record_id':[f'A{i:05}' for i in range(n)],'y_SCLC':np.arange(n)%5==0})
    tr,te=split_center_a_indices(frame); saved=split_metadata()
    assert .70<=len(tr)/n<=.80
    if n==988:
        assert len(tr)==775 and len(te)==213
    for seed in (1,42,999999):
        os.environ['MANUSCRIPT_SPLIT_SEED']=str(seed)
        other_tr,other_te=split_center_a_indices(frame)
        assert np.array_equal(tr,other_tr) and np.array_equal(te,other_te) and split_metadata()==saved
try:split_design(5)
except ValueError:pass
else:raise AssertionError('seed change accepted')
print('PASS frozen configuration applies identically regardless of environment seed; ratios valid for 100/969/988; explicit other seed rejected.')
