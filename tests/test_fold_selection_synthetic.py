from pathlib import Path
import sys, builtins, json
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'01_methods/models/src'))
import fold_selection as f
rng=np.random.default_rng(20260928)
x=pd.DataFrame(rng.normal(size=(100,32)),columns=f.DIRECT_FEATURES)
y=(x['CRP'].to_numpy()+0.3*rng.normal(size=100)>0).astype(int)
result=f.fit_scope_selection(x,y,fold_number=1)
assert len(result['methods'])==3
assert result['methods'][2]['backend']=='boruta_py'
assert all(result['consensus_counts'][name]>=2 for name in result['selected_direct'])
original=builtins.__import__
def missing(name,*args,**kwargs):
 if name=='boruta': raise ImportError('synthetic unavailable-backend test')
 return original(name,*args,**kwargs)
builtins.__import__=missing
try:
 try: f.select_boruta(x,y,1)
 except RuntimeError as exc: assert 'no substitute' in str(exc)
 else: raise AssertionError('Boruta failure substituted')
finally: builtins.__import__=original
print(json.dumps({'synthetic_selection':'passed','n_rows':100,'n_candidates':32,'selected':result['selected_direct'],'boruta_failure':'raised without substitute'},ensure_ascii=False))
