from pathlib import Path
import ast,sys
import pandas as pd
import numpy as np
from typing import Any
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r));from common.manuscript_bootstrap import canonical_rows
files=['03_figures/publication/make_ordered_layerwise_abc_figures.py','03_figures/publication/run_v4_data_scope_qc.py','03_figures/interaction_calibration/make_F6_F10_and_selection_supp.py','03_figures/weighted_analysis/make_F6_F8_weighted_layers.py','03_figures/interaction_calibration/make_F11_weighted_explainability_v2.py']
for file in files:ast.parse((r/file).read_text(encoding='utf-8'))
def functions(file,names,ns):
 tree=ast.parse((r/file).read_text(encoding='utf-8'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];exec(compile(ast.Module(body=nodes,type_ignores=[]),file,'exec'),ns)
ns={'pd':pd,'np':np,'canonical_rows':canonical_rows}
functions(files[3],['_aligned_layers'],ns)
frame=pd.DataFrame([{'dataset':'A_dev_cv_clean','layer':layer,'fold':i%2,'record_id':f'r{i}','y_SCLC':i%2,'probability':.1+.1*i} for layer in ['I','I_M_N'] for i in range(8)])
assert len(ns['_aligned_layers'](frame,'I','I_M_N')[0])==8
for bad in [frame.iloc[:-1],pd.concat([frame,frame.iloc[[-1]]]),frame.assign(y_SCLC=lambda x:np.where((x.layer=='I_M_N')&(x.record_id=='r0'),1,x.y_SCLC))]:
 try:ns['_aligned_layers'](bad,'I','I_M_N')
 except (RuntimeError,ValueError):pass
 else:raise AssertionError('case mismatch accepted')
functions(files[1],['_case_labels','_same_cases'],ns)
ref=ns['_case_labels'](frame.loc[frame.layer=='I'])
assert ns['_same_cases'](frame.loc[frame.layer=='I_M_N'].iloc[::-1],ref)
assert not ns['_same_cases'](frame.loc[frame.layer=='I_M_N'].assign(record_id=lambda x:'x'+x.record_id),ref)

ds=['A_dev_cv_clean','A_holdout_clean','B_external','C_external'];models=['weighted_voting_cv_auc','soft','stack'];layers=['I','I_M','I_M_N'];y=np.array([0,1,0,1,0,1,0,1]);p=np.array([.1,.8,.2,.7,.3,.6,.4,.5])
patients=pd.DataFrame([{'dataset':d,'model':m,'record_id':f'{d}r{i}','source_record_id':f'{d}r{i}','y_SCLC':y[i],'probability':p[i]} for d in ds for m in models for i in range(8)])
lp=pd.DataFrame([{'dataset':d,'model':models[0],'layer':layer,'record_id':f'{d}r{i}','y_SCLC':y[i],'probability':p[i]} for d in ds for layer in layers for i in range(8)])
points={'auc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'brier':brier_score_loss(y,p)}
metrics=pd.DataFrame([{'dataset':d,'model':m,**{key:value for name,val in points.items() for key,value in [(name,val),(name+'_ci_low',val),(name+'_ci_high',val)]}} for d in ds for m in models])
curves=pd.DataFrame([{'dataset':d,'model':m} for d in ds for m in models]);dca=pd.DataFrame([{'dataset':d,'model':m,'curve_type':'model','threshold':t} for d in ds for m in models for t in np.linspace(.01,.8,160)])
frames={'patient_predictions':patients,'layer_predictions':lp,'metrics':metrics,'roc':curves,'pr':curves,'calibration':curves,'dca':dca}
ns.update({'A_DEVELOPMENT':ds[0],'A_HOLDOUT':ds[1],'B_EXTERNAL':ds[2],'C_EXTERNAL':ds[3],'PRIMARY':models[0],'LAYER_ORDER':layers,'FUSIONS':models,'SIX_MODELS':models,'roc_auc_score':roc_auc_score,'average_precision_score':average_precision_score,'brier_score_loss':brier_score_loss,'metric_row':lambda table,d,m:table.loc[(table.dataset==d)&(table.model==m)].iloc[0]})
functions(files[2],['validate_frozen_data'],ns);ns['validate_frozen_data'](frames)
wrong=lp.copy();wrong.loc[0,'record_id']='other'
try:ns['validate_frozen_data']({**frames,'layer_predictions':wrong})
except RuntimeError:pass
else:raise AssertionError('same-size different layer IDs accepted')
print('PASS: syntax 5 files; arbitrary split sample size; outer paired alignment rejects missing/duplicates/outcome changes; QC matches IDs not counts; v2 accepts actual n=8 and rejects equal-size mismatched IDs')
