from pathlib import Path
import ast,sys
import numpy as np
import pandas as pd
r=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(r));import common.manuscript_bootstrap as b
changed=['common/manuscript_bootstrap.py','03_figures/weighted_analysis/plot_f9_f10_weighted_voting.py','03_figures/weighted_analysis/make_F6_F8_weighted_layers.py','03_figures/interaction_calibration/make_F6_F10_and_selection_supp.py','03_figures/publication/rebuild_f19_shap_stability.py']
for rel in changed: ast.parse((r/rel).read_text(encoding='utf-8'))
y=np.array([0,1,0,0,1]);ids=np.array(['z','b','c','a','e']);draws=b.shared_bootstrap_indices(y,'A_holdout_clean',ids)
assert len(draws)==2000 and all((y[d]==1).sum()==2 and len(d)==5 for d in draws)
perm=np.array([4,2,0,3,1]);shuffled=b.shared_bootstrap_indices(y[perm],'A_holdout_clean',ids[perm])
assert all(np.array_equal(ids[d],ids[perm][e]) for d,e in zip(draws,shuffled))
assert b.bootstrap_identity(y,ids)==b.bootstrap_identity(y[perm],ids[perm])
assert not np.array_equal(draws,b.shared_bootstrap_indices(y,'B_external',ids))
for yy,ii in [(y,['a']*5),(np.array([0,1,2,0,1]),ids),(y,[None,'b','c','d','e'])]:
 try:b.shared_bootstrap_indices(yy,'A_holdout_clean',ii)
 except ValueError:pass
 else:raise AssertionError('invalid identity/outcome accepted')
threshold,j=b.primary_youden_threshold(y,np.array([.1,.8,.2,.3,.7]));assert threshold==.7 and j==1
assert b.threshold_metric_values(y,np.array([.1,.8,.2,.3,.7]),threshold)['sensitivity']==1

p=r/changed[3];tree=ast.parse(p.read_text(encoding='utf-8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='harmonise_full_layer_ci')
ns={'pd':pd,'np':np,'BOOTSTRAP_PROTOCOL':b.BOOTSTRAP_PROTOCOL};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
try:ns['harmonise_full_layer_ci']({'metrics':pd.DataFrame({'auc':[.7]}),'layer_metrics':pd.DataFrame({'auc':[.7]})})
except RuntimeError as e:assert 'regenerated' in str(e)
else:raise AssertionError('old result accepted')
print('PASS: syntax 5 files; 2000 draws; fixed event counts; row-order invariance; dataset seeds; identity validation; OOF threshold; reject stale CI')

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from typing import Any, Sequence
ns=dict(vars(b));ns.update({'Any':Any,'Sequence':Sequence,'roc_auc_score':roc_auc_score,'average_precision_score':average_precision_score,'brier_score_loss':brier_score_loss,'LAYER_ORDER':['I','I_M_N'],'LAYER_LABELS':{'I':'I','I_M_N':'full'},'DATASET_ORDER':['A_dev_cv_clean'],'DATASET_LABELS':{'A_dev_cv_clean':'dev'},'WEIGHTED_NAME':'weighted_voting_cv_auc'})
p=r/'03_figures/weighted_analysis/make_F6_F8_weighted_layers.py';tree=ast.parse(p.read_text(encoding='utf-8'))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_metric_values','_percentile_interval','_metrics_with_ci']]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
rows=[]
for layer in ['I','I_M_N']:
 for i in (range(8) if layer=='I' else reversed(range(8))):rows.append({'layer':layer,'dataset':'A_dev_cv_clean','record_id':f'r{i}','y_SCLC':i%2,'probability':(.2,.7,.3,.6,.4,.8,.1,.5)[i],'fold':i//2})
threshold_info=b.verify_manuscript_threshold(np.array([0,1,0,1,0,1,0,1]),np.array([.1,.3,.15,.2076,.2,.8,.05,.5]))
assert threshold_info['matches_manuscript'] and threshold_info['derived_threshold']==0.2076
try:b.verify_manuscript_threshold(y,np.array([.1,.8,.2,.3,.7]))
except ValueError:pass
else:raise AssertionError('threshold mismatch accepted')
result=ns['_metrics_with_ci'](pd.DataFrame(rows),threshold_info)
for key in ['auc_ci_low','auc_ci_high','sensitivity_ci_low','specificity_ci_high','bootstrap_case_sha256','threshold']:assert result[key].nunique()==1,key
assert result['threshold'].iloc[0]==0.2076
assert result['bootstrap_n'].eq(2000).all()

assert 'bootstrap_protocol' in result
print('PASS: layer metric integration, classification CI, shared case fingerprints, identical predictions -> identical intervals; fixed threshold from OOF')

p=r/'03_figures/weighted_analysis/plot_f9_f10_weighted_voting.py';tree=ast.parse(p.read_text(encoding='utf-8'))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['percentile_interval','compute_all_threshold_metrics']]
ns2=dict(vars(b));ns2.update({'PRIMARY_MODEL':'weighted_voting_cv_auc','DATASET_ORDER':['A_holdout_clean'],'DATASET_LABELS':{'A_holdout_clean':'eval'},'MODEL_LABELS':{}})
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns2)
records=[]
for model in ['weighted_voting_cv_auc','logistic_regression']:
 for i in (range(8) if model=='weighted_voting_cv_auc' else reversed(range(8))):records.append({'model':model,'dataset':'A_holdout_clean','record_id':f'r{i}','y_SCLC':i%2,'probability':(.2,.7,.3,.6,.4,.8,.1,.5)[i]})
table=ns2['compute_all_threshold_metrics'](pd.DataFrame(records),.5,1.)
assert set(table['model'])=={'weighted_voting_cv_auc','logistic_regression'}
assert table['sensitivity_ci_low'].nunique()==1 and table['bootstrap_case_sha256'].nunique()==1
print('PASS: non-plotted logistic-regression included in threshold table, canonical rows align, same CI and case fingerprint')
