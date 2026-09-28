from pathlib import Path
import ast,sys,json
import pandas as pd
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'01_methods/models/src'))
import fold_selection as f
for source in [{}, {'selection_protocol':'old'},pd.DataFrame({'x':[1]}),pd.DataFrame({'selection_protocol':[f.SELECTION_PROTOCOL,'old']})]:
 try: f.require_selection_protocol(source)
 except RuntimeError as e: assert 'regenerate' in str(e)
 else: raise AssertionError('old protocol accepted')
f.require_selection_protocol({'selection_protocol':f.SELECTION_PROTOCOL})
f.require_selection_protocol(pd.DataFrame({'selection_protocol':[f.SELECTION_PROTOCOL]*2}))
rels=['01_methods/models/src/fold_selection.py','01_methods/training/run_common_scheme.py','02_analysis/fusion/run_a_center_fusions.py','02_analysis/fusion/run_external_fusions.py','03_figures/weighted_analysis/make_F6_F8_weighted_layers.py']
for rel in rels: ast.parse((root/rel).read_text(encoding='utf-8-sig'))

path=root/rels[-1]; tree=ast.parse(path.read_text(encoding='utf-8-sig'))
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_existing_weighted')
class FakePD:
 DataFrame=pd.DataFrame
 def read_csv(self,path): return self.frame.copy()
pd_fake=FakePD(); pd_fake.frame=pd.DataFrame({'dataset':['A_dev_cv_clean'],'fusion':['weighted_voting_cv_auc'],'score':[.3],'fold':[1],'record_id':['x'],'y_SCLC':[1]})
ns={'pd':pd_fake,'EXISTING_A_CV':'synthetic','WEIGHTED_NAME':'weighted_voting_cv_auc','require_selection_protocol':f.require_selection_protocol}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<isolated-loader>','exec'),ns)
try: ns['_existing_weighted']('A_dev_cv_clean')
except RuntimeError: pass
else: raise AssertionError('legacy prediction accepted')
pd_fake.frame['selection_protocol']=f.SELECTION_PROTOCOL
assert len(ns['_existing_weighted']('A_dev_cv_clean'))==1

audit_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_audit_i_m_n')
assert '_existing_weighted' in ast.unparse(audit_node) and 'max_abs_difference' in ast.unparse(audit_node)
print(json.dumps({'status':'passed','checks':['five modified modules parse','stale/mixed protocol rejected','new protocol accepted','isolated archived-reference loader','strict numerical identity audit retained']}))
