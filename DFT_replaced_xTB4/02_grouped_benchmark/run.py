#!/usr/bin/env python3
"""Same-fold RF/ExtraTrees/LightGBM/TabPFN comparison."""
from __future__ import annotations
import csv,json,os
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from tabpfn.constants import ModelVersion

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent
rows=list(csv.DictReader((PROJECT/'01_feature_table/xtb4_features.csv').open()))
groups=json.loads((PROJECT/'01_feature_table/feature_groups.json').read_text())
def num(v):
 try:return float(v)
 except:return np.nan
y=np.array([float(r['band_gap_eV']) for r in rows]);families=np.array([r['family'] for r in rows])
folds=list(GroupKFold(5).split(np.zeros(len(y)),y,families))
fold_ids=np.full(len(y),-1,dtype=int)
fold_audit=[]
for fold_id,(train,test) in enumerate(folds):
 train_families=set(families[train]);test_families=set(families[test])
 overlap=train_families & test_families
 assert not overlap, f'Family leakage in fold {fold_id}: {sorted(overlap)}'
 assert not (set(train) & set(test)), f'Row overlap in fold {fold_id}'
 fold_ids[test]=fold_id
 fold_audit.append({'fold':fold_id,'n_train_cofs':len(train),'n_test_cofs':len(test),
  'n_train_families':len(train_families),'n_test_families':len(test_families),
  'shared_families':len(overlap)})
assert np.all(fold_ids>=0)

def sklearn_oof(model,X):
 pred=np.full(len(y),np.nan)
 for tr,te in folds:pred[te]=model.fit(X[tr],y[tr]).predict(X[te])
 return pred
def tabpfn_oof(X):
 from tabpfn import TabPFNRegressor
 from tabpfn.constants import ModelVersion
 pred=np.full(len(y),np.nan)
 for tr,te in folds:
  imp=SimpleImputer(strategy='median');xtr=imp.fit_transform(X[tr]);xte=imp.transform(X[te])
  model=TabPFNRegressor.create_default_for_version(ModelVersion.V2)
  model.device='cpu';model.n_estimators=8;model.show_progress_bar=False
  model.fit(xtr,y[tr]);pred[te]=model.predict(xte)
 return pred
def score(p):return {'r2':float(r2_score(y,p)),'mae_eV':float(mean_absolute_error(y,p))}

results={};allpred={}
for setname,names in groups.items():
 X=np.array([[num(r.get(c)) for c in names] for r in rows])
 models={
  'random_forest':make_pipeline(SimpleImputer(strategy='median',add_indicator=True),RandomForestRegressor(n_estimators=700,min_samples_leaf=2,max_features=.7,random_state=17,n_jobs=-1)),
  'extra_trees':make_pipeline(SimpleImputer(strategy='median',add_indicator=True),ExtraTreesRegressor(n_estimators=700,min_samples_leaf=2,max_features=.7,random_state=17,n_jobs=-1)),
  'hist_gradient_boosting':make_pipeline(SimpleImputer(strategy='median',add_indicator=True),HistGradientBoostingRegressor(max_iter=300,learning_rate=.04,max_leaf_nodes=12,min_samples_leaf=12,l2_regularization=2,random_state=17))}
 for name,model in models.items():
  p=sklearn_oof(model,X);results[f'{setname}:{name}']=score(p);allpred[f'{setname}:{name}']=p
 p=tabpfn_oof(X);results[f'{setname}:tabpfn_v2']=score(p);allpred[f'{setname}:tabpfn_v2']=p
# Fixed heterogeneous ensembles: TabPFN supplies a strong small-data prior,
# while trees contribute a complementary local partition of feature space.
t=allpred['physics_informed:tabpfn_v2']
for tree_name in ('extra_trees','random_forest'):
 key=f'physics_informed:{tree_name}'; blend=.75*t+.25*allpred[key]
 outkey=f'physics_informed:75pct_tabpfn_25pct_{tree_name}'
 allpred[outkey]=blend;results[outkey]=score(blend)
(HERE/'metrics.json').write_text(json.dumps({'n':len(y),'families':len(set(families)),
 'cv':'5-fold GroupKFold identical across models; family and row overlap assertions passed',
 'fold_audit':fold_audit,'results':results},indent=2))
with (HERE/'oof_predictions.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['COF','family','fold','band_gap_eV',*allpred])
 for i,r in enumerate(rows):w.writerow([r['COF'],r['family'],fold_ids[i],y[i],*[allpred[k][i] for k in allpred]])
print(json.dumps(results,indent=2))
