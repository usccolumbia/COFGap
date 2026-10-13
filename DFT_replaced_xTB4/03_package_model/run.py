#!/usr/bin/env python3
"""Fit and save deployable full-data TabPFN v2 + RF xTB4 bundle."""
from __future__ import annotations
import csv,json,os
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.model_loading import save_fitted_tabpfn_model

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent
rows=list(csv.DictReader((PROJECT/'01_feature_table/xtb4_features.csv').open()))
names=json.loads((PROJECT/'01_feature_table/feature_groups.json').read_text())['physics_informed']
def num(v):
 try:return float(v)
 except:return np.nan
X=np.array([[num(r.get(c)) for c in names] for r in rows]);y=np.array([float(r['band_gap_eV']) for r in rows])
keep=~np.all(np.isnan(X),axis=0);X=X[:,keep];names=list(np.asarray(names)[keep])
imputer=SimpleImputer(strategy='median',add_indicator=True);Xi=imputer.fit_transform(X)
tab=TabPFNRegressor.create_default_for_version(ModelVersion.V2)
tab.device='cpu';tab.n_estimators=8;tab.show_progress_bar=False;tab.fit(Xi,y)
save_fitted_tabpfn_model(tab,HERE/'xTB4_tabpfn_v2.tabpfn_fit')
rf=RandomForestRegressor(n_estimators=700,min_samples_leaf=2,max_features=.7,random_state=17,n_jobs=-1).fit(Xi,y)
metrics=json.loads((PROJECT/'02_grouped_benchmark/metrics.json').read_text())['results']
joblib.dump({'version':'DFT_replaced_xTB4','feature_names':names,'imputer':imputer,
 'random_forest':rf,'tabpfn_fit_file':'xTB4_tabpfn_v2.tabpfn_fit',
 'ensemble_weights':{'tabpfn':.75,'random_forest':.25},'n_train':len(y),
 'requires_DFT_inputs':False,'grouped_cv_tabpfn':metrics['physics_informed:tabpfn_v2'],
 'grouped_cv_ensemble':metrics['physics_informed:75pct_tabpfn_25pct_random_forest']},
 HERE/'DFT_replaced_xTB4_bundle.joblib')
print('saved xTB4 bundle with',len(names),'nonempty source features')
