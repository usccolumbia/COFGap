#!/usr/bin/env python3
"""Raw xTB versus leakage-resistant delta-calibrated PBE0 fragment gaps."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parent; sys.path.insert(0,str(PROJECT))
from common import DESC_NAMES, grouped_oof, metrics

pairs=list(csv.DictReader((PROJECT/'01_calibration_dataset/fragment_calibration_pairs.csv').open()))
role_map={'node':0.0,'linker':1.0,'nodelinker':2.0}
xtb_names=['xtb_gap_eV','xtb_homo_eV','xtb_lumo_eV','xtb_vertical_ip_eV',
           'xtb_vertical_ea_eV','xtb_charge_mean_abs_e','xtb_charge_std_e',
           'xtb_dipole_debye','xtb_bond_order_mean','xtb_bond_order_max']
names=xtb_names+DESC_NAMES+['role_code']
def num(x):
    try:return float(x)
    except:return np.nan
X=np.array([[num(r.get(c)) for c in xtb_names+DESC_NAMES]+[role_map[r['role']]] for r in pairs])
y=np.array([float(r['pbe0_gap_eV']) for r in pairs]); raw=np.array([float(r['xtb_gap_eV']) for r in pairs])
delta=y-raw; groups=np.array([r['canonical_smiles'] for r in pairs])
models={
 'linear_gap': make_pipeline(SimpleImputer(strategy='median'),LinearRegression()),
 'ridge_delta': make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,25))),
 'extra_trees_delta': make_pipeline(SimpleImputer(strategy='median',add_indicator=True),ExtraTreesRegressor(n_estimators=600,min_samples_leaf=2,max_features=.8,random_state=31,n_jobs=-1)),
 'random_forest_delta': make_pipeline(SimpleImputer(strategy='median',add_indicator=True),RandomForestRegressor(n_estimators=600,min_samples_leaf=2,max_features=.8,random_state=31,n_jobs=-1)),
}
predictions={'raw_xtb':raw}; scores={'raw_xtb':metrics(y,raw)}; folds=None
for name,model in models.items():
    target=y if name=='linear_gap' else delta
    pred,folds=grouped_oof(model,X,target,groups)
    if name!='linear_gap': pred=raw+pred
    predictions[name]=pred; scores[name]=metrics(y,pred)
best=min(scores,key=lambda k:scores[k]['mae_eV'])
(HERE/'metrics.json').write_text(json.dumps({'n':len(y),'unique_fragments':len(set(groups)),
    'grouping':'canonical_smiles','best_model':best,'models':scores},indent=2))
with (HERE/'oof_calibrated_predictions.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['COF','role','canonical_smiles','fold','pbe0_gap_eV',*predictions])
    for i,r in enumerate(pairs):w.writerow([r['COF'],r['role'],r['canonical_smiles'],folds[i],y[i],*[predictions[k][i] for k in predictions]])
print(json.dumps(scores,indent=2)); print('best',best)
