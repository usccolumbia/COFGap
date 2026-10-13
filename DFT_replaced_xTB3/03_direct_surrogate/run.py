#!/usr/bin/env python3
"""Direct SMILES -> PBE0 gap surrogate, grouped by canonical fragment."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parent; sys.path.insert(0,str(PROJECT))
from common import DESC_NAMES,descriptors,fingerprint,grouped_oof,metrics
rows=list(csv.DictReader((PROJECT/'00_Zenodo_14855145/per_fragment_pbe0_labels.csv').open()))
role={'node':0.0,'linker':1.0,'nodelinker':2.0}
X=np.array([descriptors(r['canonical_smiles'])+fingerprint(r['canonical_smiles']).tolist()+[role[r['role']]] for r in rows])
y=np.array([float(r['pbe0_gap_eV']) for r in rows]); groups=np.array([r['canonical_smiles'] for r in rows])
models={
 'extra_trees':make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(n_estimators=700,min_samples_leaf=2,max_features=.7,random_state=41,n_jobs=-1)),
 'random_forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=700,min_samples_leaf=2,max_features=.7,random_state=41,n_jobs=-1)),
}
scores={};preds={};folds=None
for name,model in models.items():
 p,folds=grouped_oof(model,X,y,groups);preds[name]=p;scores[name]=metrics(y,p)
(HERE/'metrics.json').write_text(json.dumps({'n':len(y),'unique_fragments':len(set(groups)),
 'grouping':'canonical_smiles','models':scores},indent=2))
with (HERE/'oof_surrogate_predictions.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['COF','role','canonical_smiles','fold','pbe0_gap_eV',*preds])
 for i,r in enumerate(rows):w.writerow([r['COF'],r['role'],r['canonical_smiles'],folds[i],y[i],*[preds[k][i] for k in preds]])
print(json.dumps(scores,indent=2))
