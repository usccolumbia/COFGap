#!/usr/bin/env python3
"""Random 80/10/10 and family-proxy grouped CV on Wei complete-xTB cohort."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split,GroupKFold
from scipy.stats import spearmanr
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
HERE=Path(__file__).resolve().parent;P=HERE.parent
rows=list(csv.DictReader((P/'04_features'/'wei725_xtb_features.csv').open()));names=json.loads((P/'04_features'/'feature_names.json').read_text())
def n(x):
 try:return float(x)
 except:return np.nan
X=np.asarray([[n(r.get(c)) for c in names] for r in rows]);y=np.asarray([n(r['reference_bandgap_eV']) for r in rows]);g=np.asarray([r['family_proxy'] for r in rows]);idx=np.arange(len(y))
def model():
 m=TabPFNRegressor.create_default_for_version(ModelVersion.V3);m.device='cpu';m.n_estimators=8;m.show_progress_bar=False;return m
def score(a,b):
 rho,p=spearmanr(a,b)
 return {'r2':float(r2_score(a,b)),'mae_eV':float(mean_absolute_error(a,b)),
         'spearman_rho':float(rho),'spearman_p_value':float(p)}
tv,te=train_test_split(idx,test_size=.10,random_state=17);tr,va=train_test_split(tv,test_size=len(te)/len(tv),random_state=17)
imp=SimpleImputer(strategy='median');m=model().fit(imp.fit_transform(X[tr]),y[tr]);pv=m.predict(imp.transform(X[va]));pt=m.predict(imp.transform(X[te]))
oof=np.full(len(y),np.nan);audit=[]
for f,(a,b) in enumerate(GroupKFold(5).split(X,y,g)):
 overlap=set(g[a])&set(g[b]);assert not overlap
 im=SimpleImputer(strategy='median');oof[b]=model().fit(im.fit_transform(X[a]),y[a]).predict(im.transform(X[b]));audit.append({'fold':f,'n_train':len(a),'n_test':len(b),'shared_families':len(overlap)})
res={'cohort_n':len(y),'family_proxy_count':len(set(g)),'random_80_10_10':{'n_train':len(tr),'n_validation':len(va),'n_test':len(te),'validation':score(y[va],pv),'test':score(y[te],pt),'train_test_shared_family_proxies':len(set(g[tr])&set(g[te]))},'family_proxy_grouped_5fold':{'metrics':score(y,oof),'fold_audit':audit}}
(HERE/'metrics.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
with (HERE/'random_test_predictions.csv').open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=['filename','family_proxy','reference_bandgap_eV','prediction_eV']);w.writeheader()
 for i,pred in zip(te,pt):w.writerow({'filename':rows[i]['filename'],'family_proxy':g[i],'reference_bandgap_eV':y[i],'prediction_eV':pred})
