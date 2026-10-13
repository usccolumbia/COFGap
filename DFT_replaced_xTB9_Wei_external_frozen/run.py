#!/usr/bin/env python3
"""Fit xTB5 V3 once; externally evaluate on non-overlapping aligned Wei labels."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
XTB4=ROOT/'DFT_replaced_xTB4';XTB8=ROOT/'DFT_replaced_xTB8_Wei2025_725'
orig=list(csv.DictReader((XTB4/'01_feature_table/xtb4_features.csv').open()))
names=json.loads((XTB4/'01_feature_table/feature_groups.json').read_text())['physics_informed']
manifest={r['filename']:r for r in csv.DictReader((XTB8/'01_prepare'/'wei725_manifest.csv').open())}
external=[r for r in csv.DictReader((XTB8/'04_features'/'wei725_xtb_features.csv').open()) if manifest[r['filename']]['xtb5_id_overlap']=='no']
def number(v):
 try:return float(v)
 except:return np.nan
def matrix(rows):return np.asarray([[number(r.get(c)) for c in names] for r in rows])
X_train=matrix(orig);y_train=np.asarray([number(r['band_gap_eV']) for r in orig])

# Fit target conversion using the 156 same-structure labels only. These rows
# are excluded from the external prediction metric.
original_by_id={r['COF']:number(r['band_gap_eV']) for r in orig}
pairs=[]
for r in manifest.values():
 if r['xtb5_id_overlap']=='yes':pairs.append((number(r['reference_bandgap_eV']),original_by_id[r['source_id']]))
wei_scale=np.asarray([a for a,b in pairs]).reshape(-1,1);original_scale=np.asarray([b for a,b in pairs])
calibration=LinearRegression().fit(wei_scale,original_scale)
calibrated_pairs=calibration.predict(wei_scale)

imputer=SimpleImputer(strategy='median'); model=TabPFNRegressor.create_default_for_version(ModelVersion.V3)
model.device='cpu';model.n_estimators=8;model.show_progress_bar=False
model.fit(imputer.fit_transform(X_train),y_train)
X_ext=matrix(external);prediction=model.predict(imputer.transform(X_ext))
y_wei=np.asarray([number(manifest[r['filename']]['reference_bandgap_eV']) for r in external])
y_aligned=calibration.predict(y_wei.reshape(-1,1))
def score(target):return {'r2':float(r2_score(target,prediction)),'mae_eV':float(mean_absolute_error(target,prediction))}
metrics={'training':{'n':len(orig),'label_source':'original xTB5 target; frozen model'},
 'external':{'wei_total':725,'complete_rich_xtb_rows':476,'nonoverlapping_complete_rows':len(external),'overlap_rows_excluded_from_external_metric':156},
 'target_alignment':{'method':'ordinary least-squares: original_target = intercept + slope * wei_reference', 'pair_count':len(pairs),'slope':float(calibration.coef_[0]),'intercept':float(calibration.intercept_),'overlap_pair_r2':float(r2_score(original_scale,calibrated_pairs)),'overlap_pair_mae_eV':float(mean_absolute_error(original_scale,calibrated_pairs))},
 'frozen_xTB5_v3_external_metrics':{'against_raw_wei_labels_not_comparable':score(y_wei),'against_aligned_wei_labels_primary':score(y_aligned)}}
(HERE/'metrics.json').write_text(json.dumps(metrics,indent=2))
with (HERE/'external_predictions.csv').open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=['filename','wei_reference_eV','wei_aligned_to_original_eV','frozen_xtb5_v3_prediction_eV','absolute_error_aligned_eV']);w.writeheader()
 for r,a,b,p in zip(external,y_wei,y_aligned,prediction):w.writerow({'filename':r['filename'],'wei_reference_eV':a,'wei_aligned_to_original_eV':b,'frozen_xtb5_v3_prediction_eV':p,'absolute_error_aligned_eV':abs(b-p)})
print(json.dumps(metrics,indent=2))
