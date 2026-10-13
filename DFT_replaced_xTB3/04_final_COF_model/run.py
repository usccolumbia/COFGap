#!/usr/bin/env python3
"""Compare raw, calibrated, surrogate, and DFT fragment features in COF model."""
from __future__ import annotations
import csv, importlib.util, json
from pathlib import Path
import numpy as np
import joblib

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parent
xtb2_common=PROJECT.parent/'DFT_replaced_xTB2/common.py'
spec=importlib.util.spec_from_file_location('xtb2_common',xtb2_common)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def read_predictions(path, metric_path):
    metric=json.loads(metric_path.read_text()); best=metric.get('best_model')
    rows=list(csv.DictReader(path.open()))
    if best is None:
        candidates=[k for k in rows[0] if k not in ('COF','role','canonical_smiles','fold','pbe0_gap_eV')]
        scores=metric['models'];best=min(candidates,key=lambda x:scores[x]['mae_eV'])
    return {(r['COF'],r['role']):float(r[best]) for r in rows},best

delta,delta_best=read_predictions(PROJECT/'02_delta_calibration/oof_calibrated_predictions.csv',PROJECT/'02_delta_calibration/metrics.json')
surrogate,surrogate_best=read_predictions(PROJECT/'03_direct_surrogate/oof_surrogate_predictions.csv',PROJECT/'03_direct_surrogate/metrics.json')
labels=list(csv.DictReader((PROJECT/'00_Zenodo_14855145/per_fragment_pbe0_labels.csv').open()))
pbe0={(r['COF'],r['role']):float(r['pbe0_gap_eV']) for r in labels}
raw_pairs=list(csv.DictReader((PROJECT/'01_calibration_dataset/fragment_calibration_pairs.csv').open()))
raw={(r['COF'],r['role']):float(r['xtb_gap_eV']) for r in raw_pairs}
roles=['node','linker','nodelinker']
rows=[r for r in m.load_rows() if all((r['id'],role) in delta for role in roles)]
X_structure,structure_names,_=m.make_matrix(rows)

def electronic(table):
    base=np.array([[table[(r['id'],role)] for role in roles] for r in rows])
    derived=np.column_stack([base[:,0]-base[:,1],np.minimum(base[:,0],base[:,1]),
                             np.maximum(base[:,0],base[:,1]),base[:,2]-np.minimum(base[:,0],base[:,1])])
    return np.column_stack([base,derived])

gap_names=['node_gap','linker_gap','nodelinker_gap','node_minus_linker_gap',
           'min_fragment_gap','max_fragment_gap','nodelinker_minus_min_gap']
variants={'raw_xtb':raw,'delta_calibrated_xtb':delta,'direct_surrogate':surrogate,
          'DFT_reference':pbe0}
summary={'population':{'n':len(rows),'families':len({r['family'] for r in rows})},
         'calibration_model':delta_best,'surrogate_model':surrogate_best,'variants':{}}
for name,table in variants.items():
    X=np.column_stack([X_structure,electronic(table)])
    result=m.evaluate(rows,X,structure_names+gap_names,HERE/f'results_{name}',label=name+'+structure')
    summary['variants'][name]=result
(HERE/'comparison.json').write_text(json.dumps(summary,indent=2))
# Package the best deployable DFT-free variant. The saved model expects the
# inexpensive structure descriptors followed by the seven raw-xTB gap fields.
X_best=np.column_stack([X_structure,electronic(raw)])
best_model=m.estimators()['random_forest'].fit(X_best,np.array([r['gap'] for r in rows]))
joblib.dump({'model':best_model,'version':'DFT_replaced_xTB3',
             'feature_variant':'raw_xtb+structure','feature_names':structure_names+gap_names,
             'n_train':len(rows),'n_families':len({r['family'] for r in rows}),
             'grouped_cv':summary['variants']['raw_xtb']['random_forest'],
             'requires_DFT_inputs':False},HERE/'DFT_replaced_xTB3_model.joblib')
print(json.dumps(summary,indent=2))
