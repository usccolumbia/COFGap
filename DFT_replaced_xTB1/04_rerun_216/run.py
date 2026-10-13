#!/usr/bin/env python3
"""Rerun the xTB1 rows on the 216-COF / 72-family cohort with the xTB4 folds."""
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_rows, make_matrix, estimators

ref = list(csv.DictReader((HERE.parents[1] / "DFT_replaced_xTB4/01_feature_table/xtb4_features.csv").open()))
order = [r["COF"] for r in ref]
fam = np.array([r["family"] for r in ref])
y_ref = np.array([float(r["band_gap_eV"]) for r in ref])
by_id = {r["id"]: r for r in load_rows()}
missing = [c for c in order if c not in by_id]
assert not missing, missing[:5]
rows = [by_id[c] for c in order]
folds = list(GroupKFold(5).split(np.zeros(len(rows)), y_ref, fam))
out = {"n": len(rows), "n_families": int(len(set(fam))), "results": []}
for label, kw in [("structure_only", {}), ("structure+xtb_gap", {"include_xtb": True})]:
    X, names, valid = make_matrix(rows, **kw)
    assert all(valid), "invalid rows in 216 cohort"
    y = np.array([r["gap"] for r in rows])
    assert np.allclose(y, y_ref, atol=1e-6)
    for mname, model in estimators().items():
        p = np.zeros(len(y))
        for tr, te in folds:
            p[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
        res = {"features": label, "model": mname, "n_features": len(names),
               "r2": float(r2_score(y, p)), "mae_eV": float(mean_absolute_error(y, p))}
        print(res); out["results"].append(res)
(HERE / "metrics.json").write_text(json.dumps(out, indent=2))
