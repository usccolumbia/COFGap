#!/usr/bin/env python3
"""Same-fold TabPFN v3 vs v3.5 (alone and 75/25 with RF) on the xTB4 features."""
import csv, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

HERE = Path(__file__).resolve().parent
XTB4 = HERE.parent / "DFT_replaced_xTB4"
rows = list(csv.DictReader((XTB4 / "01_feature_table/xtb4_features.csv").open()))
names = json.loads((XTB4 / "01_feature_table/feature_groups.json").read_text())["physics_informed"]
def num(v):
    try: return float(v)
    except (TypeError, ValueError): return np.nan
y = np.array([num(r["band_gap_eV"]) for r in rows])
fam = np.array([r["family"] for r in rows])
X = np.array([[num(r.get(n)) for n in names] for r in rows])
folds = list(GroupKFold(5).split(X, y, fam))
for tr, te in folds:
    assert not (set(fam[tr]) & set(fam[te]))

def tabpfn_oof(version):
    p = np.zeros(len(y))
    for tr, te in folds:
        imp = SimpleImputer(strategy="median")
        m = TabPFNRegressor.create_default_for_version(version)
        m.device = "cpu"; m.n_estimators = 8; m.show_progress_bar = False
        m.fit(imp.fit_transform(X[tr]), y[tr]); p[te] = m.predict(imp.transform(X[te]))
    return p

def rf_oof():
    p = np.zeros(len(y))
    for tr, te in folds:
        m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
            RandomForestRegressor(n_estimators=700, min_samples_leaf=2, max_features=.7,
                                  random_state=17, n_jobs=-1))
        p[te] = m.fit(X[tr], y[tr]).predict(X[te])
    return p

score = lambda p: {"r2": float(r2_score(y, p)), "mae_eV": float(mean_absolute_error(y, p))}
pred = {"tabpfn_v3_5": tabpfn_oof(ModelVersion.V3_5), "random_forest": rf_oof()}
pred["ens_75v3_5_25rf"] = .75 * pred["tabpfn_v3_5"] + .25 * pred["random_forest"]
v3 = HERE.parent / "DFT_replaced_xTB5_TabPFN_V3/01_benchmark_v3/oof_predictions.csv"
old = {r["COF"]: r for r in csv.DictReader(v3.open())}
col = [c for c in next(iter(old.values())) if "v3" in c.lower()][0]
pred["tabpfn_v3_saved"] = np.array([float(old[r["COF"]][col]) for r in rows])
pred["ens_75v3_25rf"] = .75 * pred["tabpfn_v3_saved"] + .25 * pred["random_forest"]
res = {k: score(v) for k, v in pred.items()}
for k, v in res.items(): print(f"{k:18s} R2={v['r2']:.4f} MAE={v['mae_eV']:.4f}")
(HERE / "metrics.json").write_text(json.dumps({"n": len(y), "n_families": len(set(fam)),
    "n_features": len(names), "results": res}, indent=2))
with (HERE / "oof_predictions.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["COF", "family", "band_gap_eV", *pred])
    for i, r in enumerate(rows): w.writerow([r["COF"], r["family"], y[i], *[pred[k][i] for k in pred]])
