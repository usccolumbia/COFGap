#!/usr/bin/env python3
"""Rerun the xTB7 seed-17 random 80/10/10 holdout with TabPFN v3.5 (same partitions)."""
import csv, json
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

HERE = Path(__file__).resolve().parent
XTB4 = HERE.parent / "DFT_replaced_xTB4"
rows = list(csv.DictReader((XTB4 / "01_feature_table/xtb4_features.csv").open()))
names = json.loads((XTB4 / "01_feature_table/feature_groups.json").read_text())["physics_informed"]
split = {r["COF"]: r["random_split"] for r in
         csv.DictReader((HERE.parent / "DFT_replaced_xTB7_validation_protocols/split_assignments.csv").open())}
def num(v):
    try: return float(v)
    except (TypeError, ValueError): return np.nan
y = np.array([num(r["band_gap_eV"]) for r in rows])
X = np.array([[num(r.get(n)) for n in names] for r in rows])
fam = np.array([r["family"] for r in rows])
part = np.array([split[r["COF"]] for r in rows])
tr = part == "train"
imp = SimpleImputer(strategy="median")
m = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5)
m.device = "cpu"; m.n_estimators = 8; m.show_progress_bar = False
m.fit(imp.fit_transform(X[tr]), y[tr])
out = {}
for name in ("validation", "test"):
    s = part == name
    p = m.predict(imp.transform(X[s]))
    out[name] = {"n": int(s.sum()), "r2": float(r2_score(y[s], p)),
                 "mae_eV": float(mean_absolute_error(y[s], p)),
                 "shared_families_with_train": len(set(fam[s]) & set(fam[tr]))}
    print(name, out[name])
(HERE / "random_holdout_metrics.json").write_text(json.dumps(out, indent=2))
