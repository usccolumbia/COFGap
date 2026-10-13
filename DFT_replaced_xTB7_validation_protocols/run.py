#!/usr/bin/env python3
"""Compare random 80/10/10 and family-grouped CV for xTB5 TabPFN V3."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

HERE = Path(__file__).resolve().parent
XTB4 = HERE.parent / "DFT_replaced_xTB4"
rows = list(csv.DictReader((XTB4 / "01_feature_table/xtb4_features.csv").open()))
features = json.loads((XTB4 / "01_feature_table/feature_groups.json").read_text())["physics_informed"]

def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

X = np.asarray([[number(row.get(feature)) for feature in features] for row in rows])
y = np.asarray([number(row["band_gap_eV"]) for row in rows])
families = np.asarray([row["family"] for row in rows])

def score(y_true, prediction):
    return {"r2": float(r2_score(y_true, prediction)), "mae_eV": float(mean_absolute_error(y_true, prediction))}

def make_model():
    model = TabPFNRegressor.create_default_for_version(ModelVersion.V3)
    model.device = "cpu"; model.n_estimators = 8; model.show_progress_bar = False
    return model

# One reproducible random split: 172 train / 22 validation / 22 test.
indices = np.arange(len(y))
train_val, test = train_test_split(indices, test_size=0.10, random_state=17, shuffle=True)
train, validation = train_test_split(train_val, test_size=22 / len(train_val), random_state=17, shuffle=True)
assert (len(train), len(validation), len(test)) == (172, 22, 22)
assert not (set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test))
imputer = SimpleImputer(strategy="median")
x_train = imputer.fit_transform(X[train])
model = make_model().fit(x_train, y[train])
random_pred_validation = model.predict(imputer.transform(X[validation]))
random_pred_test = model.predict(imputer.transform(X[test]))

# Same model/representation, but no family may cross train/test in any fold.
oof = np.full(len(y), np.nan)
fold_ids = np.full(len(y), -1, dtype=int)
fold_audit = []
for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, families)):
    overlap = set(families[tr]) & set(families[te])
    assert not overlap and not (set(tr) & set(te))
    fold_imputer = SimpleImputer(strategy="median")
    fitted = make_model().fit(fold_imputer.fit_transform(X[tr]), y[tr])
    oof[te] = fitted.predict(fold_imputer.transform(X[te])); fold_ids[te] = fold
    fold_audit.append({"fold": fold, "n_train": len(tr), "n_test": len(te), "shared_families": len(overlap)})
assert np.all(fold_ids >= 0)

random_split = {"seed": 17, "n_train": len(train), "n_validation": len(validation), "n_test": len(test),
                "family_overlap_train_test": int(len(set(families[train]) & set(families[test]))),
                "family_overlap_train_validation": int(len(set(families[train]) & set(families[validation]))),
                "validation": score(y[validation], random_pred_validation), "test": score(y[test], random_pred_test)}
metrics = {"model": "TabPFN V3", "cohort": {"n": len(y), "families": len(set(families)), "features_requested": len(features)},
           "random_80_10_10": random_split,
           "family_grouped_5fold": {"metrics": score(y, oof), "fold_audit": fold_audit}}
(HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
with (HERE / "split_assignments.csv").open("w", newline="") as handle:
    writer = csv.writer(handle); writer.writerow(["COF", "family", "random_split", "group_fold", "band_gap_eV", "grouped_oof_prediction_eV"])
    membership = {i: "train" for i in train} | {i: "validation" for i in validation} | {i: "test" for i in test}
    for i, row in enumerate(rows): writer.writerow([row["COF"], families[i], membership[i], fold_ids[i], y[i], oof[i]])
print(json.dumps(metrics, indent=2))
