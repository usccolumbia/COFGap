#!/usr/bin/env python3
"""Direct, same-fold TabPFN V2 versus V3 benchmark using xTB4 features."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

HERE = Path(__file__).resolve().parent
XTB4 = HERE.parents[1] / "DFT_replaced_xTB4"
rows = list(csv.DictReader((XTB4 / "01_feature_table/xtb4_features.csv").open()))
feature_names = json.loads(
    (XTB4 / "01_feature_table/feature_groups.json").read_text()
)["physics_informed"]

def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan

y = np.asarray([number(row["band_gap_eV"]) for row in rows])
groups = np.asarray([row["family"] for row in rows])
X = np.asarray([[number(row.get(name)) for name in feature_names] for row in rows])
folds = list(GroupKFold(n_splits=5).split(X, y, groups))

def predict_oof(version):
    prediction = np.full(len(rows), np.nan)
    fold_ids = np.full(len(rows), -1, dtype=int)
    audit = []
    for fold, (train, test) in enumerate(folds):
        train_families, test_families = set(groups[train]), set(groups[test])
        assert not (train_families & test_families)
        assert not (set(train) & set(test))
        imputer = SimpleImputer(strategy="median")
        x_train, x_test = imputer.fit_transform(X[train]), imputer.transform(X[test])
        model = TabPFNRegressor.create_default_for_version(version)
        model.device = "cpu"
        model.n_estimators = 8
        model.show_progress_bar = False
        model.fit(x_train, y[train])
        prediction[test] = model.predict(x_test)
        fold_ids[test] = fold
        audit.append({"fold": fold, "n_train": len(train), "n_test": len(test),
                      "shared_families": len(train_families & test_families)})
    assert np.all(fold_ids >= 0)
    return prediction, fold_ids, audit

v2, fold_ids, audit = predict_oof(ModelVersion.V2)
v3, _, _ = predict_oof(ModelVersion.V3)

def score(prediction):
    return {"r2": float(r2_score(y, prediction)),
            "mae_eV": float(mean_absolute_error(y, prediction))}

metrics = {"cohort": {"n": len(rows), "n_families": len(set(groups)),
                       "feature_count_requested": len(feature_names),
                       "same_xTB4_features_and_folds": True},
           "fold_audit": audit,
           "results": {"tabpfn_v2": score(v2), "tabpfn_v3": score(v3)}}
(HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
with (HERE / "oof_predictions.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["COF", "family", "fold", "band_gap_eV", "tabpfn_v2_eV", "tabpfn_v3_eV"])
    for i, row in enumerate(rows):
        writer.writerow([row["COF"], row["family"], fold_ids[i], y[i], v2[i], v3[i]])
print(json.dumps(metrics, indent=2))
