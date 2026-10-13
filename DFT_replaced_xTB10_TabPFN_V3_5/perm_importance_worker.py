#!/usr/bin/env python3
"""One feature's out-of-fold permutation importance for TabPFN v3.5 (Slurm array worker).

usage: perm_importance_worker.py DATA_DIR OUT_DIR FEATURE_INDEX N_REPEATS
Fits one v3.5 model per family-grouped fold (same GroupKFold(5) folds as run.py),
shuffles feature FEATURE_INDEX within each held-out fold N_REPEATS times, and
writes the drop in pooled out-of-fold R2. Seed = 17 + FEATURE_INDEX.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

data, out, j, reps = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rows = list(csv.DictReader((data / "xtb4_features.csv").open()))
names = json.loads((data / "feature_groups.json").read_text())["physics_informed"]
def num(v):
    try: return float(v)
    except (TypeError, ValueError): return np.nan
y = np.array([num(r["band_gap_eV"]) for r in rows])
fam = np.array([r["family"] for r in rows])
X = np.array([[num(r.get(n)) for n in names] for r in rows])
res = {"index": j, "feature": names[j], "n_repeats": reps}
if not np.isfinite(X[:, j]).any():
    res["skipped"] = "all values missing"
else:
    folds = list(GroupKFold(5).split(X, y, fam))
    models, base = [], np.zeros(len(y))
    for tr, te in folds:
        imp = SimpleImputer(strategy="median").fit(X[tr])
        m = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5)
        m.device = "cpu"; m.n_estimators = 8; m.show_progress_bar = False
        m.fit(imp.transform(X[tr]), y[tr]); models.append((imp, m))
        base[te] = m.predict(imp.transform(X[te]))
    r2b = r2_score(y, base)
    rng = np.random.default_rng(17 + j)
    drops = []
    for _ in range(reps):
        pred = np.zeros(len(y))
        for (tr, te), (imp, m) in zip(folds, models):
            Xt = X[te].copy(); Xt[:, j] = Xt[rng.permutation(len(te)), j]
            pred[te] = m.predict(imp.transform(Xt))
        drops.append(r2b - r2_score(y, pred))
    res.update(baseline_r2=float(r2b), drops=[float(d) for d in drops],
               mean=float(np.mean(drops)), std=float(np.std(drops)))
(out / f"feat_{j:03d}.json").write_text(json.dumps(res))
print(json.dumps({k: res.get(k) for k in ("index", "feature", "mean", "baseline_r2", "skipped")}))
