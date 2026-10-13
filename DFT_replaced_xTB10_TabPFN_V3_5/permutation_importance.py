#!/usr/bin/env python3
"""Out-of-fold permutation importance for TabPFN v3.5 (216 COFs, family-grouped 5-fold).

For each fold, the model is fit on the training families once; each feature is
then shuffled within the held-out fold (N_REPEATS times) and the pooled
out-of-fold R2 drop relative to the unshuffled pooled R2 is recorded.
"""
import csv, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

N_REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
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
usable = [j for j in range(X.shape[1]) if np.isfinite(X[:, j]).any()]
folds = list(GroupKFold(5).split(X, y, fam))
rng = np.random.default_rng(17)

models = []
base = np.zeros(len(y))
for tr, te in folds:
    imp = SimpleImputer(strategy="median").fit(X[tr])
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5)
    m.device = "cpu"; m.n_estimators = 8; m.show_progress_bar = False
    m.fit(imp.transform(X[tr]), y[tr])
    models.append((imp, m))
    base[te] = m.predict(imp.transform(X[te]))
r2_base = r2_score(y, base)
print(f"baseline pooled OOF R2 = {r2_base:.4f}", flush=True)

drops = {}
t0 = time.time()
for k, j in enumerate(usable):
    d = []
    for _ in range(N_REPEATS):
        pred = np.zeros(len(y))
        for (tr, te), (imp, m) in zip(folds, models):
            Xt = X[te].copy()
            Xt[:, j] = Xt[rng.permutation(len(te)), j]
            pred[te] = m.predict(imp.transform(Xt))
        d.append(r2_base - r2_score(y, pred))
    drops[names[j]] = (float(np.mean(d)), float(np.std(d)))
    if k % 10 == 0: print(k, names[j], drops[names[j]], f"{time.time()-t0:.0f}s", flush=True)
out = {"model": "TabPFN v3.5", "n": len(y), "n_repeats": N_REPEATS, "baseline_r2": r2_base,
       "importance_r2_drop": dict(sorted(drops.items(), key=lambda kv: -kv[1][0]))}
(HERE / "permutation_importance.json").write_text(json.dumps(out, indent=2))
for n, (mu, sd) in list(out["importance_r2_drop"].items())[:15]: print(f"{mu:.4f} ± {sd:.4f}  {n}")
