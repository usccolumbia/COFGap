#!/usr/bin/env python3
"""Merge Slurm worker outputs (perm_importance_parts/feat_*.json) into permutation_importance.json."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
parts = [json.loads(p.read_text()) for p in sorted((HERE / "perm_importance_parts").glob("feat_*.json"))]
used = [p for p in parts if "skipped" not in p]
base = [p["baseline_r2"] for p in used]
assert max(base) - min(base) < 1e-4, (min(base), max(base))  # CPU-dependent float noise only
imp = {p["feature"]: (p["mean"], p["std"]) for p in used}
out = {"model": "TabPFN v3.5", "n": 216, "n_repeats": used[0]["n_repeats"], "baseline_r2": used[0]["baseline_r2"],
       "n_features_scored": len(used), "skipped": [p["feature"] for p in parts if "skipped" in p],
       "importance_r2_drop": dict(sorted(imp.items(), key=lambda kv: -kv[1][0]))}
(HERE / "permutation_importance.json").write_text(json.dumps(out, indent=2))
print(len(parts), "parts;", len(used), "scored; baseline", out["baseline_r2"])
for n, (m, s) in list(out["importance_r2_drop"].items())[:15]: print(f"{m:.4f} ± {s:.4f}  {n}")
