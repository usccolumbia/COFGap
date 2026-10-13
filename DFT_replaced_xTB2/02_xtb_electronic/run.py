#!/usr/bin/env python3
"""Experiment 02: structural + existing inexpensive xTB fragment gap."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import evaluate, load_rows, make_matrix

here = Path(__file__).parent
rows = load_rows()
X, names, _ = make_matrix(rows, include_xtb=True)
evaluate(rows, X, names, here / "results_all_rows", label="structure+xtb_with_missing_indicator")
complete = np.array([np.isfinite(r["xtb_gap"]) for r in rows])
complete_rows = [r for r, keep in zip(rows, complete) if keep]
evaluate(complete_rows, X[complete], names, here / "results_complete_cases",
         label="structure+xtb_complete_cases")
(here / "coverage.json").write_text(json.dumps({"labeled_rows": len(rows),
    "finite_xtb_rows": int(complete.sum()), "coverage": float(complete.mean())}, indent=2))
