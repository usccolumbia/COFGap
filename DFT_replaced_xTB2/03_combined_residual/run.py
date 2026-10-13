#!/usr/bin/env python3
"""Experiment 03: combined direct model and xTB-to-DFT residual model."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import evaluate, load_rows, make_matrix

here = Path(__file__).parent
rows = load_rows()
X, names, valid_mol = make_matrix(rows, include_molecular=True, include_xtb=True)
rows = [r for r, keep in zip(rows, valid_mol) if keep]
X = X[valid_mol]
evaluate(rows, X, names, here / "results_direct", label="combined_direct")

complete = np.array([np.isfinite(r["xtb_gap"]) for r in rows])
residual_rows = [r for r, keep in zip(rows, complete) if keep]
xtb = np.array([r["xtb_gap"] for r in residual_rows])
target = np.array([r["gap"] for r in residual_rows]) - xtb
evaluate(residual_rows, X[complete], names, here / "results_residual",
         target=target, prediction_offset=xtb, label="dft_minus_xtb_residual")
