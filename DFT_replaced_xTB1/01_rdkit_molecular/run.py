#!/usr/bin/env python3
"""Experiment 01: structural + RDKit fragment descriptors/fingerprints."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import evaluate, load_rows, make_matrix

rows = load_rows()
X, names, valid = make_matrix(rows, include_molecular=True)
rows = [r for r, keep in zip(rows, valid) if keep]
X = X[valid]
evaluate(rows, X, names, Path(__file__).parent / "results", label="structure+rdkit")
