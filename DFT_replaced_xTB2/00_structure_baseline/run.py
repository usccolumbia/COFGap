#!/usr/bin/env python3
"""Experiment 00: common structural-only control for fair comparisons."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import evaluate, load_rows, make_matrix

rows = load_rows()
X, names, _ = make_matrix(rows)
evaluate(rows, X, names, Path(__file__).parent / "results", label="structure_only")
