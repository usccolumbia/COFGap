#!/usr/bin/env python3
"""Evaluate rich-xTB feature groups without any DFT-derived inputs."""
from __future__ import annotations
import csv
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
from common import STRUCTURE, evaluate, load_rows, make_matrix

RICH = HERE.parents[0] / "04_rich_xTB_extraction" / "rich_xtb_features.csv"


def number(value):
    try: return float(value)
    except (TypeError, ValueError): return np.nan


rich_rows = list(csv.DictReader(RICH.open(encoding="utf-8-sig")))
rich = {r["file"].replace(".cif", ""): r for r in rich_rows}
feature_names = [c for c in rich_rows[0] if c not in ("file", "xtb_status")]
rows = [r for r in load_rows()
        if r["id"] in rich and rich[r["id"]]["xtb_status"] == "OK"]
derived_names = [
    "derived_homo_node_minus_linker_eV", "derived_lumo_node_minus_linker_eV",
    "derived_gap_node_minus_linker_eV", "derived_ip_node_minus_linker_eV",
    "derived_ea_node_minus_linker_eV", "derived_frontier_cross_gap_eV",
    "derived_min_fragment_gap_eV", "derived_max_fragment_gap_eV",
    "derived_nodelinker_gap_minus_min_fragment_eV",
]


def derived(record):
    value = lambda fragment, prop: number(record.get(f"{fragment}_xtb_{prop}"))
    node_gap, linker_gap = value("node", "gap_eV"), value("linker", "gap_eV")
    return [
        value("node", "homo_eV") - value("linker", "homo_eV"),
        value("node", "lumo_eV") - value("linker", "lumo_eV"),
        node_gap - linker_gap,
        value("node", "vertical_ip_eV") - value("linker", "vertical_ip_eV"),
        value("node", "vertical_ea_eV") - value("linker", "vertical_ea_eV"),
        min(value("node", "lumo_eV"), value("linker", "lumo_eV")) -
            max(value("node", "homo_eV"), value("linker", "homo_eV")),
        min(node_gap, linker_gap), max(node_gap, linker_gap),
        value("nodelinker", "gap_eV") - min(node_gap, linker_gap),
    ]


X_rich = np.array([
    [number(rich[r["id"]].get(c)) for c in feature_names] + derived(rich[r["id"]])
    for r in rows
])
feature_names += derived_names
evaluate(rows, X_rich, feature_names, HERE / "results_rich_only", label="rich_xtb_only")

X_structure, structure_names, _ = make_matrix(rows)
X_sr = np.column_stack([X_structure, X_rich])
evaluate(rows, X_sr, structure_names + feature_names,
         HERE / "results_rich_structure", label="rich_xtb+structure")

X_all, all_names, valid = make_matrix(rows, include_molecular=True)
rows_valid = [r for r, keep in zip(rows, valid) if keep]
evaluate(rows_valid, np.column_stack([X_all[valid], X_rich[valid]]), all_names + feature_names,
         HERE / "results_rich_structure_rdkit", label="rich_xtb+structure+rdkit")
