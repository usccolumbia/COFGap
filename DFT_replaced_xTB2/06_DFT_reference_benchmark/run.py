#!/usr/bin/env python3
"""Fair DFT-feature reference on exactly the rich-xTB complete-case rows."""
from __future__ import annotations
import csv
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = PROJECT.parent / "dataset_code"
sys.path.insert(0, str(PROJECT))
from common import evaluate, load_rows, make_matrix


def number(value):
    try: return float(value)
    except (TypeError, ValueError): return np.nan


lines = (DATA / "bandgap_CoRE_COF_v5.0_PCAdata.csv").read_text(
    encoding="utf-8-sig").splitlines()
header = [x.strip() for x in lines[0].split(";")]
dft = {}
for line in lines[1:]:
    row = dict(zip(header, line.split(";")))
    dft[row["COF"].strip()] = [
        number(row.get("Node_HL_gap(eV)")),
        number(row.get("Linker_HL_gap(eV)")),
        number(row.get("Node+Linker_HL_gap(eV)")),
    ]

rich_path = PROJECT / "04_rich_xTB_extraction" / "rich_xtb_features.csv"
rich = {r["file"].replace(".cif", ""): r
        for r in csv.DictReader(rich_path.open(encoding="utf-8-sig"))}
rows = [r for r in load_rows() if r["id"] in rich and
        rich[r["id"]]["xtb_status"] == "OK"]
dft_names = ["DFT_node_gap_eV", "DFT_linker_gap_eV", "DFT_nodelinker_gap_eV"]
X_dft = np.asarray([dft[r["id"]] for r in rows], dtype=float)
evaluate(rows, X_dft, dft_names, HERE / "results_DFT_only", label="DFT_reference_only")

X_structure, structure_names, _ = make_matrix(rows)
evaluate(rows, np.column_stack([X_structure, X_dft]), structure_names + dft_names,
         HERE / "results_DFT_structure", label="DFT_reference+structure")
