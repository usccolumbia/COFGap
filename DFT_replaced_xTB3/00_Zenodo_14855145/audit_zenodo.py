#!/usr/bin/env python3
"""Normalize and audit the Zenodo per-fragment PBE0 labels."""
from __future__ import annotations
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[1]
FRAGMENTS = DATA_ROOT / "DFT_cof_bandgap" / "fragments"
SOURCE = HERE / "COFsalldata_forplots.csv"

ROLE_COLUMNS = {
    "node": ("smile1", "Node_HOMO(ev)", "Node_LUMO(eV)", "Node_HL_gap(eV)"),
    "linker": ("smile2", "Linker_HOMO(ev)", "Linker_LUMO(eV)", "Linker_HL_gap(eV)"),
    "nodelinker": ("comb_smiles", "Node+Linker_HOMO(ev)",
                   "Node+Linker_LUMO(eV)", "Node+Linker_HL_gap(eV)"),
}


def canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule) if molecule is not None else smiles


def main():
    rows = list(csv.DictReader(SOURCE.open(encoding="utf-8-sig"), delimiter=";"))
    normalized = []
    for row in rows:
        cof = row["COF"].strip()
        for role, (smiles_col, homo_col, lumo_col, gap_col) in ROLE_COLUMNS.items():
            smiles = row[smiles_col].strip()
            canon = canonical(smiles)
            xyz = FRAGMENTS / cof / f"{role}.xyz"
            normalized.append({
                "COF": cof,
                "role": role,
                "smiles": smiles,
                "canonical_smiles": canon,
                "fragment_id": hashlib.sha256(canon.encode()).hexdigest()[:16],
                "pbe0_homo_eV": float(row[homo_col]),
                "pbe0_lumo_eV": float(row[lumo_col]),
                "pbe0_gap_eV": float(row[gap_col]),
                "xyz_path": str(xyz.resolve()) if xyz.exists() else "",
                "xyz_exists": int(xyz.exists()),
            })

    fields = list(normalized[0])
    with (HERE / "per_fragment_pbe0_labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(normalized)

    grouped = defaultdict(list)
    for row in normalized:
        grouped[(row["role"], row["canonical_smiles"])].append(row)
    unique_rows = []
    for (role, smiles), values in grouped.items():
        gaps = np.array([v["pbe0_gap_eV"] for v in values])
        homos = np.array([v["pbe0_homo_eV"] for v in values])
        lumos = np.array([v["pbe0_lumo_eV"] for v in values])
        unique_rows.append({
            "role": role, "canonical_smiles": smiles,
            "fragment_id": values[0]["fragment_id"], "n_occurrences": len(values),
            "pbe0_homo_eV_mean": homos.mean(), "pbe0_lumo_eV_mean": lumos.mean(),
            "pbe0_gap_eV_mean": gaps.mean(), "pbe0_gap_eV_std": gaps.std(),
            "pbe0_gap_eV_range": gaps.max() - gaps.min(),
        })
    with (HERE / "unique_fragment_pbe0_labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(unique_rows[0]))
        writer.writeheader(); writer.writerows(unique_rows)

    inventory = {}
    for role in ROLE_COLUMNS:
        subset = [r for r in normalized if r["role"] == role]
        unique = {(r["canonical_smiles"]) for r in subset}
        inventory[role] = {
            "label_occurrences": len(subset), "unique_role_fragments": len(unique),
            "xyz_available": sum(r["xyz_exists"] for r in subset),
        }
    inventory["all_roles"] = {
        "label_occurrences": len(normalized),
        "unique_role_and_smiles_pairs": len(grouped),
        "unique_smiles_across_roles": len({r["canonical_smiles"] for r in normalized}),
        "xyz_available": sum(r["xyz_exists"] for r in normalized),
    }
    (HERE / "inventory.json").write_text(json.dumps(inventory, indent=2))
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
