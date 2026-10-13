#!/usr/bin/env python3
"""Build a DFT-free COF band-gap feature table.

The original bandgap_CoRE_COF_v5.0_PCAdata.csv includes fragment electronic
features that are expensive/opaque for screening:
    Node_HL_gap(eV)
    Linker_HL_gap(eV)
    Node+Linker_HL_gap(eV)

This script excludes those columns and replaces them with cheaper, more
interpretable descriptors derived from composition counts, geometry, topology,
and the rule-based COF screener output.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


LINK_SCORE = {
    "vinylene": 1.0,
    "vinylene2": 1.0,
    "triazine_l": 0.9,
    "azine": 0.85,
    "imine": 0.8,
    "hydrazone": 0.55,
    "ketoenamine": 0.6,
    "amide_imide": 0.45,
    "boronate": 0.15,
    "boronate2": 0.15,
    "boroxine": 0.10,
    "none": 0.0,
    "?": 0.0,
}


def to_float(value, default=math.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def load_gaps(path: Path):
    rows = {}
    raw = path.read_text(encoding="utf-8-sig").splitlines()
    header = [item.strip() for item in raw[0].split(";")]
    for line in raw[1:]:
        if not line.strip():
            continue
        r = dict(zip(header, line.split(";")))
        idx = r["COF"].strip()
        rows[idx] = {
            "COF": idx,
            "band_gap_eV": to_float(r.get("band gap (eV)")),
            "volume_A3": to_float(r.get("volume")),
            "density_g_cm3": to_float(r.get("Density  (g/cm3)")),
            "sacc_m2_cm3": to_float(r.get("Sacc           (m2/cm-3)")),
            "n_count": to_float(r.get("N_count_comb_smiles"), 0.0),
            "b_count": to_float(r.get("B_count_comb_smiles"), 0.0),
            "o_count": to_float(r.get("O_count_comb_smiles"), 0.0),
        }
    return rows


def parse_flags(flags: str):
    out = {
        "conj": 0.0,
        "sp3": 0.0,
        "link_score": 0.0,
        "link_name": "none",
        "has_donor": 0,
        "has_acceptor": 0,
        "is_DA": 0,
        "has_metal": 0,
    }
    for token in flags.split(";"):
        token = token.strip()
        if token.startswith("conj:"):
            out["conj"] = to_float(token[5:], 0.0)
        elif token.startswith("sp3:"):
            out["sp3"] = to_float(token[4:], 0.0)
        elif token.startswith("link:"):
            link = token[5:]
            out["link_name"] = link
            out["link_score"] = LINK_SCORE.get(link, 0.0)
        elif token.startswith("don:"):
            out["has_donor"] = 1
        elif token.startswith("acc:"):
            out["has_acceptor"] = 1
        elif token == "D-A":
            out["is_DA"] = 1
        elif token.startswith("METAL"):
            out["has_metal"] = 1
    return out


def load_screener(path: Path):
    rows = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            idx = r["file"].replace(".cif", "").strip()
            row = parse_flags(r.get("flags", ""))
            row.update(
                {
                    "n_atoms": to_float(r.get("n_atoms")),
                    "density_screen_g_cm3": to_float(r.get("density_g_cm3")),
                    "volume_screen_A3": to_float(r.get("volume_A3")),
                    "layered": 1 if str(r.get("layered")).lower() == "yes" else 0,
                    "stack_len_A": to_float(r.get("stacking_len_A")),
                    "n_pbc_bonds_cut": to_float(r.get("n_pbc_bonds_cut")),
                    "n_unique_fragments": to_float(r.get("n_unique_fragments")),
                    "tier": r.get("tier", ""),
                    "pv_score": to_float(r.get("pv_score")),
                }
            )
            rows[idx] = row
    return rows


def add_derived(row):
    n_atoms = row.get("n_atoms", math.nan)
    volume = row.get("volume_A3", math.nan)
    n_count = row.get("n_count", 0.0)
    b_count = row.get("b_count", 0.0)
    o_count = row.get("o_count", 0.0)
    hetero_count = n_count + b_count + o_count

    row["hetero_count"] = hetero_count
    if math.isfinite(n_atoms) and n_atoms > 0:
        row["n_frac"] = n_count / n_atoms
        row["b_frac"] = b_count / n_atoms
        row["o_frac"] = o_count / n_atoms
        row["hetero_frac"] = hetero_count / n_atoms
        row["volume_per_atom_A3"] = volume / n_atoms if math.isfinite(volume) else math.nan
        row["pbc_bond_density"] = row.get("n_pbc_bonds_cut", math.nan) / n_atoms
    else:
        row["n_frac"] = math.nan
        row["b_frac"] = math.nan
        row["o_frac"] = math.nan
        row["hetero_frac"] = math.nan
        row["volume_per_atom_A3"] = math.nan
        row["pbc_bond_density"] = math.nan

    row["atom_density_A3_inv"] = (
        n_atoms / volume
        if math.isfinite(n_atoms) and math.isfinite(volume) and volume > 0
        else math.nan
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gaps-csv",
        default="bandgap_CoRE_COF_v5.0_PCAdata.csv",
        type=Path,
    )
    parser.add_argument("--screener-csv", default="pv_ranked_H.csv", type=Path)
    parser.add_argument(
        "--output",
        default="bandgap_CoRE_COF_v5.0_DFTfree_features.csv",
        type=Path,
    )
    args = parser.parse_args()

    gaps = load_gaps(args.gaps_csv)
    screener = load_screener(args.screener_csv)
    rows = []
    for idx, gap_row in gaps.items():
        if idx not in screener:
            continue
        row = {**gap_row, **screener[idx]}
        rows.append(add_derived(row))

    fieldnames = [
        "COF",
        "band_gap_eV",
        "volume_A3",
        "density_g_cm3",
        "sacc_m2_cm3",
        "n_atoms",
        "volume_per_atom_A3",
        "atom_density_A3_inv",
        "layered",
        "stack_len_A",
        "n_pbc_bonds_cut",
        "pbc_bond_density",
        "n_unique_fragments",
        "n_count",
        "b_count",
        "o_count",
        "hetero_count",
        "n_frac",
        "b_frac",
        "o_frac",
        "hetero_frac",
        "conj",
        "sp3",
        "link_name",
        "link_score",
        "has_donor",
        "has_acceptor",
        "is_DA",
        "has_metal",
        "tier",
        "pv_score",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} DFT-free feature rows to {args.output}")


if __name__ == "__main__":
    main()
