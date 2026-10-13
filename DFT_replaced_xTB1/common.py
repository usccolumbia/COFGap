#!/usr/bin/env python3
"""Shared data, features, and leakage-resistant grouped-CV evaluation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset_code"

STRUCTURE = [
    "density", "volume", "sacc", "n_atoms", "layered", "stack_len",
    "n_pbc_bonds_cut", "n_unique_fragments", "volume_per_atom",
    "pbc_bond_per_atom",
]


def _number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _read_gap_table():
    path = DATA / "bandgap_CoRE_COF_v5.0_PCAdata.csv"
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header = [x.strip() for x in lines[0].split(";")]
    out = {}
    for line in lines[1:]:
        row = dict(zip(header, line.split(";")))
        key = row["COF"].strip()
        out[key] = {
            "gap": _number(row.get("band gap (eV)")),
            "volume": _number(row.get("volume")),
            "density": _number(row.get("Density  (g/cm3)")),
            "sacc": _number(row.get("Sacc           (m2/cm-3)")),
            "n_count": _number(row.get("N_count_comb_smiles")),
            "b_count": _number(row.get("B_count_comb_smiles")),
            "o_count": _number(row.get("O_count_comb_smiles")),
        }
    return out


def _parse_link(flags):
    for token in (flags or "").split(";"):
        if token.startswith("link:"):
            return token[5:]
    return "none"


def _read_screener(path):
    out = {}
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = row["file"].replace(".cif", "").strip()
            out[key] = {
                "n_atoms": _number(row.get("n_atoms")),
                "layered": 1.0 if row.get("layered", "").lower() == "yes" else 0.0,
                "stack_len": _number(row.get("stacking_len_A")),
                "n_pbc_bonds_cut": _number(row.get("n_pbc_bonds_cut")),
                "n_unique_fragments": _number(row.get("n_unique_fragments")),
                "unit_smiles": row.get("unit_smiles", ""),
                "frag_smiles": row.get("frag_smiles", ""),
                "link": _parse_link(row.get("flags", "")),
                "xtb_gap": _number(row.get("xtb_frag_gap_eV")),
            }
    return out


def _family(row):
    b = "B" if np.nan_to_num(row["b_count"]) >= 1 else ""
    n = int(min(np.nan_to_num(row["n_count"]), 6))
    o = int(min(np.nan_to_num(row["o_count"]), 6))
    return f"{row['link']}|{b}N{n}O{o}"


def load_rows():
    gaps = _read_gap_table()
    base = _read_screener(DATA / "pv_ranked_H.csv")
    xtb = _read_screener(DATA / "pv_ranked_H_xtb.csv")
    rows = []
    for key, gap in gaps.items():
        if key not in base or not np.isfinite(gap["gap"]):
            continue
        row = {"id": key, **gap, **base[key]}
        row["xtb_gap"] = xtb.get(key, {}).get("xtb_gap", np.nan)
        n_atoms, volume = row["n_atoms"], row["volume"]
        row["volume_per_atom"] = volume / n_atoms if n_atoms > 0 else np.nan
        row["pbc_bond_per_atom"] = row["n_pbc_bonds_cut"] / n_atoms if n_atoms > 0 else np.nan
        row["xtb_missing"] = float(not np.isfinite(row["xtb_gap"]))
        row["family"] = _family(row)
        rows.append(row)
    return rows


MOLECULAR_NAMES = [
    "heavy_atoms", "mol_wt", "tpsa", "logp", "molar_refractivity",
    "aromatic_fraction", "ring_count", "aromatic_rings", "rotatable_bonds",
    "hbd", "hba", "formal_charge", "fraction_csp3", "hetero_fraction",
]


def _mol_descriptors(mol):
    heavy = mol.GetNumHeavyAtoms()
    aromatic = sum(a.GetIsAromatic() for a in mol.GetAtoms())
    hetero = sum(a.GetAtomicNum() not in (1, 6) for a in mol.GetAtoms())
    return [
        heavy, Descriptors.MolWt(mol), rdMolDescriptors.CalcTPSA(mol),
        Crippen.MolLogP(mol), Crippen.MolMR(mol), aromatic / max(heavy, 1),
        Lipinski.RingCount(mol), Lipinski.NumAromaticRings(mol),
        Lipinski.NumRotatableBonds(mol), Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol), Chem.GetFormalCharge(mol),
        rdMolDescriptors.CalcFractionCSP3(mol), hetero / max(heavy, 1),
    ]


def molecular_features(row, fp_bits=256):
    """Return size-ranked fragment descriptors and Morgan fingerprints."""
    mol = Chem.MolFromSmiles(row["frag_smiles"] or row["unit_smiles"])
    if mol is None:
        return None, None
    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    fragments.sort(key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
    fragments = fragments[:2]
    while len(fragments) < 2:
        fragments.append(None)
    values, names = [], []
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=fp_bits)
    for i, frag in enumerate(fragments, 1):
        prefix = f"fragment_{i}"
        if frag is None:
            values.extend([np.nan] * len(MOLECULAR_NAMES))
            values.extend([0.0] * fp_bits)
        else:
            values.extend(_mol_descriptors(frag))
            arr = np.zeros(fp_bits, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(frag), arr)
            values.extend(arr.tolist())
        names.extend([f"{prefix}_{x}" for x in MOLECULAR_NAMES])
        names.extend([f"{prefix}_morgan_{j}" for j in range(fp_bits)])
    return values, names


def make_matrix(rows, include_molecular=False, include_xtb=False):
    names = list(STRUCTURE)
    matrix = [[row[x] for x in STRUCTURE] for row in rows]
    valid = np.ones(len(rows), dtype=bool)
    if include_molecular:
        mol_values = []
        mol_names = None
        for i, row in enumerate(rows):
            values, feature_names = molecular_features(row)
            if values is None:
                valid[i] = False
                values = [np.nan] * (2 * (len(MOLECULAR_NAMES) + 256))
                feature_names = ([f"fragment_{k}_{x}" for k in (1, 2) for x in MOLECULAR_NAMES] +
                                 [f"fragment_{k}_morgan_{j}" for k in (1, 2) for j in range(256)])
            mol_values.append(values)
            mol_names = feature_names
        matrix = [a + b for a, b in zip(matrix, mol_values)]
        names += mol_names
    if include_xtb:
        for values, row in zip(matrix, rows):
            values.extend([row["xtb_gap"], row["xtb_missing"]])
        names += ["xtb_gap", "xtb_missing"]
    return np.asarray(matrix, dtype=float), names, valid


def estimators():
    return {
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            VarianceThreshold(),
            ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                max_features=0.7, random_state=17, n_jobs=-1),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            VarianceThreshold(),
            RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                  max_features=0.7, random_state=17, n_jobs=-1),
        ),
    }


def grouped_predictions(model, X, y, groups):
    cv = GroupKFold(n_splits=5)
    pred = np.full(len(y), np.nan)
    fold = np.full(len(y), -1, dtype=int)
    for fold_id, (train, test) in enumerate(cv.split(X, y, groups)):
        fitted = clone(model).fit(X[train], y[train])
        pred[test] = fitted.predict(X[test])
        fold[test] = fold_id
    return pred, fold


def _importance(model, names):
    imputer, selector, regressor = model.steps[-3][1], model.steps[-2][1], model.steps[-1][1]
    original = list(names) + [f"missing__{names[i]}" for i in imputer.indicator_.features_]
    kept = np.asarray(original)[selector.get_support()]
    return sorted(zip(kept, regressor.feature_importances_), key=lambda x: -x[1])


def evaluate(rows, X, names, out_dir, target=None, prediction_offset=None, label="direct"):
    out_dir.mkdir(parents=True, exist_ok=True)
    y_gap = np.array([r["gap"] for r in rows])
    y = y_gap if target is None else np.asarray(target)
    groups = np.array([r["family"] for r in rows])
    results, all_predictions = {}, {}
    for model_name, model in estimators().items():
        raw, fold = grouped_predictions(model, X, y, groups)
        pred = raw if prediction_offset is None else raw + np.asarray(prediction_offset)
        results[model_name] = {
            "r2": float(r2_score(y_gap, pred)),
            "mae_eV": float(mean_absolute_error(y_gap, pred)),
        }
        all_predictions[model_name] = pred
    best = min(results, key=lambda k: results[k]["mae_eV"])
    fitted = estimators()[best].fit(X, y)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump({"label": label, "n": len(rows), "n_families": len(set(groups)),
                   "n_features": X.shape[1], "best_model": best,
                   "models": results}, f, indent=2)
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["COF", "family", "fold", "dft_gap_eV", *all_predictions])
        for i, row in enumerate(rows):
            writer.writerow([row["id"], row["family"], int(fold[i]), y_gap[i],
                             *[all_predictions[m][i] for m in all_predictions]])
    with (out_dir / "feature_importance.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["feature", "importance"])
        writer.writerows(_importance(fitted, names))
    return results
