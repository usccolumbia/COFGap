#!/usr/bin/env python3
"""Shared grouped-CV utilities for xTB3 fragment calibration."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "00_Zenodo_14855145" / "per_fragment_pbe0_labels.csv"
XTB_CACHE = ROOT.parent / "DFT_replaced_xTB2" / "04_rich_xTB_extraction" / "rich_xtb_cache.json"
HARTREE_EV = 27.211386245988

DESC_NAMES = ["heavy_atoms", "mol_wt", "tpsa", "logp", "aromatic_fraction",
              "rings", "aromatic_rings", "rotatable_bonds", "hbd", "hba",
              "formal_charge", "fraction_csp3", "n_count", "o_count",
              "b_count", "s_count"]


def descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return [np.nan] * len(DESC_NAMES)
    heavy = max(mol.GetNumHeavyAtoms(), 1)
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    return [heavy, Descriptors.MolWt(mol), rdMolDescriptors.CalcTPSA(mol),
            Crippen.MolLogP(mol), sum(a.GetIsAromatic() for a in mol.GetAtoms())/heavy,
            Lipinski.RingCount(mol), Lipinski.NumAromaticRings(mol),
            Lipinski.NumRotatableBonds(mol), Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol), Chem.GetFormalCharge(mol),
            rdMolDescriptors.CalcFractionCSP3(mol), atoms.count(7), atoms.count(8),
            atoms.count(5), atoms.count(16)]


def fingerprint(smiles, bits=512):
    mol = Chem.MolFromSmiles(smiles)
    arr = np.zeros(bits, dtype=np.int8)
    if mol is not None:
        fp = AllChem.GetMorganGenerator(radius=2, fpSize=bits).GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def grouped_oof(model, X, y, groups, n_splits=5):
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold
    pred = np.full(len(y), np.nan); folds = np.full(len(y), -1, int)
    cv = GroupKFold(min(n_splits, len(set(groups))))
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        fitted = clone(model).fit(X[tr], y[tr]); pred[te] = fitted.predict(X[te]); folds[te] = fold
    return pred, folds


def metrics(y, pred):
    from sklearn.metrics import r2_score, mean_absolute_error
    return {"r2": float(r2_score(y, pred)),
            "mae_eV": float(mean_absolute_error(y, pred)),
            "rmse_eV": float(np.sqrt(np.mean((np.asarray(y)-np.asarray(pred))**2)))}

