#!/usr/bin/env python3
"""
cof_pv_model.py
===============
Turn the rule-based PV screener into a validated, learned baseline using the
deCOFpose 315-COF band gaps as ground truth.

Two models, each with a feature-set ABLATION so you can quantify what the
structural/topological descriptors add over fragment electronics alone
(the central claim of your COF band-gap paper):

  MODEL 1 - PV-window classifier (logistic regression)
      target: band gap in [1.5, 2.5] eV  (PV-relevant, exciton-corrected)
      reports precision/recall/ROC-AUC vs. the hand-tuned rule tiers.

  MODEL 2 - band-gap regression (ridge + random forest)
      target: DFT band gap (eV)
      feature sets compared:
        (A) atom/heteroatom composition only
        (S) structure/topology only    -> density, Sacc, volume, stacking
        (C) screener chemistry only    -> conjugation, linkage, D/A flags
        (A+S), (A+S+C) all combined

This DFT-free version intentionally excludes fragment electronic descriptors:
Node_HL_gap(eV), Linker_HL_gap(eV), and Node+Linker_HL_gap(eV).
      cross-validated R2 / MAE, plus RF feature importances.

Inputs (positional):
  gaps_csv     deCOFpose PCA CSV (semicolon-delimited: COF; ...; band gap ...)
  screener_csv pv_ranked_H.csv from pv_cof_screen.py (H-cap recommended)

Usage:
  python cof_pv_model.py bandgap_CoRE_COF_v5_0_PCAdata.csv pv_ranked_H.csv \
         --out-dir model_out

Dependencies: numpy, scikit-learn, matplotlib
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (cross_val_predict, StratifiedKFold,
                                     KFold, GroupKFold)
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                             average_precision_score, r2_score,
                             mean_absolute_error, precision_score,
                             recall_score)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PV_LO, PV_HI = 1.5, 2.5

# Linkage conjugation-retention scores (mirror the screener's ranking)
LINK_SCORE = {
    "vinylene": 1.0, "vinylene2": 1.0, "triazine_l": 0.9, "azine": 0.85,
    "imine": 0.8, "hydrazone": 0.55, "ketoenamine": 0.6, "amide_imide": 0.45,
    "boronate": 0.15, "boronate2": 0.15, "boroxine": 0.10, "none": 0.0,
    "?": 0.0,
}


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_gaps(path):
    raw = open(path, encoding="utf-8-sig").read().splitlines()
    hdr = [h.strip() for h in raw[0].split(";")]
    rows = {}
    for line in raw[1:]:
        if not line.strip():
            continue
        d = dict(zip(hdr, line.split(";")))
        idx = d["COF"].strip()
        def num(k):
            try:
                return float(d[k])
            except Exception:
                return np.nan
        rows[idx] = {
            "gap": num("band gap (eV)"),
            "volume": num("volume"),
            "density": num("Density  (g/cm3)"),
            "sacc": num("Sacc           (m2/cm-3)"),
            "n_count": num("N_count_comb_smiles"),
            "b_count": num("B_count_comb_smiles"),
            "o_count": num("O_count_comb_smiles"),
        }
    return rows


def parse_flags(flags):
    """Extract numeric chemistry features from the screener flag string."""
    f = {"conj": 0.0, "sp3": 0.0, "link_score": 0.0, "_link_name": "none",
         "has_donor": 0, "has_acceptor": 0, "is_DA": 0, "has_metal": 0}
    for tok in flags.split(";"):
        tok = tok.strip()
        if tok.startswith("conj:"):
            f["conj"] = float(tok[5:])
        elif tok.startswith("sp3:"):
            f["sp3"] = float(tok[4:])
        elif tok.startswith("link:"):
            f["link_score"] = LINK_SCORE.get(tok[5:], 0.0)
            f["_link_name"] = tok[5:]
        elif tok.startswith("don:"):
            f["has_donor"] = 1
        elif tok.startswith("acc:"):
            f["has_acceptor"] = 1
        elif tok == "D-A":
            f["is_DA"] = 1
        elif tok.startswith("METAL"):
            f["has_metal"] = 1
    return f


def load_screener(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        idx = r["file"].replace(".cif", "").strip()
        feats = parse_flags(r.get("flags", ""))
        try:
            feats["pv_score"] = float(r.get("pv_score", "nan"))
        except Exception:
            feats["pv_score"] = np.nan
        feats["tier"] = r.get("tier", "C")
        feats["layered"] = 1 if str(r.get("layered")).lower() == "yes" else 0
        for out_key, csv_key in [
            ("n_atoms", "n_atoms"),
            ("density_screen", "density_g_cm3"),
            ("volume_screen", "volume_A3"),
            ("n_pbc_bonds_cut", "n_pbc_bonds_cut"),
            ("n_unique_fragments", "n_unique_fragments"),
        ]:
            try:
                feats[out_key] = float(r.get(csv_key, "nan"))
            except Exception:
                feats[out_key] = np.nan
        try:
            feats["stack_len"] = float(r.get("stacking_len_A", "nan"))
        except Exception:
            feats["stack_len"] = np.nan
        rows[idx] = feats
    return rows


# ----------------------------------------------------------------------
# Feature assembly
# ----------------------------------------------------------------------
FEATURE_SETS = {
    "A_atomic": ["n_count", "b_count", "o_count", "hetero_count",
                 "n_frac", "b_frac", "o_frac", "hetero_frac"],
    "S_structure": ["density", "sacc", "volume", "n_atoms",
                    "volume_per_atom", "atom_density", "layered", "stack_len",
                    "n_pbc_bonds_cut", "pbc_bond_density",
                    "n_unique_fragments"],
    "C_chemistry": ["conj", "sp3", "link_score",
                    "has_donor", "has_acceptor", "is_DA", "has_metal"],
}
FEATURE_SETS["A+S"] = FEATURE_SETS["A_atomic"] + FEATURE_SETS["S_structure"]
FEATURE_SETS["A+S+C"] = (FEATURE_SETS["A_atomic"]
                         + FEATURE_SETS["S_structure"]
                         + FEATURE_SETS["C_chemistry"])
# Minimal cheap model: the two dominant non-DFT physical predictors.
FEATURE_SETS["minimal(stack+density)"] = ["stack_len", "density"]


def family_group(merged):
    """
    Coarse COF-family key for grouped CV: linkage class x heteroatom profile.
    Groups COFs that share both linkage chemistry and (B,N,O) composition
    bucket, so train/test folds don't leak near-identical frameworks.
    """
    link = merged.get("_link_name", "none")
    b = "B" if merged.get("b_count", 0) >= 1 else ""
    n = f"N{int(min(merged.get('n_count', 0), 6))}"
    o = f"O{int(min(merged.get('o_count', 0), 6))}"
    return f"{link}|{b}{n}{o}"


def build_matrix(gaps, scr, cols):
    X, y, tiers, pvs, ids, grps = [], [], [], [], [], []
    for idx, g in gaps.items():
        if idx not in scr or np.isnan(g["gap"]):
            continue
        merged = {**g, **scr[idx]}
        n_atoms = merged.get("n_atoms", np.nan)
        volume = merged.get("volume", np.nan)
        n_count = merged.get("n_count", 0.0)
        b_count = merged.get("b_count", 0.0)
        o_count = merged.get("o_count", 0.0)
        hetero_count = n_count + b_count + o_count
        merged["hetero_count"] = hetero_count
        if np.isfinite(n_atoms) and n_atoms > 0:
            merged["n_frac"] = n_count / n_atoms
            merged["b_frac"] = b_count / n_atoms
            merged["o_frac"] = o_count / n_atoms
            merged["hetero_frac"] = hetero_count / n_atoms
            merged["volume_per_atom"] = volume / n_atoms if np.isfinite(volume) else np.nan
            merged["pbc_bond_density"] = merged.get("n_pbc_bonds_cut", np.nan) / n_atoms
        else:
            merged["n_frac"] = np.nan
            merged["b_frac"] = np.nan
            merged["o_frac"] = np.nan
            merged["hetero_frac"] = np.nan
            merged["volume_per_atom"] = np.nan
            merged["pbc_bond_density"] = np.nan
        merged["atom_density"] = n_atoms / volume if np.isfinite(n_atoms) and np.isfinite(volume) and volume > 0 else np.nan
        row = [merged.get(c, np.nan) for c in cols]
        X.append(row)
        y.append(g["gap"])
        tiers.append(scr[idx]["tier"])
        pvs.append(scr[idx].get("pv_score", np.nan))
        ids.append(idx)
        grps.append(family_group(merged))
    X = np.array(X, float)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X, np.array(y), np.array(tiers), np.array(pvs), ids, np.array(grps)


# ----------------------------------------------------------------------
# Model 1: PV-window classifier
# ----------------------------------------------------------------------
def run_classifier(gaps, scr, out_dir):
    cols = FEATURE_SETS["A+S+C"]
    X, y, tiers, pvs, ids, grps = build_matrix(gaps, scr, cols)
    label = ((y >= PV_LO) & (y <= PV_HI)).astype(int)
    print(f"\n{'='*60}\nMODEL 1  PV-window classifier  (target: {PV_LO}-{PV_HI} eV)")
    print(f"{'='*60}")
    print(f"n={len(y)}  positives={label.sum()} ({100*label.mean():.0f}%)")

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    proba = cross_val_predict(clf, X, label, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    # honest grouped-CV AUC (whole COF families held out)
    gproba = cross_val_predict(clf, X, label, cv=GroupKFold(5),
                               groups=grps, method="predict_proba")[:, 1]

    auc = roc_auc_score(label, proba)
    gauc = roc_auc_score(label, gproba)
    ap = average_precision_score(label, proba)
    prec = precision_score(label, pred)
    rec = recall_score(label, pred)
    print(f"\nLearned logistic:")
    print(f"  random-CV ROC-AUC={auc:.3f}  grouped-CV ROC-AUC={gauc:.3f}  "
          f"(honest)")
    print(f"  AP={ap:.3f}  precision={prec:.2f} recall={rec:.2f}")

    # Baseline: rule tiers A+B
    rule_pred = np.array([t in ("A", "B") for t in tiers]).astype(int)
    print(f"\nRule tiers A+B (hand-tuned baseline):")
    print(f"  precision={precision_score(label, rule_pred):.2f} "
          f"recall={recall_score(label, rule_pred):.2f}")
    if not np.isnan(pvs).all():
        print(f"  rule pv_score as ranker: ROC-AUC="
              f"{roc_auc_score(label, np.nan_to_num(pvs)):.3f}")

    # Coefficients (fit on all data for interpretability)
    clf.fit(X, label)
    coef = clf.named_steps["logisticregression"].coef_[0]
    print("\nLogistic coefficients (standardized; +drivestoward PV window):")
    for c, w in sorted(zip(cols, coef), key=lambda x: -abs(x[1])):
        print(f"    {c:14s} {w:+.3f}")

    # PR curve plot
    p, r, _ = precision_recall_curve(label, proba)
    plt.figure(figsize=(5, 5))
    plt.plot(r, p, label=f"learned (AP={ap:.2f})", lw=2)
    plt.scatter([recall_score(label, rule_pred)],
                [precision_score(label, rule_pred)],
                c="red", zorder=5, label="rule A+B")
    plt.xlabel("recall"); plt.ylabel("precision")
    plt.title("PV-window detection"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "model1_pr_curve.png", dpi=130)
    plt.close()
    return {"auc": auc, "ap": ap}


# ----------------------------------------------------------------------
# Model 2: gap regression with ablation
# ----------------------------------------------------------------------
def run_regression(gaps, scr, out_dir):
    print(f"\n{'='*60}\nMODEL 2  band-gap regression  (feature-set ablation)")
    print(f"{'='*60}")
    cv = KFold(5, shuffle=True, random_state=0)
    gkf = GroupKFold(n_splits=5)
    results = {}
    order = ["A_atomic", "S_structure", "C_chemistry", "A+S", "A+S+C",
             "minimal(stack+density)"]
    print("\nCV type 'rand' leaks shared building blocks across folds and")
    print("inflates R2; 'grp' holds out whole COF families (honest estimate).")
    print(f"\n{'feature set':22s} {'model':6s} {'R2rand':>7s} {'R2grp':>7s} "
          f"{'MAEgrp':>7s}  n")
    print("-" * 60)
    for name in order:
        cols = FEATURE_SETS[name]
        X, y, tiers, pvs, ids, grps = build_matrix(gaps, scr, cols)
        for mname, mk in [
            ("ridge", lambda: make_pipeline(StandardScaler(), RidgeCV())),
            ("rf", lambda: RandomForestRegressor(n_estimators=300,
                                                 random_state=0, n_jobs=-1)),
        ]:
            pr = cross_val_predict(mk(), X, y, cv=cv)
            pg = cross_val_predict(mk(), X, y, cv=gkf, groups=grps)
            r2r = r2_score(y, pr)
            r2g = r2_score(y, pg); maeg = mean_absolute_error(y, pg)
            results[(name, mname)] = (r2r, r2g, maeg)
            print(f"{name:22s} {mname:6s} {r2r:7.3f} {r2g:7.3f} {maeg:7.3f}  "
                  f"{len(y)}")
    print(f"\n({len(set(grps))} COF families for grouped CV)")

    # RF feature importances on full DFT-free model
    cols = FEATURE_SETS["A+S+C"]
    X, y, *_ = build_matrix(gaps, scr, cols)
    rf = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1)
    rf.fit(X, y)
    print("\nRF feature importances (A+S+C, DFT-free, fit on all data):")
    for c, imp in sorted(zip(cols, rf.feature_importances_),
                         key=lambda x: -x[1]):
        print(f"    {c:14s} {imp:.3f}")

    # Bar chart: random vs grouped R2 for the RF model (the honest comparison)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(order))
    r2_rand = [results[(n, "rf")][0] for n in order]
    r2_grp = [results[(n, "rf")][1] for n in order]
    ax.bar(x - 0.2, r2_rand, 0.4, label="RF random-CV (optimistic)",
           color="#bbbbbb")
    ax.bar(x + 0.2, r2_grp, 0.4, label="RF grouped-CV (honest)",
           color="#ee6677")
    ax.axhline(0.22, ls="--", c="gray",
               label="paper node+linker (R2=0.22)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("cross-validated R$^2$")
    ax.set_title("Band-gap predictability by feature set")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "model2_ablation.png", dpi=130)
    plt.close()

    # scatter of best model (grouped-CV predictions = honest)
    cols = FEATURE_SETS["A+S+C"]
    X, y, tiers, pvs, ids, grps = build_matrix(gaps, scr, cols)
    pred = cross_val_predict(
        RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
        X, y, cv=GroupKFold(5), groups=grps)
    plt.figure(figsize=(5, 5))
    plt.scatter(y, pred, s=22, alpha=0.6, edgecolor="none")
    lims = [0, max(y.max(), pred.max()) + 0.3]
    plt.plot(lims, lims, "k--", lw=1)
    plt.axvspan(PV_LO, PV_HI, color="#2166ac", alpha=0.08)
    plt.xlabel("DFT band gap (eV)"); plt.ylabel("predicted (RF, grouped-CV)")
    plt.title(f"A+S+C RF (honest): R2={results[('A+S+C','rf')][1]:.2f} "
              f"MAE={results[('A+S+C','rf')][2]:.2f} eV")
    plt.tight_layout()
    plt.savefig(out_dir / "model2_parity.png", dpi=130)
    plt.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gaps_csv")
    ap.add_argument("screener_csv")
    ap.add_argument("--out-dir", default="model_out")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(exist_ok=True)

    gaps = load_gaps(args.gaps_csv)
    scr = load_screener(args.screener_csv)
    print(f"loaded {len(gaps)} gaps, {len(scr)} screener rows")

    run_classifier(gaps, scr, out)
    run_regression(gaps, scr, out)
    print(f"\nPlots written to {out}/")


if __name__ == "__main__":
    main()
