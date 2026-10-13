#!/usr/bin/env python3
"""
cof_pv_model.py  --  train / screen / analyze for COF photovoltaic band gaps
============================================================================

THREE MODES
-----------
1) TRAIN  -> fit a band-gap regressor on labeled COFs, save a model file.
     python cof_pv_model.py train \
         --gaps bandgap_CoRE_COF_v5_0_PCAdata.csv \
         --screener pv_ranked_H.csv \
         --features deployable \
         --model-out pv_model.joblib

2) SCREEN -> run raw CIFs through the whole pipeline, predict gaps, and copy
             the PV-window candidates into an output folder.
     python cof_pv_model.py screen \
         --dir mycof_cifs --outdir filtered_pv_cifs \
         --model pv_model.joblib

3) ANALYZE -> the feature-set ablation / calibration study (no model saved).
     python cof_pv_model.py analyze --gaps ... --screener ...

DEPLOYABILITY NOTE (important)
------------------------------
The single strongest predictor of a COF band gap in this dataset is the
node+linker HOMO-LUMO gap (`nl_hl`), which requires a *fragment DFT*
calculation and CANNOT be obtained from a bare CIF. Feature sets:

  * 'deployable'  - only features computable from a CIF (+SMILES): density,
                    volume, stacking length, layering, conjugation, linkage,
                    donor/acceptor, heteroatom counts. Honest grouped-CV
                    R2 ~ 0.28, MAE ~ 0.6 eV. This is what SCREEN mode can run
                    with no DFT.
  * 'deployable+xtb' - adds a GFN2-xTB fragment gap (seconds/COF) as a cheap
                    proxy for nl_hl. Needs `tblite`. Recovers much of the gap.
  * 'full'        - adds nl_hl and other fragment-DFT features. Best accuracy
                    (grouped R2 ~ 0.6) but SCREEN mode can only use it if you
                    supply precomputed fragment gaps; not runnable on raw CIFs.

TRAIN records which features the model needs; SCREEN checks availability and
warns/aborts if a required feature can't be produced for new CIFs.

Dependencies: numpy, scikit-learn, joblib, rdkit, pymatgen, matplotlib
              (tblite only for the xtb option). Requires cof_cif2smiles.py and
              pv_cof_screen.py in the same directory (imported for SCREEN mode).
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, GroupKFold, KFold
from sklearn.metrics import r2_score, mean_absolute_error

PV_LO, PV_HI = 1.5, 2.5

LINK_SCORE = {
    "vinylene": 1.0, "vinylene2": 1.0, "triazine_l": 0.9, "azine": 0.85,
    "imine": 0.8, "hydrazone": 0.55, "ketoenamine": 0.6, "amide_imide": 0.45,
    "boronate": 0.15, "boronate2": 0.15, "boroxine": 0.10, "none": 0.0, "?": 0.0,
}

# Feature groups
FRAGMENT_DFT = ["node_hl", "linker_hl", "nl_hl"]          # needs ORCA
HETERO = ["n_count", "b_count", "o_count"]                # from SMILES
STRUCT_CIF = ["density", "volume", "layered", "stack_len"]  # from CIF
STRUCT_PORE = ["sacc"]                                    # needs Zeo++
CHEM = ["conj", "sp3", "link_score", "has_donor",
        "has_acceptor", "is_DA", "has_metal"]             # from SMILES
XTB = ["xtb_frag_gap"]                                    # optional GFN2

FEATURE_SETS = {
    "deployable":      STRUCT_CIF + CHEM + HETERO,
    "deployable+xtb":  STRUCT_CIF + CHEM + HETERO + XTB,
    "full":            FRAGMENT_DFT + STRUCT_CIF + STRUCT_PORE + CHEM + HETERO,
    "minimal":         ["stack_len", "nl_hl"],
}


# ----------------------------------------------------------------------
# Feature parsing shared by all modes
# ----------------------------------------------------------------------
def parse_flags(flags):
    f = {"conj": 0.0, "sp3": 0.0, "link_score": 0.0, "_link_name": "none",
         "has_donor": 0, "has_acceptor": 0, "is_DA": 0, "has_metal": 0}
    for tok in (flags or "").split(";"):
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


def family_group(merged):
    link = merged.get("_link_name", "none")
    b = "B" if merged.get("b_count", 0) >= 1 else ""
    n = f"N{int(min(merged.get('n_count', 0) or 0, 6))}"
    o = f"O{int(min(merged.get('o_count', 0) or 0, 6))}"
    return f"{link}|{b}{n}{o}"


# ----------------------------------------------------------------------
# Loaders for TRAIN / ANALYZE (from the two CSVs)
# ----------------------------------------------------------------------
def load_gaps(path):
    raw = open(path, encoding="utf-8-sig").read().splitlines()
    hdr = [h.strip() for h in raw[0].split(";")]
    rows = {}
    for line in raw[1:]:
        if not line.strip():
            continue
        d = dict(zip(hdr, line.split(";")))
        def num(k):
            try:
                return float(d[k])
            except Exception:
                return np.nan
        rows[d["COF"].strip()] = {
            "gap": num("band gap (eV)"), "volume": num("volume"),
            "density": num("Density  (g/cm3)"),
            "sacc": num("Sacc           (m2/cm-3)"),
            "node_hl": num("Node_HL_gap(eV)"),
            "linker_hl": num("Linker_HL_gap(eV)"),
            "nl_hl": num("Node+Linker_HL_gap(eV)"),
            "n_count": num("N_count_comb_smiles"),
            "b_count": num("B_count_comb_smiles"),
            "o_count": num("O_count_comb_smiles"),
        }
    return rows


def load_screener(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        idx = r["file"].replace(".cif", "").strip()
        feats = parse_flags(r.get("flags", ""))
        feats["layered"] = 1 if str(r.get("layered")).lower() == "yes" else 0
        for k, col in [("stack_len", "stacking_len_A"),
                       ("xtb_frag_gap", "xtb_frag_gap_eV")]:
            try:
                feats[k] = float(r.get(col, "nan"))
            except Exception:
                feats[k] = np.nan
        feats["tier"] = r.get("tier", "C")
        rows[idx] = feats
    return rows


def build_matrix(gaps, scr, cols):
    X, y, grps, ids = [], [], [], []
    for idx, g in gaps.items():
        if idx not in scr or np.isnan(g["gap"]):
            continue
        merged = {**g, **scr[idx]}
        X.append([merged.get(c, np.nan) for c in cols])
        y.append(g["gap"])
        grps.append(family_group(merged))
        ids.append(idx)
    X = np.array(X, float)
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X, np.array(y), np.array(grps), ids, col_mean


# ----------------------------------------------------------------------
# TRAIN
# ----------------------------------------------------------------------
def mode_train(args):
    gaps = load_gaps(args.gaps)
    scr = load_screener(args.screener)
    cols = FEATURE_SETS[args.features]
    needs_dft = [c for c in cols if c in FRAGMENT_DFT + STRUCT_PORE]
    X, y, grps, ids, col_mean = build_matrix(gaps, scr, cols)
    print(f"Training on {len(y)} COFs, feature set '{args.features}' "
          f"({len(cols)} features)")
    if needs_dft:
        print(f"  NOTE: features {needs_dft} require DFT/pore analysis; this "
              f"model will NOT be runnable on raw CIFs in screen mode.")

    # honest grouped-CV estimate before fitting the deployable model
    gkf = GroupKFold(min(5, len(set(grps))))
    pred = cross_val_predict(
        RandomForestRegressor(300, random_state=0, n_jobs=-1),
        X, y, cv=gkf, groups=grps)
    r2 = r2_score(y, pred); mae = mean_absolute_error(y, pred)
    print(f"  grouped-CV: R2={r2:.3f}  MAE={mae:.3f} eV")

    model = RandomForestRegressor(400, random_state=0, n_jobs=-1)
    model.fit(X, y)
    bundle = {
        "model": model, "feature_cols": cols, "feature_set": args.features,
        "impute_means": col_mean, "pv_window": (PV_LO, PV_HI),
        "cv_r2": r2, "cv_mae": mae, "n_train": len(y),
        "needs_dft_features": needs_dft,
    }
    joblib.dump(bundle, args.model_out)
    print(f"\nSaved model -> {args.model_out}")
    print("Feature importances:")
    for c, imp in sorted(zip(cols, model.feature_importances_),
                         key=lambda x: -x[1]):
        print(f"    {c:14s} {imp:.3f}")


# ----------------------------------------------------------------------
# SCREEN  (raw CIFs -> features -> predict -> filter)
# ----------------------------------------------------------------------
def _import_pipeline():
    """Import the converter + screener modules (must sit alongside this file)."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import cof_cif2smiles as conv
        import pv_cof_screen as scr
        return conv, scr
    except Exception as e:
        sys.exit(f"SCREEN mode needs cof_cif2smiles.py and pv_cof_screen.py "
                 f"in the same folder as this script.\nImport error: {e}")


def features_for_cif(cif_path, conv, scr, cols, use_xtb):
    """Run one CIF through convert + screen and return a feature dict."""
    row = conv.process_cif(Path(cif_path), cap="H", max_atoms=2000)
    if row.get("status") != "OK":
        return None, row.get("status", "FAIL")
    scored = scr.score_row(row, use_xtb)     # adds flags, pv_score, tier, xtb
    feats = parse_flags(scored.get("flags", ""))
    feats["layered"] = 1 if str(row.get("layered")).lower() == "yes" else 0
    for k, col in [("density", "density_g_cm3"), ("volume", "volume_A3"),
                   ("stack_len", "stacking_len_A")]:
        try:
            feats[k] = float(row.get(col, "nan"))
        except Exception:
            feats[k] = np.nan
    for k in HETERO:
        # count heteroatoms from unit SMILES
        pass
    smi = row.get("unit_smiles", "") or ""
    feats["n_count"] = smi.count("N") + smi.count("n")
    feats["o_count"] = smi.upper().count("O")
    feats["b_count"] = smi.upper().count("B")
    if use_xtb:
        try:
            feats["xtb_frag_gap"] = float(scored.get("xtb_frag_gap_eV", "nan"))
        except Exception:
            feats["xtb_frag_gap"] = np.nan
    x = [feats.get(c, np.nan) for c in cols]
    return (feats, x), row


def mode_screen(args):
    bundle = joblib.load(args.model)
    cols = bundle["feature_cols"]
    means = bundle["impute_means"]
    lo, hi = bundle["pv_window"]
    if bundle["needs_dft_features"]:
        sys.exit(f"This model requires DFT features {bundle['needs_dft_features']}"
                 f" that can't be computed from raw CIFs. Retrain with "
                 f"--features deployable (or deployable+xtb).")
    use_xtb = "xtb_frag_gap" in cols
    conv, scr = _import_pipeline()

    cif_dir = Path(args.dir)
    cifs = sorted(cif_dir.glob("*.cif"))
    if not cifs:
        sys.exit(f"No CIFs in {cif_dir}")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print(f"Model '{bundle['feature_set']}' (grouped-CV R2={bundle['cv_r2']:.2f},"
          f" MAE={bundle['cv_mae']:.2f} eV)")
    print(f"Screening {len(cifs)} CIFs; PV window [{lo},{hi}] eV"
          + (" ; xTB on" if use_xtb else "") + "\n")

    results = []
    for k, cif in enumerate(cifs, 1):
        sys.stdout.write(f"\r[{k}/{len(cifs)}] {cif.name[:34]:34s}")
        sys.stdout.flush()
        out, row = features_for_cif(cif, conv, scr, cols, use_xtb)
        if out is None:
            results.append((cif.name, np.nan, "SKIP:" + row, False))
            continue
        feats, x = out
        x = np.array(x, float)
        nan = np.isnan(x)
        x[nan] = means[nan]
        gap = float(bundle["model"].predict(x.reshape(1, -1))[0])
        keep = lo <= gap <= hi
        results.append((cif.name, gap, "OK", keep))
        if keep:
            shutil.copy2(cif, outdir / cif.name)
    sys.stdout.write("\n")

    # ranked manifest
    results.sort(key=lambda r: (not r[3],
                                abs((r[1] if not np.isnan(r[1]) else 99)
                                    - (lo + hi) / 2)))
    man = outdir / "screen_results.csv"
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "pred_gap_eV", "status", "in_pv_window"])
        for name, gap, st, keep in results:
            w.writerow([name, f"{gap:.3f}" if not np.isnan(gap) else "",
                        st, "yes" if keep else "no"])
    n_keep = sum(1 for r in results if r[3])
    print(f"\nPV candidates (pred gap in [{lo},{hi}] eV): {n_keep}/{len(cifs)}")
    print(f"Copied to {outdir}/ ; manifest -> {man}")
    print("\nTop candidates:")
    for name, gap, st, keep in [r for r in results if r[3]][:10]:
        print(f"    {name:28s} pred gap = {gap:.2f} eV")


# ----------------------------------------------------------------------
# ANALYZE  (ablation study; unchanged science, no model saved)
# ----------------------------------------------------------------------
def mode_analyze(args):
    gaps = load_gaps(args.gaps)
    scr = load_screener(args.screener)
    gkf = GroupKFold(5)
    order = ["deployable", "deployable+xtb", "full", "minimal"]
    print(f"{'feature set':16s} {'R2grp':>7s} {'MAEgrp':>7s}  n  (grouped CV)")
    print("-" * 46)
    for name in order:
        cols = FEATURE_SETS[name]
        # skip xtb set if column absent in screener CSV
        X, y, grps, ids, _ = build_matrix(gaps, scr, cols)
        if X.shape[0] == 0:
            continue
        pred = cross_val_predict(
            RandomForestRegressor(300, random_state=0, n_jobs=-1),
            X, y, cv=gkf, groups=grps)
        print(f"{name:16s} {r2_score(y,pred):7.3f} "
              f"{mean_absolute_error(y,pred):7.3f}  {len(y)}")
    print(f"\n({len(set(grps))} COF families)")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("train", help="fit and save a model")
    t.add_argument("--gaps", required=True)
    t.add_argument("--screener", required=True)
    t.add_argument("--features", default="deployable",
                   choices=list(FEATURE_SETS))
    t.add_argument("--model-out", default="pv_model.joblib")
    t.set_defaults(func=mode_train)

    s = sub.add_parser("screen", help="filter a CIF directory for PV candidates")
    s.add_argument("--dir", required=True)
    s.add_argument("--outdir", required=True)
    s.add_argument("--model", default="pv_model.joblib")
    s.set_defaults(func=mode_screen)

    a = sub.add_parser("analyze", help="feature-set ablation study")
    a.add_argument("--gaps", required=True)
    a.add_argument("--screener", required=True)
    a.set_defaults(func=mode_analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
