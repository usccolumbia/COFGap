#!/usr/bin/env python3
"""
pv_cof_screen.py
================
Fast, DFT-free screening/ranking of COFs for photovoltaic potential.

Takes the CSV produced by cof_cif2smiles.py (columns: file, unit_smiles,
frag_smiles, density_g_cm3, layered, stacking_len_A, ...) and scores every
COF 0-100 using chemistry rules (RDKit SMARTS on the fragments) plus
structural descriptors. Optionally adds a semiempirical HOMO-LUMO gap of
the largest fragment via tblite (GFN2-xTB), a few seconds per COF.

Scoring rationale (all literature-motivated, weights adjustable at top):

  CHEMISTRY (on fragment SMILES)
  + conjugation fraction          - sp2/aromatic heavy-atom fraction; high
                                    conjugation -> low gap, mobile carriers
  + conjugation-preserving links  - vinylene C=C > imine/azine C=N >
                                    ketoenamine >> boronate/boroxine (B-O
                                    breaks pi-conjugation -> wide gaps)
  + donor motifs                  - triphenylamine, carbazole, thiophene,
                                    pyrene-like PAH, alkoxy-arene
  + acceptor motifs               - triazine, benzothiadiazole, quinone,
                                    (di)imide, nitrile, azole
  + D-A synergy bonus             - both donor AND acceptor present ->
                                    intramolecular charge transfer, narrow
                                    gap, built-in exciton dissociation
  - sp3 dilution penalty          - saturated content interrupts transport
  - metal / inorganic penalty     - outside the organic-PV design space

  STRUCTURE (from CSV columns)
  + layered with interlayer 3.2-3.7 A  - good pi-pi overlap along stacking
  + density                            - compact frameworks -> smaller gaps
                                         (PCA result of the deCOFpose paper)

  OPTIONAL ELECTRONIC (needs tblite)
  + fragment GFN2-xTB HOMO-LUMO gap mapped through a generic xTB->optical
    plausibility window. xTB gaps are ~2x DFT; the window used here
    (4-8 eV raw xTB on molecular fragments) empirically brackets organic
    chromophores absorbing in/near the visible. Treat as a coarse bonus
    term, NOT a gap prediction - calibrate against your own DFT tier.

Output: ranked CSV with total score, per-component breakdown, tier
(A = prioritize for DFT, B = keep in training pool, C = deprioritize),
and human-readable flags explaining each score.

Usage:
    python pv_cof_screen.py cof_smiles.csv [--out pv_ranked.csv]
                            [--xtb] [--top 20]

Dependencies: rdkit, numpy (pymatgen not needed); tblite optional (--xtb)
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# ----------------------------------------------------------------------
# Weights (sum of maxima = 100 when xTB disabled; renormalized if enabled)
# ----------------------------------------------------------------------
W = {
    "conjugation":   22.0,   # sp2/aromatic fraction
    "linkage":       18.0,   # conjugation-preserving linkage chemistry
    "donor":         12.0,
    "acceptor":      12.0,
    "da_bonus":      12.0,   # both present
    "stacking":      14.0,
    "density":       10.0,
    "xtb_gap":       20.0,   # only if --xtb
}
SP3_PENALTY_MAX = 15.0       # subtracted, scaled by sp3 carbon fraction
METAL_PENALTY = 40.0

ORGANIC_OK = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl",
              "Se", "Br", "I", "*"}

# ----------------------------------------------------------------------
# SMARTS libraries
# ----------------------------------------------------------------------
def _smarts(d):
    out = {}
    for name, s in d.items():
        m = Chem.MolFromSmarts(s)
        if m is not None:
            out[name] = m
    return out


DONORS = _smarts({
    "triarylamine":  "N(c)(c)c",
    "carbazole":     "c1ccc2c(c1)[nX3]c1ccccc12",
    "thiophene":     "c1ccsc1",
    "pyrrole":       "c1cc[nH]c1",
    "furan":         "c1ccoc1",
    "alkoxy_arene":  "cO[CX4]",
    "hydroxy_arene": "c[OX2H]",
    "dialkylamino":  "c[NX3]([CX4])[CX4]",
})

ACCEPTORS = _smarts({
    "triazine":          "c1ncncn1",
    "benzothiadiazole":  "c1ccc2nsnc2c1",
    "quinone":           "O=C1C=CC(=O)C=C1",
    "imide":             "C(=O)[NX3]C(=O)",
    "nitrile":           "C#N",
    "pyrazine":          "c1cnccn1",
    "benzoxazole":       "c1ccc2ocnc2c1",
    "benzothiazole":     "c1ccc2scnc2c1",
    "nitro":             "[NX3](=O)=O",
    "azo":               "cN=Nc",
})

LINKAGES = {          # name: (SMARTS, score 0-1)
    "vinylene":     ("c/C=C/c", 1.00),
    "vinylene2":    ("cC=Cc", 1.00),
    "azine":        ("C=NN=C", 0.85),
    "imine":        ("[cX3][CX3H1]=[NX2]", 0.80),
    "triazine_l":   ("c1ncncn1", 0.90),
    "ketoenamine":  ("O=C1C=C([NX3])C=CC1", 0.60),
    "amide_imide":  ("C(=O)[NX3]", 0.45),
    "hydrazone":    ("C=N[NX3]", 0.55),
    "boronate":     ("B1OccO1", 0.15),
    "boronate2":    ("[B]([OX2])[OX2]", 0.15),
    "boroxine":     ("B1OB(O1)", 0.10),
}
LINKAGE_PATS = {k: (Chem.MolFromSmarts(s), v) for k, (s, v) in LINKAGES.items()}


# ----------------------------------------------------------------------
# Chemistry scoring
# ----------------------------------------------------------------------
def load_frags(frag_field, unit_field):
    """Return RDKit mols for each fragment (sanitize loosely)."""
    smis = []
    if isinstance(frag_field, str) and frag_field.strip():
        smis = [s for s in frag_field.split(".") if s]
    elif isinstance(unit_field, str) and unit_field.strip():
        smis = [unit_field]
    mols = []
    for s in smis:
        m = Chem.MolFromSmiles(s, sanitize=False)
        if m is None:
            continue
        try:
            Chem.SanitizeMol(
                m,
                Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception:
            pass
        mols.append(m)
    return mols


def conjugation_fraction(mols):
    """Fraction of heavy (non-H, non-*) atoms that are aromatic or sp2/sp."""
    heavy = conj = 0
    for m in mols:
        for a in m.GetAtoms():
            if a.GetAtomicNum() in (0, 1):
                continue
            heavy += 1
            if a.GetIsAromatic() or a.GetHybridization() in (
                    Chem.HybridizationType.SP2, Chem.HybridizationType.SP):
                conj += 1
    return conj / heavy if heavy else 0.0


def sp3_carbon_fraction(mols):
    c = sp3 = 0
    for m in mols:
        for a in m.GetAtoms():
            if a.GetAtomicNum() != 6:
                continue
            c += 1
            if a.GetHybridization() == Chem.HybridizationType.SP3:
                sp3 += 1
    return sp3 / c if c else 0.0


def match_lib(mols, lib):
    hits = set()
    for m in mols:
        for name, pat in lib.items():
            try:
                if m.HasSubstructMatch(pat):
                    hits.add(name)
            except Exception:
                continue
    return hits


def linkage_score(mols):
    best, name = 0.0, ""
    for m in mols:
        for lname, (pat, sc) in LINKAGE_PATS.items():
            if pat is None:
                continue
            try:
                if m.HasSubstructMatch(pat) and sc > best:
                    best, name = sc, lname
            except Exception:
                continue
    return best, name


def has_metal(mols):
    for m in mols:
        for a in m.GetAtoms():
            if a.GetSymbol() not in ORGANIC_OK:
                return a.GetSymbol()
    return None


# ----------------------------------------------------------------------
# Optional xTB fragment gap (tblite)
# ----------------------------------------------------------------------
def xtb_fragment_gap(mol):
    """GFN2-xTB HOMO-LUMO gap (eV) of the largest H-capped fragment."""
    try:
        from rdkit.Chem import AllChem
        from tblite.interface import Calculator
    except Exception:
        return None
    try:
        m = Chem.RWMol(mol)
        # replace wildcards with H
        for a in m.GetAtoms():
            if a.GetAtomicNum() == 0:
                a.SetAtomicNum(1)
        m = m.GetMol()
        Chem.SanitizeMol(m, catchErrors=True)
        m = Chem.AddHs(m)
        if AllChem.EmbedMolecule(m, randomSeed=42) != 0:
            return None
        AllChem.MMFFOptimizeMolecule(m, maxIters=500)
        conf = m.GetConformer()
        nums = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
        pos = np.array([list(conf.GetAtomPosition(i))
                        for i in range(m.GetNumAtoms())]) / 0.529177
        calc = Calculator("GFN2-xTB", nums, pos)
        calc.set("verbosity", 0)
        res = calc.singlepoint()
        e = np.array(res.get("orbital-energies")) * 27.2114
        occ = np.array(res.get("orbital-occupations"))
        homo = e[occ > 0.5].max()
        lumo = e[occ <= 0.5].min()
        return float(lumo - homo)
    except Exception:
        return None


def xtb_gap_score(gap):
    """
    Map raw molecular xTB gap to plausibility 0-1. Window 4-8 eV raw
    (~2-4 eV DFT-ish fundamental, absorbing in/near visible after
    exciton/periodic corrections). Triangular: peak at 5.5 eV.
    """
    if gap is None:
        return None
    lo, peak, hi = 4.0, 5.5, 8.0
    if gap <= lo or gap >= hi:
        return 0.0
    if gap <= peak:
        return (gap - lo) / (peak - lo)
    return (hi - gap) / (hi - peak)


# ----------------------------------------------------------------------
# Structure scoring
# ----------------------------------------------------------------------
def stacking_score(layered, stack_len):
    if str(layered).lower() != "yes":
        return 0.3          # 3D COFs: not excluded, just less pi-stacked
    try:
        d = float(stack_len)
    except (TypeError, ValueError):
        return 0.3
    # per-layer distance may be a multiple; reduce to plausible interlayer
    while d > 7.5:
        d /= 2.0
    # triangular around 3.2-3.7 A
    if 3.2 <= d <= 3.7:
        return 1.0
    if d < 3.2:
        return max(0.0, 1.0 - (3.2 - d) / 0.6)
    return max(0.0, 1.0 - (d - 3.7) / 1.3)


def density_score(dens):
    try:
        d = float(dens)
    except (TypeError, ValueError):
        return 0.5
    # COFs ~0.1-1.6 g/cm3; favour compact (>=0.9 ~ 1.0 score)
    return float(np.clip((d - 0.3) / 0.6, 0.0, 1.0))


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def score_row(row, use_xtb):
    mols = load_frags(row.get("frag_smiles", ""), row.get("unit_smiles", ""))
    out = dict(row)
    flags = []

    if not mols:
        out.update(pv_score=0.0, tier="C", flags="NO_PARSEABLE_SMILES")
        return out

    metal = has_metal(mols)
    conj = conjugation_fraction(mols)
    sp3f = sp3_carbon_fraction(mols)
    donors = match_lib(mols, DONORS)
    acceptors = match_lib(mols, ACCEPTORS)
    lscore, lname = linkage_score(mols)

    s = 0.0
    s += W["conjugation"] * conj
    s += W["linkage"] * lscore
    s += W["donor"] * min(len(donors), 2) / 2.0
    s += W["acceptor"] * min(len(acceptors), 2) / 2.0
    if donors and acceptors:
        s += W["da_bonus"]
        flags.append("D-A")
    s += W["stacking"] * stacking_score(row.get("layered"),
                                        row.get("stacking_len_A"))
    s += W["density"] * density_score(row.get("density_g_cm3"))
    s -= SP3_PENALTY_MAX * sp3f
    if metal:
        s -= METAL_PENALTY
        flags.append(f"METAL_{metal}")

    gap = None
    if use_xtb:
        big = max(mols, key=lambda m: m.GetNumHeavyAtoms())
        gap = xtb_fragment_gap(big)
        gs = xtb_gap_score(gap)
        if gs is not None:
            s += W["xtb_gap"] * gs
            out["xtb_frag_gap_eV"] = round(gap, 2)

    max_pts = sum(v for k, v in W.items() if k != "xtb_gap") \
        + (W["xtb_gap"] if use_xtb and gap is not None else 0.0)
    s = max(0.0, 100.0 * s / max_pts)

    flags.append(f"link:{lname or 'none'}")
    if donors:
        flags.append("don:" + "|".join(sorted(donors)))
    if acceptors:
        flags.append("acc:" + "|".join(sorted(acceptors)))
    flags.append(f"conj:{conj:.2f}")
    if sp3f > 0.3:
        flags.append(f"sp3:{sp3f:.2f}")

    out["pv_score"] = round(s, 1)
    out["tier"] = "A" if s >= 65 else ("B" if s >= 40 else "C")
    out["flags"] = ";".join(flags)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_in", help="output CSV of cof_cif2smiles.py")
    ap.add_argument("--out", default="pv_ranked.csv")
    ap.add_argument("--xtb", action="store_true",
                    help="add GFN2-xTB fragment gap term (needs tblite)")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    with open(args.csv_in) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("Empty input CSV.")

    scored = [score_row(r, args.xtb) for r in rows]
    scored.sort(key=lambda r: -r.get("pv_score", 0))

    cols = list(rows[0].keys()) + ["pv_score", "tier", "flags"]
    if args.xtb:
        cols.append("xtb_frag_gap_eV")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(scored)

    print(f"{'rank':4s} {'score':6s} {'tier':4s} file / flags")
    for i, r in enumerate(scored[:args.top], 1):
        print(f"{i:<4d} {r['pv_score']:<6} {r['tier']:<4s} {r['file']}")
        print(f"     {r['flags'][:110]}")
    nA = sum(1 for r in scored if r["tier"] == "A")
    nB = sum(1 for r in scored if r["tier"] == "B")
    print(f"\nTiers: A={nA}  B={nB}  C={len(scored)-nA-nB}   -> {args.out}")


if __name__ == "__main__":
    main()
