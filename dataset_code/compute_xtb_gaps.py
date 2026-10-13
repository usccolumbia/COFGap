#!/usr/bin/env python3
"""
compute_xtb_gaps.py
===================
Compute a GFN2-xTB HOMO-LUMO gap for each COF's building block, as a cheap
DFT-free proxy for the deCOFpose `Node+Linker_HL_gap` (nl_hl) feature.

WHY
---
In the COF band-gap model, nl_hl is the single strongest predictor after
stacking length, but it requires a fragment DFT calculation (ORCA) and so
cannot be obtained for new CIFs. A GFN2-xTB gap on the same fragment costs
milliseconds of quantum time and may recover much of that signal. This
script produces the column so you can test that empirically:

    deployable        (no fragment electronics)   grouped R2 ~ 0.28
    deployable+xtb    (this script's column)      -> ?
    full  (with true nl_hl)                        grouped R2 ~ 0.57

INSTALL
-------
    pip install tblite rdkit pandas scikit-learn
  If pip's tblite wheel fails (it needs the compiled C library), use conda:
    conda install -c conda-forge tblite-python rdkit
  Alternative backend: `pip install xtb` (xtb-python) - slower to install but
  equivalent; set --backend xtb.

PERFORMANCE NOTES
-----------------
The xTB single point itself is ~0.01-1 s. The real cost is 3D conformer
embedding (ETKDG), which scales badly and can hang on large fused-ring
fragments. This script therefore:
  * canonicalises and CACHES by fragment SMILES  (COFs share building blocks,
    so ~315 COFs typically reduce to far fewer unique fragments)
  * runs each fragment in a SUBPROCESS with a hard timeout, so a pathological
    fragment is killed instead of stalling the run
  * parallelises across fragments with a thread pool (subprocesses release
    the GIL)
  * writes a resumable cache file, so re-running skips completed work

USAGE
-----
  # 1) compute the gaps (writes cache + augmented screener CSV)
  python compute_xtb_gaps.py --screener pv_ranked_H.csv \
         --out pv_ranked_H_xtb.csv --jobs 8

  # 2) retrain / compare
  python cof_pv_model.py analyze --gaps bandgap_CoRE_COF_v5_0_PCAdata.csv \
         --screener pv_ranked_H_xtb.csv

  # 3) if it helps, save a deployable model that uses it
  python cof_pv_model.py train --gaps ... --screener pv_ranked_H_xtb.csv \
         --features deployable+xtb --model-out pv_model_xtb.joblib

OPTIONS OF NOTE
  --target largest   gap of the single largest fragment (default; closest
                     analogue of the paper's node+linker construct)
  --target combined  gap of all fragments joined as one disconnected system
                     (approximates the frontier orbitals of the whole unit)
  --max-atoms 60     skip fragments larger than this (embedding cost)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ----------------------------------------------------------------------
# WORKER A: use the REAL geometry from the CIF (fast, no embedding)
# ----------------------------------------------------------------------
WORKER_CIF = r'''
import sys, numpy as np
sys.path.insert(0, sys.argv[2])          # dir containing cof_cif2smiles.py
cif = sys.argv[1]
backend = sys.argv[3] if len(sys.argv) > 3 else "tblite"
try:
    import warnings; warnings.filterwarnings("ignore")
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
    from pymatgen.core import Structure
    import cof_cif2smiles as C
    s = Structure.from_file(cif)
    G = C.build_quotient_graph(s)
    G = C.prune_overbonded(s, G)
    pos, internal, periodic = C.unwrap(s, G)
    mol = C.to_rdkit(s, pos, internal, periodic, cap="H")   # H-capped, 3D
    m = mol.GetMol()
    conf = m.GetConformer()
    nums = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    xyz = np.array([list(conf.GetAtomPosition(i))
                    for i in range(m.GetNumAtoms())])
    if len(nums) == 0:
        print("FAIL empty"); sys.exit(0)
    if int(nums.sum()) % 2 != 0:
        print("FAIL odd_electrons"); sys.exit(0)
    p = xyz / 0.529177210903
    if backend == "tblite":
        from tblite.interface import Calculator
        calc = Calculator("GFN2-xTB", nums, p)
        calc.set("verbosity", 0)
        res = calc.singlepoint()
        e = np.array(res.get("orbital-energies")) * 27.211386
        occ = np.array(res.get("orbital-occupations"))
    else:
        from xtb.interface import Calculator as XCalc, Param
        calc = XCalc(Param.GFN2xTB, nums, p)
        res = calc.singlepoint()
        e = np.array(res.get_orbital_eigenvalues()) * 27.211386
        occ = np.array(res.get_orbital_occupations())
    homo = e[occ > 0.5].max(); lumo = e[occ <= 0.5].min()
    print(f"{lumo - homo:.6f}")
except Exception as exc:
    print("FAIL " + type(exc).__name__)
'''


# ----------------------------------------------------------------------
# WORKER B: embed a SMILES fragment in 3D (fallback; slow on big fragments)
# ----------------------------------------------------------------------
WORKER_CODE = r'''
import sys, json, numpy as np
smi = sys.stdin.read().strip()
backend = sys.argv[1] if len(sys.argv) > 1 else "tblite"
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        print("FAIL parse"); sys.exit(0)
    # wildcards -> H so the fragment is a closed-shell molecule
    rw = Chem.RWMol(m)
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetAtomicNum(1)
    m = rw.GetMol()
    Chem.SanitizeMol(m, catchErrors=True)
    m = Chem.AddHs(m)
    if AllChem.EmbedMolecule(m, randomSeed=42, useRandomCoords=True,
                             maxAttempts=2) != 0:
        print("FAIL embed"); sys.exit(0)
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=400)
    except Exception:
        pass
    conf = m.GetConformer()
    nums = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    pos = np.array([list(conf.GetAtomPosition(i))
                    for i in range(m.GetNumAtoms())]) / 0.529177210903
    if backend == "tblite":
        from tblite.interface import Calculator
        calc = Calculator("GFN2-xTB", nums, pos)
        calc.set("verbosity", 0)
        res = calc.singlepoint()
        e = np.array(res.get("orbital-energies")) * 27.211386
        occ = np.array(res.get("orbital-occupations"))
    else:
        from xtb.interface import Calculator as XCalc, Param
        calc = XCalc(Param.GFN2xTB, nums, pos)
        res = calc.singlepoint()
        e = np.array(res.get_orbital_eigenvalues()) * 27.211386
        occ = np.array(res.get_orbital_occupations())
    homo = e[occ > 0.5].max()
    lumo = e[occ <= 0.5].min()
    print(f"{lumo - homo:.6f}")
except Exception as exc:
    print("FAIL " + type(exc).__name__)
'''


def run_worker_cif(cif, backend, timeout, worker_path, pkgdir):
    """Run xTB on the CIF's own geometry (no 3D embedding needed)."""
    try:
        r = subprocess.run([sys.executable, worker_path, cif, pkgdir, backend],
                           capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "").strip().splitlines()
        out = out[-1] if out else ""
        if out.startswith("FAIL") or not out:
            return None, out or "FAIL empty"
        return float(out), "ok"
    except subprocess.TimeoutExpired:
        return None, "FAIL timeout"
    except Exception as exc:
        return None, f"FAIL {type(exc).__name__}"


def run_worker(smiles, backend, timeout, worker_path):
    try:
        r = subprocess.run([sys.executable, worker_path, backend],
                           input=smiles, capture_output=True, text=True,
                           timeout=timeout)
        out = (r.stdout or "").strip().splitlines()
        out = out[-1] if out else ""
        if out.startswith("FAIL") or not out:
            return None, out or "FAIL empty"
        return float(out), "ok"
    except subprocess.TimeoutExpired:
        return None, "FAIL timeout"
    except Exception as exc:
        return None, f"FAIL {type(exc).__name__}"


# ----------------------------------------------------------------------
def canonical(smi):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return smi
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return smi


def heavy_atoms(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi, sanitize=False)
    return m.GetNumHeavyAtoms() if m is not None else 10 ** 6


def pick_fragment(row, target):
    frags = [s for s in (row.get("frag_smiles") or "").split(".") if s]
    if not frags:
        u = (row.get("unit_smiles") or "").strip()
        frags = [u] if u else []
    if not frags:
        return None
    if target == "combined":
        return ".".join(frags)
    return max(frags, key=len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screener", required=True,
                    help="pv_ranked_H.csv (must have frag_smiles)")
    ap.add_argument("--out", required=True,
                    help="augmented copy with xtb_frag_gap_eV column")
    ap.add_argument("--cache", default="xtb_gap_cache.json")
    ap.add_argument("--target", choices=["largest", "combined"],
                    default="largest")
    ap.add_argument("--backend", choices=["tblite", "xtb"], default="tblite")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--timeout", type=float, default=90.0,
                    help="seconds per fragment (embed + xTB)")
    ap.add_argument("--max-atoms", type=int, default=60,
                    help="[embed mode] skip fragments larger than this")
    ap.add_argument("--geometry", choices=["cif", "embed"], default="cif",
                    help="'cif' (default, recommended): run xTB directly on "
                         "the H-capped unit-cell molecule built from the CIF's "
                         "own 3D coordinates - no conformer generation, so it "
                         "is far faster and cannot fail to embed. 'embed': "
                         "generate a 3D conformer from frag_smiles instead "
                         "(use only if you have no CIFs).")
    ap.add_argument("--cif-dir", default=None,
                    help="[cif mode] folder holding the CIFs named in 'file'")
    ap.add_argument("--pkgdir", default=str(Path(__file__).resolve().parent),
                    help="[cif mode] folder containing cof_cif2smiles.py")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.screener)))
    if "frag_smiles" not in rows[0]:
        sys.exit("screener CSV has no frag_smiles column")
    print(f"loaded {len(rows)} rows from {args.screener}")

    cache = {}
    if Path(args.cache).is_file():
        cache = json.load(open(args.cache))
        print(f"resuming: {len(cache)} entries already in {args.cache}")

    cif_mode = (args.geometry == "cif")
    if cif_mode:
        if not args.cif_dir:
            sys.exit("--geometry cif requires --cif-dir")
        cifdir = Path(args.cif_dir)
        if not cifdir.is_dir():
            sys.exit(f"no such folder: {cifdir}")
        if not (Path(args.pkgdir) / "cof_cif2smiles.py").is_file():
            sys.exit(f"cof_cif2smiles.py not found in {args.pkgdir}; "
                     f"pass --pkgdir")
        worker_path = Path(args.cache).with_suffix(".worker_cif.py")
        worker_path.write_text(WORKER_CIF)
        per_row, todo, missing = {}, set(), 0
        for r in rows:
            f = cifdir / r["file"]
            if not f.is_file():
                missing += 1
                continue
            key = r["file"]
            per_row[key] = key
            if key not in cache:
                todo.add(str(f))
        print(f"CIF geometry mode: {len(per_row)} CIFs found"
              + (f", {missing} missing" if missing else "")
              + f"; to compute: {len(todo)}")
    else:
        worker_path = Path(args.cache).with_suffix(".worker.py")
        worker_path.write_text(WORKER_CODE)
        per_row, todo = {}, set()
        skipped_big = 0
        for r in rows:
            smi = pick_fragment(r, args.target)
            if not smi:
                continue
            c = canonical(smi)
            per_row[r["file"]] = c
            if c in cache:
                continue
            if heavy_atoms(c) > args.max_atoms:
                cache[c] = {"gap": None, "note": "skip too_large"}
                skipped_big += 1
                continue
            todo.add(c)
        print(f"embed mode: unique fragments "
              f"{len(set(per_row.values()))}   to compute: {len(todo)}   "
              f"skipped (>{args.max_atoms} atoms): {skipped_big}")
    if not todo:
        print("nothing to compute")

    t0, done = time.time(), 0
    if todo:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            if cif_mode:
                futs = {ex.submit(run_worker_cif, c, args.backend,
                                  args.timeout, str(worker_path),
                                  args.pkgdir): Path(c).name for c in todo}
            else:
                futs = {ex.submit(run_worker, s, args.backend, args.timeout,
                                  str(worker_path)): s for s in todo}
            for fut in as_completed(futs):
                s = futs[fut]
                gap, note = fut.result()
                cache[s] = {"gap": gap, "note": note}
                done += 1
                if done % 10 == 0 or done == len(todo):
                    el = time.time() - t0
                    rate = done / el if el else 0
                    eta = (len(todo) - done) / rate if rate else 0
                    ok = sum(1 for v in cache.values() if v["gap"] is not None)
                    sys.stdout.write(
                        f"\r  {done}/{len(todo)}  ok={ok}  "
                        f"{rate:.1f} frag/s  ETA {eta:5.0f}s")
                    sys.stdout.flush()
                if done % 25 == 0:
                    json.dump(cache, open(args.cache, "w"))
        sys.stdout.write("\n")
    json.dump(cache, open(args.cache, "w"))

    # write augmented CSV
    cols = list(rows[0])
    if "xtb_frag_gap_eV" not in cols:
        cols.append("xtb_frag_gap_eV")
    n_ok = 0
    for r in rows:
        c = per_row.get(r["file"])
        g = cache.get(c, {}).get("gap") if c else None
        r["xtb_frag_gap_eV"] = "" if g is None else f"{g:.4f}"
        if g is not None:
            n_ok += 1
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    gaps = [float(r["xtb_frag_gap_eV"]) for r in rows if r["xtb_frag_gap_eV"]]
    print(f"\nCOFs with an xTB gap: {n_ok}/{len(rows)}")
    if gaps:
        import statistics as st
        print(f"gap range: {min(gaps):.2f} - {max(gaps):.2f} eV  "
              f"(median {st.median(gaps):.2f})")
    fails = {}
    for v in cache.values():
        if v["gap"] is None:
            fails[v["note"]] = fails.get(v["note"], 0) + 1
    if fails:
        print("failures:", fails)
    print(f"wrote {args.out}")
    print("\nNext:  python cof_pv_model.py analyze --gaps <gaps.csv> "
          f"--screener {args.out}")


if __name__ == "__main__":
    main()
