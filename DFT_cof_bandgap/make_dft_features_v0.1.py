#!/usr/bin/env python3
"""
make_dft_features.py
====================
Generate the four COF band-gap features that CANNOT be obtained from a bare
CIF, and merge them back into the screener table:

    node_hl     HOMO-LUMO gap of the smaller building block   } fragment DFT
    linker_hl   HOMO-LUMO gap of the larger building block    } (ORCA)
    nl_hl       HOMO-LUMO gap of the node+linker construct    }
    sacc        accessible surface area (m^2/cm^3)              (Zeo++)

Together these carry ~0.48 of the random-forest importance, so producing them
is what turns the CIF-only model (grouped R2 ~ 0.28) into the full model
(~0.61).

--------------------------------------------------------------------------
SOFTWARE TO INSTALL
--------------------------------------------------------------------------
1. Python side (always needed)
       conda install -c conda-forge rdkit pymatgen numpy
   or  pip install rdkit pymatgen numpy

2. Zeo++  -- for `sacc`
       conda install -c conda-forge zeoplusplus
   or build from source: http://www.zeoplusplus.org/download.html
       tar -xzf zeo++-0.3.tar.gz && cd zeo++-0.3/voro++/src && make
       cd ../..  && make
   The binary is called `network`. Point this script at it with --network
   if it is not on your PATH.

3. ORCA  -- for the three fragment gaps
       Free for academic use, but requires registration:
       https://orcaforum.kofo.mpg.de  ->  download ORCA 6.x (or 5.0.4, the
       version used by Ernst et al.) -> unpack -> add the folder to PATH.
       ORCA needs a matching OpenMPI for parallel runs; the shared-memory
       build is fine for single-fragment jobs.
       Check with:  orca --version

--------------------------------------------------------------------------
HOW TO RUN  (five steps)
--------------------------------------------------------------------------
  # 0) one-off: check what is installed
  python make_dft_features.py check --network network --orca orca

  # 1) accessible surface area for every CIF     (seconds per structure)
  python make_dft_features.py sacc --cif-dir CoRE-COFs_DT591-v5.0/ \
         --out sacc.csv --jobs 8

  # 2) cut each COF into node / linker / node+linker and write XYZ files
  python make_dft_features.py fragment --cif-dir CoRE-COFs_DT591-v5.0/ \
         --out-dir fragments/

  # 3) write ORCA inputs                      (inspect a few before launching!)
  python make_dft_features.py orca-inputs --frag-dir fragments/ \
         --level pbe0 --nprocs 8

  # 4) run them -- locally, or better, submit to your cluster
  python make_dft_features.py orca-run --frag-dir fragments/ --jobs 4
  #    (or:  for d in fragments/*/; do sbatch my_orca.sh $d; done )

  # 5) parse the outputs and merge everything into the screener CSV
  python make_dft_features.py collect --frag-dir fragments/ \
         --sacc sacc.csv --screener pv_ranked_H.csv \
         --out pv_ranked_H_full.csv

  # then retrain:
  python cof_pv_model.py analyze --gaps bandgap_CoRE_COF_v5_0_PCAdata.csv \
         --screener pv_ranked_H_full.csv

--------------------------------------------------------------------------
COST WARNING -- READ BEFORE STEP 3
--------------------------------------------------------------------------
PBE0-D3/def2-TZVP *geometry optimisation* on a 60-100 atom fragment costs
hours of CPU. For ~600 COFs x 3 fragments that is a cluster campaign, not a
laptop job. Cheaper levels, selectable with --level:

    --level pbe0     PBE0-D3BJ/def2-TZVP  Opt      paper-faithful, expensive
    --level pbe0sp   PBE0-D3BJ/def2-TZVP  single point on the CIF geometry
                                                   ~10-50x cheaper
    --level r2scan3c r2SCAN-3c Opt                 good geometry+energy, ~5x cheaper
    --level gfn2     GFN2-xTB Opt (ORCA driver)    seconds; use to pre-relax

A sensible campaign: --level gfn2 for all fragments, then --level pbe0sp on
the xTB geometries. That reproduces the paper's *ranking* at a small fraction
of the cost. Only use plain `pbe0` if you need the published numbers exactly.

--------------------------------------------------------------------------
FRAGMENTATION
--------------------------------------------------------------------------
Implements the four deCOFpose "cutting" rules (Ernst et al., ChemRxiv 2025):
  1. C-C adjacent to a doubly-substituted N (imine/azine), not carbonyl,
     not part of a 5- or 6-membered ring
  2. B-C, not in a 5-/6-ring                       (boronate / boroxine)
  3. C-N with a triply-substituted N, not amide, not in a ring; for a
     secondary N, cut only if the C is in a ring and the N is not
  4. C-O, not in a ring; in a C1-O-C2=O ester motif cut the C1-O single bond
Fragments keep their CIF coordinates, so no conformer generation is needed.
Following the paper, fragments are NOT labelled node vs linker: the smaller
is written as `node`, the larger as `linker`, and one bonded pair as
`nodelinker`.
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HARTREE = 27.211386245988


# ======================================================================
# helpers
# ======================================================================
def _import_converter(pkgdir):
    sys.path.insert(0, str(pkgdir))
    try:
        import cof_cif2smiles as C
        return C
    except Exception as e:
        sys.exit(f"cannot import cof_cif2smiles.py from {pkgdir}: {e}")


def build_unit_molecule(cif, C, supercell=(2, 2, 1)):
    """H-capped molecule with real 3D coordinates from the CIF.

    A supercell is used so that building blocks in the interior are COMPLETE.
    In a single unit cell every fragment touching the cell boundary is cut by
    the periodic-bond capping, which turns e.g. a benzene ring into a C3H3
    stub. Returns (mol, n_real) where atoms with index >= n_real are capping
    hydrogens marking a boundary - fragments containing them are discarded.
    """
    from pymatgen.core import Structure
    st = Structure.from_file(str(cif))
    if supercell and tuple(supercell) != (1, 1, 1):
        st = st.copy()
        st.make_supercell(list(supercell))
    G = C.build_quotient_graph(st)
    G = C.prune_overbonded(st, G)
    pos, internal, periodic = C.unwrap(st, G)
    n_real = len(st)
    mol = C.to_rdkit(st, pos, internal, periodic, cap="H")
    return mol.GetMol(), n_real


# ======================================================================
# 1. deCOFpose cutting rules
# ======================================================================
# Typical bond lengths (A) above which a bond is single. Below the cutoff we
# treat it as multiple. Values from standard covalent-radius tables; the
# geometry comes straight from the CIF so this is reliable for C=O and C=N,
# the two cases the deCOFpose exception rules depend on.
DOUBLE_CUTOFF = {("C", "O"): 1.28, ("C", "N"): 1.34, ("C", "C"): 1.42,
                 ("N", "N"): 1.32, ("C", "S"): 1.68}


def perceive(mol):
    """Populate ring info and infer multiple bonds from CIF bond lengths.

    The unit-cell molecule arrives with every bond single and no ring
    perception (it was never sanitised, because full bond-order perception on
    a 100+ atom conjugated cell is prohibitively slow). Rings come from
    FastFindRings, which needs no sanitisation; bond multiplicity is read off
    the interatomic distances we already have from the crystal structure.
    """
    from rdkit import Chem
    Chem.FastFindRings(mol)
    conf = mol.GetConformer()
    multiple = set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        si = mol.GetAtomWithIdx(i).GetSymbol()
        sj = mol.GetAtomWithIdx(j).GetSymbol()
        cut = DOUBLE_CUTOFF.get(tuple(sorted((si, sj))))
        if cut is None:
            continue
        pi, pj = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
        d = ((pi.x - pj.x) ** 2 + (pi.y - pj.y) ** 2 + (pi.z - pj.z) ** 2) ** 0.5
        if d < cut:
            multiple.add(b.GetIdx())
    return multiple


def cut_bonds(mol, multiple=None):
    """Return indices of bonds to break, per the four deCOFpose rules."""
    from rdkit import Chem
    if multiple is None:
        multiple = perceive(mol)
    ri = mol.GetRingInfo()

    def in_small_ring(a, b=None):
        for ring in ri.AtomRings():
            if len(ring) in (5, 6):
                if b is None:
                    if a in ring:
                        return True
                elif a in ring and b in ring:
                    return True
        return False

    def has_double_O(atom):
        for nb in atom.GetNeighbors():
            if nb.GetSymbol() != "O":
                continue
            bd = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
            if bd is not None and bd.GetIdx() in multiple:
                return True
        return False

    def n_hydrogens(atom):
        return sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)

    def heavy_deg(atom):
        return sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() > 1)

    cuts = set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtom(), b.GetEndAtom()
        si, sj = i.GetSymbol(), j.GetSymbol()
        if in_small_ring(i.GetIdx(), j.GetIdx()):
            continue                                    # never cut ring bonds
        if b.GetIdx() in multiple:
            continue                                    # keep multiple bonds

        # ---- Rule 2: B-C -------------------------------------------------
        if {si, sj} == {"B", "C"}:
            cuts.add(b.GetIdx()); continue

        # ---- Rule 4: C-O -------------------------------------------------
        if {si, sj} == {"C", "O"}:
            c, o = (i, j) if si == "C" else (j, i)
            # ester C1-O-C2=O : cut the C1-O single bond, i.e. the C without =O
            if has_double_O(c):
                continue                                # this is the C2 side
            cuts.add(b.GetIdx()); continue

        # ---- Rule 3: C-N (triply substituted N, or secondary N) ----------
        if {si, sj} == {"C", "N"}:
            c, n = (i, j) if si == "C" else (j, i)
            if has_double_O(c):
                continue                                # amide / imide
            nd = heavy_deg(n)
            if nd >= 3:
                cuts.add(b.GetIdx()); continue
            if nd == 2 and n_hydrogens(n) >= 1:        # secondary N
                if in_small_ring(c.GetIdx()) and not in_small_ring(n.GetIdx()):
                    cuts.add(b.GetIdx()); continue

        # ---- Rule 1: C-C next to a doubly substituted N ------------------
        if si == "C" and sj == "C":
            for c1, c2 in ((i, j), (j, i)):
                if has_double_O(c2):
                    continue                            # O=C2-C1-N excluded
                for nb in c1.GetNeighbors():
                    if nb.GetSymbol() == "N" and heavy_deg(nb) == 2:
                        cuts.add(b.GetIdx()); break
                if b.GetIdx() in cuts:
                    break
    return sorted(cuts)


def fragments_from_cuts(mol, cuts):
    """Break `cuts`, cap with H, return list of (rdkit mol, atom indices)."""
    from rdkit import Chem
    if not cuts:
        return [(mol, list(range(mol.GetNumAtoms())))]
    em = Chem.RWMol(mol)
    for bi in sorted(cuts, reverse=True):
        b = mol.GetBondWithIdx(bi)
        em.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    frag_idx = Chem.GetMolFrags(em.GetMol(), asMols=False, sanitizeFrags=False)
    out = []
    for idxs in frag_idx:
        out.append((None, list(idxs)))
    return out


def write_xyz(path, symbols, coords, comment=""):
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n{comment}\n")
        for s, (x, y, z) in zip(symbols, coords):
            f.write(f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")


def cap_and_extract(mol, keep, cut_pairs):
    """Extract atoms `keep`; add an H along every bond that leaves the set."""
    conf = mol.GetConformer()
    keepset = set(keep)
    syms, xyz = [], []
    for idx in keep:
        a = mol.GetAtomWithIdx(idx)
        p = conf.GetAtomPosition(idx)
        syms.append(a.GetSymbol()); xyz.append((p.x, p.y, p.z))
    # caps
    for (u, v) in cut_pairs:
        for a, b in ((u, v), (v, u)):
            if a in keepset and b not in keepset:
                pa = conf.GetAtomPosition(a); pb = conf.GetAtomPosition(b)
                va = np.array([pa.x, pa.y, pa.z]); vb = np.array([pb.x, pb.y, pb.z])
                d = vb - va; n = np.linalg.norm(d)
                if n < 1e-6:
                    continue
                blen = {"C": 1.09, "N": 1.01, "O": 0.96, "B": 1.19,
                        "S": 1.34, "P": 1.42, "Si": 1.48}.get(
                            mol.GetAtomWithIdx(a).GetSymbol(), 1.05)
                syms.append("H"); xyz.append(tuple(va + d / n * blen))
    return syms, xyz


def cmd_fragment(args):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    C = _import_converter(args.pkgdir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cifs = sorted(Path(args.cif_dir).glob("*.cif"))
    if not cifs:
        sys.exit(f"no CIFs in {args.cif_dir}")
    manifest = []
    for k, cif in enumerate(cifs, 1):
        sys.stdout.write(f"\r[{k}/{len(cifs)}] {cif.name[:36]:36s}")
        sys.stdout.flush()
        try:
            mol, n_real = build_unit_molecule(cif, C, args.supercell)
            multiple = perceive(mol)
            cuts = cut_bonds(mol, multiple)
            pairs = [(mol.GetBondWithIdx(b).GetBeginAtomIdx(),
                      mol.GetBondWithIdx(b).GetEndAtomIdx()) for b in cuts]
            frags = fragments_from_cuts(mol, cuts)
            if len(frags) < 2:
                manifest.append({"file": cif.name, "status": "no_cut"})
                continue
            # discard fragments that touch the cell boundary: they are
            # truncated by the periodic capping, not real building blocks
            def complete(idxs):
                return all(i < n_real or
                           mol.GetAtomWithIdx(i).GetAtomicNum() == 1
                           for i in idxs) and not any(i >= n_real for i in idxs)
            whole = [(m, idxs) for m, idxs in frags if complete(idxs)]
            # also drop trivial groups (OH, H2O, single heavy atom)
            def heavy(idxs):
                return sum(1 for i in idxs
                           if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
            whole = [t for t in whole if heavy(t[1]) >= args.min_heavy]
            if len(whole) < 2:
                manifest.append({"file": cif.name,
                                 "status": f"too_few_complete_frags"})
                continue
            whole.sort(key=lambda t: heavy(t[1]))
            frags = whole
            small, large = frags[0][1], frags[-1][1]
            # node+linker: smallest fragment plus a fragment bonded to it
            nl = None
            smallset = set(small)
            for (u, v) in pairs:
                for a, b in ((u, v), (v, u)):
                    if a in smallset:
                        for _, idxs in frags:
                            if b in idxs:
                                nl = sorted(smallset | set(idxs))
                                break
                    if nl:
                        break
                if nl:
                    break
            d = out / cif.stem
            d.mkdir(exist_ok=True)
            rec = {"file": cif.name, "status": "ok",
                   "n_cuts": len(cuts), "n_frags": len(frags)}
            for tag, keep in (("node", small), ("linker", large),
                              ("nodelinker", nl)):
                if not keep:
                    continue
                syms, xyz = cap_and_extract(mol, keep, pairs)
                write_xyz(d / f"{tag}.xyz", syms, xyz,
                          f"{cif.name} {tag} ({len(syms)} atoms)")
                rec[f"{tag}_atoms"] = len(syms)
            manifest.append(rec)
        except Exception as e:
            manifest.append({"file": cif.name, "status": f"ERROR {type(e).__name__}"})
    sys.stdout.write("\n")
    cols = ["file", "status", "n_cuts", "n_frags",
            "node_atoms", "linker_atoms", "nodelinker_atoms"]
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(manifest)
    ok = sum(1 for m in manifest if m["status"] == "ok")
    print(f"fragmented {ok}/{len(cifs)} COFs -> {out}/")
    sizes = [m.get("nodelinker_atoms", 0) for m in manifest if m["status"] == "ok"]
    if sizes:
        print(f"node+linker size: median {int(np.median(sizes))} atoms, "
              f"max {max(sizes)}")
    print(f"manifest: {out/'manifest.csv'}")


# ======================================================================
# 2. ORCA
# ======================================================================
LEVELS = {
    "pbe0":     "! PBE0 D3BJ def2-TZVP def2/J RIJCOSX Opt TightSCF",
    "pbe0sp":   "! PBE0 D3BJ def2-TZVP def2/J RIJCOSX TightSCF",
    "r2scan3c": "! r2SCAN-3c Opt TightSCF",
    "gfn2":     "! XTB2 Opt",
}


def cmd_orca_inputs(args):
    frag = Path(args.frag_dir)
    dirs = [d for d in sorted(frag.iterdir()) if d.is_dir()]
    n = 0
    for d in dirs:
        for xyz in d.glob("*.xyz"):
            tag = xyz.stem
            inp = d / f"{tag}.inp"
            natoms = int(open(xyz).readline().split()[0])
            body = open(xyz).read().split("\n")[2:]
            with open(inp, "w") as f:
                f.write(LEVELS[args.level] + "\n")
                if args.nprocs > 1:
                    f.write(f"%pal nprocs {args.nprocs} end\n")
                f.write(f"%maxcore {args.maxcore}\n")
                f.write(f"# {d.name} {tag} ({natoms} atoms)\n")
                f.write("* xyz 0 1\n")
                for line in body:
                    if line.strip():
                        f.write(line + "\n")
                f.write("*\n")
            n += 1
    print(f"wrote {n} ORCA inputs at level '{args.level}' "
          f"({LEVELS[args.level]})")
    print("Inspect one before launching:  head "
          f"{dirs[0].name}/nodelinker.inp" if dirs else "")


def _run_one_orca(inp, orca, timeout):
    out = inp.with_suffix(".out")
    if out.is_file() and "ORCA TERMINATED NORMALLY" in out.read_text(errors="ignore"):
        return inp, "cached"
    try:
        with open(out, "w") as fh:
            subprocess.run([orca, inp.name], cwd=inp.parent, stdout=fh,
                           stderr=subprocess.STDOUT, timeout=timeout)
        txt = out.read_text(errors="ignore")
        return inp, ("ok" if "ORCA TERMINATED NORMALLY" in txt else "failed")
    except subprocess.TimeoutExpired:
        return inp, "timeout"
    except Exception as e:
        return inp, f"error {type(e).__name__}"


def cmd_orca_run(args):
    inps = sorted(Path(args.frag_dir).glob("*/*.inp"))
    if not inps:
        sys.exit("no .inp files - run orca-inputs first")
    print(f"{len(inps)} ORCA jobs, {args.jobs} at a time")
    done = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(_run_one_orca, i, args.orca, args.timeout) for i in inps]
        for k, fut in enumerate(as_completed(futs), 1):
            inp, st = fut.result()
            done[st] = done.get(st, 0) + 1
            sys.stdout.write(f"\r  {k}/{len(inps)}  {dict(done)}      ")
            sys.stdout.flush()
    sys.stdout.write("\n")


ORB_RE = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$")


def parse_orca_gap(path):
    """Return (homo_eV, lumo_eV, gap_eV) from the last ORBITAL ENERGIES block."""
    txt = Path(path).read_text(errors="ignore")
    if "ORCA TERMINATED NORMALLY" not in txt:
        return None
    blocks = txt.split("ORBITAL ENERGIES")
    if len(blocks) < 2:
        return None
    occ_e, vir_e = [], []
    for line in blocks[-1].split("\n"):
        m = ORB_RE.match(line)
        if not m:
            if occ_e and "----" in line:
                break
            continue
        occ = float(m.group(2)); eV = float(m.group(4))
        (occ_e if occ > 0.5 else vir_e).append(eV)
    if not occ_e or not vir_e:
        return None
    homo, lumo = max(occ_e), min(vir_e)
    return homo, lumo, lumo - homo


# ======================================================================
# 3. Zeo++
# ======================================================================
def _run_zeo(cif, network, probe, samples, timeout, workdir):
    tmp = Path(workdir) / cif.stem
    tmp.mkdir(parents=True, exist_ok=True)
    local = tmp / cif.name
    shutil.copy2(cif, local)
    try:
        subprocess.run([network, "-ha", "-sa", str(probe), str(probe),
                        str(samples), local.name],
                       cwd=tmp, capture_output=True, timeout=timeout)
    except Exception as e:
        return cif.name, None, f"FAIL {type(e).__name__}"
    sa = local.with_suffix(".sa")
    if not sa.is_file():
        return cif.name, None, "FAIL no_output"
    txt = sa.read_text()
    m = re.search(r"ASA_m\^2/cm\^3:\s*([-\d.eE+]+)", txt)
    if not m:
        return cif.name, None, "FAIL no_ASA"
    return cif.name, float(m.group(1)), "ok"


def cmd_sacc(args):
    net = shutil.which(args.network) or args.network
    if not Path(net).is_file() and not shutil.which(args.network):
        sys.exit(f"Zeo++ 'network' binary not found (looked for {args.network}).\n"
                 f"  conda install -c conda-forge zeoplusplus\n"
                 f"  or pass --network /path/to/network")
    cifs = sorted(Path(args.cif_dir).glob("*.cif"))
    print(f"{len(cifs)} CIFs, probe {args.probe} A, {args.samples} samples")
    rows, t = [], 0
    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(_run_zeo, c, net, args.probe, args.samples,
                          args.timeout, work) for c in cifs]
        for k, fut in enumerate(as_completed(futs), 1):
            name, val, note = fut.result()
            rows.append({"file": name, "sacc": "" if val is None else f"{val:.3f}",
                         "note": note})
            t += (val is not None)
            sys.stdout.write(f"\r  {k}/{len(cifs)}  ok={t}   ")
            sys.stdout.flush()
    sys.stdout.write("\n")
    rows.sort(key=lambda r: r["file"])
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "sacc", "note"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out}  ({t}/{len(cifs)} succeeded)")
    if not args.keep_workdir:
        shutil.rmtree(work, ignore_errors=True)


# ======================================================================
# 4. collect + merge
# ======================================================================
def cmd_collect(args):
    frag = Path(args.frag_dir)
    gaps = {}
    for d in sorted(p for p in frag.iterdir() if p.is_dir()):
        rec = {}
        for tag, col in (("node", "node_hl"), ("linker", "linker_hl"),
                         ("nodelinker", "nl_hl")):
            out = d / f"{tag}.out"
            if out.is_file():
                r = parse_orca_gap(out)
                if r:
                    rec[col] = r[2]
        if rec:
            gaps[d.name] = rec
    print(f"parsed ORCA gaps for {len(gaps)} COFs")

    sacc = {}
    if args.sacc and Path(args.sacc).is_file():
        for r in csv.DictReader(open(args.sacc)):
            if r.get("sacc"):
                sacc[Path(r["file"]).stem] = r["sacc"]
        print(f"loaded sacc for {len(sacc)} COFs")

    rows = list(csv.DictReader(open(args.screener)))
    cols = list(rows[0])
    for c in ("node_hl", "linker_hl", "nl_hl", "sacc"):
        if c not in cols:
            cols.append(c)
    n_full = 0
    for r in rows:
        stem = Path(r["file"]).stem
        g = gaps.get(stem, {})
        for c in ("node_hl", "linker_hl", "nl_hl"):
            r[c] = f"{g[c]:.4f}" if c in g else r.get(c, "")
        r["sacc"] = sacc.get(stem, r.get("sacc", ""))
        if all(r.get(c) for c in ("nl_hl", "sacc")):
            n_full += 1
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"{n_full}/{len(rows)} rows now have both nl_hl and sacc")
    print(f"wrote {args.out}")
    print("\nRetrain:  python cof_pv_model.py analyze --gaps <gaps.csv> "
          f"--screener {args.out}")


# ======================================================================
def cmd_check(args):
    print("Checking external programs:\n")
    for name, exe, hint in (
        ("Zeo++ (network)", args.network,
         "conda install -c conda-forge zeoplusplus"),
        ("ORCA", args.orca,
         "https://orcaforum.kofo.mpg.de  (free academic registration)"),
    ):
        w = shutil.which(exe)
        print(f"  {name:18s} {'FOUND  ' + w if w else 'NOT FOUND'}")
        if not w:
            print(f"                     -> {hint}")
    for mod in ("rdkit", "pymatgen", "numpy"):
        try:
            __import__(mod)
            print(f"  {mod:18s} ok")
        except ImportError:
            print(f"  {mod:18s} MISSING  -> conda install -c conda-forge {mod}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    here = str(Path(__file__).resolve().parent)

    c = sub.add_parser("check", help="verify ORCA / Zeo++ / python deps")
    c.add_argument("--network", default="network")
    c.add_argument("--orca", default="orca")
    c.set_defaults(func=cmd_check)

    f = sub.add_parser("fragment", help="cut COFs into node/linker/node+linker")
    f.add_argument("--cif-dir", required=True)
    f.add_argument("--out-dir", default="fragments")
    f.add_argument("--pkgdir", default=here)
    f.add_argument("--supercell", type=int, nargs=3, default=[2, 2, 1],
                   metavar=("NA", "NB", "NC"),
                   help="supercell used so interior fragments are complete "
                        "(default 2 2 1; use 2 2 2 for 3D COFs)")
    f.add_argument("--min-heavy", type=int, default=4,
                   help="ignore fragments with fewer heavy atoms (OH, H2O...)")
    f.set_defaults(func=cmd_fragment)

    o = sub.add_parser("orca-inputs", help="write ORCA .inp files")
    o.add_argument("--frag-dir", default="fragments")
    o.add_argument("--level", default="pbe0sp", choices=list(LEVELS))
    o.add_argument("--nprocs", type=int, default=8)
    o.add_argument("--maxcore", type=int, default=3000)
    o.set_defaults(func=cmd_orca_inputs)

    r = sub.add_parser("orca-run", help="run the ORCA jobs locally")
    r.add_argument("--frag-dir", default="fragments")
    r.add_argument("--orca", default="orca")
    r.add_argument("--jobs", type=int, default=2)
    r.add_argument("--timeout", type=float, default=86400)
    r.set_defaults(func=cmd_orca_run)

    z = sub.add_parser("sacc", help="accessible surface area with Zeo++")
    z.add_argument("--cif-dir", required=True)
    z.add_argument("--out", default="sacc.csv")
    z.add_argument("--network", default="network")
    z.add_argument("--probe", type=float, default=1.86,
                   help="probe radius in A (1.86 = N2, the usual choice)")
    z.add_argument("--samples", type=int, default=2000)
    z.add_argument("--jobs", type=int, default=4)
    z.add_argument("--timeout", type=float, default=600)
    z.add_argument("--workdir", default="_zeo_tmp")
    z.add_argument("--keep-workdir", action="store_true")
    z.set_defaults(func=cmd_sacc)

    k = sub.add_parser("collect", help="parse ORCA + sacc, merge into screener")
    k.add_argument("--frag-dir", default="fragments")
    k.add_argument("--sacc", default="sacc.csv")
    k.add_argument("--screener", required=True)
    k.add_argument("--out", required=True)
    k.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
