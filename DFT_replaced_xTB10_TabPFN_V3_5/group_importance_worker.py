#!/usr/bin/env python3
"""Grouped out-of-fold permutation importance for TabPFN v3.5 (Slurm array worker).

usage: group_importance_worker.py DATA_DIR OUT_DIR GROUP_INDEX N_REPEATS
All features of a group are shuffled jointly (same row permutation) within each
held-out family-grouped fold, so correlated features cannot substitute for one another.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

ROLES = ["node", "linker", "nodelinker"]
GAPS = ["node_gap", "linker_gap", "nodelinker_gap", "node_minus_linker_gap", "min_gap", "max_gap",
        "nodelinker_minus_min_gap", *[f"{r}_gap_eV" for r in ROLES], "frontier_cross_gap",
        "min_fragment_gap", "max_fragment_gap", "nodelinker_gap_lowering", "nodelinker_vs_fragment_mean",
        "gap_ratio_node_linker"]
FRONTIER = [f"{r}_{p}" for r in ROLES for p in ("homo_eV", "lumo_eV", "vertical_ip_eV", "vertical_ea_eV",
            "electronegativity_eV", "hardness_eV")] + ["homo_node_minus_linker", "lumo_node_minus_linker",
            "ip_node_minus_linker", "ea_node_minus_linker"]
CHARGE_BOND = [f"{r}_{p}" for r in ROLES for p in ("dipole_debye", "charge_mean_abs_e", "charge_std_e",
               "charge_min_e", "charge_max_e", "bond_order_mean", "bond_order_max", "bond_order_std",
               "n_xtb_atoms")] + ["dipole_change_pair_vs_sum", "charge_polarization_node_linker",
               "bond_order_pair_minus_mean", "size_ratio_node_linker"]
LATTICE = ["layered", "stack_len", "lattice_min_A", "lattice_mid_A", "lattice_max_A", "lattice_anisotropy",
           "lattice_aspect_mid_min", "alpha_deg", "beta_deg", "gamma_deg", "angle_deviation_deg"]
POROSITY = ["density", "volume", "sacc", "volume_per_atom", "sacc_per_atom", "sacc_times_density",
            "surface_volume_proxy", "packing_inverse_volume_per_atom"]
CONNECT = ["n_atoms", "n_pbc_bonds_cut", "n_unique_fragments", "pbc_bond_per_atom"]
COMPOSITION = ["n_frac", "b_frac", "o_frac", "hetero_frac"]
GROUPS = [
    ("All fragment electronic (xTB)", GAPS + FRONTIER + CHARGE_BOND),
    ("All framework structure", LATTICE + POROSITY + CONNECT),
    ("xTB gaps", GAPS), ("xTB frontier orbitals", FRONTIER), ("xTB charges/dipoles/bond orders", CHARGE_BOND),
    ("Lattice and stacking geometry", LATTICE), ("Density and porosity", POROSITY),
    ("Connectivity", CONNECT), ("Composition", COMPOSITION),
]

if __name__ == "__main__":
    data, out, g, reps = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rows = list(csv.DictReader((data / "xtb4_features.csv").open()))
    names = json.loads((data / "feature_groups.json").read_text())["physics_informed"]
    covered = set(sum((m for _, m in GROUPS[2:]), []))
    assert covered == set(names), (set(names) - covered, covered - set(names))
    def num(v):
        try: return float(v)
        except (TypeError, ValueError): return np.nan
    y = np.array([num(r["band_gap_eV"]) for r in rows]); fam = np.array([r["family"] for r in rows])
    X = np.array([[num(r.get(n)) for n in names] for r in rows])
    label, members = GROUPS[g]; cols = [names.index(m) for m in members]
    folds = list(GroupKFold(5).split(X, y, fam))
    models, base = [], np.zeros(len(y))
    for tr, te in folds:
        imp = SimpleImputer(strategy="median").fit(X[tr])
        m = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5)
        m.device = "cpu"; m.n_estimators = 8; m.show_progress_bar = False
        m.fit(imp.transform(X[tr]), y[tr]); models.append((imp, m)); base[te] = m.predict(imp.transform(X[te]))
    r2b = r2_score(y, base); rng = np.random.default_rng(1000 + g); drops = []
    for _ in range(reps):
        pred = np.zeros(len(y))
        for (tr, te), (imp, m) in zip(folds, models):
            Xt = X[te].copy(); Xt[:, cols] = Xt[rng.permutation(len(te))][:, cols]
            pred[te] = m.predict(imp.transform(Xt))
        drops.append(r2b - r2_score(y, pred))
    res = {"index": g, "group": label, "n_features": len(cols), "features": members, "baseline_r2": float(r2b),
           "drops": [float(d) for d in drops], "mean": float(np.mean(drops)), "std": float(np.std(drops))}
    (out / f"group_{g:02d}.json").write_text(json.dumps(res))
    print(json.dumps({k: res[k] for k in ("group", "n_features", "mean", "std")}))
