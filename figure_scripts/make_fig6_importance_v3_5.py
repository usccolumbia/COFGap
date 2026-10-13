"""Figure 6: out-of-fold permutation importance for TabPFN v3.5 (216 COFs, family-grouped 5-fold).
(a) grouped importance (features in a group shuffled jointly, 10 repeats);
(b) top 15 single features (5 repeats). Bars: mean drop in pooled OOF R2; whiskers: s.d. over repeats."""
import glob, json
import matplotlib.pyplot as plt
R = "DFT_replaced_xTB10_TabPFN_V3_5/"
groups = sorted((json.load(open(f)) for f in glob.glob(R + "group_importance_parts/group_*.json")), key=lambda d: d["index"])
single = json.load(open(R + "permutation_importance.json"))
LABEL = {"lattice_min_A": "Shortest lattice parameter", "nodelinker_lumo_eV": "Node+linker LUMO (xTB)",
    "stack_len": "Stacking length", "lattice_anisotropy": "Lattice anisotropy",
    "linker_bond_order_mean": "Linker mean bond order (xTB)", "nodelinker_electronegativity_eV": "Node+linker electronegativity (xTB)",
    "min_gap": "Min. of node/linker gap (xTB)", "lattice_aspect_mid_min": "Lattice aspect ratio (mid/min)",
    "min_fragment_gap": "Min. fragment gap (xTB)", "nodelinker_bond_order_mean": "Node+linker mean bond order (xTB)",
    "nodelinker_bond_order_max": "Node+linker max bond order (xTB)", "node_bond_order_mean": "Node mean bond order (xTB)",
    "linker_lumo_eV": "Linker LUMO (xTB)", "volume_per_atom": "Volume per atom",
    "dipole_change_pair_vs_sum": "Dipole change on pairing (xTB)"}
plt.rcParams.update({"font.family": "Arial", "font.size": 10})
fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=200, gridspec_kw={"width_ratios": [1, 1.15]})
g = groups[::-1]
lab = [("  " if d["index"] >= 2 else "") + f"{d['group']} ({d['n_features']})" for d in g]
col = ["#0d2643" if d["index"] < 2 else ("#3d6079" if "xTB" in d["group"] else "#7fa3bd") for d in g]
a.barh(lab, [d["mean"] for d in g], xerr=[d["std"] for d in g], color=col, height=0.65,
       error_kw=dict(ecolor="#888888", capsize=2, lw=1))
for i, d in enumerate(g):
    a.text(max(d["mean"], 0) + d["std"] + 0.012, i, f"{d['mean']:.2f}", va="center", fontsize=9)
a.axvline(0, color="#999999", lw=0.8); a.set_xlim(-0.02, 0.8)
a.set_xlabel("Drop in pooled OOF $R^2$ (group shuffled)")
a.text(-0.02, 1.02, "(a)", transform=a.transAxes, fontweight="bold", fontsize=13)
items = list(single["importance_r2_drop"].items())[:15][::-1]
xtb = ["xTB" in LABEL[k] for k, _ in items]
b.barh([LABEL[k] for k, _ in items], [v[0] for _, v in items], xerr=[v[1] for _, v in items],
       color=["#3d6079" if x else "#7fa3bd" for x in xtb], height=0.65, error_kw=dict(ecolor="#888888", capsize=2, lw=1))
for i, (_, v) in enumerate(items): b.text(v[0] + v[1] + 0.003, i, f"{v[0]:.3f}", va="center", fontsize=8.5)
b.axvline(0, color="#999999", lw=0.8); b.set_xlim(-0.005, 0.16)
b.set_xlabel("Drop in pooled OOF $R^2$ (single feature shuffled)")
b.text(-0.02, 1.02, "(b)", transform=b.transAxes, fontweight="bold", fontsize=13)
from matplotlib.patches import Patch
b.legend(handles=[Patch(color="#3d6079", label="Fragment electronic (xTB)"), Patch(color="#7fa3bd", label="Framework structure")],
         loc="lower right", fontsize=9, frameon=False)
for ax in (a, b):
    ax.grid(axis="x", ls="--", alpha=0.35)
    for s in ax.spines.values(): s.set_color("#cccccc")
fig.tight_layout(); fig.savefig("figures/pvfig6.png")
