"""Figure 1: COFGap workflow schematic."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "Arial"})
fig, ax = plt.subplots(figsize=(10, 5.3), dpi=300); ax.set_xlim(0, 100); ax.set_ylim(0, 53); ax.axis("off")
STY = {"src": ("#f1eee8", "#6b6b6b", "#3a3a3a", "#555555"), "feat": ("#ecebfb", "#5b4fc4", "#33287a", "#4a3fb3"),
       "out": ("#e2f3ec", "#2c7a5a", "#0f4d36", "#1f6b4c")}
def box(x, y, w, h, title, sub, kind):
    fc, ec, tc, sc = STY[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=12, fontweight="bold", color=tc)
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=9.2, color=sc, linespacing=1.25)
def arrow(pts):
    xs, ys = zip(*pts); ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1] + (0.9 if ys[-1] < ys[-2] else 0),), color="#8a8a8a", lw=1.6)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color="#8a8a8a", lw=1.6, mutation_scale=13))
box(30, 43.5, 40, 8.5, "CoRE-COF v5.0", "216 COFs, 72 structural families", "src")
box(4, 29.5, 34, 9.5, "Fragmentation", "Node, linker, node+linker\n(H-capped at cut bonds)", "feat")
box(4, 16, 34, 10, "GFN2-xTB fragments", "Gaps, frontier orbitals, IP/EA,\ncharges, dipoles, bond orders", "feat")
box(60, 29.5, 36, 9.5, "Framework structure", "Lattice and stacking geometry,\ndensity, porosity, connectivity", "feat")
box(2, 1, 29, 9, "xTB4 feature set", "92 usable descriptors", "out")
box(35.5, 1, 29, 9, "TabPFN v3.5", "Family-grouped 5-fold CV", "out")
box(69, 1, 29, 9, "Predicted band gap", "$E_g$ (eV)", "out")
arrow([(50, 43.5), (50, 41.5), (21, 41.5), (21, 38.5)])
arrow([(50, 41.5), (78, 41.5), (78, 38.5)])
arrow([(21, 29.5), (21, 26)])
arrow([(21, 16), (21, 13.5), (16.5, 13.5), (16.5, 10)])
arrow([(78, 29.5), (78, 13.5), (16.5, 13.5), (16.5, 10)])
arrow([(31, 5.5), (35.5, 5.5)]); arrow([(64.5, 5.5), (69, 5.5)])
fig.savefig("figures/cof_band_gap_pipeline_compact.png", bbox_inches="tight", pad_inches=0.05)
