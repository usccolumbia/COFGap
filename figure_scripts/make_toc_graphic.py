"""ACS table-of-contents graphic (3.25 x 1.75 in): fragment + framework inputs -> TabPFN v3.5 -> OOF parity."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, RegularPolygon
from sklearn.metrics import mean_absolute_error, r2_score
rows = list(csv.DictReader(open("DFT_replaced_xTB10_TabPFN_V3_5/oof_predictions.csv")))
y = np.array([float(r["band_gap_eV"]) for r in rows]); p = np.array([float(r["tabpfn_v3_5"]) for r in rows])
plt.rcParams.update({"font.family": "Arial"})
fig = plt.figure(figsize=(3.25, 1.75), dpi=600)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 325); ax.set_ylim(0, 175); ax.axis("off")
PUR, PURL, GRN, GRNL, RED = "#4a3fb3", "#ecebfb", "#1f6b4c", "#e2f3ec", "#e63946"
def panel(x, y0, w, h, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=6", fc=fc, ec=ec, lw=0.8))
# Fragment panel: hexagon node + linker, H caps
panel(6, 92, 112, 76, PURL, PUR)
for cx, cy, rr in ((36, 135, 13), (78, 135, 10)):
    ax.add_patch(RegularPolygon((cx, cy), 6, radius=rr, fc="white", ec=PUR, lw=1.1))
ax.plot([49, 68], [135, 135], color=PUR, lw=1.1)
for hx in (14, 98):
    ax.plot([hx + (8 if hx < 50 else -8), hx], [135, 135], color="#888888", lw=0.8)
    ax.add_patch(Circle((hx, 135), 3, fc="white", ec="#888888", lw=0.7))
ax.text(62, 104, "Fragments (GFN2-xTB)", ha="center", va="center", fontsize=6.2, color=PUR, fontweight="bold")
# Framework panel: stacked layers with lattice arrows
panel(6, 8, 112, 76, PURL, PUR)
for k in range(3):
    yy = 32 + k * 12
    ax.add_patch(plt.Polygon([[24 + k * 4, yy], [86 + k * 4, yy], [98 + k * 4, yy + 8], [36 + k * 4, yy + 8]], fc="white", ec=PUR, lw=0.9))
ax.annotate("", xy=(108, 66), xytext=(108, 34), arrowprops=dict(arrowstyle="<->", color=RED, lw=0.9))
ax.text(62, 20, "Framework (stacking)", ha="center", va="center", fontsize=6.2, color=PUR, fontweight="bold")
# Arrows into model
for yy in (130, 46):
    ax.annotate("", xy=(140, 88 + (yy - 88) * 0.15), xytext=(120, yy), arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.0, mutation_scale=7))
panel(140, 66, 52, 44, GRNL, GRN)
ax.text(166, 94, "TabPFN", ha="center", va="center", fontsize=7.5, color=GRN, fontweight="bold")
ax.text(166, 80, "v3.5", ha="center", va="center", fontsize=7.5, color=GRN, fontweight="bold")
ax.annotate("", xy=(206, 88), xytext=(194, 88), arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.0, mutation_scale=7))
ax.text(166, 46, "No DFT", ha="center", va="center", fontsize=6.6, color=RED, fontweight="bold")
ax.text(166, 35, "required", ha="center", va="center", fontsize=6.6, color=RED, fontweight="bold")
# Parity inset
pa = fig.add_axes([0.715, 0.17, 0.27, 0.75])
pa.scatter(y, p, s=2.2, color="#0d2643", alpha=0.65, lw=0)
pa.plot([0, 5], [0, 5], "--", color="#888888", lw=0.6)
pa.set_xlim(0, 5); pa.set_ylim(0, 5); pa.set_aspect("equal")
pa.set_xticks([0, 2.5, 5]); pa.set_yticks([0, 2.5, 5]); pa.tick_params(labelsize=4.5, length=1.5, width=0.4, pad=1)
pa.set_xlabel("DFT $E_g$ (eV)", fontsize=5, labelpad=1); pa.set_ylabel("Predicted (eV)", fontsize=5, labelpad=1)
pa.text(0.2, 4.75, f"$R^2$ = {r2_score(y, p):.2f}\nMAE = {mean_absolute_error(y, p):.2f} eV", va="top", fontsize=4.8)
for s in pa.spines.values(): s.set_linewidth(0.4); s.set_color("#999999")
fig.savefig("figures/toc_graphic.png", dpi=600); fig.savefig("figures/toc_graphic.tif", dpi=600)
