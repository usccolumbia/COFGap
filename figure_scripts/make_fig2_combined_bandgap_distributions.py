"""Create the title-free two-panel Figure 2 band-gap distributions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 1.2,
        "grid.color": "#EBEBEB",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    }
)

PRIMARY_BLUE = "#1B4965"
ACCENT_RED = "#E63946"
ROOT = Path(__file__).resolve().parents[1]
ALL_DATA = ROOT / "dataset_code/bandgap_CoRE_COF_v5.0_DFTfree_features.csv"
COMPLETE_XTB = ROOT / "DFT_replaced_xTB4/01_feature_table/xtb4_features.csv"
OUTPUT_PNG = ROOT / "figures/pvfig2.png"
OUTPUT_PDF = ROOT / "figures/pvfig2.pdf"


def read_gaps(path: Path) -> pd.Series:
    return pd.to_numeric(pd.read_csv(path)["band_gap_eV"], errors="coerce").dropna()


def plot_distribution(ax: plt.Axes, gaps: pd.Series, panel: str) -> None:
    ax.hist(gaps, bins=25, density=True, alpha=0.45, color=PRIMARY_BLUE, edgecolor="none")
    grid = np.linspace(gaps.min() - 0.1, gaps.max() + 0.1, 512)
    ax.plot(grid, gaussian_kde(gaps)(grid), color=PRIMARY_BLUE, linewidth=2.5)
    ax.axvline(gaps.mean(), color=ACCENT_RED, linestyle=":", linewidth=2,
               label=f"Mean = {gaps.mean():.2f} eV")
    ax.set_xlabel("DFT Band Gap (eV)")
    ax.set_ylabel("Density")
    ax.set_xlim(-0.15, 5.05)
    ax.text(0.03, 0.95, panel, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left")
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="#CCCCCC",
              loc="upper right")


def main() -> None:
    all_gaps = read_gaps(ALL_DATA)
    complete_gaps = read_gaps(COMPLETE_XTB)
    if (len(all_gaps), len(complete_gaps)) != (315, 216):
        raise ValueError(f"Expected N=315 and N=216; found {len(all_gaps)} and {len(complete_gaps)}")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=200, sharey=True)
    plot_distribution(axes[0], all_gaps, "(a)")
    plot_distribution(axes[1], complete_gaps, "(b)")
    fig.tight_layout(w_pad=2.3)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG} and {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
