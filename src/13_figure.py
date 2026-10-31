"""
Step 12 — The headline figure.

Two panels, in causal order:
  A  why it happens: the JAK1 model's predictions are compressed. It never
     predicts above 9.40, while measurement reaches 10.64.
  B  what it costs: compounds whose JAK activity was predicted rather than
     measured appear TYK2-selective twice as often.

Panel A is the mechanism and panel B the consequence, so they are read left to
right as one sentence.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = ROOT / "figures"

# Validated categorical slots 1 and 2 (light surface #fcfcfb).
C_MEASURED = "#2a78d6"
C_PREDICTED = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dcdbd6"


def worst_source(r):
    col = f"{r['JAK_worst']}_source"
    return r[col] if col in r else "predicted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tyk2ic50")
    args = ap.parse_args()

    d = pd.read_csv(RESULTS / f"{args.target}_selectivity.csv")
    d["worst_measured"] = d.apply(worst_source, axis=1) == "measured"

    meas = d[d.JAK1_source == "measured"]
    pred = d[d.JAK1_source == "predicted"]
    sel_m = d[d.worst_measured]
    sel_p = d[~d.worst_measured]

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.size": 9, "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "grid.color": GRID,
    })
    # Three panels, each with one job: the mechanism, the distribution it
    # distorts, and the headline number. An inset over the histogram was
    # tried first and read as marks floating on the data.
    fig = plt.figure(figsize=(12.6, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.5], wspace=0.30)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                     fig.add_subplot(gs[2]))

    # ---- A: the mechanism ----------------------------------------------
    bins = np.linspace(3.5, 11, 46)
    ax1.hist(meas.JAK1_pIC50, bins=bins, color=C_MEASURED, alpha=0.75,
             label=f"measured  (n={len(meas):,})", density=True, linewidth=0)
    ax1.hist(pred.JAK1_pIC50, bins=bins, color=C_PREDICTED, alpha=0.75,
             label=f"predicted  (n={len(pred):,})", density=True, linewidth=0)
    ax1.axvline(pred.JAK1_pIC50.max(), color=C_PREDICTED, lw=2, ls="--")
    ax1.axvline(meas.JAK1_pIC50.max(), color=C_MEASURED, lw=2, ls="--")
    y1 = ax1.get_ylim()[1]
    ax1.set_ylim(0, y1 * 1.34)      # headroom for the two callouts
    ax1.annotate(f"prediction stops\nat {pred.JAK1_pIC50.max():.2f}",
                 xy=(pred.JAK1_pIC50.max() - 0.18, y1 * 1.05),
                 ha="right", va="top", fontsize=8.5, color=C_PREDICTED,
                 weight="bold")
    ax1.annotate(f"measurement\nreaches {meas.JAK1_pIC50.max():.2f}",
                 xy=(meas.JAK1_pIC50.max() + 0.18, y1 * 1.05),
                 ha="left", va="top", fontsize=8.5, color=C_MEASURED,
                 weight="bold")
    ax1.set_xlim(3.5, 12.6)
    ax1.set_xticks([4, 5, 6, 7, 8, 9, 10, 11])
    ax1.set_xlabel("JAK1 activity  (pIC50)")
    ax1.set_ylabel("density of compounds")
    ax1.set_title("A.  The model cannot predict a potent JAK1 inhibitor",
                  loc="left", fontsize=10.5, weight="bold", color=INK, pad=10)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")

    # ---- B: the distorted distribution ---------------------------------
    bins2 = np.linspace(-4.5, 2, 46)
    pm = (sel_m.selectivity > 0).mean() * 100
    pp = (sel_p.selectivity > 0).mean() * 100
    ax2.hist(sel_m.selectivity, bins=bins2, color=C_MEASURED, alpha=0.7,
             label=f"worst JAK measured  (n={len(sel_m):,})",
             density=True, linewidth=0)
    ax2.hist(sel_p.selectivity, bins=bins2, color=C_PREDICTED, alpha=0.7,
             label=f"worst JAK predicted  (n={len(sel_p):,})",
             density=True, linewidth=0)
    y2 = ax2.get_ylim()[1]
    ax2.axvspan(0, 2, color="#0b0b0b", alpha=0.06, zorder=0)
    ax2.axvline(0, color=INK, lw=1.3)
    ax2.text(1.0, y2 * 0.045, "looks TYK2-selective", ha="center",
             fontsize=8.5, color=INK_2, style="italic")
    ax2.set_xlabel("selectivity margin   (TYK2 pIC50 \u2212 best JAK pIC50)")
    ax2.set_ylabel("density of compounds")
    ax2.set_title("B.  The whole distribution shifts right",
                  loc="left", fontsize=10.5, weight="bold", color=INK, pad=10)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")

    # ---- C: the headline -----------------------------------------------
    ax3.barh([1, 0], [pm, pp], color=[C_MEASURED, C_PREDICTED], height=0.42)
    for y, v in [(1, pm), (0, pp)]:
        ax3.text(v + 0.8, y, f"{v:.1f}%", va="center", fontsize=12,
                 weight="bold", color=INK)
    ax3.set_yticks([1, 0])
    ax3.set_yticklabels(["JAK activity\nmeasured", "JAK activity\npredicted"],
                        fontsize=9, color=INK_2)
    ax3.set_xlim(0, 30)
    ax3.set_ylim(-0.6, 1.6)
    ax3.set_xticks([])
    ax3.set_title("C.  Appear selective", loc="left",
                  fontsize=10.5, weight="bold", color=INK, pad=10)
    ax3.tick_params(length=0)

    for ax in (ax1, ax2):
        ax.grid(axis="y", lw=0.6, alpha=0.5)
        ax.set_axisbelow(True)
    for ax in (ax1, ax2, ax3):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax3.spines["bottom"].set_visible(False)
    ax3.spines["left"].set_color(GRID)

    fig.suptitle("Apparent selectivity is concentrated in the compounds nobody measured",
                 x=0.008, ha="left", fontsize=13, weight="bold", color=INK, y=1.02)
    fig.text(0.008, -0.05,
             f"TYK2 screen, {len(d):,} candidates. A JAK1 model trained on 7,758 "
             f"ChEMBL compounds regresses toward the mean, so it never predicts the "
             f"high potencies\nmeasurement finds. A compound nobody tested therefore "
             f"scores as selective by default.   github.com/Saantzai/tyk2-discovery",
             fontsize=8, color=INK_2, ha="left", va="top")

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "selectivity_illusion.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"written -> figures/{out.name}")
    print(f"  measured  JAK1 max {meas.JAK1_pIC50.max():.2f}, "
          f"selective {pm:.1f}%")
    print(f"  predicted JAK1 max {pred.JAK1_pIC50.max():.2f}, "
          f"selective {pp:.1f}%")


if __name__ == "__main__":
    main()
