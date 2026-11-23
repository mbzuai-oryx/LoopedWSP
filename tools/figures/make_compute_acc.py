"""Maze-Hard test accuracy against training compute (PFLOPs, log scale).
Trained models, any shortest path counts: LDT, TRM, HRM from Davis et al. (NeurIPS 2026), Table 3 (training compute up
to the reported checkpoint); EBR from Chen et al. (2026), Table D.1 (94.07% any shortest path, 1 sample) and Table B.1
(1.16 EFLOPs, full run). HRM: 82.2%, Davis et al.'s retrain without augmentation (Section 4.3);
compute 10,122 PFLOPs = our count of its recipe (logs/baselines_train/hrm_count_flops.py: 3.89e14 FLOPs per step at
batch 768 x 26,042 steps; the same per-step count gives Davis et al.'s 46,944 PFLOPs at their ~122K steps). Gradient descent on the architecture of the transformer Claude wrote: about 5 PFLOPs per run (5.02-5.09e15
FLOPs, REVIEW_NOTES), 0% of test mazes. The transformer Claude wrote: no gradient training, 100%. Saves compute_acc.*"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figures as mf  # noqa: E402

GREY, GREY_TEXT, GD_C, GD_TEXT = "#8f8e88", "#5f5e59", "#2c2c2c", "#2c2c2c"
HRM_TEXT = "#11805a"
X0 = 0.45                      # where "no training" sits, left of the break


def main() -> None:
    plt = mf.setup_matplotlib()
    from matplotlib.ticker import FixedLocator, NullLocator
    fs = mf.F * 8.3
    fig = plt.figure(figsize=(5.6, 4.7))
    ax = fig.add_axes([0.14, 0.17, 0.82, 0.78])
    ax.set_xscale("log")
    ax.set_xlim(0.25, 7e5)
    ax.set_ylim(-9, 112)
    ax.xaxis.set_major_locator(FixedLocator([X0, 1, 10, 100, 1e3, 1e4, 1e5]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["0", "1", "10", "100", "1K", "10K", "100K"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=mf.GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(mf.GRID)
    ax.tick_params(length=0, labelsize=fs, pad=5)
    ax.set_xlabel("Training compute (PFLOPs, log scale)", fontsize=fs, labelpad=6)
    ax.set_ylabel("Test mazes solved (%)", fontsize=fs, labelpad=8)
    # break marks between "0" and the log scale
    for xb in (0.62, 0.7):
        ax.plot([xb * 0.97, xb * 1.03], [-11.5, -6.5], color=mf.MUTED, lw=1.0, clip_on=False)

    # the transformer Claude wrote: no training compute, 100%
    ax.scatter([X0], [100], s=420, marker="*", color=mf.WRITTEN, edgecolor="white", linewidth=1.0, zorder=6,
               clip_on=False, gid="data")
    t = ax.annotate("Written by Claude, 100%", (X0, 100), xytext=(13, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=fs * 1.04, color=mf.WRITTEN_TEXT, fontweight="semibold", gid="data")
    # gradient descent on the same architecture
    ax.scatter([5], [0], s=70, marker="X", color=GD_C, edgecolor="white", linewidth=0.8, zorder=6, gid="data")
    ax.annotate("Same architecture,\ngradient descent, 0%", (5, 0), xytext=(9, 4), textcoords="offset points",
                ha="left", va="bottom", fontsize=fs * 0.9, color=GD_TEXT, linespacing=1.2, gid="data")
    # trained models (Davis et al., Table 3)
    pts = [  # x PFLOPs, y %, colour, text colour, label, logo, offset, ha, hollow
        (1303, 99.8, GREY, GREY_TEXT, "LDT, 99.8%", None, (0, 11), "center", False),
        (1160, 94.1, "#c2338a", "#c2338a", "EBR, 94.1%", None, (-10, 0), "right", False),
        (5971, 91.9, mf.TRAINED, mf.TRAINED_TEXT, "TRM, 91.9%", "samsung", (10, 0), "left", False),
        (10122, 82.2, mf.AQUA, HRM_TEXT, "HRM, 82.2%", "sapient", (10, 0), "left", False),
    ]
    for x, y, col, tcol, lab, logo, off, ha, hollow in pts:
        if hollow:
            ax.scatter([x], [y], s=54, facecolor="white", edgecolor=col, linewidth=1.6, zorder=5, gid="data")
        else:
            ax.scatter([x], [y], s=80, color=col, edgecolor="white", linewidth=1.6, zorder=5, gid="data")
        tt = ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", ha=ha,
                         va="bottom" if off[1] > 0 else ("top" if off[1] < 0 else "center"),
                         fontsize=fs * 0.95, color=tcol, gid="data")
        if logo:
            mf.add_logos(fig, ax, tt, [logo], shift=(ha == "left"))
    # the compute gap
    ax.annotate("", xy=(1160, 62), xytext=(5, 62),
                arrowprops=dict(arrowstyle="<->", color=mf.INK, lw=1.3, shrinkA=0, shrinkB=0), gid="data")
    ax.text((5 * 1160) ** 0.5, 64.5, "230× to 2,000× less compute", fontsize=fs * 1.05, color=mf.INK,
            fontweight="semibold", ha="center", va="bottom", gid="data")
    ax.text((5 * 1160) ** 0.5, 59.5, "in our gradient descent runs", fontsize=fs * 0.85, color=mf.MUTED, ha="center",
            va="top", gid="data")
    mf.save(fig, "compute_acc", "Maze-Hard accuracy against training compute")
    plt.close(fig)


if __name__ == "__main__":
    main()
