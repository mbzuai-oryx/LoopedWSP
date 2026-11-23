"""The small task signs from Figure 13 as standalone SVGs, for the run-in headings in the text
(figures/ico_<task>.svg). Same drawing code as the figure (make_figures.pictogram)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figures as mf  # noqa: E402

TASKS = {"Graph search": "graph_search", "Parity": "parity", "Reasoning chains": "reasoning_chains"}


def main() -> None:
    plt = mf.setup_matplotlib()
    pad = 0.015
    W, H = mf.PIC_W + 2 * pad, mf.PIC_H + 2 * pad
    for task, slug in TASKS.items():
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        mf.pictogram(ax, task, pad, H / 2)
        out = HERE / f"ico_{slug}.svg"
        fig.savefig(out, transparent=True, metadata={"Title": task, "Date": None, "Creator": None})
        plt.close(fig)
        print("wrote", out.name)


if __name__ == "__main__":
    main()
