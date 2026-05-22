import matplotlib.patches as mpatches
import numpy as np

from maxplotlib import Canvas


def main() -> None:
    x = np.linspace(0.5, 10, 60)
    y = np.sqrt(x)

    canvas = Canvas(width="12cm", ratio=0.55)

    canvas.add_line(x, y, color="steelblue", label="sqrt(x)")
    canvas.errorbar(
        x[::10],
        y[::10],
        yerr=0.15,
        color="tomato",
        marker="o",
        label="samples ± err",
    )
    canvas.fill_between(
        x, y - 0.1, y + 0.1, color="steelblue", alpha=0.2, label="band"
    )
    canvas.vlines([2, 5, 8], ymin=0, ymax=3.5, color="gray", linestyle="dashed")
    canvas.text(7.2, 2.8, "note", color="purple")
    canvas.annotate(
        "peak-ish", xy=(9.5, np.sqrt(9.5)), xytext=(6.0, 3.1), color="purple"
    )

    canvas.add_patch(
        mpatches.Rectangle((1.2, 0.0), 2.5, 1.2, fill=True),
        facecolor="rgba(255,0,0,0.1)",
        edgecolor="crimson",
        alpha=0.3,
    )

    canvas.set_title("Plotly backend (parity features)")
    canvas.set_xlabel("x")
    canvas.set_ylabel("y")
    canvas.set_xscale("log")
    canvas.set_grid(True)
    canvas.set_legend(True)

    canvas.savefig("plotly_parity.html", backend="plotly")


if __name__ == "__main__":
    main()
