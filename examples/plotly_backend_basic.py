import numpy as np

from maxplotlib import Canvas


def main() -> None:
    x = np.linspace(0, 2 * np.pi, 200)

    canvas = Canvas(width="12cm", ratio=0.5)
    canvas.add_line(x, np.sin(x), color="royalblue", label="sin(x)")
    canvas.scatter(x[::12], np.sin(x[::12]), color="tomato", label="samples")
    canvas.axhline(0, color="black", linestyle="dotted")
    canvas.set_title("Plotly backend (basic)")
    canvas.set_xlabel("x")
    canvas.set_ylabel("y")
    canvas.set_grid(True)
    canvas.set_legend(True)

    canvas.savefig("plotly_basic.html", backend="plotly")


if __name__ == "__main__":
    main()
