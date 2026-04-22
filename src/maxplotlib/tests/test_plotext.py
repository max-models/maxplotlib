import re

import matplotlib.patches as mpatches
import numpy as np

from maxplotlib import Canvas
from maxplotlib.backends.plotext import PlotextFigure

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def test_canvas_plot_plotext_builds_terminal_output():
    x = np.linspace(0, 2 * np.pi, 40)
    canvas, ax = Canvas.subplots(width="10cm", ratio=0.5)

    ax.plot(x, np.sin(x), color="blue", label="sin(x)")
    ax.scatter(x[::8], np.cos(x[::8]), color="red", label="samples")
    ax.axhline(0, color="white")
    ax.set_title("Terminal sine")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_grid(True)
    ax.set_legend(True)
    canvas.suptitle("Plotext demo")

    figure = canvas.plot(backend="plotext")
    output = strip_ansi(figure.build())

    assert isinstance(figure, PlotextFigure)
    assert "Plotext demo" in output
    assert "Terminal sine" in output
    assert "sin(x)" in output
    assert "samples" in output


def test_canvas_show_plotext_prints_output(capsys):
    x = np.linspace(0, 1, 12)
    canvas, ax = Canvas.subplots()
    ax.plot(x, x**2, color="green")
    ax.set_title("Quadratic")

    figure = canvas.show(backend="plotext")
    captured = strip_ansi(capsys.readouterr().out)

    assert isinstance(figure, PlotextFigure)
    assert "Quadratic" in captured


def test_canvas_plot_plotext_supports_scalar_errorbars():
    x = np.linspace(1, 10, 10)
    canvas, ax = Canvas.subplots()
    ax.errorbar(x, np.sqrt(x), yerr=0.2, color="yellow", label="samples")
    ax.set_xscale("log")
    ax.set_title("Log errors")

    output = strip_ansi(canvas.plot(backend="plotext").build())

    assert "Log errors" in output


def test_canvas_plot_plotext_supports_fill_between_curves_and_annotations():
    x = np.linspace(0, 4, 25)
    canvas, ax = Canvas.subplots()
    ax.fill_between(x, np.sin(x) + 1.5, np.cos(x) + 0.5, color="cyan", label="band")
    ax.annotate("crossing", xy=(1.5, 1.0), xytext=(2.5, 2.1), arrowprops={"color": "yellow"})
    ax.set_title("Filled band")
    ax.set_legend(True)

    output = strip_ansi(canvas.plot(backend="plotext").build())

    assert "Filled band" in output
    assert "band" in output
    assert "crossing" in output


def test_canvas_plot_plotext_supports_matrix_plots_and_patches():
    canvas, ax = Canvas.subplots()
    ax.add_imshow(np.arange(9).reshape(3, 3))
    ax.add_patch(mpatches.Rectangle((0.2, 0.2), 1.2, 0.8, fill=False, edgecolor="yellow"))
    ax.add_patch(mpatches.Circle((1.8, 1.8), 0.4, fill=False, edgecolor="cyan"))
    ax.set_title("Matrix plot")

    output = strip_ansi(canvas.plot(backend="plotext").build())

    assert "Matrix plot" in output


def test_canvas_plot_plotext_supports_colorbar_notes_symlog_aspect_and_generic_patches():
    canvas, ax = Canvas.subplots()
    ax.add_imshow(np.eye(3))
    ax.add_colorbar(label="scale")
    ax.set_title("Heatmap")
    output = strip_ansi(canvas.plot(backend="plotext").build())

    assert "Heatmap" in output
    assert "scale:" in output

    x = np.linspace(-20, 20, 81)
    canvas, ax = Canvas.subplots()
    ax.plot(x, x**3, color="cyan")
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.set_aspect("equal")
    ax.add_caption("caption text")
    ax.set_title("Symlog view")
    output = strip_ansi(canvas.plot(backend="plotext").build())

    assert "Symlog view" in output
    assert "caption text" in output

    canvas, ax = Canvas.subplots()
    ax.add_patch(
        mpatches.Ellipse(
            (1.5, 1.0),
            2.0,
            1.0,
            fill=False,
            edgecolor="yellow",
            label="ellipse",
        )
    )
    ax.set_title("Generic patch")
    ax.set_legend(True)
    output = strip_ansi(canvas.plot(backend="plotext").build())

    assert "Generic patch" in output
    assert "ellipse" in output
