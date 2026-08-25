import re

import matplotlib.patches as mpatches
import numpy as np
import pytest

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

    figure = canvas.render(backend="plotext")
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

    output = strip_ansi(canvas.render(backend="plotext").build())

    assert "Log errors" in output


def test_canvas_plot_plotext_supports_fill_between_curves_and_annotations():
    x = np.linspace(0, 4, 25)
    canvas, ax = Canvas.subplots()
    ax.fill_between(x, np.sin(x) + 1.5, np.cos(x) + 0.5, color="cyan", label="band")
    ax.annotate(
        "crossing", xy=(1.5, 1.0), xytext=(2.5, 2.1), arrowprops={"color": "yellow"}
    )
    ax.set_title("Filled band")
    ax.set_legend(True)

    output = strip_ansi(canvas.render(backend="plotext").build())

    assert "Filled band" in output
    assert "band" in output
    assert "crossing" in output


def test_canvas_plot_plotext_supports_matrix_plots_and_patches():
    canvas, ax = Canvas.subplots()
    ax.add_imshow(np.arange(9).reshape(3, 3))
    ax.add_patch(
        mpatches.Rectangle((0.2, 0.2), 1.2, 0.8, fill=False, edgecolor="yellow")
    )
    ax.add_patch(mpatches.Circle((1.8, 1.8), 0.4, fill=False, edgecolor="cyan"))
    ax.set_title("Matrix plot")

    output = strip_ansi(canvas.render(backend="plotext").build())

    assert "Matrix plot" in output


def test_canvas_plot_plotext_supports_colorbar_notes_symlog_aspect_and_generic_patches():
    canvas, ax = Canvas.subplots()
    ax.add_imshow(np.eye(3))
    ax.add_colorbar(label="scale")
    ax.set_title("Heatmap")
    output = strip_ansi(canvas.render(backend="plotext").build())

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
    output = strip_ansi(canvas.render(backend="plotext").build())

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
    output = strip_ansi(canvas.render(backend="plotext").build())

    assert "Generic patch" in output
    assert "ellipse" in output


def test_plotext_supports_axis_controls_and_line_primitives():
    canvas, ax = Canvas.subplots()
    ax.plot([0, 1], [0, 1], label="line")
    ax.hlines([0.25, 0.75], 0, 1, color="yellow")
    ax.vlines([0.25, 0.75], 0, 1, color="cyan")
    ax.axhline(0.5, color="red")
    ax.axvline(0.5, color="blue")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1], labels=["left", "middle", "right"])
    ax.set_yticks([0, 0.5, 1], labels=["low", "mid", "high"])
    ax.set_title("Axis controls")
    ax.set_xlabel("horizontal")
    ax.set_ylabel("vertical")
    ax.set_grid(True)

    output = strip_ansi(canvas.render(backend="plotext").build())

    for text in (
        "Axis controls",
        "horizontal",
        "vertical",
        "left",
        "middle",
        "right",
        "low",
        "mid",
        "high",
    ):
        assert text in output


def test_plotext_supports_multiple_subplots():
    canvas, axes = Canvas.subplots(nrows=1, ncols=2)
    axes[0].plot([0, 1], [0, 1])
    axes[0].set_title("left subplot")
    axes[1].bar([0, 1], [1, 2])
    axes[1].set_title("right subplot")

    output = strip_ansi(canvas.render(backend="plotext").build())

    assert "left subplot" in output
    assert "right subplot" in output


def test_plotext_layer_filtering_changes_rendered_content():
    canvas, ax = Canvas.subplots()
    ax.plot([0, 1], [0, 1], label="first layer", layer=0)
    ax.plot([0, 1], [1, 0], label="second layer", layer=1)
    ax.set_legend(True)

    first = strip_ansi(canvas.render(backend="plotext", layers=[0]).build())
    second = strip_ansi(canvas.render(backend="plotext", layers=[1]).build())

    assert "first layer" in first
    assert "second layer" not in first
    assert "second layer" in second
    assert "first layer" not in second


def test_plotext_savefig_writes_plain_text_and_supports_append(tmp_path):
    canvas, ax = Canvas.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("Saved output")
    output_file = tmp_path / "plot.txt"

    figure = canvas.render(backend="plotext")
    figure.savefig(output_file, keep_colors=False)
    first = output_file.read_text(encoding="utf-8")
    figure.savefig(output_file, append=True, keep_colors=False)
    combined = output_file.read_text(encoding="utf-8")

    assert "Saved output" in first
    assert "\x1b[" not in first
    assert combined == first + first


def test_plotext_rejects_unsupported_twinx():
    canvas, ax = Canvas.subplots()
    ax.plot([0, 1], [0, 1])
    canvas.twinx()

    with pytest.raises(NotImplementedError, match="twinx"):
        canvas.render(backend="plotext")


def test_plotext_supports_asymmetric_errors_baseline_fill_and_annotations():
    canvas, ax = Canvas.subplots()
    x = np.arange(4)
    ax.fill_between(x, [1, 2, 3, 2], 0, label="baseline")
    ax.errorbar(
        x,
        [1, 2, 1, 3],
        xerr=[[0.1, 0.2, 0.1, 0.2], [0.2, 0.1, 0.2, 0.1]],
        yerr=[[0.2, 0.1, 0.2, 0.1], [0.1, 0.2, 0.1, 0.2]],
        label="measurements",
    )
    ax.annotate("peak", xy=(3, 3), xytext=(2, 2.5), arrowprops={"color": "red"})
    ax.set_title("Uncertainty and annotations")
    ax.set_legend(True)

    output = strip_ansi(canvas.render(backend="plotext").build())

    assert "Uncertainty and annotations" in output
    assert "baseline" in output
    assert "measurements" in output
    assert "peak" in output


def test_plotext_supports_gantt_and_flame_chart_labels():
    canvas, ax = Canvas.subplots()
    ax.gantt(
        ["plan", "build", "test"],
        [0, 1, 3],
        [1, 2, 1],
    )
    ax.set_title("Gantt")
    gantt_output = strip_ansi(canvas.render(backend="plotext").build())

    canvas, ax = Canvas.subplots()
    ax.flame_chart(
        labels=["root", "worker", "io"],
        parents=[None, 0, 1],
        values=[5, 3, 1],
        start_times=[0, 0, 2],
    )
    ax.set_title("Flame")
    flame_output = strip_ansi(canvas.render(backend="plotext").build())

    assert "Gantt" in gantt_output
    assert all(label in gantt_output for label in ("plan", "build", "test"))
    assert "Flame" in flame_output


def test_plotext_rejects_unsupported_plot_types_and_imshow_options():
    canvas, ax = Canvas.subplots()
    ax.hist([1, 2, 2, 3])
    with pytest.raises(NotImplementedError, match="plot type: hist"):
        canvas.render(backend="plotext")

    canvas, ax = Canvas.subplots()
    ax.add_imshow(np.eye(3), cmap="viridis")
    with pytest.raises(NotImplementedError, match="imshow kwargs: cmap"):
        canvas.render(backend="plotext")


def test_plotext_empty_canvas_still_builds():
    figure = Canvas().render(backend="plotext")

    assert isinstance(figure, PlotextFigure)
    assert isinstance(figure.build(keep_colors=False), str)
