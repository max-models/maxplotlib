"""Calls that work on Matplotlib's ``Axes`` and work the same on a subplot:
format strings, short keyword names, shared axes, scale options, text in
axes coordinates and ``legend()``."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from maxplotlib import Canvas  # noqa: E402
from maxplotlib.utils.plot_format import (  # noqa: E402
    canonical_line_kwargs,
    parse_plot_format,
)

X = np.linspace(0.0, 1.0, 5)


def _mpl_axes(canvas):
    fig, axes = canvas.render(backend="matplotlib")
    return fig, np.asarray(axes)


# --- format strings ---------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt, expected",
    [
        ("o", {"marker": "o", "linestyle": "None"}),
        ("x--", {"marker": "x", "linestyle": "--"}),
        ("--x", {"marker": "x", "linestyle": "--"}),
        ("k:", {"color": "k", "linestyle": ":"}),
        ("-.", {"linestyle": "-."}),
        ("C1s-", {"color": "C1", "marker": "s", "linestyle": "-"}),
        ("ro", {"color": "r", "marker": "o", "linestyle": "None"}),
        ("C3", {"color": "C3"}),
        ("red", {"color": "red"}),
        ("0.5", {"color": "0.5"}),
        ("1", {"marker": "1", "linestyle": "None"}),
        ("", {}),
    ],
)
def test_parse_plot_format(fmt, expected):
    assert parse_plot_format(fmt) == expected


@pytest.mark.parametrize("fmt", ["oo", "---", "kr", "q", "o?"])
def test_parse_plot_format_rejects_what_matplotlib_rejects(fmt):
    with pytest.raises(ValueError, match="format string"):
        parse_plot_format(fmt)


def test_short_keyword_names_are_spelled_out():
    assert canonical_line_kwargs(
        {"ls": ":", "lw": 2, "ms": 3, "c": "k", "mfc": "none"}
    ) == {
        "linestyle": ":",
        "linewidth": 2,
        "markersize": 3,
        "color": "k",
        "markerfacecolor": "none",
    }
    with pytest.raises(TypeError, match="aliases"):
        canonical_line_kwargs({"ls": ":", "linestyle": "-"})


def test_a_format_string_draws_as_in_matplotlib():
    canvas, ax = Canvas.subplots()
    ax.plot(X, X, "o")
    ax.plot(X, 2 * X, "x--", color="C3")
    ax.plot(X, 3 * X, "k:", ls="-")  # the keyword wins over the format string
    fig, axes = _mpl_axes(canvas)
    markers, dashed, solid = axes[0, 0].get_lines()
    assert (markers.get_marker(), markers.get_linestyle()) == ("o", "None")
    assert (dashed.get_marker(), dashed.get_linestyle()) == ("x", "--")
    assert matplotlib.colors.same_color(dashed.get_color(), "C3")
    assert solid.get_linestyle() == "-"
    assert matplotlib.colors.same_color(solid.get_color(), "k")
    plt.close(fig)


def test_plot_against_the_index_and_from_the_canvas():
    canvas = Canvas()
    canvas.plot([3.0, 1.0, 2.0])
    canvas.plot([3.0, 1.0, 2.0], "s")
    canvas.plot(X, X, "v-", lw=3)
    fig, axes = _mpl_axes(canvas)
    index, squares, wide = axes[0, 0].get_lines()
    np.testing.assert_allclose(index.get_xdata(), [0, 1, 2])
    np.testing.assert_allclose(index.get_ydata(), [3, 1, 2])
    assert (squares.get_marker(), squares.get_linestyle()) == ("s", "None")
    assert (wide.get_marker(), wide.get_linewidth()) == ("v", 3)
    plt.close(fig)


def test_a_positional_layer_is_refused():
    _, ax = Canvas.subplots()
    with pytest.raises(TypeError, match="layer="):
        ax.plot(X, X, 1)
    ax.plot(X, X, layer=1)
    assert ax.layers == [1]


def test_markers_only_in_plotly_plotext_and_tikz():
    canvas, ax = Canvas.subplots()
    ax.plot(X, X, "o", ms=7, label="markers")
    ax.plot(X, 2 * X, "s-", lw=2.5)
    plotly_fig = canvas.render(backend="plotly")
    assert plotly_fig.data[0].mode == "markers"
    assert plotly_fig.data[0].marker.size == 7
    assert plotly_fig.data[1].mode == "lines+markers"
    assert plotly_fig.data[1].line.width == 2.5
    assert canvas.render(backend="plotext").build()
    tikz = canvas.render(backend="tikzfigure").generate_tikz()
    assert "only marks" in tikz


# --- shared axes --------------------------------------------------------------------


def test_shared_axes_in_matplotlib():
    canvas, axes = Canvas.subplots(nrows=2, ncols=2, sharex=True, sharey="row")
    axes[0][0].plot([0, 1], [0, 1])
    axes[1][1].plot([0, 10], [0, 5])
    fig, m = _mpl_axes(canvas)
    assert m[0, 0].get_xlim() == m[1, 1].get_xlim()  # x shared by all
    assert m[0, 0].get_ylim() == m[0, 1].get_ylim()  # y shared within a row
    assert m[0, 0].get_ylim() != m[1, 0].get_ylim()
    fig.canvas.draw()
    assert not any(t.get_visible() for t in m[0, 0].get_xticklabels())
    assert any(t.get_visible() for t in m[1, 0].get_xticklabels())
    assert not any(t.get_visible() for t in m[1, 1].get_yticklabels())
    plt.close(fig)


def test_shared_columns_keep_their_own_range():
    canvas, axes = Canvas.subplots(nrows=2, ncols=2, sharex="col")
    axes[0][0].plot([0, 1], [0, 1])
    axes[1][1].plot([0, 10], [0, 5])
    fig, m = _mpl_axes(canvas)
    assert m[0, 0].get_xlim() == m[1, 0].get_xlim()
    assert m[0, 1].get_xlim() == m[1, 1].get_xlim()
    assert m[0, 0].get_xlim() != m[0, 1].get_xlim()
    plt.close(fig)


def test_shared_axes_in_plotly_and_tikz():
    canvas, (top, bottom) = Canvas.subplots(nrows=2, sharex=True)
    top.plot([0, 1], [0, 1])
    bottom.plot([0, 10], [0, 5])
    fig = canvas.render(backend="plotly")
    # one x axis follows the other (Plotly anchors the group on the bottom row)
    assert {fig.layout.xaxis.matches, fig.layout.xaxis2.matches} - {None}
    unshared = Canvas(nrows=2)
    unshared.plot([0, 1], [0, 1], row=0, col=0)
    unshared.plot([0, 10], [0, 5], row=1, col=0)
    layout = unshared.render(backend="plotly").layout
    assert layout.xaxis.matches is None and layout.xaxis2.matches is None
    tikz = canvas.render(backend="tikzfigure").generate_tikz()
    limits = [
        line.split("xmin=")[1].split(",")[0]
        for line in tikz.splitlines()
        if "xmin=" in line
    ]
    assert len(limits) == 2 and limits[0] == limits[1]


def test_an_unknown_sharing_is_refused():
    with pytest.raises(ValueError, match="sharex"):
        Canvas(nrows=2, sharex="columns")


# --- scale options, text coordinates, legend --------------------------------------------


def test_scale_options_reach_matplotlib():
    canvas, ax = Canvas.subplots()
    ax.plot([-10, 10], [-10, 10])
    ax.set_yscale("symlog", linthresh=0.5)
    canvas.set_xscale("log", base=2)
    fig, axes = _mpl_axes(canvas)
    assert axes[0, 0].yaxis.get_transform().linthresh == 0.5
    assert axes[0, 0].xaxis.get_transform().base == 2
    plt.close(fig)


def test_text_in_axes_and_figure_coordinates():
    canvas, ax = Canvas.subplots()
    ax.plot([0, 100], [0, 100])
    ax.text(0.5, 0.9, "in the axes", transform="axes", ha="center")
    ax.text(0.01, 0.01, "in the figure", transform="figure")
    ax.text(50, 50, "in the data")
    fig, axes = _mpl_axes(canvas)
    in_axes, in_figure, in_data = axes[0, 0].texts
    assert in_axes.get_transform() == axes[0, 0].transAxes
    assert in_figure.get_transform() == fig.transFigure
    assert in_data.get_transform() == axes[0, 0].transData
    plt.close(fig)

    annotations = {
        a.text: a for a in canvas.render(backend="plotly").layout.annotations
    }
    assert annotations["in the axes"].xref == "x domain"
    assert annotations["in the axes"].y == 0.9
    assert annotations["in the figure"].xref == "paper"
    assert annotations["in the data"].xref == "x"

    assert "in the axes" in canvas.render(backend="tikzfigure").generate_tikz()


def test_an_unknown_text_coordinate_system_is_refused():
    canvas, ax = Canvas.subplots()
    ax.text(0.5, 0.5, "where?", transform="polar")
    with pytest.raises(ValueError, match="transform"):
        canvas.render(backend="matplotlib")
    with pytest.raises(NotImplementedError, match="polar"):
        canvas.render(backend="plotly")


def test_legend_is_a_method_as_in_matplotlib():
    canvas, (left, right) = Canvas.subplots(ncols=2)
    left.plot(X, X, label="a")
    right.plot(X, X, label="b")
    left.legend(loc="lower right", fontsize=7)
    canvas.legend(row=0, col=1, title="right")
    fig, axes = _mpl_axes(canvas)
    assert axes[0, 0].get_legend().get_texts()[0].get_fontsize() == 7
    assert axes[0, 1].get_legend().get_title().get_text() == "right"
    plt.close(fig)
    left.set_legend(False)
    fig, axes = _mpl_axes(canvas)
    assert axes[0, 0].get_legend() is None
    plt.close(fig)
