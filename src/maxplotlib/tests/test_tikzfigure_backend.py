"""The tikzfigure backend: drawn Matplotlib figures as pgfplots axes."""

import re
import shutil
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from maxplotlib import Canvas  # noqa: E402
from maxplotlib.backends.tikzfigure import (  # noqa: E402
    TikzConversionWarning,
    figure_to_tikz,
    latex,
)

needs_pdflatex = pytest.mark.skipif(
    shutil.which("pdflatex") is None, reason="pdflatex not installed"
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def axes_options(tikz):
    """The option lists of every ``axis`` environment."""
    return re.findall(r"\\begin\{axis\}\[(.*?)\]\n", tikz)


# -- text ------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("plain", "plain"),
        ("50% of a_b & #1", r"50\% of a\_b \& \#1"),
        (r"$\gamma/\omega_{ci}$ fit", r"$\gamma/\omega_{ci}$ fit"),
        (r"$\mathdefault{10^{-3}}$", "${10^{-3}}$"),
        (r"$\mathdefault{0.5}$", "$0.5$"),
        ("ω = 2", r"$\omega$ = 2"),
        (r"$ω_p$", r"$\omega _p$"),
        ("−1", "$-$1"),
        ("cost: $5", r"cost: \$5"),
        ("two\nlines", r"two\\lines"),
        ("", ""),
        (None, ""),
    ],
)
def test_latex(text, expected):
    assert latex(text) == expected


# -- what is drawn ---------------------------------------------------------
def test_lines_markers_and_styles():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0], "--", color="red", lw=2)
    ax.plot([0, 1, 2], [1, 2, 1], "s", ms=6, mfc="none", mec="blue")
    tikz = figure_to_tikz(fig).generate_tikz()
    assert "\\definecolor{mplFF0000}{HTML}{FF0000}" in tikz
    assert "draw=mplFF0000, line width=2pt, dash pattern=on 7.4pt off 3.2pt" in tikz
    assert "only marks, mark=square, mark size=3pt" in tikz
    assert "fill=none, draw=mpl0000FF" in tikz


def test_the_legend_keeps_matplotlib_order_and_style():
    fig, ax = plt.subplots()
    ax.fill_between([0, 1], [0, 1], alpha=0.5, label="area")
    ax.plot([0, 1], [1, 0], color="k", label="line")
    ax.legend(handles=ax.lines + ax.collections, loc="upper left")
    tikz = figure_to_tikz(fig).generate_tikz()
    entries = re.findall(r"\\addlegendentry\{(.*?)\}", tikz)
    assert entries == ["line", "area"]
    images = re.findall(r"\\addlegendimage\{(.*?)\}\n", tikz)
    assert "area legend" in images[1] and "draw=mpl000000" in images[0]
    assert "legend style={at={(" in tikz and "anchor=north west" in tikz
    assert tikz.count("forget plot") >= 2


def test_log_and_reversed_axes():
    fig, ax = plt.subplots()
    ax.semilogy([1, 2, 3], [1, 10, 100])
    ax.invert_xaxis()
    (options,) = axes_options(figure_to_tikz(fig).generate_tikz())
    assert "ymode=log" in options
    assert "x dir=reverse" in options
    assert "ytick={1,10,100}" in options


def test_text_and_annotation_arrows_are_not_clipped():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.annotate(
        "peak, 50%", xy=(0.5, 0.5), xytext=(0.8, 1.05), arrowprops={"arrowstyle": "->"}
    )
    ax.text(0.1, 0.9, "$x_0$", ha="left", va="top")
    tikz = figure_to_tikz(fig).generate_tikz()
    assert r"{peak, 50\%}" in tikz
    assert "anchor=north west" in tikz and "{$x_0$}" in tikz
    assert re.search(r"\\draw\[.*\] \(axis cs:", tikz), "the arrow is a \\draw"


def test_far_away_geometry_is_clipped_and_text_outside_the_figure_left_out():
    fig, ax = plt.subplots()
    ax.plot([0, 1e6], [0, 1])
    ax.text(1e6, 0.5, "far away")
    ax.set_xlim(0, 1)
    tikz = figure_to_tikz(fig).generate_tikz()
    xs = [float(x) for x in re.findall(r"\(([-0-9.e+]+),[-0-9.e+]+\)", tikz)]
    assert xs and max(xs) < 3
    assert "far away" not in tikz


def test_scatter_colored_by_value_is_one_plot_per_color():
    fig, ax = plt.subplots()
    ax.scatter([0, 1, 2, 3], [0, 1, 2, 3], c=[0, 0, 1, 1], s=20, cmap="viridis")
    tikz = figure_to_tikz(fig).generate_tikz()
    assert tikz.count("only marks, mark=*") == 2


def test_large_scatter_and_meshes_are_images(tmp_path):
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(np.random.default_rng(0).random((10, 20)))
    fig.colorbar(mesh, label="value [a.u.]")
    ax.scatter(*np.random.default_rng(1).random((2, 50)), s=2)
    tikz_figure = figure_to_tikz(fig, max_markers=10)
    tikz = tikz_figure.generate_tikz()
    assert len(tikz_figure.axes) == 2, "the axes and the colorbar"
    assert tikz.count("\\addplot[forget plot] graphics") == 2
    main, colorbar = axes_options(tikz)
    assert "axis on top" in main
    assert "ylabel={value [a.u.]}" in colorbar and "xtick=\\empty" in colorbar
    tikz_figure.savefig(tmp_path / "figure.tikz")
    images = sorted(path.name for path in tmp_path.glob("*.png"))
    assert len(images) == 2
    assert all(name in tikz for name in images)


def test_twin_axes_share_the_position_with_ticks_on_the_right():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    twin = ax.twinx()
    twin.plot([0, 1], [1, 0], color="C1")
    twin.set_ylabel("right")
    first, second = axes_options(figure_to_tikz(fig).generate_tikz())
    position = re.compile(r"at=\{\(([0-9.]+)in,([0-9.]+)in\)\}")
    assert position.search(first).groups() == position.search(second).groups()
    assert "ytick pos=right" in second and "ylabel near ticks" in second
    assert "xtick=\\empty" in second


def test_hidden_spines_and_tick_labels():
    fig, axes = plt.subplots(2, 1, sharex=True)
    for ax in axes:
        ax.plot([0, 1], [0, 1])
        ax.spines[["top", "right"]].set_visible(False)
    upper, lower = axes_options(figure_to_tikz(fig).generate_tikz())
    assert "axis x line*=bottom" in upper and "axis y line*=left" in upper
    assert "xticklabels={}" in upper and "xticklabels={}" not in lower


def test_categories_keep_their_labels():
    fig, ax = plt.subplots()
    ax.bar(["low", "mid_1", "high"], [1, 3, 2])
    (options,) = axes_options(figure_to_tikz(fig).generate_tikz())
    assert r"xticklabels={{low},{mid\_1},{high}}" in options


def test_polar_axes_are_an_image_with_a_warning():
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    ax.plot([0, 1, 2], [1, 2, 1])
    with pytest.warns(TikzConversionWarning, match="polar"):
        tikz = figure_to_tikz(fig).generate_tikz()
    assert "hide axis" in tikz and "graphics" in tikz


def test_the_figure_is_left_as_it_was():
    fig, ax = plt.subplots(layout="constrained")
    (line,) = ax.plot([0, 1], [0, 1])
    mesh = ax.pcolormesh(np.ones((2, 2)))
    engine = fig.get_layout_engine()
    figure_to_tikz(fig)
    assert fig.get_layout_engine() is engine
    assert line.get_visible() and mesh.get_visible() and ax.xaxis.get_visible()
    assert fig.patch.get_visible()


def test_figure_texts_are_placed_in_inches():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1])
    fig.suptitle("All of it")
    tikz = figure_to_tikz(fig).generate_tikz()
    match = re.search(
        r"\\node\[anchor=north.*\] at \(([0-9.]+)in,([0-9.]+)in\) \{All of it\}", tikz
    )
    assert match
    assert float(match.group(1)) == pytest.approx(2.0, abs=0.01)


# -- through a Canvas ------------------------------------------------------
def test_canvas_with_meshes_colorbars_and_a_grid_of_subplots():
    canvas, axes = Canvas.subplots(nrows=2, ncols=2)
    x = np.linspace(0, 1, 20)
    axes[0][0].plot(x, x**2, label="square")
    axes[0][0].set_legend(True)
    axes[0][1].pcolormesh(x, x, np.outer(x, x), cmap="magma")
    axes[0][1].add_colorbar(label="z")
    axes[1][0].scatter(x, x, color="C2")
    axes[1][1].imshow(np.eye(3))
    figure = canvas.render(backend="tikzfigure")
    assert len(figure.axes) == 5
    assert len(figure.files()) == 3  # the mesh, the image and the colorbar strip


def test_an_imported_figure_with_a_colorbar_converts():
    fig, ax = plt.subplots()
    image = ax.imshow(np.arange(6.0).reshape(2, 3))
    fig.colorbar(image)
    canvas = Canvas.from_matplotlib(fig)
    figure = canvas.render(backend="tikzfigure")
    assert len(figure.axes) == 2


def test_rendering_leaves_the_canvas_and_global_style_alone():
    canvas = Canvas()
    canvas.plot([0, 1], [0, 1])
    style = dict(plt.rcParams)
    canvas.render(backend="tikzfigure")
    assert dict(plt.rcParams) == style
    assert not getattr(canvas, "_plotted", False)


# -- compiled ---------------------------------------------------------------
@needs_pdflatex
def test_a_figure_with_everything_compiles(tmp_path):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
    t = np.linspace(0, 10, 3000)
    axes[0].plot(t, np.sin(t) * np.exp(0.2 * t), label=r"$\sin t\, e^{t/5}$")
    axes[0].axvspan(2, 3, alpha=0.2, color="C1", label="window, fit")
    axes[0].errorbar([1, 5], [1, 2], yerr=0.5, fmt="o", capsize=3)
    axes[0].set_yscale("symlog")  # not pgfplots: an image
    axes[0].legend()
    mesh = axes[1].pcolormesh(np.random.default_rng(0).random((30, 30)))
    axes[1].contour(np.random.default_rng(0).random((30, 30)), levels=[0.5], colors="k")
    fig.colorbar(mesh, ax=axes[1], label="$|B|$ [T]")
    fig.suptitle("Everything_1 & more")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TikzConversionWarning)
        figure = figure_to_tikz(fig)
    figure.savefig(tmp_path / "figure.pdf")
    assert (tmp_path / "figure.pdf").stat().st_size > 1000
