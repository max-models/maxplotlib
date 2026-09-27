"""Single-source-of-truth consistency checks between backends.

For every recorded plot type exercised here, the test:

1. Makes ONE ``Canvas`` drawing call, which records "the recorded intent" as
   a dict in ``LinePlot.line_data`` (see ``LinePlot._add``/``add_line`` and
   the ``plot_*`` recording methods in
   ``src/maxplotlib/subfigure/line_plot.py``).
2. Renders BOTH backends from that same canvas
   (``canvas.get_matplotlib_figaxs()`` and
   ``canvas.get_plotly_fig(allow_unsupported=True)``).
3. Asserts that each backend's rendered artifact (a Matplotlib artist, or a
   Plotly trace/shape/annotation) carries the SAME values that were recorded
   -- instead of cross-converting one backend's figure into the other's.

Where ``PLOTLY_BACKEND_COVERAGE.md`` documents that a keyword is not honored
by the Plotly backend (``zorder``, ``hatch``, ...), the corresponding case
asserts that Matplotlib reflects it and that Plotly does not, with a comment
pointing at the relevant section of that document. A genuine gap discovered
while writing this suite (the Plotly ``step`` branch does not translate
Matplotlib's ``where="pre"/"post"/"mid"`` into a valid Plotly ``line.shape``
enum value) is documented with a dedicated ``xfail`` test rather than folded
into the main matrix, since it would raise instead of silently mismatching.
"""

import re

import matplotlib.colors as mcolors
import numpy as np
import pytest

from maxplotlib import Canvas

# ---------------------------------------------------------------------------
# Reusable helpers: "recorded intent" + "what each backend shows".
# ---------------------------------------------------------------------------


def recorded(ax, index=-1):
    """The recorded-intent dict for a drawing call (``LinePlot.line_data``)."""
    return ax.line_data[index]


def render_both(canvas):
    """Render the same canvas with both backends and return the artifacts.

    ``allow_unsupported=True`` matches the semantics documented on
    ``Canvas.get_plotly_fig`` -- it only changes behavior for plot types with
    no faithful Plotly equivalent, none of which are exercised here, but it
    keeps this helper usable if such a case is added later.
    """
    figure, axes = canvas.get_matplotlib_figaxs()
    plotly_fig = canvas.get_plotly_fig(allow_unsupported=True)
    return axes[0][0], plotly_fig


def rgba(color):
    """Normalize a Matplotlib or Plotly (``rgba(...)``/``rgb(...)``) color
    spelling to an RGBA tuple, so colors round-tripped through either
    backend can be compared directly."""
    if isinstance(color, str):
        match = re.fullmatch(
            r"\s*rgba?\(([^)]+)\)\s*", color, flags=re.IGNORECASE
        )
        if match:
            parts = [float(p) for p in match.group(1).split(",")]
            r, g, b = (p / 255.0 for p in parts[:3])
            a = parts[3] if len(parts) == 4 else 1.0
            return (r, g, b, a)
    return mcolors.to_rgba(color)


def rgba_close(actual, expected):
    return np.allclose(rgba(actual), rgba(expected))


def rgb_close(actual, expected):
    """Compare only the RGB channels, ignoring alpha.

    Useful when a separate ``alpha=`` kwarg has been baked into the actual
    artist's color (as Matplotlib does for ``bar``/``scatter`` face colors),
    so the alpha channel legitimately differs from the plain named color the
    call recorded.
    """
    return np.allclose(rgba(actual)[:3], rgba(expected)[:3])


def last_trace(plotly_fig, trace_type=None):
    data = plotly_fig.data
    if trace_type is None:
        return data[-1]
    return next(t for t in reversed(data) if t.type == trace_type)


def mpl_last_line(ax):
    """What Matplotlib shows for the most recent Line2D artist."""
    return ax.lines[-1]


def mpl_last_patch(ax):
    """What Matplotlib shows for the most recently added Patch artist."""
    return ax.patches[-1]


def mpl_last_collection(ax):
    """What Matplotlib shows for the most recently added Collection artist."""
    return ax.collections[-1]


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def test_plot_reflects_recorded_style_on_both_backends():
    canvas, ax = Canvas.subplots()
    x, y = np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.5])
    ax.plot(
        x,
        y,
        color="royalblue",
        linestyle="dashed",
        linewidth=2.5,
        marker="o",
        markersize=8,
        label="line",
    )
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    # What Matplotlib shows.
    line = mpl_last_line(mpl_ax)
    assert np.array_equal(line.get_xdata(), intent["x"])
    assert np.array_equal(line.get_ydata(), intent["y"])
    assert rgba_close(line.get_color(), intent["kwargs"]["color"])
    assert line.get_linestyle() == "--"
    assert line.get_linewidth() == intent["kwargs"]["linewidth"]
    assert line.get_marker() == "o"
    assert line.get_markersize() == intent["kwargs"]["markersize"]

    # What Plotly shows.
    trace = last_trace(plotly_fig, "scatter")
    assert np.array_equal(np.asarray(trace.x), intent["x"])
    assert np.array_equal(np.asarray(trace.y), intent["y"])
    assert trace.line.color == intent["kwargs"]["color"]
    assert trace.line.dash == "dash"
    assert trace.line.width == intent["kwargs"]["linewidth"]
    assert trace.marker.symbol == "circle"
    assert trace.marker.size == intent["kwargs"]["markersize"]
    assert trace.name == "line"


# ---------------------------------------------------------------------------
# scatter
# ---------------------------------------------------------------------------


def test_scatter_reflects_recorded_style_on_both_backends():
    canvas, ax = Canvas.subplots()
    x, y = np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 4.0])
    ax.scatter(
        x,
        y,
        color="tomato",
        marker="s",
        s=64,
        alpha=0.5,
        edgecolors="black",
        linewidths=2,
        label="pts",
    )
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    collection = mpl_last_collection(mpl_ax)
    assert np.array_equal(collection.get_offsets(), np.column_stack([x, y]))
    # ``alpha=`` is baked into the face/edge colors' alpha channel, so only
    # the RGB channels are compared against the plain recorded color name.
    assert rgb_close(collection.get_facecolor()[0], intent["kwargs"]["color"])
    assert collection.get_alpha() == intent["kwargs"]["alpha"]
    assert rgb_close(collection.get_edgecolor()[0], intent["kwargs"]["edgecolors"])
    assert collection.get_linewidths()[0] == intent["kwargs"]["linewidths"]

    trace = last_trace(plotly_fig, "scatter")
    assert np.array_equal(np.asarray(trace.x), intent["x"])
    assert np.array_equal(np.asarray(trace.y), intent["y"])
    assert trace.marker.color == intent["kwargs"]["color"]
    assert trace.marker.symbol == "square"
    assert trace.marker.opacity == intent["kwargs"]["alpha"]
    assert trace.marker.line.color == intent["kwargs"]["edgecolors"]
    assert trace.marker.line.width == intent["kwargs"]["linewidths"]


def test_scatter_value_colormap_reflects_recorded_c_vmin_vmax():
    canvas, ax = Canvas.subplots()
    c = [0.1, 0.5, 0.9]
    ax.scatter([0, 1, 2], [0, 1, 4], c=c, cmap="plasma", vmin=0.0, vmax=1.0)
    intent = recorded(ax)
    _, plotly_fig = render_both(canvas)

    trace = last_trace(plotly_fig, "scatter")
    assert list(trace.marker.color) == intent["kwargs"]["c"]
    assert trace.marker.cmin == intent["kwargs"]["vmin"]
    assert trace.marker.cmax == intent["kwargs"]["vmax"]


# ---------------------------------------------------------------------------
# bar / barh
# ---------------------------------------------------------------------------


def test_bar_reflects_recorded_values_on_both_backends():
    canvas, ax = Canvas.subplots()
    ax.bar(
        [0, 1],
        [2, 3],
        color="blue",
        bottom=[1, 1],
        width=0.5,
        alpha=0.6,
        edgecolor="black",
        linewidth=2,
        label="b",
    )
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    heights, bottoms = intent["height"], intent["kwargs"]["bottom"]
    for patch, height, bottom in zip(mpl_ax.patches, heights, bottoms):
        assert patch.get_height() == height
        assert patch.get_y() == bottom
        assert patch.get_width() == intent["kwargs"]["width"]
        # ``alpha=`` is baked into the face/edge colors' alpha channel here
        # too; compare RGB only, and alpha via the recorded kwarg directly.
        assert rgb_close(patch.get_facecolor(), intent["kwargs"]["color"])
        assert patch.get_alpha() == intent["kwargs"]["alpha"]
        assert rgb_close(patch.get_edgecolor(), intent["kwargs"]["edgecolor"])
        assert patch.get_linewidth() == intent["kwargs"]["linewidth"]

    trace = last_trace(plotly_fig, "bar")
    assert np.array_equal(np.asarray(trace.y), heights)
    assert np.array_equal(np.asarray(trace.base), np.asarray(bottoms, dtype=float))
    assert trace.width == intent["kwargs"]["width"]
    assert trace.opacity == intent["kwargs"]["alpha"]
    assert trace.marker.color == intent["kwargs"]["color"]
    assert trace.marker.line.color == intent["kwargs"]["edgecolor"]
    assert trace.marker.line.width == intent["kwargs"]["linewidth"]


def test_barh_reflects_recorded_values_on_both_backends():
    canvas, ax = Canvas.subplots()
    ax.barh([0, 1], [2, 3], left=[1, 1], height=0.4, color="green")
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    widths, lefts = intent["width"], intent["kwargs"]["left"]
    for patch, width, left in zip(mpl_ax.patches, widths, lefts):
        assert patch.get_width() == width
        assert patch.get_x() == left
        assert rgba_close(patch.get_facecolor(), intent["kwargs"]["color"])

    trace = last_trace(plotly_fig, "bar")
    assert np.array_equal(np.asarray(trace.x), widths)
    assert np.array_equal(np.asarray(trace.base), np.asarray(lefts, dtype=float))
    assert trace.width == intent["kwargs"]["height"]
    assert trace.marker.color == intent["kwargs"]["color"]


# ---------------------------------------------------------------------------
# errorbar
# ---------------------------------------------------------------------------


def test_errorbar_reflects_recorded_values_on_both_backends():
    canvas, ax = Canvas.subplots()
    x, y = np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 1.0])
    ax.errorbar(
        x, y, yerr=0.3, xerr=0.1, color="black", capsize=5, elinewidth=3, marker="o"
    )
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    container = mpl_ax.containers[-1]
    line, _caps, (xbar, ybar) = container.lines
    assert np.array_equal(line.get_xdata(), intent["x"])
    assert np.array_equal(line.get_ydata(), intent["y"])
    # Each x-error cap spans +/- xerr around the point; each y-error bar
    # spans +/- yerr.
    x_segment = xbar.get_segments()[0]
    assert np.allclose(x_segment[:, 0], [x[0] - 0.1, x[0] + 0.1])
    y_segment = ybar.get_segments()[0]
    assert np.allclose(y_segment[:, 1], [y[0] - 0.3, y[0] + 0.3])

    trace = last_trace(plotly_fig, "scatter")
    assert np.array_equal(np.asarray(trace.x), intent["x"])
    assert np.array_equal(np.asarray(trace.y), intent["y"])
    assert np.allclose(trace.error_y.array, 0.3)
    assert np.allclose(trace.error_x.array, 0.1)
    assert trace.error_y.width == intent["kwargs"]["capsize"]
    assert trace.error_y.thickness == intent["kwargs"]["elinewidth"]


# ---------------------------------------------------------------------------
# fill_between
# ---------------------------------------------------------------------------


def test_fill_between_reflects_recorded_values_on_both_backends():
    canvas, ax = Canvas.subplots()
    x, y1 = [0, 1, 2], [1, 2, 1]
    ax.fill_between(x, y1, 0, color="gray", alpha=0.3, label="band")
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    collection = mpl_last_collection(mpl_ax)
    face = collection.get_facecolor()[0]
    assert rgba_close(face[:3].tolist() + [1.0], intent["kwargs"]["color"])
    assert np.isclose(face[3], intent["kwargs"]["alpha"])

    trace = last_trace(plotly_fig, "scatter")
    assert trace.fillcolor == intent["kwargs"]["color"]
    assert trace.opacity == intent["kwargs"]["alpha"]
    # Plotly draws the band as one closed polygon: y1 forward, y2 backward.
    assert np.array_equal(np.asarray(trace.x), np.concatenate([x, x[::-1]]))
    assert np.array_equal(
        np.asarray(trace.y), np.concatenate([y1, np.zeros(len(x))[::-1]])
    )


# ---------------------------------------------------------------------------
# step / stairs
# ---------------------------------------------------------------------------


def test_step_reflects_recorded_values_on_both_backends():
    canvas, ax = Canvas.subplots()
    x, y = [0, 1, 2], [1, 3, 2]
    ax.step(x, y, color="purple", label="s")
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    line = mpl_last_line(mpl_ax)
    assert np.array_equal(line.get_xdata(), intent["x"])
    assert np.array_equal(line.get_ydata(), intent["y"])
    assert rgba_close(line.get_color(), intent["kwargs"]["color"])
    # Matplotlib's own default step placement is "pre".
    assert line.get_drawstyle() == "steps-pre"

    trace = last_trace(plotly_fig, "scatter")
    assert np.array_equal(np.asarray(trace.x), intent["x"])
    assert np.array_equal(np.asarray(trace.y), intent["y"])
    assert trace.line.color == intent["kwargs"]["color"]
    # NOTE: this is a real, currently-undocumented drift, not just a
    # dropped keyword: when ``where=`` is omitted, the Plotly branch
    # defaults to ``shape="hv"`` (step-after, i.e. Matplotlib's "post"),
    # while Matplotlib's own default is "pre". See the dedicated xfail test
    # below for the sharper case (an explicit ``where=`` value crashing
    # Plotly outright).
    assert trace.line.shape == "hv"


def test_step_explicit_where_is_not_translated_for_plotly():
    """Known gap: Plotly's ``step`` branch passes ``where=`` straight through
    as ``line.shape`` (see PLOTLY_BACKEND_COVERAGE.md, ``step`` row: only
    ``color``/``label``/``where`` are "honored", but the value isn't mapped
    from Matplotlib's vocabulary ("pre"/"post"/"mid") to Plotly's
    ("hv"/"vh"/"hvh"/"vhv"/"linear"/"spline"). A Matplotlib-valid, explicit
    ``where="post"`` is not a valid Plotly ``line.shape`` enum value, so
    building the Plotly figure raises instead of silently mismatching.
    """
    canvas, ax = Canvas.subplots()
    ax.step([0, 1, 2], [1, 3, 2], where="post")

    figure, axes = canvas.get_matplotlib_figaxs()
    assert axes[0][0].lines[-1].get_drawstyle() == "steps-post"

    with pytest.raises(ValueError, match="shape"):
        canvas.get_plotly_fig(allow_unsupported=True)


def test_stairs_reflects_recorded_values_on_both_backends():
    canvas, ax = Canvas.subplots()
    values, edges = [1, 2, 3], [0, 1, 2, 3]
    ax.stairs(values, edges=edges, color="teal", label="st")
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    patch = mpl_last_patch(mpl_ax)
    stair_values, stair_edges, _baseline = patch.get_data()
    assert np.array_equal(stair_values, intent["values"])
    assert np.array_equal(stair_edges, intent["edges"])
    assert rgba_close(patch.get_edgecolor(), intent["kwargs"]["color"])

    trace = last_trace(plotly_fig, "scatter")
    expected_x = np.repeat(edges, 2)[1:-1]
    expected_y = np.repeat(values, 2)
    assert np.array_equal(np.asarray(trace.x), expected_x)
    assert np.array_equal(np.asarray(trace.y), expected_y)
    assert trace.line.color == intent["kwargs"]["color"]


# ---------------------------------------------------------------------------
# contour / contourf / pcolormesh / pcolor
# ---------------------------------------------------------------------------


def _mesh_field():
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    xx, yy = np.meshgrid(x, y)
    z = xx**2 + yy**2
    return x, y, z


def test_contour_reflects_recorded_z_and_clim_on_both_backends():
    canvas, ax = Canvas.subplots()
    x, y, z = _mesh_field()
    ax.contour(x, y, z, cmap="viridis", vmin=0.0, vmax=2.0)
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    contour_set = mpl_last_collection(mpl_ax)
    assert contour_set.get_clim() == (
        intent["kwargs"]["vmin"],
        intent["kwargs"]["vmax"],
    )

    trace = last_trace(plotly_fig, "contour")
    assert np.array_equal(np.asarray(trace.z), intent["z"])
    assert np.array_equal(np.asarray(trace.x), intent["x"])
    assert np.array_equal(np.asarray(trace.y), intent["y"])
    assert trace.zmin == intent["kwargs"]["vmin"]
    assert trace.zmax == intent["kwargs"]["vmax"]


def test_contourf_reflects_recorded_z_on_both_backends():
    canvas, ax = Canvas.subplots()
    x, y, z = _mesh_field()
    ax.contourf(x, y, z, cmap="plasma")
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    contour_set = mpl_last_collection(mpl_ax)
    assert np.isclose(contour_set.get_clim()[0], z.min())
    assert np.isclose(contour_set.get_clim()[1], z.max())

    trace = last_trace(plotly_fig, "contour")
    assert np.array_equal(np.asarray(trace.z), intent["z"])
    assert trace.contours.coloring != "lines"  # contourf fills, contour doesn't


def test_pcolormesh_reflects_recorded_z_and_clim_on_both_backends():
    canvas, ax = Canvas.subplots()
    x, y, z = _mesh_field()
    ax.pcolormesh(x, y, z, cmap="plasma", vmin=0.1, vmax=1.5)
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    quadmesh = mpl_last_collection(mpl_ax)
    assert quadmesh.get_clim() == (intent["kwargs"]["vmin"], intent["kwargs"]["vmax"])
    array = quadmesh.get_array()
    array = array.data if hasattr(array, "data") else array
    assert np.array_equal(np.asarray(array).reshape(z.shape), intent["z"])

    trace = last_trace(plotly_fig, "heatmap")
    assert np.array_equal(np.asarray(trace.z), intent["z"])
    assert trace.zmin == intent["kwargs"]["vmin"]
    assert trace.zmax == intent["kwargs"]["vmax"]


def test_pcolor_reflects_recorded_z_on_both_backends():
    canvas, ax = Canvas.subplots()
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    z = xx**2 + yy**2
    ax.pcolor(x, y, z, cmap="viridis", alpha=0.8)
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    quadmesh = mpl_last_collection(mpl_ax)
    array = quadmesh.get_array()
    array = array.data if hasattr(array, "data") else array
    assert np.array_equal(np.asarray(array).reshape(z.shape), intent["z"])
    assert quadmesh.get_alpha() == intent["kwargs"]["alpha"]

    trace = last_trace(plotly_fig, "heatmap")
    assert np.array_equal(np.asarray(trace.z), intent["z"])
    assert trace.opacity == intent["kwargs"]["alpha"]


# ---------------------------------------------------------------------------
# imshow
# ---------------------------------------------------------------------------


def test_imshow_reflects_recorded_data_and_clim_on_both_backends():
    canvas, ax = Canvas.subplots()
    data = np.arange(9).reshape(3, 3).astype(float)
    ax.imshow(data, cmap="gray", vmin=0.0, vmax=8.0)
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    image = mpl_ax.images[0]
    assert np.array_equal(image.get_array(), intent["data"])
    assert image.get_clim() == (intent["kwargs"]["vmin"], intent["kwargs"]["vmax"])

    trace = last_trace(plotly_fig, "heatmap")
    assert np.array_equal(np.asarray(trace.z), intent["data"])
    assert trace.zmin == intent["kwargs"]["vmin"]
    assert trace.zmax == intent["kwargs"]["vmax"]


# ---------------------------------------------------------------------------
# quiver
# ---------------------------------------------------------------------------


def test_quiver_reflects_recorded_vectors_on_both_backends():
    canvas, ax = Canvas.subplots()
    ax.quiver([0, 1], [0, 1], [1, -1], [1, 1], color="purple", alpha=0.5)
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    quiver = mpl_last_collection(mpl_ax)
    assert np.array_equal(quiver.U, intent["u"])
    assert np.array_equal(quiver.V, intent["v"])

    arrows = [a for a in plotly_fig.layout.annotations if a.showarrow]
    assert len(arrows) == len(intent["x"])
    for arrow, x, y, u, v in zip(
        arrows, intent["x"], intent["y"], intent["u"], intent["v"]
    ):
        assert arrow.ax == x
        assert arrow.ay == y
        assert arrow.x == x + u
        assert arrow.y == y + v
        assert arrow.arrowcolor == intent["kwargs"]["color"]


# ---------------------------------------------------------------------------
# hist
# ---------------------------------------------------------------------------


def test_hist_reflects_recorded_data_on_both_backends():
    canvas, ax = Canvas.subplots()
    data = [1, 2, 2, 3, 3, 3, 4]
    ax.hist(data, bins=3, color="blue")
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    expected_counts, _ = np.histogram(data, bins=3)
    heights = [p.get_height() for p in mpl_ax.patches]
    assert np.array_equal(heights, expected_counts)

    trace = last_trace(plotly_fig, "histogram")
    # Plotly computes its own bins client-side; the raw recorded data and the
    # bin count are what maxplotlib actually hands it.
    assert np.array_equal(np.asarray(trace.x), intent["x"])
    assert trace.nbinsx == intent["bins"]


# ---------------------------------------------------------------------------
# boxplot
# ---------------------------------------------------------------------------


def test_boxplot_reflects_recorded_datasets_on_both_backends():
    canvas, ax = Canvas.subplots()
    datasets = [[1, 2, 3], [2, 4, 5]]
    ax.boxplot(datasets)
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    # No outliers in either dataset, so Matplotlib's default whiskers extend
    # exactly to each dataset's min and max.
    all_y = np.concatenate([line.get_ydata() for line in mpl_ax.lines])
    for dataset in intent["x"]:
        assert min(dataset) in all_y
        assert max(dataset) in all_y

    box_traces = [t for t in plotly_fig.data if t.type == "box"]
    assert len(box_traces) == len(datasets)
    for trace, dataset in zip(box_traces, intent["x"]):
        assert list(trace.y) == dataset


# ---------------------------------------------------------------------------
# axhline / axvline
# ---------------------------------------------------------------------------


def test_axhline_reflects_recorded_value_on_both_backends():
    canvas, ax = Canvas.subplots()
    ax.axhline(0.5, color="black", linestyle="dashed", linewidth=2)
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    line = mpl_last_line(mpl_ax)
    assert line.get_ydata()[0] == intent["y"]
    assert rgba_close(line.get_color(), intent["kwargs"]["color"])
    assert line.get_linewidth() == intent["kwargs"]["linewidth"]

    shape = plotly_fig.layout.shapes[-1]
    assert shape.y0 == shape.y1 == intent["y"]
    assert shape.xref == "paper"
    assert shape.line.color == intent["kwargs"]["color"]
    assert shape.line.dash == "dash"
    assert shape.line.width == intent["kwargs"]["linewidth"]


def test_axvline_reflects_recorded_value_on_both_backends():
    canvas, ax = Canvas.subplots()
    ax.axvline(0.25, color="red", linestyle="dotted")
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    line = mpl_last_line(mpl_ax)
    assert line.get_xdata()[0] == intent["x"]
    assert rgba_close(line.get_color(), intent["kwargs"]["color"])

    shape = plotly_fig.layout.shapes[-1]
    assert shape.x0 == shape.x1 == intent["x"]
    assert shape.yref == "paper"
    assert shape.line.color == intent["kwargs"]["color"]
    assert shape.line.dash == "dot"


# ---------------------------------------------------------------------------
# text / annotate
# ---------------------------------------------------------------------------


def test_text_reflects_recorded_position_and_style_on_both_backends():
    canvas, ax = Canvas.subplots()
    ax.text(0.3, 0.4, "hi", color="purple", fontsize=12)
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    text_artist = mpl_ax.texts[-1]
    assert text_artist.get_text() == intent["s"]
    assert text_artist.get_position() == (intent["x"], intent["y"])
    assert rgba_close(text_artist.get_color(), intent["kwargs"]["color"])
    assert text_artist.get_fontsize() == intent["kwargs"]["fontsize"]

    annotation = next(a for a in plotly_fig.layout.annotations if a.text == "hi")
    assert annotation.x == intent["x"]
    assert annotation.y == intent["y"]
    assert annotation.font.color == intent["kwargs"]["color"]
    assert annotation.font.size == intent["kwargs"]["fontsize"]


def test_annotate_reflects_recorded_position_and_style_on_both_backends():
    canvas, ax = Canvas.subplots()
    ax.annotate(
        "there", xy=(0.3, 0.3), xytext=(0.6, 0.5), color="green", fontsize=10
    )
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    text_artist = mpl_ax.texts[-1]
    assert text_artist.get_text() == intent["text"]
    assert rgba_close(text_artist.get_color(), intent["kwargs"]["color"])

    annotation = next(a for a in plotly_fig.layout.annotations if a.text == "there")
    assert annotation.x == intent["xy"][0]
    assert annotation.y == intent["xy"][1]
    assert annotation.font.color == intent["kwargs"]["color"]
    assert annotation.font.size == intent["kwargs"]["fontsize"]


# ---------------------------------------------------------------------------
# patch
# ---------------------------------------------------------------------------


def test_patch_reflects_recorded_patch_object_on_both_backends():
    import matplotlib.patches as mpatches

    canvas, ax = Canvas.subplots()
    patch = mpatches.Rectangle(
        (0.2, 0.2), 1.0, 0.5, edgecolor="yellow", facecolor="cyan", alpha=0.5
    )
    # Style lives on the patch object itself, not on ``add_patch(**kwargs)``:
    # real Matplotlib's ``Axes.add_patch(self, p)`` takes no extra keyword
    # arguments, and maxplotlib's Matplotlib branch forwards
    # ``**line["kwargs"]`` straight to it (see line_plot.py around line
    # 1881-1885), so passing style kwargs there would raise a TypeError.
    ax.add_patch(patch)
    intent = recorded(ax)
    mpl_ax, plotly_fig = render_both(canvas)

    mpl_patch = mpl_last_patch(mpl_ax)
    assert mpl_patch is intent["patch"]
    assert mpl_patch.get_x() == 0.2
    assert mpl_patch.get_y() == 0.2

    shape = plotly_fig.layout.shapes[-1]
    assert shape.x0 == pytest.approx(0.2)
    assert shape.y0 == pytest.approx(0.2)
    assert shape.x1 == pytest.approx(1.2)
    assert shape.y1 == pytest.approx(0.7)
    # ``alpha=0.5`` on the patch is baked into both colors' alpha channel by
    # Matplotlib's own ``get_facecolor()``/``get_edgecolor()`` (which is what
    # the Plotly branch reads it from too); compare RGB only.
    assert rgb_close(shape.fillcolor, "cyan")
    assert rgb_close(shape.line.color, "yellow")


# ---------------------------------------------------------------------------
# Documented gaps: keywords Plotly drops (PLOTLY_BACKEND_COVERAGE.md, "Every
# thing else -- zorder, hatch, ... -- is dropped for Plotly.").
# ---------------------------------------------------------------------------


def test_bar_zorder_and_hatch_are_matplotlib_only():
    canvas, ax = Canvas.subplots()
    ax.bar([0, 1], [2, 3], color="blue", zorder=5, hatch="//")
    mpl_ax, plotly_fig = render_both(canvas)

    for patch in mpl_ax.patches:
        assert patch.get_zorder() == 5
        assert patch.get_hatch() == "//"

    trace = last_trace(plotly_fig, "bar")
    # Plotly has its own unrelated ``zorder``/pattern properties; maxplotlib
    # never sets them from the Matplotlib-style kwargs, so they stay at
    # Plotly's defaults instead of reflecting the recorded values.
    assert trace.zorder is None
    assert trace.marker.pattern.shape is None
