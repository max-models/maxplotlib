"""Behavioral regression fixtures for the importer compatibility matrix."""

import io
import json
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, StrMethodFormatter
from matplotlib.transforms import Affine2D

from maxplotlib import Canvas


@pytest.fixture(autouse=True)
def close_figures():
    with plt.rc_context({"text.usetex": False}):
        yield
    plt.close("all")


def render(source, **kwargs):
    canvas = Canvas.from_matplotlib(source, strict=True, **kwargs)
    fig, axes = canvas.render()
    fig.canvas.draw()
    return canvas, fig, axes


def test_trusted_pickle_boundary_and_report(tmp_path, monkeypatch):
    fig, ax = plt.subplots()
    ax.plot([1, 2], gid="line-id")
    data = pickle.dumps(fig)
    path = tmp_path / "figure.pickle"
    path.write_bytes(data)
    with monkeypatch.context() as boundary:

        def reject_unpickling(*args, **kwargs):
            pytest.fail("Untrusted input reached pickle")

        boundary.setattr(pickle, "load", reject_unpickling)
        boundary.setattr(pickle, "loads", reject_unpickling)
        for source in (data, path, io.BytesIO(data)):
            with pytest.raises(ValueError, match="trusted=True"):
                Canvas.from_matplotlib(source)
    for source in (data, path, io.BytesIO(data)):
        canvas = Canvas.from_matplotlib(source, trusted=True, strict=True)
        result, axes = canvas.render()
        assert axes[0, 0].lines[0].get_gid() == "line-id"
        assert canvas.import_report.diagnostics[0].artist_type == "Line2D"
        json.dumps(canvas.import_report.to_dict())


def test_native_geometry_clipping_rebinding_and_repeated_render():
    fig, ax = plt.subplots()
    circle = Circle(
        (0.5, 0.5), 0.2, transform=Affine2D().translate(0.1, 0) + ax.transAxes
    )
    ax.add_patch(circle)
    vertices = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]
    path = Path(
        vertices, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    )
    ax.add_patch(PathPatch(path, facecolor="red"))
    canvas, first, axes = render(fig)
    circle.set_radius(0.9)
    assert axes[0, 0].patches[0].radius == 0.2
    canvas.set_size_inches(9, 4)
    second, again = canvas.render()
    second.canvas.draw()
    assert again[0, 0].patches[0] is not axes[0, 0].patches[0]
    np.testing.assert_allclose(
        again[0, 0].patches[0].get_transform().transform([0, 0]),
        again[0, 0].transAxes.transform([0.6, 0.5]),
    )
    assert circle.axes is ax and circle.figure is fig
    with pytest.raises(NotImplementedError, match="matplotlib_artist"):
        canvas.render(backend="plotly")


@pytest.mark.parametrize(
    "kind",
    [
        "mesh",
        "contour",
        "tri",
        "quiver",
        "barbs",
        "stream",
        "hexbin",
        "pie",
        "box",
        "violin",
        "event",
        "stem",
        "table",
    ],
)
def test_complex_geometry(kind):
    fig, ax = plt.subplots()
    x, y = np.meshgrid(np.arange(4), np.arange(3))
    z = x + y
    if kind == "mesh":
        ax.pcolormesh(x, y, z, shading="nearest", norm=LogNorm(vmin=1, vmax=6))
    elif kind == "contour":
        c = ax.contour(x, y, z, levels=[1, 2, 3])
        ax.clabel(c)
    elif kind == "tri":
        ax.tripcolor([0, 1, 0, 1], [0, 0, 1, 1], [1, 2, 3, 4])
    elif kind == "quiver":
        q = ax.quiver(x, y, x + 1, y + 1)
        ax.quiverkey(q, 0.8, 0.9, 1, "unit")
    elif kind == "barbs":
        ax.barbs(x, y, x + 1, y + 1)
    elif kind == "stream":
        ax.streamplot(np.arange(4), np.arange(3), x + 1, y + 1)
    elif kind == "hexbin":
        ax.hexbin(x.ravel(), y.ravel(), gridsize=3)
    elif kind == "pie":
        ax.pie([1, 2, 3], labels=["a", "b", "c"], wedgeprops={"width": 0.3})
    elif kind == "box":
        ax.boxplot([[1, 2, 3], [3, 4, 6]])
    elif kind == "violin":
        ax.violinplot([[1, 2, 3], [3, 4, 6]])
    elif kind == "event":
        ax.eventplot([[1, 2], [3, 4]])
    elif kind == "stem":
        ax.stem([1, 2], [3, 4])
    elif kind == "table":
        ax.table(cellText=[["a", "b"], ["c", "d"]])
    canvas, imported, axes = render(fig)
    assert canvas.subplot().line_data
    assert imported is not fig
    again, _ = canvas.render()
    again.canvas.draw()


def test_shared_axes_empty_slots_and_spine_positions():
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    fig.delaxes(axes[0, 1])
    axes[0, 0].spines["right"].set(position=("outward", 32), color="red", linewidth=3)
    canvas, imported, copied = render(fig)
    assert len(imported.axes) == 3
    assert copied[0, 1] is None
    copied[1, 0].set_xlim(2, 8)
    assert copied[0, 0].get_xlim() == (2, 8)
    assert copied[0, 0].spines["right"].get_position() == ("outward", 32)
    assert copied[0, 0].spines["right"].get_linewidth() == 3


@pytest.mark.parametrize(
    "scale,kw",
    [
        ("log", {"base": 2}),
        ("symlog", {"linthresh": 3}),
        ("asinh", {"linear_width": 2}),
        ("logit", {}),
    ],
)
def test_scales_and_axis_configuration(scale, kw):
    fig, ax = plt.subplots()
    ax.set_xscale(scale, **kw)
    ax.set_xlim(0.1, 0.9)
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax.tick_params(axis="y", which="minor", direction="inout", length=8, color="red")
    ax.grid(True, axis="y", which="minor", linestyle="--", color="green", alpha=0.4)
    ax.grid(False, axis="x", which="both")
    canvas, _, copied = render(fig)
    result = copied[0, 0]
    assert result.get_xscale() == scale
    np.testing.assert_allclose(
        result.xaxis.get_transform().transform([0.2, 0.8]),
        ax.xaxis.get_transform().transform([0.2, 0.8]),
    )
    assert not any(line.get_visible() for line in result.get_xgridlines())
    assert result.yaxis.get_minor_ticks()[0].gridline.get_color() == "green"
    assert result.yaxis.get_minor_ticks()[0].tick1line.get_markersize() == 8
    canvas.subplot().set_title("edited")
    canvas.subplot().set_xlim(0.2, 0.8)
    _, edited = canvas.render()
    assert edited[0, 0].get_title() == "edited"
    assert edited[0, 0].get_xlim() == (0.2, 0.8)


def test_dates_categories_and_formatters():
    import datetime

    fig, axes = plt.subplots(1, 2)
    axes[0].plot([datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 2)], [1, 2])
    axes[1].plot(["apple", "pear"], [1, 2])
    fig.canvas.draw()
    _, _, copied = render(fig)
    for source, target in zip(axes, copied.flat):
        assert [t.get_text() for t in source.get_xticklabels()] == [
            t.get_text() for t in target.get_xticklabels()
        ]
        assert target.xaxis.get_major_formatter().axis is target.xaxis


def test_hidden_artists_metadata_dashes_and_order():
    fig, ax = plt.subplots()
    ax.scatter([1], [2], zorder=2)
    (line,) = ax.plot(
        [1, 2],
        [2, 3],
        linestyle=(2, (3, 2, 1, 2)),
        dash_capstyle="round",
        gapcolor="red",
        markevery=2,
        visible=False,
        gid="hidden",
        url="https://example.test",
        picker=5,
    )
    canvas, _, copied = render(fig)
    result = copied[0, 0].lines[0]
    assert not result.get_visible()
    assert result.get_gid() == "hidden"
    assert result.get_url() == "https://example.test"
    assert result.get_picker() == 5
    assert result._unscaled_dash_pattern == line._unscaled_dash_pattern
    assert result.get_gapcolor() == "red"
    assert copied[0, 0].get_children()[0] is copied[0, 0].collections[0]
    canvas.subplot().line_data[1]["kwargs"]["visible"] = True
    _, updated = canvas.render()
    assert updated[0, 0].lines[0].get_visible()


def test_legend_proxies_multiple_legends_and_figure_decorations():
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots()
    (line,) = ax.plot([1, 2], label="original")
    first = ax.legend(
        [Line2D([], [], color="red")],
        ["proxy"],
        loc="upper left",
        title="Proxy",
        framealpha=0.3,
    )
    ax.add_artist(first)
    ax.legend([line], ["renamed"], loc="lower right", ncols=2)
    fig.legend([line], ["global"], loc="upper center")
    fig.text(0.1, 0.1, "footer")
    fig.patch.set_facecolor("beige")
    fig.suptitle("Super", x=0.3, fontsize=21, color="red")
    _, copied, axes = render(fig)
    from matplotlib.legend import Legend

    legends = [
        child for child in axes[0, 0].get_children() if isinstance(child, Legend)
    ]
    assert [[t.get_text() for t in legend.get_texts()] for legend in legends] == [
        ["proxy"],
        ["renamed"],
    ]
    assert copied.get_facecolor() == fig.get_facecolor()
    assert copied._suptitle.get_position()[0] == 0.3
    assert any(isinstance(a, Legend) for a in copied.artists)


def test_colorbar_ownership_multiple_shared_and_horizontal():
    fig, axes = plt.subplots(1, 2)
    image = axes[0].imshow([[1, 2], [3, 4]], norm=LogNorm(1, 4))
    points = axes[1].scatter([1, 2], [2, 3], c=[10, 20])
    fig.colorbar(image, ax=axes, orientation="horizontal", label="image", extend="both")
    fig.colorbar(points, ax=axes[1], label="points", ticks=[10, 15, 20])
    canvas, copied, imported = render(fig)
    bars = [ax._colorbar for ax in copied.axes if hasattr(ax, "_colorbar")]
    assert bars[0].mappable is imported[0, 0].images[0]
    assert bars[1].mappable is imported[0, 1].collections[0]
    assert bars[0].orientation == "horizontal" and bars[0].extend == "both"
    np.testing.assert_allclose(bars[1].get_ticks(), [10, 15, 20])
    image.set_clim(2, 10)
    assert bars[0].mappable.norm.vmin == 1


def test_secondary_inset_and_annotation_coordinates():
    fig, ax = plt.subplots()
    secondary = ax.secondary_xaxis("top", functions=(lambda x: x * 2, lambda x: x / 2))
    secondary.set_xlabel("double")
    inset = ax.inset_axes([0.2, 0.2, 0.4, 0.4])
    inset.plot([1, 2])
    ax.annotate(
        "offset",
        (0.5, 0.5),
        xycoords="axes fraction",
        xytext=(12, 6),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    _, copied, axes = render(fig)
    result = axes[0, 0]
    assert len(result.child_axes) == 2
    result.set_xlim(1, 3)
    copied.canvas.draw()
    assert result.child_axes[0].get_xlim() == (2, 6)
    assert result.child_axes[0].get_xlabel() == "double"
    assert result.texts[0].xycoords == "axes fraction"


def test_inset_zoom_indicator_rebinds_to_reconstructed_axes():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2, 3], [0, 1, 4, 9])
    inset = ax.inset_axes([0.5, 0.5, 0.4, 0.4])
    inset.plot([0, 1, 2, 3], [0, 1, 4, 9])
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    ax.indicate_inset_zoom(inset, edgecolor="black", linewidth=2, alpha=0.75)

    _, copied, axes = render(fig)
    result = axes[0, 0]
    indicators = [
        artist for artist in result.artists if type(artist).__name__ == "InsetIndicator"
    ]
    assert len(indicators) == 1
    indicator = indicators[0]
    # The indicator must reference the *reconstructed* inset axes, not the
    # source figure's, so it keeps tracking the copy's live limits.
    assert indicator._inset_ax is result.child_axes[0]
    assert indicator.rectangle.get_edgecolor() == (0.0, 0.0, 0.0, 0.75)
    assert indicator.rectangle.get_linewidth() == 2.0


@pytest.mark.parametrize("projection", ["polar", "3d"])
def test_native_projections(projection):
    fig = plt.figure()
    ax = fig.add_subplot(projection=projection)
    if projection == "3d":
        ax.plot([0, 1], [1, 2], [3, 4])
        ax.plot_surface(*np.meshgrid([0, 1], [0, 1]), np.array([[1, 2], [3, 4]]))
        ax.view_init(elev=40, azim=25)
    else:
        ax.plot([0, 1, 2], [1, 2, 3])
    canvas, _, axes = render(fig)
    assert axes[0, 0].name == projection
    if projection == "3d":
        assert axes[0, 0].elev == 40
    with pytest.raises(NotImplementedError, match="projections"):
        canvas.render(backend="plotly")


def test_raster_loss_report_and_plotly_rgba_extent():
    fig, ax = plt.subplots()
    ax.imshow([[1, 2], [3, 4]], norm=LogNorm(1, 4), extent=(2, 6, 3, 9), origin="lower")
    canvas = Canvas.from_matplotlib(fig, strict=True)
    image = canvas.render(backend="plotly").data[0]
    assert image.type == "image"
    assert (image.x0, image.dx, image.y0, image.dy) == (3, 2, 4.5, 3)
    raster = Canvas.from_matplotlib(fig, fallback="raster")
    assert raster.import_report.diagnostics[0].fallback == "raster"
    assert "lost" in raster.import_report.diagnostics[0].message
    assert raster.render(backend="plotly").data[0].type == "image"


def test_error_limits_subsampling_and_independent_components():
    fig, ax = plt.subplots()
    error = ax.errorbar(
        [1, 2, 3, 4], [2, 3, 4, 5], yerr=0.3, errorevery=2, lolims=True, capsize=4
    )
    error.lines[1][0].set_color("red")
    _, _, copied = render(fig)
    assert len(copied[0, 0].lines) >= 3
    from matplotlib.colors import to_rgba

    assert any(
        to_rgba(line.get_color()) == to_rgba("red") for line in copied[0, 0].lines
    )


def test_raster_image_comparison_and_exports(tmp_path):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot([0, 1, 2], [1, 3, 2], color="red", linestyle=(0, (4, 2)), marker="o")
    ax.set_title("snapshot")
    ax.tick_params(direction="in", colors="blue")
    ax.grid(False)
    FigureCanvasAgg(fig).draw()
    original = np.asarray(fig.canvas.buffer_rgba()).copy()
    canvas = Canvas.from_matplotlib(fig, strict=True)
    result, _ = canvas.render(savefig=True)
    FigureCanvasAgg(result).draw()
    actual = np.asarray(result.canvas.buffer_rgba()).copy()
    assert original.shape == actual.shape
    assert np.mean(np.abs(original.astype(float) - actual)) < 2.0
    canvas.savefig(tmp_path / "figure.png")
    canvas.savefig(tmp_path / "figure.svg")
    canvas.savefig(tmp_path / "figure.html", backend="plotly")
    assert all(
        (tmp_path / name).stat().st_size > 100
        for name in ("figure.png", "figure.svg", "figure.html")
    )


@pytest.mark.parametrize("layout", ["constrained", "tight", None])
def test_layout_ratios_and_reflow(layout):
    fig, axes = plt.subplots(2, 2, layout=layout, gridspec_kw={"width_ratios": [1, 3]})
    axes[0, 0].set_ylabel("Vertical label")
    fig.canvas.draw()
    canvas, copied, imported = render(fig)
    for original, result in zip(axes.flat, imported.flat):
        np.testing.assert_allclose(
            result.get_position().bounds, original.get_position().bounds, atol=0.02
        )
    canvas.set_size_inches(9, 5)
    resized, result = canvas.render()
    resized.canvas.draw()
    assert result[0, 1].get_position().width / result[
        0, 0
    ].get_position().width == pytest.approx(3)


def test_subfigures_mosaic_and_nested_layout():
    fig = plt.figure()
    left, right = fig.subfigures(1, 2)
    a = left.subplots()
    grid = right.add_gridspec(2, 2)
    b = right.add_subplot(grid[0, :])
    c = right.add_subplot(grid[1, 0].subgridspec(1, 1)[0])
    for ax in (a, b, c):
        ax.plot([1, 2])
    _, _, imported = render(fig)
    for source, result in zip((a, b, c), imported.flat):
        expected = (
            source.get_position()
            .transformed(source.figure.transSubfigure)
            .transformed(fig.transFigure.inverted())
        )
        np.testing.assert_allclose(result.get_position().bounds, expected.bounds)


def test_composite_text_boxes_effects_and_clip_boxes():
    import matplotlib.patheffects as pe
    from matplotlib.offsetbox import AnchoredText, AnnotationBbox, TextArea
    from matplotlib.transforms import Bbox, TransformedBbox

    fig, ax = plt.subplots()
    ax.add_artist(AnnotationBbox(TextArea("boxed"), (0.5, 0.5)))
    ax.add_artist(AnchoredText("anchor", loc="upper left"))
    ax.text(0.2, 0.4, "$x^2$\nline two", linespacing=1.7, bbox={"facecolor": "yellow"})
    (line,) = ax.plot(
        [0, 1], [0, 1], path_effects=[pe.Stroke(linewidth=4), pe.Normal()]
    )
    line.set_clip_box(
        TransformedBbox(Bbox.from_bounds(0.1, 0.1, 0.6, 0.6), ax.transAxes)
    )
    canvas, copied, axes = render(fig)
    result = axes[0, 0]
    assert result.texts[0].get_bbox_patch() is not None
    assert result.texts[0]._linespacing == 1.7
    assert len(result.lines[0].get_path_effects()) == 2
    assert len(result.artists) == 2
    assert result.lines[0].get_clip_box().bounds == pytest.approx(
        line.get_clip_box().bounds
    )


def test_unit_converter_and_masked_empty_data():
    import pint

    units = pint.UnitRegistry()
    units.setup_matplotlib()
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(np.arange(3) * units.second, np.arange(3) * units.meter)
    axes[1].scatter([], [])
    axes[1].plot(np.ma.array([0, 1, 2], mask=[0, 1, 0]), [1, 2, 3])
    axes[1].imshow(np.ma.array([[1, 2], [3, 4]], mask=[[0, 1], [0, 0]]))
    _, _, imported = render(fig)
    assert str(imported[0, 0].xaxis.get_units()) == "second"
    assert len(imported[0, 1].collections[0].get_offsets()) == 0
    assert np.isnan(imported[0, 1].lines[0].get_xdata()[1])
    assert imported[0, 1].images[0].get_array().mask[0, 1]


@pytest.mark.parametrize(
    "backend,extension", [("plotext", "txt"), ("tikzfigure", "tikz")]
)
def test_portable_import_export_remaining_backends(tmp_path, backend, extension):
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [1, 3, 2], color="red", marker="o")
    canvas = Canvas.from_matplotlib(fig, strict=True)
    path = tmp_path / ("imported." + extension)
    canvas.savefig(str(path), backend=backend)
    assert path.is_file() and path.stat().st_size > 0


def test_retained_container_group_metadata():
    fig, ax = plt.subplots()
    bars = ax.bar([0, 1], [2, 3], label="group")
    errors = ax.errorbar([0, 1], [2, 3], yerr=0.5, lolims=True, label="limits")
    canvas, _, _ = render(fig)
    subplot = canvas.subplot()
    assert subplot.import_groups[id(bars)]["label"] == "group"
    np.testing.assert_allclose(subplot.import_groups[id(bars)]["datavalues"], [2, 3])
    assert (
        len(
            [
                entry
                for entry in subplot.line_data
                if id(bars) in entry["source_container_ids"]
            ]
        )
        == 2
    )
    assert (
        len(
            [
                entry
                for entry in subplot.line_data
                if id(errors) in entry["source_container_ids"]
            ]
        )
        >= 2
    )


def test_source_and_render_lifetimes_are_independent():
    import gc
    import weakref

    fig, ax = plt.subplots()
    ax.add_patch(Circle((0.5, 0.5), 0.2))
    canvas = Canvas.from_matplotlib(fig, strict=True)
    reference = weakref.ref(fig)
    plt.close(fig)
    del ax, fig
    gc.collect()
    assert reference() is None
    first, axes = canvas.render()
    axes[0, 0].patches[0].set_radius(0.8)
    second, axes = canvas.render()
    assert axes[0, 0].patches[0].radius == 0.2


def test_scatter_plotly_normalization_sizes_and_hollow_markers():
    fig, ax = plt.subplots()
    points = ax.scatter([0, 1], [1, 2], c=[1, 10], s=[9, 36], norm=LogNorm(1, 100))
    ax.scatter([2], [3], facecolors="none", edgecolors="red")
    canvas = Canvas.from_matplotlib(fig, strict=True)
    result = canvas.render(backend="plotly")
    assert result.data[0].marker.size == pytest.approx([4, 8])
    assert result.data[1].marker.symbol == "circle-open"
    expected = points.cmap(points.norm([1, 10]))
    for color, rgba in zip(result.data[0].marker.color, expected):
        r, g, b = np.round(rgba[:3] * 255).astype(int)
        assert color == f"rgba({r},{g},{b},{rgba[3]})"


def test_arbitrary_axes_rectangles_are_preserved():
    fig = plt.figure()
    first = fig.add_axes([0.1, 0.2, 0.6, 0.5])
    second = fig.add_axes([0.4, 0.3, 0.5, 0.6])
    first.plot([0, 1])
    second.plot([1, 0])
    _, _, copied = render(fig)
    for source, result in zip((first, second), copied.flat):
        np.testing.assert_allclose(
            source.get_position().bounds, result.get_position().bounds
        )


def test_hidden_fixed_ticks_and_secondary_data_location():
    fig, ax = plt.subplots()
    ax.set_xticks([0, 1, 2], ["a", "b", "c"])
    ax.get_xticklabels()[1].set_visible(False)
    _, _, copied = render(fig)
    assert [tick.get_text() for tick in copied[0, 0].get_xticklabels()] == ["a", "c"]


def test_standalone_and_projection_colorbars():
    from matplotlib.cm import ScalarMappable

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    points = ax.scatter([1, 2], [2, 3], [3, 4], c=[1, 2])
    fig.colorbar(points, ax=ax)
    fig.colorbar(ScalarMappable(norm=LogNorm(1, 10)), ax=ax, orientation="horizontal")
    _, copied, axes = render(fig)
    bars = [axis._colorbar for axis in copied.axes if hasattr(axis, "_colorbar")]
    assert bars[0].mappable is axes[0, 0].collections[0]
    assert isinstance(bars[1].norm, LogNorm)


def test_custom_title_placement_and_clearing_imported_settings():
    fig, ax = plt.subplots()
    ax.set_title("custom", x=0.2, y=0.85, pad=13)
    ax.set_title("left", loc="left", x=0.05, y=0.9)
    ax.set_xscale("symlog", linthresh=2)
    ax.grid(True, axis="y")
    canvas, _, copied = render(fig)
    assert copied[0, 0].title.get_position() == (0.2, 0.85)
    assert copied[0, 0]._left_title.get_position() == (0.05, 0.9)
    canvas.subplot().set_title("")
    canvas.subplot().set_xscale("linear")
    canvas.subplot().set_grid(True)
    _, edited = canvas.render()
    assert edited[0, 0].get_title() == ""
    assert edited[0, 0].get_xscale() == "linear"
    assert all(line.get_visible() for line in edited[0, 0].get_xgridlines())
