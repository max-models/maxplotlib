import matplotlib.pyplot as plt
import numpy as np
import pytest

from maxplotlib import Canvas


@pytest.fixture(autouse=True)
def close_figures():
    with plt.rc_context():
        yield
    plt.close("all")


@pytest.mark.parametrize("kind", ["figure", "axes", "array", "flat", "list"])
def test_input_forms_and_layout(kind):
    fig, axs = plt.subplots(2, 2, figsize=(8, 5))
    for i, ax in enumerate(axs.flat):
        ax.plot([1, 2], [i, i + 1], color="red", label=f"line {i}")
        ax.set_title(f"Panel {i}")
    source = {
        "figure": fig,
        "axes": axs[1, 1],
        "array": axs,
        "flat": axs.ravel(),
        "list": axs.tolist(),
    }[kind]
    canvas = Canvas.from_matplotlib(source, strict=True)
    rendered, imported = canvas.render(backend="matplotlib")
    assert imported.shape == ((1, 1) if kind == "axes" else (2, 2))
    for i, ax in enumerate(imported.flat):
        expected = 3 if kind == "axes" else i
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [expected, expected + 1])
        assert ax.get_title() == f"Panel {expected}"
        assert ax.lines[0].get_color() == "red"
    np.testing.assert_allclose(rendered.get_size_inches(), [8, 5])


def test_explicit_array_order_and_flat_vertical_layout():
    _, axs = plt.subplots(2, 1)
    axs[0].set_title("top")
    axs[1].set_title("bottom")
    canvas = Canvas.from_matplotlib(axs)
    assert (canvas.nrows, canvas.ncols) == (2, 1)
    canvas = Canvas.from_matplotlib(axs[::-1].reshape(1, 2))
    _, imported = canvas.render()
    assert imported.shape == (1, 2)
    assert imported[0, 0].get_title() == "bottom"


def test_data_snapshot_settings_and_plotly_render():
    fig, ax = plt.subplots()
    (line,) = ax.plot([1, 2, 3], [4, 5, 6], marker="o", label="series")
    ax.set(xlabel="Time", ylabel="Value", xscale="log", ylim=(10, 0))
    ax.legend()
    fig.suptitle("Imported")
    canvas = Canvas.from_matplotlib(fig)
    line.set_ydata([100, 200, 300])
    _, imported = canvas.render()
    result = imported[0, 0]
    np.testing.assert_array_equal(result.lines[0].get_ydata(), [4, 5, 6])
    assert result.get_xlabel() == "Time"
    assert result.get_ylabel() == "Value"
    assert result.get_xscale() == "log"
    assert result.get_ylim() == (10, 0)
    assert result.get_legend().get_texts()[0].get_text() == "series"
    plotly_fig = canvas.render(backend="plotly")
    assert len(plotly_fig.data) == 1


def test_scatter_bars_images_and_text():
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter([1, 2], [3, 4], c=[0.1, 0.8], s=[20, 50], marker="s")
    axs[0, 1].bar([1, 2], [3, 4], bottom=[2, 1], width=0.4, label="bars")
    axs[1, 0].barh([1, 2], [3, 4], left=2, height=0.3)
    axs[1, 1].imshow([[1, 2], [3, 4]], origin="lower", extent=(1, 3, 2, 6))
    axs[1, 1].text(2, 3, "hello")
    canvas = Canvas.from_matplotlib(fig, strict=True)
    _, imported = canvas.render()
    scatter = imported[0, 0].collections[0]
    np.testing.assert_allclose(scatter.get_offsets(), [[1, 3], [2, 4]])
    np.testing.assert_allclose(scatter.get_array(), [0.1, 0.8])
    np.testing.assert_allclose(scatter.get_sizes(), [20, 50])
    assert imported[0, 1].patches[0].get_y() == 2
    assert imported[0, 1].patches[0].get_height() == 3
    assert imported[1, 0].patches[0].get_x() == 2
    assert imported[1, 0].patches[0].get_width() == 3
    np.testing.assert_array_equal(
        imported[1, 1].images[0].get_array(), [[1, 2], [3, 4]]
    )
    assert imported[1, 1].texts[0].get_text() == "hello"
    plotly_fig = canvas.render(backend="plotly")
    assert len(plotly_fig.data) >= 6


def test_scatter_solid_and_hollow_colors():
    _, ax = plt.subplots()
    ax.scatter([1], [2], color="red", marker="s")
    ax.scatter([2], [3], facecolors="none", edgecolors="blue")
    canvas = Canvas.from_matplotlib(ax)
    _, imported = canvas.render()
    np.testing.assert_allclose(
        imported[0, 0].collections[0].get_facecolors(), [[1, 0, 0, 1]]
    )
    assert len(imported[0, 0].collections[1].get_facecolors()) == 0


def test_scatter_per_point_colors_render_on_both_backends():
    _, ax = plt.subplots()
    ax.scatter([1, 2], [3, 4], c=["red", "blue"])
    canvas = Canvas.from_matplotlib(ax)
    _, imported = canvas.render()
    np.testing.assert_allclose(
        imported[0, 0].collections[0].get_facecolors(),
        [[1, 0, 0, 1], [0, 0, 1, 1]],
    )
    plotly_fig = canvas.render(backend="plotly")
    assert list(plotly_fig.data[0].marker.color) == [
        "rgba(255,0,0,1.0)",
        "rgba(0,0,255,1.0)",
    ]


@pytest.mark.parametrize("grid", [True, False])
def test_grid_state(grid):
    _, ax = plt.subplots()
    ax.grid(grid)
    canvas = Canvas.from_matplotlib(ax)
    _, imported = canvas.render()
    assert all(line.get_visible() == grid for line in imported[0, 0].get_xgridlines())


def test_unsupported_artists_warn_or_raise_without_changing_source():
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    ax.pcolormesh([[1, 2], [3, 4]])
    children = ax.get_children()
    with pytest.warns(UserWarning, match="collection skipped"):
        canvas = Canvas.from_matplotlib(fig)
    assert len(canvas._subplot_matrix[0][0].line_data) == 1
    with pytest.raises(NotImplementedError, match="collection skipped"):
        Canvas.from_matplotlib(fig, strict=True)
    assert ax.get_children() == children


def test_colorbar_is_reported_and_twin_is_imported():
    fig, ax = plt.subplots()
    image = ax.imshow([[1, 2], [3, 4]])
    fig.colorbar(image, ax=ax)
    ax.twinx().plot([1, 2], [3, 4])
    with pytest.warns(UserWarning) as caught:
        canvas = Canvas.from_matplotlib(fig)
    assert any("Colorbar" in str(w.message) for w in caught)
    assert not any("Twin" in str(w.message) for w in caught)
    assert len(canvas._subplots) == 1
    assert len(canvas._twinx_subplots) == 1


@pytest.mark.parametrize("direction", ["twinx", "twiny"])
@pytest.mark.parametrize("input_kind", ["figure", "reversed", "single"])
def test_twin_axes(direction, input_kind):
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4], color="red")
    twin = getattr(ax, direction)()
    twin.plot([1, 2], [30, 40], color="blue")
    twin.set_ylabel("Secondary")
    source = {"figure": fig, "reversed": [twin, ax], "single": twin}[input_kind]
    canvas = Canvas.from_matplotlib(source, strict=True)
    rendered, axes = canvas.render()
    if input_kind == "single":
        assert len(rendered.axes) == 1
        result = axes[0, 0]
    else:
        assert len(rendered.axes) == 2
        result = rendered.axes[1]
        np.testing.assert_array_equal(axes[0, 0].lines[0].get_ydata(), [3, 4])
    np.testing.assert_array_equal(result.lines[0].get_ydata(), [30, 40])
    assert result.get_ylabel() == "Secondary"
    if direction == "twinx":
        assert len(canvas.render(backend="plotly").data) == (
            1 if input_kind == "single" else 2
        )


@pytest.mark.parametrize("xerr", [None, [0.1, 0.2]])
def test_errorbar_asymmetric_errors_are_not_duplicated(xerr):
    _, ax = plt.subplots()
    source = ax.errorbar(
        [1, 2],
        [3, 4],
        xerr=xerr,
        yerr=[[0.2, 0.3], [0.4, 0.5]],
        fmt="o-",
        capsize=5,
        label="errors",
        ecolor="red",
    )
    canvas = Canvas.from_matplotlib(ax, strict=True)
    assert len(canvas.subplot().line_data) == 1
    _, imported = canvas.render()
    result = imported[0, 0].containers[0]
    for original, copied in zip(source.lines[2], result.lines[2]):
        np.testing.assert_allclose(original.get_segments(), copied.get_segments())
    assert result.lines[1][0].get_markersize() == 10
    np.testing.assert_allclose(
        canvas.render(backend="plotly").data[0].error_y.array, [0.4, 0.5]
    )


def test_bars_with_errors_and_error_only_geometry():
    _, ax = plt.subplots()
    ax.bar([1, 2], [3, 4], yerr=[0.2, 0.4], capsize=3)
    canvas = Canvas.from_matplotlib(ax, strict=True)
    _, imported = canvas.render()
    assert len(imported[0, 0].patches) == 2
    segments = [
        line.get_xydata()
        for line in imported[0, 0].lines
        if len(line.get_xydata()) == 2
    ]
    assert any(np.allclose(segment, [[1, 2.8], [1, 3.2]]) for segment in segments)


def test_fill_between_and_polygons_preserve_geometry():
    from matplotlib.patches import Polygon

    _, ax = plt.subplots()
    region = ax.fill_between([0, 1, 2], [2, 3, 2], [1, 1, 0], color="red", alpha=0.3)
    ax.add_patch(Polygon([[0, 0], [1, 0], [0, 1]], color="blue"))
    canvas = Canvas.from_matplotlib(ax, strict=True)
    _, imported = canvas.render()
    assert len(imported[0, 0].patches) == 2
    np.testing.assert_allclose(
        imported[0, 0].patches[0].get_xy(), region.get_paths()[0].vertices[:-1]
    )
    assert len(canvas.render(backend="plotly").data) == 2


def test_line_collections_and_reference_lines():
    _, ax = plt.subplots()
    ax.hlines([1, 2], [0, 1], [3, 4], colors=["red", "blue"])
    ax.axhline(3, xmin=0.2, xmax=0.8)
    ax.axvline(2, ymin=0.1, ymax=0.7)
    canvas = Canvas.from_matplotlib(ax, strict=True)
    _, imported = canvas.render()
    assert len(imported[0, 0].lines) == 4
    np.testing.assert_allclose(imported[0, 0].lines[0].get_xdata(), [0.2, 0.8])
    np.testing.assert_allclose(imported[0, 0].lines[1].get_ydata(), [0.1, 0.7])
    canvas.render(backend="plotly")


def test_stairs_and_annotation_with_title_styling():
    _, ax = plt.subplots()
    stairs = ax.stairs([1, 3, 2], [0, 1, 2, 3], fill=True)
    annotation = ax.annotate(
        "peak",
        (1.5, 3),
        xytext=(2, 4),
        arrowprops={"arrowstyle": "->"},
        fontweight="bold",
    )
    ax.set_title("Styled", color="red", fontsize=18)
    canvas = Canvas.from_matplotlib(ax, strict=True)
    annotation.set_text("changed")
    _, imported = canvas.render()
    result = imported[0, 0]
    np.testing.assert_allclose(
        result.patches[0].get_path().vertices, stairs.get_path().vertices
    )
    assert result.texts[0].get_text() == "peak"
    assert result.texts[0].arrow_patch is not None
    assert result.title.get_color() == "red"
    assert result.title.get_fontsize() == 18
    canvas.render(backend="plotly")


def test_annotation_without_arrow_and_marker_only_errors_in_plotly():
    _, ax = plt.subplots()
    ax.annotate("label", (1, 2), xytext=(3, 4))
    ax.errorbar([1, 2], [3, 4], yerr=[0.1, 0.2], fmt="o")
    canvas = Canvas.from_matplotlib(ax, strict=True)
    result = canvas.render(backend="plotly")
    annotation = next(a for a in result.layout.annotations if a.text == "label")
    assert not annotation.showarrow
    assert (annotation.x, annotation.y) == (3, 4)
    assert result.data[0].mode == "markers"


def test_unsupported_annotation_coordinates_and_multiple_twins():
    fig, ax = plt.subplots()
    ax.annotate("label", (0.5, 0.5), xycoords="axes fraction")
    with pytest.raises(NotImplementedError, match="outside data coordinates"):
        Canvas.from_matplotlib(ax, strict=True)
    fig2, base = plt.subplots()
    base.twinx()
    base.twinx()
    with pytest.raises(NotImplementedError, match="Multiple twins"):
        Canvas.from_matplotlib(fig2, strict=True)


def test_dashed_line_collection_and_disconnected_fill():
    _, ax = plt.subplots()
    ax.hlines([1, 2], 0, 3, linestyles="dashed")
    region = ax.fill_between(
        [0, 1, 2, 3, 4], [1, 2, 3, 2, 1], where=[True, True, False, True, True]
    )
    canvas = Canvas.from_matplotlib(ax, strict=True)
    _, imported = canvas.render()
    assert len(imported[0, 0].patches) == len(region.get_paths()) == 2
    canvas.render(backend="plotly")


def test_spanning_layout_is_reported():
    fig = plt.figure()
    grid = fig.add_gridspec(2, 2)
    fig.add_subplot(grid[0, :])
    fig.add_subplot(grid[1, 0])
    with pytest.warns(UserWarning, match="layout"):
        canvas = Canvas.from_matplotlib(fig)
    assert (canvas.nrows, canvas.ncols) == (1, 2)
    with pytest.raises(NotImplementedError, match="layout"):
        Canvas.from_matplotlib(fig, strict=True)


@pytest.mark.parametrize("source", [None, 42, [1, 2], [[[1]]]])
def test_invalid_input(source):
    with pytest.raises(TypeError):
        Canvas.from_matplotlib(source)


def test_empty_duplicate_and_mixed_figures():
    with pytest.raises(ValueError, match="At least one"):
        Canvas.from_matplotlib([])
    _, ax = plt.subplots()
    _, other = plt.subplots()
    with pytest.raises(ValueError, match="Duplicate"):
        Canvas.from_matplotlib([ax, ax])
    with pytest.raises(ValueError, match="same Figure"):
        Canvas.from_matplotlib([ax, other])
