import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from maxplotlib import Canvas  # noqa: E402
from maxplotlib.utils import xarray_support  # noqa: E402


def _line():
    t = np.linspace(0, 1, 5)
    return xr.DataArray(
        t**2,
        dims="t",
        coords={"t": ("t", t, {"long_name": "Time", "units": "s"})},
        name="energy",
        attrs={"units": "J"},
    )


def _mesh():
    x = np.linspace(0, 1, 4)
    y = np.linspace(0, 2, 3)
    return xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("y", "x"),
        coords={"x": ("x", x, {"units": "m"}), "y": y},
        name="phi",
        attrs={"long_name": "Potential", "units": "V"},
    )


def test_is_dataarray():
    assert xarray_support.is_dataarray(_line())
    assert not xarray_support.is_dataarray(np.zeros(3))
    assert not xarray_support.is_dataarray(_line().to_dataset())


def test_labels_from_attrs():
    da = _mesh()
    assert xarray_support.value_label(da) == "Potential [V]"
    assert xarray_support.coord_label(da, "x") == "x [m]"
    assert xarray_support.coord_label(da, "y") == "y"
    assert xarray_support.value_label(xr.DataArray([1.0])) == ""
    assert (
        xarray_support.value_label(xr.DataArray([1.0], attrs={"units": "K"})) == "[K]"
    )


def test_dimension_without_coordinate_uses_index():
    da = xr.DataArray([3.0, 4.0, 5.0], dims="i")
    x, y, xlabel, ylabel = xarray_support.line_data(da)
    np.testing.assert_array_equal(x, [0, 1, 2])
    assert xlabel == "i"


def test_canvas_plot_dataarray_matplotlib():
    da = _line()
    canvas = Canvas()
    assert canvas.plot(da, color="red") is canvas

    fig, ax = canvas.get_matplotlib_figaxs()
    ax = np.ravel(ax)[0]
    line = ax.get_lines()[0]
    np.testing.assert_allclose(line.get_xdata(), da.t.values)
    np.testing.assert_allclose(line.get_ydata(), da.values)
    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "energy [J]"
    plt.close(fig)


def test_subplot_plot_dataarray():
    canvas, ax = Canvas.subplots()
    ax.plot(_line())
    fig = canvas.render(backend="plotly")
    assert fig.layout.xaxis.title.text == "Time [s]"
    np.testing.assert_allclose(fig.data[0].y, _line().values)


def test_explicit_labels_win_regardless_of_order():
    canvas, ax = Canvas.subplots()
    ax.set_xlabel("before")
    ax.plot(_line())
    ax.set_ylabel("after")
    fig, axes = canvas.get_matplotlib_figaxs()
    axes = np.ravel(axes)[0]
    assert axes.get_xlabel() == "before"
    assert axes.get_ylabel() == "after"
    plt.close(fig)


def test_plot_rejects_wrong_ndim():
    canvas = Canvas()
    with pytest.raises(ValueError, match="1-D DataArray.*sel"):
        canvas.plot(_mesh())
    with pytest.raises(TypeError, match="both x and y"):
        canvas.plot(np.arange(3))


def test_plot_x_y_arrays_still_work_with_dataarrays():
    da = _line()
    canvas = Canvas()
    canvas.plot(da.t, da)
    fig, ax = canvas.get_matplotlib_figaxs()
    ax = np.ravel(ax)[0]
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), da.values)
    assert ax.get_xlabel() == ""
    plt.close(fig)


def test_pcolormesh_dataarray_matplotlib():
    da = _mesh()
    canvas = Canvas()
    canvas.pcolormesh(da, cmap="magma")
    fig, axes = canvas.get_matplotlib_figaxs()
    ax = np.ravel(axes)[0]
    assert ax.get_xlabel() == "x [m]"
    assert ax.get_ylabel() == "y"
    mesh = ax.collections[0]
    assert mesh.get_cmap().name == "magma"
    np.testing.assert_allclose(mesh.get_array().reshape(3, 4), da.values)
    colorbars = [a for a in fig.axes if a is not ax]
    assert len(colorbars) == 1
    assert colorbars[0].get_ylabel() == "Potential [V]"
    plt.close(fig)


def test_pcolormesh_dataarray_transpose_and_no_colorbar():
    da = _mesh()
    canvas = Canvas()
    canvas.pcolormesh(da, xdim="y", add_colorbar=False)
    fig, axes = canvas.get_matplotlib_figaxs()
    ax = np.ravel(axes)[0]
    assert ax.get_xlabel() == "y"
    assert ax.get_ylabel() == "x [m]"
    np.testing.assert_allclose(ax.collections[0].get_array().reshape(4, 3), da.values.T)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_pcolormesh_dataarray_plotly():
    da = _mesh()
    canvas = Canvas()
    canvas.pcolormesh(da)
    fig = canvas.render(backend="plotly")
    heatmap = fig.data[0]
    np.testing.assert_allclose(heatmap.z, da.values)
    np.testing.assert_allclose(heatmap.x, da.x.values)
    assert heatmap.colorbar.title.text == "Potential [V]"
    assert fig.layout.xaxis.title.text == "x [m]"


def test_pcolormesh_dataarray_errors():
    canvas = Canvas()
    with pytest.raises(ValueError, match="2-D DataArray"):
        canvas.pcolormesh(_line())
    with pytest.raises(ValueError, match="not a dimension"):
        canvas.pcolormesh(_mesh(), xdim="t")
    with pytest.raises(ValueError, match="different"):
        canvas.pcolormesh(_mesh(), xdim="x", ydim="x")
    with pytest.raises(TypeError, match="no y or z"):
        canvas.pcolormesh(_mesh(), np.arange(3))
    with pytest.raises(TypeError, match="requires x, y and z"):
        canvas.pcolormesh(np.arange(3), np.arange(3))


def test_imshow_colorbar_uses_given_label():
    canvas = Canvas()
    canvas.imshow(np.arange(4.0).reshape(2, 2))
    canvas.colorbar(label="density")
    fig, axes = canvas.get_matplotlib_figaxs()
    assert fig.axes[-1].get_ylabel() == "density"
    plt.close(fig)


# ---------------------------------------------------------------------------
# Titles from single-value coordinates
# ---------------------------------------------------------------------------


def _cube():
    t = np.array([0.0, 0.5, 1.0])
    return xr.DataArray(
        np.arange(36.0).reshape(3, 3, 4),
        dims=("t", "y", "x"),
        coords={
            "t": ("t", t, {"units": "s"}),
            "x": np.linspace(0, 1, 4),
            "y": np.linspace(0, 2, 3),
        },
        name="phi",
        attrs={"long_name": "Potential", "units": "V"},
    )


def test_title_from_selected_coordinates():
    da = _cube().sel(t=0.5).isel(y=1)
    assert xarray_support.title(da) == "t = 0.5 s, y = 1"
    canvas = Canvas()
    canvas.plot(da)
    fig, axes = canvas.get_matplotlib_figaxs()
    assert np.ravel(axes)[0].get_title() == "t = 0.5 s, y = 1"
    plt.close(fig)


def test_title_formats_strings_and_datetimes():
    da = xr.DataArray(
        np.zeros((2, 2)),
        dims=("species", "time"),
        coords={
            "species": ["ions", "electrons"],
            "time": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
        },
    )
    assert xarray_support.title(da.isel(species=0, time=1)) == (
        "species = ions, time = 2026-01-02"
    )


def test_explicit_title_wins():
    canvas, ax = Canvas.subplots()
    ax.set_title("mine")
    ax.pcolormesh(_cube().isel(t=0))
    fig = canvas.render(backend="plotly")
    assert "mine" in [a.text for a in fig.layout.annotations]
    assert not any("t = " in (a.text or "") for a in fig.layout.annotations)


def test_no_title_without_scalar_coords():
    canvas, ax = Canvas.subplots()
    ax.plot(_line())
    assert ax._title is None


# ---------------------------------------------------------------------------
# imshow, contour, contourf, scatter
# ---------------------------------------------------------------------------


def test_imshow_dataarray_extent_and_labels():
    da = _mesh()
    canvas = Canvas()
    canvas.imshow(da)
    fig, axes = canvas.get_matplotlib_figaxs()
    ax = np.ravel(axes)[0]
    image = ax.get_images()[0]
    # x: 0..1 in 4 steps of 1/3, y: 0..2 in 3 steps of 1.
    np.testing.assert_allclose(image.get_extent(), [-1 / 6, 7 / 6, -0.5, 2.5])
    assert image.origin == "lower"
    np.testing.assert_allclose(image.get_array(), da.values)
    assert ax.get_xlabel() == "x [m]"
    assert fig.axes[-1].get_ylabel() == "Potential [V]"
    plt.close(fig)


def test_imshow_dataarray_plotly_shows_colorbar():
    canvas = Canvas()
    canvas.imshow(_mesh())
    fig = canvas.render(backend="plotly")
    heatmap = fig.data[0]
    assert heatmap.showscale is True
    assert heatmap.colorbar.title.text == "Potential [V]"
    np.testing.assert_allclose(heatmap.x0, 0.0, atol=1e-12)
    np.testing.assert_allclose(heatmap.dx, 1 / 3)


def test_imshow_dataarray_rejects_uneven_coordinates():
    da = _mesh().assign_coords(x=[0.0, 0.1, 0.5, 2.0])
    with pytest.raises(ValueError, match="'x' is not.*pcolormesh"):
        Canvas().imshow(da)


def test_contour_and_contourf_dataarray():
    da = _mesh()
    canvas, (left, right) = Canvas.subplots(ncols=2)
    left.contour(da, levels=3)
    right.contourf(da)
    fig, axes = canvas.get_matplotlib_figaxs()
    axes = np.ravel(axes)
    assert axes[0].get_xlabel() == "x [m]"
    # contour adds no colorbar by default, contourf does.
    colorbars = [a for a in fig.axes if a not in axes]
    assert [a.get_ylabel() for a in colorbars] == ["Potential [V]"]
    plt.close(fig)


def test_contour_colorbar_flag_does_not_reach_matplotlib():
    canvas = Canvas()
    canvas.contourf(_mesh(), colorbar=False)
    fig, _ = canvas.get_matplotlib_figaxs()
    plt.close(fig)


def test_scatter_dataarray():
    da = _line()
    canvas = Canvas()
    canvas.scatter(da, color="k")
    fig, axes = canvas.get_matplotlib_figaxs()
    ax = np.ravel(axes)[0]
    np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 1], da.values)
    assert ax.get_ylabel() == "energy [J]"
    plt.close(fig)


# ---------------------------------------------------------------------------
# hue
# ---------------------------------------------------------------------------


def _species():
    t = np.linspace(0, 1, 5)
    return xr.DataArray(
        np.stack([t, 2 * t]),
        dims=("species", "t"),
        coords={"species": ["ions", "electrons"], "t": ("t", t, {"units": "s"})},
        name="density",
    )


def test_plot_hue_draws_one_labelled_line_per_value():
    da = _species()
    canvas = Canvas()
    canvas.plot(da, hue="species")
    fig, axes = canvas.get_matplotlib_figaxs()
    ax = np.ravel(axes)[0]
    lines = ax.get_lines()
    assert [line.get_label() for line in lines] == [
        "species = ions",
        "species = electrons",
    ]
    np.testing.assert_allclose(lines[1].get_ydata(), da.sel(species="electrons"))
    assert ax.get_legend() is not None
    assert ax.get_xlabel() == "t [s]"
    plt.close(fig)


def test_hue_on_first_dimension_and_without_coordinate():
    da = _species().T.drop_vars("species")
    x, lines, _, _ = xarray_support.line_data(da, hue="species")
    assert [label for _, label in lines] == ["species = 0", "species = 1"]
    np.testing.assert_allclose(lines[1][0], 2 * x)


def test_hue_errors_and_add_legend():
    canvas = Canvas()
    with pytest.raises(ValueError, match="hue=<dim>"):
        canvas.plot(_species())
    with pytest.raises(ValueError, match="hue= needs a 2-D"):
        canvas.plot(_line(), hue="t")
    with pytest.raises(TypeError, match="label="):
        canvas.plot(_species(), hue="species", label="x")
    canvas, ax = Canvas.subplots()
    ax.scatter(_species(), hue="species", add_legend=False)
    assert not ax._legend
    fig = canvas.render(backend="plotly")
    assert [trace.name for trace in fig.data] == [
        "species = ions",
        "species = electrons",
    ]


# ---------------------------------------------------------------------------
# facets
# ---------------------------------------------------------------------------


def test_facet_col_wrap_shares_scale_and_colorbar():
    da = _cube()
    canvas, axes = Canvas.facet(da, col="t", col_wrap=2)
    assert len(axes) == 2 and len(axes[0]) == 2
    assert axes[1][1] is None
    fig, mpl_axes = canvas.get_matplotlib_figaxs()
    assert not mpl_axes[1][1].get_visible()
    meshes = [mpl_axes[r][c].collections[0] for r, c in [(0, 0), (0, 1), (1, 0)]]
    assert {mesh.norm.vmin for mesh in meshes} == {0.0}
    assert {mesh.norm.vmax for mesh in meshes} == {35.0}
    assert [ax.get_title() for ax in fig.axes[:4] if ax.get_visible()] == [
        "t = 0 s",
        "t = 0.5 s",
        "t = 1 s",
    ]
    # One figure-wide colorbar: 4 grid axes + 1 colorbar axis.
    assert len(fig.axes) == 5
    assert fig.axes[-1].get_ylabel() == "Potential [V]"
    # Outer labels only: (0, 1) has nothing below it, so keeps its x label.
    assert mpl_axes[0][0].get_xlabel() == ""
    assert mpl_axes[0][1].get_xlabel() == "x"
    assert mpl_axes[1][0].get_ylabel() == "y"
    assert mpl_axes[0][1].get_ylabel() == ""
    plt.close(fig)


def test_facet_plotly_single_colorbar():
    canvas, _ = Canvas.facet(_cube(), col="t")
    fig = canvas.render(backend="plotly")
    assert [trace.showscale for trace in fig.data] == [True, False, False]
    assert {trace.zmin for trace in fig.data} == {0.0}
    assert {trace.zmax for trace in fig.data} == {35.0}
    assert fig.data[0].colorbar.title.text == "Potential [V]"


def test_facet_row_and_col_lines():
    da = _cube()
    canvas, axes = Canvas.facet(da.isel(y=0), row="t", kind="plot")
    assert len(axes) == 3 and len(axes[0]) == 1
    assert axes[2][0]._title == "t = 1 s, y = 0"
    canvas, axes = Canvas.facet(da, row="t", col="y", kind="plot", color="k")
    assert len(axes) == 3 and len(axes[0]) == 3
    fig, _ = canvas.get_matplotlib_figaxs()
    assert len(fig.axes) == 9  # no colorbar for lines
    plt.close(fig)


def test_facet_lines_share_y_range_and_hide_inner_ticks():
    da = _cube().isel(y=0)  # values 0..27 over t
    canvas, axes = Canvas.facet(da, col="t", kind="plot")
    fig, mpl_axes = canvas.get_matplotlib_figaxs()
    limits = {tuple(ax.get_ylim()) for ax in np.ravel(mpl_axes)}
    assert len(limits) == 1
    np.testing.assert_allclose(limits.pop(), (-1.35, 28.35))  # data range + 5 %
    assert mpl_axes[0][0].yaxis.get_tick_params()["labelleft"]
    assert not mpl_axes[0][1].yaxis.get_tick_params()["labelleft"]
    plt.close(fig)

    canvas, axes = Canvas.facet(da, col="t", kind="plot", sharey=False)
    fig, mpl_axes = canvas.get_matplotlib_figaxs()
    assert len({tuple(ax.get_ylim()) for ax in np.ravel(mpl_axes)}) == 3
    assert mpl_axes[0][1].get_ylabel() == "Potential [V]"
    plt.close(fig)


def test_facet_contourf_shares_levels_and_can_skip_colorbar():
    canvas, axes = Canvas.facet(_cube(), col="t", kind="contourf", add_colorbar=False)
    fig, mpl_axes = canvas.get_matplotlib_figaxs()
    levels = [ax.collections[0].levels for ax in np.ravel(mpl_axes)]
    np.testing.assert_allclose(levels[0], levels[2])
    assert len(fig.axes) == 3
    plt.close(fig)


def test_facet_errors():
    da = _cube()
    with pytest.raises(ValueError, match="col= and/or row="):
        Canvas.facet(da)
    with pytest.raises(ValueError, match="not a dimension"):
        Canvas.facet(da, col="z")
    with pytest.raises(ValueError, match="kind must be"):
        Canvas.facet(da, col="t", kind="bar")
    with pytest.raises(ValueError, match="col_wrap"):
        Canvas.facet(da, row="t", col_wrap=2)
    with pytest.raises(ValueError, match="2-D DataArray"):
        Canvas.facet(da.isel(x=0), col="t")  # leaves 1-D (y) panels


# ---------------------------------------------------------------------------
# pint units
# ---------------------------------------------------------------------------


def test_pint_units_label_and_values():
    pint = pytest.importorskip("pint")
    ureg = pint.UnitRegistry()
    t = np.linspace(0, 1, 4)
    da = xr.DataArray(
        ureg.Quantity(t * 3.0, "m/s"),
        dims="t",
        coords={"t": ("t", t, {"units": "s"})},
        name="speed",
    )
    assert xarray_support.value_label(da) == "speed [m/s]"
    canvas = Canvas()
    canvas.plot(da)
    fig, axes = canvas.get_matplotlib_figaxs()
    ax = np.ravel(axes)[0]
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), t * 3.0)
    assert ax.get_ylabel() == "speed [m/s]"
    plt.close(fig)


def test_pint_units_override_attrs_and_mesh():
    pint = pytest.importorskip("pint")
    ureg = pint.UnitRegistry()
    da = _mesh().copy(data=ureg.Quantity(_mesh().values, "kV"))
    # attrs still say "V"; the quantity is authoritative.
    assert xarray_support.value_label(da) == "Potential [kV]"
    x, y, z, *_ = xarray_support.mesh_data(da)
    assert type(z) is np.ndarray
    np.testing.assert_allclose(z, _mesh().values)


# ---------------------------------------------------------------------------
# da.maxplot accessor
# ---------------------------------------------------------------------------


def test_accessor_methods_return_canvases():
    import maxplotlib.xarray  # noqa: F401

    canvas = _line().maxplot.line(color="red")
    assert isinstance(canvas, Canvas)
    fig, axes = canvas.get_matplotlib_figaxs()
    ax = np.ravel(axes)[0]
    assert ax.get_lines()[0].get_color() == "red"
    assert ax.get_xlabel() == "Time [s]"
    plt.close(fig)

    for method in ("pcolormesh", "imshow", "contour", "contourf"):
        canvas = getattr(_mesh().maxplot, method)()
        fig = canvas.render(backend="plotly")
        np.testing.assert_allclose(fig.data[0].z, _mesh().values)

    canvas = _species().maxplot.scatter(hue="species")
    assert len(canvas.render(backend="plotly").data) == 2


def test_accessor_canvas_kwargs_and_facets():
    import maxplotlib.xarray  # noqa: F401

    canvas = _mesh().maxplot.pcolormesh(canvas_kwargs={"fontsize": 14})
    assert canvas.fontsize == 14

    canvas = _cube().maxplot.pcolormesh(col="t", col_wrap=2, cmap="magma")
    assert (canvas.nrows, canvas.ncols) == (2, 2)
    fig = canvas.render(backend="plotly")
    assert [trace.showscale for trace in fig.data] == [True, False, False]

    canvas = _cube().isel(y=0).maxplot.line(col="t", sharey=False)
    assert canvas.ncols == 3


def test_accessor_call_picks_kind():
    import maxplotlib.xarray  # noqa: F401

    fig = _line().maxplot().render(backend="plotly")
    assert fig.data[0].type == "scatter"
    fig = _mesh().maxplot().render(backend="plotly")
    assert fig.data[0].type == "heatmap"
    fig = _species().maxplot(hue="species").render(backend="plotly")
    assert [trace.type for trace in fig.data] == ["scatter", "scatter"]
    canvas = _cube().maxplot(col="t")
    assert canvas.ncols == 3
    with pytest.raises(ValueError, match="1-D or 2-D"):
        _cube().maxplot()


def test_maxplotlib_does_not_import_xarray():
    import subprocess
    import sys

    code = "import sys, maxplotlib; assert 'xarray' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
