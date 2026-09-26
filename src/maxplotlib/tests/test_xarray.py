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
