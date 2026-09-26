"""Optional support for plotting labelled ``xarray.DataArray`` objects.

xarray is never imported here: an object can only be a ``DataArray`` if the
caller has already imported xarray, so detection looks it up in
``sys.modules``. Install it with ``pip install maxplotlibx[xarray]``.

Units come from the ``units`` attribute, or from the data itself when it is a
pint ``Quantity`` (as with pint-xarray).
"""

import sys

import numpy as np


def is_dataarray(obj) -> bool:
    """Return True if ``obj`` is an ``xarray.DataArray``."""
    xr = sys.modules.get("xarray")
    return xr is not None and isinstance(obj, xr.DataArray)


def _is_quantity(data) -> bool:
    return hasattr(data, "magnitude") and hasattr(data, "units")


def _pint_units(data) -> str:
    try:
        return format(data.units, "~P")
    except (TypeError, ValueError):
        return str(data.units)


def magnitude(variable):
    """Plain numpy values of a DataArray or coordinate, stripping pint units."""
    data = variable.data
    return np.asarray(data.magnitude if _is_quantity(data) else variable.values)


def units(variable) -> str:
    """Units of a DataArray or coordinate: pint units first, then ``attrs``."""
    if _is_quantity(variable.data):
        return _pint_units(variable.data)
    return str(variable.attrs.get("units", ""))


def _label(variable) -> str:
    """Build ``"long_name [units]"`` from CF-style attributes."""
    attrs = variable.attrs
    label = attrs.get("long_name") or attrs.get("standard_name") or variable.name
    label = "" if label is None else str(label)
    unit = units(variable)
    if unit:
        return f"{label} [{unit}]" if label else f"[{unit}]"
    return label


def value_label(da) -> str:
    """Label for the values of ``da``."""
    return _label(da)


def coord_label(da, dim) -> str:
    """Label for the coordinate ``dim`` of ``da`` (the dimension name if none)."""
    if dim in da.coords:
        return _label(da.coords[dim])
    return str(dim)


def coord_values(da, dim):
    """Values of the coordinate ``dim``, or ``0..n-1`` if it has none."""
    if dim in da.coords:
        return magnitude(da.coords[dim])
    return np.arange(da.sizes[dim])


def format_value(value) -> str:
    """Short text for a single coordinate value."""
    value = np.asarray(value)
    if np.issubdtype(value.dtype, np.floating):
        return f"{float(value):.4g}"
    if np.issubdtype(value.dtype, np.datetime64):
        return np.datetime_as_string(value, unit="auto")
    return str(value.item() if value.ndim == 0 else value)


def _coord_text(name, coord, value) -> str:
    unit = units(coord)
    text = f"{name} = {format_value(value)}"
    return f"{text} {unit}" if unit else text


def title(da) -> str:
    """Title from the single-value coordinates left by ``sel``/``isel``.

    For example ``da.sel(t=1.0)`` gives ``"t = 1 s"`` if ``t`` has units ``s``.
    """
    return ", ".join(
        _coord_text(name, coord, magnitude(coord))
        for name, coord in da.coords.items()
        if coord.ndim == 0
    )


def _check_dims(da, *dims):
    for dim in dims:
        if dim is not None and dim not in da.dims:
            raise ValueError(f"{dim!r} is not a dimension of {da.dims}")


def line_data(da, hue=None):
    """Return ``(x, [(y, label), ...], xlabel, ylabel)`` for plotting lines.

    A 1-D array gives one unlabelled line. A 2-D array needs ``hue``, the
    dimension to draw one line per value of.
    """
    _check_dims(da, hue)
    if hue is None and da.ndim != 1:
        raise ValueError(
            f"a 1-D DataArray is needed, got {da.ndim}-D with dims {da.dims}. "
            "Select a slice first, e.g. da.sel(...) or da.isel(...), or pass "
            "hue=<dim> to draw one line per value of a dimension."
        )
    if hue is not None and da.ndim != 2:
        raise ValueError(
            f"hue= needs a 2-D DataArray, got {da.ndim}-D with dims {da.dims}"
        )
    if hue is None:
        (dim,) = da.dims
        lines = [(magnitude(da), None)]
    else:
        (dim,) = [d for d in da.dims if d != hue]
        hue_coord = da.coords[hue] if hue in da.coords else None
        values = magnitude(da.transpose(hue, dim))
        lines = []
        for i in range(da.sizes[hue]):
            if hue_coord is None:
                label = f"{hue} = {i}"
            else:
                label = _coord_text(hue, hue_coord, magnitude(hue_coord)[i])
            lines.append((values[i], label))
    return coord_values(da, dim), lines, coord_label(da, dim), value_label(da)


def mesh_dims(da, x=None, y=None, method="pcolormesh"):
    """Return the ``(x, y)`` dimension names for plotting a 2-D DataArray.

    Like ``xarray.DataArray.plot.pcolormesh``, the first dimension goes on the
    y-axis and the second on the x-axis unless ``x`` or ``y`` names a dimension.
    """
    if da.ndim != 2:
        raise ValueError(
            f"{method}() needs a 2-D DataArray, got {da.ndim}-D with dims "
            f"{da.dims}. Select a slice first, e.g. da.sel(...) or da.isel(...)."
        )
    _check_dims(da, x, y)
    if x is None and y is None:
        y, x = da.dims
    elif x is None:
        (x,) = [d for d in da.dims if d != y]
    elif y is None:
        (y,) = [d for d in da.dims if d != x]
    if x == y:
        raise ValueError("x and y must be different dimensions")
    return x, y


def mesh_data(da, x=None, y=None, method="pcolormesh"):
    """Return ``(x, y, z, xlabel, ylabel, zlabel)`` for a 2-D DataArray.

    See :func:`mesh_dims` for which dimension goes on which axis.
    """
    x, y = mesh_dims(da, x, y, method)
    return (
        coord_values(da, x),
        coord_values(da, y),
        magnitude(da.transpose(y, x)),
        coord_label(da, x),
        coord_label(da, y),
        value_label(da),
    )


def _edges(values, name):
    """Outer edges of evenly spaced cell centres."""
    if values.size == 1:
        return values[0] - 0.5, values[0] + 0.5
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(
            f"imshow() needs a numeric coordinate {name!r}; use pcolormesh() instead"
        )
    step = np.diff(values.astype(float))
    if not np.allclose(step, step[0], rtol=1e-3, atol=0):
        raise ValueError(
            f"imshow() needs an evenly spaced coordinate, but {name!r} is not; "
            "use pcolormesh() instead"
        )
    return values[0] - step[0] / 2, values[-1] + step[0] / 2


def image_extent(x, y, xname="x", yname="y"):
    """``(left, right, bottom, top)`` for ``imshow(..., origin="lower")``."""
    return (*_edges(x, xname), *_edges(y, yname))
