"""Optional support for plotting labelled ``xarray.DataArray`` objects.

xarray is never imported here: an object can only be a ``DataArray`` if the
caller has already imported xarray, so detection looks it up in
``sys.modules``. Install it with ``pip install maxplotlibx[xarray]``.
"""

import sys

import numpy as np


def is_dataarray(obj) -> bool:
    """Return True if ``obj`` is an ``xarray.DataArray``."""
    xr = sys.modules.get("xarray")
    return xr is not None and isinstance(obj, xr.DataArray)


def _label(name, attrs) -> str:
    """Build ``"long_name [units]"`` from CF-style attributes."""
    label = attrs.get("long_name") or attrs.get("standard_name") or name or ""
    units = attrs.get("units", "")
    if units:
        return f"{label} [{units}]" if label else f"[{units}]"
    return str(label)


def value_label(da) -> str:
    """Label for the values of ``da``."""
    return _label(da.name, da.attrs)


def coord_label(da, dim) -> str:
    """Label for the coordinate ``dim`` of ``da`` (the dimension name if none)."""
    if dim in da.coords:
        return _label(dim, da.coords[dim].attrs)
    return str(dim)


def coord_values(da, dim):
    """Values of the coordinate ``dim``, or ``0..n-1`` if it has none."""
    if dim in da.coords:
        return np.asarray(da.coords[dim].values)
    return np.arange(da.sizes[dim])


def _require_ndim(da, ndim, method):
    if da.ndim != ndim:
        raise ValueError(
            f"{method}() needs a {ndim}-D DataArray, got {da.ndim}-D with dims "
            f"{da.dims}. Select a slice first, e.g. da.sel(...) or da.isel(...)."
        )


def line_data(da):
    """Return ``(x, y, xlabel, ylabel)`` for a 1-D DataArray."""
    _require_ndim(da, 1, "plot")
    (dim,) = da.dims
    return (
        coord_values(da, dim),
        np.asarray(da.values),
        coord_label(da, dim),
        value_label(da),
    )


def mesh_data(da, x=None, y=None):
    """Return ``(x, y, z, xlabel, ylabel, zlabel)`` for a 2-D DataArray.

    Like ``xarray.DataArray.plot.pcolormesh``, the first dimension goes on the
    y-axis and the second on the x-axis unless ``x`` or ``y`` names a dimension.
    """
    _require_ndim(da, 2, "pcolormesh")
    for name in (x, y):
        if name is not None and name not in da.dims:
            raise ValueError(f"{name!r} is not a dimension of {da.dims}")
    if x is None and y is None:
        y, x = da.dims
    elif x is None:
        (x,) = [d for d in da.dims if d != y]
    elif y is None:
        (y,) = [d for d in da.dims if d != x]
    if x == y:
        raise ValueError("x and y must be different dimensions")
    return (
        coord_values(da, x),
        coord_values(da, y),
        np.asarray(da.transpose(y, x).values),
        coord_label(da, x),
        coord_label(da, y),
        value_label(da),
    )
