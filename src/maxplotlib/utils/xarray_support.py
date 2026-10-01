"""Optional support for plotting labelled ``xarray.DataArray`` objects.

xarray is never imported here: an object can only be a ``DataArray`` if the
caller has already imported xarray, so detection looks it up in
``sys.modules``. Install it with ``pip install maxplotlibx[xarray]``.

Units come from the ``units`` attribute, or from the data itself when it is a
pint ``Quantity`` (as with pint-xarray).
"""

import sys
from typing import NamedTuple

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


def title(da, exclude=()) -> str:
    """Title from the single-value coordinates left by ``sel``/``isel``.

    For example ``da.sel(t=1.0)`` gives ``"t = 1 s"`` if ``t`` has units ``s``.
    Coordinates in ``exclude`` are left out.
    """
    return ", ".join(
        _coord_text(name, coord, magnitude(coord))
        for name, coord in da.coords.items()
        if coord.ndim == 0 and name not in exclude
    )


def _check_dims(da, *dims):
    for dim in dims:
        if dim is not None and dim not in da.dims:
            raise ValueError(f"{dim!r} is not a dimension of {da.dims}")


def _resolve(da, name):
    """Return ``(dims, variable)`` for a dimension or coordinate name.

    ``variable`` is the coordinate, or ``None`` for a dimension without one.
    """
    if name in da.coords:
        coord = da.coords[name]
        if not set(coord.dims) <= set(da.dims):
            raise ValueError(f"coordinate {name!r} has dims outside {da.dims}")
        return coord.dims, coord
    if name in da.dims:
        return (name,), None
    raise ValueError(
        f"{name!r} is not a dimension or coordinate of the DataArray; "
        f"dims are {da.dims}, coordinates {tuple(da.coords)}"
    )


def _axis(da, name):
    """``(values, label)`` for a 1-D dimension or coordinate."""
    dims, coord = _resolve(da, name)
    if coord is None:
        return np.arange(da.sizes[name]), str(name)
    return magnitude(coord), _label(coord)


class LineData(NamedTuple):
    """Data for :func:`line_data`: one ``(positions, values, label)`` per line."""

    lines: list
    coord_label: str
    value_label: str
    vertical: bool


def line_data(da, hue=None, x=None, y=None):
    """Return the positions, value arrays and labels for plotting lines.

    A 1-D array gives one unlabelled line. A 2-D array needs ``hue``, the
    dimension to draw one line per value of. ``x`` names the dimension or
    coordinate to plot against; ``y`` does the same but puts the coordinate
    on the y-axis (``vertical`` is then True), e.g. for profiles. With
    ``hue``, the coordinate may also vary along ``hue`` (2-D), giving each
    line its own positions.
    """
    if x is not None and y is not None:
        raise ValueError("give x= or y= for a line plot, not both")
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
    name = x if x is not None else y
    if name is None:
        (dim,) = [d for d in da.dims if d != hue]
        name = dim
        coord = None
    else:
        dims, coord = _resolve(da, name)
        along = [d for d in dims if d != hue]
        if len(along) != 1 or len(dims) > 1 + (hue is not None):
            raise ValueError(
                f"{name!r} must be a dimension, or a coordinate along one "
                f"dimension other than hue (optionally also along hue); it "
                f"has dims {dims}"
            )
        (dim,) = along
    if coord is not None and coord.ndim == 2:
        positions = magnitude(coord.transpose(hue, dim))
        position_label = _label(coord)
    else:
        positions, position_label = _axis(da, name)
    if hue is None:
        lines = [(positions, magnitude(da), None)]
    else:
        hue_coord = da.coords[hue] if hue in da.coords else None
        values = magnitude(da.transpose(hue, dim))
        lines = []
        for i in range(da.sizes[hue]):
            if hue_coord is None:
                label = f"{hue} = {i}"
            else:
                label = _coord_text(hue, hue_coord, magnitude(hue_coord)[i])
            line_positions = positions[i] if np.ndim(positions) == 2 else positions
            lines.append((line_positions, values[i], label))
    return LineData(lines, position_label, value_label(da), y is not None)


class MeshData(NamedTuple):
    """Data for :func:`mesh_data`; ``x``/``y`` are 2-D on curvilinear grids."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    xlabel: str
    ylabel: str
    zlabel: str
    xname: str
    yname: str

    @property
    def curvilinear(self) -> bool:
        return np.ndim(self.x) == 2


def mesh_data(da, x=None, y=None, method="pcolormesh"):
    """Return the coordinates, values and labels for a 2-D DataArray.

    ``x``/``y`` name a dimension or a coordinate, which may be 2-D, e.g.
    physical ``R(e1, e2)`` over logical dimensions. Like
    ``xarray.DataArray.plot.pcolormesh``, the first dimension goes on the
    y-axis and the second on the x-axis by default.
    """
    if da.ndim != 2:
        raise ValueError(
            f"{method}() needs a 2-D DataArray, got {da.ndim}-D with dims "
            f"{da.dims}. Select a slice first, e.g. da.sel(...) or da.isel(...)."
        )
    if x is None and y is None:
        y, x = da.dims
    resolved = {name: _resolve(da, name) for name in (x, y) if name is not None}
    if any(len(dims) == 2 for dims, _ in resolved.values()):
        if x is None or y is None:
            raise ValueError(
                "give both x= and y= to plot on a 2-D coordinate "
                f"({', '.join(resolved)})"
            )
        return _curvilinear_mesh(da, x, y)
    if x is None:
        (ydim,) = resolved[y][0]
        (x,) = [d for d in da.dims if d != ydim]
    elif y is None:
        (xdim,) = resolved[x][0]
        (y,) = [d for d in da.dims if d != xdim]
    (xdim,), _ = _resolve(da, x)
    (ydim,), _ = _resolve(da, y)
    if xdim == ydim:
        raise ValueError(f"x and y must lie along different dimensions ({x!r}, {y!r})")
    xvalues, xlabel = _axis(da, x)
    yvalues, ylabel = _axis(da, y)
    z = magnitude(da.transpose(ydim, xdim))
    return MeshData(xvalues, yvalues, z, xlabel, ylabel, value_label(da), x, y)


def _curvilinear_mesh(da, x, y):
    def grid(name):
        dims, coord = _resolve(da, name)
        if coord is None:
            coord = type(da)(np.arange(da.sizes[name]), dims=dims)
        return magnitude(coord.broadcast_like(da).transpose(*da.dims)), coord

    xvalues, xcoord = grid(x)
    yvalues, ycoord = grid(y)
    return MeshData(
        xvalues,
        yvalues,
        magnitude(da),
        _label(xcoord) if xcoord.name is not None else str(x),
        _label(ycoord) if ycoord.name is not None else str(y),
        value_label(da),
        x,
        y,
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


def color_limits(values, kwargs):
    """Apply xarray's color defaults to ``kwargs`` for color-mapped ``values``.

    Pops ``robust`` and ``center``. ``robust=True`` takes ``vmin``/``vmax``
    from the 2nd and 98th percentiles. Data that crosses ``center`` (default:
    0, when it has both signs and ``vmin``/``vmax`` are not both given) gets
    limits symmetric about it and, unless ``cmap`` or ``colors`` is given,
    the diverging ``"RdBu_r"`` colormap. ``center=False`` turns that off.
    A ``norm`` is left alone.
    """
    robust = kwargs.pop("robust", False)
    center = kwargs.pop("center", None)
    if kwargs.get("norm") is not None:
        return kwargs
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.number):
        return kwargs
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return kwargs
    if robust:
        low, high = np.percentile(finite, [2, 98])
    else:
        low, high = finite.min(), finite.max()
    vmin, vmax = kwargs.get("vmin"), kwargs.get("vmax")
    low = low if vmin is None else vmin
    high = high if vmax is None else vmax
    if center is None:
        both_given = vmin is not None and vmax is not None
        divergent = not both_given and low < 0 < high
        center = 0.0
    else:
        divergent = center is not False
    if divergent:
        half_range = max(abs(low - center), abs(high - center))
        low = center - half_range if vmin is None else vmin
        high = center + half_range if vmax is None else vmax
        if kwargs.get("colors") is None:  # contour(colors=...) excludes cmap
            kwargs.setdefault("cmap", "RdBu_r")
    if divergent or robust:
        kwargs["vmin"], kwargs["vmax"] = float(low), float(high)
    return kwargs
