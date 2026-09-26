"""The ``DataArray.maxplot`` and ``Dataset.maxplot`` accessors.

Importing this module registers them::

    import maxplotlib.xarray  # noqa: F401

    canvas = da.maxplot.line()
    canvas.show(backend="plotly")

Every method returns a new :class:`~maxplotlib.Canvas`; choose the backend
when rendering it. ``x=``/``y=`` name the dimension or coordinate for each
axis, as in xarray. ``col=``, ``row=``, ``col_wrap=`` and ``sharey=`` lay the
data out over several subplots with :meth:`Canvas.facet`, and
``canvas_kwargs=`` is forwarded to the Canvas constructor. Everything else is
forwarded to the Canvas method of the same name (``line`` uses ``plot``).
"""

try:
    import xarray as xr
except ImportError as error:  # pragma: no cover - depends on the environment
    raise ImportError(
        "maxplotlib.xarray needs xarray; install it with "
        "`pip install maxplotlibx[xarray]`"
    ) from error

from maxplotlib.canvas.canvas import Canvas
from maxplotlib.utils import xarray_support

__all__ = ["MaxplotAccessor", "MaxplotDatasetAccessor"]

_FACET_KWARGS = ("col", "row", "col_wrap", "sharey")


@xr.register_dataarray_accessor("maxplot")
class MaxplotAccessor:
    """Plot a DataArray with maxplotlib: ``da.maxplot.line()`` etc."""

    def __init__(self, da):
        self._da = da

    def __call__(self, **kwargs):
        """Plot with a kind chosen from the dimensions, like ``da.plot()``.

        After removing any ``col``/``row`` dimensions, 1-D data (or 2-D data
        with ``hue=``) is drawn with :meth:`line` and 2-D data with
        :meth:`pcolormesh`.
        """
        facet_dims = {kwargs.get("col"), kwargs.get("row")} - {None}
        ndim = self._da.ndim - len(facet_dims & set(self._da.dims))
        if ndim == 1 or (ndim == 2 and "hue" in kwargs):
            return self.line(**kwargs)
        if ndim == 2:
            return self.pcolormesh(**kwargs)
        raise ValueError(
            f"da.maxplot() plots 1-D or 2-D data, got dims {self._da.dims}. "
            "Select a slice first, e.g. da.sel(...), or facet with col=/row=."
        )

    def line(self, **kwargs):
        """Lines against the coordinate; ``hue=<dim>`` for one per value."""
        return self._draw("plot", kwargs)

    def scatter(self, **kwargs):
        """Points against the coordinate; ``hue=<dim>`` for one series per value."""
        return self._draw("scatter", kwargs)

    def pcolormesh(self, **kwargs):
        """A pseudocolor mesh of 2-D data, with a labelled colorbar."""
        return self._draw("pcolormesh", kwargs)

    def imshow(self, **kwargs):
        """An image of 2-D data on evenly spaced coordinates."""
        return self._draw("imshow", kwargs)

    def contour(self, **kwargs):
        """Contour lines of 2-D data."""
        return self._draw("contour", kwargs)

    def contourf(self, **kwargs):
        """Filled contours of 2-D data, with a labelled colorbar."""
        return self._draw("contourf", kwargs)

    def _draw(self, kind, kwargs):
        canvas_kwargs = kwargs.pop("canvas_kwargs", None)
        # xarray's x=/y= are xcoord=/ycoord= on Canvas methods, where x and y
        # are the positional data arguments.
        for axis in ("x", "y"):
            if axis in kwargs:
                kwargs[f"{axis}coord"] = kwargs.pop(axis)
        if any(name in kwargs for name in _FACET_KWARGS):
            canvas, _ = Canvas.facet(
                self._da, kind=kind, canvas_kwargs=canvas_kwargs, **kwargs
            )
            return canvas
        canvas = Canvas(**(canvas_kwargs or {}))
        getattr(canvas, kind)(self._da, **kwargs)
        return canvas


@xr.register_dataset_accessor("maxplot")
class MaxplotDatasetAccessor:
    """Plot one Dataset variable against another: ``ds.maxplot.scatter(x=, y=)``."""

    def __init__(self, ds):
        self._ds = ds

    def line(self, x, y, hue=None, **kwargs):
        """Lines of variable ``y`` against variable or coordinate ``x``.

        ``hue`` names a dimension to draw one line per value of. Other
        arguments are as for ``DataArray.maxplot.line``, including facets.
        """
        return self._as_dataarray(x, y).maxplot.line(x=x, hue=hue, **kwargs)

    def scatter(self, x, y, hue=None, **kwargs):
        """Points of variable ``y`` against variable or coordinate ``x``.

        ``hue`` names a dimension, for one series per value, or a variable,
        which colors the points by value with a labelled colorbar (turned off
        by ``add_colorbar=False``; ``cmap``, ``robust`` and ``center`` apply).
        """
        if hue is None or hue in self._ds.dims:
            return self._as_dataarray(x, y).maxplot.scatter(x=x, hue=hue, **kwargs)
        if any(name in kwargs for name in _FACET_KWARGS):
            raise ValueError("facets are not supported with a variable as hue=")
        return self._scatter_colored(x, y, hue, kwargs)

    def _variable(self, name):
        if name not in self._ds.variables:
            raise ValueError(
                f"{name!r} is not a variable or coordinate of the Dataset; "
                f"variables are {tuple(self._ds.data_vars)}"
            )
        return self._ds[name]

    def _as_dataarray(self, x, y):
        """``y`` with ``x`` attached as a coordinate, to plot against it."""
        xvar, yvar = self._variable(x), self._variable(y)
        if x in yvar.coords:
            return yvar
        if not set(xvar.dims) <= set(yvar.dims):
            raise ValueError(
                f"{x!r} has dims {xvar.dims}, which are not all dims of "
                f"{y!r} {yvar.dims}"
            )
        return yvar.assign_coords({x: xvar})

    def _scatter_colored(self, x, y, hue, kwargs):
        canvas_kwargs = kwargs.pop("canvas_kwargs", None)
        add_colorbar = kwargs.pop("add_colorbar", True)
        xvar, yvar, cvar = xr.broadcast(
            self._variable(x), self._variable(y), self._variable(hue)
        )
        colors = xarray_support.magnitude(cvar).ravel()
        xarray_support.color_limits(colors, kwargs)
        canvas = Canvas(**(canvas_kwargs or {}))
        canvas.scatter(
            xarray_support.magnitude(xvar).ravel(),
            xarray_support.magnitude(yvar).ravel(),
            c=colors,
            **kwargs,
        )
        canvas.set_xlabel(xarray_support.value_label(xvar))
        canvas.set_ylabel(xarray_support.value_label(yvar))
        title = xarray_support.title(yvar)
        if title:
            canvas.set_title(title)
        if add_colorbar:
            canvas.colorbar(label=xarray_support.value_label(cvar))
        return canvas
