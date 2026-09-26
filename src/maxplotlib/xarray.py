"""The ``DataArray.maxplot`` accessor.

Importing this module registers it::

    import maxplotlib.xarray  # noqa: F401

    canvas = da.maxplot.line()
    canvas.show(backend="plotly")

Every method returns a new :class:`~maxplotlib.Canvas`; choose the backend
when rendering it. ``col=``, ``row=``, ``col_wrap=`` and ``sharey=`` lay the
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

__all__ = ["MaxplotAccessor"]

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
        if any(name in kwargs for name in _FACET_KWARGS):
            canvas, _ = Canvas.facet(
                self._da, kind=kind, canvas_kwargs=canvas_kwargs, **kwargs
            )
            return canvas
        canvas = Canvas(**(canvas_kwargs or {}))
        getattr(canvas, kind)(self._da, **kwargs)
        return canvas
