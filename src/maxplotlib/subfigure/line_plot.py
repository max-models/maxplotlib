import re

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tikzfigure import TikzFigure

# Keyword arguments that every drawing method accepts and that are handled by
# maxplotlib itself rather than being forwarded to a backend drawing call.
_NEUTRAL_KWARGS = ("hover", "meta")

_TIKZ_SUPPORTED_PLOT_TYPES = {
    "plot",
    "scatter",
    "bar",
    "barh",
    "fill_between",
    "errorbar",
    "step",
    "stairs",
    "stem",
    "hlines",
    "vlines",
    "axvspan",
    "axhspan",
    "fill",
    "gantt",
    "flame_chart",
}


def _sample_colormap(colormap, count, *, css=True):
    """Sample ``count`` colors from a colormap, for either backend.

    ``colormap`` may be a Matplotlib colormap name (``"viridis"``), a Plotly
    colorscale name (``"Viridis"``), a Matplotlib ``Colormap`` instance, or an
    explicit sequence of colors.  Returns CSS color strings when ``css`` is
    true (Plotly) and Matplotlib RGBA tuples otherwise, so one ``colormap=``
    argument serves every backend.
    """
    count = max(int(count), 1)
    if isinstance(colormap, (list, tuple)):
        colors = list(colormap)
        if not colors:
            colors = ["#1f77b4"]
        sampled = [colors[index % len(colors)] for index in range(count)]
        if css:
            import matplotlib.colors as mcolors

            return [
                (
                    color
                    if isinstance(color, str)
                    else mcolors.to_hex(color, keep_alpha=True)
                )
                for color in sampled
            ]
        return sampled

    positions = np.linspace(0, 1, count) if count > 1 else np.array([0.5])
    try:
        import matplotlib.colors as mcolors

        cmap = plt.get_cmap(colormap)
        rgba = [cmap(float(position)) for position in positions]
        if css:
            return [mcolors.to_hex(color, keep_alpha=False) for color in rgba]
        return rgba
    except (ValueError, TypeError):
        pass

    from plotly.colors import sample_colorscale

    css_colors = sample_colorscale(colormap, list(positions))
    if css:
        return list(css_colors)

    import matplotlib.colors as mcolors

    converted = []
    for color in css_colors:
        numbers = [float(value) for value in re.findall(r"[\d.]+", str(color))]
        if len(numbers) >= 3:
            converted.append(tuple(value / 255.0 for value in numbers[:3]) + (1.0,))
        else:
            converted.append(mcolors.to_rgba(color))
    return converted


def _flame_frame_colors(kwargs, depths, count, *, css):
    """Resolve one color per flame-chart frame.

    ``colors=`` (one entry per frame, or a single color) wins; otherwise the
    frames are colored by stack depth from ``colormap=``.
    """
    explicit = kwargs.get("colors", kwargs.get("color"))
    if explicit is not None:
        if isinstance(explicit, str) or not hasattr(explicit, "__len__"):
            return [explicit] * count
        explicit = list(explicit)
        if explicit:
            return [explicit[index % len(explicit)] for index in range(count)]

    default = "Viridis" if css else "viridis"
    colormap = kwargs.get("colormap", default)
    max_depth = int(depths.max()) + 1 if count else 1
    palette = _sample_colormap(colormap, max_depth, css=css)
    return [palette[int(depth) % len(palette)] for depth in depths]


def _colormap_to_plotly_colorscale(colormap, steps=17):
    """Convert a Matplotlib colormap name into a Plotly colorscale.

    Plotly only recognizes its own colorscale names (``"Viridis"``, not
    ``"viridis"``, and nothing at all for names like ``"coolwarm"``), so a
    Matplotlib ``cmap=`` on a value-colored scatter has to be resampled into
    an explicit ``[[fraction, color], ...]`` list to render at all.
    """
    positions = np.linspace(0, 1, steps)
    colors = _sample_colormap(colormap, steps, css=True)
    return [[float(position), color] for position, color in zip(positions, colors)]


def _tikz_style_kwargs(kwargs, *, default_color="black"):
    """Translate common Matplotlib-style options to pgfplots/TikZ options."""
    kwargs = dict(kwargs)
    style = {}
    if kwargs.get("color") is not None:
        style["color"] = kwargs["color"]
    else:
        style["color"] = default_color
    if kwargs.get("linewidth") is not None:
        style["line_width"] = kwargs["linewidth"]
    if kwargs.get("alpha") is not None:
        style["opacity"] = kwargs["alpha"]
    if kwargs.get("linestyle") in {"--", "dashed"}:
        style["dash_pattern"] = "on 4pt off 2pt"
    elif kwargs.get("linestyle") in {":", "dotted"}:
        style["dash_pattern"] = "on 1pt off 2pt"
    elif kwargs.get("linestyle") == "-.":
        style["dash_pattern"] = "on 4pt off 2pt on 1pt off 2pt"
    if kwargs.get("marker") is not None:
        style["mark"] = kwargs["marker"]
    if kwargs.get("markersize") is not None:
        style["mark_size"] = f"{kwargs['markersize']}pt"
    return style


def _tikz_error_bounds(error, values):
    """Return lower and upper error arrays in Matplotlib's common formats."""
    if error is None:
        return None
    error = np.asarray(error, dtype=float)
    values = np.asarray(values, dtype=float)
    if error.ndim == 0:
        error = np.full(values.shape, error.item())
    if error.ndim == 2 and error.shape[0] == 2:
        return error[0], error[1]
    return error, error


def _tikz_step_coordinates(x, y, where="pre"):
    """Expand line data into explicit coordinates for a stepped path."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2:
        return x, y
    if where == "post":
        step_x = np.repeat(x, 2)[1:]
        step_y = np.repeat(y, 2)[:-1]
    elif where == "mid":
        mids = (x[:-1] + x[1:]) / 2
        step_x = np.ravel(np.column_stack((x[:-1], mids, mids, x[1:])))
        step_y = np.ravel(np.column_stack((y[:-1], y[:-1], y[1:], y[1:])))
        return step_x, step_y
    else:
        step_x = np.repeat(x, 2)[:-1]
        step_y = np.repeat(y, 2)[1:]
    return step_x, step_y


class Node:
    def __init__(self, x, y, label="", content="", layer=0, **kwargs):
        self.x = x
        self.y = y
        self.label = label
        self.content = content
        self.layer = layer
        self.options = kwargs


class Path:
    def __init__(
        self,
        nodes,
        path_actions=[],
        cycle=False,
        label="",
        layer=0,
        **kwargs,
    ):
        self.nodes = nodes
        self.path_actions = path_actions
        self.cycle = cycle
        self.layer = layer
        self.label = label
        self.options = kwargs


class LinePlot:
    """A subplot's drawing surface.

    Every drawing method accepts two backend-neutral keyword arguments in
    addition to the ones it forwards to the backend:

    ``hover``
        Hover text for the artists this call creates. A single string, one
        entry per point, or a 2-D array for image-like plots. Rendered by the
        Plotly backend (as ``hovertext``, with an invisible overlay trace for
        shapes, which cannot show hover text themselves) and ignored by the
        other backends.
    ``meta``
        An opaque tag attached to everything this call creates: ``meta`` on
        Plotly traces, ``gid`` on Matplotlib artists. Use it to find the
        artists again by identity instead of by drawing order.

    Neither is forwarded to the backend drawing call, so ``hover=`` never
    reaches ``ax.bar()``.
    """

    def __init__(
        self,
        title: str | None = None,
        grid: bool = False,
        legend: bool = False,
        xmin: float | int | None = None,
        xmax: float | int | None = None,
        ymin: float | int | None = None,
        ymax: float | int | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xscale: float | int = 1.0,
        yscale: float | int = 1.0,
        xshift: float | int = 0.0,
        yshift: float | int = 0.0,
    ):
        """
        Initialize the LinePlot class for a subplot.

        Parameters:
            title (str): Title of the plot.
            caption (str): Caption for the plot.
            description (str): Description of the plot.
            label (str): Label for the plot.
            grid (bool): Whether to display grid lines (default is False).
            legend (bool): Whether to display legend (default is False).
            xmin, xmax, ymin, ymax (float): Axis limits.
            xlabel, ylabel (str): Axis labels.
            xscale, yscale (float): Scaling factors for axes.
            xshift, yshift (float): Shifts for axes.
        """

        self._title = title
        self._caption = None
        self._grid = grid
        self._legend = legend
        self._legend_kwargs: dict = {}
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._xlabel_kwargs: dict = {}
        self._ylabel_kwargs: dict = {}
        self._title_kwargs: dict = {}
        self._tick_params: dict = {}
        self._xscale = xscale
        self._yscale = yscale
        self._xshift = xshift
        self._yshift = yshift

        # Axis scale type ('linear', 'log', 'symlog')
        self._xaxis_scale: str | None = None
        self._yaxis_scale: str | None = None
        self._axis_off = False
        self._axisbelow = None
        self._facecolor = None
        self._margins: dict = {}
        self._invert_xaxis = False
        self._invert_yaxis = False
        self._minorticks = None
        self._locator_params: dict = {}
        self._ticklabel_format: dict = {}
        self._bar_label_kwargs: dict | None = None
        self._clabel_kwargs: dict | None = None
        self._rasterization_zorder = None
        self._axis_settings: dict = {}
        self._autoscale_settings: dict | None = None
        self._autoscale_view_settings: dict | None = None
        self._relim_settings: dict | None = None
        self._box_aspect = None
        self._secondary_xaxis_settings: dict | None = None
        self._secondary_yaxis_settings: dict | None = None
        self._frame_on = None
        self._visible = None
        self._alpha = None
        self._zorder = None
        self._rasterized = None
        self._autoscale_on = None
        self._autoscalex_on = None
        self._autoscaley_on = None
        self._xmargin = None
        self._ymargin = None
        self._adjustable = None
        self._anchor = None

        # Custom tick positions and labels
        self._xticks: list | None = None
        self._xticklabels: list | None = None
        self._xtick_kwargs: dict = {}
        self._xticklabel_kwargs: dict = {}
        self._yticks: list | None = None
        self._yticklabels: list | None = None
        self._ytick_kwargs: dict = {}
        self._yticklabel_kwargs: dict = {}

        # Aspect ratio
        self._aspect = None

        # List to store line data, each entry contains x and y data, label, and plot kwargs
        self.line_data = []
        self.layered_line_data = {}

        # Initialize lists to hold Node and Path objects
        self.nodes = []
        self.paths = []

        # Counter for unnamed nodes
        self._node_counter = 0

    def add_caption(self, caption):
        self._caption = caption

    def _add(self, obj, layer):
        # ``hover`` and ``meta`` are backend-neutral: they are meaningful to
        # Plotly (hover text / addressable traces) and meaningless to the
        # other backends, which forward **kwargs straight to their drawing
        # calls.  Lift them out of ``kwargs`` here, once, so that every
        # drawing method accepts them without leaking them into Matplotlib.
        kwargs = obj.get("kwargs")
        if isinstance(kwargs, dict):
            for key in _NEUTRAL_KWARGS:
                if key in kwargs:
                    obj[key] = kwargs.pop(key)
        for key in _NEUTRAL_KWARGS:
            obj.setdefault(key, None)
        self.line_data.append(obj)
        if layer in self.layered_line_data:
            self.layered_line_data[layer].append(obj)
        else:
            self.layered_line_data[layer] = [obj]

    def add_line(
        self,
        x,
        y,
        layer=0,
        **kwargs,
    ):
        """
        Add a line to the subplot.

        Parameters:
        x (array-like): X-axis data.
        y (array-like): Y-axis data.
        layer (int): Layer index (default 0).
        **kwargs: Additional keyword arguments forwarded to the backend
            (e.g., color, linestyle, label, linewidth).
        """
        ld = {
            "x": np.array(x),
            "y": np.array(y),
            "layer": layer,
            "plot_type": "plot",
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def plot(self, x, y, layer=0, **kwargs):
        """Matplotlib-style alias for :meth:`add_line`."""
        self.add_line(x, y, layer=layer, **kwargs)

    def scatter(self, x, y, layer=0, **kwargs):
        """
        Add a scatter plot to the subplot.

        Parameters:
        x (array-like): X-axis data.
        y (array-like): Y-axis data.
        layer (int): Layer index (default 0).
        **kwargs: Additional keyword arguments forwarded to the backend
            (e.g., color, marker, s, label).
        """
        ld = {
            "x": np.array(x),
            "y": np.array(y),
            "layer": layer,
            "plot_type": "scatter",
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def bar(self, x, height, layer=0, **kwargs):
        """
        Add a bar chart to the subplot.

        Parameters:
        x (array-like): X positions of the bars.
        height (array-like): Heights of the bars.
        layer (int): Layer index (default 0).
        **kwargs: Additional keyword arguments forwarded to the backend
            (e.g., color, width, bottom, alpha, edgecolor, linewidth, label).
            ``bottom`` stacks the bars on every backend (it becomes Plotly's
            ``base``).
        """
        ld = {
            "x": np.array(x),
            "height": np.array(height),
            "layer": layer,
            "plot_type": "bar",
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def barh(self, y, width, layer=0, **kwargs):
        """Add a horizontal bar chart.

        ``left`` offsets the bars on every backend (it becomes Plotly's
        ``base``); ``height``, ``alpha`` and ``edgecolor`` are honored too.
        """
        self._add(
            {
                "y": np.array(y),
                "width": np.array(width),
                "layer": layer,
                "plot_type": "barh",
                "kwargs": kwargs,
            },
            layer,
        )

    def hist(self, x, bins=10, layer=0, **kwargs):
        """Add a histogram."""
        self._add(
            {
                "x": np.array(x),
                "bins": bins,
                "layer": layer,
                "plot_type": "hist",
                "kwargs": kwargs,
            },
            layer,
        )

    def step(self, x, y, layer=0, **kwargs):
        """Add a step plot."""
        self._add(
            {
                "x": np.array(x),
                "y": np.array(y),
                "layer": layer,
                "plot_type": "step",
                "kwargs": kwargs,
            },
            layer,
        )

    def stairs(self, values, edges=None, baseline=0, layer=0, **kwargs):
        """Add a stairs plot."""
        self._add(
            {
                "values": np.array(values),
                "edges": edges,
                "baseline": baseline,
                "layer": layer,
                "plot_type": "stairs",
                "kwargs": kwargs,
            },
            layer,
        )

    def broken_barh(self, xranges, yrange, layer=0, **kwargs):
        """Add horizontal bars with gaps between x ranges."""
        self._add(
            {
                "xranges": list(xranges),
                "yrange": yrange,
                "layer": layer,
                "plot_type": "broken_barh",
                "kwargs": kwargs,
            },
            layer,
        )

    def pie(self, x, layer=0, **kwargs):
        """Add a pie chart."""
        self._add(
            {"x": np.array(x), "layer": layer, "plot_type": "pie", "kwargs": kwargs},
            layer,
        )

    def bar_label(self, **kwargs):
        """Add labels to bar containers in the Matplotlib backend."""
        self._bar_label_kwargs = dict(kwargs)

    def clabel(self, **kwargs):
        """Label contour lines and filled contour levels."""
        self._clabel_kwargs = dict(kwargs)

    def set_rasterization_zorder(self, z):
        """Rasterize artists below the given z-order when exporting."""
        self._rasterization_zorder = z

    def stem(self, x, y, layer=0, **kwargs):
        """Add a stem plot."""
        self._add(
            {
                "x": np.array(x),
                "y": np.array(y),
                "layer": layer,
                "plot_type": "stem",
                "kwargs": kwargs,
            },
            layer,
        )

    def stackplot(self, x, *ys, layer=0, **kwargs):
        """Add a stacked area plot."""
        self._add(
            {
                "x": np.array(x),
                "ys": [np.array(y) for y in ys],
                "layer": layer,
                "plot_type": "stackplot",
                "kwargs": kwargs,
            },
            layer,
        )

    def boxplot(self, x, layer=0, **kwargs):
        """Add a box-and-whisker plot."""
        self._add(
            {"x": x, "layer": layer, "plot_type": "boxplot", "kwargs": kwargs},
            layer,
        )

    def violinplot(self, dataset, layer=0, **kwargs):
        """Add a violin plot."""
        self._add(
            {
                "dataset": dataset,
                "layer": layer,
                "plot_type": "violinplot",
                "kwargs": kwargs,
            },
            layer,
        )

    def eventplot(self, positions, layer=0, **kwargs):
        """Add an event/rug plot."""
        self._add(
            {
                "positions": positions,
                "layer": layer,
                "plot_type": "eventplot",
                "kwargs": kwargs,
            },
            layer,
        )

    def contour(self, x, y, z, layer=0, **kwargs):
        """Add contour lines for a 2D scalar field."""
        self._add(
            {
                "x": x,
                "y": y,
                "z": np.asarray(z),
                "layer": layer,
                "plot_type": "contour",
                "kwargs": kwargs,
            },
            layer,
        )

    def contourf(self, x, y, z, layer=0, **kwargs):
        """Add filled contours for a 2D scalar field."""
        self._add(
            {
                "x": x,
                "y": y,
                "z": np.asarray(z),
                "layer": layer,
                "plot_type": "contourf",
                "kwargs": kwargs,
            },
            layer,
        )

    def pcolormesh(self, x, y, z, layer=0, **kwargs):
        """Add a pseudocolor mesh."""
        self._add(
            {
                "x": x,
                "y": y,
                "z": np.asarray(z),
                "layer": layer,
                "plot_type": "pcolormesh",
                "kwargs": kwargs,
            },
            layer,
        )

    def hexbin(self, x, y, layer=0, **kwargs):
        """Add a hexagonal bin density plot."""
        self._add(
            {
                "x": np.asarray(x),
                "y": np.asarray(y),
                "layer": layer,
                "plot_type": "hexbin",
                "kwargs": kwargs,
            },
            layer,
        )

    def matshow(self, data, layer=0, **kwargs):
        """Display a matrix with matrix-oriented axes."""
        self._add(
            {
                "data": np.asarray(data),
                "layer": layer,
                "plot_type": "matshow",
                "kwargs": kwargs,
            },
            layer,
        )

    def quiver(self, x, y, u, v, layer=0, **kwargs):
        """Add a vector field."""
        self._add(
            {
                "x": x,
                "y": y,
                "u": u,
                "v": v,
                "layer": layer,
                "plot_type": "quiver",
                "kwargs": kwargs,
            },
            layer,
        )

    def triplot(self, x, y, triangles=None, layer=0, **kwargs):
        """Add an unstructured triangular grid."""
        self._add(
            {
                "x": x,
                "y": y,
                "triangles": triangles,
                "layer": layer,
                "plot_type": "triplot",
                "kwargs": kwargs,
            },
            layer,
        )

    def tripcolor(self, x, y, c, triangles=None, layer=0, **kwargs):
        """Add a colored unstructured triangular grid."""
        self._add(
            {
                "x": x,
                "y": y,
                "c": c,
                "triangles": triangles,
                "layer": layer,
                "plot_type": "tripcolor",
                "kwargs": kwargs,
            },
            layer,
        )

    def tricontour(self, x, y, z, triangles=None, layer=0, **kwargs):
        """Add contour lines on an unstructured triangular grid."""
        self._add(
            {
                "x": x,
                "y": y,
                "z": z,
                "triangles": triangles,
                "layer": layer,
                "plot_type": "tricontour",
                "kwargs": kwargs,
            },
            layer,
        )

    def tricontourf(self, x, y, z, triangles=None, layer=0, **kwargs):
        """Add filled contours on an unstructured triangular grid."""
        self._add(
            {
                "x": x,
                "y": y,
                "z": z,
                "triangles": triangles,
                "layer": layer,
                "plot_type": "tricontourf",
                "kwargs": kwargs,
            },
            layer,
        )

    def streamplot(self, x, y, u, v, layer=0, **kwargs):
        """Add streamlines for a 2D vector field."""
        self._add(
            {
                "x": x,
                "y": y,
                "u": u,
                "v": v,
                "layer": layer,
                "plot_type": "streamplot",
                "kwargs": kwargs,
            },
            layer,
        )

    def pcolor(self, x, y, z, layer=0, **kwargs):
        """Add a pseudocolor plot."""
        self._add(
            {
                "x": x,
                "y": y,
                "z": np.asarray(z),
                "layer": layer,
                "plot_type": "pcolor",
                "kwargs": kwargs,
            },
            layer,
        )

    def pcolorfast(self, x, y, z, layer=0, **kwargs):
        """Add a fast pseudocolor plot."""
        self._add(
            {
                "x": x,
                "y": y,
                "z": np.asarray(z),
                "layer": layer,
                "plot_type": "pcolorfast",
                "kwargs": kwargs,
            },
            layer,
        )

    def spy(self, matrix, layer=0, **kwargs):
        """Visualize the sparsity pattern of a matrix."""
        self._add(
            {"matrix": matrix, "layer": layer, "plot_type": "spy", "kwargs": kwargs},
            layer,
        )

    def table(self, cellText=None, layer=0, **kwargs):
        """Add a table annotation to the axes."""
        self._add(
            {
                "cellText": cellText,
                "layer": layer,
                "plot_type": "table",
                "kwargs": kwargs,
            },
            layer,
        )

    def add_table(self, cellText=None, layer=0, **kwargs):
        """Matplotlib-style alias for ``table``."""
        self.table(cellText=cellText, layer=layer, **kwargs)

    def gantt(self, tasks, start_times, durations, layer=0, **kwargs):
        """
        Add a Gantt chart to the subplot.

        Parameters:
        tasks (array-like): Task names or labels (y-axis).
        start_times (array-like): Start times for each task (x-axis).
        durations (array-like): Duration of each task.
        layer (int): Layer index (default 0).
        **kwargs: Additional keyword arguments forwarded to the backend
            (e.g., color, alpha, edgecolor, label).
        """
        ld = {
            "tasks": list(tasks),
            "start_times": np.array(start_times),
            "durations": np.array(durations),
            "layer": layer,
            "plot_type": "gantt",
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def flame_chart(self, labels, parents, values, start_times=None, layer=0, **kwargs):
        """
        Add a flame chart to the subplot for hierarchical profiling data.

        Parameters:
        labels (array-like): Labels for each stack frame/function.
        parents (array-like): Parent indices for each frame (None for root, or index of parent).
        values (array-like): Duration/sample count for each frame.
        start_times (array-like, optional): Start times for each frame. If None, computed from hierarchy.
        layer (int): Layer index (default 0).
        **kwargs: Additional keyword arguments forwarded to the backend
            (e.g., colormap, colors, edgecolor, label). ``colormap`` accepts a
            Matplotlib colormap name or a Plotly colorscale name on either
            backend; ``colors`` sets an explicit color per frame, which colors
            frames by region rather than by stack depth.
        """
        ld = {
            "labels": list(labels),
            "parents": list(parents),
            "values": np.array(values),
            "start_times": np.array(start_times) if start_times is not None else None,
            "layer": layer,
            "plot_type": "flame_chart",
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def axhline(self, y=0, layer=0, **kwargs):
        """
        Add a horizontal line spanning the full width of the axes.

        Parameters:
        y (float): Y-coordinate of the line (default 0).
        **kwargs: Additional keyword arguments (e.g., color, linestyle, label).
        """
        ld = {
            "y": y,
            "layer": layer,
            "plot_type": "axhline",
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def axvline(self, x=0, layer=0, **kwargs):
        """
        Add a vertical line spanning the full height of the axes.

        Parameters:
        x (float): X-coordinate of the line (default 0).
        **kwargs: Additional keyword arguments (e.g., color, linestyle, label).
        """
        ld = {
            "x": x,
            "layer": layer,
            "plot_type": "axvline",
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def set_xlabel(self, label: str, **kwargs):
        """Set the x-axis label and its text properties."""
        self._xlabel = label
        self._xlabel_kwargs = dict(kwargs)

    def set_ylabel(self, label: str, **kwargs):
        """Set the y-axis label and its text properties."""
        self._ylabel = label
        self._ylabel_kwargs = dict(kwargs)

    def set_title(self, title: str, **kwargs):
        """Set the subplot title and its text properties."""
        self._title = title
        self._title_kwargs = dict(kwargs)

    def set_xlim(self, left=None, right=None):
        """Set the x-axis limits."""
        if left is not None:
            self._xmin = left
        if right is not None:
            self._xmax = right

    def set_ylim(self, bottom=None, top=None):
        """Set the y-axis limits."""
        if bottom is not None:
            self._ymin = bottom
        if top is not None:
            self._ymax = top

    def set_grid(self, visible: bool = True):
        """Show or hide the grid."""
        self._grid = visible

    def tick_params(self, **kwargs):
        """Configure tick appearance using Matplotlib-style keyword arguments."""
        self._tick_params = dict(kwargs)

    def set_legend(self, visible: bool = True, **kwargs):
        """Show or hide the legend."""
        self._legend = visible
        self._legend_kwargs = dict(kwargs)

    def set_xscale(self, scale: str):
        """Set the x-axis scale type: 'linear', 'log', or 'symlog'."""
        self._xaxis_scale = scale

    def set_yscale(self, scale: str):
        """Set the y-axis scale type: 'linear', 'log', or 'symlog'."""
        self._yaxis_scale = scale

    def set_axis_off(self):
        """Hide the axis frame, ticks, and labels."""
        self._axis_off = True

    def set_axis_on(self):
        """Show the axis frame, ticks, and labels."""
        self._axis_off = False

    def set_axisbelow(self, state=True):
        """Set whether axis gridlines and ticks are drawn below plot data."""
        self._axisbelow = state

    def set_facecolor(self, color):
        """Set the subplot background color."""
        self._facecolor = color

    def set_frame_on(self, state):
        self._frame_on = state

    def set_visible(self, state):
        self._visible = state

    def set_alpha(self, alpha):
        self._alpha = alpha

    def set_zorder(self, zorder):
        self._zorder = zorder

    def set_rasterized(self, rasterized):
        self._rasterized = rasterized

    def set_autoscale_on(self, enable):
        self._autoscale_on = enable

    def set_autoscalex_on(self, enable):
        self._autoscalex_on = enable

    def set_autoscaley_on(self, enable):
        self._autoscaley_on = enable

    def set_xbound(self, lower=None, upper=None):
        self.set_xlim(lower, upper)

    def set_ybound(self, lower=None, upper=None):
        self.set_ylim(lower, upper)

    def set_xmargin(self, margin):
        self._xmargin = margin

    def set_ymargin(self, margin):
        self._ymargin = margin

    def get_adjustable(self):
        return self._adjustable

    def get_anchor(self):
        return self._anchor

    def get_alpha(self):
        return self._alpha

    def get_box_aspect(self):
        return self._box_aspect

    def get_facecolor(self):
        return self._facecolor

    def get_frame_on(self):
        return self._frame_on

    def get_legend(self):
        return self._legend

    def get_rasterization_zorder(self):
        return self._rasterization_zorder

    def get_rasterized(self):
        return self._rasterized

    def get_visible(self):
        return self._visible

    def get_zorder(self):
        return self._zorder

    def get_xbound(self):
        return self._xmin, self._xmax

    def get_ybound(self):
        return self._ymin, self._ymax

    def get_xmargin(self):
        return self._xmargin

    def get_ymargin(self):
        return self._ymargin

    def margins(self, *args, **kwargs):
        """Set x/y data margins using Matplotlib-style arguments."""
        self._margins = {"args": args, **kwargs}

    def invert_xaxis(self):
        """Invert the x-axis direction."""
        self._invert_xaxis = True

    def invert_yaxis(self):
        """Invert the y-axis direction."""
        self._invert_yaxis = True

    def minorticks_on(self):
        """Enable minor ticks."""
        self._minorticks = True

    def minorticks_off(self):
        """Disable minor ticks."""
        self._minorticks = False

    def locator_params(self, **kwargs):
        """Set axis locator parameters."""
        self._locator_params = dict(kwargs)

    def ticklabel_format(self, **kwargs):
        """Configure tick-label numeric formatting."""
        self._ticklabel_format = dict(kwargs)

    def set_xticks(self, ticks, labels=None, **kwargs):
        """Set x-axis tick positions, labels, and label properties.

        Keyword arguments are forwarded to the plotting backend. For example,
        ``rotation=45`` rotates the tick labels in the Matplotlib and Plotly
        backends.
        """
        self._xticks = list(ticks)
        self._xticklabels = list(labels) if labels is not None else None
        self._xtick_kwargs = dict(kwargs)

    def set_yticks(self, ticks, labels=None, **kwargs):
        """Set y-axis tick positions, labels, and label properties."""
        self._yticks = list(ticks)
        self._yticklabels = list(labels) if labels is not None else None
        self._ytick_kwargs = dict(kwargs)

    def set_aspect(self, aspect):
        """Set the axes aspect ratio: 'equal', 'auto', or a float."""
        self._aspect = aspect

    def set_adjustable(self, adjustable):
        self._adjustable = adjustable

    def set_anchor(self, anchor):
        self._anchor = anchor

    def set_fc(self, color):
        self.set_facecolor(color)

    def set(self, **kwargs):
        """Set common axes properties using Matplotlib-style names."""
        handlers = {
            "title": self.set_title,
            "xlabel": self.set_xlabel,
            "ylabel": self.set_ylabel,
            "xlim": lambda value: self.set_xlim(*value),
            "ylim": lambda value: self.set_ylim(*value),
            "xscale": self.set_xscale,
            "yscale": self.set_yscale,
            "facecolor": self.set_facecolor,
            "fc": self.set_fc,
            "aspect": self.set_aspect,
            "adjustable": self.set_adjustable,
            "anchor": self.set_anchor,
            "visible": self.set_visible,
            "alpha": self.set_alpha,
            "zorder": self.set_zorder,
        }
        for name, value in kwargs.items():
            if name not in handlers:
                raise AttributeError(f"Unknown LinePlot property: {name}")
            handlers[name](value)
        return kwargs

    def update(self, kwargs):
        return self.set(**dict(kwargs))

    def xaxis_inverted(self):
        return self._invert_xaxis

    def yaxis_inverted(self):
        return self._invert_yaxis

    def axis(self, *args, **kwargs):
        """Set Matplotlib-style axis limits or modes."""
        self._axis_settings = {"args": args, **kwargs}

    def autoscale(self, enable=True, axis="both", tight=None):
        """Configure autoscaling of one or both axes."""
        self._autoscale_settings = {
            "enable": enable,
            "axis": axis,
            "tight": tight,
        }

    def autoscale_view(self, tight=None, scalex=True, scaley=True):
        """Configure autoscaling using the current data limits."""
        self._autoscale_view_settings = {
            "tight": tight,
            "scalex": scalex,
            "scaley": scaley,
        }

    def relim(self, visible_only=False):
        """Recompute data limits before autoscaling."""
        self._relim_settings = {"visible_only": visible_only}

    def set_box_aspect(self, aspect):
        """Set the physical height-to-width ratio of the axes box."""
        self._box_aspect = aspect

    def secondary_xaxis(self, location="top", functions=None, **kwargs):
        """Add a Matplotlib secondary x-axis using forward/inverse functions."""
        self._secondary_xaxis_settings = {
            "location": location,
            "functions": functions,
            "kwargs": kwargs,
        }

    def secondary_yaxis(self, location="right", functions=None, **kwargs):
        """Add a Matplotlib secondary y-axis using forward/inverse functions."""
        self._secondary_yaxis_settings = {
            "location": location,
            "functions": functions,
            "kwargs": kwargs,
        }

    def semilogx(self, x, y, layer=0, **kwargs):
        """Add a line while using a logarithmic x-axis."""
        self.set_xscale("log")
        self.plot(x, y, layer=layer, **kwargs)

    def semilogy(self, x, y, layer=0, **kwargs):
        """Add a line while using a logarithmic y-axis."""
        self.set_yscale("log")
        self.plot(x, y, layer=layer, **kwargs)

    def loglog(self, x, y, layer=0, **kwargs):
        """Add a line while using logarithmic x- and y-axes."""
        self.set_xscale("log")
        self.set_yscale("log")
        self.plot(x, y, layer=layer, **kwargs)

    def set_xticklabels(self, labels, **kwargs):
        """Set x tick labels and their text properties."""
        self._xticklabels = list(labels)
        self._xticklabel_kwargs = dict(kwargs)

    def set_yticklabels(self, labels, **kwargs):
        """Set y tick labels and their text properties."""
        self._yticklabels = list(labels)
        self._yticklabel_kwargs = dict(kwargs)

    def fill_between(self, x, y1, y2=0, layer=0, **kwargs):
        """
        Fill the region between two curves.

        Parameters:
        x (array-like): X-axis data.
        y1 (array-like): Upper boundary.
        y2 (array-like or scalar): Lower boundary (default 0).
        layer (int): Layer index.
        **kwargs: Forwarded to the backend (e.g., color, alpha, label).
        """
        ld = {
            "x": np.array(x),
            "y1": np.array(y1) if not np.isscalar(y1) else y1,
            "y2": np.array(y2) if not np.isscalar(y2) else y2,
            "layer": layer,
            "plot_type": "fill_between",
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def fill(self, *args, layer=0, **kwargs):
        """Fill one or more polygonal regions."""
        self._add(
            {"args": args, "layer": layer, "plot_type": "fill", "kwargs": kwargs},
            layer,
        )

    def fill_betweenx(self, y, x1, x2=0, layer=0, **kwargs):
        """Fill the area between two x-boundaries along y."""
        self._add(
            {
                "y": np.array(y),
                "x1": np.array(x1) if not np.isscalar(x1) else x1,
                "x2": np.array(x2) if not np.isscalar(x2) else x2,
                "layer": layer,
                "plot_type": "fill_betweenx",
                "kwargs": kwargs,
            },
            layer,
        )

    def errorbar(self, x, y, yerr=None, xerr=None, layer=0, **kwargs):
        """
        Add a line plot with error bars.

        Parameters:
        x (array-like): X-axis data.
        y (array-like): Y-axis data.
        yerr (array-like or scalar, optional): Y-axis error.
        xerr (array-like or scalar, optional): X-axis error.
        layer (int): Layer index.
        **kwargs: Forwarded to the backend (e.g., color, fmt, capsize, label).
        """
        ld = {
            "x": np.array(x),
            "y": np.array(y),
            "yerr": yerr,
            "xerr": xerr,
            "layer": layer,
            "plot_type": "errorbar",
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def hlines(self, y, xmin, xmax, layer=0, **kwargs):
        """
        Draw horizontal lines at each y from xmin to xmax.

        Parameters:
        y (float or array-like): Y positions.
        xmin, xmax (float or array-like): Start and end of each line.
        **kwargs: Forwarded to the backend (e.g., colors, linestyles, label).
        """
        ld = {
            "y": y,
            "xmin": xmin,
            "xmax": xmax,
            "layer": layer,
            "plot_type": "hlines",
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def vlines(self, x, ymin, ymax, layer=0, **kwargs):
        """
        Draw vertical lines at each x from ymin to ymax.

        Parameters:
        x (float or array-like): X positions.
        ymin, ymax (float or array-like): Start and end of each line.
        **kwargs: Forwarded to the backend (e.g., colors, linestyles, label).
        """
        ld = {
            "x": x,
            "ymin": ymin,
            "ymax": ymax,
            "layer": layer,
            "plot_type": "vlines",
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def axvspan(self, xmin, xmax, layer=0, **kwargs):
        """Add a vertical shaded span across the axes."""
        self._add(
            {
                "xmin": xmin,
                "xmax": xmax,
                "layer": layer,
                "plot_type": "axvspan",
                "kwargs": kwargs,
            },
            layer,
        )

    def axhspan(self, ymin, ymax, layer=0, **kwargs):
        """Add a horizontal shaded span across the axes."""
        self._add(
            {
                "ymin": ymin,
                "ymax": ymax,
                "layer": layer,
                "plot_type": "axhspan",
                "kwargs": kwargs,
            },
            layer,
        )

    def arrow(self, x, y, dx, dy, layer=0, **kwargs):
        """Add an arrow to the axes."""
        self._add(
            {
                "x": x,
                "y": y,
                "dx": dx,
                "dy": dy,
                "layer": layer,
                "plot_type": "arrow",
                "kwargs": kwargs,
            },
            layer,
        )

    def axline(self, xy1, xy2=None, slope=None, layer=0, **kwargs):
        """Add an infinitely extending line through one or two points."""
        self._add(
            {
                "xy1": xy1,
                "xy2": xy2,
                "slope": slope,
                "layer": layer,
                "plot_type": "axline",
                "kwargs": kwargs,
            },
            layer,
        )

    def annotate(self, text, xy, xytext=None, layer=0, **kwargs):
        """
        Add a text annotation, optionally with an arrow.

        Parameters:
        text (str): Annotation text.
        xy (tuple): (x, y) position to annotate.
        xytext (tuple, optional): (x, y) position for the text.
        **kwargs: Forwarded to ax.annotate (e.g., arrowprops, fontsize, color).
        """
        ld = {
            "text": text,
            "xy": xy,
            "xytext": xytext,
            "layer": layer,
            "plot_type": "annotate",
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def text(self, x, y, s, layer=0, **kwargs):
        """
        Add a text label at position (x, y).

        Parameters:
        x, y (float): Position.
        s (str): Text string.
        **kwargs: Forwarded to ax.text (e.g., fontsize, color, ha, va).
        """
        ld = {
            "x": x,
            "y": y,
            "s": s,
            "layer": layer,
            "plot_type": "text",
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def add_imshow(self, data, layer=0, **kwargs):
        ld = {
            "data": np.array(data),
            "layer": layer,
            "plot_type": "imshow",
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def add_image(self, data, layer=0, **kwargs):
        """Matplotlib-style alias for ``imshow``."""
        self.add_imshow(data, layer=layer, **kwargs)

    def add_patch(self, patch, layer=0, **kwargs):
        ld = {
            "patch": patch,
            "layer": layer,
            "plot_type": "patch",
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def add_colorbar(self, label="", layer=0, **kwargs):
        cb = {
            "label": label,
            "layer": layer,
            "plot_type": "colorbar",
            "kwargs": kwargs,
        }
        self._add(cb, layer)

    @property
    def layers(self):
        layers = []
        for layer_name, layer_lines in self.layered_line_data.items():
            layers.append(layer_name)
        return layers

    def plot_matplotlib(
        self,
        ax,
        layers=None,
        verbose: bool = False,
    ):
        """
        Plot all lines on the provided axis.

        Parameters:
        ax (matplotlib.axes.Axes): Axis on which to plot the lines.
        """
        im = None
        contour_sets = []
        for layer_name, layer_lines in self.layered_line_data.items():
            if layers and layer_name not in layers:
                continue
            for line in layer_lines:
                artists_before = (
                    set(map(id, ax.get_children()))
                    if line.get("meta") is not None
                    else None
                )
                if line["plot_type"] == "plot":
                    ax.plot(
                        (line["x"] + self._xshift) * self._xscale,
                        (line["y"] + self._yshift) * self._yscale,
                        **line["kwargs"],
                    )
                elif line["plot_type"] == "scatter":
                    # "colorbar" mirrors the same Plotly-only flag used by
                    # contour/pcolormesh/etc.; Matplotlib shows a colorbar via
                    # a separate colorbar() call, not a scatter() kwarg.
                    scatter_kwargs = {
                        k: v for k, v in line["kwargs"].items() if k != "colorbar"
                    }
                    ax.scatter(
                        (line["x"] + self._xshift) * self._xscale,
                        (line["y"] + self._yshift) * self._yscale,
                        **scatter_kwargs,
                    )
                elif line["plot_type"] == "bar":
                    ax.bar(
                        (line["x"] + self._xshift) * self._xscale,
                        line["height"] * self._yscale,
                        **line["kwargs"],
                    )
                elif line["plot_type"] == "barh":
                    ax.barh(
                        line["y"],
                        line["width"] * self._xscale,
                        **line["kwargs"],
                    )
                elif line["plot_type"] == "hist":
                    ax.hist(line["x"], bins=line["bins"], **line["kwargs"])
                elif line["plot_type"] == "step":
                    ax.step(
                        (line["x"] + self._xshift) * self._xscale,
                        (line["y"] + self._yshift) * self._yscale,
                        **line["kwargs"],
                    )
                elif line["plot_type"] == "stairs":
                    ax.stairs(
                        line["values"],
                        edges=line["edges"],
                        baseline=line["baseline"],
                        **line["kwargs"],
                    )
                elif line["plot_type"] == "broken_barh":
                    ax.broken_barh(line["xranges"], line["yrange"], **line["kwargs"])
                elif line["plot_type"] == "pie":
                    ax.pie(line["x"], **line["kwargs"])
                elif line["plot_type"] == "stem":
                    ax.stem(line["x"], line["y"], **line["kwargs"])
                elif line["plot_type"] == "stackplot":
                    ax.stackplot(line["x"], *line["ys"], **line["kwargs"])
                elif line["plot_type"] == "boxplot":
                    ax.boxplot(line["x"], **line["kwargs"])
                elif line["plot_type"] == "violinplot":
                    ax.violinplot(line["dataset"], **line["kwargs"])
                elif line["plot_type"] == "eventplot":
                    ax.eventplot(line["positions"], **line["kwargs"])
                elif line["plot_type"] == "contour":
                    contour_sets.append(
                        ax.contour(line["x"], line["y"], line["z"], **line["kwargs"])
                    )
                elif line["plot_type"] == "contourf":
                    contour_sets.append(
                        ax.contourf(line["x"], line["y"], line["z"], **line["kwargs"])
                    )
                elif line["plot_type"] == "pcolormesh":
                    ax.pcolormesh(line["x"], line["y"], line["z"], **line["kwargs"])
                elif line["plot_type"] == "hexbin":
                    ax.hexbin(line["x"], line["y"], **line["kwargs"])
                elif line["plot_type"] == "matshow":
                    ax.matshow(line["data"], **line["kwargs"])
                elif line["plot_type"] == "quiver":
                    ax.quiver(
                        line["x"], line["y"], line["u"], line["v"], **line["kwargs"]
                    )
                elif line["plot_type"] == "triplot":
                    import matplotlib.tri as mtri

                    triangulation = mtri.Triangulation(
                        line["x"], line["y"], triangles=line["triangles"]
                    )
                    ax.triplot(triangulation, **line["kwargs"])
                elif line["plot_type"] == "tripcolor":
                    import matplotlib.tri as mtri

                    triangulation = mtri.Triangulation(
                        line["x"], line["y"], triangles=line["triangles"]
                    )
                    ax.tripcolor(triangulation, line["c"], **line["kwargs"])
                elif line["plot_type"] == "tricontour":
                    import matplotlib.tri as mtri

                    triangulation = mtri.Triangulation(
                        line["x"], line["y"], triangles=line["triangles"]
                    )
                    ax.tricontour(triangulation, line["z"], **line["kwargs"])
                elif line["plot_type"] == "tricontourf":
                    import matplotlib.tri as mtri

                    triangulation = mtri.Triangulation(
                        line["x"], line["y"], triangles=line["triangles"]
                    )
                    ax.tricontourf(triangulation, line["z"], **line["kwargs"])
                elif line["plot_type"] == "streamplot":
                    ax.streamplot(
                        line["x"], line["y"], line["u"], line["v"], **line["kwargs"]
                    )
                elif line["plot_type"] == "pcolor":
                    ax.pcolor(line["x"], line["y"], line["z"], **line["kwargs"])
                elif line["plot_type"] == "pcolorfast":
                    ax.pcolorfast(line["x"], line["y"], line["z"], **line["kwargs"])
                elif line["plot_type"] == "spy":
                    ax.spy(line["matrix"], **line["kwargs"])
                elif line["plot_type"] == "table":
                    ax.table(cellText=line["cellText"], **line["kwargs"])
                elif line["plot_type"] == "gantt":
                    tasks = line["tasks"]
                    start_times = (line["start_times"] + self._xshift) * self._xscale
                    durations = line["durations"] * self._xscale
                    y_positions = np.arange(len(tasks))
                    ax.barh(y_positions, durations, left=start_times, **line["kwargs"])
                    ax.set_yticks(y_positions)
                    ax.set_yticklabels(tasks)
                elif line["plot_type"] == "flame_chart":
                    labels = line["labels"]
                    parents = line["parents"]
                    values = line["values"] * self._xscale
                    start_times = line["start_times"]

                    # Calculate depth levels and positions
                    n = len(labels)
                    depths = np.zeros(n, dtype=int)
                    if start_times is None:
                        start_times = np.zeros(n)
                    else:
                        start_times = (start_times + self._xshift) * self._xscale

                    # Calculate depths based on parent relationships
                    for i in range(n):
                        if parents[i] is None:
                            depths[i] = 0
                        else:
                            parent_idx = (
                                parents[i]
                                if isinstance(parents[i], int)
                                else list(labels).index(parents[i])
                            )
                            depths[i] = depths[parent_idx] + 1

                    # Draw rectangles for each frame
                    import matplotlib.patches as mpatches

                    max_depth = depths.max() + 1
                    frame_colors = _flame_frame_colors(
                        line["kwargs"], depths, n, css=False
                    )

                    for i in range(n):
                        color = frame_colors[i]
                        rect = mpatches.Rectangle(
                            (start_times[i], depths[i]),
                            values[i],
                            0.9,
                            facecolor=color,
                            edgecolor=line["kwargs"].get("edgecolor", "black"),
                            linewidth=0.5,
                        )
                        ax.add_patch(rect)

                        # Add label if rectangle is wide enough
                        if values[i] > 0.1 * (start_times.max() + values.max()):
                            ax.text(
                                start_times[i] + values[i] / 2,
                                depths[i] + 0.45,
                                labels[i],
                                ha="center",
                                va="center",
                                fontsize=8,
                                color=(
                                    "white" if depths[i] / max_depth > 0.5 else "black"
                                ),
                            )

                    ax.set_ylim(-0.5, max_depth)
                    ax.set_ylabel("Stack Depth")
                elif line["plot_type"] == "fill_between":
                    ax.fill_between(
                        (line["x"] + self._xshift) * self._xscale,
                        (
                            line["y1"]
                            if np.isscalar(line["y1"])
                            else (line["y1"] + self._yshift) * self._yscale
                        ),
                        (
                            line["y2"]
                            if np.isscalar(line["y2"])
                            else (line["y2"] + self._yshift) * self._yscale
                        ),
                        **line["kwargs"],
                    )
                elif line["plot_type"] == "fill_betweenx":
                    y = (line["y"] + self._yshift) * self._yscale
                    x1 = line["x1"]
                    x2 = line["x2"]
                    if np.isscalar(x1):
                        x1 = np.full_like(y, x1, dtype=float)
                    if np.isscalar(x2):
                        x2 = np.full_like(y, x2, dtype=float)
                    ax.fill_betweenx(
                        y,
                        (np.asarray(x1) + self._xshift) * self._xscale,
                        (np.asarray(x2) + self._xshift) * self._xscale,
                        **line["kwargs"],
                    )
                elif line["plot_type"] == "fill":
                    ax.fill(*line["args"], **line["kwargs"])
                elif line["plot_type"] == "errorbar":
                    ax.errorbar(
                        (line["x"] + self._xshift) * self._xscale,
                        (line["y"] + self._yshift) * self._yscale,
                        yerr=line["yerr"],
                        xerr=line["xerr"],
                        **line["kwargs"],
                    )
                elif line["plot_type"] == "hlines":
                    ax.hlines(line["y"], line["xmin"], line["xmax"], **line["kwargs"])
                elif line["plot_type"] == "vlines":
                    ax.vlines(line["x"], line["ymin"], line["ymax"], **line["kwargs"])
                elif line["plot_type"] == "axvspan":
                    ax.axvspan(line["xmin"], line["xmax"], **line["kwargs"])
                elif line["plot_type"] == "axhspan":
                    ax.axhspan(line["ymin"], line["ymax"], **line["kwargs"])
                elif line["plot_type"] == "arrow":
                    ax.arrow(
                        line["x"],
                        line["y"],
                        line["dx"],
                        line["dy"],
                        **line["kwargs"],
                    )
                elif line["plot_type"] == "axline":
                    ax.axline(
                        line["xy1"],
                        xy2=line["xy2"],
                        slope=line["slope"],
                        **line["kwargs"],
                    )
                elif line["plot_type"] == "annotate":
                    ann_kwargs = dict(line["kwargs"])
                    if line["xytext"] is not None:
                        ann_kwargs["xytext"] = line["xytext"]
                    ax.annotate(line["text"], xy=line["xy"], **ann_kwargs)
                elif line["plot_type"] == "text":
                    ax.text(line["x"], line["y"], line["s"], **line["kwargs"])
                elif line["plot_type"] == "axvline":
                    ax.axvline(x=line["x"], **line["kwargs"])
                elif line["plot_type"] == "imshow":
                    im = ax.imshow(
                        line["data"],
                        **line["kwargs"],
                    )
                elif line["plot_type"] == "patch":
                    ax.add_patch(
                        line["patch"],
                        **line["kwargs"],
                    )
                elif line["plot_type"] == "colorbar":
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax, label="Potential (V)")

                if line.get("meta") is not None:
                    # Mirror the Plotly ``meta=`` tag onto the Matplotlib
                    # artists so callers can find them again by identity
                    # rather than by drawing order.
                    self._tag_matplotlib_artists(ax, artists_before, line["meta"])

        if self._title:
            ax.set_title(self._title, **self._title_kwargs)
        if self._xlabel:
            ax.set_xlabel(self._xlabel, **self._xlabel_kwargs)
        if self._ylabel:
            ax.set_ylabel(self._ylabel, **self._ylabel_kwargs)
        if self._legend and len(self.line_data) > 0:
            ax.legend(**self._legend_kwargs)
        if self._grid:
            ax.grid()
        if self._axis_settings:
            axis_settings = dict(self._axis_settings)
            axis_args = axis_settings.pop("args", ())
            ax.axis(*axis_args, **axis_settings)
        if self.xmin is not None:
            ax.axis(xmin=self.xmin)
        if self.xmax is not None:
            ax.axis(xmax=self.xmax)
        if self.ymin is not None:
            ax.axis(ymin=self.ymin)
        if self.ymax is not None:
            ax.axis(ymax=self.ymax)
        if self._xaxis_scale is not None:
            ax.set_xscale(self._xaxis_scale)
        if self._yaxis_scale is not None:
            ax.set_yscale(self._yaxis_scale)
        if self._xticks is not None:
            ax.set_xticks(
                self._xticks,
                labels=self._xticklabels,
                **self._xtick_kwargs,
            )
        if self._yticks is not None:
            ax.set_yticks(
                self._yticks,
                labels=self._yticklabels,
                **self._ytick_kwargs,
            )
        if self._xticklabels is not None and self._xticks is None:
            ax.set_xticklabels(self._xticklabels, **self._xticklabel_kwargs)
        if self._yticklabels is not None and self._yticks is None:
            ax.set_yticklabels(self._yticklabels, **self._yticklabel_kwargs)
        if self._tick_params:
            tick_params = dict(self._tick_params)
            # ``rotation`` is the neutral spelling accepted by every backend;
            # Matplotlib calls it ``labelrotation``.
            if "rotation" in tick_params:
                tick_params.setdefault("labelrotation", tick_params.pop("rotation"))
                tick_params.pop("rotation", None)
            ax.tick_params(**tick_params)
        if self._aspect is not None:
            ax.set_aspect(self._aspect)
        if self._adjustable is not None:
            ax.set_adjustable(self._adjustable)
        if self._anchor is not None:
            ax.set_anchor(self._anchor)
        if self._box_aspect is not None:
            ax.set_box_aspect(self._box_aspect)
        if self._axisbelow is not None:
            ax.set_axisbelow(self._axisbelow)
        if self._facecolor is not None:
            ax.set_facecolor(self._facecolor)
        if self._frame_on is not None:
            ax.set_frame_on(self._frame_on)
        if self._visible is not None:
            ax.set_visible(self._visible)
        if self._alpha is not None:
            ax.set_alpha(self._alpha)
        if self._zorder is not None:
            ax.set_zorder(self._zorder)
        if self._rasterized is not None:
            ax.set_rasterized(self._rasterized)
        if self._autoscale_on is not None:
            ax.set_autoscale_on(self._autoscale_on)
        if self._autoscalex_on is not None:
            ax.set_autoscalex_on(self._autoscalex_on)
        if self._autoscaley_on is not None:
            ax.set_autoscaley_on(self._autoscaley_on)
        if self._xmargin is not None:
            ax.set_xmargin(self._xmargin)
        if self._ymargin is not None:
            ax.set_ymargin(self._ymargin)
        if self._margins:
            margin_settings = dict(self._margins)
            margin_args = margin_settings.pop("args", ())
            ax.margins(*margin_args, **margin_settings)
        if self._invert_xaxis:
            ax.invert_xaxis()
        if self._invert_yaxis:
            ax.invert_yaxis()
        if self._minorticks is True:
            ax.minorticks_on()
        elif self._minorticks is False:
            ax.minorticks_off()
        if self._locator_params:
            ax.locator_params(**self._locator_params)
        if self._ticklabel_format:
            ax.ticklabel_format(**self._ticklabel_format)
        if self._axis_off:
            ax.set_axis_off()
        if self._relim_settings is not None:
            ax.relim(**self._relim_settings)
        if self._autoscale_settings is not None:
            ax.autoscale(**self._autoscale_settings)
        if self._autoscale_view_settings is not None:
            ax.autoscale_view(**self._autoscale_view_settings)
        if self._secondary_xaxis_settings is not None:
            settings = self._secondary_xaxis_settings
            secondary_kwargs = dict(settings["kwargs"])
            label = secondary_kwargs.pop("label", None)
            secondary = ax.secondary_xaxis(
                settings["location"],
                functions=settings["functions"],
                **secondary_kwargs,
            )
            if label:
                secondary.set_xlabel(label)
        if self._secondary_yaxis_settings is not None:
            settings = self._secondary_yaxis_settings
            secondary_kwargs = dict(settings["kwargs"])
            label = secondary_kwargs.pop("label", None)
            secondary = ax.secondary_yaxis(
                settings["location"],
                functions=settings["functions"],
                **secondary_kwargs,
            )
            if label:
                secondary.set_ylabel(label)
        if self._bar_label_kwargs:
            for container in ax.containers:
                ax.bar_label(container, **self._bar_label_kwargs)
        if self._clabel_kwargs:
            for contour_set in contour_sets:
                ax.clabel(contour_set, **self._clabel_kwargs)
        if self._rasterization_zorder is not None:
            ax.set_rasterization_zorder(self._rasterization_zorder)

    @staticmethod
    def _tag_matplotlib_artists(ax, artists_before, meta):
        """Set ``gid`` on every artist added since ``artists_before``."""
        if artists_before is None:
            return
        for artist in ax.get_children():
            if id(artist) in artists_before:
                continue
            try:
                artist.set_gid(str(meta))
            except AttributeError:
                continue

    def plot_tikzfigure(self, layers=None, verbose: bool = False) -> TikzFigure:

        tikz_figure = TikzFigure()
        for layer_name, layer_lines in self.layered_line_data.items():
            if layers and layer_name not in layers:
                continue
            for line in layer_lines:
                plot_type = line["plot_type"]
                if plot_type not in _TIKZ_SUPPORTED_PLOT_TYPES:
                    raise NotImplementedError(
                        f"{plot_type} is not supported by the tikzfigure backend"
                    )
                if plot_type == "plot":
                    x = (line["x"] + self._xshift) * self._xscale
                    y = (line["y"] + self._yshift) * self._yscale

                    nodes = [[xi, yi] for xi, yi in zip(x, y)]
                    tikz_figure.draw(
                        nodes=nodes,
                        **_tikz_style_kwargs(line["kwargs"]),
                    )
                elif plot_type == "scatter":
                    x = (line["x"] + self._xshift) * self._xscale
                    y = (line["y"] + self._yshift) * self._yscale
                    style = _tikz_style_kwargs(line["kwargs"])
                    style.setdefault("mark", "*")
                    style["line_width"] = 0
                    tikz_figure.draw(
                        nodes=[[xi, yi] for xi, yi in zip(x, y)],
                        **style,
                    )
                elif plot_type in {"bar", "barh"}:
                    kwargs = line["kwargs"]
                    style = _tikz_style_kwargs(kwargs)
                    style["fill"] = kwargs.get("color", "blue")
                    style["fill_opacity"] = kwargs.get("alpha", 1.0)
                    style["line_width"] = kwargs.get("linewidth", 0)
                    if plot_type == "bar":
                        width = kwargs.get("width", 0.8)
                        for x, height in zip(line["x"], line["height"]):
                            x = (x + self._xshift) * self._xscale
                            height = height * self._yscale
                            tikz_figure.draw(
                                nodes=[
                                    [x - width / 2, 0],
                                    [x + width / 2, 0],
                                    [x + width / 2, height],
                                    [x - width / 2, height],
                                ],
                                cycle=True,
                                **style,
                            )
                    else:
                        height = kwargs.get("height", 0.8)
                        for y, width in zip(line["y"], line["width"]):
                            y = (y + self._yshift) * self._yscale
                            width = width * self._xscale
                            tikz_figure.draw(
                                nodes=[
                                    [0, y - height / 2],
                                    [width, y - height / 2],
                                    [width, y + height / 2],
                                    [0, y + height / 2],
                                ],
                                cycle=True,
                                **style,
                            )
                elif plot_type == "fill_between":
                    x = (line["x"] + self._xshift) * self._xscale
                    y1 = np.asarray(line["y1"])
                    y2 = np.broadcast_to(line["y2"], y1.shape)
                    nodes = [[xi, yi] for xi, yi in zip(x, y1)]
                    nodes.extend([[xi, yi] for xi, yi in zip(x[::-1], y2[::-1])])
                    kwargs = line["kwargs"]
                    style = _tikz_style_kwargs(kwargs)
                    style["fill"] = kwargs.get("color", "blue")
                    style["fill_opacity"] = kwargs.get("alpha", 0.25)
                    tikz_figure.draw(nodes=nodes, cycle=True, **style)
                elif plot_type == "errorbar":
                    x = (line["x"] + self._xshift) * self._xscale
                    y = (line["y"] + self._yshift) * self._yscale
                    style = _tikz_style_kwargs(line["kwargs"])
                    tikz_figure.draw(nodes=[[xi, yi] for xi, yi in zip(x, y)], **style)
                    y_bounds = _tikz_error_bounds(line["yerr"], y)
                    if y_bounds is not None:
                        lower, upper = y_bounds
                        for xi, low, high in zip(x, y - lower, y + upper):
                            tikz_figure.draw(nodes=[[xi, low], [xi, high]], **style)
                    x_bounds = _tikz_error_bounds(line["xerr"], x)
                    if x_bounds is not None:
                        lower, upper = x_bounds
                        for yi, low, high in zip(y, x - lower, x + upper):
                            tikz_figure.draw(nodes=[[low, yi], [high, yi]], **style)
                elif plot_type in {"step", "stairs"}:
                    kwargs = line["kwargs"]
                    if plot_type == "step":
                        x = line["x"]
                        y = line["y"]
                        where = kwargs.get("where", "pre")
                    else:
                        values = line["values"]
                        edges = line["edges"]
                        if edges is None:
                            edges = np.arange(len(values) + 1)
                        x = edges
                        y = np.r_[values, values[-1]]
                        where = "post"
                    x, y = _tikz_step_coordinates(x, y, where=where)
                    x = (x + self._xshift) * self._xscale
                    y = (y + self._yshift) * self._yscale
                    tikz_figure.draw(
                        nodes=[[xi, yi] for xi, yi in zip(x, y)],
                        **_tikz_style_kwargs(kwargs),
                    )
                elif plot_type == "stem":
                    x = (line["x"] + self._xshift) * self._xscale
                    y = (line["y"] + self._yshift) * self._yscale
                    kwargs = line["kwargs"]
                    style = _tikz_style_kwargs(kwargs)
                    marker_style = dict(style)
                    marker_style.update(mark=kwargs.get("marker", "*"), line_width=0)
                    tikz_figure.draw(
                        nodes=[[xi, yi] for xi, yi in zip(x, y)], **marker_style
                    )
                    for xi, yi in zip(x, y):
                        tikz_figure.draw(nodes=[[xi, 0], [xi, yi]], **style)
                elif plot_type in {"hlines", "vlines"}:
                    kwargs = _tikz_style_kwargs(line["kwargs"])
                    if plot_type == "hlines":
                        for yi, left, right in zip(
                            np.atleast_1d(line["y"]),
                            np.atleast_1d(line["xmin"]),
                            np.atleast_1d(line["xmax"]),
                        ):
                            tikz_figure.draw(nodes=[[left, yi], [right, yi]], **kwargs)
                    else:
                        for xi, bottom, top in zip(
                            np.atleast_1d(line["x"]),
                            np.atleast_1d(line["ymin"]),
                            np.atleast_1d(line["ymax"]),
                        ):
                            tikz_figure.draw(nodes=[[xi, bottom], [xi, top]], **kwargs)
                elif plot_type in {"axvspan", "axhspan"}:
                    kwargs = line["kwargs"]
                    style = _tikz_style_kwargs(kwargs)
                    style["fill"] = kwargs.get("color", "blue")
                    style["fill_opacity"] = kwargs.get("alpha", 0.2)
                    if plot_type == "axvspan":
                        ymin, ymax = self._ymin or 0, self._ymax or 1
                        nodes = [
                            [line["xmin"], ymin],
                            [line["xmax"], ymin],
                            [line["xmax"], ymax],
                            [line["xmin"], ymax],
                        ]
                    else:
                        xmin, xmax = self._xmin or 0, self._xmax or 1
                        nodes = [
                            [xmin, line["ymin"]],
                            [xmax, line["ymin"]],
                            [xmax, line["ymax"]],
                            [xmin, line["ymax"]],
                        ]
                    tikz_figure.draw(nodes=nodes, cycle=True, **style)
                elif plot_type == "fill":
                    if len(line["args"]) < 2:
                        raise ValueError("tikzfigure fill requires x and y coordinates")
                    x, y = line["args"][:2]
                    kwargs = line["kwargs"]
                    style = _tikz_style_kwargs(kwargs)
                    style["fill"] = kwargs.get("color", "blue")
                    style["fill_opacity"] = kwargs.get("alpha", 0.25)
                    tikz_figure.draw(
                        nodes=[[xi, yi] for xi, yi in zip(x, y)],
                        cycle=True,
                        **style,
                    )
                elif line["plot_type"] == "gantt":
                    tasks = line["tasks"]
                    start_times = (line["start_times"] + self._xshift) * self._xscale
                    durations = line["durations"] * self._xscale
                    y_positions = np.arange(len(tasks))

                    # Draw horizontal bars for each task
                    for i, (task, start, duration) in enumerate(
                        zip(tasks, start_times, durations)
                    ):
                        # Create rectangle nodes for the bar
                        x_start = start
                        x_end = start + duration
                        y_pos = y_positions[i]
                        bar_height = 0.8  # Bar thickness

                        # Draw rectangle as a path
                        rect_nodes = [
                            [x_start, y_pos - bar_height / 2],
                            [x_end, y_pos - bar_height / 2],
                            [x_end, y_pos + bar_height / 2],
                            [x_start, y_pos + bar_height / 2],
                        ]
                        tikz_figure.draw(
                            nodes=rect_nodes,
                            cycle=True,
                            fill=line["kwargs"].get("color", "blue"),
                            **line["kwargs"],
                        )
                elif line["plot_type"] == "flame_chart":
                    labels = line["labels"]
                    parents = line["parents"]
                    values = line["values"] * self._xscale
                    start_times = line["start_times"]

                    # Calculate depths
                    n = len(labels)
                    depths = np.zeros(n, dtype=int)
                    if start_times is None:
                        start_times = np.zeros(n)
                    else:
                        start_times = (start_times + self._xshift) * self._xscale

                    for i in range(n):
                        if parents[i] is None:
                            depths[i] = 0
                        else:
                            parent_idx = (
                                parents[i]
                                if isinstance(parents[i], int)
                                else list(labels).index(parents[i])
                            )
                            depths[i] = depths[parent_idx] + 1

                    # Draw rectangles for each frame
                    bar_height = 0.8
                    explicit_colors = line["kwargs"].get("colors")
                    if isinstance(explicit_colors, str) or not hasattr(
                        explicit_colors, "__len__"
                    ):
                        explicit_colors = (
                            None if explicit_colors is None else [explicit_colors]
                        )
                    colors = ["red", "blue", "green", "orange", "purple", "cyan"]

                    for i in range(n):
                        x_start = start_times[i]
                        x_end = start_times[i] + values[i]
                        y_pos = depths[i]
                        if explicit_colors:
                            color = explicit_colors[i % len(explicit_colors)]
                        else:
                            color = colors[depths[i] % len(colors)]

                        rect_nodes = [
                            [x_start, y_pos - bar_height / 2],
                            [x_end, y_pos - bar_height / 2],
                            [x_end, y_pos + bar_height / 2],
                            [x_start, y_pos + bar_height / 2],
                        ]
                        tikz_figure.draw(
                            nodes=rect_nodes,
                            cycle=True,
                            fill=color,
                            **{
                                k: v
                                for k, v in line["kwargs"].items()
                                if k not in ("colormap", "colors")
                            },
                        )
        if verbose:
            print("Generated TikZ figure:")
            print(tikz_figure.generate_tikz())
        return tikz_figure

    def plot_plotly(self, layers=None, allow_unsupported=False):
        """
        Plot all lines using Plotly.

        Returns a tuple of (traces, shapes, annotations) where:
        - traces are plotly graph objects to add with fig.add_trace()
        - shapes are layout shape dicts to add with fig.add_shape()
        - annotations are layout annotation dicts to add with fig.add_annotation()
        """
        linestyle_map = {
            "solid": "solid",
            "dashed": "dash",
            "dotted": "dot",
            "dashdot": "dashdot",
        }

        marker_map = {
            "o": "circle",
            ".": "circle",
            "s": "square",
            "^": "triangle-up",
            "v": "triangle-down",
            "<": "triangle-left",
            ">": "triangle-right",
            "x": "x",
            "+": "cross",
            "*": "star",
            "D": "diamond",
        }

        traces: list[go.BaseTraceType] = []
        shapes: list[dict] = []
        annotations: list[dict] = []
        last_heatmap_idx: int | None = None
        # Set to "overlay" while building traces whenever a bar-like trace
        # positions itself with an explicit base (stacked bars, Gantt rows,
        # flame frames); Plotly's default "group" barmode would re-offset
        # those bars and undo the positioning the caller asked for.
        self._plotly_barmode_hint = None
        # Plotly shapes (unlike traces) don't participate in axis autorange,
        # so patches would otherwise be clipped or invisible unless the caller
        # sets explicit axis limits. Track each patch's bounding box here and
        # add one invisible marker trace at the end so autorange sees them.
        patch_bounds_x: list[float] = []
        patch_bounds_y: list[float] = []
        # Indices of shapes that already got their own hover overlay trace and
        # must not receive a second one from the generic ``hover=`` handling.
        hover_handled_shapes: set[int] = set()

        # These primitives have no faithful 2-D Plotly equivalent in the
        # current backend.  Keep the default strict so a mixed plot cannot
        # silently lose data, while allowing callers to deliberately render
        # the Plotly-compatible portions of a canvas.
        unsupported_plot_types = set()

        def tx(values):
            return self._transform_x(values)

        def ty(values):
            return self._transform_y(values)

        def txs(value):
            return self._transform_scalar_x(value)

        def tys(value):
            return self._transform_scalar_y(value)

        def plotly_color(value):
            if value is None:
                return None
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value).astype(float).reshape(-1)
                if arr.size in (3, 4):
                    rgb = (arr[:3] * 255.0) if np.all(arr[:3] <= 1.0) else arr[:3]
                    r, g, b = [int(round(float(x))) for x in rgb]
                    if arr.size == 4:
                        a = float(arr[3])
                        if a > 1.0:
                            a = a / 255.0
                        return f"rgba({r},{g},{b},{a})"
                    return f"rgb({r},{g},{b})"
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered.startswith(("#", "rgb", "hsl", "hwb", "lab", "lch", "ok")):
                    return value
                # Matplotlib color spellings Plotly does not accept, such as
                # "tab:blue", "C0" or the single-letter shorthands. CSS named
                # colors are passed through unchanged.
                import matplotlib.colors as mcolors

                if lowered in mcolors.CSS4_COLORS:
                    return value
                try:
                    return mcolors.to_hex(value, keep_alpha=False)
                except ValueError:
                    return value
            return value

        def bar_marker(kwargs):
            """Build a go.Bar marker honoring color, edgecolor and linewidth."""
            marker = dict(color=plotly_color(kwargs.get("color", None)))
            edgecolor = kwargs.get("edgecolor", kwargs.get("edgecolors", None))
            linewidth = kwargs.get("linewidth", kwargs.get("linewidths", None))
            if edgecolor is not None or linewidth is not None:
                marker["line"] = dict(
                    color=plotly_color(edgecolor) if edgecolor is not None else None,
                    width=(
                        linewidth
                        if linewidth is not None
                        else (1 if edgecolor is not None else None)
                    ),
                )
            return marker

        for line in self._iter_layer_lines(layers=layers):
            plot_type = line["plot_type"]
            if plot_type in unsupported_plot_types:
                if allow_unsupported:
                    continue
                raise NotImplementedError(
                    f"{plot_type} is currently supported only by the matplotlib "
                    "backend; pass allow_unsupported=True to skip it for Plotly"
                )
            trace_start = len(traces)
            shape_start = len(shapes)
            if plot_type == "plot":
                kwargs = line["kwargs"]
                marker = kwargs.get("marker")
                mode = "lines+markers" if marker is not None else "lines"
                trace = go.Scatter(
                    x=tx(line["x"]),
                    y=ty(line["y"]),
                    mode=mode,
                    name=kwargs.get("label", ""),
                    showlegend=bool(kwargs.get("label")) and bool(self._legend),
                    line=dict(
                        color=plotly_color(kwargs.get("color", None)),
                        dash=linestyle_map.get(
                            kwargs.get("linestyle", "solid"),
                            "solid",
                        ),
                        width=kwargs.get("linewidth", None),
                    ),
                    marker=(
                        dict(
                            color=plotly_color(kwargs.get("color", None)),
                            symbol=marker_map.get(marker, "circle"),
                            size=kwargs.get("markersize", None),
                        )
                        if marker is not None
                        else None
                    ),
                )
                traces.append(trace)
            elif plot_type == "scatter":
                kwargs = line["kwargs"]
                marker = kwargs.get("marker", "circle")
                # ``c`` is Matplotlib's value-per-point coloring; a plain
                # ``color`` still wins if both are given, matching ax.scatter.
                c_values = kwargs.get("c")
                if kwargs.get("color") is not None or c_values is None:
                    marker_color = plotly_color(kwargs.get("color", None))
                    colorscale = None
                else:
                    if isinstance(c_values, str) or (
                        np.ndim(c_values) > 0
                        and not np.issubdtype(np.asarray(c_values).dtype, np.number)
                    ):
                        marker_color = plotly_color(c_values)
                        colorscale = None
                    else:
                        marker_color = np.asarray(c_values, dtype=float).tolist()
                        colorscale = _colormap_to_plotly_colorscale(
                            kwargs.get("cmap", "viridis")
                        )
                marker_dict = dict(
                    color=marker_color,
                    colorscale=colorscale,
                    cmin=kwargs.get("vmin", None),
                    cmax=kwargs.get("vmax", None),
                    showscale=(colorscale is not None)
                    and bool(kwargs.get("colorbar", False)),
                    symbol=marker_map.get(marker, marker),
                    size=kwargs.get("s", None),
                    opacity=kwargs.get("alpha", None),
                )
                edgecolor = kwargs.get("edgecolors", kwargs.get("edgecolor"))
                linewidth = kwargs.get("linewidths", kwargs.get("linewidth"))
                if edgecolor is not None or linewidth is not None:
                    marker_dict["line"] = dict(
                        color=(
                            plotly_color(edgecolor) if edgecolor is not None else None
                        ),
                        width=linewidth if linewidth is not None else 1,
                    )
                trace = go.Scatter(
                    x=tx(line["x"]),
                    y=ty(line["y"]),
                    mode="markers",
                    name=kwargs.get("label", ""),
                    showlegend=bool(kwargs.get("label")) and bool(self._legend),
                    marker=marker_dict,
                )
                traces.append(trace)
            elif plot_type == "bar":
                kwargs = line["kwargs"]
                base = kwargs.get("bottom")
                if base is not None:
                    base = np.asarray(base, dtype=float) * self._yscale
                    self._plotly_barmode_hint = "overlay"
                trace = go.Bar(
                    x=tx(line["x"]),
                    y=np.asarray(line["height"]) * self._yscale,
                    base=base,
                    width=kwargs.get("width", None),
                    orientation="v",
                    name=kwargs.get("label", ""),
                    showlegend=bool(kwargs.get("label")) and bool(self._legend),
                    marker=bar_marker(kwargs),
                    opacity=kwargs.get("alpha", None),
                    offsetgroup=kwargs.get("offsetgroup", None),
                )
                traces.append(trace)
            elif plot_type == "barh":
                kwargs = line["kwargs"]
                base = kwargs.get("left")
                if base is not None:
                    base = np.asarray(base, dtype=float) * self._xscale
                    self._plotly_barmode_hint = "overlay"
                traces.append(
                    go.Bar(
                        x=np.asarray(line["width"]) * self._xscale,
                        y=ty(line["y"]),
                        base=base,
                        width=kwargs.get("height", None),
                        orientation="h",
                        name=kwargs.get("label", ""),
                        showlegend=bool(kwargs.get("label")) and bool(self._legend),
                        marker=bar_marker(kwargs),
                        opacity=kwargs.get("alpha", None),
                        offsetgroup=kwargs.get("offsetgroup", None),
                    )
                )
            elif plot_type == "hist":
                kwargs = line["kwargs"]
                bins = line["bins"]
                xbins = None
                nbinsx = None
                if np.isscalar(bins):
                    nbinsx = bins
                else:
                    edges = np.asarray(bins, dtype=float)
                    if edges.size >= 2:
                        # Plotly bins by (start, end, size) rather than by
                        # explicit edges; only an evenly spaced ``bins=``
                        # array (the common case) can be represented exactly.
                        spacing = np.diff(edges)
                        if np.allclose(spacing, spacing[0]):
                            xbins = dict(
                                start=float(edges[0]),
                                end=float(edges[-1]),
                                size=float(spacing[0]) * self._xscale,
                            )
                traces.append(
                    go.Histogram(
                        x=tx(line["x"]),
                        nbinsx=nbinsx,
                        xbins=xbins,
                        histnorm=(
                            "probability density" if kwargs.get("density") else None
                        ),
                        cumulative=(
                            dict(enabled=True) if kwargs.get("cumulative") else None
                        ),
                        name=kwargs.get("label", ""),
                        showlegend=bool(kwargs.get("label")) and bool(self._legend),
                        marker=dict(
                            color=plotly_color(kwargs.get("color", None)),
                            line=dict(
                                color=plotly_color(kwargs.get("edgecolor", None)),
                                width=kwargs.get("linewidth", None),
                            ),
                        ),
                        opacity=kwargs.get("alpha", None),
                    )
                )
            elif plot_type == "step":
                kwargs = line["kwargs"]
                traces.append(
                    go.Scatter(
                        x=tx(line["x"]),
                        y=ty(line["y"]),
                        mode="lines",
                        line=dict(
                            shape=kwargs.get("where", "hv"),
                            color=plotly_color(kwargs.get("color", None)),
                        ),
                        name=kwargs.get("label", ""),
                        showlegend=bool(kwargs.get("label")) and bool(self._legend),
                    )
                )
            elif plot_type == "stairs":
                kwargs = line["kwargs"]
                values = np.asarray(line["values"])
                edges = line["edges"]
                if edges is None:
                    edges = np.arange(values.size + 1)
                edges = np.asarray(edges)
                x_values = np.repeat(tx(edges), 2)[1:-1]
                y_values = np.repeat(ty(values), 2)
                traces.append(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode="lines",
                        line=dict(color=plotly_color(kwargs.get("color", None))),
                        name=kwargs.get("label", ""),
                        showlegend=bool(kwargs.get("label")) and bool(self._legend),
                    )
                )
            elif plot_type == "broken_barh":
                kwargs = line["kwargs"]
                for start, width in line["xranges"]:
                    traces.append(
                        go.Bar(
                            x=[width * self._xscale],
                            y=[line["yrange"][0] + line["yrange"][1] / 2],
                            base=[txs(start)],
                            orientation="h",
                            width=line["yrange"][1],
                            marker_color=plotly_color(kwargs.get("color", None)),
                            showlegend=False,
                        )
                    )
            elif plot_type == "pie":
                kwargs = line["kwargs"]
                labels = kwargs.get("labels", None)
                colors = kwargs.get("colors", None)
                explode = kwargs.get("explode", None)
                traces.append(
                    go.Pie(
                        values=line["x"],
                        labels=labels,
                        name=kwargs.get("label", ""),
                        showlegend=bool(self._legend),
                        marker=(
                            dict(colors=[plotly_color(c) for c in colors])
                            if colors is not None
                            else None
                        ),
                        pull=list(explode) if explode is not None else None,
                        textinfo=("percent" if kwargs.get("autopct") else None),
                    )
                )
            elif plot_type == "stem":
                kwargs = line["kwargs"]
                traces.append(
                    go.Scatter(
                        x=tx(line["x"]),
                        y=ty(line["y"]),
                        mode="markers+lines",
                        line=dict(color=plotly_color(kwargs.get("color", None))),
                        name=kwargs.get("label", ""),
                        showlegend=bool(kwargs.get("label")) and bool(self._legend),
                    )
                )
            elif plot_type == "stackplot":
                kwargs = line["kwargs"]
                x_values = tx(line["x"])
                colors = kwargs.get("colors", None)
                cumulative = np.zeros(len(x_values))
                for index, values in enumerate(line["ys"]):
                    next_cumulative = cumulative + np.asarray(values)
                    color = (
                        plotly_color(colors[index % len(colors)]) if colors else None
                    )
                    traces.append(
                        go.Scatter(
                            x=x_values,
                            y=ty(next_cumulative),
                            mode="lines",
                            stackgroup="one",
                            fillcolor=color,
                            line=dict(color=color),
                            name=(
                                kwargs.get("labels", [])[index]
                                if index < len(kwargs.get("labels", []))
                                else ""
                            ),
                            showlegend=bool(self._legend),
                        )
                    )
                    cumulative = next_cumulative
            elif plot_type == "boxplot":
                kwargs = line["kwargs"]
                datasets = (
                    line["x"] if isinstance(line["x"], (list, tuple)) else [line["x"]]
                )
                for index, values in enumerate(datasets):
                    showfliers = kwargs.get("showfliers", True)
                    boxpoints = "outliers" if showfliers else False
                    traces.append(
                        go.Box(
                            y=values,
                            name=str(index),
                            boxpoints=boxpoints,
                            showlegend=False,
                        )
                    )
            elif plot_type == "violinplot":
                dataset = line["dataset"]
                datasets = dataset if isinstance(dataset, (list, tuple)) else [dataset]
                for index, values in enumerate(datasets):
                    traces.append(
                        go.Violin(y=values, name=str(index), showlegend=False)
                    )
            elif plot_type == "eventplot":
                positions = line["positions"]
                for row_index, row_positions in enumerate(np.atleast_1d(positions)):
                    for position in np.atleast_1d(row_positions):
                        shapes.append(
                            dict(
                                type="line",
                                x0=txs(position),
                                x1=txs(position),
                                y0=row_index,
                                y1=row_index + 0.8,
                                line=dict(color="black"),
                            )
                        )
            elif plot_type == "contour":
                kwargs = line["kwargs"]
                contours = dict(coloring="lines")
                if self._clabel_kwargs:
                    contours["showlabels"] = True
                    contours["labelfont"] = {
                        "size": self._clabel_kwargs.get(
                            "size", self._clabel_kwargs.get("fontsize")
                        ),
                        "color": self._clabel_kwargs.get("color"),
                        "family": self._clabel_kwargs.get("family"),
                    }
                traces.append(
                    go.Contour(
                        x=line["x"],
                        y=line["y"],
                        z=line["z"],
                        contours=contours,
                        colorscale=kwargs.get("cmap", "Viridis"),
                        showscale=kwargs.get("colorbar", True),
                    )
                )
            elif plot_type == "contourf":
                kwargs = line["kwargs"]
                contours = {}
                if self._clabel_kwargs:
                    contours["showlabels"] = True
                    contours["labelfont"] = {
                        "size": self._clabel_kwargs.get(
                            "size", self._clabel_kwargs.get("fontsize")
                        ),
                        "color": self._clabel_kwargs.get("color"),
                        "family": self._clabel_kwargs.get("family"),
                    }
                traces.append(
                    go.Contour(
                        x=line["x"],
                        y=line["y"],
                        z=line["z"],
                        colorscale=kwargs.get("cmap", "Viridis"),
                        showscale=kwargs.get("colorbar", True),
                        contours=contours,
                    )
                )
            elif plot_type == "pcolormesh":
                kwargs = line["kwargs"]
                traces.append(
                    go.Heatmap(
                        x=line["x"],
                        y=line["y"],
                        z=line["z"],
                        colorscale=kwargs.get("cmap", "Viridis"),
                        showscale=kwargs.get("colorbar", True),
                    )
                )
            elif plot_type in ("pcolor", "pcolorfast"):
                # Plotly's heatmap is the closest equivalent to Matplotlib's
                # pseudocolor artists.  The cell-centered rendering differs
                # slightly from pcolor, but preserves the data and color map.
                kwargs = line["kwargs"]
                traces.append(
                    go.Heatmap(
                        x=line["x"],
                        y=line["y"],
                        z=line["z"],
                        colorscale=kwargs.get("cmap", "Viridis"),
                        showscale=kwargs.get("colorbar", True),
                        opacity=kwargs.get("alpha", None),
                    )
                )
            elif plot_type == "hexbin":
                kwargs = line["kwargs"]
                traces.append(
                    go.Histogram2d(
                        x=tx(line["x"]),
                        y=ty(line["y"]),
                        nbinsx=(
                            kwargs.get("gridsize", 30)
                            if np.isscalar(kwargs.get("gridsize", 30))
                            else 30
                        ),
                        nbinsy=(
                            kwargs.get("gridsize", 30)
                            if np.isscalar(kwargs.get("gridsize", 30))
                            else 30
                        ),
                        colorscale=kwargs.get("cmap", "Viridis"),
                    )
                )
            elif plot_type == "matshow":
                kwargs = line["kwargs"]
                traces.append(
                    go.Heatmap(
                        z=line["data"],
                        colorscale=kwargs.get("cmap", "Viridis"),
                        showscale=kwargs.get("colorbar", True),
                    )
                )
            elif plot_type == "quiver":
                kwargs = line["kwargs"]
                x_values = np.asarray(line["x"])
                y_values = np.asarray(line["y"])
                u_values = np.asarray(line["u"])
                v_values = np.asarray(line["v"])
                if u_values.ndim == 2 and x_values.ndim == 1 and y_values.ndim == 1:
                    x_values, y_values = np.meshgrid(x_values, y_values)
                x_values, y_values, u_values, v_values = np.broadcast_arrays(
                    x_values, y_values, u_values, v_values
                )
                color = plotly_color(kwargs.get("color", "black"))
                arrow_width = kwargs.get("linewidth", kwargs.get("width", 1))
                for x_start, y_start, u_value, v_value in zip(
                    x_values.flat, y_values.flat, u_values.flat, v_values.flat
                ):
                    raw_x_start = x_start
                    raw_y_start = y_start
                    x_start = txs(raw_x_start)
                    y_start = tys(raw_y_start)
                    x_end = txs(raw_x_start + u_value)
                    y_end = tys(raw_y_start + v_value)
                    annotations.append(
                        dict(
                            x=x_end,
                            y=y_end,
                            ax=x_start,
                            ay=y_start,
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=kwargs.get("headlength", 1),
                            arrowwidth=arrow_width,
                            arrowcolor=color,
                            opacity=kwargs.get("alpha", 1),
                        )
                    )
            elif plot_type == "spy":
                kwargs = line["kwargs"]
                matrix = np.asarray(line["matrix"])
                traces.append(
                    go.Heatmap(
                        z=(matrix != 0).astype(int),
                        colorscale=kwargs.get(
                            "cmap", [[0, "rgba(0,0,0,0)"], [1, "black"]]
                        ),
                        showscale=False,
                        xgap=kwargs.get("markersize", 0),
                        ygap=kwargs.get("markersize", 0),
                    )
                )
            elif plot_type == "triplot":
                import matplotlib.tri as mtri

                kwargs = line["kwargs"]
                triangulation = mtri.Triangulation(
                    line["x"], line["y"], triangles=line["triangles"]
                )
                x_values = []
                y_values = []
                x_data = np.asarray(line["x"])
                y_data = np.asarray(line["y"])
                for triangle in triangulation.triangles:
                    indices = [*triangle, triangle[0]]
                    x_values.extend(x_data[indices].tolist() + [None])
                    y_values.extend(y_data[indices].tolist() + [None])
                transformed_x = [
                    None if value is None else txs(value) for value in x_values
                ]
                transformed_y = [
                    None if value is None else tys(value) for value in y_values
                ]
                traces.append(
                    go.Scatter(
                        x=transformed_x,
                        y=transformed_y,
                        mode="lines",
                        line=dict(
                            color=plotly_color(kwargs.get("color", None)),
                            dash=linestyle_map.get(
                                kwargs.get("linestyle", "solid"), "solid"
                            ),
                            width=kwargs.get("linewidth", None),
                        ),
                        name=kwargs.get("label", ""),
                        showlegend=bool(kwargs.get("label")) and bool(self._legend),
                    )
                )
            elif plot_type == "tripcolor":
                import matplotlib.tri as mtri
                from plotly.colors import sample_colorscale

                kwargs = line["kwargs"]
                triangulation = mtri.Triangulation(
                    line["x"], line["y"], triangles=line["triangles"]
                )
                x_data = np.asarray(line["x"])
                y_data = np.asarray(line["y"])
                c_data = np.asarray(line["c"])
                if c_data.ndim == 0:
                    c_data = np.full(len(x_data), c_data.item())
                triangle_values = (
                    c_data
                    if c_data.size == len(triangulation.triangles)
                    else np.asarray(
                        [
                            np.mean(c_data[triangle])
                            for triangle in triangulation.triangles
                        ]
                    )
                )
                finite_values = triangle_values[np.isfinite(triangle_values)]
                minimum = finite_values.min() if finite_values.size else 0.0
                maximum = finite_values.max() if finite_values.size else 1.0
                scale = maximum - minimum or 1.0
                colorscale = kwargs.get("cmap", "Viridis")
                edge_color = plotly_color(kwargs.get("edgecolors", "black"))
                for triangle, value in zip(
                    triangulation.triangles, triangle_values, strict=False
                ):
                    fraction = float(np.clip((value - minimum) / scale, 0, 1))
                    try:
                        fill_color = sample_colorscale(colorscale, [fraction])[0]
                    except (KeyError, ValueError):
                        fill_color = plotly_color(kwargs.get("color", "blue"))
                    indices = [*triangle, triangle[0]]
                    traces.append(
                        go.Scatter(
                            x=tx(x_data[indices]),
                            y=ty(y_data[indices]),
                            mode="lines",
                            fill="toself",
                            fillcolor=fill_color,
                            line=dict(color=edge_color),
                            showlegend=False,
                        )
                    )
            elif plot_type == "streamplot":
                import matplotlib.pyplot as mpl_plt

                kwargs = dict(line["kwargs"])
                stream_figure, stream_axis = mpl_plt.subplots()
                try:
                    stream_set = stream_axis.streamplot(
                        line["x"], line["y"], line["u"], line["v"], **kwargs
                    )
                    segments = stream_set.lines.get_segments()
                    colors = stream_set.lines.get_colors()
                    widths = stream_set.lines.get_linewidths()
                    for index, segment in enumerate(segments):
                        if len(segment) < 2:
                            continue
                        color = plotly_color(colors[min(index, len(colors) - 1)])
                        width = widths[min(index, len(widths) - 1)]
                        traces.append(
                            go.Scatter(
                                x=tx(segment[:, 0]),
                                y=ty(segment[:, 1]),
                                mode="lines",
                                line=dict(color=color, width=width),
                                showlegend=False,
                            )
                        )
                finally:
                    mpl_plt.close(stream_figure)
            elif plot_type in ("tricontour", "tricontourf"):
                import matplotlib.pyplot as mpl_plt
                import matplotlib.tri as mtri

                kwargs = dict(line["kwargs"])
                kwargs.pop("colorbar", None)
                kwargs.pop("label", None)
                triangulation = mtri.Triangulation(
                    line["x"], line["y"], triangles=line["triangles"]
                )
                contour_figure, contour_axis = mpl_plt.subplots()
                try:
                    if plot_type == "tricontourf":
                        contour_set = contour_axis.tricontourf(
                            triangulation, line["z"], **kwargs
                        )
                        colors = contour_set.get_facecolors()
                    else:
                        contour_set = contour_axis.tricontour(
                            triangulation, line["z"], **kwargs
                        )
                        colors = contour_set.get_edgecolors()
                    for index, path in enumerate(contour_set.get_paths()):
                        vertices = path.vertices
                        if len(vertices) < 2:
                            continue
                        color = plotly_color(colors[min(index, len(colors) - 1)])
                        traces.append(
                            go.Scatter(
                                x=tx(vertices[:, 0]),
                                y=ty(vertices[:, 1]),
                                mode="lines",
                                fill="toself" if plot_type == "tricontourf" else None,
                                fillcolor=color if plot_type == "tricontourf" else None,
                                line=dict(color=color),
                                showlegend=False,
                            )
                        )
                finally:
                    mpl_plt.close(contour_figure)
            elif plot_type == "table":
                kwargs = line["kwargs"]
                cell_text = line["cellText"] or []
                col_labels = kwargs.get("colLabels")
                row_labels = kwargs.get("rowLabels")
                if col_labels is not None:
                    header_values = list(col_labels)
                    rows = cell_text
                elif cell_text:
                    header_values = []
                    rows = cell_text
                else:
                    header_values = []
                    rows = []
                columns = list(map(list, zip(*rows))) if rows else []
                if row_labels is not None:
                    columns.insert(0, list(row_labels))
                    if header_values:
                        header_values.insert(0, "")
                traces.append(
                    go.Table(
                        header=dict(values=header_values),
                        cells=dict(values=columns),
                    )
                )
            elif plot_type == "gantt":
                kwargs = line["kwargs"]
                tasks = line["tasks"]
                start_times = tx(line["start_times"])
                durations = np.asarray(line["durations"]) * self._xscale
                y_positions = list(range(len(tasks)))
                self._plotly_barmode_hint = "overlay"
                trace = go.Bar(
                    x=durations,
                    y=y_positions,
                    base=start_times,
                    width=kwargs.get("height", None),
                    orientation="h",
                    name=kwargs.get("label", ""),
                    showlegend=bool(kwargs.get("label")) and bool(self._legend),
                    marker=bar_marker(kwargs),
                    opacity=kwargs.get("alpha", None),
                )
                traces.append(trace)
            elif plot_type == "flame_chart":
                kwargs = line["kwargs"]
                labels = line["labels"]
                parents = line["parents"]
                values = np.asarray(line["values"]) * self._xscale
                start_times = line["start_times"]

                # Calculate depths
                n = len(labels)
                depths = np.zeros(n, dtype=int)
                if start_times is None:
                    start_times = np.zeros(n)
                else:
                    start_times = tx(start_times)

                for i in range(n):
                    if parents[i] is None:
                        depths[i] = 0
                    else:
                        parent_idx = (
                            parents[i]
                            if isinstance(parents[i], int)
                            else list(labels).index(parents[i])
                        )
                        depths[i] = depths[parent_idx] + 1

                # Frames are drawn as a single horizontal bar trace rather
                # than layout shapes, so they carry hover text, per-frame
                # colors, legend entries and zoom/click behaviour.
                frame_colors = [
                    plotly_color(color)
                    for color in _flame_frame_colors(kwargs, depths, n, css=True)
                ]
                hover = line.get("hover")
                if hover is None:
                    hover = [
                        f"{label}<br>{float(value):g}"
                        for label, value in zip(labels, values, strict=False)
                    ]
                self._plotly_barmode_hint = "overlay"
                traces.append(
                    go.Bar(
                        x=[float(value) for value in values],
                        y=[float(depth) + 0.45 for depth in depths],
                        base=[float(start) for start in start_times],
                        width=0.9,
                        orientation="h",
                        name=kwargs.get("label", "") or "",
                        showlegend=bool(kwargs.get("label")) and bool(self._legend),
                        marker=dict(
                            color=frame_colors,
                            line=dict(
                                color=plotly_color(kwargs.get("edgecolor", "black")),
                                width=kwargs.get("linewidth", 0.5),
                            ),
                        ),
                        opacity=kwargs.get("alpha", None),
                        customdata=list(labels),
                        hovertext=hover,
                        hoverinfo="text",
                    )
                )

                for i in range(n):
                    # Add text annotation if the frame is wide enough
                    if values[i] > 0.1 * (start_times.max() + values.max()):
                        annotations.append(
                            dict(
                                x=float(start_times[i] + values[i] / 2),
                                y=float(depths[i] + 0.45),
                                text=labels[i],
                                showarrow=False,
                                font=dict(size=8, color="white"),
                            )
                        )
            elif plot_type == "fill_between":
                kwargs = line["kwargs"]
                x = tx(line["x"])
                if np.isscalar(line["y1"]):
                    y1 = np.full_like(
                        np.asarray(x, dtype=float), float(tys(line["y1"]))
                    )
                else:
                    y1 = ty(line["y1"])
                if np.isscalar(line["y2"]):
                    y2 = np.full_like(
                        np.asarray(x, dtype=float), float(tys(line["y2"]))
                    )
                else:
                    y2 = ty(line["y2"])

                color = plotly_color(kwargs.get("color", kwargs.get("facecolor", None)))
                alpha = kwargs.get("alpha", 0.3)
                fill_trace = go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([y1, y2[::-1]]),
                    fill="toself",
                    fillcolor=color,
                    opacity=alpha,
                    line=dict(color="rgba(0,0,0,0)"),
                    name=kwargs.get("label", ""),
                    showlegend=bool(kwargs.get("label")) and bool(self._legend),
                )
                traces.append(fill_trace)
            elif plot_type == "fill_betweenx":
                kwargs = line["kwargs"]
                y = ty(line["y"])
                x1 = line["x1"]
                x2 = line["x2"]
                if np.isscalar(x1):
                    x1 = np.full_like(y, float(txs(x1)), dtype=float)
                else:
                    x1 = tx(x1)
                if np.isscalar(x2):
                    x2 = np.full_like(y, float(txs(x2)), dtype=float)
                else:
                    x2 = tx(x2)
                color = plotly_color(kwargs.get("color", kwargs.get("facecolor", None)))
                traces.append(
                    go.Scatter(
                        x=np.concatenate([x1, x2[::-1]]),
                        y=np.concatenate([y, y[::-1]]),
                        fill="toself",
                        fillcolor=color,
                        opacity=kwargs.get("alpha", 0.3),
                        line=dict(color="rgba(0,0,0,0)"),
                        name=kwargs.get("label", ""),
                        showlegend=bool(kwargs.get("label")) and bool(self._legend),
                    )
                )
            elif plot_type == "fill":
                kwargs = line["kwargs"]
                polygon_args = line["args"]
                for index in range(0, len(polygon_args), 2):
                    x_values = tx(polygon_args[index])
                    y_values = ty(polygon_args[index + 1])
                    traces.append(
                        go.Scatter(
                            x=x_values,
                            y=y_values,
                            fill="toself",
                            fillcolor=plotly_color(
                                kwargs.get("color", kwargs.get("facecolor", None))
                            ),
                            opacity=kwargs.get("alpha", None),
                            line=dict(
                                color=plotly_color(
                                    kwargs.get("edgecolor", kwargs.get("color", None))
                                )
                            ),
                            name=kwargs.get("label", ""),
                            showlegend=bool(kwargs.get("label"))
                            and bool(self._legend)
                            and index == 0,
                        )
                    )
            elif plot_type == "errorbar":
                kwargs = line["kwargs"]
                marker = kwargs.get("marker")
                mode = "lines+markers" if marker is not None else "lines"
                x_vals = tx(line["x"])
                y_vals = ty(line["y"])
                yerr = line.get("yerr")
                xerr = line.get("xerr")
                if yerr is not None and np.isscalar(yerr):
                    yerr = np.full(len(x_vals), float(yerr))
                if xerr is not None and np.isscalar(xerr):
                    xerr = np.full(len(x_vals), float(xerr))
                # Matplotlib draws error-bar caps only when ``capsize`` is
                # set; Plotly always draws them, sized by ``width``, so
                # ``capsize`` is honored and a caller who omits it (wanting no
                # caps) still gets Plotly's default rather than nothing.
                capsize = kwargs.get("capsize")
                error_width = None if capsize is None else float(capsize)
                error_linewidth = kwargs.get("elinewidth", kwargs.get("capthick"))
                trace = go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode=mode,
                    name=kwargs.get("label", ""),
                    showlegend=bool(kwargs.get("label")) and bool(self._legend),
                    line=dict(
                        color=plotly_color(kwargs.get("color", None)),
                        dash=linestyle_map.get(
                            kwargs.get("linestyle", "solid"), "solid"
                        ),
                        width=kwargs.get("linewidth", None),
                    ),
                    marker=(
                        dict(
                            color=plotly_color(kwargs.get("color", None)),
                            symbol=marker_map.get(marker, "circle"),
                            size=kwargs.get("markersize", None),
                        )
                        if marker is not None
                        else None
                    ),
                    error_y=(
                        dict(
                            type="data",
                            array=yerr,
                            visible=True,
                            width=error_width,
                            thickness=error_linewidth,
                        )
                        if yerr is not None
                        else None
                    ),
                    error_x=(
                        dict(
                            type="data",
                            array=xerr,
                            visible=True,
                            width=error_width,
                            thickness=error_linewidth,
                        )
                        if xerr is not None
                        else None
                    ),
                )
                traces.append(trace)
            elif plot_type in ("axhline", "axvline", "hlines", "vlines"):
                kwargs = line["kwargs"]
                color = plotly_color(kwargs.get("color", kwargs.get("colors", "black")))
                dash = linestyle_map.get(kwargs.get("linestyle", "solid"), "solid")
                width = kwargs.get("linewidth", 1)
                # Layout shapes never appear in a Plotly legend, unlike their
                # Matplotlib equivalents; add a zero-data dummy trace so a
                # labeled line still shows up.
                if kwargs.get("label") and self._legend:
                    traces.append(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            name=kwargs["label"],
                            line=dict(color=color, dash=dash, width=width),
                            showlegend=True,
                        )
                    )
                if plot_type == "axhline":
                    shapes.append(
                        dict(
                            type="line",
                            x0=0,
                            x1=1,
                            xref="paper",
                            y0=tys(line["y"]),
                            y1=tys(line["y"]),
                            line=dict(color=color, dash=dash, width=width),
                        )
                    )
                elif plot_type == "axvline":
                    shapes.append(
                        dict(
                            type="line",
                            y0=0,
                            y1=1,
                            yref="paper",
                            x0=txs(line["x"]),
                            x1=txs(line["x"]),
                            line=dict(color=color, dash=dash, width=width),
                        )
                    )
                elif plot_type == "hlines":
                    y_vals = np.atleast_1d(line["y"])
                    xmins = np.atleast_1d(line["xmin"])
                    xmaxs = np.atleast_1d(line["xmax"])
                    for y, xmin, xmax in zip(y_vals, xmins, xmaxs):
                        shapes.append(
                            dict(
                                type="line",
                                x0=txs(xmin),
                                x1=txs(xmax),
                                y0=tys(y),
                                y1=tys(y),
                                line=dict(color=color, dash=dash, width=width),
                            )
                        )
                elif plot_type == "vlines":
                    x_vals = np.atleast_1d(line["x"])
                    ymins = np.atleast_1d(line["ymin"])
                    ymaxs = np.atleast_1d(line["ymax"])
                    for x, ymin, ymax in zip(x_vals, ymins, ymaxs):
                        shapes.append(
                            dict(
                                type="line",
                                x0=txs(x),
                                x1=txs(x),
                                y0=tys(ymin),
                                y1=tys(ymax),
                                line=dict(color=color, dash=dash, width=width),
                            )
                        )
            elif plot_type in ("axvspan", "axhspan"):
                kwargs = line["kwargs"]
                color = plotly_color(kwargs.get("color", kwargs.get("facecolor", None)))
                span_shape = dict(
                    type="rect",
                    fillcolor=color,
                    opacity=kwargs.get("alpha", 0.3),
                    line=dict(width=0),
                )
                if plot_type == "axvspan":
                    span_shape.update(
                        x0=txs(line["xmin"]),
                        x1=txs(line["xmax"]),
                        y0=0,
                        y1=1,
                        yref="paper",
                    )
                else:
                    span_shape.update(
                        x0=0,
                        x1=1,
                        xref="paper",
                        y0=tys(line["ymin"]),
                        y1=tys(line["ymax"]),
                    )
                shapes.append(span_shape)
                if kwargs.get("label") and self._legend:
                    traces.append(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="markers",
                            name=kwargs["label"],
                            marker=dict(
                                color=color,
                                opacity=kwargs.get("alpha", 0.3),
                                symbol="square",
                                size=12,
                            ),
                            showlegend=True,
                        )
                    )
            elif plot_type == "arrow":
                kwargs = line["kwargs"]
                annotations.append(
                    dict(
                        x=txs(line["x"] + line["dx"]),
                        y=tys(line["y"] + line["dy"]),
                        ax=txs(line["x"]),
                        ay=tys(line["y"]),
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        text="",
                        showarrow=True,
                        arrowhead=kwargs.get("arrowhead", 2),
                        arrowcolor=plotly_color(kwargs.get("color", "black")),
                    )
                )
            elif plot_type == "axline":
                kwargs = line["kwargs"]
                if line["xy2"] is None and line["slope"] is None:
                    raise ValueError("axline requires xy2 or slope")
                if line["xy2"] is not None:
                    x0, y0 = line["xy1"]
                    x1, y1 = line["xy2"]
                    slope = (y1 - y0) / (x1 - x0)
                else:
                    x0, y0 = line["xy1"]
                    slope = line["slope"]
                shapes.append(
                    dict(
                        type="line",
                        x0=0,
                        x1=1,
                        xref="paper",
                        y0=tys(y0 - slope * x0),
                        y1=tys(y0 + slope * (1 - x0)),
                        line=dict(color=plotly_color(kwargs.get("color", "black"))),
                    )
                )
            elif plot_type in ("text", "annotate"):
                kwargs = line["kwargs"]
                # Matplotlib's ha/va anchor the text block to the (x, y)
                # point; Plotly's xanchor/yanchor do the same thing under a
                # different name ("center" means the same in both).
                ha_to_xanchor = {"left": "left", "right": "right", "center": "center"}
                va_to_yanchor = {
                    "top": "top",
                    "bottom": "bottom",
                    "center": "middle",
                    "baseline": "bottom",
                }
                font = dict(
                    color=plotly_color(kwargs.get("color", None)),
                    size=kwargs.get("fontsize", None),
                    family=kwargs.get("fontfamily", kwargs.get("family", None)),
                    weight=kwargs.get("fontweight", None),
                )
                if plot_type == "text":
                    x = txs(float(line["x"]))
                    y = tys(float(line["y"]))
                    text = line["s"]
                    annotations.append(
                        dict(
                            x=x,
                            y=y,
                            text=text,
                            showarrow=False,
                            xanchor=ha_to_xanchor.get(kwargs.get("ha"), "left"),
                            yanchor=va_to_yanchor.get(kwargs.get("va"), "bottom"),
                            font=font,
                        )
                    )
                else:
                    x = txs(float(line["xy"][0]))
                    y = tys(float(line["xy"][1]))
                    ann = dict(
                        x=x,
                        y=y,
                        text=line["text"],
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-30,
                        font=font,
                    )
                    if line.get("xytext") is not None:
                        tx_val = txs(float(line["xytext"][0]))
                        ty_val = tys(float(line["xytext"][1]))
                        ann.update(axref="x", ayref="y", ax=tx_val, ay=ty_val)
                    annotations.append(ann)
            elif plot_type == "imshow":
                kwargs = line["kwargs"]
                heatmap = go.Heatmap(
                    z=line["data"],
                    colorscale=kwargs.get("cmap", "Viridis"),
                    showscale=True,
                )
                traces.append(heatmap)
                last_heatmap_idx = len(traces) - 1
            elif plot_type == "colorbar":
                if last_heatmap_idx is not None:
                    label = line.get("label", "") or line["kwargs"].get("label", "")
                    if label:
                        traces[last_heatmap_idx].update(
                            colorbar=dict(title=dict(text=label))
                        )
            elif plot_type == "patch":
                kwargs = line["kwargs"]
                patch = line["patch"]
                try:
                    import matplotlib.patches as mpl_patches
                except Exception:
                    mpl_patches = None

                def _patch_line_color():
                    return plotly_color(
                        kwargs.get(
                            "edgecolor",
                            kwargs.get(
                                "color",
                                (
                                    patch.get_edgecolor()
                                    if hasattr(patch, "get_edgecolor")
                                    else "black"
                                ),
                            ),
                        )
                    )

                def _patch_fill_color():
                    return plotly_color(
                        kwargs.get(
                            "facecolor",
                            (
                                patch.get_facecolor()
                                if hasattr(patch, "get_facecolor")
                                else None
                            ),
                        )
                    )

                patch_label = kwargs.get("label")
                if patch_label is None and hasattr(patch, "get_label"):
                    raw = patch.get_label()
                    if raw and not str(raw).startswith("_"):
                        patch_label = str(raw)

                hovertext = line.get("hover")
                if hovertext is None:
                    hovertext = kwargs.get("hovertext")

                def _add_hover_trace(x_pts, y_pts, hovertext=hovertext):
                    # Plotly shapes can't show hover info themselves, so an
                    # invisible filled polygon trace is overlaid on top of
                    # the shape's outline to make the whole area hoverable.
                    if hovertext is None:
                        return
                    hover_handled_shapes.add(len(shapes) - 1)
                    traces.append(
                        go.Scatter(
                            x=list(x_pts) + [x_pts[0]],
                            y=list(y_pts) + [y_pts[0]],
                            mode="lines",
                            fill="toself",
                            fillcolor="rgba(0,0,0,0)",
                            line=dict(width=0),
                            hoveron="fills",
                            hoverinfo="text",
                            hovertext=hovertext,
                            showlegend=False,
                        )
                    )

                if mpl_patches is not None and isinstance(patch, mpl_patches.Rectangle):
                    x0 = txs(patch.get_x())
                    y0 = tys(patch.get_y())
                    x1 = txs(patch.get_x() + patch.get_width())
                    y1 = tys(patch.get_y() + patch.get_height())
                    shapes.append(
                        dict(
                            type="rect",
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                            line=dict(color=_patch_line_color()),
                            fillcolor=_patch_fill_color(),
                            opacity=kwargs.get("alpha", None),
                        )
                    )
                    patch_bounds_x.extend([x0, x1])
                    patch_bounds_y.extend([y0, y1])
                    _add_hover_trace([x0, x1, x1, x0], [y0, y0, y1, y1])
                elif mpl_patches is not None and isinstance(patch, mpl_patches.Circle):
                    cx = txs(patch.center[0])
                    cy = tys(patch.center[1])
                    rx = abs(txs(patch.center[0] + patch.radius) - cx)
                    ry = abs(tys(patch.center[1] + patch.radius) - cy)
                    path = (
                        f"M {cx - rx},{cy} "
                        f"A {rx},{ry} 0 1,0 {cx + rx},{cy} "
                        f"A {rx},{ry} 0 1,0 {cx - rx},{cy} Z"
                    )
                    shapes.append(
                        dict(
                            type="path",
                            path=path,
                            line=dict(color=_patch_line_color()),
                            fillcolor=_patch_fill_color(),
                            opacity=kwargs.get("alpha", None),
                        )
                    )
                    patch_bounds_x.extend([cx - rx, cx + rx])
                    patch_bounds_y.extend([cy - ry, cy + ry])
                    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
                    _add_hover_trace(cx + rx * np.cos(angles), cy + ry * np.sin(angles))
                elif mpl_patches is not None and isinstance(patch, mpl_patches.Ellipse):
                    angle = float(getattr(patch, "angle", 0.0) or 0.0)
                    if angle == 0.0:
                        cx = txs(patch.center[0])
                        cy = tys(patch.center[1])
                        rx = abs(txs(patch.center[0] + patch.width / 2.0) - cx)
                        ry = abs(tys(patch.center[1] + patch.height / 2.0) - cy)
                        path = (
                            f"M {cx - rx},{cy} "
                            f"A {rx},{ry} 0 1,0 {cx + rx},{cy} "
                            f"A {rx},{ry} 0 1,0 {cx - rx},{cy} Z"
                        )
                        patch_bounds_x.extend([cx - rx, cx + rx])
                        patch_bounds_y.extend([cy - ry, cy + ry])
                        hover_x = cx + rx * np.cos(
                            np.linspace(0, 2 * np.pi, 32, endpoint=False)
                        )
                        hover_y = cy + ry * np.sin(
                            np.linspace(0, 2 * np.pi, 32, endpoint=False)
                        )
                    else:
                        # A rotated ellipse is no longer axis-aligned once the
                        # per-axis x/y transforms are applied, so it can't be
                        # described with a two-arc SVG path; approximate it
                        # with a sampled polygon instead, computed in data
                        # space (where the rotation is defined) and then
                        # transformed point-by-point.
                        cx0, cy0 = patch.center
                        a = patch.width / 2.0
                        b = patch.height / 2.0
                        theta = np.radians(angle)
                        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
                        ex = a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(
                            theta
                        )
                        ey = a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(
                            theta
                        )
                        hover_x = np.array([txs(cx0 + dx) for dx in ex])
                        hover_y = np.array([tys(cy0 + dy) for dy in ey])
                        path = (
                            "M "
                            + " L ".join(
                                f"{x},{y}"
                                for x, y in zip(hover_x, hover_y, strict=False)
                            )
                            + " Z"
                        )
                        patch_bounds_x.extend(hover_x.tolist())
                        patch_bounds_y.extend(hover_y.tolist())
                    shapes.append(
                        dict(
                            type="path",
                            path=path,
                            line=dict(color=_patch_line_color()),
                            fillcolor=_patch_fill_color(),
                            opacity=kwargs.get("alpha", None),
                        )
                    )
                    _add_hover_trace(hover_x, hover_y)
                elif mpl_patches is not None and isinstance(patch, mpl_patches.Polygon):
                    pts = patch.get_xy()
                    if len(pts) >= 2:
                        pts_t = [(txs(float(x)), tys(float(y))) for x, y in pts]
                        path = "M " + " L ".join(f"{x},{y}" for x, y in pts_t) + " Z"
                        shapes.append(
                            dict(
                                type="path",
                                path=path,
                                line=dict(color=_patch_line_color()),
                                fillcolor=_patch_fill_color(),
                                opacity=kwargs.get("alpha", None),
                            )
                        )
                        patch_bounds_x.extend(x for x, _ in pts_t)
                        patch_bounds_y.extend(y for _, y in pts_t)
                        _add_hover_trace([x for x, _ in pts_t], [y for _, y in pts_t])

                # Plotly shapes don't participate in legends; add a dummy trace.
                if patch_label and bool(self._legend):
                    traces.append(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            name=patch_label,
                            line=dict(color=_patch_line_color()),
                            showlegend=True,
                        )
                    )

            self._apply_plotly_neutral(
                line,
                traces,
                shapes,
                trace_start,
                shape_start,
                skip_shapes=hover_handled_shapes,
            )

        if patch_bounds_x:
            traces.append(
                go.Scatter(
                    x=patch_bounds_x,
                    y=patch_bounds_y,
                    mode="markers",
                    marker=dict(opacity=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        if self._bar_label_kwargs:
            label_kwargs = self._bar_label_kwargs
            fmt = label_kwargs.get("fmt", "{x}")

            def format_bar_label(value):
                if callable(fmt):
                    return str(fmt(value))
                if "%" in str(fmt):
                    return str(fmt) % value
                return str(fmt).format(x=value)

            for line in self._iter_layer_lines(layers=layers):
                if line["plot_type"] == "bar":
                    x_values = np.asarray(line["x"])
                    heights = np.asarray(line["height"])
                    bottom = np.asarray(line["kwargs"].get("bottom", 0))
                    for x_value, height, base in zip(
                        x_values,
                        heights,
                        np.broadcast_to(bottom, heights.shape),
                        strict=False,
                    ):
                        edge = base + height
                        annotations.append(
                            dict(
                                x=txs(x_value),
                                y=tys(edge),
                                text=format_bar_label(height),
                                showarrow=False,
                                yshift=label_kwargs.get("padding", 0),
                            )
                        )
                elif line["plot_type"] == "barh":
                    y_values = np.asarray(line["y"])
                    widths = np.asarray(line["width"])
                    left = np.asarray(line["kwargs"].get("left", 0))
                    for y_value, width, start in zip(
                        y_values,
                        widths,
                        np.broadcast_to(left, widths.shape),
                        strict=False,
                    ):
                        edge = start + width
                        annotations.append(
                            dict(
                                x=txs(edge),
                                y=tys(y_value),
                                text=format_bar_label(width),
                                showarrow=False,
                                xshift=label_kwargs.get("padding", 0),
                            )
                        )

        return traces, shapes, annotations

    def _apply_plotly_neutral(
        self,
        line,
        traces,
        shapes,
        trace_start,
        shape_start,
        skip_shapes=(),
    ):
        """Apply the backend-neutral ``hover=`` / ``meta=`` options.

        Every trace and shape produced by ``line`` is tagged, so callers never
        have to match traces by creation order after the figure is built.
        ``hover`` may be a single string, one entry per point, or a 2-D array
        for heatmap-like traces; shapes, which cannot show hover text of their
        own, get an invisible overlay trace instead.
        """
        hover = line.get("hover")
        meta = line.get("meta")
        if hover is None and meta is None:
            return

        for trace in traces[trace_start:]:
            if meta is not None:
                try:
                    trace.update(meta=meta)
                except (ValueError, TypeError):
                    pass
            if hover is not None:
                try:
                    trace.update(hovertext=hover, hoverinfo="text")
                except (ValueError, TypeError):
                    pass

        for index in range(shape_start, len(shapes)):
            shape = shapes[index]
            if meta is not None and "name" not in shape:
                shape["name"] = str(meta)
            if hover is None or index in skip_shapes:
                continue
            corners = self._plotly_shape_corners(shape)
            if corners is None:
                continue
            x_points, y_points = corners
            traces.append(
                go.Scatter(
                    x=list(x_points) + [x_points[0]],
                    y=list(y_points) + [y_points[0]],
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(0,0,0,0)",
                    line=dict(width=0),
                    hoveron="fills",
                    hoverinfo="text",
                    hovertext=hover,
                    meta=meta,
                    showlegend=False,
                )
            )

    def _plotly_shape_corners(self, shape):
        """Return the polygon outline of a rectangular shape, if hoverable.

        Shapes anchored to the paper (full-height spans from ``axvspan``, for
        instance) have no data coordinates on that axis; they can only be
        given a hover overlay when the subplot has explicit limits to
        substitute.
        """
        if shape.get("type") != "rect":
            return None
        try:
            if shape.get("xref") == "paper":
                if self._xmin is None or self._xmax is None:
                    return None
                x0 = self._transform_scalar_x(self._xmin)
                x1 = self._transform_scalar_x(self._xmax)
            else:
                x0 = float(shape["x0"])
                x1 = float(shape["x1"])
            if shape.get("yref") == "paper":
                if self._ymin is None or self._ymax is None:
                    return None
                y0 = self._transform_scalar_y(self._ymin)
                y1 = self._transform_scalar_y(self._ymax)
            else:
                y0 = float(shape["y0"])
                y1 = float(shape["y1"])
        except (KeyError, TypeError, ValueError):
            return None
        return [x0, x1, x1, x0], [y0, y0, y1, y1]

    def _iter_layer_lines(self, layers=None):
        for layer_name in sorted(self.layered_line_data):
            layer_lines = self.layered_line_data[layer_name]
            if layers and layer_name not in layers:
                continue
            for line in layer_lines:
                yield line

    def _symlog_transform(self, values):
        array = np.asarray(values, dtype=float)
        return np.sign(array) * np.log10(1.0 + np.abs(array))

    def _symlog_inverse(self, values):
        array = np.asarray(values, dtype=float)
        return np.sign(array) * (10 ** np.abs(array) - 1.0)

    def _plotext_axis_scale(self, axis: str):
        return self._xaxis_scale if axis == "x" else self._yaxis_scale

    def _plotext_axis_transform(self, values, axis: str):
        array = np.asarray(values)
        if axis == "x":
            transformed = (array + self._xshift) * self._xscale
        else:
            transformed = (array + self._yshift) * self._yscale
        if self._plotext_axis_scale(axis) == "symlog":
            return self._symlog_transform(transformed)
        return transformed

    def _transform_x(self, values):
        return self._plotext_axis_transform(values, "x")

    def _transform_y(self, values):
        return self._plotext_axis_transform(values, "y")

    def _transform_scalar_x(self, value):
        return float(np.asarray(self._transform_x([value]))[0])

    def _transform_scalar_y(self, value):
        return float(np.asarray(self._transform_y([value]))[0])

    def _plotext_plot_kwargs(self, kwargs):
        return {
            key: kwargs[key]
            for key in ("marker", "color", "label")
            if kwargs.get(key) is not None
        }

    def _plotext_scatter_kwargs(self, kwargs):
        filtered = self._plotext_plot_kwargs(kwargs)
        if kwargs.get("style") is not None:
            filtered["style"] = kwargs["style"]
        return filtered

    def _plotext_bar_kwargs(self, kwargs):
        return {
            key: kwargs[key]
            for key in ("marker", "color", "fill", "width", "label")
            if kwargs.get(key) is not None
        }

    def _plotext_text_kwargs(self, kwargs):
        return {
            key: kwargs[key]
            for key in ("color", "background", "style", "orientation", "alignment")
            if kwargs.get(key) is not None
        }

    def _plotext_native(self, value):
        return value.item() if isinstance(value, np.generic) else value

    def _plotext_color(self, *candidates):
        for color in candidates:
            if color is None:
                continue
            if isinstance(color, tuple) and len(color) >= 4 and color[3] == 0:
                continue
            return color
        return None

    def _plotext_patch_style(self, patch, kwargs):
        edgecolor = kwargs.get(
            "edgecolor",
            kwargs.get(
                "color",
                patch.get_edgecolor() if hasattr(patch, "get_edgecolor") else None,
            ),
        )
        facecolor = kwargs.get(
            "facecolor",
            patch.get_facecolor() if hasattr(patch, "get_facecolor") else None,
        )
        fill = kwargs.get("fill")
        if fill is None and hasattr(patch, "get_fill"):
            fill = bool(patch.get_fill())
        fill = bool(fill)
        color = self._plotext_color(facecolor if fill else None, edgecolor, facecolor)
        label = kwargs.get("label")
        if label is None and hasattr(patch, "get_label"):
            patch_label = patch.get_label()
            if patch_label and not str(patch_label).startswith("_"):
                label = patch_label
        return color, fill, label

    def _plotext_patch_vertices(self, patch):
        if hasattr(patch, "get_path") and hasattr(patch, "get_patch_transform"):
            vertices = np.asarray(
                patch.get_path().transformed(patch.get_patch_transform()).vertices
            )
        elif hasattr(patch, "get_xy"):
            vertices = np.asarray(patch.get_xy())
        else:
            raise NotImplementedError(
                f"plotext backend does not support patch type: {type(patch).__name__}"
            )
        if vertices.size == 0:
            return vertices.reshape(0, 2)
        if vertices.shape[1] != 2:
            raise NotImplementedError(
                f"plotext backend does not support patch type: {type(patch).__name__}"
            )
        return vertices

    def _plotext_patch_bounds(self, patch):
        vertices = self._plotext_patch_vertices(patch)
        if vertices.size == 0:
            return [], []
        return vertices[:, 0].tolist(), vertices[:, 1].tolist()

    def _plotext_apply_patch_transform(self, vertices):
        if vertices.size == 0:
            return vertices
        transformed = np.empty_like(vertices, dtype=float)
        transformed[:, 0] = self._transform_x(vertices[:, 0])
        transformed[:, 1] = self._transform_y(vertices[:, 1])
        return transformed

    def _plotext_draw_patch(self, ax, patch, kwargs):
        color, fill, label = self._plotext_patch_style(patch, kwargs)
        vertices = self._plotext_apply_patch_transform(
            self._plotext_patch_vertices(patch)
        )
        if vertices.size == 0:
            return color
        if not np.array_equal(vertices[0], vertices[-1]):
            vertices = np.vstack([vertices, vertices[0]])
        plot_kwargs = {"color": color, "label": label}
        if fill:
            plot_kwargs["fillx"] = "internal"
        ax.plot(vertices[:, 0].tolist(), vertices[:, 1].tolist(), **plot_kwargs)
        return color

    def _coerce_numeric_array(self, values):
        if values is None:
            return None
        array = np.asarray(values)
        if array.ndim == 0:
            array = array.reshape(1)
        try:
            return array.astype(float)
        except (TypeError, ValueError):
            return None

    def _plotext_bounds(self, layers=None):
        xs = []
        ys = []

        def extend_x(values):
            array = self._coerce_numeric_array(values)
            if array is not None:
                xs.extend(array.tolist())

        def extend_y(values):
            array = self._coerce_numeric_array(values)
            if array is not None:
                ys.extend(array.tolist())

        for line in self._iter_layer_lines(layers=layers):
            plot_type = line["plot_type"]
            if plot_type in {"plot", "scatter", "errorbar"}:
                x = self._transform_x(line["x"])
                y = self._transform_y(line["y"])
                extend_x(x)
                extend_y(y)
                xerr = self._plotext_error_values(line.get("xerr"), len(x))
                yerr = self._plotext_error_values(line.get("yerr"), len(y))
                xerr = self._coerce_numeric_array(xerr)
                yerr = self._coerce_numeric_array(yerr)
                if xerr is not None:
                    extend_x(x - xerr)
                    extend_x(x + xerr)
                if yerr is not None:
                    extend_y(y - yerr)
                    extend_y(y + yerr)
            elif plot_type == "bar":
                extend_x(self._transform_x(line["x"]))
                extend_y(np.asarray(line["height"]) * self._yscale)
                extend_y([self._transform_scalar_y(0)])
            elif plot_type == "fill_between":
                extend_x(self._transform_x(line["x"]))
                y1 = (
                    line["y1"]
                    if np.isscalar(line["y1"])
                    else self._transform_y(line["y1"])
                )
                y2 = (
                    self._transform_scalar_y(line["y2"])
                    if np.isscalar(line["y2"])
                    else self._transform_y(line["y2"])
                )
                extend_y(y1)
                extend_y(y2)
            elif plot_type == "hlines":
                extend_y(self._transform_y(line["y"]))
                extend_x(self._transform_x(line["xmin"]))
                extend_x(self._transform_x(line["xmax"]))
            elif plot_type == "vlines":
                extend_x(self._transform_x(line["x"]))
                extend_y(self._transform_y(line["ymin"]))
                extend_y(self._transform_y(line["ymax"]))
            elif plot_type == "axhline":
                extend_y([self._transform_scalar_y(line["y"])])
            elif plot_type == "axvline":
                extend_x([self._transform_scalar_x(line["x"])])
            elif plot_type == "annotate":
                extend_x([self._transform_scalar_x(line["xy"][0])])
                extend_y([self._transform_scalar_y(line["xy"][1])])
                if line["xytext"] is not None:
                    extend_x([self._transform_scalar_x(line["xytext"][0])])
                    extend_y([self._transform_scalar_y(line["xytext"][1])])
            elif plot_type == "text":
                extend_x([self._transform_scalar_x(line["x"])])
                extend_y([self._transform_scalar_y(line["y"])])
            elif plot_type == "imshow":
                data = np.asarray(line["data"])
                if data.ndim >= 2 and data.shape[0] and data.shape[1]:
                    extend_x([0, data.shape[1] - 1])
                    extend_y([0, data.shape[0] - 1])
            elif plot_type == "patch":
                patch_xs, patch_ys = self._plotext_patch_bounds(line["patch"])
                extend_x(self._transform_x(patch_xs))
                extend_y(self._transform_y(patch_ys))
        return xs, ys

    def _plotext_error_values(self, error, count):
        if error is None:
            return None
        values = np.asarray(error, dtype=float)
        if values.ndim == 0:
            # Plotext's error() interprets each value as the full bar width,
            # while Matplotlib interprets it as the distance from the point
            # to one end of a symmetric error bar.
            return [float(values) * 2] * count
        if values.ndim == 2 and values.shape[0] == 2:
            # Matplotlib's asymmetric form is [lower, upper]. Plotext only
            # accepts symmetric widths, so preserve the total extent.
            values = values[0] + values[1]
        else:
            values = values * 2
        return values.tolist()

    def _plotext_ranges(self, layers=None):
        xs, ys = self._plotext_bounds(layers=layers)
        xmin = self._xmin if self._xmin is not None else (min(xs) if xs else None)
        xmax = self._xmax if self._xmax is not None else (max(xs) if xs else None)
        ymin = self._ymin if self._ymin is not None else (min(ys) if ys else None)
        ymax = self._ymax if self._ymax is not None else (max(ys) if ys else None)
        return xmin, xmax, ymin, ymax

    def _plotext_format_tick(self, value):
        value = float(value)
        if abs(value) >= 1000 or (0 < abs(value) < 0.01):
            return f"{value:.1e}"
        if value.is_integer():
            return str(int(value))
        return f"{value:.3g}"

    def _plotext_axis_limit(self, value, axis: str):
        if value is None:
            return None
        if axis == "x":
            return self._transform_scalar_x(value)
        return self._transform_scalar_y(value)

    def _plotext_symlog_ticks(self, axis: str, lower, upper, ticks=None, labels=None):
        if ticks is not None:
            positions = self._plotext_axis_transform(ticks, axis).tolist()
            if labels is None:
                labels = [self._plotext_format_tick(tick) for tick in np.asarray(ticks)]
            return positions, list(labels)

        if lower is None or upper is None:
            return None, None

        positions = np.linspace(lower, upper, 5)
        labels = [
            self._plotext_format_tick(value)
            for value in self._symlog_inverse(positions)
        ]
        return positions.tolist(), labels

    def _plotext_apply_aspect(self, ax, layers=None):
        if self._aspect in (None, "auto"):
            return
        if self._aspect == "equal":
            aspect = 1.0
        elif isinstance(self._aspect, (int, float)):
            aspect = float(self._aspect)
        else:
            raise NotImplementedError(
                f"plotext backend does not support aspect={self._aspect!r}"
            )
        if aspect <= 0:
            raise ValueError("Aspect ratio must be positive.")

        xmin, xmax, ymin, ymax = self._plotext_ranges(layers=layers)
        if None in (xmin, xmax, ymin, ymax):
            return
        x_span = abs(xmax - xmin) or 1.0
        y_span = abs(ymax - ymin) or 1.0
        height = 16
        width = int(round(height * (x_span / (y_span * aspect)) * 2.0))
        title_hint = len(
            " | ".join(
                [
                    part
                    for part in [self._title, getattr(self, "_caption", None)]
                    if part
                ]
            )
        )
        width = max(40, title_hint + 6, min(width, 80))
        ax.plotsize(width, height)

    def _plotext_colorbar_note(self, image_data, label):
        data = np.asarray(image_data)
        if data.size == 0:
            return label or "colorbar"
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return label or "colorbar"
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        prefix = f"{label}: " if label else ""
        return f"{prefix}{self._plotext_format_tick(vmin)}..{self._plotext_format_tick(vmax)}"

    def _plotext_add_legend(self, ax, entries, layers=None):
        if self._xaxis_scale in {"log", "symlog"} or self._yaxis_scale in {
            "log",
            "symlog",
        }:
            return

        unique_entries = []
        seen = set()
        for label, color in entries:
            if not label or label in seen:
                continue
            unique_entries.append((label, color))
            seen.add(label)
        if not unique_entries:
            return

        xmin, xmax, ymin, ymax = self._plotext_ranges(layers=layers)
        if None in (xmin, xmax, ymin, ymax):
            return

        x_span = xmax - xmin or 1.0
        y_span = ymax - ymin or 1.0
        x_pos = xmin + 0.7 * x_span
        y_pos = ymax - 0.08 * y_span
        y_step = 0.08 * y_span

        for index, (label, color) in enumerate(unique_entries):
            ax.text(
                label,
                x_pos,
                y_pos - index * y_step,
                color=color,
                alignment="left",
            )

    def plot_plotext(self, ax, layers=None):
        legend_entries = []
        colorbar_notes = []
        last_image_data = None
        for line in self._iter_layer_lines(layers=layers):
            plot_type = line["plot_type"]
            kwargs = line["kwargs"]
            if plot_type == "plot":
                x = self._transform_x(line["x"]).tolist()
                y = self._transform_y(line["y"]).tolist()
                plot_kwargs = self._plotext_plot_kwargs(kwargs)
                ax.plot(x, y, **plot_kwargs)
                legend_entries.append((kwargs.get("label"), kwargs.get("color")))
            elif plot_type == "scatter":
                x = self._transform_x(line["x"]).tolist()
                y = self._transform_y(line["y"]).tolist()
                scatter_kwargs = self._plotext_scatter_kwargs(kwargs)
                ax.scatter(x, y, **scatter_kwargs)
                legend_entries.append((kwargs.get("label"), kwargs.get("color")))
            elif plot_type == "bar":
                transformed_heights = np.asarray(line["height"]) * self._yscale
                bar_kwargs = self._plotext_bar_kwargs(kwargs)
                if self._plotext_axis_scale("y") == "symlog":
                    transformed_heights = self._symlog_transform(transformed_heights)
                    bar_kwargs["minimum"] = 0.0
                ax.bar(
                    self._transform_x(line["x"]).tolist(),
                    transformed_heights.tolist(),
                    **bar_kwargs,
                )
                legend_entries.append((kwargs.get("label"), kwargs.get("color")))
            elif plot_type == "gantt":
                tasks = line["tasks"]
                start_times = self._transform_x(line["start_times"]).tolist()
                durations = (np.asarray(line["durations"]) * self._xscale).tolist()
                y_positions = list(range(len(tasks)))
                gantt_kwargs = self._plotext_bar_kwargs(kwargs)
                gantt_kwargs["orientation"] = "h"
                for i, (start, duration) in enumerate(zip(start_times, durations)):
                    ax.bar(
                        [start + duration / 2],
                        [i],
                        width=duration,
                        **gantt_kwargs,
                    )
                ax.yticks(y_positions, tasks)
                legend_entries.append((kwargs.get("label"), kwargs.get("color")))
            elif plot_type == "flame_chart":
                labels = line["labels"]
                parents = line["parents"]
                values = (np.asarray(line["values"]) * self._xscale).tolist()
                start_times = line["start_times"]

                # Calculate depths
                n = len(labels)
                depths = np.zeros(n, dtype=int)
                if start_times is None:
                    start_times = np.zeros(n).tolist()
                else:
                    start_times = self._transform_x(line["start_times"]).tolist()

                for i in range(n):
                    if parents[i] is None:
                        depths[i] = 0
                    else:
                        parent_idx = (
                            parents[i]
                            if isinstance(parents[i], int)
                            else list(labels).index(parents[i])
                        )
                        depths[i] = depths[parent_idx] + 1

                # Draw bars for each frame
                flame_kwargs = self._plotext_bar_kwargs(kwargs)
                flame_kwargs["orientation"] = "h"

                # Explicit per-frame colors win; otherwise cycle by depth using
                # names the terminal backend understands.
                explicit_colors = kwargs.get("colors")
                if isinstance(explicit_colors, str) or not hasattr(
                    explicit_colors, "__len__"
                ):
                    explicit_colors = (
                        None if explicit_colors is None else [explicit_colors]
                    )
                depth_colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]

                for i in range(n):
                    if explicit_colors:
                        color = explicit_colors[i % len(explicit_colors)]
                    else:
                        color = depth_colors[depths[i] % len(depth_colors)]

                    ax.bar(
                        [start_times[i] + values[i] / 2],
                        [depths[i]],
                        width=values[i],
                        color=color,
                        **{k: v for k, v in flame_kwargs.items() if k != "color"},
                    )

                legend_entries.append((kwargs.get("label"), kwargs.get("color")))
            elif plot_type == "fill_between":
                x = self._transform_x(line["x"]).tolist()
                y1 = self._transform_y(line["y1"]).tolist()
                plot_kwargs = self._plotext_plot_kwargs(kwargs)
                if np.isscalar(line["y2"]):
                    plot_kwargs["filly"] = line["y2"]
                    ax.plot(x, y1, **plot_kwargs)
                else:
                    y2 = self._transform_y(line["y2"]).tolist()
                    polygon_x = x + x[::-1]
                    polygon_y = y1 + y2[::-1]
                    plot_kwargs["fillx"] = "internal"
                    ax.plot(polygon_x, polygon_y, **plot_kwargs)
                legend_entries.append((kwargs.get("label"), kwargs.get("color")))
            elif plot_type == "errorbar":
                x = self._transform_x(line["x"]).tolist()
                y = self._transform_y(line["y"]).tolist()
                ax.error(
                    x,
                    y,
                    xerr=self._plotext_error_values(line["xerr"], len(x)),
                    yerr=self._plotext_error_values(line["yerr"], len(y)),
                    color=kwargs.get("color"),
                    label=kwargs.get("label"),
                )
                legend_entries.append((kwargs.get("label"), kwargs.get("color")))
            elif plot_type == "hlines":
                ys = np.atleast_1d(line["y"])
                xmins = np.atleast_1d(line["xmin"])
                xmaxs = np.atleast_1d(line["xmax"])
                count = max(len(ys), len(xmins), len(xmaxs))
                ys = np.resize(ys, count)
                xmins = np.resize(xmins, count)
                xmaxs = np.resize(xmaxs, count)
                for y, xmin, xmax in zip(ys, xmins, xmaxs):
                    ax.plot(
                        self._transform_x([xmin, xmax]).tolist(),
                        [self._transform_scalar_y(y), self._transform_scalar_y(y)],
                        color=kwargs.get("color"),
                    )
            elif plot_type == "vlines":
                xs = np.atleast_1d(line["x"])
                ymins = np.atleast_1d(line["ymin"])
                ymaxs = np.atleast_1d(line["ymax"])
                count = max(len(xs), len(ymins), len(ymaxs))
                xs = np.resize(xs, count)
                ymins = np.resize(ymins, count)
                ymaxs = np.resize(ymaxs, count)
                for x, ymin, ymax in zip(xs, ymins, ymaxs):
                    ax.plot(
                        [self._transform_scalar_x(x), self._transform_scalar_x(x)],
                        self._transform_y([ymin, ymax]).tolist(),
                        color=kwargs.get("color"),
                    )
            elif plot_type == "annotate":
                text_x, text_y = (
                    line["xytext"] if line["xytext"] is not None else line["xy"]
                )
                arrowprops = kwargs.get("arrowprops")
                text_x = self._plotext_native(self._transform_scalar_x(text_x))
                text_y = self._plotext_native(self._transform_scalar_y(text_y))
                xy_x = self._plotext_native(self._transform_scalar_x(line["xy"][0]))
                xy_y = self._plotext_native(self._transform_scalar_y(line["xy"][1]))
                if arrowprops:
                    arrow_color = (
                        arrowprops.get("color")
                        if isinstance(arrowprops, dict)
                        else kwargs.get("color")
                    )
                    ax.plot(
                        [text_x, xy_x],
                        [text_y, xy_y],
                        color=arrow_color,
                    )
                ax.text(
                    line["text"],
                    text_x,
                    text_y,
                    **self._plotext_text_kwargs(kwargs),
                )
            elif plot_type == "text":
                ax.text(
                    line["s"],
                    self._transform_scalar_x(line["x"]),
                    self._transform_scalar_y(line["y"]),
                    **self._plotext_text_kwargs(kwargs),
                )
            elif plot_type == "imshow":
                unsupported = set(kwargs) - {"marker", "style", "fast"}
                if unsupported:
                    unsupported_str = ", ".join(sorted(unsupported))
                    raise NotImplementedError(
                        f"plotext backend does not support imshow kwargs: {unsupported_str}"
                    )
                ax.matrix_plot(
                    np.asarray(line["data"]).tolist(),
                    marker=kwargs.get("marker"),
                    style=kwargs.get("style"),
                    fast=kwargs.get("fast", False),
                )
                last_image_data = line["data"]
            elif plot_type == "patch":
                patch_color = self._plotext_draw_patch(ax, line["patch"], kwargs)
                patch_label = kwargs.get("label")
                if patch_label is None and hasattr(line["patch"], "get_label"):
                    candidate = line["patch"].get_label()
                    if candidate and not str(candidate).startswith("_"):
                        patch_label = candidate
                legend_entries.append((patch_label, patch_color))
            elif plot_type == "colorbar":
                colorbar_notes.append(
                    self._plotext_colorbar_note(last_image_data, line.get("label"))
                )
            elif plot_type == "axhline":
                ax.horizontal_line(
                    self._transform_scalar_y(line["y"]),
                    color=kwargs.get("color"),
                )
            elif plot_type == "axvline":
                ax.vertical_line(
                    self._transform_scalar_x(line["x"]),
                    color=kwargs.get("color"),
                )
            else:
                raise NotImplementedError(
                    f"plotext backend does not support plot type: {plot_type}"
                )

        self._plotext_apply_aspect(ax, layers=layers)
        title_parts = [
            part for part in [self._title, getattr(self, "_caption", None)] if part
        ]
        if colorbar_notes:
            title_parts.extend(colorbar_notes)
        if title_parts:
            ax.title(" | ".join(title_parts))
        if self._xlabel:
            ax.xlabel(self._xlabel)
        if self._ylabel:
            ax.ylabel(self._ylabel)
        if self._grid:
            ax.grid(True, True)
        if self.xmin is not None or self.xmax is not None:
            ax.xlim(
                self._plotext_axis_limit(self.xmin, "x"),
                self._plotext_axis_limit(self.xmax, "x"),
            )
        if self.ymin is not None or self.ymax is not None:
            ax.ylim(
                self._plotext_axis_limit(self.ymin, "y"),
                self._plotext_axis_limit(self.ymax, "y"),
            )
        if self._xaxis_scale is not None:
            if self._xaxis_scale not in {"linear", "log", "symlog"}:
                raise NotImplementedError(
                    f"plotext backend does not support xscale={self._xaxis_scale!r}"
                )
            if self._xaxis_scale == "log":
                ax.xscale("log")
        if self._yaxis_scale is not None:
            if self._yaxis_scale not in {"linear", "log", "symlog"}:
                raise NotImplementedError(
                    f"plotext backend does not support yscale={self._yaxis_scale!r}"
                )
            if self._yaxis_scale == "log":
                ax.yscale("log")
        if self._xticks is not None:
            if self._xaxis_scale == "symlog":
                positions, labels = self._plotext_symlog_ticks(
                    "x",
                    self._plotext_axis_limit(self.xmin, "x"),
                    self._plotext_axis_limit(self.xmax, "x"),
                    ticks=self._xticks,
                    labels=self._xticklabels,
                )
                ax.xticks(positions, labels)
            else:
                ax.xticks(self._transform_x(self._xticks).tolist(), self._xticklabels)
        elif self._xaxis_scale == "symlog":
            positions, labels = self._plotext_symlog_ticks(
                "x",
                self._plotext_axis_limit(self.xmin, "x"),
                self._plotext_axis_limit(self.xmax, "x"),
            )
            if positions is not None:
                ax.xticks(positions, labels)
        if self._yticks is not None:
            if self._yaxis_scale == "symlog":
                positions, labels = self._plotext_symlog_ticks(
                    "y",
                    self._plotext_axis_limit(self.ymin, "y"),
                    self._plotext_axis_limit(self.ymax, "y"),
                    ticks=self._yticks,
                    labels=self._yticklabels,
                )
                ax.yticks(positions, labels)
            else:
                ax.yticks(self._transform_y(self._yticks).tolist(), self._yticklabels)
        elif self._yaxis_scale == "symlog":
            positions, labels = self._plotext_symlog_ticks(
                "y",
                self._plotext_axis_limit(self.ymin, "y"),
                self._plotext_axis_limit(self.ymax, "y"),
            )
            if positions is not None:
                ax.yticks(positions, labels)
        if self._legend:
            self._plotext_add_legend(ax, legend_entries, layers=layers)

    @property
    def xmin(self):
        return self._xmin

    @property
    def xmax(self):
        return self._xmax

    @property
    def ymin(self):
        return self._ymin

    @property
    def ymax(self):
        return self._ymax

    # Getter and Setter for grid
    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value

    # Getter and Setter for legend
    @property
    def legend(self):
        return self._legend

    @legend.setter
    def legend(self, value):
        self._legend = value


if __name__ == "__main__":
    plotter = LinePlot(xlabel="x", ylabel="y", title="Example", legend=True)
    plotter.plot([0, 1, 2, 3], [0, 1, 4, 9], label="Line 1")
    plotter.plot(
        [0, 1, 2, 3], [0, 2, 3, 6], linestyle="dashed", color="red", label="Line 2"
    )
    plotter.scatter([0, 1, 2, 3], [0, 0.5, 2, 5], label="Scatter")
