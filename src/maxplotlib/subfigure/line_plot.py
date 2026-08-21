import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tikzfigure import TikzFigure


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
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._xscale = xscale
        self._yscale = yscale
        self._xshift = xshift
        self._yshift = yshift

        # Axis scale type ('linear', 'log', 'symlog')
        self._xaxis_scale: str | None = None
        self._yaxis_scale: str | None = None

        # Custom tick positions and labels
        self._xticks: list | None = None
        self._xticklabels: list | None = None
        self._xtick_kwargs: dict = {}
        self._yticks: list | None = None
        self._yticklabels: list | None = None
        self._ytick_kwargs: dict = {}

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
            (e.g., color, width, label).
        """
        ld = {
            "x": np.array(x),
            "height": np.array(height),
            "layer": layer,
            "plot_type": "bar",
            "kwargs": kwargs,
        }
        self._add(ld, layer)

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
            (e.g., colormap, edgecolor, label).
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

    def set_xlabel(self, label: str):
        """Set the x-axis label."""
        self._xlabel = label

    def set_ylabel(self, label: str):
        """Set the y-axis label."""
        self._ylabel = label

    def set_title(self, title: str):
        """Set the subplot title."""
        self._title = title

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

    def set_legend(self, visible: bool = True):
        """Show or hide the legend."""
        self._legend = visible

    def set_xscale(self, scale: str):
        """Set the x-axis scale type: 'linear', 'log', or 'symlog'."""
        self._xaxis_scale = scale

    def set_yscale(self, scale: str):
        """Set the y-axis scale type: 'linear', 'log', or 'symlog'."""
        self._yaxis_scale = scale

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
        for layer_name, layer_lines in self.layered_line_data.items():
            if layers and layer_name not in layers:
                continue
            for line in layer_lines:
                if line["plot_type"] == "plot":
                    ax.plot(
                        (line["x"] + self._xshift) * self._xscale,
                        (line["y"] + self._yshift) * self._yscale,
                        **line["kwargs"],
                    )
                elif line["plot_type"] == "scatter":
                    ax.scatter(
                        (line["x"] + self._xshift) * self._xscale,
                        (line["y"] + self._yshift) * self._yscale,
                        **line["kwargs"],
                    )
                elif line["plot_type"] == "bar":
                    ax.bar(
                        (line["x"] + self._xshift) * self._xscale,
                        line["height"] * self._yscale,
                        **line["kwargs"],
                    )
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

                    colormap = line["kwargs"].get("colormap", "viridis")
                    cmap = plt.get_cmap(colormap)
                    max_depth = depths.max() + 1

                    for i in range(n):
                        color = (
                            cmap(depths[i] / max_depth) if max_depth > 1 else cmap(0.5)
                        )
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

        if self._title:
            ax.set_title(self._title)
        if self._xlabel:
            ax.set_xlabel(self._xlabel)
        if self._ylabel:
            ax.set_ylabel(self._ylabel)
        if self._legend and len(self.line_data) > 0:
            ax.legend()
        if self._grid:
            ax.grid()
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
        if self._aspect is not None:
            ax.set_aspect(self._aspect)

    def plot_tikzfigure(self, layers=None, verbose: bool = False) -> TikzFigure:

        tikz_figure = TikzFigure()
        for layer_name, layer_lines in self.layered_line_data.items():
            if layers and layer_name not in layers:
                continue
            for line in layer_lines:
                if line["plot_type"] == "plot":
                    x = (line["x"] + self._xshift) * self._xscale
                    y = (line["y"] + self._yshift) * self._yscale

                    nodes = [[xi, yi] for xi, yi in zip(x, y)]
                    tikz_figure.draw(nodes=nodes, **line["kwargs"])
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
                    colors = ["red", "blue", "green", "orange", "purple", "cyan"]

                    for i in range(n):
                        x_start = start_times[i]
                        x_end = start_times[i] + values[i]
                        y_pos = depths[i]
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
                                if k != "colormap"
                            },
                        )
        if verbose:
            print("Generated TikZ figure:")
            print(tikz_figure.generate_tikz())
        return tikz_figure

    def plot_plotly(self, layers=None):
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
        # Plotly shapes (unlike traces) don't participate in axis autorange,
        # so patches would otherwise be clipped or invisible unless the caller
        # sets explicit axis limits. Track each patch's bounding box here and
        # add one invisible marker trace at the end so autorange sees them.
        patch_bounds_x: list[float] = []
        patch_bounds_y: list[float] = []

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
            return value

        for line in self._iter_layer_lines(layers=layers):
            plot_type = line["plot_type"]
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
                trace = go.Scatter(
                    x=tx(line["x"]),
                    y=ty(line["y"]),
                    mode="markers",
                    name=kwargs.get("label", ""),
                    showlegend=bool(kwargs.get("label")) and bool(self._legend),
                    marker=dict(
                        color=plotly_color(kwargs.get("color", None)),
                        symbol=marker_map.get(marker, marker),
                        size=kwargs.get("s", None),
                    ),
                )
                traces.append(trace)
            elif plot_type == "bar":
                kwargs = line["kwargs"]
                trace = go.Bar(
                    x=tx(line["x"]),
                    y=np.asarray(line["height"]) * self._yscale,
                    name=kwargs.get("label", ""),
                    showlegend=bool(kwargs.get("label")) and bool(self._legend),
                    marker_color=plotly_color(kwargs.get("color", None)),
                )
                traces.append(trace)
            elif plot_type == "gantt":
                kwargs = line["kwargs"]
                tasks = line["tasks"]
                start_times = tx(line["start_times"])
                durations = np.asarray(line["durations"]) * self._xscale
                y_positions = list(range(len(tasks)))
                trace = go.Bar(
                    x=durations,
                    y=y_positions,
                    base=start_times,
                    orientation="h",
                    name=kwargs.get("label", ""),
                    showlegend=bool(kwargs.get("label")) and bool(self._legend),
                    marker_color=plotly_color(kwargs.get("color", None)),
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

                # Create rectangles as shapes
                colormap = kwargs.get("colormap", "Viridis")
                import plotly.express as px

                colors = px.colors.sample_colorscale(
                    colormap, np.linspace(0, 1, depths.max() + 1)
                )

                for i in range(n):
                    color = colors[depths[i]] if depths.max() > 0 else colors[0]
                    shapes.append(
                        dict(
                            type="rect",
                            x0=float(start_times[i]),
                            x1=float(start_times[i] + values[i]),
                            y0=float(depths[i]),
                            y1=float(depths[i] + 0.9),
                            fillcolor=color,
                            line=dict(
                                color=kwargs.get("edgecolor", "black"), width=0.5
                            ),
                        )
                    )

                    # Add text annotation if wide enough
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
                        dict(type="data", array=yerr, visible=True)
                        if yerr is not None
                        else None
                    ),
                    error_x=(
                        dict(type="data", array=xerr, visible=True)
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
            elif plot_type in ("text", "annotate"):
                kwargs = line["kwargs"]
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
                            font=dict(
                                color=plotly_color(kwargs.get("color", None)),
                                size=kwargs.get("fontsize", None),
                            ),
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
                        font=dict(
                            color=plotly_color(kwargs.get("color", None)),
                            size=kwargs.get("fontsize", None),
                        ),
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

                hovertext = kwargs.get("hovertext")

                def _add_hover_trace(x_pts, y_pts, hovertext=hovertext):
                    # Plotly shapes can't show hover info themselves, so an
                    # invisible filled polygon trace is overlaid on top of
                    # the shape's outline to make the whole area hoverable.
                    if hovertext is None:
                        return
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
                    cx = txs(patch.center[0])
                    cy = tys(patch.center[1])
                    rx = abs(txs(patch.center[0] + patch.width / 2.0) - cx)
                    ry = abs(tys(patch.center[1] + patch.height / 2.0) - cy)
                    # Ignore rotation for now; provides useful parity for tutorials.
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

        return traces, shapes, annotations

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
                xerr = self._coerce_numeric_array(line.get("xerr"))
                yerr = self._coerce_numeric_array(line.get("yerr"))
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
        if np.isscalar(error):
            return [float(error)] * count
        return np.asarray(error).tolist()

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

                # Use different colors for different depths
                colormap = kwargs.get("colormap", "viridis")
                max_depth = int(depths.max()) + 1

                for i in range(n):
                    # Simple color cycling based on depth
                    depth_colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
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
