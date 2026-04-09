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
                elif line["plot_type"] == "axhline":
                    ax.axhline(y=line["y"], **line["kwargs"])
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
        if verbose:
            print("Generated TikZ figure:")
            print(tikz_figure.generate_tikz())
        return tikz_figure

    def plot_plotly(self):
        """
        Plot all lines using Plotly and return a list of traces for each line.
        """
        linestyle_map = {
            "solid": "solid",
            "dashed": "dash",
            "dotted": "dot",
            "dashdot": "dashdot",
        }

        traces = []
        for line in self.line_data:
            plot_type = line["plot_type"]
            if plot_type == "plot":
                trace = go.Scatter(
                    x=(line["x"] + self._xshift) * self._xscale,
                    y=(line["y"] + self._yshift) * self._yscale,
                    mode="lines+markers" if "marker" in line["kwargs"] else "lines",
                    name=line["kwargs"].get("label", ""),
                    line=dict(
                        color=line["kwargs"].get("color", None),
                        dash=linestyle_map.get(
                            line["kwargs"].get("linestyle", "solid"),
                            "solid",
                        ),
                    ),
                )
                traces.append(trace)
            elif plot_type == "scatter":
                trace = go.Scatter(
                    x=(line["x"] + self._xshift) * self._xscale,
                    y=(line["y"] + self._yshift) * self._yscale,
                    mode="markers",
                    name=line["kwargs"].get("label", ""),
                    marker=dict(color=line["kwargs"].get("color", None)),
                )
                traces.append(trace)
            elif plot_type == "bar":
                trace = go.Bar(
                    x=(line["x"] + self._xshift) * self._xscale,
                    y=line["height"] * self._yscale,
                    name=line["kwargs"].get("label", ""),
                    marker_color=line["kwargs"].get("color", None),
                )
                traces.append(trace)
            elif plot_type in ("axhline", "axvline"):
                pass  # Rendered as shape annotations; no trace needed

        return traces

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
    plotter.plot([0, 1, 2, 3], [0, 2, 3, 6], linestyle="dashed", color="red", label="Line 2")
    plotter.scatter([0, 1, 2, 3], [0, 0.5, 2, 5], label="Scatter")
