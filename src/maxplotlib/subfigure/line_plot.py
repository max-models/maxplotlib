import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.axes_grid1 import make_axes_locatable

import maxplotlib.subfigure.tikz_figure as tf
from maxplotlib.objects.layer import Tikzlayer


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
        self, nodes, path_actions=[], cycle=False, label="", layer=0, **kwargs
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
        x_data,
        y_data,
        layer=0,
        plot_type="plot",
        **kwargs,
    ):
        """
        Add a line to the plot.

        Parameters:
        label (str): Label for the line.
        x_data (list): X-axis data.
        y_data (list): Y-axis data.
        **kwargs: Additional keyword arguments for the plot (e.g., color, linestyle).
        """
        ld = {
            "x": np.array(x_data),
            "y": np.array(y_data),
            "layer": layer,
            "plot_type": plot_type,
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def add_imshow(self, data, layer=0, plot_type="imshow", **kwargs):
        ld = {
            "data": np.array(data),
            "layer": layer,
            "plot_type": plot_type,
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def add_patch(self, patch, layer=0, plot_type="patch", **kwargs):
        ld = {
            "patch": patch,
            "layer": layer,
            "plot_type": plot_type,
            "kwargs": kwargs,
        }
        self._add(ld, layer)

    def add_colorbar(self, label="", layer=0, plot_type="colorbar", **kwargs):
        cb = {
            "label": label,
            "layer": layer,
            "plot_type": plot_type,
            "kwargs": kwargs,
        }
        self._add(cb, layer)

    @property
    def layers(self):
        layers = []
        for layer_name, layer_lines in self.layered_line_data.items():
            layers.append(layer_name)
        return layers

    def plot_matplotlib(self, ax, layers=None):
        """
        Plot all lines on the provided axis.

        Parameters:
        ax (matplotlib.axes.Axes): Axis on which to plot the lines.
        """
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
            if self.xmin:
                ax.axis(xmin=self.xmin)
            if self.xmax:
                ax.axis(xmax=self.xmax)
            if self.ymin:
                ax.axis(ymin=self.ymin)
            if self.ymax:
                ax.axis(ymax=self.ymax)

    def plot_plotly(self):
        """
        Plot all lines using Plotly and return a list of traces for each line.
        """
        # Mapping Matplotlib linestyles to Plotly dash styles
        linestyle_map = {
            "solid": "solid",
            "dashed": "dash",
            "dotted": "dot",
            "dashdot": "dashdot",
        }

        traces = []
        for line in self.line_data:
            trace = go.Scatter(
                x=(line["x"] + self._xshift) * self._xscale,
                y=(line["y"] + self._yshift) * self._yscale,
                mode="lines+markers" if "marker" in line["kwargs"] else "lines",
                name=line["kwargs"].get("label", ""),
                line=dict(
                    color=line["kwargs"].get("color", None),
                    dash=linestyle_map.get(
                        line["kwargs"].get("linestyle", "solid"), "solid"
                    ),
                ),
            )
            traces.append(trace)

        return traces

    def add_node(self, x, y, label=None, content="", layer=0, **kwargs):
        """
        Add a node to the TikZ figure.

        Parameters:
        - x (float): X-coordinate of the node.
        - y (float): Y-coordinate of the node.
        - label (str, optional): Label of the node. If None, a default label will be assigned.
        - **kwargs: Additional TikZ node options (e.g., shape, color).

        Returns:
        - node (Node): The Node object that was added.
        """
        if label is None:
            label = f"node{self._node_counter}"
        node = Node(x=x, y=y, label=label, layer=layer, content=content, **kwargs)
        self.nodes.append(node)
        if layer in self.layers:
            self.layers[layer].add(node)
        else:
            # print(f"{self.layers = } {layer = }")
            self.layers[layer] = Tikzlayer(layer)
            self.layers[layer].add(node)
        self._node_counter += 1
        return node

    def add_path(self, nodes, layer=0, **kwargs):
        """
        Add a line or path connecting multiple nodes.

        Parameters:
        - nodes (list of str): List of node names to connect.
        - **kwargs: Additional TikZ path options (e.g., style, color).

        Examples:
        - add_path(['A', 'B', 'C'], color='blue')
          Connects nodes A -> B -> C with a blue line.
        """
        if not isinstance(nodes, list):
            raise ValueError("nodes parameter must be a list of node names.")

        nodes = [
            (
                node
                if isinstance(node, Node)
                else (
                    self.get_node(node)
                    if isinstance(node, str)
                    else ValueError(f"Invalid node type: {type(node)}")
                )
            )
            for node in nodes
        ]
        path = Path(nodes, **kwargs)
        self.paths.append(path)
        if layer in self.layers:
            self.layers[layer].add(path)
        else:
            self.layers[layer] = Tikzlayer(layer)
            self.layers[layer].add(path)
        return path

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
    plotter = LinePlot()
    plotter.add_line("Line 1", [0, 1, 2, 3], [0, 1, 4, 9])
    plotter.add_line("Line 2", [0, 1, 2, 3], [0, 2, 3, 6])
    latex_code = plotter.generate_latex_plot()
    with open("figures/latex_code.tex", "w") as f:
        f.write(latex_code)
    print(latex_code)
