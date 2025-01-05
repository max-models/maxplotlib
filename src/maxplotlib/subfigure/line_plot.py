import numpy as np
import plotly.graph_objects as go

import maxplotlib.subfigure.tikz_figure as tf

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
    def __init__(self, **kwargs):
        """
        Initialize the LinePlot class for a subplot.

        Parameters:
        **kwargs: Arbitrary keyword arguments.
            - figsize (tuple): Figure size (default is (10, 6)).
            - caption (str): Caption for the plot.
            - description (str): Description of the plot.
            - label (str): Label for the plot.
            - grid (bool): Whether to display grid lines (default is False).
            TODO: Add all options
        """
        # Set default values
        self._figsize = kwargs.get("figsize", (10, 6))
        self._title = kwargs.get("title", None)
        self._caption = kwargs.get("caption", None)
        self._description = kwargs.get("description", None)
        self._label = kwargs.get("label", None)
        self._grid = kwargs.get("grid", False)
        self._legend = kwargs.get("legend", True)

        self._xlabel = kwargs.get("xlabel", None)
        self._ylabel = kwargs.get("ylabel", None)
        # List to store line data, each entry contains x and y data, label, and plot kwargs
        self.line_data = []
        self.layered_line_data = {}

        # Initialize lists to hold Node and Path objects
        self.nodes = []
        self.paths = []
        #self.layers = {}

        # Counter for unnamed nodes
        self._node_counter = 0

        # Scaling
        self._xscale = kwargs.get("xscale", 1.0)
        self._yscale = kwargs.get("yscale", 1.0)
        self._xshift = kwargs.get("xshift", 0.0)
        self._yshift = kwargs.get("yshift", 0.0)

    def add_caption(self, caption):
        self._caption = caption

    def add_line(self, x_data, y_data, layer=0, plot_type='plot', **kwargs):
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
        self.line_data.append(ld)
        if layer in self.layered_line_data:
            self.layered_line_data[layer].append(ld)
        else:
            self.layered_line_data[layer] = [ld]

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
            # if self._caption:
            #     ax.set_title(self._caption)
            if self._title:
                ax.set_title(self._title)
            if self._label:
                ax.set_ylabel(self._label)
            if self._xlabel:
                ax.set_xlabel(self._xlabel)
            if self._ylabel:
                ax.set_ylabel(self._ylabel)
            if self._legend and len(self.line_data) > 0:
                ax.legend()
            if self._grid:
                ax.grid()

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

    # Getter and Setter for figsize
    @property
    def figsize(self):
        return self._figsize

    @figsize.setter
    def figsize(self, value):
        self._figsize = value

    # Getter and Setter for caption
    @property
    def caption(self):
        return self._caption

    @caption.setter
    def caption(self, value):
        self._caption = value

    # Getter and Setter for description
    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    # Getter and Setter for label
    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

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
