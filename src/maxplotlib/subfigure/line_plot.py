import numpy as np
import plotly.graph_objects as go


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
        self._caption = kwargs.get("caption", None)
        self._description = kwargs.get("description", None)
        self._label = kwargs.get("label", None)
        self._grid = kwargs.get("grid", False)
        self._legend = kwargs.get("legend", True)

        self._xlabel = kwargs.get("xlabel", None)
        self._ylabel = kwargs.get("ylabel", None)
        # List to store line data, each entry contains x and y data, label, and plot kwargs
        self.line_data = []

        # Scaling
        self._xscale = kwargs.get("xscale", 1.0)
        self._yscale = kwargs.get("yscale", 1.0)
        self._xshift = kwargs.get("xshift", 0.0)
        self._yshift = kwargs.get("yshift", 0.0)

    def add_caption(self, caption):
        self._caption = caption

    def add_line(self, x_data, y_data, **kwargs):
        """
        Add a line to the plot.

        Parameters:
        label (str): Label for the line.
        x_data (list): X-axis data.
        y_data (list): Y-axis data.
        **kwargs: Additional keyword arguments for the plot (e.g., color, linestyle).
        """
        self.line_data.append(
            {"x": np.array(x_data), "y": np.array(y_data), "kwargs": kwargs}
        )

    def plot_matplotlib(self, ax):
        """
        Plot all lines on the provided axis.

        Parameters:
        ax (matplotlib.axes.Axes): Axis on which to plot the lines.
        """
        for line in self.line_data:
            ax.plot(
                (line["x"] + self._xshift) * self._xscale,
                (line["y"] + self._yshift) * self._yscale,
                **line["kwargs"],
            )
        if self._caption:
            ax.set_title(self._caption)
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
