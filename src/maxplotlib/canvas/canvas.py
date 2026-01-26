import os
import re
from typing import Dict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from tikzpics import TikzFigure

from maxplotlib.backends.matplotlib.utils import (
    set_size,
    setup_plotstyle,
    setup_tex_fonts,
)
from maxplotlib.colors.colors import Color
from maxplotlib.linestyle.linestyle import Linestyle
from maxplotlib.subfigure.line_plot import LinePlot
from maxplotlib.utils.options import Backends


class Canvas:
    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple | None = None,
        caption: str | None = None,
        description: str | None = None,
        label: str | None = None,
        fontsize: int = 14,
        dpi: int = 300,
        width: str = "17cm",
        ratio: str = "golden",  # TODO Add literal
        gridspec_kw: Dict = {"wspace": 0.08, "hspace": 0.1},
    ):
        """
        Initialize the Canvas class for multiple subplots.

        Parameters:
        nrows (int): Number of subplot rows. Default is 1.
        ncols (int): Number of subplot columns. Default is 1.
        figsize (tuple): Figure size.
        caption (str): Caption for the figure.
        description (str): Description for the figure.
        label (str): Label for the figure.
        fontsize (int): Font size. Default is 14.
        dpi (int): DPI for the figure. Default is 300.
        width (str): Width of the figure. Default is "17cm".
        ratio (str): Aspect ratio. Default is "golden".
        gridspec_kw (dict): Gridspec keyword arguments. Default is {"wspace": 0.08, "hspace": 0.1}.
        """

        self._nrows = nrows
        self._ncols = ncols
        self._figsize = figsize
        self._caption = caption
        self._description = description
        self._label = label
        self._fontsize = fontsize
        self._dpi = dpi
        self._width = width
        self._ratio = ratio
        self._gridspec_kw = gridspec_kw
        self._plotted = False

        # Dictionary to store lines for each subplot
        # Key: (row, col), Value: list of lines with their data and kwargs
        self._subplots = {}
        self._num_subplots = 0

        self._subplot_matrix = [[None] * self.ncols for _ in range(self.nrows)]

    @property
    def subplots(self):
        return self._subplots

    @property
    def layers(self):
        layers = []
        for (row, col), subplot in self.subplots.items():
            layers.extend(subplot.layers)
        return list(set(layers))

    def generate_new_rowcol(self, row, col):
        if row is None:
            for irow in range(self.nrows):
                has_none = any(item is None for item in self._subplot_matrix[irow])
                if has_none:
                    row = irow
                    break
        assert row is not None, "Not enough rows!"

        if col is None:
            for icol in range(self.ncols):
                if self._subplot_matrix[row][icol] is None:
                    col = icol
                    break
        assert col is not None, "Not enough columns!"
        return row, col

    def add_line(
        self,
        x_data,
        y_data,
        layer=0,
        subplot: LinePlot | None = None,
        row: int | None = None,
        col: int | None = None,
        plot_type="plot",
        **kwargs,
    ):
        if row is not None and col is not None:
            try:
                subplot = self._subplot_matrix[row][col]
            except KeyError:
                raise ValueError("Invalid subplot position.")
        else:
            row, col = 0, 0
            subplot = self._subplot_matrix[row][col]

        if subplot is None:
            row, col = self.generate_new_rowcol(row, col)
            subplot = self.add_subplot(col=col, row=row)

        subplot.add_line(
            x_data=x_data,
            y_data=y_data,
            layer=layer,
            plot_type=plot_type,
            **kwargs,
        )

    def add_tikzfigure(
        self,
        col=None,
        row=None,
        label=None,
        **kwargs,
    ):
        """
        Adds a subplot to the figure.

        Parameters:
        **kwargs: Arbitrary keyword arguments.
        """

        row, col = self.generate_new_rowcol(row, col)

        # Initialize the LinePlot for the given subplot position
        tikz_figure = TikzFigure(
            label=label,
            **kwargs,
        )
        self._subplot_matrix[row][col] = tikz_figure

        # Store the LinePlot instance by its position for easy access
        if label is None:
            self._subplots[(row, col)] = tikz_figure
        else:
            self._subplots[label] = tikz_figure
        return tikz_figure

    def add_subplot(
        self,
        col: int | None = None,
        row: int | None = None,
        figsize: tuple = (10, 6),
        title: str | None = None,
        caption: str | None = None,
        description: str | None = None,
        label: str | None = None,
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
        Adds a subplot to the figure.

        Parameters:
        **kwargs: Arbitrary keyword arguments.
            - col (int): Column index for the subplot.
            - row (int): Row index for the subplot.
            - label (str): Label to identify the subplot.
        """

        row, col = self.generate_new_rowcol(row, col)

        # Initialize the LinePlot for the given subplot position
        line_plot = LinePlot(
            title=title,
            grid=grid,
            legend=legend,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            xlabel=xlabel,
            ylabel=ylabel,
            xscale=xscale,
            yscale=yscale,
            xshift=xshift,
            yshift=yshift,
        )
        self._subplot_matrix[row][col] = line_plot

        # Store the LinePlot instance by its position for easy access
        if label is None:
            self._subplots[(row, col)] = line_plot
        else:
            self._subplots[label] = line_plot
        return line_plot

    def savefig(
        self,
        filename,
        backend: Backends = "matplotlib",
        layers: list | None = None,
        layer_by_layer: bool = False,
        verbose: bool = False,
    ):
        filename_no_extension, extension = os.path.splitext(filename)
        if backend == "matplotlib":
            if layer_by_layer:
                layers = []
                for layer in self.layers:
                    layers.append(layer)
                    fig, axs = self.plot(
                        show=False,
                        backend="matplotlib",
                        savefig=True,
                        layers=layers,
                    )
                    _fn = f"{filename_no_extension}_{layers}.{extension}"
                    fig.savefig(_fn)
                    print(f"Saved {_fn}")
            else:
                if layers is None:
                    layers = self.layers
                    full_filepath = filename
                else:
                    full_filepath = f"{filename_no_extension}_{layers}.{extension}"

                if self._plotted:
                    self._matplotlib_fig.savefig(full_filepath)
                else:

                    fig, axs = self.plot(
                        backend="matplotlib",
                        savefig=True,
                        layers=layers,
                    )
                    fig.savefig(full_filepath)
                if verbose:
                    print(f"Saved {full_filepath}")

    def plot(
        self,
        backend: Backends = "matplotlib",
        savefig=False,
        layers=None,
        verbose: bool = False,
    ):
        if verbose:
            print(f"Plotting figure using backend: {backend}")

        if backend == "matplotlib":
            return self.plot_matplotlib(
                savefig=savefig,
                layers=layers,
                verbose=verbose,
            )
        elif backend == "plotly":
            return self.plot_plotly(savefig=savefig)
        elif backend == "tikzpics":
            return self.plot_tikzpics(savefig=savefig)
        else:
            raise ValueError(f"Invalid backend: {backend}")

    def show(
        self,
        backend: Backends = "matplotlib",
        verbose: bool = False,
    ):
        if verbose:
            print(f"Showing figure using backend: {backend}")

        if backend == "matplotlib":
            self.plot(
                backend="matplotlib",
                savefig=False,
                layers=None,
                verbose=verbose,
            )
            # self._matplotlib_fig.show()
        elif backend == "plotly":
            self.plot_plotly(savefig=False)
        elif backend == "tikzpics":
            fig = self.plot_tikzpics(savefig=False)
            fig.show()
        else:
            raise ValueError("Invalid backend")

    def plot_matplotlib(
        self,
        savefig: bool = False,
        layers: list | None = None,
        usetex: bool = False,
        verbose: bool = False,
    ):
        """
        Generate and optionally display the subplots.

        Parameters:
        filename (str, optional): Filename to save the figure.
        """
        if verbose:
            print("Generating Matplotlib figure...")

        tex_fonts = setup_tex_fonts(fontsize=self.fontsize, usetex=usetex)

        setup_plotstyle(
            tex_fonts=tex_fonts,
            axes_grid=True,
            axes_grid_which="major",
            grid_alpha=1.0,
            grid_linestyle="dotted",
        )
        if verbose:
            print("Plot style set up.")
            print(f"{self._figsize = } {self._width = } {self._ratio = }")
        if self._figsize is not None:
            fig_width, fig_height = self._figsize
        else:
            fig_width, fig_height = set_size(
                width=self._width,
                ratio=self._ratio,
                dpi=self.dpi,
                verbose=verbose,
            )
        if verbose:
            print(f"Figure size: {fig_width} x {fig_height} points")

        fig, axes = plt.subplots(
            self.nrows,
            self.ncols,
            figsize=(fig_width, fig_height),
            squeeze=False,
            dpi=self.dpi,
        )

        for (row, col), subplot in self.subplots.items():
            ax = axes[row][col]
            if isinstance(subplot, TikzFigure):
                plot_matplotlib(subplot, ax, layers=layers)
            else:
                subplot.plot_matplotlib(ax, layers=layers)
            # ax.set_title(f"Subplot ({row}, {col})")
            ax.grid()

        # Set caption, labels, etc., if needed
        self._plotted = True
        self._matplotlib_fig = fig
        self._matplotlib_axes = axes
        return fig, axes

    def plot_tikzpics(
        self,
        savefig=None,
        verbose=False,
    ) -> TikzFigure:
        if len(self.subplots) > 1:
            raise NotImplementedError(
                "Only one subplot is supported for tikzpics backend."
            )
        for (row, col), line_plot in self.subplots.items():
            if verbose:
                print(f"Plotting subplot at row {row}, col {col}")
                print(f"{line_plot = }")
            tikz_subplot = line_plot.plot_tikzpics(verbose=verbose)
        return tikz_subplot

    def plot_plotly(self, show=True, savefig=None, usetex=False):
        """
        Generate and optionally display the subplots using Plotly.

        Parameters:
        show (bool): Whether to display the plot.
        savefig (str, optional): Filename to save the figure if provided.
        """

        setup_tex_fonts(
            fontsize=self.fontsize,
            usetex=usetex,
        )  # adjust or redefine for Plotly if needed

        # Set default width and height if not specified
        if self._figsize is not None:
            fig_width, fig_height = self._figsize
        else:
            fig_width, fig_height = set_size(
                width=self._width,
                ratio=self._ratio,
            )
        # print(self._width, fig_width, fig_height)
        # Create subplots
        fig = make_subplots(
            rows=self.nrows,
            cols=self.ncols,
            subplot_titles=[
                f"Subplot ({row}, {col})" for (row, col) in self.subplots.keys()
            ],
        )

        # Plot each subplot
        for (row, col), line_plot in self.subplots.items():
            traces = line_plot.plot_plotly()  # Generate Plotly traces for the line_plot
            for trace in traces:
                fig.add_trace(trace, row=row + 1, col=col + 1)

        # Update layout settings
        fig.update_layout(
            # width=fig_width,
            # height=fig_height,
            font=dict(size=self.fontsize),
            margin=dict(l=10, r=10, t=40, b=10),  # Adjust margins if needed
        )

        # Optionally save the figure
        if savefig:
            fig.write_image(savefig)

        # Show or return the figure
        # if show:
        #     fig.show()
        return fig

    # Property getters

    @property
    def dpi(self):
        return self._dpi

    @property
    def fontsize(self):
        return self._fontsize

    @property
    def nrows(self):
        return self._nrows

    @property
    def ncols(self):
        return self._ncols

    @property
    def caption(self):
        return self._caption

    @property
    def description(self):
        return self._description

    @property
    def label(self):
        return self._label

    @property
    def figsize(self):
        return self._figsize

    @property
    def subplot_matrix(self):
        return self._subplot_matrix

    # Property setters

    @nrows.setter
    def nrows(self, value):
        self._nrows = value

    @ncols.setter
    def ncols(self, value):
        self._ncols = value

    @caption.setter
    def caption(self, value):
        self._caption = value

    @description.setter
    def description(self, value):
        self._description = value

    @label.setter
    def label(self, value):
        self._label = value

    @figsize.setter
    def figsize(self, value):
        self._figsize = value

    # Magic methods
    def __str__(self):
        return f"Canvas(nrows={self.nrows}, ncols={self.ncols}, figsize={self.figsize})"

    def __repr__(self):
        return f"Canvas(nrows={self.nrows}, ncols={self.ncols}, caption={self.caption}, label={self.label})"

    def __getitem__(self, key):
        """Allows accessing subplots by tuple index."""
        row, col = key
        if row >= self.nrows or col >= self.ncols:
            raise IndexError("Subplot index out of range")
        return self._subplot_matrix[row][col]

    def __setitem__(self, key, value):
        """Allows setting a subplot by tuple index."""
        row, col = key
        if row >= self.nrows or col >= self.ncols:
            raise IndexError("Subplot index out of range")
        self._subplot_matrix[row][col] = value


def plot_matplotlib(tikzfigure: TikzFigure, ax, layers=None):
    """
    Plot all nodes and paths on the provided axis using Matplotlib.

    Parameters:
    - ax (matplotlib.axes.Axes): Axis on which to plot the figure.
    """

    # TODO: Specify which layers to retreive nodes from with layers=layers
    nodes = tikzfigure.layers.get_nodes()
    paths = tikzfigure.layers.get_paths()

    for path in paths:
        x_coords = [node.x for node in path.nodes]
        y_coords = [node.y for node in path.nodes]

        # Parse path color
        path_color_spec = path.kwargs.get("color", "black")
        try:
            color = Color(path_color_spec).to_rgb()
        except ValueError as e:
            print(e)
            color = "black"

        # Parse line width
        line_width_spec = path.kwargs.get("line_width", 1)
        if isinstance(line_width_spec, str):
            match = re.match(r"([\d.]+)(pt)?", line_width_spec)
            if match:
                line_width = float(match.group(1))
            else:
                print(
                    f"Invalid line width specification: '{line_width_spec}', defaulting to 1",
                )
                line_width = 1
        else:
            line_width = float(line_width_spec)

        # Parse line style using Linestyle class
        style_spec = path.kwargs.get("style", "solid")
        linestyle = Linestyle(style_spec).to_matplotlib()

        ax.plot(
            x_coords,
            y_coords,
            color=color,
            linewidth=line_width,
            linestyle=linestyle,
            zorder=1,  # Lower z-order to place behind nodes
        )

    # Plot nodes after paths so they appear on top
    for node in nodes:
        # Determine shape and size
        shape = node.kwargs.get("shape", "circle")
        fill_color_spec = node.kwargs.get("fill", "white")
        edge_color_spec = node.kwargs.get("draw", "black")
        linewidth = float(node.kwargs.get("line_width", 1))
        size = float(node.kwargs.get("size", 1))

        # Parse colors using the Color class
        try:
            facecolor = Color(fill_color_spec).to_rgb()
        except ValueError as e:
            print(e)
            facecolor = "white"

        try:
            edgecolor = Color(edge_color_spec).to_rgb()
        except ValueError as e:
            print(e)
            edgecolor = "black"

        # Plot shapes
        if shape == "circle":
            radius = size / 2
            circle = patches.Circle(
                (node.x, node.y),
                radius,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=2,  # Higher z-order to place on top of paths
            )
            ax.add_patch(circle)
        elif shape == "rectangle":
            width = height = size
            rect = patches.Rectangle(
                (node.x - width / 2, node.y - height / 2),
                width,
                height,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=2,  # Higher z-order
            )
            ax.add_patch(rect)
        else:
            # Default to circle if shape is unknown
            radius = size / 2
            circle = patches.Circle(
                (node.x, node.y),
                radius,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=2,
            )
            ax.add_patch(circle)

        # Add text inside the shape
        if node.content:
            ax.text(
                node.x,
                node.y,
                node.content,
                fontsize=10,
                ha="center",
                va="center",
                wrap=True,
                zorder=3,  # Even higher z-order for text
            )

    # Remove axes, ticks, and legend
    ax.axis("off")

    # Adjust plot limits
    all_x = [node.x for node in nodes]
    all_y = [node.y for node in nodes]
    padding = 1  # Adjust padding as needed
    ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    ax.set_aspect("equal", adjustable="datalim")


if __name__ == "__main__":
    c = Canvas(ncols=2, nrows=2)
    sp = c.add_subplot()
    sp.add_line("Line 1", [0, 1, 2, 3], [0, 1, 4, 9])
    c.plot()
    print("done")
