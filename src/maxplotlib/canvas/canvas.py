import os
import re
from dataclasses import dataclass
from typing import Mapping

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from plotly.subplots import make_subplots
from tikzfigure import TikzFigure

from maxplotlib.backends.matplotlib.utils import (
    set_size,
    setup_plotstyle,
    setup_tex_fonts,
)
from maxplotlib.backends.plotext import PlotextFigure, create_plotext_figure
from maxplotlib.colors.colors import Color
from maxplotlib.linestyle.linestyle import Linestyle
from maxplotlib.subfigure.line_plot import LinePlot
from maxplotlib.utils.options import Backends


@dataclass(frozen=True)
class SubplotSpacing:
    """Typed spacing configuration for subplot grids."""

    wspace: float = 0.08
    hspace: float = 0.1

    def to_gridspec_kw(self) -> dict[str, float]:
        return {"wspace": self.wspace, "hspace": self.hspace}


def _parse_bool_env_var(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _display_matplotlib_figure_in_notebook(fig) -> bool:
    """Display a Matplotlib figure through IPython when running in Jupyter."""
    try:
        from IPython.display import display
    except ImportError:
        return False

    if not _running_in_jupyter():
        return False

    display(fig)
    return True


def _running_in_jupyter() -> bool:
    """Return whether the current process is running in a Jupyter kernel."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False

    shell = get_ipython()
    return shell is not None and "IPKernelApp" in getattr(shell, "config", {})


def _apply_matplotlib_customizations(fig, axes, customizations) -> None:
    """Apply declarative method calls to a Matplotlib figure and its axes."""
    if callable(customizations):
        customizations(fig, axes)
        return
    if not isinstance(customizations, Mapping):
        raise TypeError("matplotlib_customizations must be a mapping or callable")

    unknown_targets = set(customizations) - {"figure", "axes"}
    if unknown_targets:
        raise ValueError(
            "matplotlib_customizations only supports 'figure' and 'axes' targets; "
            f"got {sorted(unknown_targets)!r}"
        )

    for target, objects in (
        ("figure", (fig,)),
        ("axes", tuple(axes.flat)),
    ):
        for method_name, spec in customizations.get(target, {}).items():
            if not isinstance(method_name, str):
                raise TypeError("Matplotlib customization method names must be strings")
            if isinstance(spec, Mapping) and ("args" in spec or "kwargs" in spec):
                args = tuple(spec.get("args", ()))
                kwargs = dict(spec.get("kwargs", {}))
            elif isinstance(spec, Mapping):
                args = ()
                kwargs = dict(spec)
            elif spec is None:
                args = ()
                kwargs = {}
            else:
                args = (spec,)
                kwargs = {}

            for obj in objects:
                method = getattr(obj, method_name)
                method(*args, **kwargs)


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
                fontsize=self._fontsize,
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


class Canvas:
    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple | None = None,
        caption: str | None = None,
        description: str | None = None,
        label: str | None = None,
        fontsize: int = 10,
        dpi: int | None = None,
        width: str | None = None,
        ratio: str = "golden",  # TODO Add literal
        usetex: bool | None = None,
        subplot_spacing: SubplotSpacing | None = None,
        gridspec_kw: Mapping[str, float] | None = None,
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
        fontsize (int): Font size. Default is 10.
        dpi (int | None): Optional export/render DPI override.
        width (str | None): Optional figure width, e.g. "7cm".
        ratio (str): Aspect ratio. Default is "golden".
        usetex (bool | None): Default text.usetex behavior for this canvas.
            If None, read from MAXPLOTLIB_USETEX environment variable.
        subplot_spacing (SubplotSpacing): Typed subplot spacing.
            Default is SubplotSpacing(wspace=0.08, hspace=0.1).
        gridspec_kw (Mapping[str, float]): Optional matplotlib gridspec kwargs.
            Kept for compatibility with existing code.
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
        self._usetex = (
            _parse_bool_env_var("MAXPLOTLIB_USETEX", default=False)
            if usetex is None
            else usetex
        )
        if subplot_spacing is not None and gridspec_kw is not None:
            raise ValueError("Pass either subplot_spacing or gridspec_kw, not both.")
        if subplot_spacing is None and gridspec_kw is None:
            subplot_spacing = SubplotSpacing()
        if subplot_spacing is not None:
            self._gridspec_kw = subplot_spacing.to_gridspec_kw()
        else:
            self._gridspec_kw = dict(gridspec_kw)
        self._plotted = False
        self._matplotlib_fig = None
        self._matplotlib_axes = None
        self._plotext_figure = None
        self._suptitle: str | None = None
        self._suptitle_kwargs: dict = {}
        self._supxlabel: str | None = None
        self._supxlabel_kwargs: dict = {}
        self._supylabel: str | None = None
        self._supylabel_kwargs: dict = {}
        self._subplots_adjust_kwargs: dict = {}
        self._tight_layout_kwargs: dict | None = None

        # Dictionary to store lines for each subplot
        # Key: (row, col), Value: list of lines with their data and kwargs
        self._subplots = {}
        self._twinx_subplots = {}
        self._matplotlib_twin_axes = {}
        self._num_subplots = 0

        self._subplot_matrix = [[None] * self.ncols for _ in range(self.nrows)]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def subplots(
        cls,
        nrows: int = 1,
        ncols: int = 1,
        squeeze: bool = True,
        wspace: float | None = None,
        hspace: float | None = None,
        **canvas_kwargs,
    ):
        """
        Create a Canvas pre-filled with LinePlot subplots, mirroring
        ``matplotlib.pyplot.subplots()``.

        Parameters:
        nrows, ncols (int): Grid dimensions.
        squeeze (bool): If True, return a single subplot instead of a 1-element
            list when the grid is 1×1 or when one dimension is 1.
        wspace, hspace (float): Convenience subplot spacing arguments.
            These map to matplotlib gridspec spacing values.
        **canvas_kwargs: Forwarded to the Canvas constructor.

        Returns:
        (canvas, axes): A tuple of the Canvas and either a single LinePlot,
            a flat list (when one dimension is 1 and squeeze=True), or a
            2-D list of LinePlots.

        Examples:
        >>> canvas, ax = Canvas.subplots()
        >>> canvas, (ax1, ax2) = Canvas.subplots(ncols=2)
        >>> canvas, axes = Canvas.subplots(nrows=2, ncols=2)  # axes[row][col]
        """
        spacing_given = wspace is not None or hspace is not None
        if spacing_given and (
            "subplot_spacing" in canvas_kwargs or "gridspec_kw" in canvas_kwargs
        ):
            raise ValueError(
                "Use either wspace/hspace or subplot_spacing/gridspec_kw, not both."
            )
        if spacing_given:
            canvas_kwargs["subplot_spacing"] = SubplotSpacing(
                wspace=0.08 if wspace is None else wspace,
                hspace=0.1 if hspace is None else hspace,
            )

        canvas = cls(nrows=nrows, ncols=ncols, **canvas_kwargs)
        axes = [
            [canvas.add_subplot(row=r, col=c) for c in range(ncols)]
            for r in range(nrows)
        ]
        if squeeze:
            if nrows == 1 and ncols == 1:
                return canvas, axes[0][0]
            if nrows == 1:
                return canvas, axes[0]
            if ncols == 1:
                return canvas, [row[0] for row in axes]
        return canvas, axes

    @property
    def _subplot_dict(self):
        return self._subplots

    @property
    def layers(self):
        layers = []
        for (row, col), subplot in self._subplot_dict.items():
            layers.extend(subplot.layers)
            twin_subplot = self._twinx_subplots.get((row, col))
            if twin_subplot is not None:
                layers.extend(twin_subplot.layers)
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
        x,
        y,
        layer=0,
        subplot: LinePlot | None = None,
        row: int | None = None,
        col: int | None = None,
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
            x=x,
            y=y,
            layer=layer,
            **kwargs,
        )

    def _get_or_create_subplot(self, row, col):
        """Return the subplot at (row, col), creating it if needed."""
        if row is not None and col is not None:
            try:
                sp = self._subplot_matrix[row][col]
            except (IndexError, KeyError):
                raise ValueError("Invalid subplot position.")
        else:
            row, col = 0, 0
            sp = self._subplot_matrix[row][col]
        if sp is None:
            row, col = self.generate_new_rowcol(row, col)
            sp = self.add_subplot(col=col, row=row)
        return sp

    def scatter(
        self,
        x,
        y,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """
        Add a scatter plot to the canvas (matplotlib-style convenience method).

        Parameters:
        x (array-like): X-axis data.
        y (array-like): Y-axis data.
        layer (int): Layer index (default 0).
        row, col (int): Subplot position (default top-left).
        **kwargs: Forwarded to the backend (e.g., color, marker, s, label).
        """
        sp = self._get_or_create_subplot(row, col)
        sp.scatter(x, y, layer=layer, **kwargs)

    def bar(
        self,
        x,
        height,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """
        Add a bar chart to the canvas (matplotlib-style convenience method).

        Parameters:
        x (array-like): X positions of the bars.
        height (array-like): Heights of the bars.
        layer (int): Layer index (default 0).
        row, col (int): Subplot position (default top-left).
        **kwargs: Forwarded to the backend (e.g., color, width, label).
        """
        sp = self._get_or_create_subplot(row, col)
        sp.bar(x, height, layer=layer, **kwargs)

    def barh(
        self,
        y,
        width,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add a horizontal bar chart to a subplot."""
        self._get_or_create_subplot(row, col).barh(y, width, layer=layer, **kwargs)

    def hist(
        self,
        x,
        bins=10,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add a histogram to a subplot."""
        self._get_or_create_subplot(row, col).hist(x, bins=bins, layer=layer, **kwargs)

    def step(
        self, x, y, layer=0, row: int | None = None, col: int | None = None, **kwargs
    ):
        """Add a step plot to a subplot."""
        self._get_or_create_subplot(row, col).step(x, y, layer=layer, **kwargs)

    def stairs(
        self,
        values,
        edges=None,
        baseline=0,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add a stairs plot to a subplot."""
        self._get_or_create_subplot(row, col).stairs(
            values, edges=edges, baseline=baseline, layer=layer, **kwargs
        )

    def broken_barh(
        self,
        xranges,
        yrange,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add broken horizontal bars to a subplot."""
        self._get_or_create_subplot(row, col).broken_barh(
            xranges, yrange, layer=layer, **kwargs
        )

    def pie(self, x, layer=0, row: int | None = None, col: int | None = None, **kwargs):
        """Add a pie chart to a subplot."""
        self._get_or_create_subplot(row, col).pie(x, layer=layer, **kwargs)

    def bar_label(self, row: int | None = None, col: int | None = None, **kwargs):
        """Add labels to bar containers in the Matplotlib backend."""
        self._get_or_create_subplot(row, col).bar_label(**kwargs)

    def stem(
        self, x, y, layer=0, row: int | None = None, col: int | None = None, **kwargs
    ):
        """Add a stem plot to a subplot."""
        self._get_or_create_subplot(row, col).stem(x, y, layer=layer, **kwargs)

    def stackplot(
        self, x, *ys, layer=0, row: int | None = None, col: int | None = None, **kwargs
    ):
        """Add a stacked area plot to a subplot."""
        self._get_or_create_subplot(row, col).stackplot(x, *ys, layer=layer, **kwargs)

    def boxplot(
        self, x, layer=0, row: int | None = None, col: int | None = None, **kwargs
    ):
        """Add a box-and-whisker plot to a subplot."""
        self._get_or_create_subplot(row, col).boxplot(x, layer=layer, **kwargs)

    def violinplot(
        self,
        dataset,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add a violin plot to a subplot."""
        self._get_or_create_subplot(row, col).violinplot(dataset, layer=layer, **kwargs)

    def eventplot(
        self,
        positions,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add an event/rug plot to a subplot."""
        self._get_or_create_subplot(row, col).eventplot(
            positions, layer=layer, **kwargs
        )

    def contour(
        self,
        x,
        y,
        z,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add contour lines to a subplot."""
        self._get_or_create_subplot(row, col).contour(x, y, z, layer=layer, **kwargs)

    def contourf(
        self,
        x,
        y,
        z,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add filled contours to a subplot."""
        self._get_or_create_subplot(row, col).contourf(x, y, z, layer=layer, **kwargs)

    def pcolormesh(
        self,
        x,
        y,
        z,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add a pseudocolor mesh to a subplot."""
        self._get_or_create_subplot(row, col).pcolormesh(x, y, z, layer=layer, **kwargs)

    def hexbin(
        self,
        x,
        y,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add a hexagonal density plot to a subplot."""
        self._get_or_create_subplot(row, col).hexbin(x, y, layer=layer, **kwargs)

    def matshow(
        self, data, layer=0, row: int | None = None, col: int | None = None, **kwargs
    ):
        """Display a matrix with matrix-oriented axes."""
        self._get_or_create_subplot(row, col).matshow(data, layer=layer, **kwargs)

    def quiver(
        self,
        x,
        y,
        u,
        v,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add a vector field to a subplot."""
        self._get_or_create_subplot(row, col).quiver(x, y, u, v, layer=layer, **kwargs)

    def triplot(
        self,
        x,
        y,
        triangles=None,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add an unstructured triangular grid to a subplot."""
        self._get_or_create_subplot(row, col).triplot(
            x, y, triangles=triangles, layer=layer, **kwargs
        )

    def tripcolor(
        self,
        x,
        y,
        c,
        triangles=None,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add colored unstructured triangles to a subplot."""
        self._get_or_create_subplot(row, col).tripcolor(
            x, y, c, triangles=triangles, layer=layer, **kwargs
        )

    def tricontour(
        self,
        x,
        y,
        z,
        triangles=None,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add unstructured contour lines to a subplot."""
        self._get_or_create_subplot(row, col).tricontour(
            x, y, z, triangles=triangles, layer=layer, **kwargs
        )

    def tricontourf(
        self,
        x,
        y,
        z,
        triangles=None,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add filled unstructured contours to a subplot."""
        self._get_or_create_subplot(row, col).tricontourf(
            x, y, z, triangles=triangles, layer=layer, **kwargs
        )

    def streamplot(
        self,
        x,
        y,
        u,
        v,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add streamlines for a vector field to a subplot."""
        self._get_or_create_subplot(row, col).streamplot(
            x, y, u, v, layer=layer, **kwargs
        )

    def pcolor(
        self,
        x,
        y,
        z,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add a pseudocolor plot to a subplot."""
        self._get_or_create_subplot(row, col).pcolor(x, y, z, layer=layer, **kwargs)

    def pcolorfast(
        self,
        x,
        y,
        z,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add a fast pseudocolor plot to a subplot."""
        self._get_or_create_subplot(row, col).pcolorfast(x, y, z, layer=layer, **kwargs)

    def spy(
        self,
        matrix,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Visualize a matrix sparsity pattern in a subplot."""
        self._get_or_create_subplot(row, col).spy(matrix, layer=layer, **kwargs)

    def table(
        self,
        cellText=None,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add a table annotation to a subplot."""
        self._get_or_create_subplot(row, col).table(
            cellText=cellText, layer=layer, **kwargs
        )

    def gantt(
        self,
        tasks,
        start_times,
        durations,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """
        Add a Gantt chart to the canvas (matplotlib-style convenience method).

        Parameters:
        tasks (array-like): Task names or labels (y-axis).
        start_times (array-like): Start times for each task (x-axis).
        durations (array-like): Duration of each task.
        layer (int): Layer index (default 0).
        row, col (int): Subplot position (default top-left).
        **kwargs: Forwarded to the backend (e.g., color, alpha, edgecolor, label).
        """
        sp = self._get_or_create_subplot(row, col)
        sp.gantt(tasks, start_times, durations, layer=layer, **kwargs)

    def flame_chart(
        self,
        labels,
        parents,
        values,
        start_times=None,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """
        Add a flame chart to the canvas (matplotlib-style convenience method).

        Parameters:
        labels (array-like): Labels for each stack frame/function.
        parents (array-like): Parent indices for each frame (None for root, or index of parent).
        values (array-like): Duration/sample count for each frame.
        start_times (array-like, optional): Start times for each frame. If None, computed from hierarchy.
        layer (int): Layer index (default 0).
        row, col (int): Subplot position (default top-left).
        **kwargs: Forwarded to the backend (e.g., colormap, edgecolor, label).
        """
        sp = self._get_or_create_subplot(row, col)
        sp.flame_chart(
            labels, parents, values, start_times=start_times, layer=layer, **kwargs
        )

    def set_xlabel(
        self,
        label: str,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Set the x-axis label and text properties for a subplot."""
        self._get_or_create_subplot(row, col).set_xlabel(label, **kwargs)

    def set_ylabel(
        self,
        label: str,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Set the y-axis label and text properties for a subplot."""
        self._get_or_create_subplot(row, col).set_ylabel(label, **kwargs)

    def set_title(
        self,
        title: str,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Set the title and text properties for a subplot."""
        self._get_or_create_subplot(row, col).set_title(title, **kwargs)

    def set_xlim(
        self, left=None, right=None, row: int | None = None, col: int | None = None
    ):
        """Set the x-axis limits for a subplot (default top-left)."""
        self._get_or_create_subplot(row, col).set_xlim(left, right)

    def set_ylim(
        self, bottom=None, top=None, row: int | None = None, col: int | None = None
    ):
        """Set the y-axis limits for a subplot (default top-left)."""
        self._get_or_create_subplot(row, col).set_ylim(bottom, top)

    def set_grid(
        self, visible: bool = True, row: int | None = None, col: int | None = None
    ):
        """Show or hide the grid for a subplot (default top-left)."""
        self._get_or_create_subplot(row, col).set_grid(visible)

    def set_legend(
        self, visible: bool = True, row: int | None = None, col: int | None = None
    ):
        """Show or hide the legend for a subplot (default top-left)."""
        self._get_or_create_subplot(row, col).set_legend(visible)

    def tick_params(self, row: int | None = None, col: int | None = None, **kwargs):
        """Configure major/minor tick appearance for a subplot."""
        self._get_or_create_subplot(row, col).tick_params(**kwargs)

    def set_xscale(self, scale: str, row: int | None = None, col: int | None = None):
        """Set x-axis scale ('linear', 'log', 'symlog') for a subplot."""
        self._get_or_create_subplot(row, col).set_xscale(scale)

    def set_yscale(self, scale: str, row: int | None = None, col: int | None = None):
        """Set y-axis scale ('linear', 'log', 'symlog') for a subplot."""
        self._get_or_create_subplot(row, col).set_yscale(scale)

    def set_axis_off(self, row: int | None = None, col: int | None = None):
        """Hide the axis frame, ticks, and labels for a subplot."""
        self._get_or_create_subplot(row, col).set_axis_off()

    def set_axis_on(self, row: int | None = None, col: int | None = None):
        """Show the axis frame, ticks, and labels for a subplot."""
        self._get_or_create_subplot(row, col).set_axis_on()

    def set_axisbelow(self, state=True, row: int | None = None, col: int | None = None):
        """Set whether gridlines and ticks are drawn below plot data."""
        self._get_or_create_subplot(row, col).set_axisbelow(state)

    def set_facecolor(self, color, row: int | None = None, col: int | None = None):
        """Set a subplot's background color."""
        self._get_or_create_subplot(row, col).set_facecolor(color)

    def margins(self, *args, row: int | None = None, col: int | None = None, **kwargs):
        """Set x/y data margins for a subplot."""
        self._get_or_create_subplot(row, col).margins(*args, **kwargs)

    def invert_xaxis(self, row: int | None = None, col: int | None = None):
        """Invert a subplot's x-axis."""
        self._get_or_create_subplot(row, col).invert_xaxis()

    def invert_yaxis(self, row: int | None = None, col: int | None = None):
        """Invert a subplot's y-axis."""
        self._get_or_create_subplot(row, col).invert_yaxis()

    def minorticks_on(self, row: int | None = None, col: int | None = None):
        """Enable minor ticks for a subplot."""
        self._get_or_create_subplot(row, col).minorticks_on()

    def minorticks_off(self, row: int | None = None, col: int | None = None):
        """Disable minor ticks for a subplot."""
        self._get_or_create_subplot(row, col).minorticks_off()

    def locator_params(self, row: int | None = None, col: int | None = None, **kwargs):
        """Set axis locator parameters for a subplot."""
        self._get_or_create_subplot(row, col).locator_params(**kwargs)

    def ticklabel_format(
        self, row: int | None = None, col: int | None = None, **kwargs
    ):
        """Configure numeric tick-label formatting for a subplot."""
        self._get_or_create_subplot(row, col).ticklabel_format(**kwargs)

    def set_xticks(
        self,
        ticks,
        labels=None,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Set x-axis ticks and optional label properties for a subplot."""
        self._get_or_create_subplot(row, col).set_xticks(ticks, labels, **kwargs)

    def set_yticks(
        self,
        ticks,
        labels=None,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Set y-axis ticks and optional label properties for a subplot."""
        self._get_or_create_subplot(row, col).set_yticks(ticks, labels, **kwargs)

    def fill_between(
        self,
        x,
        y1,
        y2=0,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Fill the region between two curves on a subplot."""
        self._get_or_create_subplot(row, col).fill_between(
            x, y1, y2, layer=layer, **kwargs
        )

    def fill_betweenx(
        self,
        y,
        x1,
        x2=0,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Fill the area between two x-boundaries along y."""
        self._get_or_create_subplot(row, col).fill_betweenx(
            y, x1, x2, layer=layer, **kwargs
        )

    def errorbar(
        self,
        x,
        y,
        yerr=None,
        xerr=None,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add an error-bar line to a subplot."""
        self._get_or_create_subplot(row, col).errorbar(
            x, y, yerr=yerr, xerr=xerr, layer=layer, **kwargs
        )

    def axhline(
        self, y=0, layer=0, row: int | None = None, col: int | None = None, **kwargs
    ):
        """Add a full-width horizontal reference line to a subplot."""
        self._get_or_create_subplot(row, col).axhline(y=y, layer=layer, **kwargs)

    def axvline(
        self, x=0, layer=0, row: int | None = None, col: int | None = None, **kwargs
    ):
        """Add a full-height vertical reference line to a subplot."""
        self._get_or_create_subplot(row, col).axvline(x=x, layer=layer, **kwargs)

    def hlines(
        self,
        y,
        xmin,
        xmax,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add horizontal lines at specified y positions to a subplot."""
        self._get_or_create_subplot(row, col).hlines(
            y, xmin, xmax, layer=layer, **kwargs
        )

    def vlines(
        self,
        x,
        ymin,
        ymax,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add vertical lines at specified x positions to a subplot."""
        self._get_or_create_subplot(row, col).vlines(
            x, ymin, ymax, layer=layer, **kwargs
        )

    def axvspan(
        self,
        xmin,
        xmax,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add a vertical shaded span across a subplot."""
        self._get_or_create_subplot(row, col).axvspan(xmin, xmax, layer=layer, **kwargs)

    def axhspan(
        self,
        ymin,
        ymax,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add a horizontal shaded span across a subplot."""
        self._get_or_create_subplot(row, col).axhspan(ymin, ymax, layer=layer, **kwargs)

    def arrow(
        self,
        x,
        y,
        dx,
        dy,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add an arrow to a subplot."""
        self._get_or_create_subplot(row, col).arrow(x, y, dx, dy, layer=layer, **kwargs)

    def axline(
        self,
        xy1,
        xy2=None,
        slope=None,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add an infinitely extending line to a subplot."""
        self._get_or_create_subplot(row, col).axline(
            xy1, xy2=xy2, slope=slope, layer=layer, **kwargs
        )

    def annotate(
        self,
        text,
        xy,
        xytext=None,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add an annotation (with optional arrow) to a subplot."""
        self._get_or_create_subplot(row, col).annotate(
            text, xy, xytext=xytext, layer=layer, **kwargs
        )

    def text(
        self, x, y, s, layer=0, row: int | None = None, col: int | None = None, **kwargs
    ):
        """Add a text label at (x, y) on a subplot."""
        self._get_or_create_subplot(row, col).text(x, y, s, layer=layer, **kwargs)

    def imshow(
        self,
        data,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add an image/matrix plot to a subplot."""
        self._get_or_create_subplot(row, col).add_imshow(data, layer=layer, **kwargs)

    def add_patch(
        self,
        patch,
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add a Matplotlib patch to a subplot."""
        self._get_or_create_subplot(row, col).add_patch(patch, layer=layer, **kwargs)

    def colorbar(
        self,
        label: str = "",
        layer=0,
        row: int | None = None,
        col: int | None = None,
        **kwargs,
    ):
        """Add a colorbar to the most recent imshow() on a subplot (matplotlib backend)."""
        self._get_or_create_subplot(row, col).add_colorbar(
            label=label, layer=layer, **kwargs
        )

    # ------------------------------------------------------------------
    # Multi-subplot helpers
    # ------------------------------------------------------------------

    def subplot(self, row: int = 0, col: int = 0) -> LinePlot:
        """
        Return the LinePlot at position (row, col).

        Raises ValueError if no subplot has been created there yet.
        """
        sp = self._subplot_matrix[row][col]
        if sp is None:
            raise ValueError(
                f"No subplot at ({row}, {col}). "
                "Call add_subplot() or use Canvas.subplots() first."
            )
        return sp

    def twinx(self, row: int | None = None, col: int | None = None) -> LinePlot:
        """Create or return a secondary y-axis sharing a subplot's x-axis.

        The returned subplot accepts the same plotting methods as a regular
        subplot. With the Matplotlib backend it is rendered on ``ax.twinx()``.
        Only one secondary y-axis is supported per subplot.
        """
        self._get_or_create_subplot(row, col)
        if row is None:
            row, col = 0, 0
        key = (row, col)
        if key not in self._twinx_subplots:
            self._twinx_subplots[key] = LinePlot()
        return self._twinx_subplots[key]

    @property
    def twinx_axes(self):
        """Return materialized Matplotlib secondary axes by ``(row, col)``."""
        return dict(self._matplotlib_twin_axes)

    def iter_subplots(self):
        """Yield (row, col, subplot) for every initialized subplot, row-major."""
        for r in range(self.nrows):
            for c in range(self.ncols):
                sp = self._subplot_matrix[r][c]
                if sp is not None:
                    yield r, c, sp

    def suptitle(self, title: str, **kwargs):
        """
        Set a figure-level title (rendered above all subplots).

        Parameters:
        title (str): Title text.
        **kwargs: Forwarded to matplotlib's fig.suptitle (e.g., fontsize, y).
        """
        self._suptitle = title
        self._suptitle_kwargs = kwargs

    def supxlabel(self, label: str, **kwargs):
        """Set a figure-level x-axis label."""
        self._supxlabel = label
        self._supxlabel_kwargs = dict(kwargs)

    def supylabel(self, label: str, **kwargs):
        """Set a figure-level y-axis label."""
        self._supylabel = label
        self._supylabel_kwargs = dict(kwargs)

    def subplots_adjust(self, **kwargs):
        """Adjust subplot spacing after the figure is created."""
        self._subplots_adjust_kwargs = dict(kwargs)

    def tight_layout(self, **kwargs):
        """Apply Matplotlib's automatic tight layout after plotting."""
        self._tight_layout_kwargs = dict(kwargs)

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
                    savefig_kwargs = {"dpi": self.dpi} if self.dpi is not None else {}
                    fig.savefig(_fn, **savefig_kwargs)
                    print(f"Saved {_fn}")
            else:
                if layers is None:
                    layers = self.layers
                    full_filepath = filename
                else:
                    full_filepath = f"{filename_no_extension}_{layers}.{extension}"

                if self._plotted:
                    savefig_kwargs = {"dpi": self.dpi} if self.dpi is not None else {}
                    self._matplotlib_fig.savefig(full_filepath, **savefig_kwargs)
                else:
                    fig, axs = self.plot(
                        backend="matplotlib",
                        savefig=True,
                        layers=layers,
                    )
                    savefig_kwargs = {"dpi": self.dpi} if self.dpi is not None else {}
                    fig.savefig(full_filepath, **savefig_kwargs)
                if verbose:
                    print(f"Saved {full_filepath}")
        elif backend == "plotext":
            if layer_by_layer:
                layers = []
                for layer in self.layers:
                    layers.append(layer)
                    figure = self.plot(
                        backend="plotext",
                        savefig=False,
                        layers=layers,
                    )
                    _fn = f"{filename_no_extension}_{layers}.{extension}"
                    figure.savefig(_fn)
                    print(f"Saved {_fn}")
            else:
                if layers is None:
                    layers = self.layers
                    full_filepath = filename
                else:
                    full_filepath = f"{filename_no_extension}_{layers}.{extension}"
                figure = self.plot(
                    backend="plotext",
                    savefig=False,
                    layers=layers,
                )
                figure.savefig(full_filepath)
                if verbose:
                    print(f"Saved {full_filepath}")
        elif backend == "plotly":
            if layer_by_layer:
                layers = []
                for layer in self.layers:
                    layers.append(layer)
                    full_filepath = f"{filename_no_extension}_{layers}{extension}"
                    fig = self.plot(
                        backend="plotly",
                        savefig=False,
                        layers=layers,
                    )
                    self._save_plotly(fig, full_filepath)
                    if verbose:
                        print(f"Saved {full_filepath}")
            else:
                if layers is None:
                    layers = self.layers
                    full_filepath = filename
                else:
                    full_filepath = f"{filename_no_extension}_{layers}{extension}"
                fig = self.plot(
                    backend="plotly",
                    savefig=False,
                    layers=layers,
                )
                self._save_plotly(fig, full_filepath)
                if verbose:
                    print(f"Saved {full_filepath}")
        elif backend == "tikzfigure":
            if layers is not None:
                raise NotImplementedError(
                    "Layer-by-layer rendering is not supported for tikzfigure backend"
                )
            fig = self.plot(backend="tikzfigure", savefig=False)
            fig.savefig(filename)
            if verbose:
                print(f"Saved {filename}")

    def plot(
        self,
        backend: Backends = "matplotlib",
        savefig: bool = False,
        layers: list | None = None,
        usetex: bool | None = None,
        verbose: bool = False,
        matplotlib_postprocess=None,
        matplotlib_customizations=None,
    ):
        """Render the canvas.

        ``matplotlib_customizations`` accepts either a callable receiving
        ``(figure, axes)`` or a mapping whose ``figure`` methods run once and
        whose ``axes`` methods run on every axes. Values are keyword arguments,
        or use ``{"args": [...], "kwargs": {...}}`` for positional and keyword
        arguments. A scalar value is passed as one positional argument.

        ``matplotlib_postprocess`` is an optional callable receiving
        ``(figure, axes)`` after a Matplotlib figure has been created. It can
        call any Matplotlib API, including APIs not wrapped by maxplotlib.
        Both options are only valid with the Matplotlib backend.
        """
        resolved_usetex = self._usetex if usetex is None else usetex

        if verbose:
            print(f"Plotting figure using backend: {backend}")

        if backend == "matplotlib":
            return self.plot_matplotlib(
                savefig=savefig,
                layers=layers,
                usetex=resolved_usetex,
                verbose=verbose,
                matplotlib_postprocess=matplotlib_postprocess,
                matplotlib_customizations=matplotlib_customizations,
            )
        elif (
            matplotlib_postprocess is not None or matplotlib_customizations is not None
        ):
            raise ValueError(
                "Matplotlib customizations are only supported with the matplotlib backend"
            )
        elif backend == "plotly":
            return self.plot_plotly(
                savefig=savefig,
                layers=layers,
                usetex=resolved_usetex,
                verbose=verbose,
            )
        elif backend == "plotext":
            return self.plot_plotext(
                savefig=savefig,
                layers=layers,
                verbose=verbose,
            )
        elif backend == "tikzfigure":
            return self.plot_tikzfigure(savefig=savefig, verbose=verbose)
        else:
            raise ValueError(f"Invalid backend: {backend}")

    def show(
        self,
        backend: Backends = "matplotlib",
        layers: list | None = None,
        usetex: bool | None = None,
        verbose: bool = False,
        block: bool = True,
        matplotlib_postprocess=None,
        matplotlib_customizations=None,
    ):
        """
        Render and display the canvas.

        Parameters
        ----------
        block : bool, optional
            matplotlib backend only (default: True). Whether to block until
            the figure window is closed before returning. Passed straight to
            ``plt.show(block=...)`` rather than left at its default, because
            that default follows ``matplotlib.is_interactive()`` -- which
            other imported code (IPython, a prior interactive session) can
            flip to True behind this call's back, silently turning off
            blocking. Forcing it explicitly is what makes repeated
            ``canvas.show()`` calls in a loop display one figure at a time
            instead of every window appearing together.
        """
        if verbose:
            print(f"Showing canvas using backend: {backend}")
        if backend != "matplotlib" and (
            matplotlib_postprocess is not None or matplotlib_customizations is not None
        ):
            raise ValueError(
                "Matplotlib customizations are only supported with the matplotlib backend"
            )

        if backend == "matplotlib":
            if verbose:
                print("Generating Matplotlib figure for display...")
            fig, axes = self.plot(
                backend="matplotlib",
                savefig=False,
                layers=layers,
                usetex=usetex,
                verbose=verbose,
                matplotlib_postprocess=matplotlib_postprocess,
                matplotlib_customizations=matplotlib_customizations,
            )
            if verbose:
                print("Displaying Matplotlib figure...")
            if _display_matplotlib_figure_in_notebook(fig):
                # IPython has rendered the figure already. Closing it prevents
                # a later implicit pyplot display and releases its resources.
                plt.close(fig)
            else:
                plt.show(block=block)
            return fig, axes
        elif backend == "plotly":
            resolved_usetex = self._usetex if usetex is None else usetex
            fig = self.plot_plotly(
                savefig=False, layers=layers, usetex=resolved_usetex, verbose=verbose
            )
            fig.show()
            return fig
        elif backend == "plotext":
            figure = self.plot_plotext(
                savefig=False,
                layers=layers,
                verbose=verbose,
            )
            figure.show()
            return figure
        elif backend == "tikzfigure":
            fig = self.plot_tikzfigure(savefig=False, verbose=verbose)
            # TikzFigure handles all rendering (single or multi-subplot)
            fig.show(transparent=False)
            # TikzFigure.__repr__ returns the generated TikZ source. Returning
            # it from a notebook cell would therefore print the source after
            # TikzFigure has already displayed the rendered image.
            return None if _running_in_jupyter() else fig
        else:
            raise ValueError("Invalid backend")

    def plot_matplotlib(
        self,
        savefig: bool = False,
        layers: list | None = None,
        usetex: bool | None = None,
        verbose: bool = False,
        matplotlib_postprocess=None,
        matplotlib_customizations=None,
    ):
        """
        Generate and optionally display the subplots.

        Parameters:
        filename (str, optional): Filename to save the figure.
        """
        if verbose:
            print("Generating Matplotlib figure...")

        resolved_usetex = self._usetex if usetex is None else usetex
        tex_fonts = setup_tex_fonts(fontsize=self.fontsize, usetex=resolved_usetex)
        render_dpi = self.dpi if savefig else None

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
        subplot_kwargs = {
            "squeeze": False,
            "gridspec_kw": self._gridspec_kw,
        }
        if self._figsize is not None:
            subplot_kwargs["figsize"] = self._figsize
        elif self._width is not None:
            fig_width, fig_height = set_size(
                width=self._width,
                ratio=self._ratio,
                dpi=render_dpi,
                verbose=verbose,
            )
            subplot_kwargs["figsize"] = (fig_width, fig_height)
        if verbose:
            if "figsize" in subplot_kwargs:
                fig_width, fig_height = subplot_kwargs["figsize"]
                print(f"Figure size: {fig_width} x {fig_height} inches")
            else:
                print("Figure size: Matplotlib default")
            print(f"Render DPI override: {render_dpi} (export DPI: {self.dpi})")

        if render_dpi is not None:
            subplot_kwargs["dpi"] = render_dpi

        fig, axes = plt.subplots(
            self.nrows,
            self.ncols,
            **subplot_kwargs,
        )

        if verbose:
            print(f"Created Matplotlib figure and axes with shape {axes.shape}")

        for (row, col), subplot in self._subplot_dict.items():
            ax = axes[row][col]
            if isinstance(subplot, TikzFigure):
                plot_matplotlib(subplot, ax, layers=layers)
            else:
                subplot.plot_matplotlib(ax, layers=layers)
            # ax.set_title(f"Subplot ({row}, {col})")
            ax.grid()

        if verbose:
            print("Finished plotting subplots.")

        if self._suptitle:
            suptitle_kwargs = dict(self._suptitle_kwargs)
            suptitle_kwargs.setdefault("fontsize", self.fontsize)
            fig.suptitle(self._suptitle, **suptitle_kwargs)

        if self._supxlabel:
            fig.supxlabel(self._supxlabel, **self._supxlabel_kwargs)
        if self._supylabel:
            fig.supylabel(self._supylabel, **self._supylabel_kwargs)
        if self._subplots_adjust_kwargs:
            fig.subplots_adjust(**self._subplots_adjust_kwargs)
        if self._tight_layout_kwargs is not None:
            fig.tight_layout(**self._tight_layout_kwargs)

        if verbose:
            print("Set suptitle.")

        # Set caption, labels, etc., if needed
        self._plotted = True
        self._matplotlib_fig = fig
        self._matplotlib_axes = axes
        self._matplotlib_twin_axes = {}
        for (row, col), twin_subplot in self._twinx_subplots.items():
            twin_axis = axes[row][col].twinx()
            twin_subplot.plot_matplotlib(twin_axis, layers=layers)
            self._matplotlib_twin_axes[(row, col)] = twin_axis
        if matplotlib_customizations is not None:
            _apply_matplotlib_customizations(fig, axes, matplotlib_customizations)
        if matplotlib_postprocess is not None:
            if not callable(matplotlib_postprocess):
                raise TypeError("matplotlib_postprocess must be callable")
            matplotlib_postprocess(fig, axes)
        return fig, axes

    def plot_tikzfigure(
        self,
        savefig: bool = False,
        verbose: bool = False,
    ) -> TikzFigure:
        """
        Generate a TikZ figure from subplots.

        For now, returns the first subplot's TikzFigure.
        Full multi-subplot support requires TikzFigure's subfigure_axis API.

        Parameters:
        verbose (bool): If True, print debug information.

        Returns:
        TikzFigure: Figure object that can be shown, saved, or compiled.
        """
        if verbose:
            print(f"Plotting tikzfigure with {len(self._subplot_dict)} subplot(s)")

        if self._twinx_subplots:
            raise NotImplementedError(
                "twinx plots are currently supported only by the matplotlib and plotly backends"
            )

        # Check for unsupported layouts
        if self.nrows > 1:
            raise NotImplementedError(
                "Vertical/grid layouts (nrows > 1) are not yet supported for tikzfigure backend. "
                "Use horizontal layouts (1×n) only."
            )

        # Validate that at least one subplot exists
        if len(self._subplot_dict) == 0:
            raise ValueError(
                "No subplots to plot. Call add_subplot() or Canvas.subplots() first."
            )

        axis_width, axis_height = self._get_tikzfigure_axis_dimensions()
        fig = TikzFigure()

        # Add each subplot as a subfigure axis
        for (row, col), line_plot in self._subplot_dict.items():
            if verbose:
                print(f"Plotting subplot at row {row}, col {col}")

            # Create subfigure axis with subplot metadata
            ax = fig.subfigure_axis(
                xlabel=line_plot._xlabel or "",
                ylabel=line_plot._ylabel or "",
                xlim=(
                    (line_plot._xmin, line_plot._xmax)
                    if line_plot._xmin is not None
                    else None
                ),
                ylim=(
                    (line_plot._ymin, line_plot._ymax)
                    if line_plot._ymin is not None
                    else None
                ),
                grid=line_plot._grid,
                title=line_plot._title or f"Subplot {col + 1}",
                width=0.45,
                axis_width=axis_width,
                height=axis_height,
            )

            # Add each plot line to the subfigure
            for line_data in line_plot.line_data:
                if line_data.get("plot_type") == "plot":
                    # Extract and transform x, y data
                    x = (line_data["x"] + line_plot._xshift) * line_plot._xscale
                    y = (line_data["y"] + line_plot._yshift) * line_plot._yscale
                    kwargs = line_data.get("kwargs", {})
                    if verbose:
                        print(f"Line {kwargs = }")
                    # Add plot to subfigure axis
                    ax.add_plot(
                        x=x,
                        y=y,
                        # label=kwargs.get("label", ""),
                        color=kwargs.get("color", "black"),
                        line_width=kwargs.get("linewidth", 1.0),
                    )
                elif line_data.get("plot_type") == "gantt":
                    tasks = line_data["tasks"]
                    start_times = (
                        line_data["start_times"] + line_plot._xshift
                    ) * line_plot._xscale
                    durations = line_data["durations"] * line_plot._xscale
                    y_positions = np.arange(len(tasks))
                    kwargs = line_data.get("kwargs", {})

                    # Draw horizontal bars for each task as filled rectangles
                    for i, (task, start, duration) in enumerate(
                        zip(tasks, start_times, durations)
                    ):
                        x_start = float(start)
                        x_end = float(start + duration)
                        y_pos = float(y_positions[i])
                        bar_height = 0.8

                        # Create rectangle coordinates for the bar
                        x_coords = [x_start, x_end, x_end, x_start, x_start]
                        y_coords = [
                            y_pos - bar_height / 2,
                            y_pos - bar_height / 2,
                            y_pos + bar_height / 2,
                            y_pos + bar_height / 2,
                            y_pos - bar_height / 2,
                        ]

                        # Add as a filled plot
                        color = kwargs.get("color", "blue")
                        ax.add_plot(
                            x=x_coords,
                            y=y_coords,
                            color=color,
                            fill=True,
                            line_width=0,
                        )

                    # Set y-axis ticks to show task names
                    if line_plot._yticks is None:
                        ax.set_ticks("y", list(y_positions), tasks)

            # Add legend if requested
            if line_plot._legend and len(line_plot.line_data) > 0:
                ax.set_legend(position="north east")

        return fig

    def _get_tikzfigure_axis_dimensions(self) -> tuple[str | None, str | None]:
        if self._width is None:
            return None, None

        total_width_in, total_height_in = set_size(
            width=self._width,
            ratio=self._ratio,
            dpi=self._dpi if self._dpi is not None else 300,
        )
        total_width_cm = total_width_in * 2.54
        total_height_cm = total_height_in * 2.54
        horizontal_sep_cm = getattr(TikzFigure, "GROUPPLOT_HORIZONTAL_SEP_CM", 1.5)
        available_width_cm = total_width_cm - horizontal_sep_cm * (self.ncols - 1)
        if available_width_cm <= 0:
            raise ValueError(
                f'Canvas width "{self._width}" is too small for {self.ncols} '
                "tikzfigure subplot(s)."
            )

        axis_width_cm = available_width_cm / self.ncols
        return f"{axis_width_cm:.6g}cm", f"{total_height_cm:.6g}cm"

    def plot_plotext(
        self,
        savefig: bool = False,
        layers: list | None = None,
        verbose: bool = False,
    ) -> PlotextFigure:
        if self._twinx_subplots:
            raise NotImplementedError(
                "twinx plots are not supported by the plotext backend"
            )
        if verbose:
            print("Generating plotext figure...")

        figure = create_plotext_figure(self.nrows, self.ncols)

        for row, col, subplot in self.iter_subplots():
            ax = (
                figure
                if (self.nrows, self.ncols) == (1, 1)
                else figure.subplot(row + 1, col + 1)
            )
            if isinstance(subplot, TikzFigure):
                raise NotImplementedError(
                    "tikzfigure subplots cannot be rendered with the plotext backend."
                )
            subplot.plot_plotext(ax, layers=layers)

        wrapped = PlotextFigure(figure=figure, suptitle=self._suptitle)
        if savefig and isinstance(savefig, str):
            wrapped.savefig(savefig)

        self._plotext_figure = wrapped
        return wrapped

    def plot_plotly(
        self,
        show=True,
        savefig=None,
        layers: list | None = None,
        usetex: bool | None = None,
        verbose: bool = False,
    ):
        """
        Generate and optionally display the subplots using Plotly.

        Parameters:
        show (bool): Whether to display the plot.
        savefig (str, optional): Filename to save the figure if provided.
        verbose (bool): Whether to print verbose output.

        """

        resolved_usetex = self._usetex if usetex is None else usetex

        setup_tex_fonts(
            fontsize=self.fontsize,
            usetex=resolved_usetex,
        )  # adjust or redefine for Plotly if needed

        # Create subplot titles in row-major order (Plotly expects rows*cols entries)
        subplot_titles = [""] * (self.nrows * self.ncols)
        for (row, col), sp in self._subplot_dict.items():
            index = row * self.ncols + col
            subplot_titles[index] = sp._title or f"({row}, {col})"

        specs = [
            [{"secondary_y": (r, c) in self._twinx_subplots} for c in range(self.ncols)]
            for r in range(self.nrows)
        ]
        fig = make_subplots(
            rows=self.nrows,
            cols=self.ncols,
            subplot_titles=subplot_titles,
            specs=specs,
        )

        # Plot each subplot and propagate axis labels/scale
        for (row, col), line_plot in self._subplot_dict.items():
            traces, shapes, annotations = line_plot.plot_plotly(layers=layers)
            for trace in traces:
                if trace.type == "pie":
                    fig.add_trace(trace)
                else:
                    fig.add_trace(trace, row=row + 1, col=col + 1)

            # Axis indices are row-major: (row*ncols + col + 1)
            axis_index = row * self.ncols + col + 1
            xref = "x" if axis_index == 1 else f"x{axis_index}"
            yref = "y" if axis_index == 1 else f"y{axis_index}"

            twin_subplot = self._twinx_subplots.get((row, col))
            if twin_subplot is not None:
                twin_traces, twin_shapes, twin_annotations = twin_subplot.plot_plotly(
                    layers=layers
                )
                for trace in twin_traces:
                    if trace.type == "pie":
                        fig.add_trace(trace)
                    else:
                        fig.add_trace(
                            trace,
                            row=row + 1,
                            col=col + 1,
                            secondary_y=True,
                        )
                for shape in twin_shapes:
                    shape = dict(shape)
                    shape["yref"] = yref
                    fig.add_shape(shape)
                for annotation in twin_annotations:
                    fig.add_annotation(dict(annotation))

            for shape in shapes:
                shape = dict(shape)
                if shape.get("xref") not in {"paper"}:
                    shape["xref"] = xref
                if shape.get("yref") not in {"paper"}:
                    shape["yref"] = yref
                fig.add_shape(shape)

            for annotation in annotations:
                annotation = dict(annotation)
                annotation.setdefault("xref", xref)
                annotation.setdefault("yref", yref)
                fig.add_annotation(annotation)

            # Apply per-axis config in a row/col-safe way
            xaxis_kwargs = dict(
                title_text=line_plot._xlabel or None,
                showgrid=bool(line_plot._grid),
                row=row + 1,
                col=col + 1,
            )
            if line_plot._xaxis_scale == "log":
                xaxis_kwargs["type"] = "log"
            fig.update_xaxes(**xaxis_kwargs)

            yaxis_kwargs = dict(
                title_text=line_plot._ylabel or None,
                showgrid=bool(line_plot._grid),
                row=row + 1,
                col=col + 1,
            )
            if line_plot._yaxis_scale == "log":
                yaxis_kwargs["type"] = "log"
            fig.update_yaxes(**yaxis_kwargs)

            if twin_subplot is not None:
                fig.update_yaxes(
                    title_text=twin_subplot._ylabel or None,
                    secondary_y=True,
                    row=row + 1,
                    col=col + 1,
                )

            # Axis limits
            if line_plot._xmin is not None or line_plot._xmax is not None:
                x_range = [line_plot._xmin, line_plot._xmax]
                if x_range[0] is not None:
                    x_range[0] = line_plot._transform_scalar_x(x_range[0])
                if x_range[1] is not None:
                    x_range[1] = line_plot._transform_scalar_x(x_range[1])
                if (
                    line_plot._xaxis_scale == "log"
                    and x_range[0] is not None
                    and x_range[1] is not None
                    and x_range[0] > 0
                    and x_range[1] > 0
                ):
                    x_range = [np.log10(x_range[0]), np.log10(x_range[1])]
                fig.update_xaxes(
                    range=x_range,
                    row=row + 1,
                    col=col + 1,
                )
            if line_plot._ymin is not None or line_plot._ymax is not None:
                y_range = [line_plot._ymin, line_plot._ymax]
                if y_range[0] is not None:
                    y_range[0] = line_plot._transform_scalar_y(y_range[0])
                if y_range[1] is not None:
                    y_range[1] = line_plot._transform_scalar_y(y_range[1])
                if (
                    line_plot._yaxis_scale == "log"
                    and y_range[0] is not None
                    and y_range[1] is not None
                    and y_range[0] > 0
                    and y_range[1] > 0
                ):
                    y_range = [np.log10(y_range[0]), np.log10(y_range[1])]
                fig.update_yaxes(
                    range=y_range,
                    row=row + 1,
                    col=col + 1,
                )

            # Custom ticks (positions + optional labels)
            if line_plot._xticks is not None:
                tickvals = [line_plot._transform_scalar_x(v) for v in line_plot._xticks]
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=line_plot._xticklabels,
                    tickangle=line_plot._xtick_kwargs.get("rotation"),
                    row=row + 1,
                    col=col + 1,
                )
            if line_plot._yticks is not None:
                tickvals = [line_plot._transform_scalar_y(v) for v in line_plot._yticks]
                fig.update_yaxes(
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=line_plot._yticklabels,
                    tickangle=line_plot._ytick_kwargs.get("rotation"),
                    row=row + 1,
                    col=col + 1,
                )

            # Aspect ratio
            if line_plot._aspect == "equal":
                fig.update_yaxes(scaleanchor=xref, row=row + 1, col=col + 1)
            elif isinstance(line_plot._aspect, (int, float)):
                fig.update_yaxes(
                    scaleanchor=xref,
                    scaleratio=float(line_plot._aspect),
                    row=row + 1,
                    col=col + 1,
                )

        # Update layout settings
        fig.update_layout(
            font=dict(size=self.fontsize),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        if self._suptitle:
            fig.update_layout(title=dict(text=self._suptitle, x=0.5))

        if savefig:
            try:
                fig.write_image(savefig)
            except Exception as exc:
                raise RuntimeError(
                    "Plotly image export failed. If you are exporting to PNG/PDF/SVG, "
                    "install kaleido (e.g., `pip install -U kaleido`)."
                ) from exc

        return fig

    def _save_plotly(self, fig, filename: str) -> None:
        _, extension = os.path.splitext(filename)
        extension = extension.lower()
        if extension in {".html", ".htm"}:
            fig.write_html(filename)
            return
        try:
            fig.write_image(filename)
        except Exception as exc:
            raise RuntimeError(
                "Plotly image export failed. For PNG/PDF/SVG export, install kaleido "
                "(e.g., `pip install -U kaleido`), or export to HTML instead."
            ) from exc

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
    def usetex(self):
        return self._usetex

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

    def __repr__(self):
        return f"Canvas(nrows={self.nrows}, ncols={self.ncols}, caption={self.caption}, label={self.label})"

    # Magic methods
    def __str__(self):
        return f"Canvas(nrows={self.nrows}, ncols={self.ncols}, figsize={self.figsize})"


if __name__ == "__main__":
    c = Canvas(ncols=2, nrows=2)
    sp = c.add_subplot()
    sp.plot([0, 1, 2, 3], [0, 1, 4, 9], label="Line 1")
    c.plot(backend="matplotlib")
    print("done")
