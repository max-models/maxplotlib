"""A drawn Matplotlib figure as a tikzfigure figure of pgfplots axes.

Every axes becomes a pgfplots ``axis`` at the place and size it has in the
figure, with its limits, scales, labels, title, ticks, grid, spines and
legend. What is drawn in it is converted artist by artist, in Matplotlib's
drawing order:

* vector graphics, in pgfplots code: lines and markers
  (:class:`~matplotlib.lines.Line2D`), scatter plots, line and polygon
  collections (``hlines``, ``fill_between``, error bars, ...), contour
  lines, patches (bars, spans, polygons, arrows) and text and annotations;
* raster images, rendered by Matplotlib itself and placed with
  ``\\addplot graphics``: meshes, images, filled contours, quivers, and
  every artist that has no vector counterpart here or would be too large
  for TeX (``max_markers``, ``max_items``, ``max_points``).

Colorbars are axes of their own: their color strip is an image, their
ticks and label pgfplots text. Axes that pgfplots cannot represent (polar
or 3-D projections, symlog scales) become one image each, as do figure
legends. Text is converted with :func:`~maxplotlib.backends.tikzfigure.text.latex`.
"""

from __future__ import annotations

import io
import warnings

import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.collections import (
    Collection,
    LineCollection,
    PathCollection,
    PolyCollection,
)
from matplotlib.contour import ContourSet
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.path import Path
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox
from tikzfigure import TikzFigure
from tikzfigure.core.axis import Axis2D
from tikzfigure.core.plot import format_number

from .text import is_multiline, latex

__all__ = ["TikzConversionWarning", "figure_to_tikz"]


class TikzConversionWarning(UserWarning):
    """A part of a Matplotlib figure that is drawn as an image or left out."""


# Matplotlib marker -> (pgfplots mark when filled, when hollow, rotation, size factor)
_MARKS = {
    "o": ("*", "o", 0, 0.5),
    ".": ("*", "o", 0, 0.25),
    ",": ("*", "o", 0, 0.1),
    "s": ("square*", "square", 0, 0.5),
    "^": ("triangle*", "triangle", 0, 0.5),
    "v": ("triangle*", "triangle", 180, 0.5),
    "<": ("triangle*", "triangle", 90, 0.5),
    ">": ("triangle*", "triangle", 270, 0.5),
    "D": ("diamond*", "diamond", 0, 0.5),
    "d": ("diamond*", "diamond", 0, 0.45),
    "p": ("pentagon*", "pentagon", 0, 0.5),
    "h": ("pentagon*", "pentagon", 0, 0.5),
    "H": ("pentagon*", "pentagon", 0, 0.5),
    "*": ("star", "star", 0, 0.5),
    "+": ("+", "+", 0, 0.5),
    "P": ("+", "+", 0, 0.5),
    "x": ("x", "x", 0, 0.5),
    "X": ("x", "x", 0, 0.5),
    "|": ("|", "|", 0, 0.5),
    "_": ("-", "-", 0, 0.5),
    "1": ("Mercedes star", "Mercedes star", 180, 0.5),
    "2": ("Mercedes star", "Mercedes star", 0, 0.5),
}
_NO_MARKER = ("None", "none", "", " ", None)
_NO_LINE = ("None", "none", "", " ")
_HA = {"left": "west", "center": "", "right": "east"}
_VA = {
    "top": "north",
    "center": "",
    "bottom": "south",
    "baseline": "base",
    "center_baseline": "mid",
}
# formatters whose labels pgfplots writes just as well by itself
_AUTO_FORMATTERS = (
    mticker.ScalarFormatter,
    mticker.LogFormatterSciNotation,
    mticker.LogFormatterMathtext,
)
_AUTO_LOCATORS = (
    mticker.AutoLocator,
    mticker.MaxNLocator,
    mticker.LogLocator,
    mticker.AutoMinorLocator,
)


def figure_to_tikz(
    figure,
    *,
    raster_dpi: float = 300,
    max_markers: int = 2000,
    max_items: int = 500,
    max_points: int = 20000,
    precision: int = 6,
) -> TikzFigure:
    """Convert a Matplotlib figure into a :class:`~tikzfigure.TikzFigure`.

    The figure is drawn first (without showing it), so that its layout,
    autoscaled limits, ticks and legend positions are final, and is left
    as it was afterwards.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to convert.
    raster_dpi : float, optional
        Resolution of the parts drawn as images (meshes, images, ...).
        Default: 300.
    max_markers : int, optional
        Scatter plots with more points are drawn as an image. Default: 2000.
    max_items : int, optional
        Collections with more differently styled items (e.g. a line
        collection colored by value) are drawn as an image. Default: 500.
    max_points : int, optional
        Lines with more points, after Matplotlib's path simplification, are
        drawn as an image. Default: 20000.
    precision : int, optional
        Significant digits of the written coordinates. Default: 6.

    Returns
    -------
    tikzfigure.TikzFigure
        The figure: ``.generate_tikz()`` for the code, ``.savefig("f.pdf")``
        to compile it, ``.savefig("f.tikz")`` to write the code and its
        images.
    """
    converter = _FigureConverter(
        figure,
        raster_dpi=raster_dpi,
        max_markers=max_markers,
        max_items=max_items,
        max_points=max_points,
        precision=precision,
    )
    return converter.convert()


class _Colors:
    """Colors used in the figure, defined once as ``\\definecolor``."""

    def __init__(self):
        self.names: dict[str, str] = {}

    def __call__(self, color):
        """``(name, opacity)`` of a color; name ``None`` for a transparent one."""
        rgba = mcolors.to_rgba(color)
        if rgba[3] <= 0:
            return None, 0.0
        code = mcolors.to_hex(rgba, keep_alpha=False)[1:].upper()
        name = self.names.setdefault(code, f"mpl{code}")
        return name, float(rgba[3])

    def definitions(self) -> str:
        return "\n".join(
            f"\\definecolor{{{name}}}{{HTML}}{{{code}}}"
            for code, name in self.names.items()
        )


def _font(size) -> str:
    size = float(size)
    return f"\\fontsize{{{size:g}}}{{{1.2 * size:g}}}\\selectfont"


def _pt(value) -> str:
    return f"{float(value):.4g}pt"


def _dash(pattern) -> str | None:
    """A Matplotlib ``(offset, [on, off, ...])`` dash pattern in points as TikZ."""
    if pattern is None:
        return None
    offset, sequence = pattern
    if not sequence:
        return None
    parts = []
    for index, length in enumerate(sequence):
        parts.append(("on" if index % 2 == 0 else "off") + " " + _pt(length))
    option = "dash pattern=" + " ".join(parts)
    if offset:
        option += f", dash phase={_pt(offset)}"
    return option


class _FigureConverter:
    def __init__(self, figure, **options):
        self.figure = figure
        self.options = options
        self.colors = _Colors()
        self.tikz = TikzFigure()
        self.width, self.height = figure.get_size_inches()

    # -- the figure ----------------------------------------------------------
    def convert(self) -> TikzFigure:
        figure = self.figure
        figure.draw_without_rendering()
        # images are rendered from this figure; a layout engine would move the
        # axes while other parts are hidden
        engine = figure.get_layout_engine()
        figure.set_layout_engine("none")
        try:
            for ax in figure.axes:
                self._axes_tree(ax)
            for text in figure.texts:
                self._figure_text(text)
            for legend in figure.legends:
                if legend.get_visible():
                    self._as_image(legend, "a figure legend")
        finally:
            figure._layout_engine = engine
        self.tikz.add_package("lmodern")
        self.tikz.add_package("amsmath")
        definitions = self.colors.definitions()
        if definitions:
            self.tikz.add_raw(definitions)
        return self.tikz

    def _axes_tree(self, ax):
        if not ax.get_visible():
            return
        if ax.name != "rectilinear" or not _supported_scales(ax):
            what = (
                f"{ax.name} axes"
                if ax.name != "rectilinear"
                else f"{ax.get_xscale()}/{ax.get_yscale()} scaled axes"
            )
            self._as_image(ax, what)
            return
        _AxesConverter(self, ax).convert()
        for child in ax.child_axes:
            self._axes_tree(child)

    def _figure_text(self, text):
        if not text.get_visible() or not text.get_text().strip():
            return
        if isinstance(text, Annotation):
            text.update_positions(self.figure._get_renderer())
        x, y = _text_display_position(text)
        if not self.figure.bbox.contains(x, y):
            return
        x_in, y_in = x / self.figure.dpi, y / self.figure.dpi
        node = _text_node(text, self.colors, f"({x_in:.4f}in,{y_in:.4f}in)")
        self.tikz.add_raw(node)

    # -- images of whole parts ---------------------------------------------
    def _as_image(self, artist, what):
        """Draw ``artist`` (axes or legend) with its decorations as one image."""
        warnings.warn(
            f"{what} cannot be drawn with pgfplots; it is included as an image",
            TikzConversionWarning,
            stacklevel=4,
        )
        renderer = self.figure._get_renderer()
        bbox = artist.get_tightbbox(renderer)
        if bbox is None or bbox.width <= 0 or bbox.height <= 0:
            return
        inches = bbox.transformed(self.figure.dpi_scale_trans.inverted())
        data = self.render_only([artist], inches)
        axis = Axis2D(
            xlim=(0, 1),
            ylim=(0, 1),
            grid=False,
            width=f"{inches.width:.4f}in",
            height=f"{inches.height:.4f}in",
            options=[
                "hide axis",
                "scale only axis",
                "anchor=south west",
                f"at={{({inches.x0:.4f}in,{inches.y0:.4f}in)}}",
            ],
        )
        axis.add_graphics(0, 1, 0, 1, data=data, plot_options=["forget plot"])
        self.tikz.axes.append(axis)

    def render_only(self, artists, inches: Bbox, keep_axes=None) -> bytes:
        """A PNG of the figure region ``inches`` showing only ``artists``.

        ``keep_axes`` is the axes whose decorations stay hidden while its
        artists in ``artists`` are drawn; every other axes is hidden.
        """
        figure = self.figure
        keep = set(map(id, artists))
        hidden = []

        def hide(artist):
            if id(artist) not in keep and artist.get_visible():
                artist.set_visible(False)
                hidden.append(artist)

        # the axes containing a kept artist, and the axes containing those,
        # stay visible (an artist of a hidden axes is not drawn), with their
        # other children hidden
        parents = {id(child): ax for ax in _all_axes(figure) for child in ax.child_axes}
        owners = set()
        for artist in artists:
            owner = getattr(artist, "axes", None)
            if owner is artist:
                owner = parents.get(id(artist))
            while owner is not None:
                owners.add(id(owner))
                owner = parents.get(id(owner))
        for ax in _all_axes(figure):
            if id(ax) in keep:
                continue
            if id(ax) in owners:
                for child in ax.get_children():
                    if id(child) not in owners:
                        hide(child)
            else:
                hide(ax)
        hide(figure.patch)
        for text in figure.texts:
            hide(text)
        for legend in figure.legends:
            hide(legend)
        for artist in artists:  # the artists themselves are drawn
            if not artist.get_visible():
                artist.set_visible(True)
                hidden.append(("shown", artist))
        buffer = io.BytesIO()
        try:
            figure.savefig(
                buffer,
                format="png",
                dpi=self.options["raster_dpi"],
                transparent=True,
                bbox_inches=inches,
                pad_inches=0,
            )
        finally:
            for artist in hidden:
                if isinstance(artist, tuple):
                    artist[1].set_visible(False)
                else:
                    artist.set_visible(True)
        return buffer.getvalue()


def _all_axes(figure):
    out = []

    def walk(ax):
        out.append(ax)
        for child in ax.child_axes:
            walk(child)

    for ax in figure.axes:
        walk(ax)
    return out


def _supported_scales(ax) -> bool:
    return ax.get_xscale() in ("linear", "log") and ax.get_yscale() in (
        "linear",
        "log",
    )


def _text_display_position(text):
    return text.get_transform().transform(text.get_unitless_position())


def _anchor(text) -> str:
    vertical = _VA.get(text.get_va(), "")
    horizontal = _HA.get(text.get_ha(), "")
    if vertical == "base" and not horizontal:
        return "base"
    if vertical == "mid" and not horizontal:
        return "mid"
    anchor = " ".join(part for part in (vertical, horizontal) if part)
    return anchor or "center"


def _text_node(text, colors, at: str) -> str:
    """A ``\\node`` for a Matplotlib text, placed at the TikZ point ``at``."""
    options = [f"anchor={_anchor(text)}", "inner sep=0pt"]
    rotation = text.get_rotation()
    if rotation:
        options.append(f"rotate={rotation:g}")
    color, opacity = colors(text.get_color())
    if color is not None and color != "mpl000000":
        options.append(f"text={color}")
    alpha = text.get_alpha()
    if alpha is not None and alpha < 1:
        opacity *= alpha
    if opacity < 1:
        options.append(f"text opacity={opacity:.3g}")
    options.append(f"font={{{_font(text.get_fontsize())}}}")
    if is_multiline(text.get_text()):
        alignment = getattr(text, "_multialignment", None) or text.get_ha()
        options.append(f"align={alignment}")
    box = text.get_bbox_patch()
    if box is not None:
        face, face_opacity = colors(box.get_facecolor())
        edge, _ = colors(box.get_edgecolor())
        options[1] = "inner sep=2pt"
        if face is not None:
            options.append(f"fill={face}")
            if face_opacity < 1:
                options.append(f"fill opacity={face_opacity:.3g}, text opacity=1")
        if edge is not None and box.get_linewidth() > 0:
            options.append(f"draw={edge}, line width={_pt(box.get_linewidth())}")
        if "round" in type(box.get_boxstyle()).__name__.lower():
            options.append("rounded corners=2pt")
    return f"\\node[{', '.join(options)}] at {at} {{{latex(text.get_text())}}};"


class _AxesConverter:
    """One Matplotlib axes as one pgfplots axis."""

    def __init__(self, parent: _FigureConverter, ax):
        self.parent = parent
        self.figure = parent.figure
        self.ax = ax
        self.colors = parent.colors
        self.options = parent.options
        self.colorbar = getattr(ax, "_colorbar", None)
        self.legend_entries: list = []
        self.has_images = False

    # -- the axis ----------------------------------------------------------
    def convert(self):
        ax = self.ax
        position = ax.get_position()
        width, height = self.parent.width, self.parent.height
        x0, y0 = position.x0 * width, position.y0 * height
        self.size = (position.width * width, position.height * height)
        self.inches = Bbox.from_bounds(x0, y0, *self.size)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        options = [
            "scale only axis",
            "anchor=south west",
            f"at={{({x0:.4f}in,{y0:.4f}in)}}",
            "clip mode=individual",
            "unbounded coords=jump",
            "every axis plot/.append style={line join=round}",
        ]
        if xlim[0] > xlim[1]:
            options.append("x dir=reverse")
        if ylim[0] > ylim[1]:
            options.append("y dir=reverse")
        if not ax.axison:
            options.append("hide axis")
        else:
            options.extend(self._frame())
            for name in ("x", "y"):
                options.extend(self._ticks(name))
            options.extend(self._labels())
        options.extend(self._background())

        self.axis = Axis2D(
            xlabel=latex(ax.get_xlabel()) if ax.xaxis.get_visible() else "",
            ylabel=latex(ax.get_ylabel()) if ax.yaxis.get_visible() else "",
            title=latex(self._title()),
            xlim=(float(min(xlim)), float(max(xlim))),
            ylim=(float(min(ylim)), float(max(ylim))),
            xlog=ax.get_xscale() == "log",
            ylog=ax.get_yscale() == "log",
            grid=None,
            width=f"{self.size[0]:.4f}in",
            height=f"{self.size[1]:.4f}in",
            options=options,
        )
        self._legend_setup()
        self._contents()
        self._legend_entries()
        if self.has_images:
            self.axis.options.append("axis on top")
        grid = self._grid()
        if grid:
            self.axis.options.extend(grid)
        self.parent.tikz.axes.append(self.axis)

    def _title(self) -> str:
        for location in ("center", "left", "right"):
            title = self.ax.get_title(location)
            if title:
                return title
        return ""

    def _frame(self) -> list[str]:
        spines = self.ax.spines
        visible = {
            name: name in spines and spines[name].get_visible()
            for name in ("left", "right", "top", "bottom")
        }
        if "outline" in spines and spines["outline"].get_visible():  # a colorbar
            return []
        if all(visible.values()):
            return []
        options = []
        x = (
            "box"
            if visible["bottom"] and visible["top"]
            else "bottom" if visible["bottom"] else "top" if visible["top"] else None
        )
        y = (
            "box"
            if visible["left"] and visible["right"]
            else "left" if visible["left"] else "right" if visible["right"] else None
        )
        options.append(f"axis x line*={x}" if x else "axis x line=none")
        options.append(f"axis y line*={y}" if y else "axis y line=none")
        return options

    def _ticks(self, name) -> list[str]:
        ax = self.ax
        axis = getattr(ax, f"{name}axis")
        if not axis.get_visible():
            return [f"{name}tick=\\empty", f"{name}ticklabels={{}}"]
        options = []
        locator, formatter = axis.get_major_locator(), axis.get_major_formatter()
        lo, hi = sorted(getattr(ax, f"get_{name}lim")())
        # the ticks where Matplotlib has them; their labels written by pgfplots,
        # unless Matplotlib's formatter writes something else than numbers
        tolerance = 1e-9 * abs(hi - lo)
        locs = [
            loc
            for loc in axis.get_majorticklocs()
            if lo - tolerance <= loc <= hi + tolerance
        ]
        if isinstance(locator, mticker.NullLocator) or not locs:
            options.append(f"{name}tick=\\empty")
        else:
            options.append(
                f"{name}tick={{{','.join(f'{float(loc):.10g}' for loc in locs)}}}"
            )
            if not isinstance(formatter, _AUTO_FORMATTERS):
                labels = formatter.format_ticks(locs)
                options.append(
                    f"{name}ticklabels={{{','.join('{' + latex(lab) + '}' for lab in labels)}}}"
                )
        ticks = axis.get_major_ticks()
        if ticks:
            tick = ticks[0]
            first = tick.tick1line.get_visible()
            second = tick.tick2line.get_visible()
            side = "both" if first and second else "left" if first else "right"
            if not first and not second:
                options.append(f"major {name} tick style={{draw=none}}")
            else:
                options.append(f"{name}tick pos={side}")
            label1, label2 = tick.label1.get_visible(), tick.label2.get_visible()
            if not label1 and not label2:
                options.append(f"{name}ticklabels={{}}")
                options.append(f"scaled {name} ticks=false")
            else:
                options.append(
                    f"{name}ticklabel pos={'right' if label2 and not label1 else 'left'}"
                )
                options.append(
                    f"{name} tick label style={{font={{{_font(tick.label1.get_fontsize())}}}}}"
                )
            if name == "x":
                direction = getattr(tick, "_tickdir", "out")
                align = {"in": "inside", "out": "outside", "inout": "center"}.get(
                    direction, "outside"
                )
                options.append(f"tick align={align}")
                options.append(f"major tick length={_pt(tick._size)}")
        if axis.get_label_position() in ("top", "right"):
            options.append(f"{name}label near ticks")
            if not any(option.startswith(f"{name}ticklabel pos") for option in options):
                options.append(f"{name}ticklabel pos=right")
        return options

    def _labels(self) -> list[str]:
        options = []
        for key, label in (
            ("xlabel", self.ax.xaxis.label),
            ("ylabel", self.ax.yaxis.label),
            ("title", self.ax.title),
        ):
            style = [f"font={{{_font(label.get_fontsize())}}}"]
            color, _ = self.colors(label.get_color())
            if color is not None and color != "mpl000000":
                style.append(f"text={color}")
            if is_multiline(label.get_text()):
                style.append("align=center")
            options.append(f"{key} style={{{', '.join(style)}}}")
        return options

    def _background(self) -> list[str]:
        if self.colorbar is not None:
            return []
        patch = self.ax.patch
        if not patch.get_visible():
            return []
        color, opacity = self.colors(patch.get_facecolor())
        if color is None or color == "mplFFFFFF":
            return []
        fill = f"fill={color}" + (
            f", fill opacity={opacity:.3g}" if opacity < 1 else ""
        )
        return [f"axis background/.style={{{fill}}}"]

    def _grid(self) -> list[str]:
        options = []
        style = None
        for name in ("x", "y"):
            axis = getattr(self.ax, f"{name}axis")
            ticks = axis.get_major_ticks()
            if ticks and ticks[0].gridline.get_visible():
                options.append(f"{name}majorgrids")
                style = style or ticks[0].gridline
        if style is not None:
            color, opacity = self.colors(style.get_color())
            alpha = style.get_alpha()
            if alpha is not None:
                opacity *= alpha
            parts = [f"draw={color}", f"line width={_pt(style.get_linewidth())}"]
            if opacity < 1:
                parts.append(f"draw opacity={opacity:.3g}")
            dash = _dash(style._dash_pattern) if style.is_dashed() else None
            parts.append(dash or "solid")
            options.append(f"major grid style={{{', '.join(parts)}}}")
        return options

    # -- the legend --------------------------------------------------------
    def _legend_setup(self):
        legend = self.ax.get_legend()
        if legend is None or not legend.get_visible():
            return
        handles = getattr(legend, "legend_handles", None)
        if handles is None:  # Matplotlib < 3.7
            handles = legend.legendHandles
        self.legend_entries = [
            (handle, text.get_text())
            for handle, text in zip(handles, legend.get_texts())
        ]
        bbox = legend.get_window_extent().transformed(self.ax.transAxes.inverted())
        center_x, center_y = (bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2
        vertical = "north" if center_y > 0.5 else "south"
        horizontal = "east" if center_x > 0.5 else "west"
        at = (
            bbox.x1 if horizontal == "east" else bbox.x0,
            bbox.y1 if vertical == "north" else bbox.y0,
        )
        style = []
        frame = legend.get_frame()
        if legend.get_frame_on() and frame.get_visible():
            face, face_opacity = self.colors(frame.get_facecolor())
            edge, _ = self.colors(frame.get_edgecolor())
            style.append(f"fill={face}" if face else "fill=none")
            if face_opacity < 1:
                style.append(f"fill opacity={face_opacity:.3g}, text opacity=1")
            style.append(f"draw={edge}" if edge else "draw=none")
            if "round" in type(frame.get_boxstyle()).__name__.lower():
                style.append("rounded corners=2pt")
        else:
            style.extend(["draw=none", "fill=none"])
        if legend.get_texts():
            style.append(f"font={{{_font(legend.get_texts()[0].get_fontsize())}}}")
        style.append("cells={anchor=west}")
        self.axis.set_legend(
            at=at,
            anchor=f"{vertical} {horizontal}",
            columns=getattr(legend, "_ncols", 1) or None,
            style=style,
        )

    def _legend_label(self, artist) -> str:
        """Plots never make legend entries themselves; see :meth:`_legend_entries`."""
        return ""

    def _legend_entries(self):
        """The legend as Matplotlib draws it: an image and a text per entry.

        The plots are left out of the legend (``forget plot``), so that the
        entries keep the legend's order and style, also for artists drawn as
        images.
        """
        for handle, label in self.legend_entries:
            options = self._legend_image(handle)
            self.axis.add_raw(
                f"\\addlegendimage{{{', '.join(options)}}}\n"
                f"\\addlegendentry{{{latex(label)}}}"
            )

    def _legend_image(self, handle) -> list[str]:
        if isinstance(handle, Line2D):
            options = self._line_options(handle)
            if options:
                return options
        elif isinstance(handle, Patch):
            face = handle.get_facecolor() if handle.get_fill() else "none"
            # the patch's colors include its alpha
            options = self._area_options(
                face,
                handle.get_edgecolor(),
                handle.get_linewidth(),
                _dash(handle._dash_pattern),
            )
            if options:
                return options
        elif isinstance(handle, PathCollection):
            handle.update_scalarmappable()
            faces, edges = handle.get_facecolors(), handle.get_edgecolors()
            sizes = handle.get_sizes()
            widths = np.atleast_1d(handle.get_linewidths())
            paths = handle.get_paths()
            return ["only marks"] + self._mark(
                _marker_name(paths[0]) if len(paths) else "o",
                float(np.sqrt(sizes[0])) if len(sizes) else 6.0,
                faces[0] if len(faces) else "none",
                edges[0] if len(edges) and not isinstance(edges, str) else "none",
                float(widths[0]) if len(widths) else 1.0,
            )
        elif isinstance(handle, LineCollection):
            colors = handle.get_colors()
            widths = np.atleast_1d(handle.get_linewidths())
            styles = handle.get_linestyles()
            stroke = self._stroke(
                colors[0] if len(colors) else "black",
                float(widths[0]) if len(widths) else 1.0,
                _dash(styles[0]) if len(styles) else None,
            )
            if stroke:
                return stroke + ["mark=none"]
        return ["empty legend"]

    # -- what is drawn -----------------------------------------------------
    def _contents(self):
        ax = self.ax
        children = list(getattr(ax, "_children", []))
        if not children:  # an older Matplotlib
            children = (
                ax.collections
                + ax.patches
                + ax.lines
                + ax.texts
                + ax.images
                + ax.tables
            )
        children = [child for child in children if child.get_visible()]
        children.sort(key=lambda artist: artist.get_zorder())  # stable: drawing order
        pending_images = []
        for artist in children:
            emit = None if self.colorbar is not None else self._vector(artist)
            if emit is None:
                pending_images.append(artist)
                continue
            self._flush_images(pending_images)
            pending_images = []
            emit()
        self._flush_images(pending_images)

    def _flush_images(self, artists):
        if not artists:
            return
        reasons = sorted(
            {
                type(artist).__name__
                for artist in artists
                if not isinstance(artist, _ALWAYS_RASTER)
            }
        )
        if reasons and self.colorbar is None:
            warnings.warn(
                f"drawing {', '.join(reasons)} as an image in the pgfplots axis",
                TikzConversionWarning,
                stacklevel=6,
            )
        data = self.parent.render_only(artists, self.inches)
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.axis.add_graphics(
            float(min(xlim)),
            float(max(xlim)),
            float(min(ylim)),
            float(max(ylim)),
            data=data,
            plot_options=["forget plot"],
        )
        self.has_images = True

    def _vector(self, artist):
        """A function adding ``artist`` to the axis as vector graphics, or None."""
        if isinstance(artist, Line2D):
            return self._line(artist)
        if isinstance(artist, ContourSet):
            return self._contour_lines(artist)
        if isinstance(artist, PathCollection):
            return self._scatter(artist)
        if isinstance(artist, LineCollection):
            return self._line_collection(artist)
        if type(artist) is PolyCollection or type(artist).__name__ in (
            "FillBetweenPolyCollection",
        ):
            return self._polygons(artist)
        if isinstance(artist, Patch):
            return self._patch(artist)
        if isinstance(artist, Text):
            return self._text(artist)
        return None

    # coordinates
    def _view(self):
        """The axes box in display coordinates, grown by one box size on each side.

        Geometry is clipped to it: what lies further out is not shown, and its
        coordinates could exceed the largest dimension TeX can hold.
        """
        box = self.ax.bbox
        return (
            box.x0 - box.width,
            box.y0 - box.height,
            box.x1 + box.width,
            box.y1 + box.height,
        )

    def _inside(self, display):
        x0, y0, x1, y1 = self._view()
        display = np.asarray(display, dtype=float).reshape(-1, 2)
        with np.errstate(invalid="ignore"):
            return (
                (display[:, 0] >= x0)
                & (display[:, 0] <= x1)
                & (display[:, 1] >= y0)
                & (display[:, 1] <= y1)
            )

    def _clip_line(self, display, simplify=False):
        """A polyline in display coordinates clipped to :meth:`_view`; gaps as nan.

        With ``simplify``, it is also simplified as Matplotlib does when drawing.
        """
        display = np.asarray(display, dtype=float).reshape(-1, 2)
        if not simplify and self._inside(display).all():
            return display
        finite = np.isfinite(display).all(axis=1)
        if not finite.any():
            return np.empty((0, 2))
        path = Path(display).cleaned(
            remove_nans=True, clip=self._view(), simplify=simplify
        )
        return _with_gaps(path)

    def _clip_polygon(self, display):
        """A polygon in display coordinates clipped to :meth:`_view` (empty if outside)."""
        display = np.asarray(display, dtype=float).reshape(-1, 2)
        display = display[np.isfinite(display).all(axis=1)]
        if len(display) < 3 or self._inside(display).all():
            return display
        closed = Path(np.vstack([display, display[:1]]), closed=True)
        clipped = closed.clip_to_bbox(Bbox.from_extents(*self._view()))
        polygons = clipped.to_polygons()
        return polygons[0] if polygons else np.empty((0, 2))

    def _to_data(self, display):
        display = np.asarray(display, dtype=float).reshape(-1, 2)
        with np.errstate(all="ignore"):
            data = self.ax.transData.inverted().transform(display)
        # display -> data leaves round-off (1e-16 for 0): snap it on linear axes
        for index, name in enumerate(("x", "y")):
            if getattr(self.ax, f"get_{name}scale")() == "linear":
                lo, hi = getattr(self.ax, f"get_{name}lim")()
                tiny = 1e-9 * abs(hi - lo)
                with np.errstate(invalid="ignore"):
                    data[np.abs(data[:, index]) < tiny, index] = 0.0
        return data

    def _add(self, data, options, label="", cycle=False, clip=True):
        data = np.asarray(data, dtype=float)
        if not clip:
            self.axis.add_raw(self._draw(data, options, cycle))
            return
        self.axis.add_plot(
            x=data[:, 0].tolist(),
            y=data[:, 1].tolist(),
            label=label,
            options=list(options) + ["forget plot"],
            cycle=cycle,
            precision=self.options["precision"],
        )

    def _draw(self, data, options, cycle) -> str:
        """A ``\\draw`` path in axis coordinates, which pgfplots does not clip."""
        precision = self.options["precision"]
        keep = [
            option
            for option in options
            if not option.startswith(
                ("mark", "only marks", "forget plot", "area legend")
            )
        ]
        parts = []
        connect = False
        for x, y in data:
            if not (np.isfinite(x) and np.isfinite(y)):
                connect = False
                continue
            point = f"(axis cs:{format_number(float(x), precision)},{format_number(float(y), precision)})"
            parts.append(("-- " if connect else "") + point)
            connect = True
        if cycle and parts:
            parts.append("-- cycle")
        return f"\\draw[{', '.join(keep)}] {' '.join(parts)};"

    def _stroke(self, color, linewidth, dash, alpha=None) -> list[str] | None:
        name, opacity = self.colors(color)
        if name is None or linewidth <= 0:
            return None
        if alpha is not None:
            opacity *= alpha
        options = [f"draw={name}", f"line width={_pt(linewidth)}"]
        if opacity < 1:
            options.append(f"draw opacity={opacity:.3g}")
        options.append(dash or "solid")
        return options

    def _fill(self, color, alpha=None) -> list[str] | None:
        name, opacity = self.colors(color)
        if name is None:
            return None
        if alpha is not None:
            opacity *= alpha
        options = [f"fill={name}"]
        if opacity < 1:
            options.append(f"fill opacity={opacity:.3g}")
        return options

    def _mark(self, marker, size, face, edge, edge_width, alpha=None) -> list[str]:
        """``mark=...`` options for a Matplotlib marker of ``size`` points."""
        if isinstance(marker, str) and marker in _MARKS:
            filled_mark, hollow_mark, rotation, factor = _MARKS[marker]
        else:
            filled_mark, hollow_mark, rotation, factor = _MARKS["o"]
        fill = self._fill(face, alpha)
        mark = filled_mark if fill else hollow_mark
        mark_options = ["solid"]
        if rotation:
            mark_options.append(f"rotate={rotation}")
        mark_options.extend(fill or ["fill=none"])
        stroke = self._stroke(edge, edge_width, None, alpha)
        if stroke:
            mark_options.extend(option for option in stroke if option != "solid")
        else:
            mark_options.append("draw=none")
        return [
            f"mark={mark}",
            f"mark size={_pt(max(size * factor, 0.1))}",
            f"mark options={{{', '.join(mark_options)}}}",
        ]

    # artists
    def _line(self, line):
        path = line.get_path()
        if not len(path.vertices):
            return lambda: None
        display = line.get_transform().transform(path.vertices)
        marker = line.get_marker()
        has_marker = marker not in _NO_MARKER and line.get_markersize() > 0
        if has_marker:  # the markers stay where they are; far ones are left out
            display = np.where(self._inside(display)[:, None], display, np.nan)
        else:
            display = self._clip_line(display, simplify=len(display) > 2000)
        if len(display) > self.options["max_points"]:
            return None
        data = self._to_data(display)
        options = self._line_options(line)
        if options is None:
            return lambda: None
        label = self._legend_label(line)
        clip = line.get_clip_on()
        return lambda: self._add(data, options, label, clip=clip)

    def _line_options(self, line) -> list[str] | None:
        """The style of a Line2D as plot options; None if it draws nothing."""
        marker = line.get_marker()
        has_marker = marker not in _NO_MARKER and line.get_markersize() > 0
        has_line = line.get_linestyle() not in _NO_LINE and line.get_linewidth() > 0
        alpha = line.get_alpha()
        options = []
        if has_line:
            dash = _dash(line._dash_pattern) if line.is_dashed() else None
            stroke = self._stroke(line.get_color(), line.get_linewidth(), dash, alpha)
            if stroke is None:
                has_line = False
            else:
                options.extend(stroke)
        if not has_line:
            if not has_marker:
                return None
            options.append("only marks")
        if has_marker:
            face = line.get_markerfacecolor()
            if line.get_fillstyle() == "none":
                face = "none"
            options.extend(
                self._mark(
                    marker,
                    line.get_markersize(),
                    face,
                    line.get_markeredgecolor(),
                    line.get_markeredgewidth(),
                    alpha,
                )
            )
        else:
            options.append("mark=none")
        return options

    def _contour_lines(self, contours):
        if contours.filled:
            return None
        paths = contours.get_paths()
        transform = contours.get_transform()
        colors = contours.get_edgecolor()
        widths = np.atleast_1d(contours.get_linewidth())
        styles = contours.get_linestyle()
        pieces = []
        total = 0
        for index, path in enumerate(paths):
            lines = transform.transform_path(path).to_polygons(closed_only=False)
            if not lines:
                continue
            joined = []
            for line in lines:
                joined.append(self._clip_line(line))
                joined.append([[np.nan, np.nan]])
            display = np.concatenate(joined[:-1])
            total += len(display)
            dash = _dash(styles[index % len(styles)]) if len(styles) else None
            stroke = self._stroke(
                colors[index % len(colors)], widths[index % len(widths)], dash
            )
            if stroke is not None:
                pieces.append((self._to_data(display), stroke + ["mark=none"]))
        if total > self.options["max_points"]:
            return None
        texts = list(getattr(contours, "labelTexts", []))

        def emit():
            for data, options in pieces:
                self._add(data, options)
            for text in texts:
                self._text(text)()

        return emit

    def _scatter(self, collection):
        offsets = np.ma.asarray(collection.get_offsets()).filled(np.nan)
        paths = collection.get_paths()
        count = len(offsets)
        if count > self.options["max_markers"] or len(paths) != 1:
            return None
        if count == 0:
            return lambda: None
        display = collection.get_offset_transform().transform(offsets)
        inside = self._inside(display)
        data = self._to_data(display)
        collection.update_scalarmappable()
        faces = collection.get_facecolors()
        edges = collection.get_edgecolors()
        if isinstance(edges, str) or len(edges) == 0:
            edges = np.zeros((1, 4))
        sizes = collection.get_sizes()
        widths = np.atleast_1d(collection.get_linewidths())
        marker = _marker_name(paths[0])

        def rows(values):
            values = np.asarray(values)
            if len(values) == 0:
                return np.zeros((count, 4))
            return values[np.arange(count) % len(values)]

        faces, edges = rows(faces), rows(edges)
        # marker sizes are areas in pt^2; marks are sized by their diameter
        if len(sizes):
            sizes = np.sqrt(np.asarray(sizes, dtype=float))[
                np.arange(count) % len(sizes)
            ]
        else:
            sizes = np.full(count, 6.0)
        widths = widths[np.arange(count) % len(widths)]
        # one plot per style: colors rounded to 1/63, sizes to quarter points
        keys = np.column_stack(
            [
                np.round(faces * 63),
                np.round(edges * 63),
                np.round(sizes * 4),
                np.round(widths * 4),
            ]
        )
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = np.ravel(inverse)
        if len(unique) > self.options["max_items"]:
            return None
        label = self._legend_label(collection)
        groups = []
        for group in range(len(unique)):
            members = np.flatnonzero((inverse == group) & inside)
            if not len(members):
                continue
            first = members[0]
            options = ["only marks"] + self._mark(
                marker,
                sizes[first],
                faces[first],
                edges[first],
                widths[first],
            )
            groups.append((data[members], options))

        def emit():
            for index, (points, options) in enumerate(groups):
                self._add(points, options, label if index == 0 else "")

        return emit

    def _line_collection(self, collection):
        offsets = collection.get_offsets()
        if len(collection.get_transforms()) or (len(offsets) and np.any(offsets)):
            return None
        collection.update_scalarmappable()
        segments = collection.get_segments()
        if not segments:
            return lambda: None
        transform = collection.get_transform()
        colors = collection.get_colors()
        widths = np.atleast_1d(collection.get_linewidths())
        styles = collection.get_linestyles()
        alpha = None  # the colors include the collection's alpha
        # consecutive segments of one style become one plot, separated by nan
        runs = []
        for index, segment in enumerate(segments):
            if len(segment) == 0:
                continue
            color = colors[index % len(colors)] if len(colors) else (0, 0, 0, 0)
            width = widths[index % len(widths)]
            style = styles[index % len(styles)] if len(styles) else (0, None)
            key = (tuple(np.round(color, 4)), float(width), str(style))
            display = self._clip_line(
                transform.transform(np.asarray(segment, dtype=float))
            )
            if not len(display):
                continue
            if runs and runs[-1][0] == key:
                runs[-1][1].append(display)
            else:
                runs.append((key, [display], color, width, style))
        if len(runs) > self.options["max_items"]:
            return None
        pieces = []
        for key, displays, color, width, style in runs:
            stroke = self._stroke(color, width, _dash(style), alpha)
            if stroke is None:
                continue
            joined = []
            for display in displays:
                joined.extend([display, [[np.nan, np.nan]]])
            pieces.append(
                (self._to_data(np.concatenate(joined[:-1])), stroke + ["mark=none"])
            )
        if sum(len(data) for data, _ in pieces) > self.options["max_points"]:
            return None
        label = self._legend_label(collection)

        def emit():
            for index, (data, options) in enumerate(pieces):
                self._add(data, options, label if index == 0 else "")

        return emit

    def _polygons(self, collection):
        offsets = collection.get_offsets()
        if len(collection.get_transforms()) or (len(offsets) and np.any(offsets)):
            return None
        paths = collection.get_paths()
        if len(paths) > self.options["max_items"]:
            return None
        collection.update_scalarmappable()
        transform = collection.get_transform()
        faces = collection.get_facecolors()
        edges = collection.get_edgecolors()
        widths = np.atleast_1d(collection.get_linewidths())
        styles = collection.get_linestyles()
        pieces = []
        for index, path in enumerate(paths):
            face = faces[index % len(faces)] if len(faces) else "none"
            edge = edges[index % len(edges)] if len(edges) else "none"
            width = widths[index % len(widths)]
            style = styles[index % len(styles)] if len(styles) else (0, None)
            options = self._area_options(face, edge, width, _dash(style))
            if options is None:
                continue
            for polygon in transform.transform_path(path).to_polygons():
                polygon = self._clip_polygon(polygon)
                if len(polygon):
                    pieces.append((self._to_data(polygon), options))
        label = self._legend_label(collection)

        def emit():
            for index, (data, options) in enumerate(pieces):
                self._add(data, options, label if index == 0 else "", cycle=True)

        return emit

    def _area_options(self, face, edge, width, dash, alpha=None):
        fill = self._fill(face, alpha)
        stroke = self._stroke(edge, width, dash, alpha)
        if fill is None and stroke is None:
            return None
        options = list(fill or ["fill=none"])
        options.extend(stroke or ["draw=none"])
        options.extend(["mark=none", "area legend"])
        return options

    def _patch(self, patch, clip=None):
        if clip is None:
            clip = patch.get_clip_on()
        transform = patch.get_transform()
        path = patch.get_path()
        fillable = None
        if isinstance(patch, FancyArrowPatch):
            try:
                paths, fillable = patch._get_path_in_displaycoord()
            except Exception:  # pragma: no cover - private Matplotlib API
                return None
            if not np.iterable(fillable):
                paths, fillable = [paths], [fillable]
            transform = None
        else:
            paths, fillable = [path], [True]
        # the patch's colors include its alpha
        face = patch.get_facecolor() if patch.get_fill() else "none"
        dash = _dash(patch._dash_pattern)
        pieces = []
        for sub_path, can_fill in zip(paths, fillable):
            if transform is not None:
                sub_path = transform.transform_path(sub_path)
            polygons = sub_path.to_polygons(closed_only=False)
            options = self._area_options(
                face if can_fill else "none",
                patch.get_edgecolor(),
                patch.get_linewidth(),
                dash,
            )
            if options is None:
                continue
            for polygon in polygons:
                closed = can_fill and len(polygon) > 2
                polygon = (
                    self._clip_polygon(polygon) if closed else self._clip_line(polygon)
                )
                if len(polygon):
                    pieces.append((self._to_data(polygon), options, closed))
        label = self._legend_label(patch)

        def emit():
            for index, (data, options, closed) in enumerate(pieces):
                self._add(
                    data, options, label if index == 0 else "", cycle=closed, clip=clip
                )

        return emit

    def _text(self, text):
        if not text.get_text().strip():
            return lambda: None
        if isinstance(text, Annotation):
            text.update_positions(self.figure._get_renderer())
        arrow = (
            getattr(text, "arrow_patch", None) if isinstance(text, Annotation) else None
        )
        arrow_emit = None
        if arrow is not None and arrow.get_visible():
            arrow_emit = self._patch(arrow, clip=False)
            if arrow_emit is None:
                return None
        display = _text_display_position(text)
        if not self.figure.bbox.contains(*display):
            return arrow_emit or (lambda: None)
        fx, fy = self.ax.transAxes.inverted().transform(display)
        node = _text_node(text, self.colors, f"(axis description cs:{fx:.5g},{fy:.5g})")

        def emit():
            if arrow_emit is not None:
                arrow_emit()
            self.axis.add_raw(node)

        return emit


# artists drawn as images without a warning: images, meshes, filled contours,
# quivers and the like have no better form in pgfplots
_ALWAYS_RASTER = (AxesImage, Collection)


def _marker_name(path) -> str:
    for marker in _MARKS:
        style = MarkerStyle(marker)
        candidate = style.get_path().transformed(style.get_transform())
        if candidate.vertices.shape == path.vertices.shape and np.allclose(
            candidate.vertices, path.vertices
        ):
            return marker
    return "o"


def _with_gaps(path):
    """The vertices of a path as one array, its separate pieces joined by nan."""
    out = []
    for vertex, code in zip(path.vertices, path.codes):
        if code == Path.STOP:
            break
        if code == Path.MOVETO and out:
            out.append((np.nan, np.nan))
        out.append(tuple(vertex))
    return np.asarray(out, dtype=float).reshape(-1, 2)
