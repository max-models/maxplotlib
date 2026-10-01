"""Detached Matplotlib state used by the object importer.

Transforms must be rebound, not frozen in source display coordinates.  The
snapshot memo cuts ownership links before copying and replaces them with tokens;
each render binds those tokens to its new figure/axes.  No source artists are
reparented, and repeated renders do not share mutable Matplotlib state.
"""

import copy
from dataclasses import asdict, dataclass, field

from matplotlib.artist import Artist
from matplotlib.cbook import CallbackRegistry
from matplotlib.collections import Collection
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.table import Table
from matplotlib.text import Text
from matplotlib.transforms import Bbox, BboxTransformTo


@dataclass
class ImportDiagnostic:
    artist_id: int
    artist_type: str
    label: str
    severity: str
    message: str
    fallback: str | None = None
    backends: tuple = ("matplotlib",)


@dataclass
class ImportReport:
    """Import decisions, including losses and backend-specific representations."""

    diagnostics: list = field(default_factory=list)

    def add(
        self,
        artist,
        message,
        *,
        severity="info",
        fallback=None,
        backends=("matplotlib",),
    ):
        item = ImportDiagnostic(
            id(artist),
            type(artist).__name__,
            str(getattr(artist, "get_label", lambda: "")()),
            severity,
            message,
            fallback,
            backends,
        )
        self.diagnostics.append(item)
        return item

    def to_dict(self):
        return asdict(self)


def root_figure(figure):
    while getattr(figure, "figure", figure) is not figure:
        figure = figure.figure
    return figure


def figure_bounds(ax, figure):
    return tuple(
        ax.get_position(original=True)
        .transformed(ax.figure.transSubfigure)
        .transformed(figure.transFigure.inverted())
        .bounds
    )


def _bindings(ax, fig):
    bindings = {
        "figure": fig,
        "transFigure": fig.transFigure,
        "transSubfigure": fig.transSubfigure,
        "dpi_scale_trans": fig.dpi_scale_trans,
        "fig_bbox": fig.bbox,
    }
    if ax is not None:
        bindings.update(
            axes=ax,
            transData=ax.transData,
            transAxes=ax.transAxes,
            transScale=ax.transScale,
            transLimits=ax.transLimits,
            ax_bbox=ax.bbox,
            xaxis=ax.xaxis,
            yaxis=ax.yaxis,
            xaxis_transform=ax.get_xaxis_transform(),
            yaxis_transform=ax.get_yaxis_transform(),
        )
    return bindings


class _BindingToken:
    """Weak-referenceable marker for copied Transform parent links."""


class ReboundSnapshot:
    """Copy a payload without retaining its owning axes, figure or callbacks."""

    def __init__(self, payload, ax=None, fig=None):
        owner = fig if fig is not None else ax.figure
        fig = root_figure(owner)
        self.owner_bounds = None
        self.tokens = {}
        memo = {}
        bindings = _bindings(ax, fig)
        if owner is not fig:
            self.owner_bounds = tuple(
                owner.bbox.transformed(fig.transFigure.inverted()).bounds
            )
            bindings.update(owner=owner, owner_transform=owner.transSubfigure)
        for name, value in bindings.items():
            # Some canonical transforms are aliases of one another.
            if id(value) not in memo:
                self.tokens[name] = memo[id(value)] = _BindingToken()
        artists = []
        if isinstance(payload, Artist):
            artists = payload.findobj()
        for artist in artists:
            for name in ("_remove_method", "stale_callback"):
                value = getattr(artist, name, None)
                if value is not None:
                    memo[id(value)] = None
            callbacks = getattr(artist, "_callbacks", None)
            if callbacks is not None:
                memo[id(callbacks)] = CallbackRegistry()
        self.payload = copy.deepcopy(payload, memo)

    def clone(self, ax=None, fig=None):
        fig = fig if fig is not None else ax.figure
        bindings = _bindings(ax, fig)
        if self.owner_bounds is not None:
            bindings.update(
                owner=fig,
                owner_transform=BboxTransformTo(Bbox.from_bounds(*self.owner_bounds))
                + fig.transFigure,
            )
        return copy.deepcopy(
            self.payload,
            {id(token): bindings[name] for name, token in self.tokens.items()},
        )

    def draw(self, ax, **overrides):
        artist = self.clone(ax)
        artist.set(**overrides)
        clipbox, clippath = artist.get_clip_box(), artist.get_clip_path()
        # add_* establishes the new removal and stale callbacks.
        if isinstance(artist, Collection):
            ax.add_collection(artist, autolim=False)
        elif isinstance(artist, Line2D):
            ax.add_line(artist)
        elif isinstance(artist, Patch):
            ax.add_patch(artist)
        elif isinstance(artist, AxesImage):
            ax.add_image(artist)
        elif isinstance(artist, Table):
            ax.add_table(artist)
        elif isinstance(artist, Text):
            ax._add_text(artist)
        else:
            ax.add_artist(artist)
        artist.set_clip_path(clippath)
        artist.set_clip_box(clipbox)
        return artist


def capture_axis_state(ax):
    """Capture decorations which have no backend-neutral equivalent yet."""
    state = {"axes": {}, "spines": {}, "titles": {}}
    for name in ("x", "y"):
        axis = getattr(ax, name + "axis")
        state["axes"][name] = dict(
            scale=ReboundSnapshot(axis._scale, ax),
            major_locator=ReboundSnapshot(axis.get_major_locator(), ax),
            minor_locator=ReboundSnapshot(axis.get_minor_locator(), ax),
            major_formatter=ReboundSnapshot(axis.get_major_formatter(), ax),
            minor_formatter=ReboundSnapshot(axis.get_minor_formatter(), ax),
            units=copy.deepcopy(axis.get_units()),
            converter=copy.deepcopy(
                axis.get_converter()
                if hasattr(axis, "get_converter")
                else axis.converter
            ),
            major_kw=_tick_params(axis, "major"),
            minor_kw=_tick_params(axis, "minor"),
            ticks={
                which: [
                    _tick_style(tick)
                    for tick in getattr(axis, "get_" + which + "_ticks")()
                ]
                for which in ("major", "minor")
            },
            label_position=axis.get_label_position(),
            offset_style=_text_properties(axis.get_offset_text()),
        )
    for name, spine in ax.spines.items():
        state["spines"][name] = dict(
            visible=spine.get_visible(),
            edgecolor=spine.get_edgecolor(),
            linewidth=spine.get_linewidth(),
            linestyle=spine.get_linestyle(),
            bounds=spine.get_bounds(),
        )
        if spine.spine_type in ("left", "right", "top", "bottom"):
            state["spines"][name]["position"] = copy.deepcopy(spine.get_position())
    for loc, title in (
        ("left", ax._left_title),
        ("center", ax.title),
        ("right", ax._right_title),
    ):
        state["titles"][loc] = (
            title.get_text(),
            _text_properties(title),
            title.get_position(),
        )
    state["autotitlepos"] = ax._autotitlepos
    state["label_coords"] = {
        name: (
            axis.label.get_position(),
            ReboundSnapshot(axis.label.get_transform(), ax),
        )
        for name, axis in (("x", ax.xaxis), ("y", ax.yaxis))
        if not axis._autolabelpos
    }
    return state


def _text_properties(text):
    return dict(
        fontproperties=copy.deepcopy(text.get_fontproperties()),
        color=text.get_color(),
        rotation=text.get_rotation(),
        rotation_mode=text.get_rotation_mode(),
        horizontalalignment=text.get_ha(),
        verticalalignment=text.get_va(),
        visible=text.get_visible(),
        usetex=text.get_usetex(),
    )


def _tick_params(axis, which):
    tick = getattr(axis, "get_" + which + "_ticks")(1)[0]
    defaults = dict(
        length=tick._size,
        width=tick._width,
        pad=tick._base_pad,
        direction=tick._tickdir,
        color=tick.tick1line.get_color(),
    )
    defaults.update(copy.deepcopy(getattr(axis, "_" + which + "_tick_kw")))
    return defaults


def _tick_line_style(line):
    return {
        name: getattr(line, "get_" + name)()
        for name in (
            "color",
            "marker",
            "markersize",
            "markeredgewidth",
            "markeredgecolor",
            "visible",
            "zorder",
        )
    }


def _tick_style(tick):
    return dict(
        tick1line=_tick_line_style(tick.tick1line),
        tick2line=_tick_line_style(tick.tick2line),
        label1=_text_properties(tick.label1),
        label2=_text_properties(tick.label2),
        gridline=dict(
            visible=tick.gridline.get_visible(),
            color=tick.gridline.get_color(),
            linewidth=tick.gridline.get_linewidth(),
            linestyle=tick.gridline.get_linestyle(),
            alpha=tick.gridline.get_alpha(),
        ),
    )


def apply_axis_state(ax, state, *, units=True):
    for name, props in state["spines"].items():
        props = copy.deepcopy(props)
        bounds = props.pop("bounds")
        ax.spines[name].set(**props)
        if bounds is not None:
            ax.spines[name].set_bounds(*bounds)
    for name, settings in state["axes"].items():
        axis = getattr(ax, name + "axis")
        # Scales are installed before locators/formatters: changing a scale
        # installs defaults and would otherwise erase the imported tick setup.
        getattr(ax, "set_" + name + "scale")(settings["scale"].clone(ax))
        if units:
            converter = copy.deepcopy(settings["converter"])
            if hasattr(axis, "set_converter"):
                if not getattr(axis, "_converter_is_explicit", False):
                    axis.set_converter(converter)
            else:
                axis.converter = converter
            axis.set_units(copy.deepcopy(settings["units"]))
        for kind in ("major", "minor"):
            getattr(axis, "set_" + kind + "_locator")(
                settings[kind + "_locator"].clone(ax)
            )
            getattr(axis, "set_" + kind + "_formatter")(
                settings[kind + "_formatter"].clone(ax)
            )
            axis.set_tick_params(which=kind, **copy.deepcopy(settings[kind + "_kw"]))
            ticks = getattr(axis, "get_" + kind + "_ticks")()
            styles = settings["ticks"][kind]
            for i, tick in enumerate(ticks):
                if styles:
                    style = styles[min(i, len(styles) - 1)]
                    for part, props in style.items():
                        getattr(tick, part).set(**props)
        axis.set_label_position(settings["label_position"])
        axis.get_offset_text().set(**settings["offset_style"])
    for loc, (text, props, position) in (
        state["titles"].items() if hasattr(ax, "set_title") else ()
    ):
        title = ax.set_title(text, loc=loc, **props)
        title.set_position(position)
    ax._autotitlepos = state["autotitlepos"]
    for name, (position, transform) in state["label_coords"].items():
        getattr(ax, name + "axis").set_label_coords(
            *position, transform=transform.clone(ax)
        )
