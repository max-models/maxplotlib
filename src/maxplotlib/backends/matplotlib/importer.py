"""Translate Matplotlib artists into backend-independent canvas entries."""

import copy
import os
import pickle
import warnings

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import AxLine
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Polygon, StepPatch
from matplotlib.path import Path
from matplotlib.text import Annotation
from matplotlib.ticker import FixedLocator
from matplotlib.transforms import IdentityTransform

from .import_state import (
    ImportReport,
    ReboundSnapshot,
    capture_axis_state,
    figure_bounds,
    root_figure,
)


def import_matplotlib(
    canvas_cls,
    source,
    *,
    strict=False,
    trusted=False,
    fallback="native",
    **canvas_kwargs,
):
    if fallback not in ("native", "skip", "raster"):
        raise ValueError("fallback must be 'native', 'skip', or 'raster'")
    if isinstance(source, (str, bytes, os.PathLike)) or hasattr(source, "read"):
        if not trusted:
            raise ValueError(
                "Serialized Matplotlib input requires trusted=True; "
                "unpickling can execute arbitrary Python code"
            )
        if isinstance(source, bytes):
            source = pickle.loads(source)
        elif hasattr(source, "read"):
            source = pickle.load(source)
        else:
            with open(source, "rb") as stream:
                source = pickle.load(stream)
    report = ImportReport()

    def unsupported(message, artist=None):
        artist = artist if artist is not None else unsupported.artist
        report.add(
            artist,
            message,
            severity="error" if strict else "warning",
            fallback="skipped",
        )
        if strict:
            error = NotImplementedError(message)
            error.import_report = report
            raise error
        warnings.warn(message, UserWarning, stacklevel=3)

    unsupported.artist = source
    unsupported.report = report
    unsupported.fallback = fallback

    whole_figure = isinstance(source, Figure)
    explicit_shape = None
    if whole_figure:
        axes = list(source.axes)
    elif isinstance(source, Axes):
        axes = [source]
        explicit_shape = (1, 1)
    else:
        try:
            array = np.asarray(source, dtype=object)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Expected a Figure, Axes, or rectangular array of Axes"
            ) from exc
        if array.ndim not in (1, 2):
            raise TypeError("Expected a Figure, Axes, or 1D/2D array of Axes")
        axes = list(array.flat)
        if array.ndim == 2:
            explicit_shape = array.shape
    if not axes:
        raise ValueError("At least one Matplotlib Axes is required")
    if not all(isinstance(ax, Axes) for ax in axes):
        raise TypeError("Every array element must be a Matplotlib Axes")
    if len({id(ax) for ax in axes}) != len(axes):
        raise ValueError("Duplicate Axes are not supported")
    figure = root_figure(source if whole_figure else axes[0].get_figure())
    if any(root_figure(ax.get_figure()) is not figure for ax in axes):
        raise ValueError("All Axes must belong to the same Figure")

    if fallback == "raster":
        return _raster_import(
            canvas_cls, figure, axes, whole_figure, report, **canvas_kwargs
        )

    # Colorbar axes are decorations, not independent data subplots.
    selected = []
    twins = []
    colorbars = []
    # The source figure's creation order identifies the primary axes even when
    # the caller passes the twin first in an array.
    for index, ax in sorted(
        enumerate(axes), key=lambda item: figure.axes.index(item[1])
    ):
        if getattr(ax, "_colorbar", None) is not None:
            colorbars.append(ax._colorbar)
        else:
            parent = next(
                (
                    previous
                    for _, previous in selected
                    if ax in previous._twinned_axes.get_siblings(previous)
                ),
                None,
            )
            if parent is None:
                selected.append((index, ax))
            else:
                direction = "x" if ax.get_shared_x_axes().joined(ax, parent) else "y"
                twins.append((parent, ax, direction))
    selected.sort(key=lambda item: item[0])
    if not selected:
        raise ValueError("No supported Axes to import")

    if explicit_shape is not None:
        nrows, ncols = explicit_shape
        positioned = [(i // ncols, i % ncols, ax) for i, ax in selected]
    else:
        specs = [ax.get_subplotspec() for _, ax in selected]
        common_grid = specs[0].get_gridspec() if specs[0] is not None else None
        simple_grid = common_grid is not None and all(
            spec is not None
            and spec.get_gridspec() is common_grid
            and spec.rowspan.stop - spec.rowspan.start == 1
            and spec.colspan.stop - spec.colspan.start == 1
            for spec in specs
        )
        if simple_grid and len({spec.num1 for spec in specs}) == len(specs):
            nrows, ncols = common_grid.get_geometry()
            positioned = [
                (spec.rowspan.start, spec.colspan.start, ax)
                for spec, (_, ax) in zip(specs, selected)
            ]
        else:
            # Irregular layouts use stable editable slots and retain their
            # actual axes rectangles for the Matplotlib renderer.
            nrows, ncols = 1, len(selected)
            positioned = [(0, i, ax) for i, (_, ax) in enumerate(selected)]

    if "nrows" in canvas_kwargs or "ncols" in canvas_kwargs:
        raise TypeError("nrows and ncols are inferred from the imported axes")
    canvas_kwargs.setdefault("figsize", tuple(figure.get_size_inches()))
    canvas_kwargs.setdefault("dpi", figure.dpi)
    canvas = canvas_cls(nrows=nrows, ncols=ncols, **canvas_kwargs)
    canvas.import_report = report
    canvas._import_layout = {}
    canvas._import_in_layout = {}
    canvas._import_shared_axes = []
    canvas._import_extra_twins = []
    canvas._import_colorbars = []
    canvas._import_figure_artists = []
    source_targets = {}
    source_slots = {}
    for row, col, ax in positioned:
        target = canvas.add_subplot(row=row, col=col)
        source_targets[ax] = target
        source_slots[ax] = (row, col)
        if explicit_shape is None:
            canvas._import_layout[(row, col)] = figure_bounds(ax, figure)
            canvas._import_in_layout[(row, col)] = ax.get_in_layout()
        _import_axes(ax, target, unsupported)
        for parent, twin, direction in twins:
            if parent is ax:
                factory = canvas.twinx if direction == "x" else canvas.twiny
                registry = (
                    canvas._twinx_subplots
                    if direction == "x"
                    else canvas._twiny_subplots
                )
                if (row, col) in registry:
                    from maxplotlib.subfigure.line_plot import LinePlot

                    twin_target = LinePlot()
                    canvas._import_extra_twins.append(
                        ((row, col), direction, twin_target)
                    )
                else:
                    twin_target = factory(row=row, col=col)
                source_targets[twin] = twin_target
                _import_axes(twin, twin_target, unsupported)
    if explicit_shape is None and all(ax.figure is figure for _, ax in selected):
        specs = {source_slots[ax]: ax.get_subplotspec() for _, ax in selected}
        canvas._import_layout_specs = ReboundSnapshot(specs, fig=figure)
        canvas._import_layout_engine = copy.deepcopy(figure.get_layout_engine())
        if canvas._import_layout_engine is None:
            canvas.subplots_adjust(
                **{
                    name: getattr(figure.subplotpars, name)
                    for name in ("left", "right", "bottom", "top", "wspace", "hspace")
                }
            )
    for name in ("x", "y"):
        for index, (_, ax) in enumerate(selected):
            parent = next(
                (
                    previous
                    for _, previous in selected[:index]
                    if getattr(ax, "get_shared_" + name + "_axes")().joined(
                        ax, previous
                    )
                ),
                None,
            )
            if parent is not None:
                canvas._import_shared_axes.append(
                    (name, source_slots[parent], source_slots[ax])
                )
    for colorbar in colorbars:
        owner = getattr(colorbar.mappable, "axes", None)
        if owner is not None and owner not in source_targets:
            unsupported("Colorbar mappable is outside the selected axes", colorbar.ax)
            continue
        canvas._import_colorbars.append(
            _capture_colorbar(colorbar, source_targets.get(owner))
        )
        report.add(colorbar.ax, "Colorbar is linked to its imported mappable")
    if whole_figure:
        canvas._import_figure_patch = ReboundSnapshot(figure.patch, fig=figure)
        canvas._import_figure_style = dict(
            facecolor=figure.get_facecolor(),
            edgecolor=figure.get_edgecolor(),
            linewidth=figure.patch.get_linewidth(),
            frameon=figure.get_frameon(),
        )
        for getter, setter in (
            ("get_suptitle", "suptitle"),
            ("get_supxlabel", "supxlabel"),
            ("get_supylabel", "supylabel"),
        ):
            value = getattr(figure, getter, lambda: "")()
            if value:
                text = getattr(figure, "_" + setter)
                getattr(canvas, setter)(
                    value,
                    x=text.get_position()[0],
                    y=text.get_position()[1],
                    **_text_style(text),
                )
        titles = {figure._suptitle, figure._supxlabel, figure._supylabel}
        for artist in (
            list(figure.legends)
            + list(figure.artists)
            + list(figure.lines)
            + list(figure.patches)
            + list(figure.images)
            + [text for text in figure.texts if text not in titles]
        ):
            canvas._import_figure_artists.append(ReboundSnapshot(artist, fig=figure))
            report.add(
                artist, "Figure decoration retained for Matplotlib", fallback="native"
            )

        def subfigure_decorations(owner):
            for subfigure in owner.subfigs:
                background = ReboundSnapshot(subfigure.patch, fig=subfigure)
                background.payload.set_zorder(
                    min((ax.get_zorder() for ax in figure.axes), default=0) - 1
                )
                canvas._import_figure_artists.append(background)
                for artist in (
                    list(subfigure.texts)
                    + list(subfigure.legends)
                    + list(subfigure.artists)
                    + list(subfigure.lines)
                    + list(subfigure.patches)
                    + list(subfigure.images)
                ):
                    canvas._import_figure_artists.append(
                        ReboundSnapshot(artist, fig=subfigure)
                    )
                subfigure_decorations(subfigure)

        subfigure_decorations(figure)
    return canvas


def _style(artist):
    return dict(
        label=artist.get_label(),
        alpha=copy.deepcopy(artist.get_alpha()),
        zorder=artist.get_zorder(),
        visible=artist.get_visible(),
        gid=artist.get_gid(),
        url=artist.get_url(),
        rasterized=artist.get_rasterized(),
        clip_on=artist.get_clip_on(),
        snap=artist.get_snap(),
        in_layout=artist.get_in_layout(),
    )


def _scatter_marker(path):
    # Keep standard marker names portable to the other renderers.
    for marker in (
        "o",
        "s",
        "^",
        "v",
        "<",
        ">",
        "D",
        "d",
        "*",
        "+",
        "x",
        "p",
        "h",
        "H",
        ".",
        "P",
        "X",
    ):
        style = MarkerStyle(marker)
        candidate = style.get_path().transformed(style.get_transform())
        if np.array_equal(path.vertices, candidate.vertices) and np.array_equal(
            path.codes, candidate.codes
        ):
            return marker
    return copy.deepcopy(path)


def _line_style(line):
    return dict(
        color=line.get_color(),
        linestyle=(
            copy.deepcopy(line._unscaled_dash_pattern)
            if line.is_dashed()
            else line.get_linestyle()
        ),
        linewidth=line.get_linewidth(),
        marker=None if line.get_marker() in ("None", "", " ") else line.get_marker(),
        markersize=line.get_markersize(),
        markerfacecolor=line.get_markerfacecolor(),
        markeredgecolor=line.get_markeredgecolor(),
        markeredgewidth=line.get_markeredgewidth(),
        drawstyle=line.get_drawstyle(),
        dash_capstyle=line.get_dash_capstyle(),
        dash_joinstyle=line.get_dash_joinstyle(),
        solid_capstyle=line.get_solid_capstyle(),
        solid_joinstyle=line.get_solid_joinstyle(),
        markevery=copy.deepcopy(line.get_markevery()),
        gapcolor=line.get_gapcolor(),
        antialiased=line.get_antialiased(),
        markerfacecoloralt=line.get_markerfacecoloralt(),
        fillstyle=line.get_fillstyle(),
        **_style(line),
    )


def _text_style(text):
    kwargs = dict(
        color=text.get_color(),
        fontsize=text.get_fontsize(),
        fontweight=text.get_fontweight(),
        fontstyle=text.get_fontstyle(),
        fontfamily=list(text.get_fontfamily()),
        rotation=text.get_rotation(),
        ha=text.get_ha(),
        va=text.get_va(),
        usetex=text.get_usetex(),
        rotation_mode=text.get_rotation_mode(),
        linespacing=text._linespacing,
        fontstretch=text.get_fontproperties().get_stretch(),
        fontvariant=text.get_fontproperties().get_variant(),
        parse_math=text.get_parse_math(),
        wrap=text.get_wrap(),
    )

    box = text.get_bbox_patch()
    if box is not None:
        kwargs["bbox"] = dict(
            boxstyle=copy.deepcopy(box.get_boxstyle()),
            facecolor=box.get_facecolor(),
            edgecolor=box.get_edgecolor(),
            linewidth=box.get_linewidth(),
            linestyle=box.get_linestyle(),
            alpha=box.get_alpha(),
            hatch=box.get_hatch(),
        )
    return kwargs


def _import_errorbar(container, ax, target, unsupported):
    """Return whether the container was consumed as one semantic entry."""
    line, caps, ranges = container.lines
    # Without a data line the original center of asymmetric errors is lost.
    # Leave these artists for the geometry import instead of inventing centers.
    if line is None:
        return False
    if line.get_transform() != ax.transData or any(
        bars.get_transform() != ax.transData for bars in ranges
    ):
        return False
    components = [line, *caps, *ranges]
    if any(_needs_native_style(artist) or artist.get_visible() != line.get_visible()
           for artist in components):
        return False
    if any(cap.get_marker() not in ("_", "|") for cap in caps):
        return False
    if caps and any(
        any(
            getattr(cap, "get_" + prop)() != getattr(caps[0], "get_" + prop)()
            for prop in ("color", "markersize", "markeredgewidth", "alpha", "visible")
        )
        for cap in caps[1:]
    ):
        return False
    x, y = (np.array(v, copy=True) for v in line.get_data(orig=False))
    errors = {}
    for axis, bars in zip(
        [i for i, flag in enumerate((container.has_xerr, container.has_yerr)) if flag],
        ranges,
    ):
        segments = bars.get_segments()
        if len(segments) != len(x):
            return False
        bounds = np.full((len(x), 2), np.nan)
        for i, segment in enumerate(segments):
            if len(segment) == 2:
                bounds[i] = segment[:, axis]
        centers = x if axis == 0 else y
        error = np.array([centers - bounds[:, 0], bounds[:, 1] - centers])
        if np.any(error < -1e-12):
            unsupported(
                "Error-bar centers are outside their ranges; importing geometry"
            )
            return False
        errors["xerr" if axis == 0 else "yerr"] = np.maximum(error, 0)
    kwargs = _line_style(line)
    kwargs["label"] = container.get_label()
    if any(
        not np.array_equal(bars.get_colors(), ranges[0].get_colors())
        or not np.array_equal(bars.get_linewidths(), ranges[0].get_linewidths())
        for bars in ranges[1:]
    ):
        return False
    if ranges:
        base_zorder = ranges[0].get_zorder()
        delta = line.get_zorder() - base_zorder
        if not np.isclose(abs(delta), 0.1) or any(bar.get_zorder() != base_zorder for bar in ranges):
            return False
        kwargs["zorder"] = base_zorder
        kwargs["barsabove"] = delta < 0
        colors = ranges[0].get_colors()
        widths = ranges[0].get_linewidths()
        if len(colors):
            kwargs["ecolor"] = tuple(colors[0])
        if len(widths):
            kwargs["elinewidth"] = float(widths[0])
    kwargs["capsize"] = caps[0].get_markersize() / 2 if caps else 0
    if caps:
        kwargs["capthick"] = caps[0].get_markeredgewidth()
    target.errorbar(x, y, **errors, **kwargs)
    return True


def _collection_style(collection, index):
    kwargs = _style(collection)
    if index:
        kwargs["label"] = ""
    for key, values in (
        ("facecolor", collection.get_facecolors()),
        ("edgecolor", collection.get_edgecolors()),
        ("linewidth", collection.get_linewidths()),
        ("linestyle", getattr(collection, "_us_linestyles", collection.get_linestyles())),
        ("antialiased", collection._antialiaseds),
    ):
        kwargs[key] = (
            copy.deepcopy(values[index % len(values)]) if len(values) else "none"
        )
    offset, dashes = kwargs["linestyle"]
    kwargs["linestyle"] = (offset, tuple(dashes) if dashes is not None else None)
    return kwargs


def _import_collection(collection, ax, target, unsupported):
    """Import plain line/polygon collections; return False for scatter."""
    if not isinstance(collection, (LineCollection, PolyCollection)):
        return False
    if (
        collection.get_transform() != ax.transData
        or np.any(collection.get_offsets())
        or collection.get_array() is not None
        or np.size(collection.get_transforms()) != 0
        or np.ndim(collection.get_alpha()) != 0
    ):
        _native(collection, ax, target, unsupported)
        return True
    if isinstance(collection, LineCollection):
        for i, segment in enumerate(collection.get_segments()):
            if not len(segment):
                continue
            kwargs = _collection_style(collection, i)
            kwargs.pop("facecolor")
            kwargs["color"] = kwargs.pop("edgecolor")
            target.plot(segment[:, 0].copy(), segment[:, 1].copy(), **kwargs)
    else:
        for i, path in enumerate(collection.get_paths()):
            if path.codes is not None and (
                np.count_nonzero(path.codes == Path.MOVETO) > 1
                or np.any(
                    ~np.isin(path.codes, [Path.MOVETO, Path.LINETO, Path.CLOSEPOLY])
                )
            ):
                _native(collection, ax, target, unsupported)
                return True
            vertices = path.vertices
            if path.codes is not None and path.codes[-1] == Path.CLOSEPOLY:
                vertices = vertices[:-1]
            target.fill(
                vertices[:, 0].copy(),
                vertices[:, 1].copy(),
                hatch=collection.get_hatch(),
                **_collection_style(collection, i),
            )
    return True


def _import_axes(ax, target, unsupported):
    if ax.name != "rectilinear" or type(ax).__module__.startswith("mpl_toolkits."):
        target._import_projection = ReboundSnapshot(ax, fig=ax.figure)
        target._import_projection_artist_ids = {
            name: [id(artist) for artist in getattr(ax, name)]
            for name in (
                "lines",
                "collections",
                "images",
                "patches",
                "texts",
                "artists",
            )
        }
        unsupported.report.add(
            ax,
            "Projection, camera and native artists retained for "
            "Matplotlib; edit through the rendered axes",
            fallback="native",
        )
        return
    consumed = set()
    target.import_groups = {
        id(container): dict(
            type=type(container).__name__,
            label=container.get_label(),
            artist_ids=[id(artist) for artist in container.get_children()],
            orientation=getattr(container, "orientation", None),
            datavalues=copy.deepcopy(getattr(container, "datavalues", None)),
        )
        for container in ax.containers
    }
    # A bar container retains grouping and orientation, unlike raw rectangles.
    for container in _tracked(ax.containers, ax, target, unsupported):
        if isinstance(container, ErrorbarContainer):
            if _import_errorbar(container, ax, target, unsupported):
                consumed.update(container.get_children())
            continue
        if isinstance(container, StemContainer):
            unsupported.report.add(
                container, "Stem components retained as geometry", fallback="geometry"
            )
            continue
        if not isinstance(container, BarContainer):
            unsupported(f"Unsupported {type(container).__name__}; container skipped")
            consumed.update(container.get_children())
            continue
        for i, patch in enumerate(container.patches):
            consumed.add(patch)
            if _needs_native_style(patch):
                _native(patch, ax, target, unsupported)
                target.line_data[-1]["source_order"] = ax.get_children().index(patch)
                continue
            kwargs = _style(patch)
            kwargs.update(
                color=patch.get_facecolor(),
                edgecolor=patch.get_edgecolor(),
                linewidth=patch.get_linewidth(),
                hatch=patch.get_hatch(),
                linestyle=patch.get_linestyle(),
                antialiased=patch.get_antialiased(),
                fill=patch.get_fill(),
                label=container.get_label() if i == 0 else "",
            )
            if container.orientation == "horizontal":
                target.barh(
                    [patch.get_y() + patch.get_height() / 2],
                    [patch.get_width()],
                    left=patch.get_x(),
                    height=patch.get_height(),
                    **kwargs,
                )
            else:
                target.bar(
                    [patch.get_x() + patch.get_width() / 2],
                    [patch.get_height()],
                    bottom=patch.get_y(),
                    width=patch.get_width(),
                    **kwargs,
                )

            target.line_data[-1]["source_order"] = ax.get_children().index(patch)

    for line in _tracked(ax.lines, ax, target, unsupported):
        if line in consumed:
            continue
        if (
            isinstance(line, AxLine)
            or _needs_native_style(line)
            or getattr(line._marker, "_user_transform", None) is not None
        ):
            _native(line, ax, target, unsupported)
            continue
        x, y = line.get_data(orig=False)
        if (
            line.get_transform() == ax.get_yaxis_transform()
            and len(y) == 2
            and y[0] == y[1]
        ):
            target.axhline(y[0], xmin=x[0], xmax=x[1], **_line_style(line))
            continue
        if (
            line.get_transform() == ax.get_xaxis_transform()
            and len(x) == 2
            and x[0] == x[1]
        ):
            target.axvline(x[0], ymin=y[0], ymax=y[1], **_line_style(line))
            continue
        if line.get_transform() != ax.transData:
            _native(line, ax, target, unsupported)
            continue
        target.plot(
            np.array(x, copy=True),
            np.array(y, copy=True),
            **_line_style(line),
        )
    for collection in _tracked(ax.collections, ax, target, unsupported):
        if collection in consumed:
            continue
        if _needs_native_style(collection):
            _native(collection, ax, target, unsupported)
            continue
        if _import_collection(collection, ax, target, unsupported):
            continue
        if not isinstance(collection, PathCollection):
            _native(collection, ax, target, unsupported)
            continue
        paths = collection.get_paths()
        if (
            len(paths) != 1
            or collection.get_offset_transform() != ax.transData
            or not isinstance(collection.get_transform(), IdentityTransform)
        ):
            _native(collection, ax, target, unsupported)
            continue
        offsets = np.ma.asarray(collection.get_offsets()).filled(np.nan)
        kwargs = _style(collection)
        kwargs.update(
            s=collection.get_sizes().copy(),
            marker=_scatter_marker(paths[0]),
            edgecolors=collection.get_edgecolors().copy(),
            linewidths=collection.get_linewidths().copy(),
        )
        values = collection.get_array()
        if values is not None:
            kwargs.update(
                c=values.copy(),
                cmap=copy.copy(collection.get_cmap()),
                norm=copy.deepcopy(collection.norm),
            )
        else:
            colors = collection.get_facecolors()
            if len(colors) == 1:
                kwargs["color"] = tuple(colors[0])
            elif len(colors):
                kwargs["c"] = colors.copy()
            else:
                kwargs["facecolors"] = "none"
        target.scatter(offsets[:, 0].copy(), offsets[:, 1].copy(), **kwargs)
    for image in _tracked(ax.images, ax, target, unsupported):
        if image.get_transform() != ax.transData or _needs_native_style(image):
            _native(image, ax, target, unsupported)
            continue
        target.add_imshow(
            image.get_array().copy(),
            extent=tuple(image.get_extent()),
            origin=image.origin,
            cmap=copy.copy(image.get_cmap()),
            norm=copy.deepcopy(image.norm),
            interpolation=image.get_interpolation(),
            interpolation_stage=getattr(image, "_interpolation_stage", "data"),
            filternorm=image.get_filternorm(),
            filterrad=image.get_filterrad(),
            resample=image.get_resample(),
            **_style(image),
        )
    for text in _tracked(ax.texts, ax, target, unsupported):
        if (
            text.get_bbox_patch() is not None
            or _needs_native_style(text)
            or text.get_wrap()
        ):
            _native(text, ax, target, unsupported)
            continue
        if isinstance(text, Annotation):
            if text.xycoords != "data" or text.anncoords != "data":
                _native(text, ax, target, unsupported)
                continue
            arrowprops = text.arrowprops
            if arrowprops and any(key in arrowprops for key in ("patchA", "patchB")):
                _native(text, ax, target, unsupported)
                continue
            target.annotate(
                text.get_text(),
                xy=tuple(text.xy),
                xytext=tuple(text.get_position()),
                arrowprops=copy.deepcopy(arrowprops),
                annotation_clip=text.get_annotation_clip(),
                **_text_style(text),
                **_style(text),
            )
            continue
        if text.get_transform() != ax.transData:
            _native(text, ax, target, unsupported)
            continue
        target.text(
            *text.get_position(),
            text.get_text(),
            **_text_style(text),
            **_style(text),
        )
    for patch in _tracked(ax.patches, ax, target, unsupported):
        if patch in consumed:
            continue
        if patch.get_data_transform() != ax.transData or _needs_native_style(patch):
            _native(patch, ax, target, unsupported)
            consumed.add(patch)
            continue
        kwargs = dict(
            facecolor=patch.get_facecolor(),
            edgecolor=patch.get_edgecolor(),
            linewidth=patch.get_linewidth(),
            linestyle=patch.get_linestyle(),
            hatch=patch.get_hatch(),
            fill=patch.get_fill(),
            **_style(patch),
        )
        if isinstance(patch, StepPatch):
            data = patch.get_data()
            target.stairs(
                data.values.copy(),
                data.edges.copy(),
                baseline=copy.deepcopy(data.baseline),
                orientation=patch.orientation,
                **kwargs,
            )
            consumed.add(patch)
        elif isinstance(patch, Polygon):
            xy = patch.get_xy()
            target.fill(
                xy[:, 0].copy(), xy[:, 1].copy(), closed=patch.get_closed(), **kwargs
            )
            consumed.add(patch)
        else:
            _native(patch, ax, target, unsupported)
    for artist in _tracked(list(ax.artists) + list(ax.tables), ax, target, unsupported):
        if not isinstance(artist, Legend):
            _native(artist, ax, target, unsupported)
    target._import_child_axes = []
    for child in ax.child_axes:
        _import_child(child, ax, target, unsupported)

    title_kwargs = _text_style(ax.title)
    title_kwargs["x"] = ax.title.get_position()[0]
    title_kwargs["pad"] = ax.titleOffsetTrans.transform((0, 0))[1] * 72 / ax.figure.dpi
    if not ax._autotitlepos:
        title_kwargs["y"] = ax.title.get_position()[1]
    target.set_title(ax.get_title(), **title_kwargs)
    target.set_xlabel(
        ax.get_xlabel(), labelpad=ax.xaxis.labelpad, **_text_style(ax.xaxis.label)
    )
    target.set_ylabel(
        ax.get_ylabel(), labelpad=ax.yaxis.labelpad, **_text_style(ax.yaxis.label)
    )
    target.set_xlim(*ax.get_xlim())
    target.set_ylim(*ax.get_ylim())
    for name in ("x", "y"):
        scale = getattr(ax, f"get_{name}scale")()
        getattr(target, f"set_{name}scale")(
            scale
        )
        axis = getattr(ax, f"{name}axis")
        if isinstance(axis.get_major_locator(), FixedLocator):
            getattr(target, f"set_{name}ticks")(
                axis.get_majorticklocs().copy(),
                labels=axis.get_major_formatter().format_ticks(axis.get_majorticklocs()),
            )
    target.set_grid(
        any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())
    )
    target.set_legend(ax.get_legend() is not None and ax.get_legend().get_visible())
    target.set_facecolor(ax.get_facecolor())
    target.set_axisbelow(ax.get_axisbelow())
    target.set_aspect(ax.get_aspect())
    target.set_visible(ax.get_visible())
    if not ax.axison:
        target.set_axis_off()
    for name in (
        "adjustable",
        "anchor",
        "box_aspect",
        "frame_on",
        "alpha",
        "zorder",
        "rasterized",
        "autoscalex_on",
        "autoscaley_on",
    ):
        getattr(target, "set_" + name)(getattr(ax, "get_" + name)())
    target.set_xmargin(ax.margins()[0])
    target.set_ymargin(ax.margins()[1])
    unsupported.report.add(
        ax,
        "Axis scales, tick and spine styles, and legend layout "
        "retained for Matplotlib",
        fallback="native",
    )
    target._import_grid = target._grid
    target._import_scales = (target._xaxis_scale, target._yaxis_scale)
    target._import_axis_state = capture_axis_state(ax)
    target._import_legends = [
        ReboundSnapshot(legend, ax)
        for legend in ax.get_children()
        if isinstance(legend, Legend)
    ]
    target.line_data.sort(key=lambda entry: entry.get("source_order", 0))
    for entries in target.layered_line_data.values():
        entries.sort(key=lambda entry: entry.get("source_order", 0))


def _needs_native_style(artist):
    box = artist.get_clip_box()
    custom_box = (
        box is not None
        and artist.axes is not None
        and not np.array_equal(box.bounds, artist.axes.bbox.bounds)
    )
    return (
        custom_box
        or artist.get_path_effects()
        or artist.get_agg_filter() is not None
        or artist.get_sketch_params() is not None
        or artist.get_clip_path() is not None
    )


def _native(artist, ax, target, unsupported):
    if unsupported.fallback == "skip" or not type(artist).__module__.startswith(
        ("matplotlib.", "mpl_toolkits.")
    ):
        unsupported(f"Unsupported {type(artist).__name__}; artist skipped", artist)
        return
    snapshot = ReboundSnapshot(artist, ax)
    target._add(
        dict(plot_type="matplotlib_artist", snapshot=snapshot, layer=0, kwargs={}), 0
    )
    unsupported.report.add(
        artist,
        "Editable artist geometry retained for Matplotlib; "
        "other backends require a raster import",
        fallback="native",
    )


def _tracked(artists, ax, target, unsupported):
    order = {id(artist): i for i, artist in enumerate(ax.get_children())}
    for artist in artists:
        unsupported.artist = artist
        start = len(target.line_data)
        yield artist
        entries = target.line_data[start:]
        children = getattr(artist, "get_children", lambda: [])()
        index = order.get(
            id(artist),
            min(
                (order.get(id(child), len(order)) for child in children),
                default=len(order),
            ),
        )
        for entry in entries:
            entry["source_artist_id"] = id(artist)
            entry["source_container_ids"] = [
                identifier
                for identifier, group in target.import_groups.items()
                if identifier == id(artist) or id(artist) in group["artist_ids"]
            ]
            entry.setdefault("source_order", index)
            if isinstance(artist, Artist):
                entry["source_sticky_edges"] = (
                    list(artist.sticky_edges.x),
                    list(artist.sticky_edges.y),
                )
                entry["source_picker"] = artist.get_picker()
        if entries:
            unsupported.report.add(
                artist,
                "Imported " + str(len(entries)) + " plot entries",
                fallback="geometry" if len(entries) > 1 else None,
            )


def _capture_colorbar(colorbar, target):
    axis = (
        colorbar.ax.yaxis if colorbar.orientation == "vertical" else colorbar.ax.xaxis
    )
    return dict(
        target=target,
        artist_id=id(colorbar.mappable),
        standalone=(
            dict(
                norm=copy.deepcopy(colorbar.mappable.norm),
                cmap=copy.copy(colorbar.mappable.get_cmap()),
            )
            if target is None
            else None
        ),
        position=tuple(colorbar.ax.get_position().bounds),
        kwargs=dict(
            orientation=colorbar.orientation,
            extend=colorbar.extend,
            extendfrac=colorbar.extendfrac,
            extendrect=colorbar.extendrect,
            spacing=colorbar.spacing,
            drawedges=colorbar.drawedges,
            boundaries=copy.deepcopy(colorbar.boundaries),
            values=copy.deepcopy(colorbar.values),
        ),
        label=axis.label.get_text(),
        label_style=_text_style(axis.label),
        ticks=copy.deepcopy(colorbar.get_ticks()),
        formatter=ReboundSnapshot(colorbar.formatter, colorbar.ax),
        state=capture_axis_state(colorbar.ax),
    )


def _import_child(child, ax, target, unsupported):
    from matplotlib.axes._secondary_axes import SecondaryAxis

    from maxplotlib.subfigure.line_plot import LinePlot

    if isinstance(child, SecondaryAxis):
        target._import_child_axes.append(
            dict(
                kind="secondary",
                orientation=child._orientation,
                location=child._loc,
                functions=child._functions,
                locator=ReboundSnapshot(child.get_axes_locator(), ax),
                state=capture_axis_state(child),
                xlabel=child.get_xlabel(),
                ylabel=child.get_ylabel(),
            )
        )
    else:
        subplot = LinePlot()
        _import_axes(child, subplot, unsupported)
        bounds = ax.transAxes.inverted().transform_bbox(child.bbox).bounds
        target._import_child_axes.append(
            dict(kind="inset", bounds=tuple(bounds), subplot=subplot,
                 locator=ReboundSnapshot(child.get_axes_locator(), ax))
        )
    unsupported.report.add(
        child, "Child axes retained for Matplotlib", fallback="native"
    )


def _raster_import(canvas_cls, figure, axes, whole_figure, report, **kwargs):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    snapshot = copy.deepcopy(figure)
    if not whole_figure:
        indices = [figure.axes.index(ax) for ax in axes]
        for i, ax in enumerate(list(snapshot.axes)):
            if i not in indices:
                ax.remove()
    FigureCanvasAgg(snapshot).draw()
    rgba = np.asarray(snapshot.canvas.buffer_rgba()).copy()
    kwargs.setdefault("figsize", tuple(figure.get_size_inches()))
    kwargs.setdefault("dpi", figure.dpi)
    canvas = canvas_cls(**kwargs)
    subplot = canvas.add_subplot(row=0, col=0)
    subplot.add_imshow(rgba)
    subplot.set_axis_off()
    subplot.set_grid(False)
    canvas.import_report = report
    report.add(
        figure,
        "Raster snapshot: data, artist editing, vector paths and interactive "
        "state are lost",
        severity="warning",
        fallback="raster",
        backends=("matplotlib", "plotly"),
    )
    return canvas
