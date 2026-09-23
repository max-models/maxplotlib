"""Translate Matplotlib artists into backend-independent canvas entries."""

import copy
import warnings

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Polygon, StepPatch
from matplotlib.path import Path
from matplotlib.text import Annotation
from matplotlib.ticker import FixedLocator
from matplotlib.transforms import IdentityTransform


def import_matplotlib(canvas_cls, source, *, strict=False, **canvas_kwargs):
    def unsupported(message):
        if strict:
            raise NotImplementedError(message)
        warnings.warn(message, UserWarning, stacklevel=3)

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
    figure = source if whole_figure else axes[0].get_figure()
    if any(ax.get_figure() is not figure for ax in axes):
        raise ValueError("All Axes must belong to the same Figure")

    # Colorbar axes are decorations, not independent data subplots.
    selected = []
    twins = []
    # The source figure's creation order identifies the primary axes even when
    # the caller passes the twin first in an array.
    for index, ax in sorted(
        enumerate(axes), key=lambda item: figure.axes.index(item[1])
    ):
        if getattr(ax, "_colorbar", None) is not None:
            unsupported("Colorbar axes are not imported")
        elif ax.name != "rectilinear":
            unsupported(f"Unsupported {ax.name!r} projection; axes skipped")
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
                if any(p is parent and d == direction for p, _, d in twins):
                    unsupported("Multiple twins in the same direction are not imported")
                else:
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
            if len(selected) > 1:
                unsupported(
                    "Irregular, nested, or spanning subplot layout cannot be preserved; "
                    "importing axes in one row"
                )
            nrows, ncols = 1, len(selected)
            positioned = [(0, i, ax) for i, (_, ax) in enumerate(selected)]

    if "nrows" in canvas_kwargs or "ncols" in canvas_kwargs:
        raise TypeError("nrows and ncols are inferred from the imported axes")
    canvas_kwargs.setdefault("figsize", tuple(figure.get_size_inches()))
    canvas_kwargs.setdefault("dpi", figure.dpi)
    canvas = canvas_cls(nrows=nrows, ncols=ncols, **canvas_kwargs)
    for row, col, ax in positioned:
        target = canvas.add_subplot(row=row, col=col)
        _import_axes(ax, target, unsupported)
        for parent, twin, direction in twins:
            if parent is ax:
                factory = canvas.twinx if direction == "x" else canvas.twiny
                _import_axes(twin, factory(row=row, col=col), unsupported)
    if whole_figure:
        for getter, setter in (
            ("get_suptitle", "suptitle"),
            ("get_supxlabel", "supxlabel"),
            ("get_supylabel", "supylabel"),
        ):
            value = getattr(figure, getter, lambda: "")()
            if value:
                getattr(canvas, setter)(value)
        if figure.legends:
            unsupported("Figure-level legends are not imported")
        if figure.artists or figure.lines or figure.patches or figure.images:
            unsupported("Figure-level artists are not imported")
        titles = {figure._suptitle, figure._supxlabel, figure._supylabel}
        if any(text not in titles for text in figure.texts):
            unsupported("Figure-level text is not imported")
    return canvas


def _style(artist):
    return dict(
        label=artist.get_label(),
        alpha=copy.deepcopy(artist.get_alpha()),
        zorder=artist.get_zorder(),
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
        linestyle=line.get_linestyle(),
        linewidth=line.get_linewidth(),
        marker=None if line.get_marker() in ("None", "", " ") else line.get_marker(),
        markersize=line.get_markersize(),
        markerfacecolor=line.get_markerfacecolor(),
        markeredgecolor=line.get_markeredgecolor(),
        markeredgewidth=line.get_markeredgewidth(),
        drawstyle=line.get_drawstyle(),
        **_style(line),
    )


def _text_style(text):
    return dict(
        color=text.get_color(),
        fontsize=text.get_fontsize(),
        fontweight=text.get_fontweight(),
        fontstyle=text.get_fontstyle(),
        fontfamily=list(text.get_fontfamily()),
        rotation=text.get_rotation(),
        ha=text.get_ha(),
        va=text.get_va(),
    )


def _import_errorbar(container, ax, target, unsupported):
    """Return whether the container was consumed as one semantic entry."""
    line, caps, ranges = container.lines
    # Without a data line the original center of asymmetric errors is lost.
    # Leave these artists for the geometry import instead of inventing centers.
    if line is None:
        return False
    if not line.get_visible():
        return False
    if line.get_transform() != ax.transData or any(
        bars.get_transform() != ax.transData for bars in ranges
    ):
        unsupported("Error bars with non-data transforms are not imported")
        return True
    if any(cap.get_marker() not in ("_", "|") for cap in caps):
        unsupported("Error-bar limit arrows are not supported; importing geometry")
        return False
    x, y = (np.array(v, copy=True) for v in line.get_data(orig=False))
    errors = {}
    for axis, bars in zip(
        [i for i, flag in enumerate((container.has_xerr, container.has_yerr)) if flag],
        ranges,
    ):
        segments = bars.get_segments()
        if len(segments) != len(x):
            unsupported("Subsampled error bars are imported as geometry")
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
    if ranges:
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
        ("linestyle", collection.get_linestyles()),
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
    ):
        unsupported("Transformed or scalar-mapped collection is not imported")
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
                unsupported("Compound or curved polygon paths are not imported")
                continue
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
    consumed = set()
    # A bar container retains grouping and orientation, unlike raw rectangles.
    for container in ax.containers:
        if isinstance(container, ErrorbarContainer):
            if _import_errorbar(container, ax, target, unsupported):
                consumed.update(container.get_children())
            continue
        if not isinstance(container, BarContainer):
            unsupported(f"Unsupported {type(container).__name__}; container skipped")
            consumed.update(container.get_children())
            continue
        for i, patch in enumerate(container.patches):
            consumed.add(patch)
            if not patch.get_visible():
                continue
            kwargs = _style(patch)
            kwargs.update(
                color=patch.get_facecolor(),
                edgecolor=patch.get_edgecolor(),
                linewidth=patch.get_linewidth(),
                hatch=patch.get_hatch(),
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

    for line in ax.lines:
        if line in consumed or not line.get_visible():
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
            unsupported("Line with a non-data transform is not imported")
            continue
        target.plot(
            np.array(x, copy=True),
            np.array(y, copy=True),
            **_line_style(line),
        )
    for collection in ax.collections:
        if collection in consumed or not collection.get_visible():
            continue
        if _import_collection(collection, ax, target, unsupported):
            continue
        if not isinstance(collection, PathCollection):
            unsupported(f"Unsupported {type(collection).__name__}; collection skipped")
            continue
        paths = collection.get_paths()
        if (
            len(paths) != 1
            or collection.get_offset_transform() != ax.transData
            or not isinstance(collection.get_transform(), IdentityTransform)
        ):
            unsupported("Collection is not a supported scatter plot; skipped")
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
    for image in ax.images:
        if not image.get_visible():
            continue
        if image.get_transform() != ax.transData:
            unsupported("Image with a non-data transform is not imported")
            continue
        target.add_imshow(
            image.get_array().copy(),
            extent=tuple(image.get_extent()),
            origin=image.origin,
            cmap=copy.copy(image.get_cmap()),
            norm=copy.deepcopy(image.norm),
            interpolation=image.get_interpolation(),
            **_style(image),
        )
    for text in ax.texts:
        if not text.get_visible():
            continue
        if isinstance(text, Annotation):
            if text.xycoords != "data" or text.anncoords != "data":
                unsupported("Annotations outside data coordinates are not imported")
                continue
            arrowprops = text.arrowprops
            if arrowprops and any(key in arrowprops for key in ("patchA", "patchB")):
                unsupported(
                    "Annotation arrows referring to other patches are not imported"
                )
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
            unsupported("Non-data-coordinate text is not imported")
            continue
        target.text(
            *text.get_position(),
            text.get_text(),
            **_text_style(text),
            **_style(text),
        )
    for patch in ax.patches:
        if patch in consumed or not patch.get_visible():
            continue
        if patch.get_data_transform() != ax.transData:
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
    if any(p not in consumed and p.get_visible() for p in ax.patches) or ax.artists:
        unsupported("Unsupported patches or custom artists are not imported")
    if ax.child_axes:
        unsupported("Inset and secondary child axes are not imported")

    target.set_title(ax.get_title(), **_text_style(ax.title))
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
        if scale not in ("linear", "log"):
            unsupported(f"{name}-axis scale {scale!r} is not imported")
        else:
            getattr(target, f"set_{name}scale")(scale)
        axis = getattr(ax, f"{name}axis")
        if isinstance(axis.get_major_locator(), FixedLocator):
            getattr(target, f"set_{name}ticks")(
                axis.get_majorticklocs().copy(),
                labels=[label.get_text() for label in axis.get_ticklabels()],
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
