"""Web figures: Plotly figures as theme-neutral JSON for the max-models docs sites.

A web figure is a Plotly figure saved as JSON (``data``, ``layout``, ``frames``) without
plotly.js and without anything that fixes its look to one page: no template, no fixed width or
height, transparent backgrounds. The ``PlotlyFigure`` component of the max-models docs theme
(starlight-theme) embeds it on a page, loads plotly.js only when the figure scrolls into view,
and gives it the site's colours in light and dark mode:

- text, grid lines and backgrounds from the theme;
- the theme's data palette for every trace in a default colour. Traces without a colour take it
  from ``layout.colorway``, which the page sets to the palette; traces with an explicit colour
  from a default cycle (Matplotlib's ``C0``..``C9``, Plotly's default colorway, the Okabe-Ito
  palette) carry the index of that colour as ``trace.meta["color_slot"]``, and the page gives
  them the palette colour of that index. Black traces take the text colour. Other explicit
  colours and colormaps stay as they are.

Hints for the page are in ``layout.meta["web"]``:

``version``
    The format version, :data:`FORMAT_VERSION`.
``title``
    The figure's title, also the default caption.
``alt``
    A text alternative for readers who cannot see the figure.
``aspect``
    Width over height, so the page reserves the right space before the figure loads.
``white_plot``
    Keep a white plot area in dark mode too, for colours chosen for a white background
    (e.g. a figure converted from Matplotlib with white overlays).

Examples
--------
>>> import plotly.graph_objects as go
>>> from maxplotlib.web import write_web_figure
>>> fig = go.Figure(go.Scatter(x=[0, 1, 2], y=[1, 3, 2]))
>>> write_web_figure(fig, "public/figures/line.plotly.json", alt="A rising line")
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1

# The default colour cycles whose colours become palette slots, each in its own order.
MATPLOTLIB_CYCLE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
PLOTLY_COLORWAY = (
    "#636efa",
    "#ef553b",
    "#00cc96",
    "#ab63fa",
    "#ffa15a",
    "#19d3f3",
    "#ff6692",
    "#b6e880",
    "#ff97ff",
    "#fecb52",
)
OKABE_ITO = (
    "#0072b2",
    "#e69f00",
    "#009e73",
    "#d55e00",
    "#cc79a7",
    "#56b4e9",
    "#f0e442",
    "#000000",
)
DEFAULT_CYCLES = (MATPLOTLIB_CYCLE, PLOTLY_COLORWAY, OKABE_ITO)

_RGB = re.compile(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)")
_CARTESIAN_AXIS = re.compile(r"[xy]axis\d*")


def _hex(color: Any) -> str | None:
    """A colour as ``#rrggbb``, or ``None`` if it is not a single colour string."""
    if not isinstance(color, str):
        return None
    text = color.strip().lower()
    if re.fullmatch(r"#[0-9a-f]{6}", text):
        return text
    if re.fullmatch(r"#[0-9a-f]{3}", text):
        return "#" + "".join(c * 2 for c in text[1:])
    match = _RGB.match(text)
    if match:
        r, g, b = (round(float(v)) for v in match.groups())
        return f"#{r:02x}{g:02x}{b:02x}"
    return None


def color_slot(color: Any, cycles=DEFAULT_CYCLES) -> int | None:
    """Return the position of a colour in one of the default colour cycles.

    Parameters
    ----------
    color : str
        A colour, as ``#rrggbb``, ``#rgb``, ``rgb(...)`` or ``rgba(...)``.
    cycles : sequence of sequence of str, optional
        The cycles to look in, in order. Default: :data:`DEFAULT_CYCLES`.

    Returns
    -------
    int or None
        The position, or ``None`` if the colour is in none of the cycles.

    Examples
    --------
    >>> color_slot("#ff7f0e")  # Matplotlib's C1
    1
    >>> color_slot("black") is None
    True
    """
    code = _hex(color)
    if code is None:
        return None
    for cycle in cycles:
        if code in cycle:
            return cycle.index(code)
    return None


def _trace_slot(trace: dict) -> int | None:
    """The palette slot of a trace: of its line colour, else of its marker or fill colour."""
    for path in (("line", "color"), ("marker", "color"), ("fillcolor",)):
        value: Any = trace
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        slot = color_slot(value)
        if slot is not None:
            return slot
    return None


def _json_default(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if tolist is not None:
        return tolist()
    raise TypeError(f"cannot write {type(value).__name__} as JSON")


def _figure_dict(figure: Any) -> dict:
    """The figure as plain JSON data, with Plotly's own encoding of arrays."""
    if isinstance(figure, dict):
        return json.loads(json.dumps(figure, default=_json_default))
    import plotly.io as pio

    return json.loads(pio.to_json(figure, validate=False))


def _title_text(layout: dict) -> str | None:
    title = layout.get("title")
    if isinstance(title, dict):
        title = title.get("text")
    return title if isinstance(title, str) and title else None


def _with_slot(trace: dict) -> dict:
    slot = _trace_slot(trace)
    if slot is None:
        return trace
    meta = trace.get("meta")
    if meta is not None and not isinstance(meta, dict):
        meta = {"value": meta}
    return {**trace, "meta": {**(meta or {}), "color_slot": slot}}


def web_figure(
    figure: Any,
    *,
    title: str | None = None,
    alt: str | None = None,
    aspect: float | None = None,
    white_plot: bool | None = None,
) -> dict:
    """Make a Plotly figure theme-neutral, as a dict ready to write as JSON.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure or dict
        The figure, or its JSON as a dict (e.g. read from a file).
    title : str, optional
        The title and default caption. Default: the figure's own title.
    alt : str, optional
        A text alternative for the figure. Default: none.
    aspect : float, optional
        Width over height. Default: from the figure's width and height when both are set.
    white_plot : bool, optional
        Keep a white plot area in dark mode. Default: ``True`` for a figure converted from
        Matplotlib (``layout.meta["converted_from"] == "matplotlib"``), else ``False``.

    Returns
    -------
    dict
        ``data``, ``layout`` and ``frames`` (when the figure has frames) of the web figure.

    Examples
    --------
    >>> web = web_figure(fig, alt="Field energy over time")
    >>> web["layout"]["meta"]["web"]["version"]
    1
    """
    source = _figure_dict(figure)
    layout = dict(source.get("layout") or {})
    meta = layout.get("meta")
    meta = (
        dict(meta)
        if isinstance(meta, dict)
        else ({} if meta is None else {"value": meta})
    )

    width, height = layout.pop("width", None), layout.pop("height", None)
    if aspect is None and width and height:
        aspect = float(width) / float(height)
    if white_plot is None:
        white_plot = meta.get("converted_from") == "matplotlib"
    title = title if title is not None else _title_text(layout)

    layout.pop("template", None)
    layout["autosize"] = True
    # the page sets the size, so make room for tick labels and axis titles at any size
    for key, axis in layout.items():
        if _CARTESIAN_AXIS.fullmatch(key) and isinstance(axis, dict):
            axis.setdefault("automargin", True)
    layout["paper_bgcolor"] = "rgba(0,0,0,0)"
    layout["plot_bgcolor"] = "white" if white_plot else "rgba(0,0,0,0)"
    hints: dict[str, Any] = {"version": FORMAT_VERSION, "white_plot": bool(white_plot)}
    if title:
        hints["title"] = title
    if alt:
        hints["alt"] = alt
    if aspect:
        hints["aspect"] = round(float(aspect), 4)
    meta["web"] = hints
    layout["meta"] = meta

    web = {
        "data": [_with_slot(trace) for trace in source.get("data") or []],
        "layout": layout,
    }
    frames = source.get("frames")
    if frames:
        web["frames"] = [
            {**frame, "data": [_with_slot(trace) for trace in frame.get("data") or []]}
            for frame in frames
        ]
    return web


def write_web_figure(figure: Any, path: str | Path, **hints: Any) -> Path:
    """Write a Plotly figure as a web figure, ``<name>.plotly.json``.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure or dict
        The figure.
    path : str or pathlib.Path
        The file; its folder is created when missing.
    **hints
        ``title``, ``alt``, ``aspect`` and ``white_plot``, as for :func:`web_figure`.

    Returns
    -------
    pathlib.Path
        The file written.

    Examples
    --------
    >>> write_web_figure(fig, "docs/public/figures/energy.plotly.json", alt="Energy over time")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(web_figure(figure, **hints), separators=(",", ":")))
    return path
