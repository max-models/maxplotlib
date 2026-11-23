"""Matplotlib's shorthand for line plots: format strings and keyword aliases."""

import re

from matplotlib.colors import is_color_like

#: Matplotlib's one-character markers.
MARKERS = frozenset(".,ov^<>1234sp*hH+xDd|_PX")
#: Matplotlib's one-letter colours.
COLORS = frozenset("bgrcmykw")
#: Line styles of a format string, the two-character ones first.
LINESTYLES = ("--", "-.", "-", ":")

#: Short keyword names of a line and the ones the backends read.
LINE_ALIASES = {
    "c": "color",
    "ls": "linestyle",
    "lw": "linewidth",
    "ms": "markersize",
    "mec": "markeredgecolor",
    "mew": "markeredgewidth",
    "mfc": "markerfacecolor",
}

_CYCLE_COLOR = re.compile(r"C\d+")


def parse_plot_format(fmt: str) -> dict:
    """Return the ``color``, ``marker`` and ``linestyle`` of a format string.

    The format is Matplotlib's ``"[marker][line][color]"`` in any order, for
    example ``"o"``, ``"x--"``, ``"k:"`` or ``"C1s-"``; a string that is a colour
    on its own (``"red"``, ``"C2"``, ``"0.5"``) sets the colour only. As in
    Matplotlib, a marker without a line style draws no line
    (``linestyle="None"``). Only the parts the string gives are returned.

    Raises:
    ValueError: for a character that is none of the three, or a part given twice.
    """
    if not isinstance(fmt, str):
        raise TypeError(f"a format string must be a str, got {type(fmt).__name__}")
    # "1" to "4" are markers (tri_down, ...), not the grey levels they also spell
    if fmt not in MARKERS and (_CYCLE_COLOR.fullmatch(fmt) or is_color_like(fmt)):
        return {"color": fmt}

    parts = {}

    def take(key, value):
        if key in parts:
            raise ValueError(f"Illegal format string {fmt!r}: two {key} symbols")
        parts[key] = value

    i = 0
    while i < len(fmt):
        cycle = _CYCLE_COLOR.match(fmt, i)
        style = next((s for s in LINESTYLES if fmt.startswith(s, i)), None)
        if cycle:
            take("color", cycle.group())
            i = cycle.end()
        elif style:
            take("linestyle", style)
            i += len(style)
        elif fmt[i] in MARKERS:
            take("marker", fmt[i])
            i += 1
        elif fmt[i] in COLORS:
            take("color", fmt[i])
            i += 1
        else:
            raise ValueError(
                f"Unrecognized character {fmt[i]!r} in format string {fmt!r}"
            )
    if "marker" in parts and "linestyle" not in parts:
        parts["linestyle"] = "None"
    return parts


def canonical_line_kwargs(kwargs: dict) -> dict:
    """Return ``kwargs`` with Matplotlib's short names spelled out.

    ``ls``, ``lw``, ``ms``, ``c``, ``mec``, ``mew`` and ``mfc`` become
    ``linestyle``, ``linewidth``, ``markersize``, ``color``, ``markeredgecolor``,
    ``markeredgewidth`` and ``markerfacecolor``, which is what every backend
    reads. Giving a short name together with its long one is an error, as in
    Matplotlib.
    """
    out = {}
    for key, value in kwargs.items():
        name = LINE_ALIASES.get(key, key)
        if name in out or (name != key and name in kwargs):
            raise TypeError(f"Got both {name!r} and {key!r}, which are aliases")
        out[name] = value
    return out


def apply_plot_format(fmt: str | None, kwargs: dict) -> dict:
    """Return the line keywords of ``fmt`` and ``kwargs``; a keyword wins."""
    kwargs = canonical_line_kwargs(kwargs)
    if fmt is None:
        return kwargs
    return {**parse_plot_format(fmt), **kwargs}


def draws_no_line(linestyle) -> bool:
    """Whether a line style means markers only (``"None"``, ``"none"``, ``""``)."""
    return isinstance(linestyle, str) and linestyle.strip().lower() in ("", "none")
