from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from plotext import figure as _plotext_figure

_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


class _Plotext6Axes:
    """Small compatibility surface for maxplotlib's plotext renderer."""

    def __init__(self, figure):
        self._figure = figure

    def _draw(self, signal, label=None):
        if label is not None:
            signal.label(label)
        self._figure.draw(signal)

    def plot(self, x, y, **kwargs):
        signal = self._figure.signal(x, y, marker=kwargs.get("marker"))
        self._draw(signal, kwargs.get("label"))

    scatter = plot

    def bar(self, *args, **kwargs):
        label = kwargs.pop("label", None)
        kwargs.pop("color", None)
        kwargs.pop("fill", None)
        signal = self._figure.bar(*args, **kwargs)
        self._draw(signal, label)

    def error(self, x, y, *, xerr=None, yerr=None, color=None, label=None):
        signal = self._figure.error(x, y, yerr, xerr, pixel=color)
        self._draw(signal, label)

    def matrix_plot(self, data, **kwargs):
        signal = self._figure.heatmap(data, symbol=kwargs.get("marker"))
        self._draw(signal)

    def text(self, label, x, y, **kwargs):
        # Plotext 6 keeps text colour in its marker object; the renderer only
        # needs the portable text/alignment arguments here.
        kwargs.pop("color", None)
        signal = self._figure.text(x, y, label, **kwargs)
        self._figure.draw(signal)

    def title(self, label):
        self._figure.title(label)

    def xlabel(self, label):
        self._figure.label(label, axis=0)

    def ylabel(self, label):
        self._figure.label(label, axis=1)

    def grid(self, active=True, *_args):
        self._figure.ruler("x").grid(active)
        self._figure.ruler("y").grid(active)

    def xlim(self, lower=None, upper=None):
        self._figure.ruler("x").lim(lower, upper)

    def ylim(self, lower=None, upper=None):
        self._figure.ruler("y").lim(lower, upper)

    def xscale(self, scale):
        self._figure.ruler("x").scale(scale)

    def yscale(self, scale):
        self._figure.ruler("y").scale(scale)

    def xticks(self, positions=None, labels=None):
        self._figure.ruler("x").ticks(positions, labels)

    def yticks(self, positions=None, labels=None):
        self._figure.ruler("y").ticks(positions, labels)

    def plotsize(self, width=None, height=None):
        self._figure.plot_size(width, height)

    def horizontal_line(self, position, **kwargs):
        kwargs.pop("color", None)
        self._figure.line(position, orientation=0, **kwargs)

    def vertical_line(self, position, **kwargs):
        kwargs.pop("color", None)
        self._figure.line(position, orientation=1, **kwargs)

    def subplots(self, rows=None, cols=None):
        self._figure.subplots(rows, cols)
        return self

    def subplot(self, row=None, col=None):
        return _Plotext6Axes(self._figure.subplot(row, col))

    def __getattr__(self, name):
        return getattr(self._figure, name)


def create_plotext_figure(nrows: int = 1, ncols: int = 1):
    # Plotext 6 exposes the figure through its public API rather than the
    # removed private ``plotext._figure._figure_class``.  It is a singleton,
    # so reset it before handing it to maxplotlib as a fresh canvas.
    figure = _Plotext6Axes(_plotext_figure.clear())
    if nrows > 1 or ncols > 1:
        figure.subplots(nrows, ncols)
    return figure


class PlotextFigure:
    def __init__(self, figure: Any, suptitle: str | None = None):
        self.figure = figure
        self.suptitle = suptitle

    def build(self, keep_colors: bool = True) -> str:
        output = str(self.figure.build())
        if self.suptitle:
            output = f"{self.suptitle}\n{output}"
        return output if keep_colors else strip_ansi(output)

    def show(self) -> str:
        output = self.build()
        print(output)
        return output

    def savefig(self, path, append: bool = False, keep_colors: bool = False) -> None:
        destination = Path(path)
        mode = "a" if append else "w"
        with destination.open(mode, encoding="utf-8") as handle:
            handle.write(self.build(keep_colors=keep_colors))
            handle.write("\n")

    save_fig = savefig

    def __getattr__(self, name):
        return getattr(self.figure, name)

    def __str__(self) -> str:
        return self.build(keep_colors=False)
