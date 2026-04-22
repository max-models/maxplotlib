from __future__ import annotations

import re
from pathlib import Path

from plotext._figure import _figure_class

_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def create_plotext_figure(nrows: int = 1, ncols: int = 1) -> _figure_class:
    figure = _figure_class()
    if nrows > 1 or ncols > 1:
        figure.subplots(nrows, ncols)
    return figure


class PlotextFigure:
    def __init__(self, figure: _figure_class, suptitle: str | None = None):
        self.figure = figure
        self.suptitle = suptitle

    def build(self, keep_colors: bool = True) -> str:
        output = self.figure.build()
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
