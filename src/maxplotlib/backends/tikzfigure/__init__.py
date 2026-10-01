"""The tikzfigure backend: pgfplots figures from what Matplotlib draws.

See :func:`figure_to_tikz`; ``Canvas.render(backend="tikzfigure")`` uses it.
"""

from .convert import TikzConversionWarning, figure_to_tikz
from .text import latex

__all__ = ["TikzConversionWarning", "figure_to_tikz", "latex"]
