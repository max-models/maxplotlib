---
title: Installation
description: Install maxplotlib from PyPI, with the optional xarray support and LaTeX for the TikZ backend.
---

maxplotlib is on PyPI as `maxplotlibx` (the import name is `maxplotlib`):

```sh
pip install maxplotlibx
```

With the [xarray](https://docs.xarray.dev) accessor (`import maxplotlib.xarray`):

```sh
pip install "maxplotlibx[xarray]"
```

Then, in Python:

```python
from maxplotlib import Canvas
```

## Backends

Matplotlib, Plotly, plotext and [tikzfigure](https://github.com/max-models/tikzfigure) are
installed with the package. The TikZ backend writes LaTeX code; to compile it into a PDF or PNG
(`canvas.savefig("figure.pdf", backend="tikzfigure")`, or the figures in the TikZ tutorials) you
also need a LaTeX distribution with `pdflatex` and PGF/TikZ with pgfplots, for example TeX Live
(`texlive-latex-extra texlive-pictures` on Debian/Ubuntu) or MacTeX.

## From source

```sh
git clone https://github.com/max-models/maxplotlib
cd maxplotlib
pip install -e ".[dev]"
```
