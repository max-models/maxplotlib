# Maxplotlib

A clean, expressive wrapper around **Matplotlib**, **Plotly**,
**plotext**, and **tikzfigure** for producing publication-quality
figures with minimal boilerplate. Swap backends without rewriting your
data — render the same canvas as a crisp PNG, an interactive Plotly
chart, a terminal-native plotext figure, or camera-ready **TikZ** code
for LaTeX.

## Install

``` bash
pip install maxplotlibx
```

## Showcase

### Quickstart

``` python
import numpy as np
from maxplotlib import Canvas

x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)

canvas, ax = Canvas.subplots()
ax.plot(x, y)
```

Plot the figure with the default (matplotlib) backend:

``` python
canvas.show()
```

![](README_files/figure-markdown_strict/cell-3-output-1.png)

Render the same line graph directly in the terminal with the `plotext`
backend:

``` python
terminal_fig = canvas.plot(backend="plotext")
print(terminal_fig.build(keep_colors=False))
```

Or plot with the TikZ backend:

``` python
canvas.show(backend="tikzfigure")
```

![](README_files/figure-markdown_strict/cell-4-output-1.png)

### Terminal backend

The `plotext` backend is designed for terminal-first workflows. It
currently supports line plots, scatter plots, bars, filled regions,
error bars, reference lines, text/annotations, labels/titles, log
axes, layers, matrix-style `imshow()` rendering, common patches, and
multi-subplot canvases.

``` python
x = np.linspace(1, 10, 40)

canvas, ax = Canvas.subplots()
ax.plot(x, np.sqrt(x), color="cyan", label="sqrt(x)")
ax.errorbar(x[::8], np.sqrt(x[::8]), yerr=0.15, color="yellow", label="samples")
ax.set_title("Terminal plot")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xscale("log")
ax.set_legend(True)

canvas.show(backend="plotext")
```

## Examples

Runnable example scripts live in `examples/`:

``` bash
python examples/plotly_backend_basic.py
python examples/plotly_backend_parity.py
```

### Layers

``` python
x = np.linspace(0, 2 * np.pi, 200)

canvas, ax = Canvas.subplots(width="10cm", ratio=0.55)

ax.plot(x, np.sin(x), color="steelblue", label=r"$\sin(x)$", layer=0)
ax.plot(x, np.cos(x), color="tomato", label=r"$\cos(x)$", layer=1)
ax.plot(
    x,
    np.sin(x) * np.cos(x),
    color="seagreen",
    label=r"$\sin(x)\cos(x)$",
    linestyle="dashed",
    layer=2,
)

ax.set_xlabel("x")
ax.set_legend(True)
```

Show layer 0 only, then layers 0 and 1, then everything:

``` python
canvas.show(layers=[0])
```

![](README_files/figure-markdown_strict/cell-6-output-1.png)

Show all layers:

``` python
canvas.show()
```

![](README_files/figure-markdown_strict/cell-7-output-1.png)
