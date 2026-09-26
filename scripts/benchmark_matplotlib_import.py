"""Measure import time and peak Python allocation for reproducible fixtures.

Run from an installed checkout: python scripts/benchmark_matplotlib_import.py
Timings are observations, not brittle pass/fail thresholds. Native renderer
allocations outside Python are not included in tracemalloc's peak measurement.
"""

import argparse
import json
import time
import tracemalloc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from maxplotlib import Canvas


def benchmark(points, panels):
    fig, axes = plt.subplots(panels, 1, figsize=(8, max(3, panels * 2)))
    x = np.linspace(0, 10, points)
    for index, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(x, np.sin(x + index))
        ax.scatter([], [])
    tracemalloc.start()
    started = time.perf_counter()
    canvas = Canvas.from_matplotlib(fig, strict=True)
    imported = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result, _ = canvas.render()
    result.canvas.draw()
    finished = time.perf_counter()
    stats = dict(
        matplotlib=matplotlib.__version__,
        points_per_panel=points,
        panels=panels,
        import_seconds=imported - started,
        render_seconds=finished - imported,
        import_peak_python_mib=peak / 1024**2,
        entries=sum(len(plot.line_data) for _, _, plot in canvas.iter_subplots()),
    )
    plt.close(fig)
    plt.close(result)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=100_000)
    parser.add_argument("--panels", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.points < 1 or arguments.panels < 1:
        parser.error("points and panels must be positive")
    print(json.dumps(benchmark(arguments.points, arguments.panels), indent=2))
