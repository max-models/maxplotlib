"""Write the example web figures of the docs' "Web figures" guide to docs/public/figures/.

The docs build runs this before ``astro build`` (``pre-build`` in .github/workflows/docs.yml);
run it by hand for ``astro dev``:

    python scripts/web_figure_examples.py

The code of each figure is also shown on the page, in docs/src/content/docs/guides/web-figures.mdx;
keep the two the same.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from maxplotlib import Canvas
from maxplotlib.web import write_web_figure

OUT = Path(__file__).resolve().parents[1] / "docs" / "public" / "figures"


def waves():
    """Lines without colours of their own, and a black reference line."""
    x = np.linspace(0, 4 * np.pi, 400)
    canvas = Canvas(width="12cm", ratio=0.5)
    for k in range(4):
        canvas.plot(
            x, np.exp(-0.1 * x) * np.sin(x - k * np.pi / 4), label=f"phase {k}/4 π"
        )
    canvas.plot(x, 0 * x, color="black", linewidth=0.8)
    canvas.set_xlabel("t")
    canvas.set_ylabel("amplitude")
    canvas.set_legend(True)
    canvas.savefig(OUT / "waves.plotly.json", backend="plotly")


def heatmap():
    """A colormap, which stays as it is, and a contour in Matplotlib's C1."""
    x = np.linspace(-2, 2, 80)
    y = np.linspace(-1.5, 1.5, 60)
    X, Y = np.meshgrid(x, y)
    z = np.exp(-(X**2 + Y**2)) * np.cos(3 * X) + 0.3 * np.exp(
        -((X - 1) ** 2 + Y**2) * 4
    )
    fig = go.Figure(
        [
            go.Heatmap(x=x, y=y, z=z, colorscale="Viridis", colorbar={"title": "φ"}),
            go.Contour(
                x=x,
                y=y,
                z=z,
                contours={"coloring": "none", "start": 0.5, "end": 0.5},
                line={"color": "#ff7f0e", "width": 2},
                showscale=False,
                name="φ = 0.5",
            ),
        ],
        layout={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "width": 640,
            "height": 420,
            "margin": {"t": 30},
        },
    )
    write_web_figure(
        fig, OUT / "heatmap.plotly.json", alt="A potential φ(x, y) with its 0.5 contour"
    )


def animation():
    """An animation with a slider and Play button: a wave packet travelling to the right."""
    x = np.linspace(0, 10, 200)
    times = np.linspace(0, 6, 31)

    def packet(t):
        return np.exp(-((x - 2 - t) ** 2)) * np.cos(6 * (x - 2 - t))

    frames = [
        # a frame names only what changes: the y values
        go.Frame(name=f"{t:.1f}", data=[go.Scatter(y=packet(t))])
        for t in times
    ]
    steps = [
        {
            "label": frame.name,
            "method": "animate",
            "args": [[frame.name], {"mode": "immediate", "frame": {"duration": 0}}],
        }
        for frame in frames
    ]
    fig = go.Figure(
        data=[go.Scatter(x=x, y=packet(0), name="u(x, t)")],
        frames=frames,
        layout={
            "xaxis": {"title": {"text": "x", "standoff": 8}, "range": [0, 10]},
            "yaxis": {"title": "u", "range": [-1.1, 1.1]},
            "width": 700,
            "height": 380,
            "sliders": [
                {"steps": steps, "currentvalue": {"prefix": "t = "}, "pad": {"t": 50}}
            ],
            "updatemenus": [
                {
                    "type": "buttons",
                    "x": 0,
                    "y": 1.15,
                    "xanchor": "left",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {"frame": {"duration": 60}, "fromcurrent": True},
                            ],
                        }
                    ],
                }
            ],
        },
    )
    write_web_figure(
        fig, OUT / "packet.plotly.json", alt="A wave packet travelling to the right"
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for example in (waves, heatmap, animation):
        example()
        print(f"wrote the {example.__name__} example")
