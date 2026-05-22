import numpy as np


def test_plotly_backend_supports_common_primitives():
    from maxplotlib import Canvas

    x = np.linspace(0, 1, 10)

    canvas, ax = Canvas.subplots()
    ax.plot(x, x, color="royalblue", label="line")
    ax.scatter(x, x**2, color="tomato", label="points")
    ax.errorbar(x[::2], (x**2)[::2], yerr=0.1, color="black", label="err")
    ax.fill_between(x, x - 0.1, x + 0.1, color="gray", alpha=0.2, label="band")
    ax.axhline(0.5, color="black", linestyle="dotted")
    ax.axvline(0.25, color="black", linestyle="dashed")
    ax.text(0.8, 0.8, "hi", color="purple")
    ax.annotate("there", xy=(0.3, 0.3), xytext=(0.6, 0.5), color="purple")
    ax.set_grid(True)
    ax.set_legend(True)

    fig = canvas.plot(backend="plotly")

    assert fig is not None
    assert len(fig.data) >= 4  # line, scatter, errorbar, fill_between
    assert len(getattr(fig.layout, "shapes", []) or []) >= 2  # axhline + axvline
    assert len(getattr(fig.layout, "annotations", []) or []) >= 2  # subplot title + text/annotate


def test_plotly_backend_respects_layers():
    from maxplotlib import Canvas

    x = np.linspace(0, 1, 10)
    canvas, ax = Canvas.subplots()
    ax.plot(x, x, color="black", label="L0", layer=0)
    ax.plot(x, x**2, color="red", label="L1", layer=1)

    fig0 = canvas.plot(backend="plotly", layers=[0])
    fig1 = canvas.plot(backend="plotly", layers=[1])

    assert len(fig0.data) == 1
    assert len(fig1.data) == 1

