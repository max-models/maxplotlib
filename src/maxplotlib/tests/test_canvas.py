def test():
    pass


def test_canvas_plot_tikzfigure_horizontal_subplots():
    """Test that Canvas.plot_tikzfigure() works with horizontal (1×n) layouts."""
    import numpy as np

    from maxplotlib import Canvas

    # Create a 1×2 canvas
    canvas, (ax1, ax2) = Canvas.subplots(ncols=2, width="10cm", ratio=0.3)

    # Add data to both subplots
    x = np.linspace(0, 2 * np.pi, 50)
    ax1.plot(x, np.sin(x), label="sin(x)", color="royalblue")
    ax1.set_title("Sine")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2.plot(x, np.cos(x), label="cos(x)", color="tomato")
    ax2.set_title("Cosine")
    ax2.set_xlabel("x")

    canvas.suptitle("Trig Functions")

    # This should NOT raise NotImplementedError
    result = canvas.plot_tikzfigure(verbose=False)

    # Result should be a TikzFigure or string containing LaTeX
    assert result is not None


def test_canvas_plot_tikzfigure_three_subplots():
    """Test 1×3 layout with tikzfigure backend."""
    import numpy as np

    from maxplotlib import Canvas

    x = np.linspace(0, 2 * np.pi, 50)
    canvas, axes = Canvas.subplots(ncols=3, width="12cm", ratio=0.3)

    axes[0].plot(x, np.sin(x), color="blue")
    axes[0].set_title("Sin")

    axes[1].plot(x, np.cos(x), color="red")
    axes[1].set_title("Cos")

    axes[2].plot(x, np.tan(x), color="green")
    axes[2].set_title("Tan")

    result = canvas.plot_tikzfigure()

    assert result is not None
    if isinstance(result, str):
        assert "\\subfigure" in result or "subfigure" in result


def test_canvas_plot_tikzfigure_respects_width_and_ratio():
    import numpy as np

    from maxplotlib import Canvas

    x = np.linspace(0, 1, 5)
    canvas, ax = Canvas.subplots(width="10cm", ratio=2)
    ax.plot(x, x**2, color="black")
    ax.set_title("Parabola")

    tikz = canvas.plot_tikzfigure().generate_tikz()

    assert "width=10cm" in tikz
    assert "height=20cm" in tikz
    assert "title=Parabola" in tikz


def test_canvas_plot_tikzfigure_vertical_not_supported():
    """Test that vertical layouts raise NotImplementedError."""
    import numpy as np
    import pytest

    from maxplotlib import Canvas

    x = np.linspace(0, 2 * np.pi, 50)
    # Create 2×1 layout (nrows=2)
    canvas, axes = Canvas.subplots(nrows=2, width="10cm")

    axes[0].plot(x, np.sin(x))
    axes[1].plot(x, np.cos(x))

    # Should raise NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        canvas.plot_tikzfigure()

    assert "nrows > 1" in str(exc_info.value)


def test_canvas_matplotlib_gridspec_kw_affects_row_spacing():
    """Test that hspace changes the vertical spacing between rows."""
    import matplotlib.pyplot as plt
    import numpy as np

    from maxplotlib import Canvas

    x = np.linspace(0, 1, 5)

    tight_canvas, tight_axes = Canvas.subplots(
        nrows=2,
        ncols=1,
        width="10cm",
        ratio=0.7,
        gridspec_kw={"hspace": 0.02, "wspace": 0.08},
    )
    for ax in tight_axes:
        ax.plot(x, x)
    tight_fig, tight_matplotlib_axes = tight_canvas.plot()
    tight_gap = (
        tight_matplotlib_axes[0, 0].get_position().y0
        - tight_matplotlib_axes[1, 0].get_position().y1
    )

    loose_canvas, loose_axes = Canvas.subplots(
        nrows=2,
        ncols=1,
        width="10cm",
        ratio=0.7,
        gridspec_kw={"hspace": 0.5, "wspace": 0.08},
    )
    for ax in loose_axes:
        ax.plot(x, x)
    loose_fig, loose_matplotlib_axes = loose_canvas.plot()
    loose_gap = (
        loose_matplotlib_axes[0, 0].get_position().y0
        - loose_matplotlib_axes[1, 0].get_position().y1
    )

    assert loose_gap > tight_gap
    plt.close(tight_fig)
    plt.close(loose_fig)


def test_canvas_matplotlib_gridspec_kw_affects_2x2_line_spacing():
    """Test that wspace/hspace change spacing for 2×2 line subplot grids."""
    import matplotlib.pyplot as plt
    import numpy as np

    from maxplotlib import Canvas

    x = np.linspace(0, 1, 20)

    tight_canvas, tight_axes = Canvas.subplots(
        nrows=2,
        ncols=2,
        width="12cm",
        ratio=0.7,
        hspace=0.03,
        wspace=0.03,
    )
    idx = 0
    for row_axes in tight_axes:
        for ax in row_axes:
            ax.plot(x, (idx + 1) * x)
            idx += 1
    tight_fig, tight_matplotlib_axes = tight_canvas.plot(backend="matplotlib")
    tight_hgap = (
        tight_matplotlib_axes[0, 1].get_position().x0
        - tight_matplotlib_axes[0, 0].get_position().x1
    )
    tight_vgap = (
        tight_matplotlib_axes[0, 0].get_position().y0
        - tight_matplotlib_axes[1, 0].get_position().y1
    )

    loose_canvas, loose_axes = Canvas.subplots(
        nrows=2,
        ncols=2,
        width="12cm",
        ratio=0.7,
        hspace=0.45,
        wspace=0.45,
    )
    idx = 0
    for row_axes in loose_axes:
        for ax in row_axes:
            ax.plot(x, (idx + 1) * x)
            idx += 1
    loose_fig, loose_matplotlib_axes = loose_canvas.plot(backend="matplotlib")
    loose_hgap = (
        loose_matplotlib_axes[0, 1].get_position().x0
        - loose_matplotlib_axes[0, 0].get_position().x1
    )
    loose_vgap = (
        loose_matplotlib_axes[0, 0].get_position().y0
        - loose_matplotlib_axes[1, 0].get_position().y1
    )

    assert loose_hgap > tight_hgap
    assert loose_vgap > tight_vgap
    plt.close(tight_fig)
    plt.close(loose_fig)


def test_canvas_matplotlib_gridspec_kw_affects_2x2_imshow_spacing():
    """Test spacing control also works for 2×2 color (imshow) subplot grids."""
    import matplotlib.pyplot as plt
    import numpy as np

    from maxplotlib import Canvas

    data = np.arange(100).reshape(10, 10)

    tight_canvas, tight_axes = Canvas.subplots(
        nrows=2,
        ncols=2,
        width="12cm",
        ratio=0.8,
        hspace=0.03,
        wspace=0.03,
    )
    idx = 0
    for row_axes in tight_axes:
        for ax in row_axes:
            ax.add_imshow(data + idx, cmap="viridis")
            ax.set_title(f"Heatmap {idx + 1}")
            idx += 1
    tight_fig, tight_matplotlib_axes = tight_canvas.plot(backend="matplotlib")
    tight_hgap = (
        tight_matplotlib_axes[0, 1].get_position().x0
        - tight_matplotlib_axes[0, 0].get_position().x1
    )
    tight_vgap = (
        tight_matplotlib_axes[0, 0].get_position().y0
        - tight_matplotlib_axes[1, 0].get_position().y1
    )

    loose_canvas, loose_axes = Canvas.subplots(
        nrows=2,
        ncols=2,
        width="12cm",
        ratio=0.8,
        hspace=0.45,
        wspace=0.45,
    )
    idx = 0
    for row_axes in loose_axes:
        for ax in row_axes:
            ax.add_imshow(data + idx, cmap="viridis")
            ax.set_title(f"Heatmap {idx + 1}")
            idx += 1
    loose_fig, loose_matplotlib_axes = loose_canvas.plot(backend="matplotlib")
    loose_hgap = (
        loose_matplotlib_axes[0, 1].get_position().x0
        - loose_matplotlib_axes[0, 0].get_position().x1
    )
    loose_vgap = (
        loose_matplotlib_axes[0, 0].get_position().y0
        - loose_matplotlib_axes[1, 0].get_position().y1
    )

    assert loose_hgap > tight_hgap
    assert loose_vgap > tight_vgap
    plt.close(tight_fig)
    plt.close(loose_fig)


def test_canvas_spacing_and_gridspec_kw_are_mutually_exclusive():
    import pytest

    from maxplotlib import Canvas, SubplotSpacing

    with pytest.raises(ValueError):
        Canvas(
            subplot_spacing=SubplotSpacing(wspace=0.2, hspace=0.2),
            gridspec_kw={"wspace": 0.3, "hspace": 0.3},
        )


def test_canvas_subplots_spacing_args_and_explicit_spacing_are_mutually_exclusive():
    import pytest

    from maxplotlib import Canvas, SubplotSpacing

    with pytest.raises(ValueError):
        Canvas.subplots(
            wspace=0.2,
            hspace=0.2,
            subplot_spacing=SubplotSpacing(wspace=0.3, hspace=0.3),
        )


def test_canvas_usetex_reads_environment_default(monkeypatch):
    from maxplotlib import Canvas

    monkeypatch.setenv("MAXPLOTLIB_USETEX", "true")
    canvas = Canvas()
    assert canvas.usetex is True


def test_canvas_usetex_constructor_overrides_environment(monkeypatch):
    from maxplotlib import Canvas

    monkeypatch.setenv("MAXPLOTLIB_USETEX", "true")
    canvas = Canvas(usetex=False)
    assert canvas.usetex is False


def test_canvas_plot_usetex_precedence(monkeypatch):
    import matplotlib.pyplot as plt

    import maxplotlib.canvas.canvas as canvas_module
    from maxplotlib import Canvas

    captured: list[bool] = []

    def fake_setup_tex_fonts(fontsize=10, usetex=False):
        captured.append(usetex)
        return {}

    monkeypatch.setattr(canvas_module, "setup_tex_fonts", fake_setup_tex_fonts)

    canvas = Canvas(usetex=True)
    subplot = canvas.add_subplot()
    subplot.plot([0, 1], [0, 1])

    fig, _ = canvas.plot_matplotlib()
    plt.close(fig)

    fig, _ = canvas.plot_matplotlib(usetex=False)
    plt.close(fig)

    assert captured == [True, False]


def test_canvas_show_uses_matplotlib_show(monkeypatch):
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    calls = []

    monkeypatch.setattr(
        plt, "show", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    canvas = Canvas()
    subplot = canvas.add_subplot()
    subplot.plot([0, 1], [0, 1])

    fig, axes = canvas.show()

    assert calls == [((), {})]
    assert fig is not None
    assert axes is not None


def test_show_canvas_script_invokes_canvas_show(monkeypatch):
    import maxplotlib

    calls = []

    def fake_show(self, *args, **kwargs):
        calls.append((args, kwargs))
        return object(), object()

    monkeypatch.setattr(maxplotlib.Canvas, "show", fake_show)

    canvas = maxplotlib.Canvas(width="10cm", ratio=0.4)
    canvas.add_line(x=[1, 2, 3], y=[4, 5, 6])
    canvas.show(backend="tikzfigure", verbose=True)

    assert calls == [((), {"backend": "tikzfigure", "verbose": True})]


def test_canvas_plot_uses_screen_dpi_when_not_saving():
    import matplotlib.pyplot as plt
    import pytest

    from maxplotlib import Canvas

    default_fig = plt.figure()
    default_dpi = default_fig.dpi
    plt.close(default_fig)

    canvas = Canvas(dpi=300)
    subplot = canvas.add_subplot()
    subplot.plot([0, 1], [0, 1])

    fig, _ = canvas.plot()

    assert fig.dpi == pytest.approx(default_dpi)


def test_canvas_without_explicit_size_uses_matplotlib_defaults():
    import matplotlib.pyplot as plt
    import pytest

    from maxplotlib import Canvas

    default_fig = plt.figure()
    default_size = default_fig.get_size_inches().copy()
    default_dpi = default_fig.dpi
    plt.close(default_fig)

    canvas = Canvas()
    subplot = canvas.add_subplot()
    subplot.plot([0, 1], [0, 1])

    fig, _ = canvas.plot()

    assert fig.get_size_inches()[0] == pytest.approx(default_size[0])
    assert fig.get_size_inches()[1] == pytest.approx(default_size[1])
    assert fig.dpi == pytest.approx(default_dpi)


def test_canvas_savefig_uses_configured_export_dpi(monkeypatch, tmp_path):
    from matplotlib.figure import Figure

    from maxplotlib import Canvas

    savefig_calls = []
    original_savefig = Figure.savefig

    def wrapped_savefig(self, *args, **kwargs):
        savefig_calls.append(kwargs.get("dpi"))
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", wrapped_savefig)

    canvas = Canvas(dpi=300)
    subplot = canvas.add_subplot()
    subplot.plot([0, 1], [0, 1])
    canvas.plot()
    canvas.savefig(tmp_path / "figure.png")

    assert savefig_calls[-1] == 300


def test_canvas_width_in_centimeters_is_preserved():
    import pytest

    from maxplotlib import Canvas

    canvas, _ = Canvas.subplots(width="7cm")
    fig, _ = canvas.plot()

    assert fig.get_size_inches()[0] == pytest.approx(7 / 2.54, abs=0.01)


def test_canvas_fontsize_controls_axes_text():
    import pytest

    from maxplotlib import Canvas

    canvas = Canvas(fontsize=10)
    canvas.add_subplot()
    canvas.set_title("Title")
    canvas.set_xlabel("X")
    canvas.set_ylabel("Y")
    canvas.suptitle("Figure title")

    fig, axes = canvas.plot()

    ax = axes[0, 0]
    assert ax.title.get_fontsize() == pytest.approx(10)
    assert ax.xaxis.label.get_fontsize() == pytest.approx(10)
    assert ax.yaxis.label.get_fontsize() == pytest.approx(10)
    assert fig._suptitle is not None
    assert fig._suptitle.get_fontsize() == pytest.approx(10)


if __name__ == "__main__":
    test()
