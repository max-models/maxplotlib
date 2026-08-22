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

    # block=True is passed explicitly (not left at plt.show()'s own default)
    # so a caller looping over several canvases shows them one at a time
    # regardless of matplotlib's interactive-mode state.
    assert calls == [((), {"block": True})]
    assert fig is not None
    assert axes is not None


def test_canvas_show_uses_ipython_display_in_jupyter(monkeypatch):
    import sys
    import types

    import matplotlib.pyplot as plt
    import pytest

    from maxplotlib import Canvas

    displayed = []
    closed = []
    fig = object()

    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: types.SimpleNamespace(config={"IPKernelApp": {}})
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = displayed.append

    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)
    monkeypatch.setattr(plt, "close", lambda value: closed.append(value))
    monkeypatch.setattr(Canvas, "_render", lambda *args, **kwargs: (fig, object()))
    monkeypatch.setattr(plt, "show", lambda: pytest.fail("pyplot.show was called"))

    canvas = Canvas()
    result = canvas.show()
    assert result[0] is fig
    assert result[1] is not None
    assert displayed == [fig]
    assert closed == [fig]


def test_canvas_show_falls_back_to_pyplot_outside_jupyter(monkeypatch):
    import sys
    import types

    import matplotlib.pyplot as plt
    import pytest

    from maxplotlib import Canvas

    calls = []
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: types.SimpleNamespace(config={})
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda fig: pytest.fail("IPython display was called")

    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: calls.append(kwargs))

    canvas = Canvas()
    canvas.add_subplot().plot([0, 1], [0, 1])
    canvas.show(block=False)

    assert calls == [{"block": False}]


def test_canvas_show_block_false_is_forwarded(monkeypatch):
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    calls = []

    monkeypatch.setattr(
        plt, "show", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    canvas = Canvas()
    subplot = canvas.add_subplot()
    subplot.plot([0, 1], [0, 1])

    canvas.show(block=False)

    assert calls == [((), {"block": False})]


def test_canvas_tick_label_rotation_is_forwarded_to_matplotlib():
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.plot([0, 1], [0, 1])
    canvas.set_xticks([0, 1], labels=["zero", "one"], rotation=45)

    fig, axes = canvas.plot()
    assert [label.get_rotation() for label in axes[0][0].get_xticklabels()] == [
        45.0,
        45.0,
    ]
    plt.close(fig)


def test_axis_label_and_tick_settings_are_forwarded_to_matplotlib():
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.plot([0, 1], [0, 1])
    canvas.set_xlabel("Time", fontsize=14, fontweight="bold", labelpad=12)
    canvas.set_ylabel("Value", color="crimson")
    canvas.set_title("Results", fontsize=16, color="navy")
    canvas.tick_params(axis="both", labelsize=12, colors="darkgreen", length=7)

    fig, axes = canvas.plot()
    matplotlib_axis = axes[0][0]
    assert matplotlib_axis.xaxis.label.get_fontsize() == 14
    assert matplotlib_axis.xaxis.label.get_fontweight() == "bold"
    assert matplotlib_axis.xaxis.labelpad == 12
    assert matplotlib_axis.yaxis.label.get_color() == "crimson"
    assert matplotlib_axis.title.get_fontsize() == 16
    assert matplotlib_axis.xaxis.majorTicks[0].tick1line.get_markersize() == 7
    assert matplotlib_axis.xaxis.get_ticklabels()[0].get_fontsize() == 12
    plt.close(fig)


def test_common_axis_and_figure_controls_are_forwarded_to_matplotlib():
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.plot([1, 2, 3], [1, 4, 9])
    canvas.set_facecolor("whitesmoke")
    canvas.set_axisbelow(True)
    canvas.margins(x=0.2, y=0.1)
    canvas.invert_yaxis()
    canvas.minorticks_on()
    canvas.set_axis_on()
    canvas.supxlabel("Shared time", fontsize=11)
    canvas.supylabel("Shared value", fontsize=11)
    canvas.subplots_adjust(left=0.2)

    fig, axes = canvas.plot()
    matplotlib_axis = axes[0][0]
    assert matplotlib_axis.get_facecolor() == (0.9607843137254902,) * 3 + (1.0,)
    assert matplotlib_axis.get_axisbelow() is True
    assert matplotlib_axis.yaxis_inverted()
    assert matplotlib_axis.xaxis.get_minorticklocs().size > 0
    assert fig._supxlabel.get_text() == "Shared time"
    assert fig._supylabel.get_text() == "Shared value"
    plt.close(fig)


def test_twinx_renders_a_secondary_matplotlib_y_axis():
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    canvas, primary = Canvas.subplots()
    secondary = canvas.twinx()
    primary.plot([0, 1, 2], [0, 1, 2], color="tab:blue", label="temperature")
    secondary.plot([0, 1, 2], [10, 20, 30], color="tab:red", label="pressure")
    primary.set_ylabel("Temperature", color="tab:blue")
    secondary.set_ylabel("Pressure", color="tab:red")

    fig, axes = canvas.plot()
    twin_axis = canvas.twinx_axes[(0, 0)]

    assert twin_axis is not axes[0][0]
    assert twin_axis.get_ylabel() == "Pressure"
    assert axes[0][0].get_ylabel() == "Temperature"
    assert len(twin_axis.lines) == 1
    plt.close(fig)


def test_common_matplotlib_plot_primitives_are_supported():
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.barh([0, 1], [2, 3], color="steelblue")
    axis.hist([0, 1, 1, 2, 2, 2], bins=3, alpha=0.5)
    axis.fill_betweenx([0, 1, 2], 0.5, [1, 1.5, 2], alpha=0.2)
    axis.axvspan(0.25, 0.75, alpha=0.1)
    axis.axhspan(0.5, 1.5, alpha=0.1)
    axis.arrow(0, 0, 1, 1, length_includes_head=True)
    axis.axline((0, 0), slope=1, linestyle="--")

    fig, axes = canvas.plot()
    matplotlib_axis = axes[0][0]
    assert len(matplotlib_axis.patches) > 0
    assert len(matplotlib_axis.lines) > 0
    plt.close(fig)


def test_additional_matplotlib_plot_primitives_are_supported():
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.step([0, 1, 2], [1, 3, 2], color="black")
    axis.stairs([1, 2, 1], edges=[0, 1, 2, 3], color="purple")
    axis.broken_barh([(0, 1), (1.5, 0.5)], (0, 0.4), color="orange")
    axis.bar([0, 1], [2, 3])
    axis.bar_label(fmt="%d")

    fig, axes = canvas.plot()
    matplotlib_axis = axes[0][0]
    assert len(matplotlib_axis.containers) >= 1
    assert len(matplotlib_axis.lines) >= 1
    assert len(matplotlib_axis.patches) >= 2
    assert len(matplotlib_axis.texts) >= 2
    plt.close(fig)


def test_statistical_and_event_plot_primitives_are_supported():
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.stem([0, 1, 2], [1, 3, 2])
    axis.stackplot([0, 1, 2], [1, 2, 1], [2, 1, 2])
    axis.boxplot([[1, 2, 3], [2, 4, 5]])
    axis.violinplot([[1, 2, 3], [2, 4, 5]])
    axis.eventplot([[0.2, 0.5], [1.0, 1.5]])

    fig, axes = canvas.plot()
    matplotlib_axis = axes[0][0]
    assert len(matplotlib_axis.lines) > 0
    assert len(matplotlib_axis.collections) > 0
    plt.close(fig)


def test_scientific_field_plot_primitives_are_supported():
    import matplotlib.pyplot as plt
    import numpy as np

    from maxplotlib import Canvas

    x = np.linspace(-1, 1, 8)
    y = np.linspace(-1, 1, 8)
    xx, yy = np.meshgrid(x, y)
    z = xx**2 + yy**2

    canvas, axis = Canvas.subplots()
    axis.contour(x, y, z)
    axis.contourf(x, y, z, alpha=0.4)
    axis.pcolormesh(x, y, z)
    axis.hexbin(np.ravel(xx), np.ravel(yy), gridsize=8)
    axis.matshow(z)

    fig, axes = canvas.plot()
    matplotlib_axis = axes[0][0]
    assert len(matplotlib_axis.collections) > 0
    assert len(matplotlib_axis.images) > 0
    plt.close(fig)


def test_vector_and_triangulated_plot_primitives_are_supported():
    import matplotlib.pyplot as plt
    import numpy as np

    from maxplotlib import Canvas

    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    triangles = [[0, 1, 2], [1, 3, 2]]
    z = x + y

    canvas, axis = Canvas.subplots()
    axis.quiver(x, y, np.ones(4), np.ones(4))
    axis.triplot(x, y, triangles=triangles)
    axis.tripcolor(x, y, z, triangles=triangles, alpha=0.3)
    axis.tricontour(x, y, z, triangles=triangles)
    axis.tricontourf(x, y, z, triangles=triangles, alpha=0.2)

    fig, axes = canvas.plot()
    matplotlib_axis = axes[0][0]
    assert len(matplotlib_axis.collections) > 0
    assert len(matplotlib_axis.lines) > 0
    plt.close(fig)


def test_stream_matrix_and_table_primitives_are_supported():
    import numpy as np

    from maxplotlib import Canvas

    x = np.linspace(-1, 1, 8)
    y = np.linspace(-1, 1, 8)
    xx, yy = np.meshgrid(x, y)
    z = xx**2 + yy**2
    matrix = np.eye(5)

    canvas, axis = Canvas.subplots()
    axis.streamplot(x, y, -yy, xx)
    axis.pcolor(x, y, z, alpha=0.2)
    axis.pcolorfast(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9), z, alpha=0.2)
    axis.spy(matrix)
    axis.table(cellText=[["A", "B"], ["1", "2"]], loc="upper right")

    fig, axes = canvas.plot()
    matplotlib_axis = axes[0][0]
    assert len(matplotlib_axis.collections) > 0
    assert len(matplotlib_axis.tables) == 1


def test_contour_labels_and_rasterization_zorder_are_supported():
    import matplotlib.pyplot as plt
    import numpy as np

    from maxplotlib import Canvas

    x = np.linspace(-1, 1, 5)
    xx, yy = np.meshgrid(x, x)
    canvas, axis = Canvas.subplots()
    axis.contour(x, x, xx**2 + yy**2)
    axis.clabel(inline=True, fontsize=8)
    axis.set_rasterization_zorder(2)

    fig, axes = canvas.plot(backend="matplotlib")

    assert len(axes[0, 0].texts) > 0
    assert axes[0, 0].get_rasterization_zorder() == 2
    plt.close(fig)


def test_axis_layout_and_log_shortcuts_are_supported():
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.fill([1, 2, 3], [1, 4, 1], alpha=0.2)
    axis.loglog([1, 2, 4], [1, 4, 16])
    axis.axis([1, 4, 1, 16])
    axis.autoscale_view(tight=True)
    axis.relim()
    axis.set_box_aspect(1)
    axis.set_xticklabels(["one", "two", "four"], rotation=30)
    axis.set_yticklabels(["low", "high"], color="navy")

    fig, axes = canvas.plot(backend="matplotlib")
    matplotlib_axis = axes[0, 0]

    assert matplotlib_axis.get_xscale() == "log"
    assert matplotlib_axis.get_yscale() == "log"
    assert matplotlib_axis.get_box_aspect() == 1
    assert len(matplotlib_axis.patches) == 1
    plt.close(fig)


def test_secondary_axes_are_supported_by_matplotlib():
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.plot([0, 1], [0, 1])
    axis.secondary_xaxis(
        "top", functions=(lambda x: x * 2, lambda x: x / 2), label="double"
    )
    axis.secondary_yaxis(
        "right", functions=(lambda y: y + 1, lambda y: y - 1), label="offset"
    )

    fig, axes = canvas.plot(backend="matplotlib")

    assert len(axes[0, 0].child_axes) == 2
    plt.close(fig)


def test_twiny_is_supported_by_matplotlib():
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    secondary = canvas.twiny()
    axis.plot([0, 1], [0, 1])
    secondary.plot([10, 20], [0, 1], color="red")

    fig, axes = canvas.plot(backend="matplotlib")

    assert canvas.twiny_axes[(0, 0)] is not axes[0, 0]
    plt.close(fig)


def test_axis_state_setter_aliases_are_supported():
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.plot([0, 1], [0, 1])
    axis.set_frame_on(False)
    axis.set_visible(True)
    axis.set_alpha(0.8)
    axis.set_zorder(3)
    axis.set_rasterized(True)
    axis.set_autoscale_on(True)
    axis.set_autoscalex_on(False)
    axis.set_autoscaley_on(True)
    axis.set_autoscale_on(False)
    axis.set_xbound(-1, 2)
    axis.set_ybound(-2, 3)

    fig, axes = canvas.plot(backend="matplotlib")
    matplotlib_axis = axes[0, 0]

    assert matplotlib_axis.get_frame_on() is False
    assert matplotlib_axis.get_visible() is True
    assert matplotlib_axis.get_alpha() == 0.8
    assert matplotlib_axis.get_zorder() == 3
    assert matplotlib_axis.get_rasterized() is True
    assert matplotlib_axis.get_xlim() == (-1, 2)
    assert matplotlib_axis.get_ylim() == (-2, 3)
    plt.close(fig)


def test_figure_size_and_dpi_setters_are_supported():
    import pytest

    from maxplotlib import Canvas

    canvas, _ = Canvas.subplots(figsize=(4, 3), dpi=120)
    assert canvas.get_size_inches() == pytest.approx([4, 3])
    assert canvas.get_figwidth() == 4
    assert canvas.get_figheight() == 3
    assert canvas.get_dpi() == 120

    canvas.set_figwidth(5)
    canvas.set_figheight(2)
    canvas.set_dpi(150)

    assert canvas.get_size_inches() == pytest.approx([5, 2])
    assert canvas.get_dpi() == 150


def test_generic_axis_setters_and_metadata_getters_are_supported():
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    canvas.suptitle("Figure title")
    canvas.supxlabel("Shared x")
    canvas.supylabel("Shared y")
    axis.plot([0, 1], [0, 1])
    axis.set(fc="lavender", adjustable="box", anchor="C")
    axis.update({"aspect": "equal"})
    axis.invert_xaxis()

    fig, axes = canvas.plot(backend="matplotlib")
    matplotlib_axis = axes[0, 0]

    assert canvas.get_suptitle() == "Figure title"
    assert canvas.get_supxlabel() == "Shared x"
    assert canvas.get_supylabel() == "Shared y"
    assert len(canvas.get_axes()) == 1
    assert canvas.xaxis_inverted() is True
    assert matplotlib_axis.get_adjustable() == "box"
    assert matplotlib_axis.get_anchor() == "C"
    plt.close(fig)


def test_figure_layout_helpers_and_aliases_are_supported():
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.plot([0, 1], [0, 1], label="line")
    axis.set_legend(loc="upper left")
    axis.add_table(cellText=[["A"]])
    axis.add_image([[1, 2], [3, 4]])
    canvas.set_tight_layout(True)
    canvas.align_labels()
    canvas.align_titles()
    canvas.align_xlabels()
    canvas.align_ylabels()
    canvas.autofmt_xdate(rotation=20)

    fig, axes = canvas.plot(backend="matplotlib")

    assert axes[0, 0].get_legend() is not None
    assert len(axes[0, 0].tables) == 1
    assert len(axes[0, 0].images) == 1
    plt.close(fig)


def test_axis_getters_reflect_configured_state():
    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.set_adjustable("datalim")
    axis.set_anchor("SW")
    axis.set_alpha(0.5)
    axis.set_box_aspect(1.2)
    axis.set_facecolor("pink")
    axis.set_frame_on(False)
    axis.set_legend(True)
    axis.set_rasterization_zorder(4)
    axis.set_rasterized(True)
    axis.set_visible(False)
    axis.set_zorder(7)
    axis.set_xbound(-1, 2)
    axis.set_ybound(-2, 3)
    axis.set_xmargin(0.1)
    axis.set_ymargin(0.2)

    assert canvas.get_adjustable() == "datalim"
    assert canvas.get_anchor() == "SW"
    assert canvas.get_alpha() == 0.5
    assert canvas.get_box_aspect() == 1.2
    assert canvas.get_facecolor() == "pink"
    assert canvas.get_frame_on() is False
    assert canvas.get_legend() is True
    assert canvas.get_rasterization_zorder() == 4
    assert canvas.get_rasterized() is True
    assert canvas.get_visible() is False
    assert canvas.get_zorder() == 7
    assert canvas.get_xbound() == (-1, 2)
    assert canvas.get_ybound() == (-2, 3)
    assert canvas.get_xmargin() == 0.1
    assert canvas.get_ymargin() == 0.2


def test_render_is_the_explicit_rendering_alias():
    from maxplotlib import Canvas

    canvas = Canvas()
    canvas.add_line([0, 1], [0, 1])

    rendered = canvas.render(backend="plotly")

    assert rendered is not None


def test_plot_adds_line_data_when_given_x_and_y():
    from maxplotlib import Canvas

    canvas = Canvas()
    result = canvas.plot([0, 1], [1, 2], color="purple", label="line")

    assert result is canvas
    assert canvas.render(backend="plotly").data[0].name == "line"


def test_legacy_plot_backend_form_warns():
    import pytest

    from maxplotlib import Canvas

    canvas = Canvas()
    canvas.add_line([0, 1], [0, 1])

    with pytest.warns(FutureWarning, match=r"canvas\.render"):
        canvas.plot(backend="plotly")


def test_matplotlib_postprocess_can_customize_figure_and_axes():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    from maxplotlib import Canvas

    canvas, (axis0, axis1) = Canvas.subplots(nrows=1, ncols=2)
    axis0.plot([0, 1], [0, 1])
    axis1.plot([0, 1], [1, 0])
    received = []

    def customize(fig, axes):
        received.append((fig, axes))
        fig.suptitle("Customized")
        for matplotlib_axis in axes.flat:
            matplotlib_axis.set_facecolor("lightgray")

    fig, axes = canvas.plot(matplotlib_postprocess=customize)

    assert received == [(fig, axes)]
    assert fig._suptitle.get_text() == "Customized"
    assert all(axis.get_facecolor() == to_rgba("lightgray") for axis in axes.flat)
    plt.close(fig)


def test_matplotlib_customizations_apply_declarative_figure_and_axes_methods():
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    canvas, (axis0, axis1) = Canvas.subplots(ncols=2)
    axis0.plot([0, 1], [0, 1])
    axis1.plot([0, 1], [1, 0])

    fig, axes = canvas.plot(
        matplotlib_customizations={
            "figure": {"suptitle": "Customized"},
            "axes": {
                "set_facecolor": "lightgray",
                "tick_params": {
                    "kwargs": {"axis": "both", "length": 6},
                },
            },
        }
    )

    from matplotlib.colors import to_rgba

    assert fig._suptitle.get_text() == "Customized"
    assert all(axis.get_facecolor() == to_rgba("lightgray") for axis in axes.flat)
    assert all(
        axis.xaxis.majorTicks[0].tick1line.get_markersize() == 6 for axis in axes.flat
    )
    plt.close(fig)


def test_matplotlib_customizations_accept_a_callable():
    import matplotlib.pyplot as plt

    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.plot([0, 1], [0, 1])
    received = []

    def customize(fig, axes):
        received.append((fig, axes))
        fig.suptitle("Customized")

    fig, axes = canvas.plot(matplotlib_customizations=customize)

    assert received == [(fig, axes)]
    assert fig._suptitle.get_text() == "Customized"
    plt.close(fig)


def test_matplotlib_postprocess_rejects_non_matplotlib_backends():
    import pytest

    from maxplotlib import Canvas

    with pytest.raises(ValueError, match="only supported with the matplotlib backend"):
        Canvas().plot(backend="plotly", matplotlib_postprocess=lambda fig, axes: None)


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


def test_tikzfigure_show_does_not_return_repr_in_jupyter(monkeypatch):
    import sys
    import types

    from maxplotlib import Canvas

    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: types.SimpleNamespace(config={"IPKernelApp": {}})
    monkeypatch.setitem(sys.modules, "IPython", ipython)

    class FakeTikzFigure:
        def show(self, **kwargs):
            self.show_kwargs = kwargs

    figure = FakeTikzFigure()
    monkeypatch.setattr(Canvas, "plot_tikzfigure", lambda *args, **kwargs: figure)

    assert Canvas().show(backend="tikzfigure") is None
    assert figure.show_kwargs == {"transparent": False}


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
