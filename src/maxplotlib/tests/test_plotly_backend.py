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
    assert (
        len(getattr(fig.layout, "annotations", []) or []) >= 2
    )  # subplot title + text/annotate


def test_plotly_backend_supports_tick_label_rotation():
    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.plot([0, 1], [0, 1])
    axis.set_xticks([0, 1], labels=["zero", "one"], rotation=45)

    fig = canvas.plot(backend="plotly")

    assert fig.layout.xaxis.tickangle == 45


def test_plotly_backend_supports_twinx():
    from maxplotlib import Canvas

    canvas, primary = Canvas.subplots()
    secondary = canvas.twinx()
    primary.plot([0, 1], [0, 1], color="blue")
    secondary.plot([0, 1], [10, 20], color="red")
    secondary.set_ylabel("Secondary")

    fig = canvas.plot(backend="plotly")

    assert len(fig.data) == 2
    assert fig.layout.yaxis2.title.text == "Secondary"


def test_plotly_backend_supports_common_added_primitives():
    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.barh([0, 1], [2, 3])
    axis.hist([0, 1, 1, 2], bins=3)
    axis.fill_betweenx([0, 1], 0, 1, alpha=0.2)
    axis.axvspan(0.25, 0.75, alpha=0.1)
    axis.axhspan(0.25, 0.75, alpha=0.1)
    axis.arrow(0, 0, 1, 1)

    fig = canvas.plot(backend="plotly")

    assert len(fig.data) >= 3
    assert len(fig.layout.shapes) >= 2
    assert any(annotation.showarrow for annotation in fig.layout.annotations)


def test_plotly_backend_supports_step_stairs_broken_barh_and_pie():
    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.step([0, 1, 2], [1, 3, 2])
    axis.stairs([1, 2], edges=[0, 1, 2])
    axis.broken_barh([(0, 1), (2, 0.5)], (0, 0.5))
    axis.pie([2, 3, 4], labels=["A", "B", "C"])

    fig = canvas.plot(backend="plotly")

    assert any(trace.type == "pie" for trace in fig.data)
    assert len(fig.data) >= 5


def test_plotly_backend_supports_statistical_and_event_plots():
    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.stem([0, 1, 2], [1, 3, 2])
    axis.stackplot([0, 1, 2], [1, 2, 1], [2, 1, 2])
    axis.boxplot([[1, 2, 3], [2, 4, 5]])
    axis.violinplot([[1, 2, 3], [2, 4, 5]])
    axis.eventplot([[0.2, 0.5], [1.0, 1.5]])

    fig = canvas.plot(backend="plotly")

    assert any(trace.type == "box" for trace in fig.data)
    assert any(trace.type == "violin" for trace in fig.data)
    assert len(fig.layout.shapes) >= 4


def test_plotly_backend_supports_scientific_field_plots():
    import numpy as np

    from maxplotlib import Canvas

    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    xx, yy = np.meshgrid(x, y)
    z = xx**2 + yy**2

    canvas, axis = Canvas.subplots()
    axis.contour(x, y, z)
    axis.contourf(x, y, z)
    axis.pcolormesh(x, y, z)
    axis.hexbin(xx.ravel(), yy.ravel(), gridsize=5)
    axis.matshow(z)

    fig = canvas.plot(backend="plotly")

    assert any(trace.type == "contour" for trace in fig.data)
    assert any(trace.type == "heatmap" for trace in fig.data)
    assert any(trace.type == "histogram2d" for trace in fig.data)


def test_plotly_backend_supports_contour_labels():
    from maxplotlib import Canvas

    x = np.linspace(-1, 1, 5)
    xx, yy = np.meshgrid(x, x)
    canvas, axis = Canvas.subplots()
    axis.contour(x, x, xx**2 + yy**2)
    axis.clabel(fontsize=10, color="black")

    fig = canvas.plot(backend="plotly")

    assert fig.data[0].contours.showlabels is True
    assert fig.data[0].contours.labelfont.size == 10


def test_plotly_backend_supports_fill_log_scales_and_ticklabels():
    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.fill([1, 2, 4], [1, 4, 1], color="orange", alpha=0.3)
    axis.loglog([1, 2, 4], [1, 4, 16])
    axis.set_xticklabels(["one", "two", "four"], color="navy")

    fig = canvas.plot(backend="plotly")

    assert any(trace.fill == "toself" for trace in fig.data)
    assert fig.layout.xaxis.type == "log"
    assert fig.layout.yaxis.type == "log"
    assert fig.layout.xaxis.ticktext == ("one", "two", "four")


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


def test_plotly_backend_supports_common_patches_and_symlog():
    import matplotlib.patches as mpatches

    from maxplotlib import Canvas

    canvas, ax = Canvas.subplots()
    ax.add_patch(
        mpatches.Rectangle(
            (0.2, 0.2), 1.3, 0.7, fill=False, edgecolor="yellow", label="r"
        )
    )
    ax.add_patch(
        mpatches.Circle((2.2, 1.6), 0.45, fill=False, edgecolor="cyan", label="c")
    )
    ax.add_patch(
        mpatches.Polygon(
            [[3.0, 0.5], [3.8, 1.2], [3.4, 2.0]],
            fill=True,
            facecolor="green",
            label="p",
        )
    )
    ax.add_patch(
        mpatches.Ellipse((2.8, 1.0), 0.8, 0.5, fill=False, edgecolor="white", label="e")
    )
    ax.set_title("patches")
    ax.set_legend(True)

    fig = canvas.plot(backend="plotly")
    assert fig is not None
    assert len(getattr(fig.layout, "shapes", []) or []) >= 4
    # patch labels become dummy legend traces
    assert any(getattr(t, "name", "") == "p" for t in fig.data)

    canvas2, ax2 = Canvas.subplots()
    x = np.linspace(-20, 20, 41)
    ax2.plot(x, x**3, color="cyan", label="x^3")
    ax2.set_xscale("symlog")
    ax2.set_yscale("symlog")
    fig2 = canvas2.plot(backend="plotly")
    assert fig2 is not None


def test_plotly_backend_renders_mixed_vector_primitives():
    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.plot([0, 1], [0, 1])
    axis.streamplot(
        np.linspace(0, 1, 3),
        np.linspace(0, 1, 3),
        np.ones((3, 3)),
        np.ones((3, 3)),
    )

    fig = canvas.plot(backend="plotly")
    assert len(fig.data) > 1


def test_plotly_backend_supports_bar_labels():
    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.bar([0, 1], [2, 3])
    axis.bar_label(fmt="%d")

    fig = canvas.plot(backend="plotly")
    labels = [annotation.text for annotation in fig.layout.annotations]
    assert labels[-2:] == ["2", "3"]


def test_plotly_backend_supports_pseudocolor_spy_table_and_triplot():
    from maxplotlib import Canvas

    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    z = xx**2 + yy**2
    points_x = np.array([0.0, 1.0, 0.0, 1.0])
    points_y = np.array([0.0, 0.0, 1.0, 1.0])
    triangles = np.array([[0, 1, 2], [1, 3, 2]])

    canvas, axis = Canvas.subplots()
    axis.pcolor(x, y, z)
    axis.pcolorfast(x, y, z)
    axis.spy([[1, 0], [0, 2]])
    axis.table(cellText=[["A", "B"], ["1", "2"]])
    axis.triplot(points_x, points_y, triangles=triangles)

    fig = canvas.plot(backend="plotly")

    assert sum(trace.type == "heatmap" for trace in fig.data) >= 3
    assert any(trace.type == "table" for trace in fig.data)
    assert any(trace.type == "scatter" for trace in fig.data)


def test_plotly_backend_supports_quiver():
    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.quiver([0, 1], [0, 1], [1, -1], [1, 1], color="purple", alpha=0.5)

    fig = canvas.plot(backend="plotly")

    arrows = [annotation for annotation in fig.layout.annotations if annotation.showarrow]
    assert len(arrows) == 2
    assert arrows[0].arrowcolor == "purple"


def test_plotly_backend_supports_tripcolor():
    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    axis.tripcolor(
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 2, 3],
        triangles=[[0, 1, 2], [1, 3, 2]],
    )

    fig = canvas.plot(backend="plotly")

    assert sum(trace.fill == "toself" for trace in fig.data) == 2


def test_plotly_backend_supports_triangulated_contours():
    from maxplotlib import Canvas

    canvas, axis = Canvas.subplots()
    x = [0, 1, 0, 1]
    y = [0, 0, 1, 1]
    triangles = [[0, 1, 2], [1, 3, 2]]
    values = [0, 1, 2, 3]
    axis.tricontour(x, y, values, triangles=triangles, levels=3)
    axis.tricontourf(x, y, values, triangles=triangles, levels=3)

    fig = canvas.plot(backend="plotly")

    assert any(trace.fill is None for trace in fig.data)
    assert any(trace.fill == "toself" for trace in fig.data)


def test_plotly_backend_supports_streamplot():
    from maxplotlib import Canvas

    coordinates = np.linspace(0, 1, 5)
    canvas, axis = Canvas.subplots()
    axis.streamplot(
        coordinates,
        coordinates,
        np.ones((5, 5)),
        np.ones((5, 5)),
        color="darkgreen",
    )

    fig = canvas.plot(backend="plotly")

    assert len(fig.data) > 0
    assert all(trace.type == "scatter" for trace in fig.data)
