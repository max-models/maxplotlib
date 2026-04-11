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


if __name__ == "__main__":
    test()
