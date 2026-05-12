import matplotlib.pyplot as plt
import numpy as np

from maxplotlib import Canvas


def test_python_example_nxm_line_subplots_spacing_changes():
    """Python example: 2x2 line subplots honor wspace/hspace settings."""
    x = np.linspace(0, 2 * np.pi, 200)

    tight_canvas, tight_axes = Canvas.subplots(
        nrows=2,
        ncols=2,
        width="12cm",
        ratio=0.7,
        wspace=0.05,
        hspace=0.05,
    )
    for i, row in enumerate(tight_axes):
        for j, ax in enumerate(row):
            ax.plot(x, np.sin((i + 1) * (j + 1) * x))
    tight_fig, tight_m_axes = tight_canvas.plot(backend="matplotlib")
    tight_hgap = tight_m_axes[0, 1].get_position().x0 - tight_m_axes[0, 0].get_position().x1
    tight_vgap = tight_m_axes[0, 0].get_position().y0 - tight_m_axes[1, 0].get_position().y1

    loose_canvas, loose_axes = Canvas.subplots(
        nrows=2,
        ncols=2,
        width="12cm",
        ratio=0.7,
        wspace=0.45,
        hspace=0.45,
    )
    for i, row in enumerate(loose_axes):
        for j, ax in enumerate(row):
            ax.plot(x, np.sin((i + 1) * (j + 1) * x))
    loose_fig, loose_m_axes = loose_canvas.plot(backend="matplotlib")
    loose_hgap = loose_m_axes[0, 1].get_position().x0 - loose_m_axes[0, 0].get_position().x1
    loose_vgap = loose_m_axes[0, 0].get_position().y0 - loose_m_axes[1, 0].get_position().y1

    assert loose_hgap > tight_hgap
    assert loose_vgap > tight_vgap
    plt.close(tight_fig)
    plt.close(loose_fig)


def test_python_example_nxm_color_subplots_spacing_changes():
    """Python example: 2x2 color subplots (imshow) honor wspace/hspace."""
    base = np.arange(100).reshape(10, 10)

    tight_canvas, tight_axes = Canvas.subplots(
        nrows=2,
        ncols=2,
        width="12cm",
        ratio=0.8,
        wspace=0.05,
        hspace=0.05,
    )
    idx = 0
    for row in tight_axes:
        for ax in row:
            ax.add_imshow(base + idx, cmap="viridis")
            idx += 1
    tight_fig, tight_m_axes = tight_canvas.plot(backend="matplotlib")
    tight_hgap = tight_m_axes[0, 1].get_position().x0 - tight_m_axes[0, 0].get_position().x1
    tight_vgap = tight_m_axes[0, 0].get_position().y0 - tight_m_axes[1, 0].get_position().y1

    loose_canvas, loose_axes = Canvas.subplots(
        nrows=2,
        ncols=2,
        width="12cm",
        ratio=0.8,
        wspace=0.45,
        hspace=0.45,
    )
    idx = 0
    for row in loose_axes:
        for ax in row:
            ax.add_imshow(base + idx, cmap="viridis")
            idx += 1
    loose_fig, loose_m_axes = loose_canvas.plot(backend="matplotlib")
    loose_hgap = loose_m_axes[0, 1].get_position().x0 - loose_m_axes[0, 0].get_position().x1
    loose_vgap = loose_m_axes[0, 0].get_position().y0 - loose_m_axes[1, 0].get_position().y1

    assert loose_hgap > tight_hgap
    assert loose_vgap > tight_vgap
    plt.close(tight_fig)
    plt.close(loose_fig)
