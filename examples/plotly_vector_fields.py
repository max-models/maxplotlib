"""Plotly examples for quiver, streamplot, and triangular plotting."""

import numpy as np

from maxplotlib import Canvas


def main() -> None:
    coordinates = np.linspace(-2, 2, 12)
    xx, yy = np.meshgrid(coordinates, coordinates)

    canvas, (vector_axis, triangle_axis) = Canvas.subplots(ncols=2, width="14cm")
    vector_axis.quiver(
        xx,
        yy,
        -yy,
        xx,
        color="darkgreen",
        alpha=0.7,
        linewidth=1.2,
    )
    vector_axis.streamplot(
        coordinates,
        coordinates,
        -yy,
        xx,
        color="royalblue",
        density=1.0,
    )
    vector_axis.set_title("quiver + streamplot")

    points_x = np.array([0.0, 1.0, 0.0, 1.0])
    points_y = np.array([0.0, 0.0, 1.0, 1.0])
    triangles = [[0, 1, 2], [1, 3, 2]]
    triangle_axis.tripcolor(
        points_x,
        points_y,
        [0.0, 1.0, 2.0, 3.0],
        triangles=triangles,
    )
    triangle_axis.triplot(points_x, points_y, triangles=triangles, color="black")
    triangle_axis.set_title("tripcolor + triplot")

    canvas.savefig("plotly_vector_fields.html", backend="plotly")


if __name__ == "__main__":
    main()
