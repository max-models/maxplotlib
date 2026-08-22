"""Plotly examples for scalar fields, triangular meshes, and tables."""

import numpy as np

from maxplotlib import Canvas


def main() -> None:
    coordinates = np.linspace(-2, 2, 30)
    xx, yy = np.meshgrid(coordinates, coordinates)
    values = np.exp(-(xx**2 + yy**2))

    canvas, (field_axis, table_axis) = Canvas.subplots(ncols=2, width="14cm")
    field_axis.pcolor(coordinates, coordinates, values, cmap="Viridis")
    field_axis.contour(coordinates, coordinates, values, levels=6, color="white")
    field_axis.set_title("pcolor + contour")
    field_axis.set_xlabel("x")
    field_axis.set_ylabel("y")

    table_axis.table(
        cellText=[["Viridis", "30×30"], ["Peak", f"{values.max():.2f}"]],
        colLabels=["Field", "Value"],
        rowLabels=["Map", "Statistic"],
    )
    table_axis.set_axis_off()
    table_axis.set_title("table")

    canvas.savefig("plotly_field_primitives.html", backend="plotly")


if __name__ == "__main__":
    main()
