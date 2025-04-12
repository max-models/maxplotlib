"""
Tutorial 4.

Add raw tikz code to the tikz subplot.
"""

import maxplotlib.canvas.canvas as canvas

c = canvas.Canvas(width=800, ratio=0.5)
tikz = c.add_tikzfigure(grid=False)

# Add nodes
tikz.add_node(0, 0, "A", shape="circle", draw="black", fill="blue", layer=0)
tikz.add_node(10, 0, "B", shape="circle", draw="black", fill="blue", layer=0)
tikz.add_node(10, 10, "C", shape="circle", draw="black", fill="blue", layer=0)
tikz.add_node(0, 10, "D", shape="circle", draw="black", fill="blue", layer=2)


# Add a line between nodes
tikz.add_path(
    ["A", "B", "C", "D"],
    path_actions=["draw", "rounded corners"],
    fill="red",
    opacity=0.5,
    cycle=True,
    layer=1,
)

raw_tikz = r"""
\foreach \i in {0, 45, 90, 135, 180, 225, 270, 315} {
    % Place a node at angle \i
    \node[circle, draw, fill=green] at (\i:3) (N\i) {};
}

% Draw lines connecting the nodes
\foreach \i/\j in {0/45, 45/90, 90/135, 135/180, 180/225, 225/270, 270/315, 315/0} {
    \draw (N\i) -- (N\j);
}
"""

tikz.add_raw(raw_tikz)

tikz.add_node(0.5, 0.5, content="Cube", layer=10)

# Generate the TikZ script
script = tikz.generate_tikz()
print(script)
# print(tikz.generate_standalone())
tikz.compile_pdf("figures/tutorial_04_figure_01.pdf")
#
