import numpy as np

import maxplotlib.canvas.canvas as canvas

c = canvas.Canvas(width=2000, ratio=0.5)
sp = c.add_subplot(grid=True, xlabel="x", ylabel="y")

node_a = sp.add_node(
    0, 0, "A", content="Node A", shape="circle", draw="black", fill="blue!20"
)
node_b = sp.add_node(
    1, 1, "B", content="Node B", shape="circle", draw="black", fill="blue!20"
)
# sp.add_node(2, 2, 'B', content="$a^2 + b^2 = c^2$", shape='rectangle', draw='red', fill='white', layer=1)
# sp.add_node(2, 5, 'C', shape='rectangle', draw='red', fill='red')
# last_node = sp.add_node(-1, 5, shape='rectangle', draw='red', fill='red', layer=-10)

# Add a line between nodes
sp.add_path(["A", "B"], color="green", style="solid", line_width="2", layer=-5)

x = np.arange(0, 2 * np.pi, 0.01)
y = np.sin(x)
sp.add_line(x, y, label=r"$\sin(x)$")
# c.plot()
c.savefig(filename="figures/tutorial_05_figure_01.pdf")
