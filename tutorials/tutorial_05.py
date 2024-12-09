import maxplotlib.canvas.canvas as canvas
import numpy as np

c = canvas.Canvas(width=2000, ratio=0.5)
sp = c.add_subplot(grid=True, xlabel='x', ylabel='y')
x = np.arange(0,2 * np.pi, 0.01)
y = np.sin(x)
sp.add_line(x, y, label=r"$\sin(x)$")
# c.plot()
c.savefig(filename='figures/tutorial_05_figure_01.pdf')