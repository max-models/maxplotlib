import numpy as np
import maxplotlib.canvas.canvas as canvas
import matplotlib.pyplot as plt
c = canvas.Canvas(width=2000, ratio=0.5)
sp = c.add_subplot(grid=False, xlabel='x', ylabel='y')
# sp.add_line([0, 1, 2, 3], [0, 1, 4, 9], label="Line 1",layer=1)
data = np.random.random((10,10))
sp.add_imshow(data, extent=[1,10,1,20],layer=1)
#c.plot()
c.savefig(layer_by_layer=True, filename='figures/tutorial_06_figure_01.pdf')
