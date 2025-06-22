import maxplotlib.canvas.canvas as canvas

c = canvas.Canvas(width="17cm", ratio=0.5)
sp = c.add_subplot(
    grid=False, xlabel="(x - 10) * 0.1", ylabel="10y", yscale=10, xshift=-10, xscale=0.1
)
sp.add_line([0, 1, 2, 3], [0, 1, 4, 9], label="Line 1", layer=0)
sp.add_line(
    [0, 1, 2, 3], [0, 2, 3, 4], linestyle="dashed", color="red", label="Line 2", layer=1
)
# c.plot()
c.savefig(layer_by_layer=True, filename="figures/tutorial_03_figure_01.pdf")
