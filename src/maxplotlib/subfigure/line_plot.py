class LinePlot:
    def __init__(self, figsize=(10, 6), caption=None, description=None, label=None):
        """
        Initialize the LinePlot class for a subplot.

        Parameters:
        figsize (tuple): Figure size.
        caption (str): Caption for the plot.
        description (str): Description of the plot.
        label (str): Label for the plot.
        """
        self.figsize = figsize
        # List to store line data, with each entry containing x and y data, label, and plot kwargs
        self.line_data = []

        self.caption = caption
        self.description = description
        self.label = label

    def add_caption(self, caption):
        self.caption = caption    

    def add_line(self, x_data, y_data, **kwargs):
        """
        Add a line to the plot.

        Parameters:
        label (str): Label for the line.
        x_data (list): X-axis data.
        y_data (list): Y-axis data.
        **kwargs: Additional keyword arguments for the plot (e.g., color, linestyle).
        """
        self.line_data.append({
            'x': x_data,
            'y': y_data,
            'kwargs': kwargs
        })

    def plot(self, ax):
        """
        Plot all lines on the provided axis.

        Parameters:
        ax (matplotlib.axes.Axes): Axis on which to plot the lines.
        """
        for line in self.line_data:
            ax.plot(
                line['x'], line['y'],
                # label=line['label'],
                **line['kwargs']
            )
        if self.caption:
            ax.set_title(self.caption)
        if self.label:
            ax.set_ylabel(self.label)
        ax.set_xlabel("X-axis")
        ax.legend()



if __name__ == "__main__":
    plotter = LinePlot()
    plotter.add_line("Line 1", [0, 1, 2, 3], [0, 1, 4, 9])
    plotter.add_line("Line 2", [0, 1, 2, 3], [0, 2, 3, 6])
    latex_code = plotter.generate_latex_plot()
    with open("figures/latex_code.tex", "w") as f:
        f.write(latex_code)
    print(latex_code)
