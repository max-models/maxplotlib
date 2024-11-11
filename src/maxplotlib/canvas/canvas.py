import matplotlib.pyplot as plt
import maxplotlib.subfigure.line_plot as lp
import maxplotlib.backends.matplotlib.utils as plt_utils
class Canvas:
    def __init__(self, nrows=1, ncols=1, caption=None, description=None, label=None, figsize=(10, 6)):
        """
        Initialize the Canvas class for multiple subplots.

        Parameters:
        nrows (int): Number of subplot rows. Default is 1.
        ncols (int): Number of subplot columns. Default is 1.
        figsize (tuple): Figure size.
        """
        self._nrows = nrows
        self._ncols = ncols
        self._figsize = figsize
        self._caption = caption
        self._description = description
        self._label = label
        
        # Dictionary to store lines for each subplot
        # Key: (row, col), Value: list of lines with their data and kwargs
        self.subplots = {}
        self._num_subplots = 0

        self._subplot_matrix = [[None] * ncols for _ in range(nrows)]

    # Property getters
    @property
    def nrows(self):
        return self._nrows

    @property
    def ncols(self):
        return self._ncols

    @property
    def caption(self):
        return self._caption

    @property
    def description(self):
        return self._description

    @property
    def label(self):
        return self._label

    @property
    def figsize(self):
        return self._figsize

    @property
    def subplot_matrix(self):
        return self._subplot_matrix

    # Property setters
    @nrows.setter
    def nrows(self, value):
        self._nrows = value

    @ncols.setter
    def ncols(self, value):
        self._ncols = value

    @caption.setter
    def caption(self, value):
        self._caption = value

    @description.setter
    def description(self, value):
        self._description = value

    @label.setter
    def label(self, value):
        self._label = value

    @figsize.setter
    def figsize(self, value):
        self._figsize = value

    # Magic methods
    def __str__(self):
        return f"Canvas(nrows={self.nrows}, ncols={self.ncols}, figsize={self.figsize})"

    def __repr__(self):
        return f"Canvas(nrows={self.nrows}, ncols={self.ncols}, caption={self.caption}, label={self.label})"

    def __getitem__(self, key):
        """Allows accessing subplots by tuple index."""
        row, col = key
        if row >= self.nrows or col >= self.ncols:
            raise IndexError("Subplot index out of range")
        return self._subplot_matrix[row][col]

    def __setitem__(self, key, value):
        """Allows setting a subplot by tuple index."""
        row, col = key
        if row >= self.nrows or col >= self.ncols:
            raise IndexError("Subplot index out of range")
        self._subplot_matrix[row][col] = value

    def add_subplot(self, **kwargs):
        """
        Adds a subplot to the figure.

        Parameters:
        **kwargs: Arbitrary keyword arguments.
            - col (int): Column index for the subplot.
            - row (int): Row index for the subplot.
            - label (str): Label to identify the subplot.
        """
        col = kwargs.get('col', None)
        row = kwargs.get('row', None)
        label = kwargs.get('label', None)

        if row is None:
            for irow in range(self.nrows):
                has_none = any(item is None for item in self._subplot_matrix[irow])
                if has_none:
                    row = irow
                    break
        assert row is not None, "Not enough rows!"

        if col is None:
            for icol in range(self.ncols):
                if self._subplot_matrix[row][icol] is None:
                    col = icol
                    break
        assert col is not None, "Not enough columns!"
        
        # Initialize the LinePlot for the given subplot position
        line_plot = lp.LinePlot(**kwargs)
        self._subplot_matrix[row][col] = line_plot

        # Store the LinePlot instance by its position for easy access
        if label is None:
            self.subplots[(row, col)] = line_plot
        else:
            self.subplots[label] = line_plot
        return line_plot
    def savefig(self, filename, backend = 'matplotlib'):
        if backend == 'matplotlib':
            fig = self.plot(show=False, savefig=True)
            fig.savefig(filename)
            #plt.savefig(filename)
    # def add_line(self, label, x_data, y_data, **kwargs):

    def plot(self, backend='matplotlib', show=True, savefig=False):
        if backend == 'matplotlib':
            return self.plot_matplotlib(show=show, savefig=savefig)
    def plot_matplotlib(self, show=True, savefig=False):
        """
        Generate and optionally display the subplots.

        Parameters:
        filename (str, optional): Filename to save the figure.
        show (bool): Whether to display the plot.
        """
        fontsize = 14
        tex_fonts = plt_utils.setup_tex_fonts(fontsize=fontsize)
        plt_utils.setup_plotstyle(
            tex_fonts=tex_fonts,
            axes_grid=False,
            axes_grid_which="major",
            grid_alpha=1.0,
            grid_linestyle="dotted",
        )
        fig, axes = plt.subplots(self.nrows, self.ncols, figsize=self.figsize, squeeze=False)
        
        for (row, col), line_plot in self.subplots.items():
            ax = axes[row][col]
            line_plot.plot(ax)  # Assuming LinePlot's `plot` method accepts an axis object
            # ax.set_title(f"Subplot ({row}, {col})")

        # Set caption, labels, etc., if needed
        plt.tight_layout()
        
        if show:
            plt.show()
        else:
            plt.close()
        return fig

    # def generate_matplotlib_code(self):
    #     """Generate code for plotting the data using matplotlib."""
    #     code = "import matplotlib.pyplot as plt\n\n"
    #     code += f"fig, axes = plt.subplots({self.nrows}, {self.ncols}, figsize={self.figsize})\n\n"
    #     if self.nrows == 1 and self.ncols == 1:
    #         code += "axes = [axes]  # Single subplot\n\n"
    #     else:
    #         code += "axes = axes.flatten()\n\n"
    #     for idx, (subplot_idx, lines) in enumerate(self.subplots.items()):
    #         code += f"# Subplot {subplot_idx}\n"
    #         code += f"ax = axes[{idx}]\n"
    #         for line in lines:
    #             x_data = line['x']
    #             y_data = line['y']
    #             label = line['label']
    #             kwargs = line.get('kwargs', {})
    #             kwargs_str = ', '.join(f"{k}={repr(v)}" for k, v in kwargs.items())
    #             code += f"ax.plot({x_data}, {y_data}, label={repr(label)}"
    #             if kwargs_str:
    #                 code += f", {kwargs_str}"
    #             code += ")\n"
    #         code += "ax.set_xlabel('X-axis')\n"
    #         code += "ax.set_ylabel('Y-axis')\n"
    #         if self.nrows * self.ncols > 1:
    #             code += f"ax.set_title('Subplot {subplot_idx}')\n"
    #         code += "ax.legend()\n\n"
    #     code += "plt.tight_layout()\nplt.show()\n"
    #     return code

    # def generate_latex_plot(self):
    #     """Generate LaTeX code for plotting the data using pgfplots in subplots."""
    #     latex_code = "\\begin{figure}[h!]\n\\centering\n"
    #     total_subplots = self.nrows * self.ncols
    #     for idx in range(total_subplots):
    #         subplot_idx = divmod(idx, self.ncols)
    #         lines = self.subplots.get(subplot_idx, [])
    #         if not lines:
    #             continue  # Skip empty subplots
    #         latex_code += "\\begin{subfigure}[b]{0.45\\textwidth}\n"
    #         latex_code += "    \\begin{tikzpicture}\n"
    #         latex_code += "        \\begin{axis}[\n"
    #         latex_code += "            xlabel={X-axis},\n"
    #         latex_code += "            ylabel={Y-axis},\n"
    #         if self.nrows * self.ncols > 1:
    #             latex_code += f"            title={{Subplot {subplot_idx}}},\n"
    #         latex_code += "            legend style={at={(1.05,1)}, anchor=north west},\n"
    #         latex_code += "            legend entries={" + ", ".join(f"{{{line['label']}}}" for line in lines) + "}\n"
    #         latex_code += "        ]\n"
    #         for line in lines:
    #             options = []
    #             kwargs = line.get('kwargs', {})
    #             if 'color' in kwargs:
    #                 options.append(f"color={kwargs['color']}")
    #             if 'linestyle' in kwargs:
    #                 linestyle_map = {'-': 'solid', '--': 'dashed', '-.': 'dash dot', ':': 'dotted'}
    #                 linestyle = linestyle_map.get(kwargs['linestyle'], kwargs['linestyle'])
    #                 options.append(f"style={linestyle}")
    #             options_str = f"[{', '.join(options)}]" if options else ""
    #             latex_code += f"        \\addplot {options_str} coordinates {{\n"
    #             for x, y in zip(line['x'], line['y']):
    #                 latex_code += f"            ({x}, {y})\n"
    #             latex_code += "        };\n"
    #         latex_code += "        \\end{axis}\n"
    #         latex_code += "    \\end{tikzpicture}\n"
    #         latex_code += "\\end{subfigure}\n"
    #         latex_code += "\\hfill\n" if (idx + 1) % self.ncols != 0 else "\n"
    #     latex_code += "\\caption{Multiple Subplots}\n"
    #     latex_code += "\\end{figure}\n"
    #     return latex_code
    

if __name__ == '__main__':
    c = Canvas(ncols=2,nrows=2)
    sp = c.add_subplot()
    sp.add_line("Line 1", [0, 1, 2, 3], [0, 1, 4, 9])
    c.plot()
    print('done')