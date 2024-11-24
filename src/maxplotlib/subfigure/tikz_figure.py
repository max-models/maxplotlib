import subprocess
import os
import tempfile
from matplotlib.image import imread

class Node:
    def __init__(self, x, y, label="", content="",**kwargs):
        """
        Represents a TikZ node.

        Parameters:
        - x (float): X-coordinate of the node.
        - y (float): Y-coordinate of the node.
        - name (str, optional): Name of the node. If None, a default name will be assigned.
        - **kwargs: Additional TikZ node options (e.g., shape, color).
        """
        self.x = x
        self.y = y
        self.label = label
        self.content = content
        self.options = kwargs

    def to_tikz(self):
        """
        Generate the TikZ code for this node.

        Returns:
        - tikz_str (str): TikZ code string for the node.
        """
        options = ', '.join(f"{k.replace('_', ' ')}={{{v}}}" for k, v in self.options.items())
        if options:
            options = f"[{options}]"
        return f"    \\node{options} ({self.label}) at ({self.x}, {self.y}) {{{self.content}}};\n"

class Path:
    def __init__(self, nodes, **kwargs):
        """
        Represents a path (line) connecting multiple nodes.

        Parameters:
        - nodes (list of str): List of node names to connect.
        - **kwargs: Additional TikZ path options (e.g., style, color).
        """
        self.nodes = nodes
        self.options = kwargs

    def to_tikz(self):
        """
        Generate the TikZ code for this path.

        Returns:
        - tikz_str (str): TikZ code string for the path.
        """
        options = ', '.join(f"{k.replace('_', ' ')}={{{v}}}" for k, v in self.options.items())
        if options:
            options = f"[{options}]"
        path_str = ' -- '.join(f"({node_label})" for node_label in self.nodes)
        return f"    \\draw{options} {path_str};\n"

class TikzFigure:
    def __init__(self, **kwargs):
        """
        Initialize the TikzFigure class for creating TikZ figures.

        Parameters:
        **kwargs: Arbitrary keyword arguments.
            - figsize (tuple): Figure size (default is (10, 6)).
            - caption (str): Caption for the figure.
            - description (str): Description of the figure.
            - label (str): Label for the figure.
            - grid (bool): Whether to display grid lines (default is False).
            TODO: Add all options
        """
        # Set default values
        self._figsize = kwargs.get('figsize', (10, 6))
        self._caption = kwargs.get('caption', None)
        self._description = kwargs.get('description', None)
        self._label = kwargs.get('label', None)
        self._grid = kwargs.get('grid', False)

        # Initialize lists to hold Node and Path objects
        self.nodes = []
        self.paths = []

        # Counter for unnamed nodes
        self._node_counter = 0

    def add_node(self, x, y, label=None, content="", **kwargs):
        """
        Add a node to the TikZ figure.

        Parameters:
        - x (float): X-coordinate of the node.
        - y (float): Y-coordinate of the node.
        - label (str, optional): Label of the node. If None, a default label will be assigned.
        - **kwargs: Additional TikZ node options (e.g., shape, color).

        Returns:
        - node (Node): The Node object that was added.
        """
        if label is None:
            label = f"node{self._node_counter}"
        node = Node(x=x, y=y, label=label, content=content, **kwargs)
        self.nodes.append(node)
        self._node_counter += 1
        return node

    def add_path(self, nodes, **kwargs):
        """
        Add a line or path connecting multiple nodes.

        Parameters:
        - nodes (list of str): List of node names to connect.
        - **kwargs: Additional TikZ path options (e.g., style, color).

        Examples:
        - add_path(['A', 'B', 'C'], color='blue')
          Connects nodes A -> B -> C with a blue line.
        """
        if not isinstance(nodes, list):
            raise ValueError("nodes parameter must be a list of node names.")
        
        nodes = [
            node.label if isinstance(node, Node)
            else node if isinstance(node, str)
            else ValueError(f"Invalid node type: {type(node)}")
            for node in nodes
        ]

        path = Path(nodes, **kwargs)
        self.paths.append(path)
        return path

    def generate_tikz(self):
        """
        Generate the TikZ script for the figure.

        Returns:
        - tikz_script (str): The TikZ script as a string.
        """
        tikz_script = "\\begin{tikzpicture}\n"

        # Add grid if enabled
        if self._grid:
            tikz_script += "    \\draw[step=1cm, gray, very thin] (-10,-10) grid (10,10);\n"

        # Add nodes
        for node in self.nodes:
            tikz_script += node.to_tikz()

        # Add paths
        for path in self.paths:
            tikz_script += path.to_tikz()

        tikz_script += "\\end{tikzpicture}"

        # Wrap in figure environment if necessary
        if self._caption or self._description or self._label:
            figure_env = "\\begin{figure}\n" + tikz_script + "\n"
            if self._caption:
                figure_env += f"    \\caption{{{self._caption}}}\n"
            if self._label:
                figure_env += f"    \\label{{{self._label}}}\n"
            figure_env += "\\end{figure}"
            tikz_script = figure_env

        return tikz_script

    def compile_pdf(self, filename='output.pdf'):
        """
        Compile the TikZ script into a PDF using pdflatex.

        Parameters:
        - filename (str): The name of the output PDF file (default is 'output.pdf').

        Notes:
        - Requires 'pdflatex' to be installed and accessible from the command line.
        """
        tikz_code = self.generate_tikz()

        # Create a minimal LaTeX document
        latex_document = (
            "\\documentclass[border=10pt]{standalone}\n"
            "\\usepackage{tikz}\n"
            "\\begin{document}\n"
            f"{tikz_code}\n"
            "\\end{document}"
        )

        # Use a temporary directory to store the LaTeX files
        with tempfile.TemporaryDirectory() as tempdir:
            tex_file = os.path.join(tempdir, 'figure.tex')
            with open(tex_file, 'w') as f:
                f.write(latex_document)

            # Run pdflatex
            try:
                subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', tex_file],
                    cwd=tempdir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError as e:
                print("An error occurred while compiling the LaTeX document:")
                print(e.stderr.decode())
                return

            # Move the output PDF to the desired location
            pdf_output = os.path.join(tempdir, 'figure.pdf')
            if os.path.exists(pdf_output):
                os.rename(pdf_output, filename)
                print(f"PDF successfully compiled and saved as '{filename}'.")
            else:
                print("PDF compilation failed. Please check the LaTeX log for details.")
    def plot_matplotlib(self, ax):
        """
        Plot all lines on the provided axis.

        Parameters:
        ax (matplotlib.axes.Axes): Axis on which to plot the lines.
        """
        
