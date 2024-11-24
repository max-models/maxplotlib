import subprocess
import os
import tempfile

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

        # Initialize nodes and lines
        self.nodes = []
        self.lines = []

    def add_node(self, name, x, y, **kwargs):
        """
        Add a node to the TikZ figure.

        Parameters:
        - name (str): Name of the node.
        - x (float): X-coordinate of the node.
        - y (float): Y-coordinate of the node.
        - **kwargs: Additional TikZ node options (e.g., shape, color).
        """
        node = {
            'name': name,
            'x': x,
            'y': y,
            'options': kwargs
        }
        self.nodes.append(node)

    def add_line(self, nodes, **kwargs):
        """
        Add a line or path connecting multiple nodes.

        Parameters:
        - nodes (list of str): List of node names to connect.
        - **kwargs: Additional TikZ line options (e.g., style, color).

        Examples:
        - add_line(['A', 'B', 'C'], color='blue')
          Connects nodes A -> B -> C with a blue line.
        """
        if not isinstance(nodes, list):
            raise ValueError("nodes parameter must be a list of node names.")

        line = {
            'nodes': nodes,
            'options': kwargs
        }
        self.lines.append(line)

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
            options = ', '.join(f"{k.replace('_',' ')}={{{v}}}" for k, v in node['options'].items())
            if options:
                options = f"[{options}]"
            tikz_script += f"    \\node{options} ({node['name']}) at ({node['x']}, {node['y']}) {{{node['name']}}};\n"

        # Add lines
        for line in self.lines:
            options = ', '.join(f"{k.replace('_',' ')}={{{v}}}" for k, v in line['options'].items())
            if options:
                options = f"[{options}]"
            # Create the path by connecting all nodes in the list
            path = ' -- '.join(f"({node_name})" for node_name in line['nodes'])
            tikz_script += f"    \\draw{options} {path};\n"

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
