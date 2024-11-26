import subprocess
import os
import tempfile
from matplotlib.image import imread
import numpy as np
import re


import matplotlib.patches as patches
from maxplotlib.colors.colors import Color
from maxplotlib.linestyle.linestyle import Linestyle

class Node:
    def __init__(self, x, y, label="", content="", layer=0, **kwargs):
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
        self.layer = layer
        self.options = kwargs

    def to_tikz(self):
        """
        Generate the TikZ code for this node.

        Returns:
        - tikz_str (str): TikZ code string for the node.
        """
        options = ', '.join(f"{k.replace('_', ' ')}={v}" for k, v in self.options.items())
        if options:
            options = f"[{options}]"
        return f"    \\node{options} ({self.label}) at ({self.x}, {self.y}) {{{self.content}}};\n"

class Path:
    def __init__(self, nodes, path_actions=[], cycle=False, layer=0, **kwargs):
        """
        Represents a path (line) connecting multiple nodes.

        Parameters:
        - nodes (list of str): List of node names to connect.
        - **kwargs: Additional TikZ path options (e.g., style, color).
        """
        self.nodes = nodes
        self.path_actions = path_actions
        self.cycle = cycle
        self.layer = layer
        self.options = kwargs

    def to_tikz(self):
        """
        Generate the TikZ code for this path.

        Returns:
        - tikz_str (str): TikZ code string for the path.
        """
        options = ', '.join(f"{k.replace('_', ' ')}={v}" for k, v in self.options.items())
        if len(self.path_actions) > 0:
            options = ', '.join(self.path_actions) + ', ' + options
        if options:
            options = f"[{options}]"
        path_str = ' -- '.join(f"({node_label}.center)" for node_label in self.nodes)
        if self.cycle:
            path_str += ' -- cycle'
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
        self.layers = {}

        # Counter for unnamed nodes
        self._node_counter = 0

    def add_node(self, x, y, label=None, content="", layer = 0, **kwargs):
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
        if layer in self.layers:
            self.layers[layer].append(node)
        else:
            self.layers[layer] = [node]
        self._node_counter += 1
        return node

    def add_path(self, nodes, layer=0, **kwargs):
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
        if layer in self.layers:
            self.layers[layer].append(path)
        else:
            self.layers[layer] = [path]
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

        for key, layer_items in self.layers.items():
            tikz_script += f"\n    % Layer {key}\n"
            for item in layer_items:
                tikz_script += item.to_tikz()
        # # Add nodes
        # for node in self.nodes:
        #     tikz_script += node.to_tikz()

        # # Add paths
        # for path in self.paths:
        #     tikz_script += path.to_tikz()

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
        Plot all nodes and paths on the provided axis using Matplotlib.

        Parameters:
        - ax (matplotlib.axes.Axes): Axis on which to plot the figure.
        """

        # Plot paths first so they appear behind nodes
        for path in self.paths:
            x_coords = [next(node.x for node in self.nodes if node.label == label) for label in path.nodes]
            y_coords = [next(node.y for node in self.nodes if node.label == label) for label in path.nodes]

            # Parse path color
            path_color_spec = path.options.get('color', 'black')
            try:
                color = Color(path_color_spec).to_rgb()
            except ValueError as e:
                print(e)
                color = 'black'

            # Parse line width
            line_width_spec = path.options.get('line_width', 1)
            if isinstance(line_width_spec, str):
                match = re.match(r'([\d.]+)(pt)?', line_width_spec)
                if match:
                    line_width = float(match.group(1))
                else:
                    print(f"Invalid line width specification: '{line_width_spec}', defaulting to 1")
                    line_width = 1
            else:
                line_width = float(line_width_spec)

            # Parse line style using Linestyle class
            style_spec = path.options.get('style', 'solid')
            linestyle = Linestyle(style_spec).to_matplotlib()

            ax.plot(
                x_coords,
                y_coords,
                color=color,
                linewidth=line_width,
                linestyle=linestyle,
                zorder=1  # Lower z-order to place behind nodes
            )

        # Plot nodes after paths so they appear on top
        for node in self.nodes:
            # Determine shape and size
            shape = node.options.get('shape', 'circle')
            fill_color_spec = node.options.get('fill', 'white')
            edge_color_spec = node.options.get('draw', 'black')
            linewidth = float(node.options.get('line_width', 1))
            size = float(node.options.get('size', 1))

            # Parse colors using the Color class
            try:
                facecolor = Color(fill_color_spec).to_rgb()
            except ValueError as e:
                print(e)
                facecolor = 'white'

            try:
                edgecolor = Color(edge_color_spec).to_rgb()
            except ValueError as e:
                print(e)
                edgecolor = 'black'

            # Plot shapes
            if shape == 'circle':
                radius = size / 2
                circle = patches.Circle(
                    (node.x, node.y),
                    radius,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    zorder=2  # Higher z-order to place on top of paths
                )
                ax.add_patch(circle)
            elif shape == 'rectangle':
                width = height = size
                rect = patches.Rectangle(
                    (node.x - width / 2, node.y - height / 2),
                    width,
                    height,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    zorder=2  # Higher z-order
                )
                ax.add_patch(rect)
            else:
                # Default to circle if shape is unknown
                radius = size / 2
                circle = patches.Circle(
                    (node.x, node.y),
                    radius,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    zorder=2
                )
                ax.add_patch(circle)

            # Add text inside the shape
            if node.content:
                ax.text(
                    node.x,
                    node.y,
                    node.content,
                    fontsize=10,
                    ha='center',
                    va='center',
                    wrap=True,
                    zorder=3  # Even higher z-order for text
                )

        # Remove axes, ticks, and legend
        ax.axis('off')

        # Adjust plot limits
        all_x = [node.x for node in self.nodes]
        all_y = [node.y for node in self.nodes]
        padding = 1  # Adjust padding as needed
        ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
        ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
        ax.set_aspect('equal', adjustable='datalim')
