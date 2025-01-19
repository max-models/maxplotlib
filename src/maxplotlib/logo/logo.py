# import maxplotlib.canvas.canvas as canvas
from maxplotlib.subfigure.tikz_figure import TikzFigure
def tikz_logo():
    tikz = TikzFigure()

    path_actions = ['draw', 'rounded corners', 'line width=3']
    
    # M
    nodes = [[0,0],[0,3],[1,2],[2,3],[2,0]]
    for i, node_data in enumerate(nodes):
        tikz.add_node(node_data[0], node_data[1], f"M{i}", layer=0)
    tikz.add_path([f"M{i}" for i in range(len(nodes))], path_actions=path_actions, layer=1)
    
    # P
    nodes = [[3,0],[3,3],[4,2.5],[4,1.5],[3,1]]
    for i, node_data in enumerate(nodes):
        tikz.add_node(node_data[0], node_data[1], f"P{i}", layer=0)
    tikz.add_path([f"P{i}" for i in range(len(nodes))], path_actions=path_actions, layer=1)
    
    # L
    nodes = [[5,3],[5,0],[7,0]]
    for i, node_data in enumerate(nodes):
        tikz.add_node(node_data[0], node_data[1], f"L{i}", layer=0)
    tikz.add_path([f"L{i}" for i in range(len(nodes))], path_actions=path_actions, layer=1)

    return tikz