from future.utils import iteritems
from past.builtins import xrange
import networkx as nx
import numpy as np

from growing_neural_gas import GNG
import utils

pos = None
G = None


def readFile():
    """Read the file and return the indices as list of lists."""
    filename = "s.txt"
    with open(filename) as file:
        array2d = [[int(digit) for digit in line.split()] for line in file]
    return array2d


def read_file_draw_graph() -> nx.Graph:
    """Create the graph and returns the networkx version of it 'G'."""
    global pos
    global G
    array2d = readFile()

    ROW, COLUMN = len(array2d), len(array2d[0])
    count = 0

    G = nx.Graph()

    for j in xrange(COLUMN):
        for i in xrange(ROW):
            if array2d[ROW - 1 - i][j] == 0:
                G.add_node(count, pos=(j, i))
                count += 1

    pos = nx.get_node_attributes(G, "pos")

    for index in pos.keys():
        for index2 in pos.keys():
            if pos[index][0] == pos[index2][0] and pos[index][1] == pos[index2][1] - 1:
                G.add_edge(index, index2, weight=1)
            if pos[index][1] == pos[index2][1] and pos[index][0] == pos[index2][0] - 1:
                G.add_edge(index, index2, weight=1)

    return G


def load_graph():
    """."""
    global pos, G
    G = read_file_draw_graph()

    inList = []
    for key, value in iteritems(pos):
        inList.append([value[0], value[1]])

    mat = np.array(inList, dtype="float64")
    return mat


if __name__ == "__main__":
    data = load_graph()
    grng = GNG(data, G)
    output_images_dir = "images/example_1"
    output_gif = "output_1.gif"
    if grng is not None:
        grng.train(max_iterations=3000, output_images_dir=output_images_dir)
        utils.convert_images_to_gif(output_images_dir, output_gif)
