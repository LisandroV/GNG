from future.utils import iteritems
from past.builtins import xrange
import networkx as nx

from growing_neural_gas import GNG
import utils


def readFile(filename: str):
    """Read the file and return the indices as list of lists."""
    with open(filename) as file:
        array2d = [[int(digit) for digit in line.split()] for line in file]
    return array2d


def load_graph_from_file(filename: str) -> nx.Graph:
    """Create the graph and returns the networkx version of it 'G'."""
    array2d = readFile(filename)

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


if __name__ == "__main__":
    train_G = load_graph_from_file("s.txt")
    grng = GNG()
    output_images_dir = "output/example_1/images"
    output_gif = "output/example_1/sequence.gif"
    grng.train(train_G, max_iterations=3000, output_images_dir=output_images_dir)
    utils.convert_images_to_gif(output_images_dir, output_gif)
