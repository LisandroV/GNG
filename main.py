import numpy as np
import networkx as nx
import imageio
from matplotlib import pylab as pl
import re
import glob
from past.builtins import xrange
from future.utils import iteritems

from growing_neural_gas import GNG

pos = None
G = None


def readFile():
    """Read the file and return the indices as list of lists."""
    filename = 's.txt'
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

    pos = nx.get_node_attributes(G, 'pos')

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

    mat = np.array(inList, dtype='float64')
    return mat


def sort_nicely(limages):
    """."""
    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    limages = sorted(limages, key=alphanum_key)
    return limages


def convert_images_to_gif(output_images_dir, output_gif):
    """Convert a list of images to a gif."""

    image_dir = "{0}/*.png".format(output_images_dir)
    list_images = glob.glob(image_dir)
    file_names = sort_nicely(list_images)
    images = [imageio.imread(fn) for fn in file_names]
    imageio.mimsave(output_gif, images)


if __name__ == "__main__":

    data = load_graph()
    grng = GNG(data,G)
    output_images_dir = 'images'
    output_gif = "output.gif"
    if grng is not None:
        grng.train(max_iterations=10000)
        convert_images_to_gif(output_images_dir, output_gif)
