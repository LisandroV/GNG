import glob
import imageio
import networkx as nx
import numpy as np
import re


def sort_nicely(limages):
    """."""

    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    limages = sorted(limages, key=alphanum_key)
    return limages


def convert_images_to_gif(output_images_dir, output_gif):
    """Convert a list of images to a gif."""

    image_dir = "{0}/*.png".format(output_images_dir)
    list_images = glob.glob(image_dir)
    file_names = sort_nicely(list_images)
    images = [imageio.imread(fn) for fn in file_names]
    imageio.mimsave(output_gif, images)


def create_polygon_graph(coords: np.ndarray) -> nx.Graph:
    """
    Creates a graph with the coordinates.
    The graph will be a polygon.
    """
    G = nx.Graph()
    # add nodes
    for index, p in enumerate(coords):
        G.add_node(index, pos=tuple(p))

    # add edges
    num_points = len(coords)
    for index in range(num_points):
        G.add_edge(index, (index + 1) % num_points)
    return G


dict_to_list = lambda dict: [dict[key] for key in dict]
