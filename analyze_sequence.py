import networkx as nx
import numpy as np

from growing_neural_gas import GNG
from sequence import polygon_sequence
import utils


if __name__ == "__main__":
    polygon_points = np.array(polygon_sequence[50])
    G = utils.create_polygon_graph(polygon_points)
    grng = GNG(G, max_nodes=47)
    output_images_dir = "images/sequence"
    output_gif = "output_sequence.gif"
    grng.train(max_iterations=3000, output_images_dir=output_images_dir)
    utils.convert_images_to_gif(output_images_dir, output_gif)
