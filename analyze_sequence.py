import networkx as nx
import numpy as np
import os

from growing_neural_gas import GNG
import utils

# chose sequence to analyze:
from video_sequences.shortside_v2 import polygon_sequence


if __name__ == "__main__":
    MAX_NODES = 47
    NUM_ITERATIONS_FIRST_MODEL = 5000
    NUM_ITERATIONS_NEXT_MODELS = 301
    START_INDEX_POLYGON = 43
    OUTPUT_NAME = "sponge_center"

    time_id: str = utils.time_id()
    output_images_dir = f"output/{OUTPUT_NAME}_{time_id}/images"
    output_gif_file = f"output/{OUTPUT_NAME}_{time_id}/sequence.gif"
    output_npy_file = f"output/{OUTPUT_NAME}_{time_id}/sequence.npy"
    os.makedirs(f"./output/{OUTPUT_NAME}_{time_id}/images")
    polygon_points = np.array(polygon_sequence[START_INDEX_POLYGON])

    # First GNG is trained
    first_GNG = GNG(max_nodes=MAX_NODES)
    train_G = utils.create_polygon_graph(polygon_points)
    first_GNG.train(
        train_G,
        max_iterations=NUM_ITERATIONS_FIRST_MODEL,
        output_images_dir=output_images_dir,
        image_title=f"Polygon #{START_INDEX_POLYGON+1}",
        png_prefix=str(START_INDEX_POLYGON) + "_",
    )
    new_polygon_sequence = [utils.dict_to_list(nx.get_node_attributes(first_GNG.graph, "pos"))]
    if len(new_polygon_sequence[0]) < MAX_NODES:
        raise Exception("Maximum number of nodes was not reached. Needs more training time")

    gas_G = first_GNG.graph.copy()
    for t in range(START_INDEX_POLYGON - 1, -1, -1):
        print(f"Gas Graph {t}:")
        print("    #Nodes " + str(len(nx.get_node_attributes(gas_G, "pos").values())))
        polygon_points = np.array(polygon_sequence[t])
        train_G = utils.create_polygon_graph(polygon_points)
        gng = GNG(graph=gas_G, max_nodes=MAX_NODES, eps_b=0.1, eps_n=0.001)  # init with previus graph
        gng.train(
            train_G,
            max_iterations=NUM_ITERATIONS_NEXT_MODELS,
            output_images_dir=output_images_dir,
            image_title=f"Polygon #{t+1}",
            png_prefix=str(t) + "_",
        )
        gas_G = gng.graph
        new_polygon = utils.dict_to_list(nx.get_node_attributes(gas_G, "pos"))
        new_polygon_sequence.append(new_polygon)

    # analize all polygons after START_INDEX_POLYGON
    gas_G = first_GNG.graph.copy()
    for t in range(START_INDEX_POLYGON + 1, len(polygon_sequence)):
        print(f"Gas Graph {t}:")
        print("    #Nodes " + str(len(nx.get_node_attributes(gas_G, "pos").values())))
        polygon_points = np.array(polygon_sequence[t])
        train_G = utils.create_polygon_graph(polygon_points)
        gng = GNG(graph=gas_G, max_nodes=MAX_NODES)  # init with previus graph # eps_b=0.005, eps_n=0.0001
        gng.train(
            train_G,
            max_iterations=NUM_ITERATIONS_NEXT_MODELS,
            output_images_dir=output_images_dir,
            image_title=f"Polygon #{t+1}",
            png_prefix=str(t) + "_",
        )
        gas_G = gng.graph
        new_polygon = utils.dict_to_list(nx.get_node_attributes(gas_G, "pos"))
        new_polygon_sequence.insert(0, new_polygon)

    utils.convert_images_to_gif(output_images_dir, output_gif_file)
    np.save(output_npy_file, np.array(new_polygon_sequence))
    x = np.load(output_npy_file)
    print(x.shape)
