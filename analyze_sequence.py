import networkx as nx
import numpy as np

from growing_neural_gas import GNG
from sequence2 import polygon_sequence
import utils


if __name__ == "__main__":
    MAX_NODES = 47
    NUM_ITERATIONS_FIRST_MODEL = 5000
    NUM_ITERATIONS_NEXT_MODELS = 200
    START_INDEX_POLYGON = 40
    OUTPUT_NAME = "sequence2_B"

    polygon_points = np.array(polygon_sequence[START_INDEX_POLYGON])
    train_G = utils.create_polygon_graph(polygon_points)
    first_GNG = GNG(max_nodes=MAX_NODES)
    output_images_dir = f"images/{OUTPUT_NAME}"
    output_gif = f"output_{OUTPUT_NAME}.gif"
    first_GNG.train(train_G, max_iterations=NUM_ITERATIONS_FIRST_MODEL, output_images_dir=output_images_dir, image_title=f"Polygon #{START_INDEX_POLYGON+1}", png_prefix=str(START_INDEX_POLYGON)+"_")
    new_polygon_sequence =  [utils.dict_to_list(nx.get_node_attributes(first_GNG.graph, "pos"))]
    if len(new_polygon_sequence[0]) < MAX_NODES:
        raise Exception("Maximum number of nodes was not reached. Needs more training time")

    # analize all polygons after START_INDEX_POLYGON
    gas_G = first_GNG.graph.copy()
    for t in range(START_INDEX_POLYGON + 1, len(polygon_sequence)):
        print(f"Gas Graph {t}:")
        print("    #Nodes " + str(len(nx.get_node_attributes(gas_G, "pos").values())))
        polygon_points = np.array(polygon_sequence[t])
        train_G = utils.create_polygon_graph(polygon_points)
        gng = GNG(graph=gas_G, max_nodes=MAX_NODES)# init with previus graph
        gng.train(train_G, max_iterations=NUM_ITERATIONS_NEXT_MODELS, output_images_dir=output_images_dir, image_title=f"Polygon #{t+1}", png_prefix=str(t)+"_")
        gas_G = gng.graph
        new_polygon = utils.dict_to_list(nx.get_node_attributes(gas_G, "pos"))
        new_polygon_sequence.insert(0, new_polygon)

    # analize all polygons before START_INDEX_POLYGON
    gas_G = first_GNG.graph.copy()
    for t in range(START_INDEX_POLYGON-1, -1, -1): 
        print(f"Gas Graph {t}:")
        print("    #Nodes " + str(len(nx.get_node_attributes(gas_G, "pos").values())))
        polygon_points = np.array(polygon_sequence[t])
        train_G = utils.create_polygon_graph(polygon_points)
        gng = GNG(graph=gas_G, max_nodes=MAX_NODES)# init with previus graph
        gng.train(train_G, max_iterations=NUM_ITERATIONS_NEXT_MODELS, output_images_dir=output_images_dir, image_title=f"Polygon #{t+1}", png_prefix=str(t)+"_")
        gas_G = gng.graph
        new_polygon = utils.dict_to_list(nx.get_node_attributes(gas_G, "pos"))
        new_polygon_sequence.append(new_polygon)

    utils.convert_images_to_gif(output_images_dir, output_gif)
    np.save(f"{OUTPUT_NAME}.npy", np.array(new_polygon_sequence))
    x = np.load(f"{OUTPUT_NAME}.npy")
    print(x.shape)