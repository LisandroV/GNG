import networkx as nx
import numpy as np

from growing_neural_gas import GNG
from sequence import polygon_sequence
import utils


if __name__ == "__main__":
    start_step = 50
    polygon_points = np.array(polygon_sequence[start_step])
    train_G = utils.create_polygon_graph(polygon_points)
    MAX_NODES = 10
    first_GNG = GNG(max_nodes=MAX_NODES)
    output_images_dir = "images/sequence"
    output_gif = "output_sequence.gif"
    first_GNG.train(train_G, max_iterations=1000, output_images_dir=output_images_dir, image_title="Polygon #50", png_prefix=str(start_step)+"_")
    new_polygon_sequence =  [utils.dict_to_list(nx.get_node_attributes(first_GNG.graph, "pos"))]

    gas_G = first_GNG.graph
    coords = utils.dict_to_list(nx.get_node_attributes(gas_G, "pos"))
    for t in range(start_step + 1, len(polygon_sequence)): # analize all polygons after start_step
        print(f"Gas Graph {t}:")
        print(nx.get_node_attributes(gas_G, "pos").values())
        print("Nodes " + str(len(nx.get_node_attributes(gas_G, "pos").values())))
        polygon_points = np.array(polygon_sequence[t])
        train_G = utils.create_polygon_graph(polygon_points)
        gng = GNG(graph=gas_G, max_nodes=MAX_NODES)#init with previus graph
        gng.train(train_G, max_iterations=200, output_images_dir=output_images_dir, image_title=f"Polygon #{t+1}", png_prefix=str(t)+"_")
        gas_G = gng.graph
        new_polygon = [utils.dict_to_list(nx.get_node_attributes(gas_G, "pos"))]
        new_polygon_sequence.insert(0, new_polygon)

    gas_G = first_GNG.graph
    for t in range(start_step-1, -1, -1): # analize all polygons before start_step
        print(f"Gas Graph {t}:")
        print(nx.get_node_attributes(gas_G, "pos").values())
        print("Nodes " + str(len(nx.get_node_attributes(gas_G, "pos").values())))
        polygon_points = np.array(polygon_sequence[t])
        train_G = utils.create_polygon_graph(polygon_points)
        gng = GNG(graph=gas_G, max_nodes=MAX_NODES)#init with previus graph
        gng.train(train_G, max_iterations=200, output_images_dir=output_images_dir, image_title=f"Polygon #{t+1}", png_prefix=str(t)+"_")
        gas_G = gng.graph
        new_polygon = [utils.dict_to_list(nx.get_node_attributes(gas_G, "pos"))]
        new_polygon_sequence.append(new_polygon)

    utils.convert_images_to_gif(output_images_dir, output_gif)
    import pdb;pdb.set_trace();