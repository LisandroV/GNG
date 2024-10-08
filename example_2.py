import numpy as np

from growing_neural_gas import GNG
import utils


if __name__ == "__main__":
    # Graph defined with coordinates
    data = np.array(
        [
            [239.0049298, 160.67136298],
            [239.63813117, 175.88549987],
            [236.45600277, 187.70997627],
            [231.6018699, 198.74273079],
            [219.12219667, 203.68898851],
            [205.67636249, 204.65116477],
            [191.60745019, 205.74988693],
            [175.96343651, 205.97366309],
            [163.74898094, 205.78650301],
            [151.93925348, 204.43092855],
            [136.80507369, 204.41020109],
            [128.85407618, 204.84589351],
            [120.89713683, 205.28201698],
            [110.79691304, 205.76387005],
            [100.29934764, 205.96986856],
            [85.39427526, 205.52912972],
            [77.89865649, 205.25407014],
            [70.41232748, 204.90729277],
            [61.24252067, 200.91664683],
            [56.70769205, 191.87102681],
            [55.17580039, 177.36290788],
            [54.9607657, 167.97422532],
            [54.63553199, 157.81898855],
            [54.00840503, 141.33474613],
            [53.94495292, 130.10985744],
            [53.54710414, 119.40181151],
            [52.04611431, 104.86463933],
            [59.86082517, 97.19351829],
            [70.46389078, 94.16064177],
            [84.09664262, 94.55903181],
            [95.1608796, 95.10921638],
            [107.73003073, 94.60989793],
            [122.85303998, 95.63961973],
            [130.64871668, 95.91150183],
            [138.66128057, 96.47776602],
            [152.16023967, 97.3639549],
            [167.95492588, 97.99086899],
            [182.77872292, 97.86565601],
            [196.90133136, 97.03280369],
            [213.58516529, 97.04388666],
            [222.06478405, 97.45789814],
            [231.00650596, 98.02849205],
            [242.49284587, 106.89950555],
            [241.96316971, 118.24571877],
            [239.4950714, 129.27728298],
            [239.02576176, 143.07971147],
            [239.01339427, 151.4838181],
        ]
    )
    train_G = utils.create_polygon_graph(data)
    grng = GNG(max_nodes=47)
    output_images_dir = "output/example_2/images"
    output_gif = "output/example_2/sequence.gif"
    grng.train(train_G, max_iterations=3000, output_images_dir=output_images_dir, image_title="Polygon")
    utils.convert_images_to_gif(output_images_dir, output_gif)
