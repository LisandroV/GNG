import cv2
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from pprint import pprint
import math
# SCRIPT TO GET THE RED POINTS FROM THE CONTOUR

def plot_contour(points_array):
    fig = plt.figure("Hiperparámetros en la búsqueda aleatoria")
    fig.suptitle("Hiperparámetros en la búsqueda aleatoria")
    ax = fig.add_subplot(111)

    x = points_array[:,0]
    y = points_array[:,1]

    ax.scatter(x, y)
    plt.show()

def get_contour_points(frame: int):
    #read image
    #src = cv2.imread('/Users/ndroid/Documents/tesis/repos/Deformation-Tracker/data/sponge_shortside/images/frame50.jpg', cv2.IMREAD_UNCHANGED)
    src = cv2.imread(f"contour_images/frame{str(frame)}.jpg")

    cv2.imshow('GFG', src)

    #extract red channel
    red_channel = src
    red_points = []

    freq = 0
    last_x = 0
    distance_threshold = 5
    for y in range(964):
        for x in range(1288):
            #print(src[x,y])
            if red_channel[y][x][0]  != src[y][x][1] and red_channel[y][x][2] > 80 and red_channel[y][x][0]< 70:

                if len(red_points)==0:
                    red_points.append([x,y])
                has_neigbors = False
                for red_point in red_points:
                    if math.dist(red_point, [x,y]) < distance_threshold:
                        has_neigbors = True

                if not has_neigbors:
                    red_points.append([x,y])

    rp_array = np.array(red_points)
    #plot_contour(rp_array)

    return rp_array.tolist()


if __name__ == "__main__":
    print("contour = [")
    for time_step in range(84):
        contour_points = get_contour_points(time_step)
        print(str(contour_points)+",")
    print("]")
