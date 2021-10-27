import matplotlib.pyplot as plt
import numpy as np
import sys


def read_normalize_reshape():
    image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
    centroids = np.loadtxt(centroids_fname)  # load centroids
    orig_pixels = plt.imread(image_fname)
    pixels = orig_pixels.astype(float) / 255.
    # Reshape the image(128x128x3) into an Nx3 matrix where N = number of pixels.
    pixels = pixels.reshape(-1, 3)
    return centroids, pixels, out_fname


def calculate_centroids_until_convergance(centroids, pixels, out_fname):
    '''
    1. I will open a file named after out_name
    2. I will iterate in while true - condition will be new cents are not equal to old ones
        - init k sets (one for each cent)
        2.1 for each pixel :
            -init minimum dist and update after each of the following calcs
            2.1.1 calc dist to cent 1
            2.1.2 calc dist to cent 2
            .
            .
            2.1.k calc dist to cent k
            - add the pixel to cent x (the closest) set
        - now that each pixel belong to cent, calc avg within the cent
        - new cents = update each cent to be its avg
        - print their format : iter ...
        - if new cents = old cents : break or condition is broken




    :param centroids:
    :param pixels:
    :param out_fname:
    :return:

    open questions : Does cents are in calculation of their own avg?
    tmp answer :  test my results with out1.txt and out3.txt
    '''
    pass

if __name__ == '__main__':
    initial_centroids, pixels, out_fname = read_normalize_reshape()
    # calculate_centroids_until_convergance(initial_centroids, pixels, out_fname)
    '''
    This is the way to calc 3d points distance
    '''
    # dist = np.linalg.norm(a - b)



