import matplotlib.pyplot as plt
import numpy as np
import sys





def calculate_centroids_until_convergance(centroids, pixels):
    # init old_cents to be as in file
    old_cents = centroids
    new_cents = np.empty_like(old_cents)
    k = len(old_cents)
    iteration = 0
    iters_list_for_plot = []
    costs_list_for_plot = []
    while True:
        # map from centroid to all pixel belongs to him in this iteration to come
        cents_to_its_pixels = {}
        total_distance_of_pixels = 0
        for i, centroid in enumerate(old_cents):
            cents_to_its_pixels[i] = []
        for pixel_index, pixel in enumerate(pixels):
            min_dist = sys.maxsize
            index_of_min = -1
            for i, centroid in enumerate(old_cents):
                pixel_dist_from_cent = np.linalg.norm(pixel - centroid)
                if min_dist > pixel_dist_from_cent:
                    min_dist = pixel_dist_from_cent
                    index_of_min = i
            cents_to_its_pixels[index_of_min].append(pixel_index)
            total_distance_of_pixels += min_dist
        for key in cents_to_its_pixels.keys():
            pixels_in_group = []
            for i in cents_to_its_pixels[key]:
                pixels_in_group.append(pixels[i])
            new_cents[key] = np.average(pixels_in_group, axis=0)
        new_cents = new_cents.round(4)
        cost = total_distance_of_pixels / len(pixels)
        costs_list_for_plot.append(cost)
        iters_list_for_plot.append(iteration)
        print(f"[iter {iteration}]:{','.join([str(i) for i in new_cents])}")
        # plt.plot(iteration, cost)
        print(cost)
        if np.array_equal(old_cents, new_cents) or iteration == 19:
            plt.plot(iters_list_for_plot, costs_list_for_plot)
            plt.xlabel('iteration')
            plt.ylabel('average cost')
            plt.title("K=" + str(k))
            plt.show()
            break
        old_cents = new_cents.round(4)
        iteration += 1

def distance(p1, p2):
    return np.sum((p1 - p2)**2)
# initialization algorithm
def plus_plus(data, k):
    '''
    initialized the centroids for K-means++
    inputs:
        data - numpy array of data points having shape (200, 2)
        k - number of clusters
    '''
    ## initialize the centroids list and add
    ## a randomly selected data point to the list
    centroids = []
    centroids.append(data[np.random.randint(
        data.shape[0]), :])
    # print("THIS IS DATA SHAPE[0]:", data.shape)
    # plot(data, np.array(centroids))

    ## compute remaining k - 1 centroids
    for c_id in range(k - 1):

        ## initialize a list to store distances of data
        ## points from nearest centroid
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize

            ## compute distance of 'point' from each of the previously
            ## selected centroid and store the minimum distance
            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)

        ## select data point with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
        # plot(data, np.array(centroids))
    print(centroids)
    return centroids


def get_pixels():
    image_fname = sys.argv[1]
    orig_pixels = plt.imread(image_fname)
    pixels = orig_pixels.astype(float) / 255.
    # Reshape the image(128x128x3) into an Nx3 matrix where N = number of pixels.
    pixels = pixels.reshape(-1, 3)
    return pixels

if __name__ == '__main__':
    pixels = get_pixels()
    initial_centroids = plus_plus(pixels, 2)
    calculate_centroids_until_convergance(initial_centroids, pixels)

