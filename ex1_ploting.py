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
    return np.sum((p1 - p2) ** 2)


# initialization algorithm


def plus_plus_initialize(dataset, k):
    # Select the first centroid randomly from the data set
    centroids = [dataset[np.random.randint(
        len(dataset))]]

    while len(centroids) < k:

        # Will hold all pixels distances from their closest centroid
        distances_from_centroid = []
        for pixel_index in range(len(dataset)):
            pixel = dataset[pixel_index]
            min_dist = sys.maxsize
            # Compute for each pixel its distance from the nearest centroid (from chosen
            # centroids)
            for centroid_index in range(len(centroids)):
                pixel_distance_from_centroid = np.sum((pixel - centroids[centroid_index]) ** 2)
                min_dist = min(min_dist, pixel_distance_from_centroid)
            distances_from_centroid.append(min_dist)

        distances_from_centroid = np.array(distances_from_centroid)
        # Select the pixel with the maximum distance to be the next centroid
        next_centroid = dataset[np.argmax(distances_from_centroid)]
        centroids.append(next_centroid)
        distances_from_centroid
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
    initial_centroids = plus_plus_initialize(pixels, 8)
    calculate_centroids_until_convergance(initial_centroids, pixels)
