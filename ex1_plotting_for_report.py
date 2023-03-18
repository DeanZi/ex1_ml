import matplotlib.pyplot as plt
import numpy as np
import sys


def calculate_centroids_until_convergence(centroids, pixels, max_iterations=20):
    old_cents = centroids
    k = len(old_cents)
    iters_list_for_plot = []
    costs_list_for_plot = []

    for iteration in range(max_iterations):
        cents_to_its_pixels = map_pixels_to_centroids(old_cents, pixels)
        new_cents = update_centroids(cents_to_its_pixels, pixels, old_cents)
        cost = calculate_cost(cents_to_its_pixels, pixels, old_cents)
        costs_list_for_plot.append(cost)
        iters_list_for_plot.append(iteration)

        print(f"[iter {iteration}]:{','.join([str(i) for i in new_cents])}")
        print(cost)

        if np.array_equal(old_cents, new_cents) or iteration == max_iterations - 1:
            plot_convergence(iters_list_for_plot, costs_list_for_plot, k)
            break

        old_cents = new_cents.round(4)


def map_pixels_to_centroids(centroids, pixels):
    cents_to_its_pixels = {}
    for i, centroid in enumerate(centroids):
        cents_to_its_pixels[i] = []
    for pixel_index, pixel in enumerate(pixels):
        closest_centroid_index = get_closest_centroid_index(centroids, pixel)
        cents_to_its_pixels[closest_centroid_index].append(pixel_index)
    return cents_to_its_pixels


def get_closest_centroid_index(centroids, pixel):
    min_dist = sys.maxsize
    index_of_min = -1
    for i, centroid in enumerate(centroids):
        pixel_dist_from_cent = np.linalg.norm(pixel - centroid)
        if min_dist > pixel_dist_from_cent:
            min_dist = pixel_dist_from_cent
            index_of_min = i
    return index_of_min


def update_centroids(cents_to_its_pixels, pixels, old_cents):
    new_cents = np.empty_like(old_cents)
    for key in cents_to_its_pixels.keys():
        pixels_in_group = get_pixels_for_cent(cents_to_its_pixels[key], pixels)
        if len(pixels_in_group) > 0:
            new_cents[key] = np.mean(pixels_in_group, axis=0)
        else:
            new_cents[key] = old_cents[key]
    return new_cents.round(4)


def get_pixels_for_cent(pixel_indices, pixels):
    return [pixels[i] for i in pixel_indices]


def calculate_cost(cents_to_its_pixels, pixels, old_cents):
    total_distance_of_pixels = 0
    for key in cents_to_its_pixels.keys():
        pixels_in_group = get_pixels_for_cent(cents_to_its_pixels[key], pixels)
        for pixel in pixels_in_group:
            total_distance_of_pixels += np.linalg.norm(pixel - old_cents[key])
    return total_distance_of_pixels / len(pixels)


def plot_convergence(iters_list_for_plot, costs_list_for_plot, k):
    plt.plot(iters_list_for_plot, costs_list_for_plot)
    plt.xlabel('iteration')
    plt.ylabel('average cost')
    plt.title("K=" + str(k))
    plt.show()


def distance(p1, p2):
    return np.sum((p1 - p2) ** 2)


def plus_plus_initialize(dataset, k):
    """
    initialization algorithm
    :param dataset:
    :param k:
    :return:
    """

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
    calculate_centroids_until_convergence(initial_centroids, pixels)
