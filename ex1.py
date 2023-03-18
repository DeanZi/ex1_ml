import matplotlib.pyplot as plt
import numpy as np
import sys


def read_normalize_reshape():
    """
    A method that reads the command line arguments: image file, initial centroids file path and path to output file (for example: python3 k_means.py dog.jpeg cents1.txt out.txt).
    It returns the initial centroids, the normalized (and reshaped) pixels from the image file and the output file name.
    :return:
    """

    # read the command line arguments as specified
    image_file_name, centroids_file_name, output_file_name = sys.argv[1], sys.argv[2], sys.argv[3]

    # load initial centroids from file
    centroids = np.loadtxt(centroids_file_name)

    # load image and normalize pixel values
    orig_pixels = plt.imread(image_file_name)
    normalized_pixels = orig_pixels.astype(float) / 255.

    # reshape the image(128x128x3) into an Nx3 matrix where N = number of pixels
    reshaped_pixels = normalized_pixels.reshape(-1, 3)

    return centroids, reshaped_pixels, output_file_name


def find_closest_centroids(pixels, centroids):
    """
    Returns an array containing the index of the closest centroid to each pixel.
    """
    num_pixels = pixels.shape[0]
    num_centroids = centroids.shape[0]
    distances = np.zeros((num_pixels, num_centroids))
    for i in range(num_centroids):
        distances[:, i] = np.linalg.norm(pixels - centroids[i], axis=1)
    return np.argmin(distances, axis=1)


def calculate_new_centroids(groups, pixels, old_cents):
    """
    Calculates the new centroids based on the given pixel groups.
    """
    new_centroids = np.empty_like(old_cents)
    for key in groups.keys():
        pixels_in_group = pixels[groups[key]]
        if len(pixels_in_group) > 0:
            new_centroids[key] = np.mean(pixels_in_group, axis=0)
        else:
            new_centroids[key] = old_cents[key]
    return new_centroids.round(4)


def write_output_file(iteration, centroids, out_fname):
    """
    Writes the current iteration's centroids to the output file.
    """
    with open(out_fname, 'a') as outfile:
        line = f"[iter {iteration}]:{','.join([str(i) for i in centroids])}"
        outfile.write(line + '\n')


def calculate_centroids_until_convergence(centroids, pixels, out_fname):
    """
    Calculates the centroids until convergence or maximum number of iterations is reached.
    """
    old_cents = centroids
    k = len(old_cents)
    with open(out_fname, 'w'):
        iteration = 0
        while True:
            # Assign pixels to their closest centroid
            groups = {i: [] for i in range(k)}
            idx = find_closest_centroids(pixels, old_cents)
            for i in range(len(idx)):
                groups[idx[i]].append(i)

            # Calculate new centroids
            new_cents = calculate_new_centroids(groups, pixels, old_cents)

            # Write the output to a file
            write_output_file(iteration, new_cents, out_fname)

            # Check for convergence
            if np.array_equal(old_cents, new_cents) or iteration == 19:
                break
            old_cents = new_cents.round(4)
            iteration += 1


if __name__ == '__main__':
    initial_centroids, pixels, out_file_name = read_normalize_reshape()
    calculate_centroids_until_convergence(initial_centroids, pixels, out_file_name)