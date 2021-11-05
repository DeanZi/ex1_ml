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
    # init old_cents to be as in file
    old_cents = centroids
    new_cents = np.empty_like(old_cents)
    k = len(old_cents)
    with open(out_fname, 'w') as outfile:
        iteration = 0
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
                sum_of_pixels_in_group = 0
                for i in cents_to_its_pixels[key]:
                    pixels_in_group.append(pixels[i])
                    sum_of_pixels_in_group += pixels[i]
                #new_cents[key] = np.average(pixels_in_group, axis=0)
                new_cents[key] = sum_of_pixels_in_group / len(pixels_in_group)
            new_cents = new_cents.round(4)
            line = f"[iter {iteration}]:{','.join([str(i) for i in new_cents])}"
            outfile.write(line + '\n')
            if np.array_equal(old_cents, new_cents) or iteration == 19:
                break
            old_cents = new_cents.round(4)
            iteration += 1




if __name__ == '__main__':
    initial_centroids, pixels, out_fname = read_normalize_reshape()
    calculate_centroids_until_convergance(initial_centroids, pixels, out_fname)



# TODO later till submission :

'''
Finish theoretical part (ex1.pdf)
Compare (maybe ask) the out.txt for cents5.txt (Try without np.average)
'''
