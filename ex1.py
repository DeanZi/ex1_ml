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
                for i in cents_to_its_pixels[key]:
                    pixels_in_group.append(pixels[i])
                new_cents[key] = np.average(pixels_in_group, axis=0)
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

# TODO tomorrow :
'''
1. Run results on university servers, see all good with out1.txt and out3.txt
2. Generate files with k=2,4,8,16 cents with an initialization function

    My centroids initialization process:
    
    I have used the K-mean++ initialization algorithm.
    It works the following way:
        1. Select the first centroid randomly from the data set
        2. While length of chosen centroids < k
            2.1 Compute for each point (pixel) its distance from the nearest centroid (from chosen centroids).
            2.2 Select the point with the maximum distance to be the next centroid.
        3. Return centroids
        
        
     
3. Implement Loss calculation in an iteration
4. Print plots of iteration vs loss

'''

# TODO later till submission :

'''
Finish theoretical part (ex1.pdf)
'''