'''Your code will take as input a grayscale image im (a np.ndarray with dtype np.uint8 and shape (3, H, W)) and an integer nbin. 
It should compute a normalized color histogram of the image, quantized with nbin bins on each color plane.

The output should be a np.ndarray with shape (3*nbin, ), containing the concatenation of the histograms computed on each color plane 
(in the same order of the input tensor).
The output should be L1-normalized (i.e. all bins of the final histogram should sum up to 1).
Quantization strategy: a pixel should go in the bin with index b iif: pixel*n_bin // 256 == b'''

import random
import numpy as np
from skimage import data

im = data.astronaut()
im = np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)
nbin = random.randint(32,128)
#
color_hist = []
for c in range(3):
    histogram = np.zeros((nbin,))
    for row in range(im.shape[1]):
        for col in range(im.shape[2]):
            pixel = im[c, row, col]
            bin = pixel * nbin // 256
            histogram[bin] += 1

    color_hist = np.concatenate((color_hist, histogram))

out = color_hist/np.sum(color_hist)
