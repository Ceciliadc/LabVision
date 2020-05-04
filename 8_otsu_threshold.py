'''Given an input grayscale image im (a np.ndarray with shape (H, W) and dtype np.uint8), write a code which computes the Otsu threshold 
for im stores the result in out.

Notice: beware of how the threshold is defined in the Otsu formulas. Your output should be compliant with our first definition of threshold.'''

import numpy as np
from skimage import data
from skimage.transform import resize

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(np.uint8)
im = np.swapaxes(im, 0, 1)
#
hist, bins = np.histogram(im, np.arange(0, 257))
thr = -1
val = -1
arr = np.arange(256)

for t in range(len(bins)):
    w1 = np.sum(hist[:t+1])
    w2 = np.sum(hist[t+1:])

    mu1 = np.sum(arr[:t+1] * hist[:t+1])
    mu2 = np.sum(arr[t+1:] * hist[t+1:])

    if w1 == 0 or w2 == 0:
        continue

    mu1 /= w1
    mu2 /= w2

    sigma = (w1 * w2) * ((mu1 - mu2)**2)

    if sigma > val:
        val = sigma
        thr = int(t)

print(thr)
