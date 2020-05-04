'''Your code will take as input two color images im_a and im_b (np.ndarray with dtype np.uint8 and shape (3, H, W)), depicting the same 
scene from two different perspectives.

You then need to:
    Manually identify (at least) four corresponding pairs of points
    Estimate the homography between the first and the second image using the detected point pairs.
    Warp the second image using the estimated transformation matrix.
    "Merge" the two images in a single one by sticking one on top of the other.

The code is expected to show the final result using pyplot (e.g. calling the imshow function). When doing this, pay attention to the axis 
order (their format is (H, W, 3)).

If you employ OpenCV functions, recall that the OpenCV format is also (H, W, 3).'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

with open("gallery_0.jpg", "rb") as gallery_0:
    bytes = np.asarray(bytearray(gallery_0.read()), dtype=np.uint8)
im_a = cv2.imdecode(bytes, cv2.IMREAD_COLOR)
im_a = np.swapaxes(np.swapaxes(im_a, 0, 2), 1, 2)
im_a = im_a[::-1, :, :]  # from BGR to RGB

with open("gallery_1.jpg", "rb") as gallery_1:
    bytes = np.asarray(bytearray(gallery_1.read()), dtype=np.uint8)
im_b = cv2.imdecode(bytes, cv2.IMREAD_COLOR)
im_b = np.swapaxes(np.swapaxes(im_b, 0, 2), 1, 2)
im_b = im_b[::-1, :, :]  # from BGR to RGB

#
im_a = im_a.transpose(1, 2, 0)
im_b = im_b.transpose(1, 2, 0)
im_a = im_a[:-1, :, :]

M1 = np.array([[144, 55], [137, 190], [94,165], [331,194], [336, 62]], dtype="float32")
M2 = np.array([[195, 40], [183, 232], [111,210], [308,207], [314, 96]], dtype="float32")

ret, _ = cv2.findHomography(M1, M2, cv2.RANSAC)

out = cv2.warpPerspective(im_b, ret, (im_b.shape[1], im_b.shape[0]))

mask = np.all(out == [0, 0, 0], axis=-1)

out[mask] = im_a[mask]

plt.imshow(out)
plt.show()

