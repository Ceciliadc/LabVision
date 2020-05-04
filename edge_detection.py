'''Your code will take as input a grayscale image im (a np.ndarray with dtype np.uint8 and shape (H,W)). It needs then to:

    Apply the horizontal and vertical Sobel masks (with a kernel size of 3) to obtain horizontal and vertical derivatives;
    Compute the magnitude and direction of the gradient, and normalize them properly (see slides);
    Diplay the gradient magnitude and derivative jointly in an HSV image, and then convert it in RGB format (see slides).

The code is expected to show the final result using pyplot (e.g. calling the imshow function). When doing this, pay attention to the axis order.'''

from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import data

im = data.coins()
im = np.swapaxes(im, 0, 1)
#
mask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float)
mask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float)

Gx = convolve2d(im, mask_x)
Gy = convolve2d(im, mask_y)

m = np.sqrt(Gx**2 + Gy**2)
V = (m / 1081) * 255

theta = np.arctan2(Gy, Gx)
H = (theta + np.pi)/(2*np.pi)
H *= 180

S = np.full(H.shape, 255)

HSV_img = np.transpose([H, S, V], [1,2,0]).astype(np.uint8)
rgb = cv2.cvtColor(HSV_img, cv2.COLOR_HSV2RGB)
print(rgb.shape)

plt.imshow(rgb)
plt.show()