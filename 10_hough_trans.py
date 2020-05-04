'''Your code will take as input a greyscale image im, (np.ndarray with dtype np.uint8 and shape (H, W)).

You then need to:

    Apply an edge detector.
    Apply the Hough transform for circles, using the corresponding OpenCV function.

The code is expected to show the final result using pyplot (e.g. calling the imshow function).'''

import numpy as np
import cv2
from skimage import data
import matplotlib.pyplot as plt

im = data.coins()[160:230, 70:270]

im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

canny = cv2.Canny(im, 200, 400, 3)

circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 50,
                     param1=100, param2=35, minRadius=4, maxRadius=0)

circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    cv2.circle(im, (i[0], i[1]), i[2], (0,255,0), 2) #outer circle

    cv2.circle(im, (i[0], i[1]), 2, (0, 0, 255), 3) #inner circle

plt.imshow(im)
plt.show()
