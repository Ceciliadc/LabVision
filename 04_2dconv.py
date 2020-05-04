'''Your code will take an input tensor input with shape (n, iC, H, W) and a kernel kernel with shape (oC, iC, kH, kW). 
It needs then to apply a 2D convolution over input, using kernel as kernel tensor and no bias, using a stride of 1, no dilation, 
no grouping, and no padding, and store the result in out. Both input and kernel have dtype np.float32.'''

import random
import numpy as np
n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)

input = np.random.rand(n, iC, H, W).astype(np.float32)
kernel = np.random.rand(oC, iC, kH, kW).astype(np.float32)

#n, iC, H, W = input.shape
#oC, iC, kH, kW = kernel.shape
oH = H-(kH-1)
oW = W-(kW-1)

out = np.zeros((n, oC, oH, oW))

for num in range(n):
    for k in range(oC):
        for x in range(oH):
            for y in range(oW):
                out[num, k, x, y] = np.sum(input[num, :, x:x+kH, y:y+kW]*kernel[k, :, :, :], axis = (-1, -2))


''' modo alternativo con solo due loop
for row in range(oH):
    for col in range(oW):
       this_input = np.expand_dims(input[:, :, row:row+kH, col:col+kW], 1)
        #(n, 1, iC, kH, kW) broadcast
        this_kernel = np.expand_dims(kernel, 0)
        #(1, oC, iC, kH, kW) broadcast

        #now input and kernel are compatible
        out[:, :, row, col] = np.sum(this_input * this_kernel)'''
