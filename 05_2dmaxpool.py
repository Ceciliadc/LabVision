'''Your code will take as input:

    a tensor input with shape (n, iC, H, W);
    a kernel height kH and a kernel width kW;
    a stride s;

It needs then to apply a 2D max-pooling over input, using the given kernel size and stride, and store the result in out. 
Input input has dtype np.float32.'''

import random
import numpy as np

n = random.randint(2, 6)
iC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 5)
kW = random.randint(2, 5)
s = random.randint(1, 2)

input = np.random.rand(n, iC, H, W)
#
oH = int((H - kH)/s+1)
oW = int((W - kW)/s+1)

out = np.zeros((n, iC, oH, oW))

for row in range(oH):
    for col in range(oW):
        out[:, :, row, col] = np.max(input[:, :, s*row:s*row+kH, s*col:s*col+kW], axis = (-1, -2))
        #faccio il max sugli ultimi due assi (-1 ultimo asse col, -2 penultimo asse row)
