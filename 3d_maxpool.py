'''Your code will take as input:

a tensor input with shape (n, iC, T, H, W);
a kernel temporal span kT, height kH and width kW;
a stride s;
It needs then to apply a 3D max-pooling over input, using the given kernel size and the same stride s on all axes, and store the result in out. Input input has dtype torch.float32. s is an integer.'''

import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
T = random.randint(10, 20)
H = random.randint(10, 20)
W = random.randint(10, 20)
kT = random.randint(2, 5)
kH = random.randint(2, 5)
kW = random.randint(2, 5)
s = random.randint(2, 3)
input = torch.rand(n, iC, T, H, W)

#
oT = int((T - kT)/s+1)
oH = int((H - kH)/s+1)
oW = int((W - kW)/s+1)

out = torch.zeros((n, iC, oT, oH, oW))

for t in range(oT):
    for row in range(oH):
        for col in range(oW):
            out[:, :, t, row, col] = torch.max(
                        torch.max(
                            torch.max(input[:, :, s*t:s*t+kT, s*row:s*row+kH, s*col:s*col+kW], -3)[0], -2)[0], -1)[0]

print(out)