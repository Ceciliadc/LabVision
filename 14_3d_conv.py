'''Your code will take an input tensor input with shape (n, iC, T, H, W), a kernel kernel with shape (oC, iC, kT, kH, kW) and a bias bias with shape (oC, ).

It needs then to apply a 3D convolution over input, using kernel as kernel tensor and bias as bias, using a stride of 1, no dilation, no grouping, and no padding, and store the result in out.

input, kernel and bias are torch.Tensor with dtype torch.float32.'''

import numpy as np
import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
T = random.randint(10, 20)
H = random.randint(10, 20)
W = random.randint(10, 20)
kT = random.randint(2, 6)
kH = random.randint(2, 6)
kW = random.randint(2, 6)


input = torch.rand(n, iC, T, H, W)
kernel = torch.rand(oC, iC, kT, kH, kW)
bias = torch.rand(oC)
#

#n, iC, T, H, W = input.shape
#oC, iC, kT, kH, kW = kernel.shape
oT = int(np.floor((T - kT + 1)))
oH = int(np.floor((H - kH + 1)))
oW = int(np.floor((W - kW + 1)))

out = torch.zeros(n, oC, oT, oH, oW)

print(input.size())
print(kernel.size())

for t in range(oT):
    for i in range(oH):
        for j in range(oW):
            #faccio unsqueeze per il broadcasting di input e kernel
            out[:, :, t, i, j] = torch.sum(torch.unsqueeze(input[:, :, t:t+kT, i:i+kH, j:j+kW], 1) * torch.unsqueeze(kernel, 0), (-1, -2, -3, -4)) + bias
            #faccio la somma sugli ultimi 4 assi (-1 oW, -2 oH, -3 oT, -4 oC)

print(out)
