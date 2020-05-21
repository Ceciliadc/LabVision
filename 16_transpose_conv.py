'''Your code will take an input tensor input with shape (n, iC, H, W), a kernel kernel with shape (iC, oC, kH, kW) and a stride s.

It needs then to apply a 2D Transpose convolution over input, using kernel as kernel tensor, using a stride of s on both axes, no dilation, no grouping, and no padding, and store the result in out.

input and kernel are torch.Tensor with dtype torch.float32. s is an integer.'''

import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)
s = random.randint(2, 6)

input = torch.rand(n, iC, H, W)
kernel = torch.rand(iC, oC, kH, kW)

#

oH = int((H-1) * s + kH)
oW = int((W-1) * s + kW)

out = torch.zeros(n, oC, oH, oW)
kernel.transpose_(0, 1) #inverto i primi due assi del kernel iC e oC

for h in range(H):
    for w in range(W):
        this_input = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(input[:, :, h, w], 1), 3), 4)
        #broadcasting dell'input aggiungendo un asse in pos 1 e cambiando i valori di H e W mettendo un 1
        this_kernel = torch.unsqueeze(kernel, 0)
        #broadcasting del kernel aggiungendo un asse in pos 0
        out[:, :, s * h:s * h + kH, s * w:s * w + kW] += torch.sum(this_input * this_kernel, 2)
        #torch.sum sui canali e infatti metto axis= alla fine
