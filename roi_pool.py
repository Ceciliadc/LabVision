'''Implement a ROI Pooling operator. Your code will be given the following variables:

    input, a mini-batch of feature maps (a torch.Tensor with shape (n, C, H, W) and dtype torch.float32)
    boxes, a list of bounding box coordinates on which you need to perform the ROI Pooling. boxes will be a list of (L,4) torch.Tensor with dtype torch.float32, where boxes[i] will refer to the i-th element of the batch, and contain L coordinates in the format (y1, x1, y2, x2)
    a tuple of integers output_size, containing the number of cells over which pooling is performed, in the format (heigth, width)

The code should produce an output torch.Tensor out with dtype torch.float32 and shape (n, L, C, output_size[0], output_size[1]).'''

import random
import torch
import numpy as np

n = random.randint(1, 3)
C = random.randint(10, 20)
H = random.randint(5, 10)
W = random.randint(5, 10)
oH = random.randint(2, 4)
oW = random.randint(2, 4)
L = random.randint(2, 6)
input = torch.rand(n, C, H, W)
boxes = [torch.zeros(L, 4) for _ in range(n)]
for i in range(n):
  boxes[i][:, 0] = torch.rand(L) * (H-oH)       # y
  boxes[i][:, 1] = torch.rand(L) * (W-oW)       # x
  boxes[i][:, 2] = oH + torch.rand(L) * (H-oH)  # w
  boxes[i][:, 3] = oW + torch.rand(L) * (W-oW)  # h

  boxes[i][:,2:] += boxes[i][:,:2]
  boxes[i][:,2] = torch.clamp(boxes[i][:,2], max=H-1)
  boxes[i][:,3] = torch.clamp(boxes[i][:,3], max=W-1)
output_size = (oH, oW)

#
n, C, H, W = input.shape
L = len(boxes[0])
oH, oW = output_size
out = torch.zeros(n, L, C, oH, oW)

for k in range(n):
    boxes[k] = boxes[k].round()
    y1 = boxes[k][:,0]
    x1 = boxes[k][:,1]
    y2 = boxes[k][:,2]
    x2 = boxes[k][:,3]

    for l in range(L):
        for i in range(oH):
            y_start = np.floor((y1+i*(y2-y1+1)/oH))
            y_end = np.ceil((y1+(i+1)*(y2-y1+1)/oH))
            for j in range(oW):
                x_start = np.floor((x1+j*(x2-x1+1)/oW))
                x_end = np.ceil((x1+(j+1)*(x2-x1+1)/oW))

                y_s, y_e = int(y_start[l]), int(y_end[l])
                x_s, x_e = int(x_start[l]), int(x_end[l])

                out[k, l, :, i, j] = torch.max(
                    torch.max(input[k, :, y_s:y_e, x_s:x_e], 1)[0], 1)[0]

print(out)