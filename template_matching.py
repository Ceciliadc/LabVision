'''Your code will take as input a mini-batch of feature maps input (a np.ndarray tensor with dtype np.float32 and shape (n, H, W)), and a template template (a np.ndarray with dtype np.float32 and shape (kH, kW)). It then needs to compare the template against all samples in the mini-batch in a sliding window fashion, and store the result in out.

out will have shape (n, oH, oW), where oH=iH-(kH-1) and oW=iW-(kW-1), and out[i, :, :] will contain the similarity between the template and the i-th feature map at all valid locations. Use the sum of squared differences as comparison function.'''

import random
import numpy as np

n = random.randint(1, 3)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)
input = np.random.rand(n, H, W).astype(np.float32)
template = np.random.rand(kH, kW).astype(np.float32)
#
#n, H, W = input.shape
#kH, kW = template.shape

oH = H-(kH-1)
oW = W-(kW-1)
out = np.zeros((n, oH, oW))

for row in range(oH):
    for col in range(oW):
        this_template = np.expand_dims(template, 0)
        out[:, row, col] = np.sum((this_template - input[:, row:row+kH, col:col+kW])**2, axis = (-1, -2))
        #devo fare la somma solo sugli ultimi due indici, il numero di canali non li tocco, quindi faccio axis = (-1 -2)
        #per prendere il penultimo e ultimo parametro
print(out)