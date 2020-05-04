'''A residual block is defined as
y=σ(F(x)+G(x))
where x and y represent the input and output tensors of the block, σ is the ReLU activation function, F is the residual function to be learned and G is a projection shortcut used to match dimensions between F(x) and x.

Your code needs to define a ResidualBlock class (inherited from nn.Module) which implements a residual block. In your code, F
will be implemented with two convolutional layers with a ReLU non-linearity between them, i.e. F=conv2(σ(conv1(x))).
Batch normalization will also be adopted right after each convolution operation.

The constructor of the ResidualBlock class needs to take the following arguments as input:
- inplanes, the number of channels of x;
- planes, the number of output channels of conv1 and conv2;
- stride, the stride of conv1;

If the shapes of F(x)
and x do not match (either because inplanes != planes, or because stride > 1) ResidualBlock also needs to apply a projection shortcut G, composed of a convolutional layer with kernel size 1×1,
no bias, no padding and stride stride, followed by a batch normalization.

The forward method of ResidualBlock will take as input the input tensor x
and return the output tensor y, after performing all the operations of a Residual block.'''

import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.F = nn.Sequential(nn.Conv2d(inplanes, planes, 3, 1, 1, False),
                               nn.BatchNorm2d(planes),
                               nn.ReLU(),
                               nn.Conv2d(planes, planes, 3, 1, 1, False),
                               nn.BatchNorm2d(planes))

        self.G = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, 1, stride, 0, False),
                               nn.BatchNorm2d(planes))
        self.stride, self.inplanes, self.planes = stride, inplanes, planes

    def forward(self, x):
        gx = x
        x = self.F(x)

        if self.stride > 1 or self.inplanes != self.planes:
            gx = self.G(x)

        out = x + gx
        y = nn.ReLU(out)
        return y