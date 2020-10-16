import torch
import torch.nn as nn



class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, t, p, r, sp, l):
        """
        Args:
            in_channels: input_channels
            p = dropout ratio
            sp = stochastic dropout ratio
            l = number of layer

        """
        self.p = p
        self.residual = nn.Sequential(

        )

swish = Swish()

img = torch.Tensor(4, 10, 20, 3)

print(swish(img).shape)
