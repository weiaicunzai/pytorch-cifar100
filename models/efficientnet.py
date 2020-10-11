import torch
import torch.nn as nn



class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class MBConvBlock(nn.Module):
    def __init__(self, p, in_channels, out_channels, stride):
        self.p = p
        self.residual = nn.Sequential(

        )

swish = Swish()

img = torch.Tensor(4, 10, 20, 3)

print(swish(img).shape)
