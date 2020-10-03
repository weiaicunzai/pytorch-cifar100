import torch
import torch.nn as nn



class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class MBConvBlock(nn.Module):

swish = Swish()

img = torch.Tensor(4, 10, 20, 3)

print(swish(img).shape)
