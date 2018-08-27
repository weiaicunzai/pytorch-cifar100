""" inceptionv3 in pytorch


[1] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna

    Rethinking the Inception Architecture for Computer Vision
    https://arxiv.org/abs/1512.00567v3
"""

import torch
import torch.nn as nn


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class InceptionA(nn.Module):

    def __init__(self, input_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(input_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3_1 = BasicConv2d(input_channels, 64, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branchpool_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branchpool_2 = BasicConv2d(96, 96, kernel_size=3, padding=1)
    
    def forward(self, x):
        
        #x -> 1x1
        branch1x1 = self.branch1x1(x)

        #x -> 1x1 -> 5x5
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        #x -> 1x1 -> 3x3 -> 3x3
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        #x -> pool -> 1x1
        branchpool = self.branchpool_1(x)
        branchpool = self.branchpool_2(branchpool)

        outputs = [branch1x1, branch5x5, branch3x3, branchpool]

        return torch.cat(outputs, 1)
