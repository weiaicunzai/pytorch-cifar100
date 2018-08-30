""" inceptionv4 in pytorch


[1] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi

    Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    https://arxiv.org/abs/1602.07261
"""

import torch
import torch.nn as nn

class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Inception_Stem(nn.Module):

    #"""Figure 3. The schema for stem of the pure Inception-v4 and 
    #Inception-ResNet-v2 networks. This is the input part of those 
    #networks."""
    def __init__(self, input_channels, class_nums=100):
        super().__init__()
        self.conv1 = nn.Sequential(
            BasicConv2d(input_channels, 32, kernel_size=3),
            BasicConv2d(32, 32, kernel_size=3, padding=1),
            BasicConv2d(32, 64, kernel_size=3, padding=1)
        )

        self.branch3x3_conv = BasicConv2d(64, 96, kernel_size=3)
        self.branch3x3_pool = nn.MaxPool2d(3, stride=1)

        self.branch7x7a = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(64, 96, kernel_size=3)
        )

        self.branch7x7b = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3)
        )

        self.branchpoola = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branchpoolb = BasicConv2d(192, 192, kernel_size=3, stride=2, padding=1)

    def forward(self, x):

        x = self.conv1(x)

        x = [
            self.branch3x3_conv(x),
            self.branch3x3_pool(x)
        ]
        x = torch.cat(x, 1)

        x = [
            self.branch7x7a(x),
            self.branch7x7b(x)
        ]
        x = torch.cat(x, 1)

        x = [
            self.branchpoola(x),
            self.branchpoolb(x)
        ]
        x = torch.cat(x, 1)

        return x

class InceptionA(nn.Module):

    #"""Figure 4. The schema for 35 × 35 grid modules of the pure 
    #Inception-v4 network. This is the Inception-A block of Figure 9."""
    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1)
        )

        self.branch1x1 = BasicConv2d(input_channels, 96, kernel_size=1)

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 96, kernel_size=1)
        )

    def forward(self, x):

        x = [
            self.branch3x3stack(x),
            self.branch3x3(x),
            self.branch1x1(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)

class ReductionA(nn.Module):

    #"""Figure 7. The schema for 35 × 35 to 17 × 17 reduction module. 
    #Different variants of this blocks (with various number of filters) 
    #are used in Figure 9, and 15 in each of the new Inception(-v4, - ResNet-v1,
    #-ResNet-v2) variants presented in this paper. The k, l, m, n numbers 
    #represent filter bank sizes which can be looked up in Table 1.
    def __init__(self, input_channels, k, l, m, n):

        super().__init__()
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, k, kernel_size=1),
            BasicConv2d(k, l, kernel_size=3, padding=1),
            BasicConv2d(l, m, kernel_size=3, stride=2)
        )

        self.branch3x3 = BasicConv2d(input_channels, n, kernel_size=3, stride=2)
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):

        x = [
            self.branch3x3stack(x),
            self.branch3x3(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)

class 
#net = Inception_Stem(3)
#net = ReductionA(3, 192, 224, 256, 384)
net = ReductionA(3, 192, 224, 256, 384)

print(net(torch.Tensor(3, 3, 32, 32)).shape)





        
