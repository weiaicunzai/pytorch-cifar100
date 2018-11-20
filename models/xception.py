"""xception in pytorch


[1] François Chollet

    Xception: Deep Learning with Depthwise Separable Convolutions
    https://arxiv.org/abs/1610.02357
"""

import torch
import torch.nn as nn

class SeperableConv2d(nn.Module):

    #***Figure 4. An “extreme” version of our Inception module, 
    #with one spatial convolution per output channel of the 1x1 
    #convolution."""
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()
        self.depthwise = nn.Conv2d(
            input_channels, 
            input_channels, 
            kernel_size, 
            groups=input_channels,
            **kwargs
        )

        self.pointwise = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x

class Block(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=3, **kwargs):

        super().__init__()
        self.separabel_conv1 = SeperableConv2d(
            input_channels, 
            output_channels, 
            kernel_size, 
            **kwargs
        )

        self.separabel_conv2 = SeperableConv2d(
            output_channels,
            output_channels,
            kernel_size,
            **kwargs
        )
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=2, ceil_mode=True)

        #"""Note that all Convolution and SeparableConvolution layers are 
        #followed by batch normalization"""
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=2)
    
    def forward(self, x):
        residual = self.separabel_conv1(x)
        residual = self.bn(residual)
        residual = self.relu(residual)
        residual = self.separabel_conv2(residual)
        residual = self.bn(residual)
        residual = self.maxpool(residual)

        shortcut = self.shortcut(x)
        shortcut = self.bn(shortcut)

        output = shortcut + residual
        output = self.relu(output)
        return output

class Xception(nn.Module):

    #"""Figure 5. The Xception architecture: the data first goes through the entry flow, 
    #then through the middle flow which is repeated eight times, and finally through 
    #the exit flow. """
    def __init__(self, class_nums=100):

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.entry_flow = nn.Sequential(
            Block(64, 128, padding=1),
            Block(128, 256, padding=1),
            Block(256, 728, padding=1)
        )

        self.middle_flow_residual = nn.Sequential(
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.middle_flow_shortcut = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

        self.exit_flow_residual = nn.Sequential(
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        self.exit_flow_shortcut = nn.Sequential(
            nn.Conv2d(728, 1024, 1, stride=2),
            nn.BatchNorm2d(1024)
        )

        self.tail = nn.Sequential(
            SeperableConv2d(1024, 1536, 3, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeperableConv2d(1536, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            #2x2
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.linear = nn.Linear(2048, class_nums)

    
    def forward(self, x):
        x = self.stem(x)

        #entry flow
        x = self.entry_flow(x)

        #middle flow
        x = self.middle_flow_residual(x) + self.middle_flow_shortcut(x)
        x = self.relu(x)

        #exit flow
        x = self.exit_flow_residual(x) + self.exit_flow_shortcut(x)

        #no relu applied here
        x = self.tail(x)
        x = x.view(-1, 2048)
        x = self.linear(x)

        return x

def xception():
    return Xception()

#image = torch.Tensor(1, 3, 32, 32)

#a = group_conv(image)
#print(a.shape)
#net = Block(3, 50, padding=1)
#net = Xception()
#print(net)
#print(net(image).shape)
#print(sum([p.numel() for p  in net.parameters()]))

#net = xception()
#print(net(image).shape)
#print(sum([p.numel() for p  in net.parameters()]))