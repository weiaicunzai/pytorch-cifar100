import random

import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

# NAS or manually designed?
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, t, r, sp):
        """
        Args:
            in_channels: input_channels
            sp = stochastic dropout ratio
            t = expansion factor
            r = se ratio (less than 1)
        """
        super().__init__()

        self.sp = sp
        # expansion phase (inverted residual block)
        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            Swish()
        )

        # depthwise phase
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels * t, in_channels * t, kernel_size, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            Swish()
        )

        # equeeze and excitation
        squeeze_channels = max(int(in_channels * t * r), 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels * t, squeeze_channels),
            Swish(),
            nn.Linear(squeeze_channels, in_channels * t),
            nn.Sigmoid()
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)

        # expansion
        expansion = self.expansion(x)

        # depthwise
        depthwise = self.depthwise(expansion)

        # squeeze and excitation
        squeezed = self.squeeze(depthwise)
        squeezed = squeezed.view(squeezed.size(0), -1)
        excitation = self.excitation(squeezed)
        excitation = excitation.view(depthwise.size(0), depthwise.size(1), 1, 1)
        #print(excitation.shape)
        depthwise = depthwise * excitation

        #print(depthwise.shape)
        # pointwise
        pointwise = self.pointwise(depthwise)

        # stochastic depth
        if self.train:
            if random.random() < self.sp:
                x = shortcut + pointwise

            else:
                x = pointwise

        else:
            x = pointwise * self.sp + shortcut

        return x


class EfficientNet(nn.Module):
    def __init__(self, num_classes=100):
        self.num_classes = num_classes


net = MBConvBlock(3, 10, 5, 2, 6, 0.25, 0.5)

#swish = Swish()

img = torch.Tensor(4, 3, 20, 3)

print(net(img).shape)
