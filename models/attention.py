"""residual attention network in pytorch



[1] Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang

    Residual Attention Network for Image Classification
    https://arxiv.org/abs/1704.06904
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#"""The Attention Module is built by pre-activation Residual Unit [11] with the 
#number of channels in each stage is the same as ResNet [10]."""

class ResidualUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride):

        self.residual_function = nn.Sequential(

            #1x1 conv
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1),

            #3x3 conv
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=stride, bias=False),

            #1x1 conv
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, 1)
        )

        self.shortcut = nn.Sequential()
        if stride != 2 or (in_channels != out_channels * 4):
            self.shortcut = nn.Conv2d(out_channels, out_channels * 4, 1, stride=stride)
    
    def forward(self, x):

        res = self.residual_function(x)
        x = self.shortcut(x)

        return res + x

class SoftBranch(nn.Module):

    def __init__(self, in_channels, out_channels, r):
        
        self.pre = self._make_risidual(in_channels, out_channels, r)
        self.mid = self._make_risidual(in_channels, out_channels, 2 * r)
        self.last = self._make_risidual(in_channels, out_channels, r)
        self.shortcut = ResidualUnit(in_channels, int(out_channels / 4), 1)
        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):

        shape1 = (x.size(2).item(), x.size(3).item())
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.pre(x)

        shape2 = (x.size(2).item(), x.size(3).item())
        x_mid = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_mid = self.mid(x_mid)
        x_mid = F.upsample_bilinear(x_mid, size=shape2)
        x_shortcut = self.shortcut(x)

        x = x_mid + x_shortcut

        x = self.last(x)
        x = F.upsample_bilinear(x, size=shape1)
        x = self.sigmoid(x)

        return x

    def _make_residual(self, in_channels, out_channels, n):
        """
        Args:
            in_channels: SoftBranch residual unit input channels
            out_channels: SoftBranch residual unit output channels
            n: number of residuals we need
        """
        layers = []
        for i in range(n):
            layers.append(ResidualUnit(in_channels, int(out_channels / 4), 1))

        return nn.Sequential(*layers)

class AttentionModule(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride, p=1, t=2, r=1):
        #"""The hyper-parameter p denotes the number of pre-processing Residual 
        #Units before splitting into trunk branch and mask branch. t denotes 
        #the number of Residual Units in trunk branch. r denotes the number of 
        #Residual Units between adjacent pooling layer in the mask branch."""

        self.pre = self._make_pre(in_channels, out_channels, p)
        self.trunk = self._make_trunk(out_channels * 4, out_channels, t)
        self.soft_mask = SoftBranch(out_channels * 4, out_channels, r)
    
    def forward(self, x):
        x = self.pre(x)
        x_t = self.trunk(x)
        x_s = self.soft_mask(x)

        return x_t + x_t * x_s
    
    def _make_trunk(self, in_channels, out_channels, t):

        layers = []

        for i in range(t):
            layers.append(ResidualUnit(in_channels, out_channels, 1))
        
        return nn.Sequential(*layers)

    def _make_pre(self, in_channels, out_channels, p):


        layers = []
        for i in range(p):
            layers.append(ResidualUnit(in_channels, out_channels, 1))
            in_channels = out_channels * 4

        return nn.Sequential(*layers)

class Attention(nn.Module):

    def __init__(self, block, block_num, class_num=100):
        
        self.in_channels = 64

        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(64, AttentionModule, block_num[0])
        self.stage2 = self._make_stage(128, AttentionModule, block_num[1])
        self.stage3 = self._make_stage(256, AttentionModule, block_num[2])
        self.stage4 = self._make_stage(512, AttentionModule, block_num[3])
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(2048, 100)
    
    def forward(self, x):
        x = self.pre_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x
        




    def _make_stage(self, out_channels, block, num):

        layers = []
        layers.append(ResidualUnit(self.in_channels, out_channels, 2))
        self.in_channels = out_channels * 4

        for i in range(num):
            layers.append(AttentionModule(self.in_channels, out_channels, 1))

        return nn.Sequential(*layers)