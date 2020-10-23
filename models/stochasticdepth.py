"""
resnet with stochastic depth

[1] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Weinberger
    Deep Networks with Stochastic Depth

    https://arxiv.org/abs/1603.09382v3
"""
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
import random


class StochasticDepthBasicBlock(torch.jit.ScriptModule):

    expansion=1

    def __init__(self, p, in_channels, out_channels, stride=1):
        super().__init__()

        #self.p = torch.tensor(p).float()
        self.p = p
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * StochasticDepthBasicBlock.expansion, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * StochasticDepthBasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * StochasticDepthBasicBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    def survival(self):
        var = torch.bernoulli(torch.tensor(self.p).float())
        return torch.equal(var, torch.tensor(1).float().to(var.device))

    @torch.jit.script_method
    def forward(self, x):

        if self.training:
            if self.survival():
                # official torch implementation
                # function ResidualDrop:updateOutput(input)
                #    local skip_forward = self.skip:forward(input)
                #    self.output:resizeAs(skip_forward):copy(skip_forward)
                #    if self.train then
                #        if self.gate then -- only compute convolutional output when gate is open
                #            self.output:add(self.net:forward(input))
                #        end
                #    else
                #            self.output:add(self.net:forward(input):mul(1-self.deathRate))
                #        end
                #    return self.output
                # end

                # paper:
                # Hl = ReLU(bl*fl(Hl−1) + id(Hl−1)).

                # paper and their official implementation are different
                # paper use relu after output
                # official implementation dosen't
                #
                # other implementions which use relu:
                # https://github.com/jiweeo/pytorch-stochastic-depth/blob/a6f95aaffee82d273c1cd73d9ed6ef0718c6683d/models/resnet.py
                # https://github.com/dblN/stochastic_depth_keras/blob/master/train.py

                # implementations which doesn't use relu:
                # https://github.com/transcranial/stochastic-depth/blob/master/stochastic-depth.ipynb
                # https://github.com/shamangary/Pytorch-Stochastic-Depth-Resnet/blob/master/TYY_stodepth_lineardecay.py

                # I will just stick with the official implementation, I think
                # whether add relu after residual won't effect the network
                # performance too much
                x = self.residual(x) + self.shortcut(x)
            else:
                # If bl = 0, the ResBlock reduces to the identity function
                x = self.shortcut(x)

        else:
            x = self.residual(x) * self.p + self.shortcut(x)

        return x


class StochasticDepthBottleNeck(torch.jit.ScriptModule):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, p, in_channels, out_channels, stride=1):
        super().__init__()

        self.p = p
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * StochasticDepthBottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * StochasticDepthBottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * StochasticDepthBottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * StochasticDepthBottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * StochasticDepthBottleNeck.expansion)
            )

    def survival(self):
        var = torch.bernoulli(torch.tensor(self.p).float())
        return torch.equal(var, torch.tensor(1).float().to(var.device))

    @torch.jit.script_method
    def forward(self, x):

        if self.training:
            if self.survival():
                x = self.residual(x) + self.shortcut(x)
            else:
                x = self.shortcut(x)
        else:
            x = self.residual(x) * self.p + self.shortcut(x)

        return x

class StochasticDepthResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.step = (1 - 0.5) / (sum(num_block) - 1)
        self.pl = 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.pl, self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
            self.pl -= self.step

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def stochastic_depth_resnet18():
    """ return a ResNet 18 object
    """
    return StochasticDepthResNet(StochasticDepthBasicBlock, [2, 2, 2, 2])

def stochastic_depth_resnet34():
    """ return a ResNet 34 object
    """
    return StochasticDepthResNet(StochasticDepthBasicBlock, [3, 4, 6, 3])

def stochastic_depth_resnet50():

    """ return a ResNet 50 object
    """
    return StochasticDepthResNet(StochasticDepthBottleNeck, [3, 4, 6, 3])

def stochastic_depth_resnet101():
    """ return a ResNet 101 object
    """
    return StochasticDepthResNet(StochasticDepthBottleNeck, [3, 4, 23, 3])

def stochastic_depth_resnet152():
    """ return a ResNet 152 object
    """
    return StochasticDepthResNet(StochasticDepthBottleNeck, [3, 8, 36, 3])

