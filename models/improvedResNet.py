"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 缩小一半，以压缩特征图，与3*3卷积核设置步长为2作用相似。

        # residual function
        self.residual_function = nn.Sequential(
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>以下为改进的残差结构的卷积部分
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 把原有的3*3卷积层(填充为1）改为了1*1卷积层（填充为0），可以保持尺寸不变
            # 改成1*1，可以缩小参数量，但是这缩小了感受野，这可能导致准确率降低！
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        )

        # shortcut
        # 传统的残差网络是默认不进行任何处理（identity mapping，恒等映射)，shortcut connection就选择输入x本身
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )
        # 因为卷积核大小为1,所以设置步长为2会得到H或W向上取整的特征图；
        # 并且得到的特征图与灰缺少部分信息，这不合要求，所以采用添加池化层的方式。
        if stride == 2:
            self.residual_function.append(self.pool1)
            self.shortcut.append(self.pool1)

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class improvedResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 由4*4的
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    # 得到一个模块（包含num_blocks个残差块，其中只有第一个残差块是有最大池化的）
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            # 每个残差块，stride=2表示论文中的/2, 即缩小图片的H和W为原来的一半
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        # print("此时的形状为：", output.shape) # 特征图的H和W此时都为4,与理论一致
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def improvedResnet18():
    """ return a ResNet 18 object
    """
    return improvedResNet(BasicBlock, [2, 2, 2, 2])
