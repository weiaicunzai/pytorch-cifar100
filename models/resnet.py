# resnet.py
#!/usr/bin/env	python3
"""resnet in pytorch




[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun 
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class BasicBlock(nn.Module):
    """Regualer residual block

    """

    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):


        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.shortcut = nn.Sequential()
#        if stride != 1 or 
    

class test:
    aa = 10

    def hello():
        nonlocal aa
        print(aa)


#a = nn.Sequential()
#print(a(Variable(torch.FloatTensor([[1, 1], [1, 1]]))))
