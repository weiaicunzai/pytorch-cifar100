"""resnet in resnet in pytorch



[1] Sasha Targ, Diogo Almeida, Kevin Lyman.

    Resnet in Resnet: Generalizing Residual Architectures
    https://arxiv.org/abs/1603.08029v1
"""

import torch
import torch.nn as nn

#geralized  
class ResnetInit(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()

        #"""The modular unit of the generalized residual network architecture is a 
        #generalized residual block consisting of parallel states for a residual stream, 
        #r, which contains identity shortcut connections and is similar to the structure 
        #of a residual block from the original ResNet with a single convolutional layer 
        #(parameters W l,r→r )
        self.residual_stream_conv = nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=stride)

        #"""and a transient stream, t, which is a standard convolutional layer
        #(W l,t→t )."""
        self.transient_stream_conv = nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=stride)

        #"""Two additional sets of convolutional filters in each block (W l,r→t , W l,t→r )
        #also transfer information across streams."""
        self.residual_stream_conv_across = nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=stride)

        #"""We use equal numbers of filters for the residual and transient streams of the 
        #generalized residual network, but optimizing this hyperparameter could lead to 
        #further potential improvements."""
        self.transient_stream_conv_across = nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=stride)

        self.residual_bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.transient_bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        #"""The form of the shortcut connection can be an identity function with
        #the appropriate padding or a projection as in He et al. (2015b)."""
        self.short_cut = nn.Sequential()
        if in_channel != out_channel or stride != 1:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)
            )
        

    def forward(self, x):
        x_residual, x_transient = x
        residual_r_r = self.residual_stream_conv(x_residual)
        residual_r_t = self.residual_stream_conv_across(x_residual)
        residual_shortcut = self.short_cut(x_residual)

        transient_t_t = self.transient_stream_conv(x_transient)
        transient_t_r = self.transient_stream_conv_across(x_transient)

        #transient_t_t = self.transient_stream_conv(x_residual)
        #transient_t_r = self.transient_stream_conv_across(x_residual)
        #"""Same-stream and cross-stream activations are summed (along with the 
        #shortcut connection for the residual stream) before applying batch 
        #normalization and ReLU nonlinearities (together σ) to get the output 
        #states of the block (Equation 1) (Ioffe & Szegedy, 2015)."""
        x_residual = self.residual_bn_relu(residual_r_r + transient_t_r + residual_shortcut)
        x_transient = self.transient_bn_relu(residual_r_t + transient_t_t)

        return x_residual, x_transient
    


class RiRBlock(nn.Module):
    def __init__(self, in_channel, out_channel, layer_num, stride, layer=ResnetInit):
        super().__init__()
        self.resnetinit = self._make_layers(in_channel, out_channel, layer_num, stride)

        #self.short_cut = nn.Sequential()
        #if stride != 1 or in_channel != out_channel:
        #    self.short_cut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride) 

    def forward(self, x):
        x_residual, x_transient = self.resnetinit(x)
        #x_residual = x_residual + self.short_cut(x[0])
        #x_transient = x_transient + self.short_cut(x[1])

        return (x_residual, x_transient)

    #"""Replacing each of the convolutional layers within a residual
    #block from the original ResNet (Figure 1a) with a generalized residual block 
    #(Figure 1b) leads us to a new architecture we call ResNet in ResNet (RiR) 
    #(Figure 1d)."""
    def _make_layers(self, in_channel, out_channel, layer_num, stride, layer=ResnetInit):
        strides = [stride] + [1] * (layer_num - 1)
        layers = nn.Sequential()
        for index, s in enumerate(strides):
            layers.add_module("generalized layers{}".format(index), layer(in_channel, out_channel, s))
            in_channel = out_channel

        return layers

class ResnetInResneet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        base = int(96 / 2)
        self.residual_pre_conv = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True)
        )
        self.transient_pre_conv = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True)
        )

        self.rir1 = RiRBlock(base, base, 2, 1)
        self.rir2 = RiRBlock(base, base, 2, 1)
        self.rir3 = RiRBlock(base, base * 2, 2, 2)
        self.rir4 = RiRBlock(base * 2, base * 2, 2, 1)
        self.rir5 = RiRBlock(base * 2, base * 2, 2, 1)
        self.rir6 = RiRBlock(base * 2, base * 4, 2, 2)
        self.rir7 = RiRBlock(base * 4, base * 4, 2, 1)
        self.rir8 = RiRBlock(base * 4, base * 4, 2, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(384, num_classes, kernel_size=3, stride=2), #without this convolution, loss will soon be nan
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(900, 450),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(450, 100),
        )

        self._weight_init()
    
    def forward(self, x):
        x_residual = self.residual_pre_conv(x)
        x_transient = self.transient_pre_conv(x)

        x_residual, x_transient = self.rir1((x_residual, x_transient))
        x_residual, x_transient = self.rir2((x_residual, x_transient))
        x_residual, x_transient = self.rir3((x_residual, x_transient))
        x_residual, x_transient = self.rir4((x_residual, x_transient))
        x_residual, x_transient = self.rir5((x_residual, x_transient))
        x_residual, x_transient = self.rir6((x_residual, x_transient))
        x_residual, x_transient = self.rir7((x_residual, x_transient))
        x_residual, x_transient = self.rir8((x_residual, x_transient))
        h = torch.cat([x_residual, x_transient], 1)
        h = self.conv1(h)
        h = h.view(h.size()[0], -1)
        h = self.classifier(h)

        return h

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight)
                m.bias.data.fill_(0.01)    
    

def resnet_in_resnet():
    return ResnetInResneet()

#from torch.autograd import Variable
#
#net = resnet_in_resnet()
#print(net(Variable(torch.randn(3, 3, 32, 32))).shape)
