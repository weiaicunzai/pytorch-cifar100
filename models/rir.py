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
        self.residual_stream_conv_across = nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=1)

        #"""We use equal numbers of filters for the residual and transient streams of the 
        #generalized residual network, but optimizing this hyperparameter could lead to 
        #further potential improvements."""
        self.transient_stream_conv_across = nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=1)

        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        #"""The form of the shortcut connection can be an identity function with
        #the appropriate padding or a projection as in He et al. (2015b)."""
        self.short_cut = nn.Sequential()
        if in_channel != out_channel:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=1, stride=stride)
            )

    def forward(self, x):
        x_residual, x_transient = x
        residual_r_r = self.residual_stream_conv(x_residual)
        residual_r_t = self.residual_stream_conv_across(x_residual)
        residual_shortcut = self.short_cut(x_residual)

        transient_t_t = self.transient_stream_conv(x_transient)
        transient_t_r = self.transient_stream_conv_across(x_transient)

        #"""Same-stream and cross-stream activations are summed (along with the 
        #shortcut connection for the residual stream) before applying batch 
        #normalization and ReLU nonlinearities (together σ) to get the output 
        #states of the block (Equation 1) (Ioffe & Szegedy, 2015)."""
        x_residual = self.bn_relu(residual_r_r + residual_r_t + residual_shortcut)
        x_transient = self.bn_relu(transient_t_t + transient_t_r)

        return x_residual, x_transient
    
class RiRBlock(nn.Module):
    def __init__(self, in_channel, out_channel, layer_num, stride, layer=ResnetInit):
        self.short_cut = nn.Sequential()
        self.resnetinit = self._make_layers(in_channel, out_channel, layer_num, stride)
        if stride != 1:
            self.short_cut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride) 

    def forward(self, x):
        x_residual, x_transient = self.resnetinit(x)
        x_residual = x_residual + self.short_cut(x[0])
        x_transient = x_transient + self.short_cut(x[1])

        return (x_residual, x_transient)

    #"""Replacing each of the convolutional layers within a residual
    #block from the original ResNet (Figure 1a) with a generalized residual block 
    #(Figure 1b) leads us to a new architecture we call ResNet in ResNet (RiR) 
    #(Figure 1d)."""
    def _make_layers(self, in_channel, out_channel, layer_num, stride, layer=ResnetInit):
        strides = stride + [1] * (layer_num - 1)
        layers = nn.Sequential()
        for s in strides:
            layers.add_module(layer(in_channel, out_channel, s))

        return layers

class ResnetInResneet(nn.Module):
    def __init__(self, block_nums, layer_num, stride=1, block=RiRBlock, num_classes=100):

        self.base = 96 / 2
        self.residual_pre_conv = nn.Conv2d(3, self.base, kernel_size=3)
        self.transient_pre_conv = nn.Conv2d(3, self.base, kernel_size=3)

        self.blocks = self._make_blocks(block_nums, layer_num)
        self.classifier = nn.Conv2d(self.base, num_classes, 1)
    
    def forward(self, x):
        x_residual = self.residual_pre_conv(x)
        x_transient = self.transient_pre_conv(x)
        x_residual, x_transient = self.blocks((x_residual, x_transient))
        h = torch.cat([x_residual, x_transient], 1)
        h = self.classifier(h)
        h = h.view(h.size()[0], -1)
        
        return h
        


    def _make_blocks(self, block_nums, layer_num, stride=2, block=RiRBlock):
        
        layers = nn.Sequential()
        for block in range(block_nums):
            self.base *= 2
            layers.add_module(RiRBlock(self.base, self.base, layer_num, stride))
        
        return layers
            


def resnet_in_resnet():
    return ResnetInResneet(9, 2)