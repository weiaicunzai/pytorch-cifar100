"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''
#import torch
#import torch.nn as nn
#
#
#cfg = {
#    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#}
#
#
#class VGG(nn.Module):
#    def __init__(self, vgg_name):
#        super(VGG, self).__init__()
#        self.features = self._make_layers(cfg[vgg_name])
#        self.classifier = nn.Linear(512, 100)
#
#    def forward(self, x):
#        out = self.features(x)
#        out = out.view(out.size(0), -1)
#        out = self.classifier(out)
#        return out
#
#    def _make_layers(self, cfg):
#        layers = []
#        in_channels = 3
#        for x in cfg:
#            if x == 'M':
#                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#            else:
#                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                           nn.BatchNorm2d(x),
#                           nn.ReLU(inplace=True)]
#                in_channels = x
#        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#        return nn.Sequential(*layers)
#
#def vgg16():
#    return VGG('VGG16')
#
#from torch.autograd import Variable
#def test():
#    net = VGG('VGG11')
#    x = torch.randn(2,3,32,32)
#    y = net(Variable(x))
#    print(y.size())
#
#net = vgg16()
#print(net)
#






import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        #3 fc layers
        #self.classifier = nn.Sequential(
        #    nn.Linear(512, 4096),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(),
        #    nn.Linear(4096, 4096),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(),
        #    nn.Linear(4096, num_class)
        #)

        #1 fc layer without dropout
        self.classifier = nn.Sequential(
            nn.Linear(512, num_class)
        )
    
    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
    
        return output



def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return nn.Sequential(*layers)


def vgg11():
    return VGG(make_layers(cfg['A']))

def vgg13():
    return VGG(make_layers(cfg['B']))

def vgg16():
    return VGG(make_layers(cfg['D']))

def vgg19():
    return VGG(make_layers(cfg['E']))

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))



