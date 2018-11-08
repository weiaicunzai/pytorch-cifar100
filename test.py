#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
from dataset import *

from skimage import io
from matplotlib import pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from conf import settings

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    args = parser.parse_args()

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn().cuda()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121().cuda()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161().cuda()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201().cuda()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet().cuda()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3().cuda()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4().cuda()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101().cuda()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception().cuda()
    else:
        print('the network name you have entered is not supported yet')


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(settings.CIFAR100_MEAN, settings.CIFAR100_STD),
    ])
    cifar100_test = CIFAR100Test(settings.CIFAR100_PATH, transform_test)
    cifar100_test_loader = DataLoader(cifar100_test, batch_size=16, shuffle=True, num_workers=2)

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    for n_iter, (label, image) in enumerate(cifar100_test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter, len(cifar100_test_loader)))
        image = Variable(image).cuda()
        label = Variable(label).cuda()
        output = net(image)
        _, pred = output.topk(5, 1, largest=True, sorted=True)

        label = label.view(16, -1).expand_as(pred)
        correct = pred.eq(label).float()

        #compute top 5
        correct_5 += correct[:, :5].sum()

        #compute top1 
        correct_1 += correct[:, :1].sum()


    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))