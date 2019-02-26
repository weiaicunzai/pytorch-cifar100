#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
#from dataset import *

#from skimage import io
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from conf import settings
from utils import get_network, get_test_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    args = parser.parse_args()

    net = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    net.load_state_dict(torch.load(args.weights), args.gpu)
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    for n_iter, (image, label) in enumerate(cifar100_test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
        image = Variable(image).cuda()
        label = Variable(label).cuda()
        output = net(image)
        _, pred = output.topk(5, 1, largest=True, sorted=True)

        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()

        #compute top 5
        correct_5 += correct[:, :5].sum()

        #compute top1 
        correct_1 += correct[:, :1].sum()


    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))