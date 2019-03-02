
import argparse
import glob
import os

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

#from PIL import Image
#import transforms 
from torchvision import transforms
#from tensorboardX import SummaryWriter
from conf import settings
from utils import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from torch.optim.lr_scheduler import _LRScheduler


class FindLR(_LRScheduler):
    """exponentially increasing learning rate

    Args:
        optimizer: optimzier(e.g. SGD)
        num_iter: totoal_iters 
        max_lr: maximum  learning rate
    """
    def __init__(self, optimizer, max_lr=10, num_iter=100, last_epoch=-1):
        
        self.total_iters = num_iter
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        return [base_lr * (self.max_lr / base_lr) ** (self.last_epoch / (self.total_iters + 1e-32)) for base_lr in self.base_lrs]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-base_lr', type=float, default=1e-7, help='min learning rate')
    parser.add_argument('-max_lr', type=float, default=10, help='max learning rate')
    parser.add_argument('-num_iter', type=int, default=100, help='num of iteration')
    parser.add_argument('-gpus', nargs='+', type=int, default=0, help='gpu device')
    args = parser.parse_args()

    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
    )
    
    net = get_network(args)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    #set up warmup phase learning rate scheduler
    lr_scheduler = FindLR(optimizer, max_lr=args.max_lr, num_iter=args.num_iter)
    epoches = int(args.num_iter / len(cifar100_training_loader)) + 1

    n = 0

    learning_rate = []
    losses = []
    for epoch in range(epoches):

        #training procedure
        net.train()
        
        for batch_index, (images, labels) in enumerate(cifar100_training_loader):
            if n > args.num_iter:
                break

            lr_scheduler.step()

            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            predicts = net(images)
            loss = loss_function(predicts, labels)
            if torch.isnan(loss).any():
                n += 1e8
                break
            loss.backward()
            optimizer.step()

            print('Iterations: {iter_num} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.8f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                iter_num=n,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(cifar100_training_loader.dataset),
            ))

            learning_rate.append(optimizer.param_groups[0]['lr'])
            losses.append(loss.item())
            n += 1

    learning_rate = learning_rate[10:-5]
    losses = losses[10:-5]

    fig, ax = plt.subplots(1,1)
    ax.plot(learning_rate, losses)
    ax.set_xlabel('learning rate')
    ax.set_ylabel('losses')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))

    fig.savefig('result.jpg')
