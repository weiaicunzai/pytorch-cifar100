# train.py
#!/usr/bin/env	python3

""" train network using pytorch
"""

#import argparse
import os

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from dataset import CIFAR100Train, CIFAR100Test
from models.resnet import *
from torch.autograd import Variable

from settings import *

#parser = argparse.ArgumentParser(description='image classification with Pytorch')
#parser.add_argument('--')

cifar100_training = CIFAR100Train(g_cifar_100_path)
cifar100_training_loader = DataLoader(cifar100_training, shuffle=True, num_workers=2, batch_size=16)

cifar100_test = CIFAR100Test(g_cifar_100_path)
cifar100_test_loader = DataLoader(cifar100_training, shuffle=True, num_workers=2, batch_size=16)

net = ResNet101().cuda()







loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)


def train(epoch):
    net.train()
    for batch_index, (labels, images) in enumerate(cifar100_training_loader):

        images = Variable(images.permute(0, 3, 1, 2).float())
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\t'.format(
            loss.data[0],
            epoch=epoch,
            trained_samples=batch_index * len(images),
            total_samples=len(cifar100_training)
        ))

def test(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for batch_index, (labels, images) in enumerate(cifar100_test_loader):
        images = Variable(images.permute(0, 3, 1, 2).float()).cuda()
        labels = Variable(labels).cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.data[0]
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().data[0]

    print(test_loss / len(cifar100_test))
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar100_test),
        correct / len(cifar100_test)
    ))
    print()

if __name__ == '__main__':

    checkpoint_path = os.path.join('checkpoint', 'resnet101-{epoch}.pt')
    for epoch in range(200):
        train(epoch)
        test(epoch)

        if not epoch % 50:
            torch.save(net.state_dict(), checkpoint.format(epoch=epoch))
        
 

    
