# train.py
#!/usr/bin/env	python3

""" train network using pytorch
"""

from torch.utils.data import DataLoader
import torch

from dataset import CIFAR100Train 
from models.resnet import *
import torch.optim as optim
from torch.autograd import Variable

from settings import *

cifar100_train = DataLoader(CIFAR100Train(g_cifar_100_path), shuffle=True, num_workers=2, batch_size=16)

net = ResNet101()

net = net.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)



if __name__ == '__main__':
    train_loss = 0.0
    correct = 0.0
    total = 0.0
    net.train()
    for batch_index, (labels, images) in enumerate(cifar100_train):
        #labels, images = label.to(device), image.to(device)
    
        images = Variable(images.permute(0, 3, 1, 2).float())
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()
        #optimizer.zero_grad()
        ##print(Variable(image.permute(0, )).size())

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.data
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().data[0]
        
        print(total)
        print(correct)
        print("loss: ", train_loss/(batch_index +1))
        print('correct: ', correct/total)
    
    torch.save(net.state_dict(), 'resnet.pt')