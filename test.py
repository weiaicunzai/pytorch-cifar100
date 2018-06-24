#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

from dataset import *

from skimage import io
from matplotlib import pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.resnet import *

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(g_cifar100_mean, g_cifar100_std),
])
cifar100_test = CIFAR100Test(g_cifar100_path, transform_test)
cifar100_test_loader = DataLoader(cifar100_test, batch_size=16, shuffle=True, num_workers=2)


net = ResNet101()


#====================================
#load the model you want to test here
#====================================
net.load_state_dict(torch.load('checkpoint/resnet101-150.pt'))
net.eval()

correct_1 = 0.0
correct_5 = 0.0
total = 0

for n_iter, (label, image) in enumerate(cifar100_test_loader):
    print("iteration: {}\ttotal {} iterations".format(n_iter, len(cifar100_test_loader)))
    image = Variable(image)
    label = Variable(label)
    output = net(image)
    _, pred = output.topk(5, 1, largest=True, sorted=True)

    label = label.view(16, -1).expand_as(pred)
    correct = pred.eq(label)

    #compute top 5
    correct_5 += correct[:, :5].sum().data[0]

    #compute top1 
    correct_1 += correct[:, :1].sum().data[0]


print()
print("Top 1 err: ", 1 - correct_1 / len(cifar100_test))
print("Top 5 err: ", 1 - correct_5 / len(cifar100_test))
print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))