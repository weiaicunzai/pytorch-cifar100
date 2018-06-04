import os
import sys
import pickle

from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset

from settings import *



class CIFAR100Train(Dataset):
    def __init__(self, path):
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')

    def __len__(self):
        return len(self.data['fine_labels'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))
        return label, image



print(sys.getdefaultencoding())
cifar100 = CIFAR100Train(g_cifar_100_path)
label, image = cifar100[24565]

io.imshow(image)
plt.show()
print(label)