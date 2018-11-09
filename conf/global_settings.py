import os
from datetime import datetime

#g_cifar100_path = '/home/baiyu/Downloads/cifar-100-python'
g_cifar100_path = '/nfs/private/cifar100/cifar-100-python'

g_cifar100_mean = (0.5071, 0.4867, 0.4408)
g_cifar100_std = (0.2675, 0.2565, 0.2761)

#CIFAR100 dataset path (python version)
CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 200 
MILESTONES = [80, 140]

#initial learning rate
INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().isoformat()








