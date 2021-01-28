""" helper function

author baiyu
"""

import sys
import importlib

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings

def load_net(args):
    models = {
        'resnet18' : 'resnet',
        'resnet34' : 'resnet',
        'resnet50' : 'resnet',
        'resnet101' : 'resnet',
        'resnet152' : 'resnet',
    }

    path = ''
    num_classes = 0
    if args.dataset == 'cifar100':
        path = 'models'
        num_classes = 100
    elif args.dataset == 'pet':
        path = 'models_large'
        num_classes = 37
    else:
        raise ValueError('dataset type not supported')

    try:
        path = path + '.' + models[args.net]
    except ValueError:
        print('network name not supported:'.format(args.net))

    module = importlib.import_module(path)
    net = getattr(module, args.net)
    net = net(num_classes)
    return net


#def load_module(module_path):
#    return module





def get_network(args):
    """ return given network
    """

    net = load_net(args)
    #from torchvision.models import resnet50
    #net = resnet50(num_classes=37, pretrained=False)
    #if args.net == 'vgg16':
    #    from models.vgg import vgg16_bn
    #    net = vgg16_bn()
    #elif args.net == 'vgg13':
    #    from models.vgg import vgg13_bn
    #    net = vgg13_bn()
    #elif args.net == 'vgg11':
    #    from models.vgg import vgg11_bn
    #    net = vgg11_bn()
    #elif args.net == 'vgg19':
    #    from models.vgg import vgg19_bn
    #    net = vgg19_bn()
    #elif args.net == 'densenet121':
    #    from models.densenet import densenet121
    #    net = densenet121()
    #elif args.net == 'densenet161':
    #    from models.densenet import densenet161
    #    net = densenet161()
    #elif args.net == 'densenet169':
    #    from models.densenet import densenet169
    #    net = densenet169()
    #elif args.net == 'densenet201':
    #    from models.densenet import densenet201
    #    net = densenet201()
    #elif args.net == 'googlenet':
    #    from models.googlenet import googlenet
    #    net = googlenet()
    #elif args.net == 'inceptionv3':
    #    from models.inceptionv3 import inceptionv3
    #    net = inceptionv3()
    #elif args.net == 'inceptionv4':
    #    from models.inceptionv4 import inceptionv4
    #    net = inceptionv4()
    #elif args.net == 'inceptionresnetv2':
    #    from models.inceptionv4 import inception_resnet_v2
    #    net = inception_resnet_v2()
    #elif args.net == 'xception':
    #    from models.xception import xception
    #    net = xception()
    #elif args.net == 'resnet18':
    #    if args.dataset == 'cifar100':
    #        from models.resnet import resnet18
    #        net = resnet18(100)
    #    elif args.dataset == 'pet':
    #        from models_large.resnet import resnet18
    #        net = resnet18(37)
    #elif args.net == 'resnet34':
    #    if args.dataset == 'cifar100':
    #        from models.resnet import resnet34
    #        net = resnet34(100)
    #    elif args.dataset == 'pet':
    #        from models_large.resnet import resnet34
    #        net = resnet34(37)
    #elif args.net == 'resnet50':
    #    if args.dataset == 'cifar100':
    #        from models.resnet import resnet50
    #        #from models_large.resnet import resnet50
    #        net = resnet50(100)
    #    elif args.dataset == 'pet':
    #        from models_large.resnet import resnet50
    #        net = resnet50(num_classes=37)
    #elif args.net == 'resnet101':
    #    if args.dataset == 'cifar100':
    #        from models.resnet import resnet101
    #        net = resnet101(100)
    #    elif args.dataset == 'pet':
    #        from models_large.resnet import resnet101
    #        net = resnet101(37)
    #elif args.net == 'resnet152':
    #    if args.dataset == 'cifar100':
    #        from models.resnet import resnet152
    #        net = resnet152(100)
    #    elif args.dataset == 'pet':
    #        from models_large.resnet import resnet152
    #        #net = resnet50(num_classes=37)
    #        net = resnet152(37)
    #elif args.net == 'preactresnet18':
    #    from models.preactresnet import preactresnet18
    #    net = preactresnet18()
    #elif args.net == 'preactresnet34':
    #    from models.preactresnet import preactresnet34
    #    net = preactresnet34()
    #elif args.net == 'preactresnet50':
    #    from models.preactresnet import preactresnet50
    #    net = preactresnet50()
    #elif args.net == 'preactresnet101':
    #    from models.preactresnet import preactresnet101
    #    net = preactresnet101()
    #elif args.net == 'preactresnet152':
    #    from models.preactresnet import preactresnet152
    #    net = preactresnet152()
    #elif args.net == 'resnext50':
    #    from models.resnext import resnext50
    #    net = resnext50()
    #elif args.net == 'resnext101':
    #    from models.resnext import resnext101
    #    net = resnext101()
    #elif args.net == 'resnext152':
    #    from models.resnext import resnext152
    #    net = resnext152()
    #elif args.net == 'shufflenet':
    #    from models.shufflenet import shufflenet
    #    net = shufflenet()
    #elif args.net == 'shufflenetv2':
    #    from models.shufflenetv2 import shufflenetv2
    #    net = shufflenetv2()
    #elif args.net == 'squeezenet':
    #    from models.squeezenet import squeezenet
    #    net = squeezenet()
    #elif args.net == 'mobilenet':
    #    from models.mobilenet import mobilenet
    #    net = mobilenet()
    #elif args.net == 'mobilenetv2':
    #    from models.mobilenetv2 import mobilenetv2
    #    net = mobilenetv2()
    #elif args.net == 'nasnet':
    #    from models.nasnet import nasnet
    #    net = nasnet()
    #elif args.net == 'attention56':
    #    from models.attention import attention56
    #    net = attention56()
    #elif args.net == 'attention92':
    #    from models.attention import attention92
    #    net = attention92()
    #elif args.net == 'seresnet18':
    #    from models.senet import seresnet18
    #    net = seresnet18()
    #elif args.net == 'seresnet34':
    #    from models.senet import seresnet34
    #    net = seresnet34()
    #elif args.net == 'seresnet50':
    #    from models.senet import seresnet50
    #    net = seresnet50()
    #elif args.net == 'seresnet101':
    #    from models.senet import seresnet101
    #    net = seresnet101()
    #elif args.net == 'seresnet152':
    #    from models.senet import seresnet152
    #    net = seresnet152()
    #elif args.net == 'efficientnetb0':
    #    if args.dataset == 'cifar100':
    #        from models.efficientnet import efficientnetb0
    #        net = efficientnetb0(num_classes=100)
    #    elif args.dataset == 'pet':
    #        from models.resnet import resnet50
    #        #net = resnet50()
    #        #from torchvision.models import resnet50
    #        #net = resnet50(num_classes=37)
    #        #from models_large.efficientnet import efficientnetb0
    #        from models_large.efficientnet1 import efficientnetb0
    #        net = efficientnetb0(num_classes=37)
    #elif args.net == 'efficientnetb1':
    #    from models.efficientnet import efficientnetb1
    #    net = efficientnetb1()
    #elif args.net == 'efficientnetb2':
    #    from models.efficientnet import efficientnetb2
    #    net = efficientnetb2()
    #elif args.net == 'efficientnetb3':
    #    from models.efficientnet import efficientnetb3
    #    net = efficientnetb3()
    #elif args.net == 'efficientnetb4':
    #    from models.efficientnet import efficientnetb4
    #    net = efficientnetb4()
    #elif args.net == 'efficientnetb5':
    #    from models.efficientnet import efficientnetb5
    #    net = efficientnetb5()
    #elif args.net == 'efficientnetb6':
    #    from models.efficientnet import efficientnetb6
    #    net = efficientnetb6()
    #elif args.net == 'efficientnetb7':
    #    from models.efficientnet import efficientnetb7
    #    net = efficientnetb7()
    #elif args.net == 'efficientnetl2':
    #    from models.efficientnet import efficientnetl2
    #    net = efficientnetl2()

    #else:
    #    print('the network name you have entered is not supported yet')
    #    sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def cifar100_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def cifar100_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def pet_training_dataloader(download, mean, std, batch_size, num_workers, shuffle):
    transforms_train = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(torch.nn.ModuleList([
                    transforms.ColorJitter()
            ]), p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
    ])

    from dataset.pet import OxfordPet
    dataset = OxfordPet(
        'data',
        image_set='train',
        download=download,
        transforms=transforms_train
    )

    return DataLoader(
        dataset,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size
    )


def pet_test_dataloader(download, mean, std, batch_size, num_workers, shuffle):
    transforms_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
    ])

    from dataset.pet import OxfordPet
    dataset = OxfordPet(
        'data',
        image_set='val',
        download=download,
        transforms=transforms_test)

    return DataLoader(
        dataset,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size
    )

def dataloader(args, image_set):
    if args.dataset == 'cifar100':
        if image_set == 'train':
            cifar100_training_loader = cifar100_training_dataloader(
                settings.CIFAR100_TRAIN_MEAN,
                settings.CIFAR100_TRAIN_STD,
                num_workers=4,
                batch_size=args.b,
                shuffle=True
            )
            return cifar100_training_loader

        elif image_set == 'val':
            cifar100_test_loader = cifar100_test_dataloader(
                settings.CIFAR100_TRAIN_MEAN,
                settings.CIFAR100_TRAIN_STD,
                num_workers=4,
                batch_size=args.b,
                shuffle=True
            )
            return cifar100_test_loader
        else:
            raise ValueError('wrong image_set value')

    elif args.dataset == 'pet':
        if image_set == 'train':
            pet_training_loader = pet_training_dataloader(
                args.download,
                settings.PET_TRAIN_MEAN,
                settings.PET_TRAIN_STD,
                num_workers=4,
                batch_size=args.b,
                shuffle=True
            )
            return pet_training_loader

        elif image_set == 'val':
            pet_test_loader = pet_test_dataloader(
                args.download,
                settings.PET_TRAIN_MEAN,
                settings.PET_TRAIN_STD,
                num_workers=4,
                batch_size=args.b,
                shuffle=True
            )
            return pet_test_loader
        else:
            raise ValueError('wrong image_set value')

    else:
        raise ValueError('wrong dataset')


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
