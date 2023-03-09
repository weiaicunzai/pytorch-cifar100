""" train and test dataset

author baiyu
"""

import pickle
from pathlib import Path

import numpy
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

from batch_shift_utils import RandomShift
from batch_shift_utils import ZipDataset

# mean and std of cifar100 dataset
_CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
_CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


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


class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        # if transform is given, we transoform data using
        with (Path(path) / 'train').open('rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['fine_labels'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image


class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with (Path(path) / 'test').open('rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image


def get_train_cifar100_with_augs(img_size, mean, std, p_apply=0.35):
    return CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            RandomShift(img_size, padding=4, p_apply=p_apply),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    )


def get_training_dataloader(
        mean, std, img_size=32,
        batch_size=128, num_workers=2,
        shuffle=True, multiply_data=1,
        prob_aug=1.,
):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
        multiply_data: multiply dataset in several times
    Returns: train_data_loader:torch dataloader object
    """

    datasets = [
        get_train_cifar100_with_augs(img_size, mean, std, p_apply=prob_aug)
        for _ in range(multiply_data)
    ]

    cifar100_training_loader = DataLoader(
        dataset=ZipDataset(datasets),
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=round(batch_size / multiply_data)
    )

    return cifar100_training_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, img_size=32, img_pad=0):
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
    if img_pad:
        resize = transforms.Pad(img_pad)
    else:
        resize = transforms.Resize(img_size)

    transform_test = transforms.Compose([
        resize,
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_test = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


def get_cifar100_dataloaders(args):
    training_loader = get_training_dataloader(
        _CIFAR100_TRAIN_MEAN,
        _CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        multiply_data=args.multiply_data,
        prob_aug=args.prob_aug
    )

    test_loader = get_test_dataloader(
        _CIFAR100_TRAIN_MEAN,
        _CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    return training_loader, test_loader
