""" train and test dataset

author baiyu
"""

from torch import tensor
from pathlib import Path
from torchvision.datasets import CIFAR100

from datasets.utils import construct_dataloaders
from datasets.utils import get_test_dataloader

# mean and std of cifar100 dataset
_CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
_CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def unnormalize_batch(batch):
    return batch * tensor(_CIFAR100_TRAIN_STD)[None, :, None, None] + tensor(_CIFAR100_TRAIN_MEAN)[None, :, None, None]  # unnormalize


def get_cifar100_test_dataloader(batch_size=128, shuffle=False, num_workers=4, img_pad=0, data_root="."):
    test_loader = get_test_dataloader(
        CIFAR100, root=Path(data_root) / 'data',
        mean=_CIFAR100_TRAIN_MEAN,
        std=_CIFAR100_TRAIN_STD,
        img_size=32,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        img_pad=img_pad
    )

    return test_loader


def get_cifar100_dataloaders(args, data_root="."):
    training_loader, test_loader = construct_dataloaders(
        CIFAR100, root=Path(data_root) / 'data',
        mean=_CIFAR100_TRAIN_MEAN,
        std=_CIFAR100_TRAIN_STD,
        batch_size=args.b,
        multiply_data=args.multiply_data,
        prob_aug=args.prob_aug,
        img_size=32
    )

    return training_loader, test_loader
