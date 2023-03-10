""" train and test dataset

author baiyu
"""

from torchvision.datasets import CIFAR100

from datasets.utils import construct_dataloaders

# mean and std of cifar100 dataset
_CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
_CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def get_cifar100_dataloaders(args):
    training_loader, test_loader = construct_dataloaders(
        CIFAR100, root='./data/',
        mean=_CIFAR100_TRAIN_MEAN,
        std=_CIFAR100_TRAIN_STD,
        batch_size=args.b,
        multiply_data=args.multiply_data,
        prob_aug=args.prob_aug,
        img_size=32
    )

    return training_loader, test_loader
