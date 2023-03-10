import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from batch_shift_utils import RandomShift
from batch_shift_utils import ZipDataset


def compute_mean_std(dataset: Dataset):
    """compute the mean and std of dataset
    Args:
        dataset: class torch.utils.data.Dataset

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = torch.stack([dataset[i][1][:, :, 0] for i in range(len(dataset))])
    data_g = torch.stack([dataset[i][1][:, :, 1] for i in range(len(dataset))])
    data_b = torch.stack([dataset[i][1][:, :, 2] for i in range(len(dataset))])
    mean = torch.mean(data_r), torch.mean(data_g), torch.mean(data_b)
    std = torch.std(data_r), torch.std(data_g), torch.std(data_b)

    return mean, std


def get_train_augs(img_size, mean, std, p_apply=0.35):
    return transforms.Compose([
        RandomShift(img_size, padding=4, p_apply=p_apply),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def get_training_dataloader(
        dataset_class: Dataset, root,
        mean, std, img_size=32,
        batch_size=128, num_workers=2,
        shuffle=True, multiply_data=1,
        prob_aug=1.,
):
    """ return training dataloader
    Args:
        dataset_class:
        img_size:
        prob_aug:
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
        dataset_class(
            root=root,
            train=True,
            download=True,
            transform=get_train_augs(img_size, mean, std, p_apply=prob_aug)
        )
        for _ in range(multiply_data)
    ]

    training_loader = DataLoader(
        dataset=ZipDataset(datasets),
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=round(batch_size / multiply_data)
    )

    return training_loader


def get_test_dataloader(
        dataset_class: Dataset, root,
        mean, std, batch_size=16,
        num_workers=2, shuffle=True,
        img_size=32, img_pad=0
):
    """ return training dataloader
    Args:
        dataset_class:
        img_size:
        img_pad:
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

    test_loader = DataLoader(
        dataset_class(root=root, train=False, download=True, transform=transform_test),
        shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader


def construct_dataloaders(
        dataset_class: Dataset, root,
        mean, std, batch_size, multiply_data, prob_aug, img_size=32, num_workers=4
):
    training_loader = get_training_dataloader(
        dataset_class, root,
        mean, std,
        img_size=img_size,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        multiply_data=multiply_data,
        prob_aug=prob_aug
    )

    test_loader = get_test_dataloader(
        dataset_class, root,
        mean, std,
        img_size=img_size,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True
    )

    return training_loader, test_loader
