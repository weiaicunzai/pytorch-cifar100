from typing import Callable
from typing import List
from typing import Sized

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Pad
from torchvision.transforms import RandomCrop
from torchvision.transforms import Resize


# def batch_shift(img_batch, batch_shift_scale: int = 2):
#     return img_batch.repeat_interleave(batch_shift_scale, 0)


# def increase_labels(labels, batch_shift_scale: int = 2):
#     return labels.repeat_interleave(batch_shift_scale, 0)


class RepeatBatchSampler(Sampler):

    def __init__(self, len_dataset, batch_size, repeat_batch_scale, data_source: Sized, shuffle=False):
        super().__init__(data_source)
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.

        if not isinstance(len_dataset, int) or len_dataset <= 0:
            raise ValueError("len_dataset should be a positive integer value, "
                             "but got len_dataset={}".format(len_dataset))

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))

        if not isinstance(repeat_batch_scale, int) or repeat_batch_scale <= 0:
            raise ValueError("repeat_batch_scale should be a positive integer value, "
                             "but got repeat_batch_scale={}".format(repeat_batch_scale))

        self.n_sample = len_dataset // batch_size  # drop_last is True
        self.indexes = torch.arange(self.n_sample * batch_size, device='cpu')

        self.indexes = self.indexes.repeat_interleave(repeat_batch_scale)
        self.indexes = self.indexes.reshape(batch_size * repeat_batch_scale, -1)

        if shuffle:
            self.indexes = self.indexes[torch.randperm(self.indexes.shape[0])]

        self.indexes = self.indexes.tolist()

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.n_sample


class Identity(object):
    def __call__(self, img):
        return img


class RandomShift(Callable):
    def __init__(self, size: int, padding, p_apply=0.35, mode="pad"):
        super().__init__()

        if mode == "pad":
            self.up_size = Pad(padding)
        elif mode == "resize":
            self.up_size = Resize(size + 2 * padding)
        else:
            raise ValueError(f'mode: {mode} (need "pad" or "resize")')

        self.rcrop = RandomCrop(size)

        assert 0. <= p_apply <= 1., "p_apply is not probability"
        self.p_apply = p_apply
        print('self.p_apply', self.p_apply)
        print('self.mode', mode)

    def __call__(self, img):
        ran_sample = np.random.random_sample()  # in [0., 1.)
        if ran_sample < self.p_apply:
            return self.rcrop(self.up_size(img))

        return img


class RepeatDataset(Dataset):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times

        self._orig_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._orig_len]

    def __len__(self):
        """Length after repetition."""

        return self.times * self._orig_len


class InterleaveDataset(Dataset):

    def __init__(
            self,
            datasets: List[Dataset],
            shuffle=True
    ):

        self.datasets = datasets
        self.ds_num = len(datasets)
        assert self.ds_num > 0, 'datasets should not be an empty iterable'

        # check datasets length
        self.n_samples = len(datasets[0])
        for dataset in datasets[1:]:
            if not self.n_samples == len(dataset):
                raise ValueError(f"Dataset '{dataset}' has wrong length")

        self.samples_idxs = None
        self.build_samples_idxs(shuffle=shuffle)

    def build_samples_idxs(self, shuffle=True):

        self.samples_idxs = np.arange(self.n_samples, dtype=int)
        if shuffle:
            np.random.shuffle(self.samples_idxs)

        self.samples_idxs = self.samples_idxs.repeat(self.ds_num, axis=0)

    def __getitem__(self, global_idx):
        # global_idx = 0 1 2 3 4 5, ds_idx = 0 1 0 1 0 1, samples_idxs = 0 0 1 1 2 2

        ds_idx = global_idx % self.ds_num  # 2

        return self.datasets[ds_idx][self.samples_idxs[global_idx]]

    def __len__(self):

        return self.ds_num * self.n_samples


class ZipDataset(Dataset):

    def __init__(self, datasets):

        self.datasets = datasets
        self.ds_num = len(datasets)
        if self.ds_num <= 0:
            raise ValueError('datasets should not be an empty')

        # check datasets length
        self.n_samples = len(datasets[0])
        for dataset in datasets[1:]:
            if not self.n_samples == len(dataset):
                raise ValueError(f"Dataset '{dataset}' has wrong length")

    def __getitem__(self, idx):

        images, labels = [], []

        for dataset in self.datasets:
            img, label = dataset[idx]
            images.append(img)
            labels.append(label)

        return images, labels

    def __len__(self):
        return self.n_samples
