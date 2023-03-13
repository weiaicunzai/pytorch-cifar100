import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import requests
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from datasets.utils import construct_dataloaders
from datasets.utils import get_test_dataloader

"""
TinyImageNetPath
├── test
│   └── images
│       ├── test_0.JPEG
│       ├── t...
│       └── ...
├── train
│   ├── n01443537
│   │   ├── images
│   │   │   ├── n01443537_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01443537_boxes.txt
│   ├── n01629819
│   │   ├── images
│   │   │   ├── n01629819_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01629819_boxes.txt
│   ├── n...
│   │   ├── images
│   │   │   ├── ...
│   │   │   └── ...
├── val
│   ├── images
│   │   ├── val_0.JPEG
│   │   ├── v...
│   │   └── ...
│   └── val_annotations.txt
├── wnids.txt
└── words.txt
"""

TINY_IMAGENET_FOLDER = 'tiny-imagenet-200'
TINY_IMAGENET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

_TINY_IMAGENET_TRAIN_MEAN = (0.485, 0.456, 0.406)
_TINY_IMAGENET_TRAIN_STD = (0.229, 0.224, 0.225)


def download(url: str, filename: Union[str, Path], chunk_size=1024):
    url = str(url)
    filename = Path(filename)
    stream_dl = requests.get(url, stream=True)

    # check if archive already downloaded
    total_size = int(stream_dl.headers['content-length'])
    if filename.is_file():
        if total_size == filename.stat().st_size:
            print("Files already downloaded and verified")
            return

    try:
        with filename.open('wb') as file:
            print('connect to', url)

            bar = tqdm(
                desc='Download',
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=chunk_size
            )

            for data in stream_dl.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

    except Exception:  # write context manager
        filename.unlink()
        print(f"file {filename} removed")


def download_and_unzip(root_dir):
    filename = Path(root_dir) / (TINY_IMAGENET_FOLDER + ".zip")
    download(TINY_IMAGENET_URL, filename)

    with zipfile.ZipFile(filename) as zf:

        list_of_files = zf.infolist()

        # check if archive already extracted
        if len(list_of_files) == len(list(filename.with_suffix('').rglob("*"))) + 1:
            return

        for member in tqdm(list_of_files, desc='Extracting'):
            try:
                zf.extract(member, root_dir)
            except zipfile.error:
                pass


class TinyImageNetPaths:
    def __init__(self, data_dir, download=False):
        """Creates a paths datastructure for the tiny imagenet.
        Args:
            data_dir: Where the data is located
            download: Download if the data is not there
        Members:
            label_id:
            ids:
            nit_to_words:
            data_dict:
        """

        data_dir = Path(data_dir)
        if download:
            download_and_unzip(data_dir)

        root_dir = data_dir / TINY_IMAGENET_FOLDER

        train_path = root_dir / 'train'
        val_path = root_dir / 'val'
        # test_path = root_dir / 'test'

        wnids_path = root_dir / 'wnids.txt'
        words_path = root_dir / 'words.txt'

        self._make_paths(
            train_path, val_path,
            # test_path,
            wnids_path, words_path
        )

    def _make_paths(
            self,
            train_path,
            val_path,
            # test_path,
            wnids_path,
            words_path
    ):

        wnids = []
        with wnids_path.open('r') as idf:
            for nid in idf:
                wnids.append(nid.strip())

        self.nid_to_words = defaultdict(list)
        with words_path.open('r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            'train': [],
            'val': [],
            # 'test': [os.path.join(test_path, x) for x in os.listdir(test_path)]
        }

        # Get the validation paths and labels
        with (val_path / 'val_annotations.txt').open('r') as valf:
            val_im_path = val_path / 'images'
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = str(val_im_path / fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = wnids.index(nid)

                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        for nid_path in train_path.iterdir():
            nid = nid_path.name
            imgs_path = nid_path / 'images'
            label_id = wnids.index(nid)

            anno_path = nid_path / (nid + '_boxes.txt')
            with anno_path.open('r') as anno_file:
                for line in anno_file:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = str(imgs_path / fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)

                    self.paths['train'].append((fname, label_id, nid, bbox))


class TinyImageNet(Dataset):
    def __init__(
            self, root, train=True,
            transform=None, download=False,
            max_samples=None
    ):

        """Datastructure for the tiny image dataset.
        Args:
            root: Root directory for the data
            train: mode of dataset ("train", "test", or "val")
            transform: Transformation to use at the retrieval time
            download: Download the dataset
        Members:
            tinp: Instance of the TinyImageNetPaths
            img_data: Image data
            label_data: Label data
    """

        tinp = TinyImageNetPaths(root, download)

        self.mode = 'train' if train else 'val'
        self.label_idx = 1  # from [image, id, nid, box]
        self.transform = transform
        self.transform_results = dict()

        self.max_samples = max_samples
        self.samples = tinp.paths[self.mode]
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[:self.samples_num]

        self.img_data = []
        self.label_data = []

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):

        s = self.samples[idx]
        img = Image.open(s[0])
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = None if self.mode == 'test' else s[self.label_idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def get_tiny_imagenet_test_dataloader(batch_size=128, shuffle=False, num_workers=4, img_pad=0):
     
    test_loader = get_test_dataloader(
        TinyImageNet, root="./data/",
        mean=_TINY_IMAGENET_TRAIN_MEAN, std=_TINY_IMAGENET_TRAIN_STD,
        batch_size=batch_size,
        img_size=64,
        num_workers=num_workers,
        shuffle=shuffle,
        img_pad=img_pad
     )
    
    return test_loader


def get_tiny_imagenet_dataloaders(args):
    training_loader, test_loader = construct_dataloaders(
        TinyImageNet, root="./data/",
        mean=_TINY_IMAGENET_TRAIN_MEAN, std=_TINY_IMAGENET_TRAIN_STD,
        batch_size=args.b,
        multiply_data=args.multiply_data,
        prob_aug=args.prob_aug,
        img_size=64
    )

    return training_loader, test_loader
