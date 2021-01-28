""" train and test dataset

author baiyu
"""
import os
import sys
import tarfile
from PIL import Image
import base64
import json
import shutil
from io import BytesIO

import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import lmdb
from tqdm import tqdm


class OxfordPet(Dataset):
    def __init__(self, path, download=False, image_set='train', transforms=None):
        """
        """
        super().__init__()
        #self.path = path
        self.transforms = transforms
        data_md5 = '5c4f3ee8e5d25df40f4fd59a7f44e54c'
        anno_md5 = '95a8c909bbe2e81eed6a22bccdf3f68f'
        dataset_url = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'
        anno_url = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz'

        data_filename = 'images.tar.gz'
        anno_filename = 'annotations.tar.gz'

        if download:
            download_url(dataset_url, path, data_filename, md5=data_md5)
            print()
            download_url(anno_url, path, anno_filename, md5=anno_md5)
            print()

        lmdb_fp = os.path.join(path, 'pet', image_set)

        def track_progress(members, total=None):
            for member in tqdm(members, total=total):
                yield member

        if not os.path.exists(lmdb_fp):
            print('Extracting file {}'.format(data_filename))
            with tarfile.open(os.path.join(path, data_filename), "r") as tar:
                tar.extractall(path=path, members=track_progress(tar, 7394))
            print('Done.\n')

            print('Extracting file {}'.format(anno_filename))
            with tarfile.open(os.path.join(path, anno_filename), "r") as tar:
                tar.extractall(path=path, members=track_progress(tar, 18474))
            print('Done.\n')

            images = []
            labels = []
            if image_set == 'train':
                anno_path = os.path.join(path, 'annotations', 'trainval.txt')
            elif image_set == 'val':
                anno_path = os.path.join(path, 'annotations', 'test.txt')
            else:
                raise ValueError('wrong image_set arguments')

            with open(anno_path, 'r') as f:
                for line in f.readlines():
                    image, label = line.split()[:2]
                    image += '.jpg'
                    images.append(os.path.join(path, 'images', image))
                    labels.append(int(label) - 1)

            os.makedirs(lmdb_fp)

            db_size = 1 << 40
            env = lmdb.open(lmdb_fp, map_size=db_size)
            print('Converting data to lmdb format.....')
            with env.begin(write=True) as txn:
                for image_path, label in tqdm(zip(images, labels), total=len(images)):
                    with open(image_path, 'rb') as f:
                        img_str = base64.encodebytes(f.read()).decode('utf-8')
                        label_str = str(label)
                        image_name = os.path.basename(image_path).encode('utf-8')
                        txn.put(image_name, json.dumps([img_str, label_str]).encode('utf-8'))

            env.close()
            print('Done.\n')
            print('Removing folder {}'.format(os.path.join(path, 'annotations')))
            shutil.rmtree(os.path.join(path, 'annotations'))
            print('Done.\n')
            print('Removing folder {}'.format(os.path.join(path, 'images')))
            shutil.rmtree(os.path.join(path, 'images'))
            print('Done.\n')


        self.env = lmdb.open(lmdb_fp, map_size=1099511627776, readonly=True, lock=False)

        with self.env.begin(write=False) as txn:
            self.image_names = [key for key in txn.cursor().iternext(keys=True, values=False)]



    def __len__(self):
            return len(self.image_names)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
           image_str, label_str = json.loads(txn.get(self.image_names[index]).decode('utf-8'))
           image_bytes = image_str.encode('utf-8')
           image_buffer = base64.decodebytes(image_bytes)

           image = Image.open(BytesIO(image_buffer)).convert('RGB')
           label = int(label_str)

        if self.transforms:
            image = self.transforms(image)

        return image, label

#mean=(0.485, 0.456, 0.406)
#std=(0.229, 0.224, 0.225)
#
#dataset = OxfordPet('/data/by/pytorch-cifar100/tmp1', download=True)
#
#i, l = dataset[33]
#for i, l in dataset:
#    print(i, l)