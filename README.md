# Pytorch-cifar100

practice on cifar100 using pytorch

## Requirements
- python3.5
- pytorch4.0
- tensorflow1.4
- cuda8.0
- cudnnv5
- tensorboardX1.4
## Usage

### 1. enter directory
```bash
$ cd pytorch-cifar100
```

### 2. change cifar100 dataset path in settings.py
```g_cifar100_path``` is the path to cifar100 dataset, you can download cifar100 by clicking [here](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz), or download from the offical website [here](https://www.cs.toronto.edu/~kriz/cifar.html). Noet that please download the python version cifar100 dataset.

### 3. run tensorbard
```bash
$ mkdir runs
$ tensorboard --logdir='runs' --port=6006
```

### 4. train the model
You need to specify the network you want to train in these two lines
```python
from models.inceptionv3 import *
net = inceptionv3().cuda()
```
Change the code, assign ```net``` variable to the network you want to train.

then train the model
```bash
$ python train.py
```

## Implementated NetWork

- vgg [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)
- google net [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842v1)
- inceptionv3 [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567v3)
- inceptionv4, inception_resnet_v2 [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
- xception [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- resnet [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)
- resnet in resnet [Resnet in Resnet: Generalizing Residual Architectures](https://arxiv.org/abs/1603.08029v1)
- densenet [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993v5)
    
## Training
I train model for 140 epoch(most of the time, but not always)
set learning rate at:
- epoch < 60, lr = 0.1
- epoch < 100, lr = 0.01
- epoch < 140, lr = 0.001

I found that training more epoch when lr = 0.1 can improve
my model prformance by %1 or %2, but adding more epoch at lr = 0.01
or lr = 0.001 won't make much difference.So I decide to train my
model for more epoch when lr = 0.1

## Results
Best result I can get from a certain model, you can try yourself.

|dataset|network|params|top1 err|top5 err|memory|epoch(lr = 0.1)|epoch(lr = 0.01)|epoch(lr = 0.001)|total epoch|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|cifar100|vgg16_bn|34.0M|27.77|8.84|2.83GB|140|40|40|220|
|cifar100|densenet121|7.0M|22.99|6.45|1.28GB|60|40|40|140|
|cifar100|resnet101|42.7M|22.22|5.61|3.72GB|60|40|40|140|
|cifar100|densenet161|26M|21.56|6.04|2.10GB|80|40|40|160|
|cifar100|densenet201|18M|21.46|5.9|2.10GB|100|40|40|180|
|cifar100|googlenet|6.2M|22.09|5.94|2.10GB|100|40|40|180|



