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

### 2. change cifar100 dataset path in conf/global_settings.py
```CIFAR100_PATH``` is the path to cifar100 dataset, you can download cifar100 by clicking [here](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz), or download from the offical website [here](https://www.cs.toronto.edu/~kriz/cifar.html). Note that please download the python version cifar100 dataset.

### 3. run tensorbard
```bash
$ mkdir runs
$ tensorboard --logdir='runs' --port=6006 --host='localhost'
```

### 4. train the model
You need to specify the net you want to train using arg -net

```bash
$ python train.py -net vgg16
```
The supported net args are:
```
vgg16
densenet121
densenet161
densenet201
googlenet
inceptionv3
inceptionv4
xception
resnet18
resnet34
resnet50
resnet101
resnet150
resnext50
resnext101
resnext152
```
Normally, the weights file with the best accuracy would be written to the disk(default in checkpoint folder).

### 5. test the model
Test the model using test.py
```bash
$ python test.py -net vgg16 -weights path_to_vgg16_weights_file
```

## Implementated NetWork

- vgg [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)
- google net [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842v1)
- inceptionv3 [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567v3)
- inceptionv4, inception_resnet_v2 [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
- xception [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- resnet [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)
- resnext [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431v2)
- resnet in resnet [Resnet in Resnet: Generalizing Residual Architectures](https://arxiv.org/abs/1603.08029v1)
- densenet [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993v5)
    
## Training Details
I found that training more epoch when lr = 0.1 can improve
my model prformance by %1 or %2, but adding more epoch at lr = 0.01
or lr = 0.001 won't make much difference.So I decide to train my
model for more epoch when lr = 0.1.

You can choose whether to use TensorBoard to visualize your training procedure

## Results
Best result I can get from a certain model, you can try yourself.

|dataset|network|params|top1 err|top5 err|memory|epoch(lr = 0.1)|epoch(lr = 0.01)|epoch(lr = 0.001)|total epoch|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|cifar100|vgg16_bn|34.0M|27.77|8.84|2.83GB|140|40|40|220|
|cifar100|resnet18|11.2M|24.39|6.95|3.02GB|80|60|60|200|
|cifar100|resnet34|21.3M|23.24|6.63|3.22GB|80|60|60|200|
|cifar100|resnet50|23.7M|22.61|6.04|3.40GB|80|60|60|200|
|cifar100|resnet101|42.7M|22.22|5.61|3.72GB|80|60|60|200|
|cifar100|resnet152|58.3M|22.31|5.81|4.36GB|80|60|60|200|
|cifar100|densenet121|7.0M|22.99|6.45|1.28GB|60|40|40|140|
|cifar100|densenet161|26M|21.56|6.04|2.10GB|80|40|40|160|
|cifar100|densenet201|18M|21.46|5.9|2.10GB|100|40|40|180|
|cifar100|googlenet|6.2M|22.09|5.94|2.10GB|100|40|40|180|
|cifar100|inceptionv3|22.3M|22.81|6.39|2.26GB|140|80|60|280|



