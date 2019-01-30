# Pytorch-cifar100

practice on cifar100 using pytorch

## Requirements

This is my experiment eviroument, pytorch0.4 should also be fine
- python3.5
- pytorch1.0
- tensorflow1.5
- cuda8.0
- cudnnv5
- tensorboardX1.6


## Usage

### 1. enter directory
```bash
$ cd pytorch-cifar100
```

### 2. change cifar100 dataset path in conf/global_settings.py
I will use cifar100 dataset from torchvision since it's more convient, but I also
kept the code sample for write your own dataset module in dataset folder, as an
example for people don't know how to write it.

### 3. run tensorbard
Install tensorboardX (a tensorboard wrapper for pytorch)
```bash
$ pip install tensorboardX
$ mkdir runs
Run tensorboard
$ tensorboard --logdir='runs' --port=6006 --host='localhost'
```

### 4. train the model
You need to specify the net you want to train using arg -net

```bash
$ python train.py -net vgg16
```
The supported net args are:
```
shufflenet
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
resnet152
resnext101
resnext152
```
Normally, the weights file with the best accuracy would be written to the disk with name suffix 'best'(default in checkpoint folder).

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
- shufflenet [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083v2)
    
## Training Details
I follow the hyperparameter settings in paper [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1701.06548v1), which is init lr = 0.1 divide by 5 at 60th, 120th, 160th epochs, train for 200
epochs with batchsize 128 and weight decay 5e-4, Nesterov momentum of 0.9. You could also use the hyperparameters from
paper [Regularizing Neural Networks by Penalizing Confident Output Distributions](https://arxiv.org/abs/1701.06548) and [Random Erasing Data Augmentation](https://arxiv.org/abs/1701.06548), which is 
initial lr = 0.1, lr divied by 10 at 150 and 225, and training for 300 epochs with batchsize 126, this is more commonly
used. You could decrese the batchsize to 64 or whatever suits you, if you dont have enough gpu memory.

You can choose whether to use TensorBoard to visualize your training procedure

## Results
The result I can get from a certain model, you can try yourself by finetuning the hyperparameters.
I didn't use any training tricks to improve accuray, if you want to learn more about training tricks,
please refer to my another [repo](https://github.com/weiaicunzai/Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks), contains
various common training tricks and their pytorch implementations.

|dataset|network|params|top1 err|top5 err|memory|epoch(lr = 0.1)|epoch(lr = 0.01)|epoch(lr = 0.001)|total epoch|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|cifar100|shufflenet|1.0M|29.94|8.35|0.84GB|140|40|40|220|
|cifar100|vgg16_bn|34.0M|27.77|8.84|2.83GB|140|40|40|220|
|cifar100|resnet18|11.2M|24.39|6.95|3.02GB|80|60|60|200|
|cifar100|resnet34|21.3M|23.24|6.63|3.22GB|80|60|60|200|
|cifar100|resnet50|23.7M|22.61|6.04|3.40GB|80|60|60|200|
|cifar100|resnet101|42.7M|22.22|5.61|3.72GB|80|60|60|200|
|cifar100|resnet152|58.3M|22.31|5.81|4.36GB|80|60|60|200|
|cifar100|resnext50|14.8M|22.23|6.00|1.91GB|80|60|60|200|
|cifar100|resnext101|25.3M|22.22|5.99|2.63GB|80|60|60|200|
|cifar100|resnext152|33.3M|22.40|5.58|3.18GB|80|60|60|200|
|cifar100|densenet121|7.0M|22.99|6.45|1.28GB|60|40|40|140|
|cifar100|densenet161|26M|21.56|6.04|2.10GB|80|40|40|160|
|cifar100|densenet201|18M|21.46|5.9|2.10GB|100|40|40|180|
|cifar100|googlenet|6.2M|22.09|5.94|2.10GB|100|40|40|180|
|cifar100|inceptionv3|22.3M|22.81|6.39|2.26GB|140|80|60|280|
|cifar100|xception|21.0M|25.07|7.32|1.67GB|140|80|60|200|



