# pytorch-cifar-100

practice on cifar-100 using pytorch


inspired by [this](https://github.com/kuangliu/pytorch-cifar) repo

## Requirements
- python3.5
- pytorch3.1
- tensorflow1.4
- cuda8.0
- cudnnv5

## Usage

### 1. enter directory
```bash
$ cd pytorch-cifar-100
```

### 2. run tensorbard
```bash
$ tensorboard --logdir='runs' --port=6006
```
### 3. train the model
```bash
$ python train.py
```
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
Best result I can get from a certain model
|dataset|network|params|top1 err|top5 err|memory|
|:---:|:---:|:---:|:---:|:---:|:---:|
|cifar100|vgg16_bn|14.8M|28.70|8.48|2.83GB|
|cifar100|densenet121|7.0M|22.99|6.45|1.28GB|
|cifar100|resnet101|42.7M|22.22|5.61|3.72GB|
|cifar100|densenet161||||2.10GB|

